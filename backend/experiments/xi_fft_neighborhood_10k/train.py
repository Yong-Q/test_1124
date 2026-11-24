"""Train the FFT neighborhood MLP and log physical curvature on held-out states."""

import sys

import torch

from experiments.xi_all_projection_loss.train import emit
from experiments.xi_continuous_pair_mlp import train as pair_train
from experiments.xi_conv3d_local_el.train import LocalOccupancy
from experiments.xi_positive_total_pair.operators import IdealProjection


def curvature_summary(predicted_c2, target_c2):
    # The project uses c2 = -D²(βF). Symmetrize in probe coordinates.
    predicted = -0.5 * (predicted_c2 + predicted_c2.mT)
    reference = -0.5 * (target_c2 + target_c2.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(reference)
    predicted_eigenvalues = torch.linalg.eigvalsh(predicted)
    directional = (eigenvectors * (predicted @ eigenvectors)).sum(-2)
    tolerance = eigenvalues.abs().amax(1, keepdim=True).clamp_min(1.e-20) * 1.e-5
    return dict(
        matrix_rmse=(predicted-reference).square().mean((1, 2)).sqrt().tolist(),
        directional_rmse=(directional-eigenvalues).square().mean(1).sqrt().tolist(),
        minimum_reference=eigenvalues[:, 0].tolist(),
        minimum_predicted=predicted_eigenvalues[:, 0].tolist(),
        negative_reference_modes=(eigenvalues < -tolerance).sum(1).tolist(),
        negative_predicted_modes=(predicted_eigenvalues < -tolerance).sum(1).tolist(),
        positive_to_negative=((eigenvalues > tolerance) & (directional < -tolerance)).sum(1).tolist(),
        negative_to_positive=((eigenvalues < -tolerance) & (directional > tolerance)).sum(1).tolist(),
    )


def with_curvature(factory):
    def configure(*args, **kwargs):
        original_batch = factory(*args, **kwargs)

        def batch(model, source, coeff, targets, scales, block, update_size, correction=None):
            result = original_batch(model, source, coeff, targets, scales, block, update_size,
                                    correction=correction)
            if update_size is not None:
                return result
            with torch.no_grad():
                if not hasattr(source, 'ideal_curvature'):
                    occupancy = LocalOccupancy.from_source(source)
                    source.ideal_curvature = IdealProjection(occupancy.counts, source.dv)
                    source.curvature_offset = 0
                coeff = coeff.double()
                _, ideal_c2 = source.ideal_curvature.project(coeff, coeff[:, :, 1:])
                predicted_c2 = result[1].detach().to(device='cpu', dtype=torch.float64)
                target_c2 = targets[1].to(device='cpu', dtype=torch.float64)
                ideal_c2 = ideal_c2.cpu()
                emit(stage='held_out_curvature', case=source.local_case, offset=source.curvature_offset,
                     probes=coeff.shape[-1]-1, excess=curvature_summary(predicted_c2, target_c2),
                     total=curvature_summary(predicted_c2+ideal_c2, target_c2+ideal_c2))
                source.curvature_offset += len(coeff)
            return result

        return batch

    return configure


if __name__ == '__main__':
    if '--direct' in sys.argv:
        raise SystemExit('Use the nonlinear FFT neighborhood MLP for this experiment')
    pair_train.batch_factory = with_curvature(pair_train.batch_factory)
    pair_train.main()
