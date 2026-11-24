"""Add local equilibrium supervision on training-only, reweighted atomic fields."""

import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.xi_all_projection_loss.train import emit, main
from experiments.xi_conv3d.bulk import INDEX
from experiments.xi_conv3d_sixlayer.bulk import SixLayerMultiscaleBulk, SixLayerSingleScaleBulk
from experiments.xi_conv3d_sixlayer.network import (
    SixLayerMultiscaleXi, SixLayerSingleScaleXi, valid_conv,
)
from experiments.xi_conv3d_sixlayer.source import SixLayerSource
from experiments.xi_conv3d_sixlayer.train import batch as projection_batch


class LocalOccupancy:
    """Sparse matrix of element/voxel occupations by trajectory frame."""

    def __init__(self, counts):
        self.counts = counts.coalesce()
        self.transpose = self.counts.T.coalesce()
        self.second = torch.sparse_coo_tensor(
            self.counts.indices(), self.counts.values().square(), self.counts.shape,
            is_coalesced=True,
        )

    @classmethod
    def from_source(cls, source):
        element = source.elements[source.observations % source.atoms]
        voxel = torch.from_numpy(np.repeat(
            np.arange(len(source.voxels)), source.count_per_voxel,
        )).to(element.device)
        indices = torch.stack((4 * voxel + element, source.frame_index))
        counts = torch.sparse_coo_tensor(
            indices, source.mass * source.dv, (4 * len(source.voxels), source.frames),
        )
        return cls(counts)

    @torch.no_grad()
    def loss_and_tangent(self, weights, excess, return_tangent=True):
        # A local potential on occupation N changes weights by -w(N-<N>).
        # With uniform canonical base weights, beta*V(frame) = -log(w) + const.
        # That constant drops out on every canonical tangent.
        weights = weights / weights.sum(-1, keepdim=True)
        mean = (self.counts @ weights.T).T
        tiny = torch.finfo(weights.dtype).tiny
        log_mean = torch.where(mean > 0, mean.clamp_min(tiny).log(), 0.)
        reference = (self.transpose @ log_mean.T).T - weights.clamp_min(tiny).log()

        def covariance(values):
            centered = values - (weights * values).sum(-1, keepdim=True)
            return (self.counts @ (weights * centered).T).T

        second = (self.second @ weights.T).T
        variance = second - mean.square()
        # Ignore zero-variance sites and cancellation at roundoff level.
        active = variance > (8 * torch.finfo(weights.dtype).eps) * second
        inv_std = torch.where(active, variance.clamp_min(tiny).rsqrt(), 0.)
        size = active.sum(-1, keepdim=True).clamp_min(1)
        scale = ((covariance(reference) * inv_std).square().sum(-1, keepdim=True)
                 / size).clamp_min(1.e-6)
        residual = covariance(excess + reference) * inv_std
        loss = (residual.square().sum(-1, keepdim=True) / (size * scale)).mean()
        tangent = None
        if return_tangent:
            site_grad = 2 * residual * inv_std / (size * scale * len(weights))
            by_frame = (self.transpose @ site_grad.T).T
            tangent = weights * (by_frame - (weights * by_frame).sum(-1, keepdim=True))
        return loss, tangent


class LocalELTileEnergy:
    def tile_energy(self, fields, shape):
        features = self.angular_invariants(fields[:, :, 0])
        h = features.reshape(len(fields), *shape, self.features)
        for index, layer in enumerate(self.layers):
            z = F.linear(h, layer.weight)
            if index:
                z = valid_conv(z, self.context[index - 1])
            h = F.softplus(z + layer.bias) - F.softplus(layer.bias)
        return F.linear(h, self.output.weight).sum((1, 2, 3, 4))


class LocalELXi(LocalELTileEnergy, SixLayerSingleScaleXi):
    pass


class LocalELMultiscaleXi(LocalELTileEnergy, SixLayerMultiscaleXi):
    pass


def excess_frame_gradient(model, source, weights):
    gradient = torch.zeros_like(weights)
    for fields, shape in source.chunks(model, weights.unsqueeze(-1)):
        energy = model.tile_energy(fields, shape) * source.dv
        gradient += torch.autograd.grad(energy.sum(), weights)[0]
    return gradient


def local_backward(model, source, coeff, update_size):
    if not hasattr(source, 'local_occupancy'):
        source.local_occupancy = LocalOccupancy.from_source(source)
    weights = coeff[:, :, 0].detach().requires_grad_()
    gradient = excess_frame_gradient(model, source, weights)
    loss, tangent = source.local_occupancy.loss_and_tangent(weights, gradient)
    if not torch.isfinite(loss) or not torch.isfinite(tangent).all():
        raise FloatingPointError('Nonfinite local equilibrium loss')
    local_coeff = torch.stack((weights.detach(), tangent), dim=-1)
    for fields, shape in source.chunks(model, local_coeff):
        first, _ = model.volume_jets(fields, shape)
        (-first.sum() * source.dv * (len(coeff) / update_size)).backward()
    return loss


def batch(model, source, coeff, targets, scales, block, update_size, correction=None):
    if update_size is not None:
        params = tuple(model.parameters())
        old = [None if p.grad is None else p.grad.detach().clone() for p in params]
        while True:
            try:
                loss = local_backward(model, source, coeff, update_size)
                break
            except RuntimeError as exc:
                oom = isinstance(exc, torch.cuda.OutOfMemoryError) or 'CUFFT_ALLOC_FAILED' in str(exc)
                if not oom or source.block <= 4:
                    raise
            for p, previous in zip(params, old):
                p.grad = None if previous is None else previous.clone()
            source.block -= 2
            source.transfer_cache.clear()
            source.observation_cache.clear()
            source.observation_cache_bytes = 0
            gc.collect()
            torch.cuda.empty_cache()
            emit(stage='retry_local_el_tile', block=source.block)
        if not hasattr(source, 'local_measurements'):
            source.local_measurements = 0
        if source.local_measurements % 50 == 0:
            emit(stage='local_el', case=source.local_case, state_offset=source.local_measurements,
                 local_el_loss=float(loss), weight=1.)
        source.local_measurements += len(coeff)
    return projection_batch(model, source, coeff, targets, scales, block, update_size, correction)


class LocalELSource(SixLayerSource):
    def __init__(self, case, *args):
        super().__init__(case, *args)
        self.local_case = case['case']


def entry():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--variant', choices=('single', 'multiscale'), default='single')
    args, remaining = parser.parse_known_args()
    model, bulk = {
        'single': (LocalELXi, SixLayerSingleScaleBulk),
        'multiscale': (LocalELMultiscaleXi, SixLayerMultiscaleBulk),
    }[args.variant]

    groups = json.loads(INDEX.read_text())['splits']
    labels = {case: path for split in groups.values() for case, path in split.items()}

    def prepare(case, data, device):
        return bulk(case, data, device, labels)

    main(model_factory=model, case_factory=prepare, source_factory=LocalELSource,
         batch_fn=batch, save_case_checkpoints=True, argv=remaining)


if __name__ == '__main__':
    entry()
