"""Sample principal Hessian blocks while retaining all first derivatives."""

from dataclasses import dataclass
import gc
import hashlib

import numpy as np
import torch

from experiments.xi_all_projection_loss.train import Metrics, SEED, emit
from experiments.xi_conv3d_local_el.train import LocalOccupancy, batch as full_batch, excess_frame_gradient
from experiments.xi_conv3d_sixlayer.train import pass_once


@dataclass
class SampledPrediction:
    c1: torch.Tensor
    c2: torch.Tensor
    indices: np.ndarray
    inverse_inclusion: np.ndarray | None = None


def inverse_probabilities(p, k, like):
    if not 2 <= k <= p:
        raise ValueError('Expected between 2 and P directions')
    probs = like.new_full((k, k), p*(p-1)/(k*(k-1)))
    return probs.fill_diagonal_(p/k)


def stratified_probes(rng, keep):
    """Sample nonempty Hessian blocks with known entry inclusion probabilities."""
    spatial = np.arange(24)
    angular = np.arange(24, 152).reshape(8, 16)
    if keep == 8:
        strata = [(spatial, 2), (angular.ravel(), 6)]
    elif keep in (16, 24):
        angular_counts = {16: (2, 3, 3, 4), 24: (2, 4, 5, 7)}[keep]
        strata = [(spatial, keep//4)] + [
            (angular[:, l*l:(l+1)**2].ravel(), count) for l, count in enumerate(angular_counts)
        ]
    else:
        raise ValueError('Stratified sampling supports 8, 16 or 24 probes')
    indices = np.concatenate([rng.choice(group, count, replace=False) for group, count in strata])
    sizes = np.concatenate([np.full(count, len(group)) for group, count in strata])
    counts = np.concatenate([np.full(count, count) for _, count in strata])
    groups = np.repeat(np.arange(len(strata)), [count for _, count in strata])
    inv = np.outer(sizes/counts, sizes/counts)
    same = groups[:, None] == groups[None, :]
    inv = np.where(same, (sizes*(sizes-1)/(counts*(counts-1)))[:, None], inv)
    np.fill_diagonal(inv, sizes/counts)
    order = indices.argsort()
    return indices[order], inv[np.ix_(order, order)]


class SampledMetrics(Metrics):
    def update(self, targets, prediction):
        if not isinstance(prediction, SampledPrediction):
            return super().update(targets, prediction)
        first, second = (y.astype(np.float64) for y in targets)
        indices = prediction.indices
        self.n += len(first)
        self.moments[0][0] += np.square(prediction.c1.detach().cpu().numpy() - first).sum(0)
        inv = prediction.inverse_inclusion
        estimated = prediction.c2.detach().cpu().numpy()
        if indices.ndim == 1:
            sub = np.ix_(indices, indices)
            if inv is None:
                inv = inverse_probabilities(first.shape[1], len(indices), prediction.c2).cpu().numpy()
            self.moments[1][0][sub] += np.square(estimated-second[:, sub[0], sub[1]]).sum(0)*inv
        else:
            for ref, pred, directions, weights in zip(second, estimated, indices, inv):
                sub = np.ix_(directions, directions)
                self.moments[1][0][sub] += np.square(pred-ref[sub])*weights
        for moment, target in zip(self.moments, (first, second)):
            moment[1] += target.sum(0)
            moment[2] += np.square(target).sum(0)


def first_and_el(model, source, coeff, target, scale, fraction):
    if not hasattr(source, 'local_occupancy'):
        source.local_occupancy = LocalOccupancy.from_source(source)
    weights = coeff[:, :, 0].detach()
    frame_adjoint = getattr(source, 'frame_energy_gradient', None)
    if frame_adjoint is None:
        frame_gradient = excess_frame_gradient(model, source, weights.requires_grad_())
    else:
        frame_gradient = frame_adjoint(model, weights)
    loss, el_tangent = source.local_occupancy.loss_and_tangent(weights, frame_gradient)
    directions = coeff[:, :, 1:]
    predicted = -(directions.mT @ frame_gradient.unsqueeze(-1)).squeeze(-1)
    cotangent = 2.*fraction*(predicted-target)/(target.numel()*scale**2)
    tangent = (directions @ cotangent.unsqueeze(-1)).squeeze(-1) - fraction*el_tangent
    if not all(torch.isfinite(t).all() for t in (predicted, loss, tangent)):
        raise FloatingPointError('Nonfinite first derivative or local EL residual')
    combined = torch.stack((weights.detach(), tangent.detach()), dim=-1)
    for fields, shape in source.chunks(model, combined):
        first, _ = model.volume_jets(fields, shape)
        (first.sum()*source.dv).backward()
    return predicted.detach(), loss


def sampled_pass(model, source, coeff, targets, scales, update_size, indices, *, stage=None, inverse_inclusion=None):
    p = targets[0].shape[-1]
    b, fraction = len(coeff), len(coeff)/update_size
    first, el_loss = first_and_el(model, source, coeff, targets[0], scales[0], fraction)
    if stage is not None:
        stage('full_c1_and_el')
    column = torch.as_tensor(indices, device=coeff.device, dtype=torch.long)
    small = torch.cat((coeff[:, :, :1], coeff[:, :, 1:].index_select(-1, column)), dim=-1)
    target = targets[1].index_select(1, column).index_select(2, column)
    target_shape = (targets[0].index_select(1, column), target)
    _, predicted = pass_once(model, source, small, target_shape, scales, update_size=None)
    prob = (inverse_probabilities(p, len(indices), predicted) if inverse_inclusion is None
            else torch.as_tensor(inverse_inclusion, device=predicted.device, dtype=predicted.dtype))
    factor = 2.*fraction/(b*p*p*scales[1]**2)
    cotangent = factor*(predicted-target)*prob
    for fields, shape in source.chunks(model, small):
        _, c2 = model.volume_jets(fields, shape)
        ((c2*cotangent).sum()*source.dv).backward()
    if stage is not None:
        stage('sampled_c2')
    return SampledPrediction(first, predicted, indices, inverse_inclusion), el_loss


def batch_factory(keep, *, stratified=False):
    samplers = {}

    def batch(model, source, coeff, targets, scales, block, update_size, correction=None):
        p = targets[0].shape[-1]
        if update_size is None or keep >= p:
            return full_batch(model, source, coeff, targets, scales, block, update_size, correction)
        if source.local_case not in samplers:
            token = hashlib.sha256(f'{SEED}:c2:{source.local_case}'.encode()).digest()[:8]
            samplers[source.local_case] = np.random.default_rng(int.from_bytes(token, 'little'))
        rng = samplers[source.local_case]
        if stratified:
            if p != 152:
                raise ValueError('Expected the 152-probe basis for stratified sampling')
            indices, inv = stratified_probes(rng, keep)
        else:
            indices, inv = np.sort(rng.choice(p, keep, replace=False)), None
        parameters = tuple(model.parameters())
        old = [q.grad.detach().clone() if q.grad is not None else None for q in parameters]
        while True:
            try:
                prediction, loss = sampled_pass(model, source, coeff, targets, scales, update_size, indices,
                                                inverse_inclusion=inv)
            except RuntimeError as exc:
                oom = isinstance(exc, torch.cuda.OutOfMemoryError) or 'CUFFT_ALLOC_FAILED' in str(exc)
                if not oom or source.block <= 4:
                    raise
            else:
                count = getattr(source, 'local_measurements', 0)
                if count % 50 == 0:
                    emit(stage='local_el', case=source.local_case, state_offset=count, local_el_loss=float(loss),
                         weight=1., sampled_c2_directions=indices.tolist())
                source.local_measurements = count + len(coeff)
                return prediction
            for param, prior in zip(parameters, old):
                param.grad = prior.clone() if prior is not None else None
            source.block = max(4, source.block-2)
            if hasattr(source, 'region_edge'):
                source.region_edge = min(source.region_edge, 16)
            source.transfer_cache.clear()
            source.observation_cache.clear()
            source.observation_cache_bytes = 0
            gc.collect()
            torch.cuda.empty_cache()
            emit(stage='retry_sampled_c2_tile', block=source.block)
    return batch
