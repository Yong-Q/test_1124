"""Fit a state-independent C2 kernel and integrate it analytically for C1."""

import numpy as np
import torch
from torch import nn

from experiments.xi_continuous_pair_mlp.network import PairMLP, SCALARS
from experiments.xi_conv3d_152_shared_region.source import SharedRegionSource, blocks
from experiments.xi_conv3d_sixlayer.network import overlap


class DirectPair(PairMLP):
    def __init__(self, shape=None, spacing=None):
        super().__init__(shape, spacing)
        self.layers = nn.ModuleList([nn.Identity()])
        del self.output
        self.reference = nn.Parameter(torch.zeros_like(self.scalar_pair))

    def curvature(self, field):
        scalar, angular, long = self.blocks(field)
        upper = self.angular_pair.new_zeros((len(angular), self.rank, self.rank))
        i, j = self.triangle
        upper[:, i, j] = self.angular_pair.reshape(len(angular), -1)
        matrices = upper + upper.mT
        mixed = [(matrix @ values).flatten(-2) for matrix, values in zip(matrices, angular)]
        return torch.cat((scalar*self.scalar_pair[:SCALARS], *mixed, long*self.scalar_pair[SCALARS:]), -1)

    def energy_gradient(self, field):
        anchor = torch.cat((-self.reference[:SCALARS], field.new_zeros(self.short_channels-SCALARS),
                            -self.reference[SCALARS:]))
        return self.curvature(field) + anchor

    def energy_density(self, fields):
        scalar = self.scalar_features(fields)
        return .5*(fields*self.curvature(fields)).sum(-1) - scalar @ self.reference

    def volume_jets(self, fields, shape):
        mean, tangent = fields[:, :, 0], fields[:, :, 1:]
        first = (tangent*self.energy_gradient(mean).unsqueeze(2)).sum((1, 3))
        second = overlap(tangent, self.curvature(tangent))
        return -first, -second


class DirectPairSource(SharedRegionSource):
    @torch.no_grad()
    def frame_energy_gradient(self, model, weights):
        """Apply the transpose of deposition, smoothing and mean subtraction."""
        by_frame = torch.zeros_like(weights)
        coeff = weights.unsqueeze(-1)
        limit = self.shape if self.region is None else np.minimum(self.shape, self.region)
        for lo, hi in blocks(np.zeros(3, dtype=int), limit, self.region_edge):
            branches = []
            fields = []
            for long, halo in ((False, self.short_halo), (True, self.long_halo)):
                prepared = self.prepare(model, coeff, lo, hi, halo, long)
                indices = [torch.arange(h, h+n, device=weights.device) for h, n in zip(halo, hi-lo)]
                fields.append(self.convolve(model, coeff, prepared, indices, long)[:, :, 0])
                branches.append(prepared)
            short, long = (model.energy_gradient(torch.cat(fields, -1))*self.dv).split(
                (model.short_channels, model.long_channels), -1)
            for is_long, gradient, (shape, tiers), halo, mix in (
                (False, short, branches[0], self.short_halo, model.short_weights()),
                (True, long, branches[1], self.long_halo, model.long_mix),
            ):
                spectrum = self.kernel_transfer(shape, is_long, weights)
                transfer = (mix @ spectrum.flatten(1)).reshape(len(mix), *spectrum.shape[-3:])
                window = tuple(slice(h, h+n) for h, n in zip(halo, hi-lo))
                padded = gradient.new_zeros((len(weights), len(mix), *shape))
                padded[(..., *window)] = gradient.mT.reshape(len(weights), len(mix), *(hi-lo))
                adjoint = torch.fft.irfftn(torch.fft.rfftn(padded, dim=(-3, -2, -1))*transfer,
                                          s=shape, dim=(-3, -2, -1)).flatten(2).mT
                for destinations, frame_ids, encoded in tiers:
                    pullback = torch.einsum('bvc,vnc->bvn', adjoint[:, destinations], encoded)
                    by_frame.scatter_add_(1, frame_ids.flatten().expand(len(weights), -1), pullback.flatten(1))
                if is_long:
                    by_frame -= (gradient.sum(1)*(self.count_density*mix.sum(1))).sum(1, keepdim=True)
        return by_frame
