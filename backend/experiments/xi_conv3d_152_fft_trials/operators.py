"""Equivalent first-layer operators for FFT and deposition benchmarks."""

from itertools import product

import numpy as np
import torch
from torch.nn.functional import one_hot

from experiments.xi_conv3d_152_fast.source import GroupedLocalELSource


class TrialSource(GroupedLocalELSource):
    def prepare(self, model, coeff, lower, upper, halo, long):
        shape, tiers = self.tile_observations(lower, upper, halo, coeff.device)
        encoded_tiers = []
        for destinations, ids, active in tiers:
            observations = self.observations[ids]
            species = one_hot(self.elements[observations % self.atoms], num_classes=4).to(coeff)
            if long:
                encoded = species
            else:
                marks = self.marks[observations]
                encoded = model.encode(torch.cat((torch.ones_like(marks[..., :1]), marks), -1), species)
            encoded_tiers.append((destinations, self.frame_index[ids], encoded * (self.mass[ids] * active)[..., None]))
        return tuple(shape), encoded_tiers

    def convolve(self, model, coeff, prepared, indices, long):
        shape, tiers = prepared
        channels = 4 if long else model.short_channels
        basis = self.kernel_transfer(shape, long, coeff)
        weights = model.long_mix if long else model.short_weights()
        transfer = (weights @ basis.flatten(1)).reshape(channels, *basis.shape[-3:])
        b = coeff.shape[0]
        pieces = []
        for part in coeff.split(self.directions_per_fft, -1):
            p = part.shape[-1]
            lifted = coeff.new_zeros((int(np.prod(shape)), b * p, channels))
            for destinations, frame_ids, encoded in tiers:
                directions = part[:, frame_ids].permute(1, 0, 3, 2).flatten(1, 2)
                lifted.index_copy_(0, destinations, torch.bmm(directions, encoded))
            field = lifted.permute(1, 2, 0).reshape(b * p, channels, *shape)
            chunks = []
            for start in range(0, channels, 32):
                end = min(start + 32, channels)
                spectrum = torch.fft.rfftn(field[:, start:end].contiguous(), dim=(-3, -2, -1))
                result = torch.fft.irfftn(spectrum * transfer[start:end], s=shape, dim=(-3, -2, -1))
                for axis, ids in enumerate(indices, start=2):
                    result = result.index_select(axis, ids)
                chunks.append(result.flatten(2).transpose(1, 2))
            cropped = torch.cat(chunks, -1).reshape(b, p, -1, channels)
            if long:
                mean = part.sum(1)[..., None] * self.count_density
                cropped = cropped - (mean * weights.sum(-1))[:, :, None, :]
            pieces.append(cropped)
        return torch.cat(pieces, 1).transpose(1, 2)

    def shared_tile(self, model, coeff, lower, upper, halo, long):
        indices = [torch.arange(h, h + n, device=coeff.device) for h, n in zip(halo, upper - lower)]
        return self.convolve(model, coeff, self.prepare(model, coeff, lower, upper, halo, long), indices, long)

    def global_crop(self, model, coeff, lower, upper, long):
        zero = np.zeros(3, dtype=int)
        indices = [torch.arange(a, b, device=coeff.device) % n for a, b, n in zip(lower, upper, self.shape)]
        return self.convolve(model, coeff, self.prepare(model, coeff, zero, self.shape, zero, long), indices, long)


def boxes(region, block, halo):
    """Core blocks plus CNN halo, in the same order as the existing source."""
    for start in product(range(0, region, block), repeat=3):
        start = np.array(start)
        yield start - halo, np.minimum(start + block, region) + halo
