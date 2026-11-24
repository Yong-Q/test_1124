"""Reuse the kernel mix inside a tile, without storing whole-box fields."""

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from experiments.xi_conv3d_152_hybrid.source import HybridSource


class TileFFTSource(HybridSource):
    def shared_tile(self, model, coeff, lower, upper, halo, long):
        shape, tiers = self.prepare(model, coeff, lower, upper, halo, long)
        channels = 4 if long else model.short_channels
        basis = self.kernel_transfer(shape, long, coeff)
        weights = model.long_mix if long else model.short_weights()
        transfer = (weights @ basis.flatten(1)).reshape(channels, *basis.shape[-3:])
        window = tuple(slice(h, h+n) for h, n in zip(halo, upper-lower))
        mode_sum = weights.sum(-1) if long else weights.new_empty(0)
        cells = int(np.prod(shape))

        def field_group(part, spectral_kernel, dc):
            b, _, p = part.shape
            lifted = part.new_zeros((cells, b*p, channels))
            for destinations, frame_ids, encoded in tiers:
                direction = part[:, frame_ids].permute(1, 0, 3, 2).flatten(1, 2)
                lifted.index_copy_(0, destinations, torch.bmm(direction, encoded))
            field = lifted.permute(1, 2, 0).reshape(b*p, channels, *shape)
            outputs = []
            for start in range(0, channels, 32):
                end = min(start+32, channels)
                spectrum = torch.fft.rfftn(field[:, start:end].contiguous(), dim=(-3, -2, -1))
                result = torch.fft.irfftn(spectrum*spectral_kernel[start:end], s=shape, dim=(-3, -2, -1))
                outputs.append(result[(..., *window)].flatten(2).transpose(1, 2))
            cropped = torch.cat(outputs, -1).reshape(b, p, -1, channels)
            if long:
                mean = part.sum(1)[..., None] * self.count_density
                cropped = cropped - (mean*dc)[:, :, None, :]
            return cropped.transpose(1, 2)

        if coeff.shape[-1] <= self.directions_per_fft:
            return field_group(coeff, transfer, mode_sum)
        crops = []
        for part in coeff.split(self.directions_per_fft, -1):
            if torch.is_grad_enabled():
                crops.append(checkpoint(field_group, part, transfer, mode_sum,
                                        use_reentrant=False, preserve_rng_state=False))
            else:
                crops.append(field_group(part, transfer, mode_sum))
        return torch.cat(crops, dim=2)
