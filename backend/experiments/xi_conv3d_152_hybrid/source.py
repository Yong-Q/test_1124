"""Memory-bounded short FFTs and one periodic long FFT per projection pass."""

from itertools import product

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from experiments.xi_conv3d_152_fft_trials.operators import TrialSource
from experiments.xi_conv3d_sixlayer.network import HALO


class HybridSource(TrialSource):
    directions_per_fft = 8

    def shared_tile(self, model, coeff, lower, upper, halo, long):
        indices = [torch.arange(h, h+n, device=coeff.device) for h, n in zip(halo, upper-lower)]
        prepared = self.prepare(model, coeff, lower, upper, halo, long)
        convolve = super().convolve

        def part_fields(part):
            return convolve(model, part, prepared, indices, long)

        if torch.is_grad_enabled() and coeff.shape[-1] > self.directions_per_fft:
            return torch.cat([checkpoint(part_fields, part, use_reentrant=False, preserve_rng_state=False)
                              for part in coeff.split(self.directions_per_fft, -1)], dim=2)
        return part_fields(coeff)

    def chunks(self, model, coeff):
        limit = self.shape if self.region is None else np.minimum(self.region, self.shape)
        by_weights = torch.is_grad_enabled() and coeff.requires_grad
        global_long = model.multiscale and not by_weights
        long_field = None
        if global_long:
            zero = np.zeros(3, dtype=int)
            long_field = self.global_crop(model, coeff, zero, self.shape, True)
            long_values = long_field.detach()
            long_grad = torch.zeros_like(long_values, memory_format=torch.contiguous_format) if long_field.requires_grad else None

        for start in product(*(range(0, n, self.block) for n in limit)):
            lo = np.asarray(start)
            hi = np.minimum(lo+self.block, limit)
            lower, upper = lo-HALO, hi+HALO
            short = self.shared_tile(model, coeff, lower, upper, self.short_halo, False)
            if global_long:
                x, y, z = [torch.arange(a, b, device=coeff.device) % n
                           for a, b, n in zip(lower, upper, self.shape)]
                indices = ((x[:, None, None]*self.shape[1]+y[None, :, None])*self.shape[2]+z[None, None, :]).flatten()
                tile = long_values.index_select(1, indices)
                if long_grad is not None:
                    tile.requires_grad_()

                    def accumulate(gradient, ids=indices):
                        long_grad.index_add_(1, ids, gradient)

                    tile.register_hook(accumulate)
                short = torch.cat((short, tile), -1)
            elif model.multiscale:
                # EL's frame-weight VJP must remain connected inside each tile.
                long = self.shared_tile(model, coeff, lower, upper, self.long_halo, True)
                short = torch.cat((short, long), -1)
            yield short, tuple(upper-lower)

        if long_field is not None and long_grad is not None:
            long_field.backward(long_grad)
