"""Reuse first-layer fields across neighboring tiles with bounded GPU storage."""

from itertools import product
import math

import numpy as np
import torch

from experiments.xi_conv3d_152_direct_fft.source import DirectFFTSource


def blocks(lower, upper, edge):
    for start in product(*(range(a, b, edge) for a, b in zip(lower, upper))):
        lo = np.asarray(start)
        yield lo, np.minimum(lo+edge, upper)


class SharedRegionSource(DirectFFTSource):
    region_edge = 16
    buffer_limit = 3 * 2**30

    def filtered(self, model, coeff, lo, hi):
        short = self.shared_tile(model, coeff, lo, hi, self.short_halo, False)
        if model.multiscale:
            long = self.shared_tile(model, coeff, lo, hi, self.long_halo, True)
            return torch.cat((short, long), -1)
        return short

    def chunks(self, model, coeff):
        halo = len(model.layers)-1
        limit = self.shape if self.region is None else np.minimum(self.region, self.shape)
        zero = np.zeros(3, dtype=int)
        channels = model.short_channels + model.long_channels
        storage = math.prod(np.minimum(limit, self.region_edge)+2*halo)
        storage *= len(coeff)*coeff.shape[-1]*channels*coeff.element_size()*2
        if coeff.requires_grad or storage > self.buffer_limit:
            # EL differentiates each tile directly with respect to frame weights.
            for lo, hi in blocks(zero, limit, self.block):
                lower, upper = lo-halo, hi+halo
                yield self.filtered(model, coeff, lower, upper), tuple(upper-lower)
            return

        for lo, hi in blocks(zero, limit, self.region_edge):
            lower, upper = lo-halo, hi+halo
            values = self.filtered(model, coeff, lower, upper)
            stored = values.detach()
            cotangent = torch.zeros_like(stored) if values.requires_grad else None
            _, ny, nz = upper-lower
            for child_lo, child_hi in blocks(lo, hi, self.block):
                start, stop = child_lo-halo, child_hi+halo
                x, y, z = [torch.arange(a-base, b-base, device=coeff.device)
                           for a, b, base in zip(start, stop, lower)]
                idx = ((x[:, None, None]*ny + y[None, :, None])*nz + z[None, None, :]).flatten()
                tile = stored.index_select(1, idx)
                if cotangent is not None:
                    tile.requires_grad_()

                    def collect(grad, ids=idx, buffer=cotangent):
                        buffer.index_add_(1, ids, grad)

                    tile.register_hook(collect)
                yield tile, tuple(stop-start)
            if cotangent is not None:
                values.backward(cotangent)
