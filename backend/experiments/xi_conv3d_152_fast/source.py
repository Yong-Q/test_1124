"""Bound the FFT workspace while retaining the full set of direction fields."""

from itertools import product

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from experiments.xi_conv3d_local_el.train import LocalELSource
from experiments.xi_conv3d_sixlayer.network import HALO


class GroupedLocalELSource(LocalELSource):
    directions_per_fft = 4
    region = None

    def filtered_tile(self, model, coeff, lower, upper, halo, long):
        filter_tile = super().filtered_tile
        if coeff.shape[-1] <= self.directions_per_fft:
            return filter_tile(model, coeff, lower, upper, halo, long)

        def filtered(part):
            return filter_tile(model, part, lower, upper, halo, long)

        fields = [checkpoint(filtered, part, use_reentrant=False, preserve_rng_state=False)
                  if torch.is_grad_enabled() else filtered(part)
                  for part in coeff.split(self.directions_per_fft, dim=-1)]
        return torch.cat(fields, dim=2)

    def chunks(self, model, coeff):
        if self.region is None:
            yield from super().chunks(model, coeff)
            return

        # Benchmark only the same central-cell region for each tile size.
        # The convolution halos still read the real, full periodic geometry.
        limit = np.minimum(self.region, self.shape)
        for start in product(*(range(0, n, self.block) for n in limit)):
            lo = np.asarray(start)
            hi = np.minimum(lo + self.block, limit)
            lower, upper = lo - HALO, hi + HALO
            field = self.filtered_tile(model, coeff, lower, upper, self.short_halo, False)
            if model.multiscale:
                long = self.filtered_tile(model, coeff, lower, upper, self.long_halo, True)
                field = torch.cat((field, long), dim=-1)
            yield field, tuple(upper - lower)
