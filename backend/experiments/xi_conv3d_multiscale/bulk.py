"""Homogeneous element matrix matching for the exact two-scale convolution."""

import numpy as np
import torch

from experiments.xi_conv3d.bulk import BulkCase
from experiments.xi_conv3d.network import FEATURES, SCALARS
from experiments.xi_conv3d_multiscale.network import (
    LONG_CUTOFF_A, LONG_WIDTHS_A, SHORT_CUTOFF_A, SHORT_WIDTHS_A, radial_bank,
)


class MultiscaleBulk(BulkCase):
    def __init__(self, case, data, device, labels, include_long=True):
        super().__init__(case, data, device, labels)
        with np.load(labels[case['case']]) as reference:
            shape = tuple(reference['shape'])
            spacing = reference['lengths_A'] / reference['shape']
            resolved = reference['reliable'].any(-1)
            integer = reference['mode_indices'][resolved]
            inds = tuple((integer[:, d] % n) for d, n in enumerate(shape))

        def modes(widths, radius, long):
            bank = radial_bank(shape, spacing, widths, radius, long=long)
            values = []
            for kernel in bank:
                spectrum = torch.fft.rfftn(kernel.float()).real
                values.append(spectrum[inds])
            return torch.stack(values, dim=-1).to(device)

        self.short_modes = modes(SHORT_WIDTHS_A, SHORT_CUTOFF_A, False)
        self.long_modes = modes(LONG_WIDTHS_A, LONG_CUTOFF_A, True) if include_long else None

    def bulk_loss(self, model):
        mixed = model.scalar_mix(self.inputs)
        weights = model.spatial[:SCALARS]
        local_mean = (self.rho @ mixed) * weights.sum(1)
        mean = torch.cat((local_mean, local_mean.new_zeros(FEATURES - SCALARS + 4)))
        ids = torch.as_tensor([*range(SCALARS), *range(FEATURES, FEATURES+4)],
                              device=mean.device)
        directions = torch.eye(FEATURES+4, dtype=mean.dtype, device=mean.device)[ids]
        _, curvature, _ = model.invariant_jets(mean, directions)
        short = mixed[None] * (self.short_modes @ weights.T)[:, None, :]
        long = self.inputs[None, :, 1:5] * (self.long_modes @ model.long_mix.T)[:, None, :]
        d = torch.cat((short, long), dim=-1) * self.rho.sqrt()[None, :, None]
        h = torch.einsum('kec,cd,kfd->kef', d, curvature, d)
        h = h + torch.eye(len(self.rho), device=h.device, dtype=h.dtype)
        projected = self.basis.mH @ h.to(self.basis.dtype) @ self.basis
        error = (projected - self.targets) * self.pair_mask
        return error.abs().square().sum() / self.targets.square().sum()
