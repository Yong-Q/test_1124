"""Exact overlap-save FFT convolution of reweighted atomic fields."""

from itertools import product

import h5py
import numpy as np
import torch

from experiments.xi_conv3d.network import ANGULAR, SCALARS
from experiments.xi_conv3d_multiscale.network import (
    LONG_CUTOFF_A, LONG_WIDTHS_A, SHORT_CUTOFF_A, SHORT_WIDTHS_A, radial_bank,
)
from kernel import box_lengths
from xi_response import deposition_stencil


DIMS = SCALARS + sum(ANGULAR)


class MultiscaleTrajectory:
    def __init__(self, case, data, charges, device, block):
        self.dv = float(data['voxel_volume'])
        self.shape = np.asarray(data['shape'], dtype=int)
        steps = data['source_steps']
        with h5py.File(case['cache'], 'r') as h:
            np.testing.assert_array_equal(h['step'][:], steps)
            if h.attrs['radial_basis'] != 'zero_centered_gaussian':
                raise ValueError('Expected the zero-centered 129-channel cache')
            cell = h['cell'][0]
            self.spacing = box_lengths(cell) / self.shape
            np.testing.assert_allclose(self.dv, np.prod(self.spacing))
            elements = h['elements'][:]
            coords = h['positions'][:]
            marks = h['atomic_xi_nlm'][:].reshape(-1, 128)
        self.counts = torch.bincount(torch.as_tensor(elements, device=device, dtype=torch.long),
                                     minlength=4)
        self.count_density = (self.counts / (np.prod(self.shape) * self.dv)).float()
        self.atoms = len(elements)
        self.frames = len(steps)
        self.marks = torch.as_tensor(marks, device=device, dtype=torch.float32)
        self.elements = torch.as_tensor(elements, device=device, dtype=torch.long)
        bins, mass = deposition_stencil(coords, cell, self.shape, 'smoothstep')
        active = np.flatnonzero(mass.ravel() > 0)
        voxel_ids = bins.ravel()[active]
        order = np.argsort(voxel_ids, kind='stable')
        active = active[order]
        self.voxels, self.count_per_voxel = np.unique(voxel_ids, return_counts=True)
        self.offset = np.r_[0, self.count_per_voxel.cumsum()]
        frame_atom = active // mass.shape[-1]
        self.observations = torch.as_tensor(frame_atom, dtype=torch.long, device=device)
        self.frame_index = torch.as_tensor(frame_atom // len(elements), device=device, dtype=torch.long)
        self.mass = torch.as_tensor(mass.ravel()[active] / self.dv, device=device, dtype=torch.float32)
        self.lookup = np.full(int(np.prod(self.shape)), -1, dtype=np.int32)
        self.lookup[self.voxels] = np.arange(len(self.voxels), dtype=np.int32)
        self.lookup = self.lookup.reshape(self.shape)
        self.block = block
        self.short_halo = np.ceil(SHORT_CUTOFF_A / self.spacing).astype(int)
        self.long_halo = np.ceil(LONG_CUTOFF_A / self.spacing).astype(int)
        self.transfer_cache = {}
        self.observation_cache = {}
        self.observation_cache_bytes = 0

    def tile_observations(self, lower, upper, halo, device):
        key = (tuple(lower), tuple(upper), tuple(halo))
        if key in self.observation_cache:
            shape, cpu = self.observation_cache[key]
        else:
            shape = tuple(upper - lower + 2 * halo)
            indices = np.ix_(*(np.arange(a-h, b+h) % n for a, b, h, n in
                               zip(lower, upper, halo, self.shape)))
            refs = self.lookup[indices].ravel()
            positions = np.flatnonzero(refs >= 0)
            voxels = refs[positions]
            counts = self.count_per_voxel[voxels]
            cpu = []
            powers = np.ceil(np.log2(counts)).astype(int)
            for power in np.unique(powers):
                select = np.flatnonzero(powers == power)
                cols = np.arange(counts[select].max())
                valid = cols < counts[select, None]
                ids = self.offset[voxels[select], None] + cols * valid
                cpu.append((positions[select].astype(np.int32), ids.astype(np.int32), valid))
            used = sum(array.nbytes for tier in cpu for array in tier)
            if self.observation_cache_bytes + used <= 4 * 2**30:
                self.observation_cache[key] = (shape, cpu)
                self.observation_cache_bytes += used
        return shape, [(torch.as_tensor(dst, device=device, dtype=torch.long),
                        torch.as_tensor(ids, device=device),
                        torch.as_tensor(valid, device=device))
                       for dst, ids, valid in cpu]

    def kernel_transfer(self, shape, long, like):
        key = (shape, long)
        if key not in self.transfer_cache:
            widths = LONG_WIDTHS_A if long else SHORT_WIDTHS_A
            cutoff = LONG_CUTOFF_A if long else SHORT_CUTOFF_A
            kernel = radial_bank(shape, self.spacing, widths, cutoff, long=long)
            transfer = torch.fft.rfftn(kernel.float(), dim=(-3, -2, -1)).real
            self.transfer_cache[key] = transfer.to(like)
        return self.transfer_cache[key]

    def filtered_tile(self, model, coeff, lower, upper, halo, long):
        shape, tiers = self.tile_observations(lower, upper, halo, coeff.device)
        b, _, cols = coeff.shape
        channels = 4 if long else getattr(model, 'short_channels', DIMS)
        size = int(np.prod(shape))
        lifted = coeff.new_zeros((size, b * cols, channels))
        for destinations, ids, active in tiers:
            observations = self.observations[ids]
            groups = torch.nn.functional.one_hot(
                self.elements[observations % self.atoms], num_classes=4).to(coeff)
            if long:
                encoded = groups
            else:
                marks = self.marks[observations]
                marked = torch.cat((torch.ones_like(marks[..., :1]), marks), dim=-1)
                encoded = model.encode(marked, groups)
            encoded = encoded * (self.mass[ids] * active)[..., None]
            weights = coeff[:, self.frame_index[ids]].permute(1, 0, 3, 2).flatten(1, 2)
            projected = torch.bmm(weights, encoded)
            lifted.index_copy_(0, destinations, projected)

        basis = self.kernel_transfer(shape, long, coeff)
        weights = model.long_mix if long else model.short_weights()
        transfer = (weights @ basis.flatten(1)).reshape(channels, *basis.shape[-3:])
        field = lifted.permute(1, 2, 0).reshape(b*cols, channels, *shape)
        # Grouping the FFT's batch axes avoids holding extra spectral workspaces.
        window = tuple(slice(h, h+n) for h, n in zip(halo, upper-lower))
        outputs = []
        for sub in field.split(4, dim=0):
            spectrum = torch.fft.rfftn(sub.contiguous(), dim=(-3, -2, -1))
            result = torch.fft.irfftn(spectrum * transfer, s=shape, dim=(-3, -2, -1))
            outputs.append(result[(..., *window)].flatten(2).transpose(1, 2).contiguous())
        result = torch.cat(outputs).reshape(b, cols, -1, channels)
        if long:
            mean = coeff.sum(1)[..., None] * self.count_density
            result = result - (mean * weights.sum(-1))[:, :, None, :]
        return result.transpose(1, 2)

    def chunks(self, model, coeff):
        for start in product(*(range(0, n, self.block) for n in self.shape)):
            lower = np.asarray(start)
            upper = np.minimum(lower + self.block, self.shape)
            short = self.filtered_tile(model, coeff, lower, upper, self.short_halo, False)
            long = self.filtered_tile(model, coeff, lower, upper, self.long_halo, True)
            yield torch.cat((short, long), dim=-1)
