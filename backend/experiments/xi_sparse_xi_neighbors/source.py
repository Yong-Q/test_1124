"""Aggregate the existing Xi components on a sparse 3D neighbor stencil."""

import numpy as np
import torch
from torch.nn import functional as F

from experiments.xi_conv3d_152_direct_fft.source import lift
from experiments.xi_conv3d_152_shared_region.source import SharedRegionSource
from experiments.xi_conv3d_multiscale.network import (
    LONG_CUTOFF_A, LONG_WIDTHS_A, SHORT_CUTOFF_A, SHORT_WIDTHS_A,
)


class SparseXiSource(SharedRegionSource):
    """Use edge distances to weight the Xi carried by axial and diagonal neighbors."""

    def __init__(self, *args):
        super().__init__(*args)
        self.sampling = {}
        self.short_halo = self.register_stencil(False, (.65, 2.5), SHORT_WIDTHS_A, SHORT_CUTOFF_A)
        self.long_halo = self.register_stencil(True, (4.5, 6.5), LONG_WIDTHS_A, LONG_CUTOFF_A)

    def register_stencil(self, long, radii, widths, cutoff):
        self.sampling[long] = radial_stencil(self.spacing, long, radii, widths, cutoff)
        return np.maximum.reduce([step for step,_ in self.sampling[long]])

    def shared_tile(self, model, coeff, lower, upper, halo, long):
        shape, tiers = self.prepare(model, coeff, lower, upper, halo, long)
        channels = model.long_channels if long else model.short_channels
        mix = model.long_mix if long else model.short_weights()
        kernels = []
        for step, basis in self.sampling[long]:
            weight = (mix @ coeff.new_tensor(basis).flatten(1)).reshape(channels,1,3,3,3)
            window = tuple(slice(int(h-d),int(h+n+d)) for h,n,d in zip(halo,upper-lower,step))
            kernels.append((weight, window, tuple(map(int, step))))
        tiles = []
        for part in coeff.split(self.directions_per_fft, -1):
            field = lift(part, tiers, channels, shape)
            result = sum(F.conv3d(field[(..., *window)].contiguous(), weight,
                                  dilation=step, groups=channels) for weight,window,step in kernels)
            values = result.flatten(2).mT.reshape(len(part),part.shape[-1],-1,channels)
            if long:
                mean = part.sum(1)[...,None] * self.count_density
                values = values-(mean*mix.sum(-1))[:,:,None,:]
            tiles.append(values)
        return torch.cat(tiles,1).transpose(1,2)


def radial_stencil(spacing, long, radii, widths, cutoff):
    steps = tuple(np.maximum(1, np.rint(r/spacing).astype(int)) for r in radii)
    axis = np.arange(-1, 2)
    offsets = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1)
    basis = []
    for index, step in enumerate(steps):
        distance = np.linalg.norm(offsets*(step*spacing), axis=-1)
        u = np.minimum(distance/cutoff, 1.)
        envelope = (1-u)**3*(1+3*u+6*u**2+10*u**3+15*u**4+21*u**5)
        if long:
            turn = np.clip((distance-4.)/2., 0., 1.)
            envelope *= turn**3*(10.-15.*turn+6.*turn**2)
        elif index:
            envelope[1,1,1] = 0.
        sigma = np.asarray(widths)[:, None, None, None]
        basis.append(envelope*np.exp(-.5*(distance/sigma)**2))
    normalizer = np.sum(basis, axis=(0,2,3,4))
    if np.any(normalizer <= 0):
        raise ValueError('Neighbor stencil has no support for a radial basis')
    return tuple((step, weight/normalizer[:,None,None,None]) for step, weight in zip(steps,basis))
