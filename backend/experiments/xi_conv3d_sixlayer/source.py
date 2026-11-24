"""The first convolution supplies an exact five-voxel halo for later layers."""

from itertools import product

import numpy as np
import torch

from experiments.xi_conv3d.network import STEPS
from experiments.xi_conv3d_multiscale.source import MultiscaleTrajectory
from experiments.xi_conv3d_sixlayer.network import HALO


class SixLayerSource(MultiscaleTrajectory):
    def chunks(self, model, coeff):
        for start in product(*(range(0, n, self.block) for n in self.shape)):
            lo = np.asarray(start)
            hi = np.minimum(lo+self.block,self.shape)
            outer_lo, outer_hi = lo-HALO, hi+HALO
            field = self.filtered_tile(model,coeff,outer_lo,outer_hi,self.short_halo,False)
            if model.multiscale:
                long = self.filtered_tile(model,coeff,outer_lo,outer_hi,self.long_halo,True)
                field = torch.cat((field,long),dim=-1)
            yield field, tuple(outer_hi-outer_lo)


class SixLayerAxisSource(SixLayerSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.short_halo = np.full(3,max(STEPS),dtype=int)

    def kernel_transfer(self, shape, long, like):
        if long:
            raise ValueError('Axis model has no separate long-distance branch')
        key = (shape, False)
        if key not in self.transfer_cache:
            kernel = like.new_zeros((1+len(STEPS), *shape))
            kernel[0,0,0,0] = 1.
            for shell, step in enumerate(STEPS,1):
                for axis in range(3):
                    for sign in (-1,1):
                        v = [0,0,0]
                        v[axis] = sign*step % shape[axis]
                        kernel[(shell,*v)] += 1./6
            self.transfer_cache[key] = torch.fft.rfftn(kernel,dim=(-3,-2,-1)).real
        return self.transfer_cache[key]
