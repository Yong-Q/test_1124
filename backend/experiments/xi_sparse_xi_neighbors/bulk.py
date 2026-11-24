"""Evaluate the sparse stencil exactly at the wavevectors used by the bulk loss."""

import numpy as np
import torch

from experiments.xi_continuous_pair_mlp.train import PairBulk
from experiments.xi_conv3d.bulk import BulkCase
from experiments.xi_sparse_xi_neighbors.source import (
    LONG_CUTOFF_A, LONG_WIDTHS_A, SHORT_CUTOFF_A, SHORT_WIDTHS_A, radial_stencil,
)


class SparseXiBulk(PairBulk):
    def __init__(self, case, data, device, labels):
        BulkCase.__init__(self, case, data, device, labels)
        with np.load(labels[case['case']]) as reference:
            shape = reference['shape']
            spacing = reference['lengths_A']/shape
            k = 2.*np.pi*reference['mode_indices'][reference['reliable'].any(-1)]/shape
        axis = np.arange(-1,2)
        offsets = np.stack(np.meshgrid(axis,axis,axis,indexing='ij'), -1).reshape(-1,3)

        def symbol(long, radii, widths, cutoff):
            stencil = radial_stencil(spacing,long,radii,widths,cutoff)
            modes = sum(np.cos(k @ (offsets*step).T) @ values.reshape(len(widths),-1).T
                        for step,values in stencil)
            if long:
                modes[(k==0).all(-1)] = 0.
            return torch.as_tensor(modes,device=device,dtype=self.rho.dtype)

        self.short_modes = symbol(False,(.65,2.5),SHORT_WIDTHS_A,SHORT_CUTOFF_A)
        self.long_modes = symbol(True,(4.5,6.5),LONG_WIDTHS_A,LONG_CUTOFF_A)
