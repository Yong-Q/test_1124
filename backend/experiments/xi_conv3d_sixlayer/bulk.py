"""Homogeneous element Hessian for the six serial convolution stages."""

import torch
from torch.nn import functional as F

from experiments.xi_conv3d.bulk import BulkCase
from experiments.xi_conv3d_full_radial.network import INVARIANTS, SCALARS
from experiments.xi_conv3d_multiscale.bulk import MultiscaleBulk
from experiments.xi_conv3d_sixlayer.network import softplus_slope


def predicted_hessian(case, model, short_modes, long_modes=None):
    first_weights = model.spatial[:SCALARS]
    mean_short = (case.rho @ case.inputs) * first_weights.sum(1)
    mean = torch.cat((mean_short,mean_short.new_zeros(model.features-SCALARS)))
    short = case.inputs[None] * (short_modes @ first_weights.T)[:,None,:]
    if long_modes is None:
        variation = short
        columns = list(range(SCALARS))
    else:
        long = case.inputs[None,:,1:5] * (long_modes @ model.long_mix.T)[:,None,:]
        variation = torch.cat((short,long),dim=-1)
        columns = [*range(SCALARS),*range(INVARIANTS,model.features)]
    variation = variation * case.rho.sqrt()[None,:,None]
    mean_h = mean
    d_h = variation
    dzs, slopes = [], []
    for i, layer in enumerate(model.layers):
        z = F.linear(mean_h,layer.weight)
        if i:
            weights = model.context[i-1]
            k = weights[:,0] + case.shells[:,1,None]*weights[:,1]
            z = z*weights.sum(1)
            dz = F.linear(d_h,layer.weight)*k[:,None,:]
        else:
            dz = F.linear(d_h,layer.weight[:,columns])
        z = z+layer.bias
        slope = softplus_slope(z)
        mean_h = F.softplus(z)-F.softplus(layer.bias)
        d_h = dz*slope
        dzs.append(dz)
        slopes.append(slope)
    g = model.output.weight[0]
    n = len(case.rho)
    value = torch.eye(n, dtype=mean.dtype, device=mean.device)
    for i in range(len(model.layers)-1,-1,-1):
        slope = slopes[i]
        dz = dzs[i]
        adj = g*slope
        value = value + (dz*(adj*(1.-slope)))@dz.transpose(-1,-2)
        if i:
            adj = adj*model.context[i-1].sum(1)
        g = F.linear(adj,model.layers[i].weight.T)
    return value


def match(case, value):
    observed = case.basis.mH @ value.to(case.basis.dtype) @ case.basis
    error = (observed-case.targets)*case.pair_mask
    return error.abs().square().sum()/case.targets.square().sum()


class SixLayerBulk(BulkCase):
    def bulk_loss(self, model):
        return match(self,predicted_hessian(self,model,self.shells))


class SixLayerMultiscaleBulk(MultiscaleBulk):
    def bulk_loss(self, model):
        return match(self,predicted_hessian(self,model,self.short_modes,self.long_modes))


class SixLayerSingleScaleBulk(MultiscaleBulk):
    def __init__(self, case, data, device, labels):
        super().__init__(case,data,device,labels,include_long=False)

    def bulk_loss(self, model):
        return match(self,predicted_hessian(self,model,self.short_modes))
