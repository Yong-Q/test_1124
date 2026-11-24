"""Untrained atomic Omega[xi; v]. xi[spatial x, atom i, 1+n*16].

Channel 0 is the atomic number-density anchor; remaining channels retain n,l,m.
External v is independent and has the same layout, with beta V_i(x) in channel
0 and optional conjugate directional fields in the remaining channels.
"""
import torch
from torch import nn


class InvariantStandardization(nn.Module):
    def __init__(self, center, scale):
        super().__init__()
        self.register_buffer('center', center)
        self.register_buffer('scale', scale)

    def forward(self, values):
        return (values - self.center) / self.scale


class XiGrandPotential(nn.Module):
    def __init__(self,radials=8,width=32):
        super().__init__();self.radials=radials
        self.readout=nn.Sequential(nn.Linear(radials+3*radials*radials,width),nn.Softplus(),nn.Linear(width,width),nn.Softplus(),nn.Linear(width,1))

    def excess_density(self, xi):
        """Local beta F_ex contribution per unit volume for each xi entry."""
        rho = xi[..., 0]
        marks = (xi[..., 1:] / rho.clamp_min(1e-30)[..., None]).reshape(
            *rho.shape, self.radials, 16)
        invariants = [marks[..., 0]]
        for ell in range(1, 4):
            angular = marks[..., ell*ell:(ell+1)**2]
            gram = angular @ angular.transpose(-1, -2)
            invariants.append(gram.flatten(-2))
        return rho * self.readout(torch.cat(invariants, dim=-1)).squeeze(-1)

    def forward(self,xi,dv,external,chemical,species,thermal_volume=None):
        if xi.ndim!=3 or xi.shape[-1]!=1+16*self.radials:raise ValueError('xi must be [spatial x, atom i, 1+n*16]')
        if external.shape!=xi.shape or chemical.shape!=(xi.shape[1],):raise ValueError('Per-atom external/chemical dimensions mismatch')
        if species.shape!=(xi.shape[1],) or not bool(((species>=0)&(species<4)).all()):raise ValueError('Atomic element index required')
        if not all(bool(torch.isfinite(a).all()) for a in [xi,external,chemical]):raise ValueError('Nonfinite fields')
        rho=xi[...,0]
        if bool((rho<0).any()):raise ValueError('Negative number-density anchor')
        if bool((xi[...,1:][rho==0]!=0).any()):raise ValueError('Nonzero marks at zero atomic density')
        volume=torch.as_tensor(dv,dtype=xi.dtype,device=xi.device)
        if volume.ndim==0:volume=volume.expand(xi.shape[0])
        if volume.shape!=(xi.shape[0],) or not bool((volume>0).all()):raise ValueError('Positive quadrature volumes required')
        thermal=xi.new_ones(4) if thermal_volume is None else thermal_volume
        if thermal.shape!=(4,) or not bool((thermal>0).all()):raise ValueError('Positive Lambda_alpha^3 required')
        byspecies=torch.stack([rho[:,species==e].sum(1) for e in range(4)],1)
        logarg=(byspecies*thermal).clamp_min(torch.finfo(xi.dtype).tiny)
        ideal=(volume[:,None]*byspecies*(torch.log(logarg)-1)).sum()
        excess=(volume[:,None]*self.excess_density(xi)).sum()
        ext=(volume[:,None,None]*xi*external).sum()
        chem=(volume[:,None]*rho*chemical).sum()
        return {'ideal':ideal,'excess':excess,'external':ext,'chemical':chem,'omega':ideal+excess+ext-chem}
