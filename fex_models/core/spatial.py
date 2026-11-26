"""Spatial pair energy from Fourier components of the weighted observable Xi(x)."""

import torch
from torch import nn

from .functional import XiOnly


class SpatialXi(XiOnly):
    def __init__(self,scale):
        super().__init__(scale,coupled=False)
        self.spatial_embedding=nn.Linear(8,8,bias=False)
        self.log_sigma=nn.Parameter(torch.tensor([.7,1.5,3.,6.]).log())
        self.spatial_mix=nn.Parameter(torch.zeros(8,4))

    def transfer(self,wave):
        return torch.exp(-.5*(wave[:,None]*self.log_sigma.exp()[None])**2)@self.spatial_mix.mT

    def weighted_spectrum(self,spectrum):
        return spectrum@self.spatial_embedding.weight.mT.to(spectrum.dtype)

    def spatial_energy(self,spectrum,wave,volume):
        latent=self.weighted_spectrum(spectrum)
        return (latent.abs().square()*self.transfer(wave)).sum()/volume

    def spatial_jets(self,spectrum,directions,wave,volume):
        latent=self.weighted_spectrum(spectrum)
        tangent=self.weighted_spectrum(directions)
        K=self.transfer(wave)
        d=tangent.flatten(1)
        first=-2*(d.conj()@(latent*K).flatten()).real/volume
        second=-2*(d.conj()@(tangent*K).flatten(1).mT).real/volume
        return first,second
