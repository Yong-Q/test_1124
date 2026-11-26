"""Xi-conditioned radial energy, with the analytic derivatives of one scalar."""

import torch
from torch import nn
from torch.nn import functional as F

from .readout import NormalizedReadout


class ConditionalRadial(NormalizedReadout):
    def __init__(self,scale):
        super().__init__(scale)
        self.radial_modes=nn.Linear(8,8,bias=False)
        self.context=nn.Linear(116,24)
        self.gate=nn.Linear(24,8)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def local_energy(self,feature):
        x=feature/self.feature_scale
        radial=self.radial_modes(x[...,:8])
        gate=self.gate(torch.tanh(self.context(x)))
        return super().local_energy(feature)+.5*(gate*radial.square()).sum()

    def local_curvature(self,field):
        x,dx=(field/self.feature_scale).split((1,field.shape[1]-1),dim=1)
        x=x[:,0]
        r=self.radial_modes(x[:,:8])
        dr=F.linear(dx[...,:8],self.radial_modes.weight)
        z=torch.tanh(self.context(x))
        dz=F.linear(dx,self.context.weight)
        gain=self.gate(z)
        dg=F.linear(dz*(1-z.square())[:,None],self.gate.weight)

        def contract(a,b):
            return a.transpose(0,1).flatten(1)@b.transpose(0,1).flatten(1).mT

        cross=contract(dr,dg*r[:,None])
        gate_curvature=(.5*r.square()@self.gate.weight)*(-2*z*(1-z.square()))
        return (super().local_curvature(field)+contract(dr,dr*gain[:,None])+
                cross+cross.mT+contract(dz,dz*gate_curvature[:,None]))
