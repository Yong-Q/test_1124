"""Condition local energy on the same Xi invariants with a nonlinear residual."""

import torch
from torch import nn
from torch.nn import functional as F

from .atom_functional import SOGFunctional


def slope(z):
    return torch.where(z>20.,1.,z.sigmoid())


class NormalizedReadout(SOGFunctional):
    def __init__(self,scale,deep=False):
        super().__init__(True)
        self.register_buffer('feature_scale',scale)
        with torch.no_grad():
            self.local.weight.mul_(scale.to(self.local.weight))
        self.deep=deep
        if deep:
            self.residual=nn.Linear(32,32)
            self.residual_out=nn.Linear(32,1,bias=False)
            nn.init.zeros_(self.residual_out.weight)

    def local_energy(self,feature):
        activation=F.softplus(self.local(feature/self.feature_scale))-F.softplus(self.local.bias)
        result=self.output(activation).sum()
        if self.deep:
            residual=F.softplus(self.residual(activation))-F.softplus(self.residual.bias)
            result=result+self.residual_out(residual).sum()
        return result

    def local_curvature(self,field):
        field=field/self.feature_scale
        main=super().local_curvature(field)
        if not self.deep:
            return main
        x,dx=field[:,0],field[:,1:]
        z1=self.local(x)
        a1=F.softplus(z1)-F.softplus(self.local.bias)
        dj1=F.linear(dx,self.local.weight)
        s1=slope(z1)
        z2=self.residual(a1)
        dj2=F.linear(dj1*s1[:,None],self.residual.weight)
        s2=slope(z2)
        g2=s2*self.residual_out.weight[0]
        g1=(g2@self.residual.weight)*s1*(1-s1)

        def contraction(d,g):
            scalar=d.transpose(0,1).flatten(1)
            return scalar @ (d*g[:,None]).transpose(0,1).flatten(1).T

        return main+contraction(dj2,g2*(1-s2))+contraction(dj1,g1)
