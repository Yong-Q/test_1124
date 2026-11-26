"""A scalar functional of one weighted Xi tensor, with optional center coupling."""

import torch
from torch import nn
from torch.nn import functional as F

from .atom_functional import invariant_features
from .conditional import ConditionalRadial


class XiOnly(ConditionalRadial):
    def __init__(self,scale,coupled):
        super().__init__(scale)
        self.include_long=False
        del self.latent,self.log_width,self.amplitude
        self.coupled=coupled
        if coupled:
            self.context_projection=nn.Linear(116,12)
            self.pair_mix=nn.Parameter(torch.zeros(12))

    def pair_value(self,feature):
        z=self.context_projection(feature/self.feature_scale)
        q=(z.tanh()-self.context_projection.bias.tanh()).sum(0)
        return .5*(self.pair_mix*q.square()).sum()/len(feature)

    def value(self,xi):
        return self.invariant_energy(invariant_features(xi))

    def invariant_energy(self,feature):
        local=self.local_energy(feature)
        return local+self.pair_value(feature) if self.coupled else local

    def pair_curvature(self,base,directions):
        z=self.context_projection(base/self.feature_scale).tanh()
        dz=F.linear(directions/self.feature_scale,self.context_projection.weight)
        slope=1-z.square()
        q=(z-self.context_projection.bias.tanh()).sum(0)
        first=(slope[:,None]*dz).sum(0)
        coupled=(first*self.pair_mix)@first.mT
        coefficient=-2*z*slope*(self.pair_mix*q)
        u=dz.transpose(0,1).flatten(1)
        v=(dz*coefficient[:,None]).transpose(0,1).flatten(1)
        return (coupled+u@v.mT)/len(base)

    def jets(self,xi,delta,training):
        upper=torch.triu_indices(8,8,device=xi.device)
        features=invariant_features(xi)
        buffers=[delta[...,0]]
        for ell in (1,2,3):
            sl=slice(ell*ell,(ell+1)**2)
            v=delta[...,sl]
            term=v@xi[:,None,:,sl].mT
            buffers.append((term+term.mT)[...,upper[0],upper[1]])
        dx=torch.cat(buffers,-1)
        with torch.enable_grad():
            base=features.detach().requires_grad_()
            grad=torch.autograd.grad(self.invariant_energy(base),base,create_graph=training)[0]
        first=(dx*grad[:,None]).sum((0,2))
        second=self.local_curvature(torch.cat((features[:,None],dx),dim=1))
        if self.coupled:
            second=second+self.pair_curvature(features,dx)
        for j,ell in enumerate((1,2,3)):
            picked=grad[:,8+j*36:8+(j+1)*36]
            matrix=picked.new_zeros((len(xi),8,8))
            matrix[:,upper[0],upper[1]]=picked
            v=delta[...,ell*ell:(ell+1)**2]
            action=(matrix+matrix.mT)[:,None]@v
            second=second+v.transpose(0,1).flatten(1)@action.transpose(0,1).flatten(1).mT
        return -first,-second
