"""Grid-free short-range atomic readout and periodic SOG free-energy tail."""


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def invariant_features(xi):
    """Keep all 8 radial modes and their cross-radial inner products."""
    upper = torch.triu_indices(8, 8, device=xi.device)
    blocks = [xi[..., 0]]
    for ell in (1, 2, 3):
        harmonics = xi[..., ell*ell:(ell+1)**2]
        gram = harmonics @ harmonics.transpose(-1, -2)
        blocks.append(gram[..., upper[0], upper[1]])
    return torch.cat(blocks, dim=-1)


def long_features(xi):
    return torch.cat((xi[...,0], *(xi[..., ell*ell:(ell+1)**2].square().mean((-2,-1))
                                    .sqrt().unsqueeze(-1) for ell in (1,2,3))), dim=-1)


def reciprocal_modes(lengths, kmax):
    kvec = 2*np.pi/np.asarray(lengths)
    bounds = np.floor(kmax/kvec).astype(int)
    integers = np.stack(np.meshgrid(*(np.arange(-n,n+1) for n in bounds),indexing='ij'),-1).reshape(-1,3)
    half = ((integers[:,0] > 0) |
            ((integers[:,0] == 0) & ((integers[:,1] > 0) |
             ((integers[:,1] == 0) & (integers[:,2] > 0)))))
    wave = integers*kvec
    return wave[half & (np.linalg.norm(wave,axis=-1) <= kmax)]


def structure_factors(positions, features, wave):
    """Direct Fourier sum in bounded batches. No gridding or interpolator."""
    result = torch.empty((len(positions), len(wave), features.shape[-1]),
                         dtype=torch.complex64 if positions.dtype==torch.float32 else torch.complex128,
                         device=positions.device)
    for start in range(0,len(wave),64):
        section = slice(start,start+64)
        angle = positions @ wave[section].T
        phase = torch.complex(angle.cos(),-angle.sin())
        result[:,section] = phase.transpose(-1,-2) @ features.to(phase.dtype)
    return result


class SOGFunctional(nn.Module):
    def __init__(self, include_long=True):
        super().__init__()
        self.local = nn.Linear(116,32)
        self.output = nn.Linear(32,1,bias=False)
        self.include_long = include_long
        if include_long:
            self.latent = nn.Linear(11,4,bias=False)
            self.log_width = nn.Parameter(torch.tensor([3.,5.,8.]).log().repeat(4,1))
            self.amplitude = nn.Parameter(torch.zeros(4,3))

    def local_energy(self, feature):
        return self.output(F.softplus(self.local(feature))-F.softplus(self.local.bias)).sum()

    def local_curvature(self, field):
        base, directions = field[:,0], field[:,1:]
        z = self.local(base)
        dz = F.linear(directions,self.local.weight)
        slope = torch.where(z>20.,1.,z.sigmoid())
        d = dz.transpose(0,1).flatten(1)
        action = dz*(self.output.weight[0]*slope*(1-slope))[:,None]
        return d @ action.transpose(0,1).flatten(1).T

    def long_energy(self, factors, k, volume):
        if not self.include_long:
            return factors.real.new_zeros(())
        modes = factors @ self.latent.weight.T.to(factors.dtype)
        return (modes.abs().square()*self.kernel(k)).sum()/volume

    def kernel(self,k):
        return (self.amplitude[None]*torch.exp(-.5*(k[:,None,None]*self.log_width.exp())**2)).sum(-1)

    def long_curvature(self, factors,k,volume):
        if not self.include_long:
            return factors.real.new_zeros((len(factors),len(factors)))
        modes = factors @ self.latent.weight.T.to(factors.dtype)
        return 2*(modes.flatten(1).conj() @ (modes*self.kernel(k)).flatten(1).T).real/volume
