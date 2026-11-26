"""Periodic quadrature for a compact radial and angular Xi interaction."""
import numpy as np
from scipy.special import spherical_jn
import torch
from torch import nn

from .angular_model import AngularSpatial, angular_basis
from .atom_functional import reciprocal_modes

CUTOFF = 5.
KMAX = 4.5
SMOOTH = .30
CENTERS = (0., 2., 4.)
SIGMAS = (.70, .78, .78)


def short_modes(lengths, quota=96):
    """Keep zero and sample independent half-space modes in three bands."""
    pool = reciprocal_modes(lengths,KMAX)
    magnitude = np.linalg.norm(pool,axis=1)
    rng = np.random.default_rng(808)
    vectors = [np.zeros((1,3))]
    weights = [np.array([.5])]
    for start,stop in zip((0.,1.3,2.7),(1.3,2.7,KMAX)):
        ids = np.flatnonzero((magnitude>start)&(magnitude<=stop))
        count = min(quota,len(ids))
        if count:
            selected = np.sort(rng.choice(ids,count,replace=False))
            vectors.append(pool[selected])
            weights.append(np.full(count,len(ids)/count))
    return np.concatenate(vectors),np.concatenate(weights)


def axial_modes(lengths):
    k = 2*np.pi/lengths[-1]
    nums = np.arange(1+int(np.floor(KMAX/k)))
    vectors = np.zeros((len(nums),3))
    vectors[:,2] = nums*k
    weights = np.ones(len(nums))
    weights[0] = .5
    return vectors,weights


def setup(vectors,weights,device):
    wave = np.linalg.norm(vectors,axis=-1)
    nonzero = wave>0
    basis = torch.zeros((len(wave),8),device=device,dtype=torch.complex64)
    if nonzero.any():
        basis[torch.as_tensor(nonzero,device=device)] = angular_basis(vectors[nonzero],device)[:,:8].to(basis)
    r = np.linspace(0.,CUTOFF,257)
    u = np.clip((r-4.1)/(CUTOFF-4.1),0.,1.)
    cutoff = 1-u**3*(10-15*u+6*u*u)
    banks = np.exp(-.5*((r[:,None]-CENTERS)/SIGMAS)**2)*cutoff[:,None]
    banks /= np.trapz(r[:,None]**2*banks,r,axis=0)
    radial = []
    for ell in range(3):
        order = spherical_jn(ell,wave[:,None]*r[None,:])*r[None,:]**2
        radial.append(np.trapz(order[:,:,None]*banks[None]*(r[None,:,None]/CUTOFF)**ell,r,axis=1))
    kernels = torch.as_tensor(np.stack(radial,axis=1)*weights[:,None,None],device=device,dtype=torch.float32)
    return basis,kernels


def project(spectrum,basis):
    """Project m onto the wave direction while keeping all eight radial channels."""
    values = [spectrum[...,0]]
    for ell in (1,2):
        a,b = ell**2,(ell+1)**2
        values.append((spectrum[...,a:b]*basis[:,None,a-1:b-1]).sum(-1))
    return torch.stack(values,dim=-1)


class ShortSpatial(AngularSpatial):
    def __init__(self,scale,ell_max,width=32,context_width=24):
        super().__init__(scale,width,context_width)
        self.ell_max = ell_max
        self.shell_weight = nn.Parameter(torch.zeros(ell_max+1,8,3))

    def weights(self,kernels):
        return torch.einsum('klb,lnb->knl',kernels[:,:self.ell_max+1],self.shell_weight)

    def action(self,x,kernels):
        w = self.weights(kernels)
        if self.ell_max==0:
            return x*2*w
        scalar = 2*w[...,0]*x[...,0]+(w[...,1:]*x[...,1:]).sum(-1)
        return torch.cat((scalar.unsqueeze(-1),w[...,1:]*x[...,:1]),dim=-1)

    def forward(self,xi,spectrum,magnitude,angular_spectrum,basis,angular_magnitude,
                volume,mode_weight,short,kernels):
        return (super().forward(xi,spectrum,magnitude,angular_spectrum,basis,
                                angular_magnitude,volume,mode_weight)
                + self.short_energy(short,kernels,volume))

    def short_energy(self,short,kernels,volume):
        x = short[...,:self.ell_max+1]
        return .5*(x.conj()*self.action(x,kernels)).real.sum()/volume

    def short_jets(self,short,directions,kernels,volume):
        x = short[...,:self.ell_max+1]
        d = directions[...,:self.ell_max+1]
        left = d.flatten(1).conj()
        g = (left@self.action(x,kernels).flatten()).real
        h = (left@self.action(d,kernels).flatten(1).mT).real
        return -g/volume,-h/volume
