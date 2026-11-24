"""Ideal derivatives and a bounded nonlinear correction to a positive pair mode."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn.functional import one_hot, softplus

from experiments.xi_continuous_pair_mlp.direct import DirectPair
from experiments.xi_conv3d_full_radial.network import SCALARS


IDEAL_FLOOR = 1.e-4
CORRECTION_BOUND = .2
NEURONS = ((.05, 0.), (.2, -2.))


class StableTotalPair(DirectPair):
    """Export the excess component to combine with the ordinary ideal term."""

    def __init__(self, shape=None, spacing=None, channels=(0,)):
        super().__init__(shape, spacing)
        self.channels = tuple(channels)
        self.environment_weights = nn.Parameter(torch.zeros(len(self.channels)*len(NEURONS)))

    def energy_components(self, xi, species):
        field = self.dense_first(xi, species)
        energy = self.energy_density(field)
        for channel, weights in zip(self.channels, self.environment_weights.reshape(-1,len(NEURONS))):
            q = field[:,channel if channel < SCALARS else self.short_channels+channel-SCALARS]
            for coefficient,(scale,bias) in zip(weights,NEURONS):
                origin = q.new_tensor(bias)
                x = q/scale
                activation = 4*scale**2*(softplus(x+origin)-softplus(origin)-origin.sigmoid()*x)
                energy = energy + coefficient*activation
        density = xi[..., 0] @ one_hot(species,num_classes=4).to(xi)
        log_density = density.clamp_min(torch.finfo(density.dtype).tiny).log()
        reference = (density*log_density-density).sum(-1)
        return energy, reference

    def total_density(self, xi, species):
        energy, reference = self.energy_components(xi, species)
        return energy+IDEAL_FLOOR*reference

    def excess_density(self, xi, species):
        energy, reference = self.energy_components(xi, species)
        return energy-(1-IDEAL_FLOOR)*reference


class IdealProjection:
    def __init__(self, counts, dv):
        self.dv = dv
        matrix = counts.double().to_sparse_csr()
        ptr = matrix.crow_indices().cpu().numpy()
        indices, values = matrix.col_indices(), matrix.values()
        self.blocks = []
        for lo in range(0,matrix.shape[0],16384):
            hi = min(lo+16384,matrix.shape[0])
            start, stop = ptr[lo],ptr[hi]
            sparse = torch.sparse_csr_tensor(torch.as_tensor(ptr[lo:hi+1]-start,device=values.device),
                                             indices[start:stop],values[start:stop],
                                             size=(hi-lo,matrix.shape[1]))
            self.blocks.append((sparse, sparse.to_sparse_coo().T.coalesce()))

    def project(self, coeff, selected):
        w, directions = coeff[:,:,0],coeff[:,:,1:]
        b,frames,p = selected.shape
        first, second = coeff.new_zeros((b,directions.shape[-1])),coeff.new_zeros((b,p,p))
        for matrix, transpose in self.blocks:
            mean = (matrix @ w.T.contiguous()).T
            active = mean > 0.
            safe = mean.clamp_min(torch.finfo(mean.dtype).tiny)
            grad = torch.where(active,torch.log(safe/self.dv),0.)
            by_frame = transpose @ grad.T.contiguous()
            first -= (directions.mT @ by_frame.T[...,None]).squeeze(-1)
            dx = (matrix @ selected.transpose(0,1).reshape(frames,b*p)).reshape(-1,b,p).transpose(0,1)
            ratio = torch.where(active[...,None], dx/safe[...,None], 0.)
            second -= dx.mT @ ratio
        return first,second


class EnvironmentBasis:
    """Smoothed density fields with bounded softplus curvatures."""

    def __init__(self, model, source):
        shape = tuple(int(x) for x in source.shape)
        cells = math.prod(shape)
        mass = source.mass
        voxel = torch.as_tensor(np.repeat(source.voxels,source.count_per_voxel),device=mass.device)
        element = source.elements[source.observations % source.atoms]
        self.fields = []
        # Keep large trajectories on the host and move only a spatial slab at a time.
        capacity = len(model.channels)*source.frames*cells*mass.element_size()
        free, _ = torch.cuda.mem_get_info(mass.device)
        storage = mass.device if capacity < .30*free else torch.device('cpu')
        self.storage = str(storage)
        for channel in model.channels:
            long = channel >= SCALARS
            atom = channel-SCALARS if long else channel-1
            selected = mass if channel==0 else mass*(element==atom)
            kernel = source.kernel_transfer(shape,long,mass)
            mixing = model.long_mix[atom] if long else model.spatial[channel]
            transfer = (mixing @ kernel.flatten(1)).reshape(kernel.shape[1:])
            field = torch.empty((source.frames,cells),device=storage,dtype=mass.dtype)
            for lo in range(0,source.frames,32):
                hi = min(lo+32,source.frames)
                keep = (source.frame_index >= lo) & (source.frame_index < hi)
                indices = (source.frame_index[keep]-lo)*cells + voxel[keep]
                dense = mass.new_zeros((hi-lo)*cells)
                dense.scatter_add_(0,indices,selected[keep])
                modes = torch.fft.rfftn(dense.reshape(hi-lo,*shape),dim=(-3,-2,-1))
                filtered = torch.fft.irfftn(modes*transfer,s=shape,dim=(-3,-2,-1)).flatten(1)
                if long:
                    filtered -= source.count_density[atom]*mixing.sum()
                field[lo:hi].copy_(filtered)
            self.fields.append(field)
        self.dv = source.dv

    def project(self, coeff, selected):
        w, directions = coeff[:,:,0],coeff[:,:,1:]
        terms = len(self.fields)*len(NEURONS)
        first = coeff.new_zeros((len(w),directions.shape[-1],terms))
        second = coeff.new_zeros((len(w),selected.shape[-1],selected.shape[-1],terms))
        for field_index,field in enumerate(self.fields):
            for start in range(0,field.shape[1],32768):
                q = field[:,start:start+32768].to(device=coeff.device, dtype=coeff.dtype)
                mean = w @ q
                d = selected.mT @ q
                for i,(scale,bias) in enumerate(NEURONS,start=field_index*len(NEURONS)):
                    s = torch.sigmoid(mean/scale+bias)
                    origin = 1/(1+math.exp(-bias))
                    g = 4*scale*(s-origin) @ q.T
                    first[...,i] -= (directions.mT @ g[...,None]).squeeze(-1)*self.dv
                    second[...,i] -= ((d*(4*s*(1-s))[:,None,:]) @ d.mT)*self.dv
        return first,second


def positive_transform(sizes, environment, channels=(0,)):
    """Map nonnegative template weights to quadratic and bounded nonlinear terms."""
    ns, na, nr = sizes
    total = sum(sizes)+len(channels)*len(NEURONS)
    columns = []
    for i in range(ns):
        unit = np.zeros(total)
        unit[i] = 1.
        columns.append(unit)
    for block in range(na//3):
        for u,v in ((1.,0.),(0.,1.),(2**-.5,2**-.5),(2**-.5,-2**-.5)):
            term = np.zeros(total)
            start = ns+3*block
            term[start:start+3] = (.5*u*u,u*v,.5*v*v)
            columns.append(term)
    if environment:
        for group,channel in enumerate(channels):
            for j in range(len(NEURONS)):
                for sign in (-1,1):
                    term = np.zeros(total)
                    term[channel] = 1.
                    term[sum(sizes)+group*len(NEURONS)+j] = sign*CORRECTION_BOUND
                    columns.append(term)
    constrained = len(columns)
    for i in range(ns+na,sum(sizes)):
        term = np.zeros(total)
        term[i] = 1.
        columns.append(term)
    return np.column_stack(columns),constrained
