"""PRL-style pair matching plus real-field stationarity for the atomic density block."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import softmax, xlogy
import torch
from torch import nn

from pair_kernel_audit.md_field_equilibrium import field_input, potential, stencil
from pair_kernel_audit.run import MANIFEST


TRAIN = ('CH4', 'H2O', 'C2H4')
TEST = ('NH3', 'C2H6')
RADII = (1.5, 2.5, 4., 6.)
RHO0 = .04
FLOOR = 1e-12


def periodic_kernels(spacing):
    """Real-space, normalized radial kernels on the common 0.5 Å voxel grid."""
    kernels = []
    for radius in RADII:
        extent = np.ceil(radius / spacing).astype(int)
        indices = np.stack(np.meshgrid(*(np.arange(-n, n+1) for n in extent), indexing='ij'), axis=-1)
        kernel = np.maximum(1 - np.sum((indices*spacing/radius)**2, axis=-1), 0.)**2
        live = kernel > 0.
        kernels.append((indices[live], kernel[live]/kernel.sum()))
    return kernels


class PrlDensityIntrinsic(nn.Module):
    """Scalar free energy of the density component of ξ on a 3D physical grid."""
    def __init__(self, coefficients):
        super().__init__()
        self.register_buffer('coefficients', torch.as_tensor(coefficients, dtype=torch.float64))

    def forward(self, xi, spacing):
        if xi.ndim != 5 or np.shape(spacing) != (3,):
            raise ValueError('Expected ξ[x,y,z,atom,channel] and three spatial spacings')
        rho = xi[...,0].sum(-1)
        spectrum = torch.fft.rfftn(rho)
        local = []
        spacing = np.asarray(spacing)
        for coordinates, weights in periodic_kernels(spacing):
            kernel = torch.zeros_like(rho)
            indices = tuple(torch.as_tensor(coordinates[:,i]%rho.shape[i],device=rho.device) for i in range(3))
            kernel.index_put_(indices, torch.as_tensor(weights,dtype=rho.dtype,device=rho.device), accumulate=True)
            local.append(torch.fft.irfftn(torch.fft.rfftn(kernel)*spectrum,s=rho.shape).clamp_min(0))
        averages = torch.stack(local)
        density = torch.cat((rho[None],averages),dim=0)+FLOOR
        entropy = (density*density.log()-density).sum((1,2,3))
        squares = averages.square().sum((1,2,3))/(2*RHO0)
        gradients = (rho-averages).square().sum((1,2,3))/(2*RHO0)
        cubic = averages.pow(3).sum((1,2,3))/(6*RHO0**2)
        return float(np.prod(spacing))*(self.coefficients.to(xi)@torch.cat((entropy,squares,gradients,cubic)))


def bulk_modes(case, max_wavevector=1.7, max_mode=10):
    """Fourier values of the smoothstep deposition at physical periodic modes."""
    with h5py.File(case['cache'], 'r') as h:
        xyz = h['positions'][:]
        length = np.diag(h['cell'][0])
        elements = h['elements'][:]
    shape = np.ceil(length/.5).astype(int)
    spacing = length/shape
    z = (xyz % length)/spacing-.5
    lower = np.floor(z)
    fraction = z-lower
    upper = fraction**2*(3-2*fraction)
    axes = np.array(((1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)))
    kernels = periodic_kernels(spacing)
    for mode in range(1, max_mode+1):
        for integer in axes*mode:
            k = 2*np.pi*integer/length
            if np.linalg.norm(k)>max_wavevector:
                continue
            phasor = np.ones(z.shape[:2],dtype=np.complex128)
            for q in np.flatnonzero(integer):
                phase = k[q]*spacing[q]
                phasor *= np.exp(1j*phase*(lower[:,:,q]+.5))*(1+upper[:,:,q]*np.expm1(1j*phase))
            transfer = np.array([np.cos((coords*spacing)@k)@weights for coords,weights in kernels])
            yield phasor, elements, float(np.prod(length)), transfer


def structure_factor(case):
    """Connected atomic structure factor of the same smoothly deposited density."""
    design, values = [], []
    for phasor, elements, volume, transfer in bulk_modes(case):
        atoms = len(elements)
        rho = atoms/volume
        amp = phasor.sum(1)
        connected = abs(amp-amp.mean())**2/atoms
        s = connected.mean()
        blocks = np.array([x.mean() for x in np.array_split(connected,10)])
        sem = blocks.std(ddof=1)/np.sqrt(len(blocks))
        if s <= .04 or sem > .35*s:
            continue
        design.append(np.r_[1., transfer**2, rho/RHO0*transfer**2, rho/RHO0*(1-transfer)**2,
                            np.full(len(RADII),rho**2/RHO0**2)*transfer**2])
        values.append(1/s)
    if not design:
        raise ValueError(f'No resolved bulk expectation for {case["case"]}')
    return np.stack(design), np.asarray(values)


class AxialProjection:
    """Lateral marginal of the full 3D convolution, exact for a planar field."""
    def __init__(self, shape, spacing):
        n = shape[2]
        self.identity = np.eye(n)
        self.kernels = []
        for coordinates, weights in periodic_kernels(spacing):
            lag = np.bincount(coordinates[:,2]%n, weights=weights, minlength=n)
            self.kernels.append(lag[(np.arange(n)[:,None]-np.arange(n)[None,:])%n])

    def local_only(self):
        local = AxialProjection.__new__(AxialProjection)
        local.identity = self.identity
        local.kernels = [self.identity] * len(self.kernels)
        return local

    def jets(self, rho, value=False, hessian=True):
        first, second, energies = [], [], []
        terms = [(self.identity,'entropy')]
        terms += [(k,'entropy') for k in self.kernels]
        terms += [(k,'square') for k in self.kernels]
        terms += [(self.identity-k,'square') for k in self.kernels]
        terms += [(k,'cubic') for k in self.kernels]
        for matrix, kind in terms:
            x = matrix@rho
            if kind == 'entropy':
                x = x+FLOOR
                e, g, h = xlogy(x,x)-x, np.log(x), 1/x
            elif kind == 'square':
                e, g, h = x*x/(2*RHO0), x/RHO0, np.full_like(x,1/RHO0)
            else:
                e, g, h = x**3/(6*RHO0**2), x*x/(2*RHO0**2), x/RHO0**2
            first.append(matrix.T@g)
            if value:
                energies.append(e.sum())
                if hessian:
                    second.append(matrix.T@(h[:,None]*matrix))
        return np.stack(first,axis=1), energies, second


def external_samples(folder, species, separate_hydrogen=False):
    raw = folder/'reference'/f'{species}_r05_field'
    source = json.loads((raw/'metadata.json').read_text())
    with h5py.File(raw/'source.h5','r') as h:
        xyz = h['positions'][1:,:,2]
        length = np.diag(h['cell'][0])
        elements = h['elements'][:]
    shape = np.ceil(length/.5).astype(int)
    nz = shape[2]
    spacing = length/shape
    volume = np.prod(length)
    grid = (np.arange(nz)+.5)*spacing[2]
    force, _ = field_input(Path(source['original_force_input']),length[2])
    field = potential(grid,length[2],*force)
    bins, mass = stencil(xyz,length[2],nz)
    keys = bins+np.arange(len(xyz))[:,None,None]*nz
    groups = (elements==1,elements!=1) if separate_hydrogen else (np.ones(len(elements),bool),)
    profiles = np.stack([np.bincount(keys[:,keep].ravel(),weights=mass[:,keep].ravel(),
                         minlength=len(xyz)*nz).reshape(-1,nz)/(volume/nz) for keep in groups],axis=-1)
    return shape,spacing,profiles,field,grid,volume/nz


def external_profiles(folder, species):
    shape,spacing,profiles,field,grid,_ = external_samples(folder,species)
    ref = dict(species=species,rho=profiles.mean(0)[:,0],field=field,frames=len(profiles))
    return AxialProjection(shape,spacing), ref, grid


def gradients(operator, profile):
    rho, field = profile['rho'], profile['field']
    basis, _, _ = operator.jets(rho)
    # Canonical stationarity has an unknown additive chemical potential.
    weight = rho/rho.sum()
    return np.sqrt(weight)[:,None]*(basis-weight@basis), np.sqrt(weight)*(-field+weight@field)


def fit(bulk, field, active=None):
    groups = [g for g in (bulk,field) if g]
    pairs = [tuple(map(np.concatenate,zip(*g))) for g in groups]
    matrix = np.concatenate([x/np.linalg.norm(y) for x,y in pairs])
    target = np.concatenate([y/np.linalg.norm(y) for _,y in pairs])
    shape = matrix.shape[1]
    if active is None:
        active = np.arange(shape)
    matrix = matrix[:,active]
    norm = np.maximum(np.linalg.norm(matrix,axis=0),1e-12)
    matrix = matrix/norm
    ridge = .001
    rng = np.random.default_rng(20261119)
    start = rng.uniform(.06,.12,len(norm))

    def objective(params):
        error = matrix@params-target
        return .5*float(error@error+ridge*(params@params)),matrix.T@error+ridge*params

    result = minimize(objective,start*norm,jac=True,method='L-BFGS-B',
                      bounds=[(.05*norm[0],None)]+[(0.,None)]*(len(norm)-1),
                      options={'maxiter':1500,'gtol':1e-9,'ftol':1e-13})
    if not result.success:
        raise RuntimeError(str(result.message))
    coefficients = np.zeros(shape)
    coefficients[active] = result.x/norm
    return coefficients, start


def compare(operator, profile, coefficients, *, total_from_metadata=None):
    rho, v = profile['rho'], profile['field']
    total = rho.sum() if total_from_metadata is None else float(total_from_metadata)
    initial = softmax(-v)

    def objective(w):
        g,e,_ = operator.jets(w*total,True,hessian=False)
        return coefficients@e/total + v@w, g@coefficients+v

    def gradient(w):
        return operator.jets(w*total)[0]@coefficients+v

    result = minimize(objective,initial,jac=True,method='SLSQP',bounds=[(1e-14,None)]*len(v),
                      constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],
                      options={'maxiter':500,'ftol':1e-12})
    w = np.maximum(result.x,0.)
    w /= w.sum()
    initial_gap = float(w @ gradient(w)-gradient(w).min())
    for polish in range(2000):
        g = gradient(w)
        enter = int(g.argmin())
        if w@g-g[enter] <= 1e-7:
            break
        leave = int(np.argmax(np.where(w>0,g,-np.inf)))
        direction = np.zeros_like(w)
        direction[[enter,leave]] = [1.,-1.]

        def slope(amount):
            local = gradient(w+amount*direction)
            return local[enter]-local[leave]

        amount = w[leave]
        if slope(amount)>0.:
            amount = brentq(slope,0.,amount,xtol=1e-16)
        w += amount*direction
    for newton in range(16):
        g = gradient(w)
        if w@g-g.min()<=1e-7:
            break
        active = np.flatnonzero((w>0.)|(g<w@g-1e-7))
        _, _, pieces = operator.jets(w*total,True)
        h = total*np.einsum('k,kij->ij',coefficients,pieces)[np.ix_(active,active)]
        scale = np.sqrt(h.diagonal())
        basis = 1/scale
        answers = np.linalg.solve(h*basis[:,None]*basis,np.column_stack((g[active]*basis,basis)))
        multiplier = (basis@answers[:,0])/(basis@answers[:,1])
        d = (multiplier*answers[:,1]-answers[:,0])*basis
        direction = np.zeros_like(w)
        direction[active] = d
        decreasing = d<0.
        width = min(1.,.99*np.min(-w[active[decreasing]]/d[decreasing])) if decreasing.any() else 1.
        base = objective(w)[0]
        accepted = False
        for _ in range(20):
            trial = w+width*direction
            if width>1e-12 and np.min(trial)>=0. and objective(trial)[0]<=base+1e-14:
                w = trial
                accepted = True
                break
            width *= .5
        if not accepted:
            break
    predicted = w*total
    ref_basis, ref_y = gradients(operator,profile)
    _, _, terms = operator.jets(predicted,True)
    h = np.einsum('k,kij->ij',coefficients,terms)
    final_gradient = gradient(w)
    gap = float(w@final_gradient-final_gradient.min())
    norm = np.linalg.norm(rho)
    return predicted, initial*total, dict(solver_success=gap<=1e-7, sqp_steps=int(result.nit), polish_steps=polish,
                    newton_polish_steps=newton,
                    stationarity_before_polish=initial_gap, stationarity_gap=gap,
                    min_total_curvature=float(np.linalg.eigvalsh(h)[0]),
                    reference_gradient_rmse=float(np.linalg.norm(ref_basis@coefficients-ref_y)),
                    ideal_reference_gradient_rmse=float(np.linalg.norm(ref_basis[:,0]-ref_y)),
                    initial_relative_l2=float(np.linalg.norm(initial*total-rho)/norm),
                    predicted_relative_l2=float(np.linalg.norm(predicted-rho)/norm))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--postprocess', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--ablation', action='store_true')
    args = p.parse_args()
    args.output.mkdir()
    cohorts = json.loads(MANIFEST.read_text())['splits']

    def anchors(role,species):
        cases = sorted((c for c in cohorts[role] if c['species']==species),key=lambda c:c['case'])
        return [cases[i] for i in np.unique(np.linspace(0,len(cases)-1,3,dtype=int))]

    bulk_cases = [c for species in TRAIN for c in anchors('train',species)]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        bulk = list(executor.map(structure_factor,bulk_cases))
    refs = {species:external_profiles(args.postprocess,species) for species in TRAIN}
    fields = [gradients(op,r) for op,r,_ in refs.values()]
    coefficients, initial = fit(bulk,fields)
    print('Training finished',len(coefficients),flush=True)
    torch.save(PrlDensityIntrinsic(coefficients).state_dict(),args.output/'model.pt')
    ablations = {}
    if args.ablation:
        local_bulk = []
        for case, (x,y) in zip(bulk_cases,bulk):
            with h5py.File(case['cache'],'r') as h:
                rho = len(h['elements']) / np.linalg.det(h['cell'][0])
            row = np.zeros(x.shape[1])
            row[[0,5,13]] = [1,rho/RHO0,(rho/RHO0)**2]
            local_bulk.append((np.broadcast_to(row,x.shape),y))
        local_fields = [gradients(op.local_only(),r) for op,r,_ in refs.values()]
        ablations = dict(bulk_only=fit(bulk,[])[0], field_only=fit([],fields)[0],
                         local_only_joint=fit(local_bulk,local_fields,active=[0,5,13])[0])
    scores, bulk_scores, ablation_scores = {}, {}, {}
    for species in TEST:
        with ProcessPoolExecutor(max_workers=min(3,args.workers)) as executor:
            vb, yb = map(np.concatenate,zip(*executor.map(structure_factor,anchors('test',species))))
        bulk_scores[species] = dict(modes=len(yb), rmse=float(np.sqrt(np.mean((vb@coefficients-yb)**2))),
                                    ideal_rmse=float(np.sqrt(np.mean((1-yb)**2))))
        op, state, z = external_profiles(args.postprocess,species)
        predicted, guess, scores[species] = compare(op,state,coefficients)
        np.savez(args.output/f'{species}_density.npz',z_A=z,md=state['rho'],external_guess=guess,predicted=predicted)
        print(species,scores[species],flush=True)
        ablation_scores[species] = {}
        for label, parameters in ablations.items():
            substitute = op.local_only() if label=='local_only_joint' else op
            values, _, metrics = compare(substitute,state,parameters)
            ablation_scores[species][label] = metrics
            print(species,label,metrics,flush=True)
            np.savez(args.output/f'{species}_{label}.npz',z_A=z,md=state['rho'],predicted=values)
    with (args.output/'summary.json').open('x') as f:
        json.dump(dict(status='complete', density_channel='sum of atomic anchor densities',
                       bulk_target='inverse connected atomic structure factor at positive 3D modes',
                       training_species=list(TRAIN), test_species=list(TEST),
                       train_bulk=[c['case'] for c in bulk_cases],
                       reweighted=False, train_field_frames={s:value[1]['frames'] for s,value in refs.items()},
                       coefficients=coefficients.tolist(), initialization=initial.tolist(),
                       bulk_holdout=bulk_scores, external_holdout=scores, ablation=ablation_scores), f,
                  indent=2,allow_nan=False)


if __name__ == '__main__':
    main()
