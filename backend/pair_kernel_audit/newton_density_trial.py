"""Safeguarded Newton comparison on a full three-dimensional translation grid."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import softmax
import torch

from equilibrium import simplex_project
from kernel import grid
from pair_kernel_audit.rigid_equilibrium_trial import BOX, SPACING, errors, geometry, neighbors
from xi_response import deposition_stencil


METHODS = ('pooled_original','voxel','convex_smooth')
FLOOR = 1e-15
TOLERANCE = 1e-7
COEFFICIENTS = Path('evidence/xi_intrinsic10k_clean_density_gpu_20261119_101122/summary.json')


def density_jet(mass, name):
    """Scalar energy, gradient and Hessian action on the density mass field."""
    n = mass.shape[-1]
    voxel = mass+FLOOR
    if name=='pooled_original':
        p = neighbors(voxel)/27
        value = np.sum(voxel*np.log(p))/n
        gradient = (np.log(p)+neighbors(voxel/p)/27)/n

        def hessian(d):
            pd = neighbors(d)/27
            return (pd/p+neighbors(d/p-voxel*pd/p**2)/27)/n

    elif name in ('voxel','convex_smooth'):
        local = 1. if name=='voxel' else .25
        smooth = 1.-local
        p = neighbors(voxel)/27
        value = np.sum(local*voxel*np.log(voxel)+smooth*p*np.log(p))/n
        gradient = (local*(1.+np.log(voxel))+smooth*neighbors(1.+np.log(p))/27)/n

        def hessian(d):
            return (local*d/voxel+smooth*neighbors((neighbors(d)/27)/p)/27)/n

    else:
        raise ValueError(f'Unknown density functional {name}')
    return float(value),gradient,hessian


def center(x):
    return x-np.mean(x)


def conjugate_direction(rhs, action, maxiter=35):
    x = np.zeros_like(rhs)
    r = rhs.copy()
    p = r.copy()
    square = float(r@r)
    goal = max(1e-22,square*1e-10)
    for _ in range(maxiter):
        if square<=goal:
            break
        h = action(p)
        curvature = float(p@h)
        if curvature<=1e-14*float(p@p):
            return None
        step = square/curvature
        x += step*p
        r -= step*h
        nxt = float(r@r)
        p = r + nxt/square*p
        square = nxt
    return center(x)


def newton(objective, initial, maxiter):
    w = initial.copy()
    v,g,h = objective(w)
    energies = [v]
    counts = dict(newton_steps=0,damped_steps=0,projected_steps=0,negative_curvature_events=0)

    def line_search(direction, bound):
        slope = float(g@direction)
        if slope>=0 or bound<=0:
            return None
        step = min(1., bound)
        for _ in range(25):
            trial = w+step*direction
            a,b,c = objective(trial)
            if a<=v+1e-4*step*slope:
                return trial,a,b,c
            step *= .5
        return None

    for _ in range(maxiter):
        if float(w@g-g.min()) <= TOLERANCE:
            break
        accepted = None
        if w.min()>1e-9/len(w):
            for damping in (0., .05, .25, 1., 4.):
                d = conjugate_direction(-center(g),lambda z:center(h(z))+damping*len(w)*z)
                if d is None:
                    counts['negative_curvature_events']+=1
                    continue
                decreasing = d<0
                bound = .995*np.min(-w[decreasing]/d[decreasing]) if decreasing.any() else 1.
                if bound<1e-4:
                    break
                accepted = line_search(d,bound)
                if accepted is not None:
                    counts['damped_steps' if damping>0 else 'newton_steps']+=1
                    break
        if accepted is None:
            d = simplex_project(torch.from_numpy(w-center(g)/len(w))).numpy()-w
            accepted = line_search(d,1.)
            if accepted is None:
                break
            counts['projected_steps']+=1
        w,v,g,h = accepted
        energies.append(v)
    gap = float(w@g-g.min())
    return w,dict(**counts, status='stationary' if gap<=TOLERANCE else 'residual_above_tolerance',
                  stationarity_gap=gap, omega_initial=energies[0], omega=v, trace=energies)


class Translations:
    """A rigid molecule translated over every cell center of an 8 Å periodic box."""
    def __init__(self, species):
        coords, self.shape, self.dv = grid(np.eye(3)*BOX,SPACING)
        relative,_ = geometry(species)
        atoms = len(relative)
        nodes, weight = deposition_stencil(coords[:,None]+relative,np.eye(3)*BOX,self.shape,'smoothstep')
        cols = np.broadcast_to(np.arange(len(coords))[:,None,None],nodes.shape).ravel()
        rows = (nodes*atoms+np.arange(atoms)[None,:,None]).ravel()
        self.lift = csr_matrix((weight.ravel(),(rows,cols)),shape=(len(coords)*atoms,len(coords)))
        self.lift.eliminate_zeros()
        self.back = self.lift.T.tocsr()
        self.grid = (*self.shape,atoms)
        x,y,z = (2*np.pi*coords/BOX).T
        v = .8*(np.cos(x)+.5*np.sin(y)+.35*np.cos(z+x))/atoms
        self.external = np.asarray(self.back@np.repeat(v,atoms))
        self.perturb = np.broadcast_to(((-1.)**np.arange(self.shape[0]))[:,None,None],self.shape).ravel()

    def field(self,w):
        return (self.lift@w).reshape(self.grid)

    def objective(self,w,external,name,coefficient):
        value,gradient,hessian = density_jet(self.field(w),name)
        return (coefficient*value+external@w, coefficient*(self.back@gradient.ravel())+external,
                lambda d:coefficient*(self.back@hessian(self.field(d)).ravel()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--maxiter',type=int,default=80)
    p.add_argument('--voxel-fit',type=Path,help='Use a fitted voxel-entropy coefficient for the voxel branch')
    args = p.parse_args()
    if args.maxiter<1:
        p.error('--maxiter must be positive')
    args.output.mkdir()
    coefficient = json.loads(COEFFICIENTS.read_text())['coefficients'][0]
    scales = dict.fromkeys(METHODS,coefficient)
    if args.voxel_fit:
        fit = json.loads(args.voxel_fit.read_text())
        if fit['configuration']['mode'] != 'voxel' or fit['status'] != 'complete':
            raise ValueError('Expected a completed voxel-entropy fit')
        scales['voxel'] = fit['coefficients'][0]
    results = {}
    for species in ('NH3','C2H6'):
        cloud = Translations(species)
        results[species] = {}
        n = len(cloud.external)
        for case, v, initial in (('zero_checkerboard',np.zeros(n),(1.+.025*cloud.perturb)/n),
                                 ('spatial_from_uniform',cloud.external,np.full(n,1/n))):
            reference = cloud.field(softmax(-v))
            results[species][case] = {}
            for method in METHODS:
                objective = lambda w:cloud.objective(w,v,method,scales[method])
                uniform = np.full(n,1/n)
                d = cloud.perturb/n
                curvature = float(d@objective(uniform)[2](d))
                w,info = newton(objective,initial,args.maxiter)
                x,y = cloud.field(w),reference
                info.update(density_rmse=float(np.sqrt(np.mean((x-y)**2))/cloud.dv),
                            density_relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(y)),
                            initial_mode=float(initial@cloud.perturb), final_mode=float(w@cloud.perturb),
                            checkerboard_curvature=curvature, particle_error=float(np.max(
                                np.abs(x.sum(axis=(0,1,2))-1))), mass_min=float(w.min()))
                results[species][case][method]=info
                print(species,case,method,info['status'],info['density_relative_l2'],flush=True)
    temporary = args.output/'summary.pending'
    with temporary.open('x') as f:
        json.dump(dict(status='complete',shape=[int(x) for x in cloud.shape],coefficient=coefficient,
                       coefficients_by_method=scales,tolerance=TOLERANCE,
                       scope='all translations, fixed orientation, exact discrete Boltzmann reference',
                       results=results),f,indent=2,allow_nan=False)
    temporary.rename(args.output/'summary.json')


if __name__ == '__main__':
    main()
