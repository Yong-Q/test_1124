"""Three-dimensional variational benchmark on independently generated rigid molecules."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation
from scipy.special import softmax
from scipy.stats import qmc
import torch

from equilibrium import simplex_project
from kernel import Basis, grid
from pair_kernel_audit.intrinsic_angular_terms import TAU, GROUPS
from xi_response import deposition_stencil


ROOT = Path('evidence')
CHECKPOINTS = {'density': ROOT/'xi_intrinsic10k_clean_density_gpu_20261119_101122/model.pt',
               'angular': ROOT/'xi_intrinsic10k_clean_joint_gpu2_20261119_101121/model.pt'}
BOX = 8.
SPACING = .5
CASES = ('zero', 'spatial', 'directional')
ANGULAR_INDEX = 12  # packed anchor plus n=4, l=3, m=0


def geometry(species):
    angle = 2*np.pi*np.arange(3)/3
    if species == 'NH3':
        cosine = -np.sqrt((np.cos(np.deg2rad(106.67))+.5)/1.5)
        r = 1.012*np.sqrt(1-cosine**2)
        xyz = np.vstack((np.zeros((1,3)), np.column_stack((r*np.cos(angle), r*np.sin(angle),
                                                          np.full(3,1.012*cosine)))))
        charge = np.array([7.,1.,1.,1.])
    elif species == 'C2H6':
        carbon = .5*1.536
        theta = np.deg2rad(110.91)
        z, r = carbon-1.091*np.cos(theta), 1.091*np.sin(theta)
        xyz = np.vstack(([[0.,0.,carbon],[0.,0.,-carbon]],
                         np.column_stack((r*np.cos(angle),r*np.sin(angle),np.full(3,z))),
                         np.column_stack((r*np.cos(angle+np.pi/3),r*np.sin(angle+np.pi/3),np.full(3,-z)))))
        charge = np.r_[6.,6.,np.ones(6)]
    else:
        raise ValueError(f'Unknown rigid species: {species}')
    return xyz-xyz.mean(0), charge


def environment(rotations, vectors, charges):
    atoms = len(vectors)
    i,j = np.where(~np.eye(atoms,dtype=bool))
    displacements = vectors[j]-vectors[i]
    packed = np.ones((len(rotations), atoms, 16))
    basis = Basis()
    for lo in range(0,len(rotations),512):
        chunk = rotations[lo:lo+512]
        directions = np.einsum('fsd,pd->fps',chunk,displacements)
        marks = basis.values(directions.reshape(-1,3))[:,65:80].reshape(len(chunk),atoms,atoms-1,15)
        packed[lo:lo+len(chunk),:,1:] = np.einsum('fiac,ia->fic',marks,charges[j].reshape(atoms,-1))
    return packed


def rotations_from_unit_cube(cube):
    a,b,c = cube.T
    return Rotation.from_quat(np.column_stack((np.sqrt(1-a)*np.sin(2*np.pi*b),
                    np.sqrt(1-a)*np.cos(2*np.pi*b),np.sqrt(a)*np.sin(2*np.pi*c),
                    np.sqrt(a)*np.cos(2*np.pi*c)))).as_matrix()


class RigidCloud:
    def __init__(self, species, power, seed, box=BOX):
        vectors, charges = geometry(species)
        self.atoms = len(charges)
        uni = qmc.Sobol(6,scramble=True,seed=seed).random_base2(power)
        rotations = rotations_from_unit_cube(uni[:,3:])
        cell = np.eye(3)*box
        coords, self.shape, self.dv = grid(cell, SPACING)
        size = len(coords)
        pos = np.einsum('fsd,id->fis',rotations,vectors) + box*uni[:,None,:3]
        nodes, weights = deposition_stencil(pos,cell,self.shape,'smoothstep')
        t = len(pos)
        indices = np.broadcast_to(np.arange(t*self.atoms).reshape(t,self.atoms,1),nodes.shape)
        rows = nodes*self.atoms+np.arange(self.atoms)[None,:,None]
        self.lift = csr_matrix((weights.ravel(),(rows.ravel(),indices.ravel())),
                               shape=(size*self.atoms,t*self.atoms))
        self.lift.eliminate_zeros()
        self.back = self.lift.T.tocsr()
        self.marks = environment(rotations,vectors,charges)
        angle_norm = np.linalg.norm(environment(np.eye(3)[None],vectors,charges)[0,0,9:16])
        if angle_norm < 1e-12:
            raise ValueError('Expected a nonzero l=3 multiplet on the first atomic site')
        self.orientation = self.marks[:,0,ANGULAR_INDEX]/angle_norm
        x,y,z = (2*np.pi*coords/box).T
        potential = .8*(np.cos(x)+.5*np.sin(y)+.35*np.cos(z+x))/self.atoms
        spatial = (self.back @ np.repeat(potential,self.atoms)).reshape(t,self.atoms).sum(1)
        self.u = {'zero': np.zeros(t), 'spatial': spatial, 'directional': spatial+.8*self.orientation}

    def field(self, weights, with_angular=True):
        n = self.marks.shape[-1] if with_angular else 1
        marked = (weights[:,None,None]*self.marks[...,:n]).reshape(-1,n)
        return (self.lift @ marked).reshape(*self.shape,self.atoms,n)

    def pullback(self, gradient):
        n = gradient.shape[-1]
        pulled = (self.back @ gradient.reshape(-1,n)).reshape(-1,self.atoms,n)
        return np.sum(pulled*self.marks[...,:n], axis=(1,2))


def neighbors(values):
    for a in range(3):
        values = values+np.roll(values,1,a)+np.roll(values,-1,a)
    return values


def intrinsic_gradient(field, coefficient):
    mass = field[...,0]
    pooled = np.maximum(neighbors(mass),np.finfo(field.dtype).tiny)
    logp = np.log(pooled)
    density = coefficient[0]
    energy = density*np.sum(mass*logp)
    grad = np.zeros_like(field)
    grad[...,0] = density*(logp+neighbors(mass/pooled))
    if field.shape[-1] > 1:
        den = mass+TAU
        for j,s in enumerate(GROUPS):
            sl = slice(s.start+1,s.stop+1)
            a = field[...,sl]/den[...,None]
            square = np.square(a).sum(-1)
            weight = coefficient[j+1]
            energy += .5*weight*np.sum(square*den)
            grad[...,0] -= .5*weight*square
            grad[...,sl] = weight*a
    return float(energy/mass.shape[-1]), grad/mass.shape[-1]


def iterate(objective, initial, steps):
    if not np.all(initial>0):
        raise ValueError('Positive initial probabilities are required')
    energy0, _ = objective(initial)
    path = [energy0]
    last = None

    def transform(logits):
        nonlocal last
        w = softmax(logits)
        value,g = objective(w)
        last = (logits.copy(),value)
        return value, w*(g-w@g)

    def record(logits):
        path.append(last[1] if np.array_equal(last[0],logits) else transform(logits)[0])

    solver = minimize(transform, np.log(initial), method='L-BFGS-B', jac=True,
                      bounds=[(-40.,20.)]*len(initial), callback=record,
                      options={'maxiter':steps,'ftol':1e-13,'gtol':1e-10})
    w = softmax(solver.x)
    value,g = objective(w)
    gap = float(w@g-g.min())
    scale = 1./len(w)
    projected_steps = 0
    for _ in range(4*steps):
        if gap <= 1e-5:
            break
        direction = simplex_project(torch.from_numpy(w-scale*g)).numpy()-w
        slope = g@direction
        if slope >= 0:
            break
        fraction = 1.
        for _ in range(40):
            update = fraction*direction
            nxt = w+update
            new_value,new_g = objective(nxt)
            if new_value <= value+1e-4*fraction*slope:
                break
            fraction *= .5
        else:
            break
        curvature = update@(new_g-g)
        scale = float(np.clip((update@update)/curvature, 1e-8, 1.)) if curvature>0 else 1./len(w)
        w,value,g = nxt,new_value,new_g
        path.append(value)
        projected_steps += 1
        gap = float(w@g-g.min())
    return w, dict(status='stationary' if gap<=1e-5 else 'residual_above_tolerance',
                   optimizer_stop=str(solver.message),steps=int(solver.nit)+projected_steps,
                   projected_steps=projected_steps,
                   stationarity_gap=gap,omega=value,omega_initial=energy0,omega_trace=path,
                   monotone=bool(np.all(np.diff(path)<=1e-10)))


def solve(cloud, config, coefficient, starts, maxiter, functional=intrinsic_gradient):
    u = cloud.u[config]
    with_angular = bool(np.any(coefficient[1:]))
    def objective(weights):
        energy, grad = functional(cloud.field(weights,with_angular),coefficient)
        return energy+u@weights, cloud.pullback(grad)+u
    outputs, meta = {}, {}
    for start in starts:
        initial = softmax(-.25*u) if start=='external' else np.full(len(u),1/len(u))
        weights, info = iterate(objective,initial,maxiter)
        field = cloud.field(weights)
        info['orientation'] = float(weights@cloud.orientation)
        info['particle_conservation_error'] = float(np.max(np.abs(field[...,0].sum(axis=(0,1,2))-1)))
        info['minimum_density'] = float(field[...,0].min()/cloud.dv)
        outputs[start], meta[start] = field, info
    return outputs, meta


def errors(p, r, dv):
    x,y = p[...,0]/dv, r[...,0]/dv
    nx,ny = p[...,9:16]/dv,r[...,9:16]/dv
    return dict(density_rmse=float(np.sqrt(np.mean((x-y)**2))),
                density_relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(y)),
                angular_l3_rmse=float(np.sqrt(np.mean((nx-ny)**2))))


def independent_reference(species, power):
    result = {}
    for seed in (6021,6022):
        cloud = RigidCloud(species,power,seed)
        result[seed] = {}
        for case in CASES:
            p = softmax(-cloud.u[case])
            result[seed][case] = cloud.field(p),float(p@cloud.orientation)
    return result


def uniform_orientation_case(species, coefficients, power, reference_power, maxiter):
    vectors, charge = geometry(species)
    voxels = round(BOX/SPACING)**3
    normalization = np.linalg.norm(environment(np.eye(3)[None],vectors,charge)[0,0,9:16])

    def sample(seed, n):
        unit = qmc.Sobol(3,scramble=True,seed=seed).random_base2(n)
        marks = environment(rotations_from_unit_cube(unit),vectors,charge)[...,1:]
        return marks, marks[:,0,ANGULAR_INDEX-1]/normalization

    marks, observable = sample(6019,power)
    external = .8*observable
    result = {}
    for name, coeff in coefficients.items():
        channel_weights = np.zeros(15)
        for weight, group in zip(coeff[1:],GROUPS):
            channel_weights[group] = weight
        design = (marks*np.sqrt(channel_weights/(len(vectors)*(1+TAU*voxels)))).reshape(len(marks),-1)
        ideal_constant = coeff[0]*np.log(27/voxels)

        def objective(w):
            angle = w@design
            return ideal_constant+.5*(angle@angle)+w@external, design@angle+external

        result[name] = {}
        for start, init in (('uniform', np.full(len(marks),1/len(marks))), ('external',softmax(-.25*external))):
            w, meta = iterate(objective,init,maxiter)
            meta['orientation'] = float(w@observable)
            result[name][start] = meta
    references = []
    for seed in (6021,6022):
        _, angles = sample(seed,reference_power)
        references.append(float(softmax(-.8*angles)@angles))
    return dict(models=result,reference_orientation=float(np.mean(references)),reference_replicates=references,
                finite_support_reference=float(softmax(-external)@observable))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--support-power', type=int, default=12)
    ap.add_argument('--reference-power', type=int, default=15)
    ap.add_argument('--maxiter', type=int, default=180)
    ap.add_argument('--uniform-orientation-only', action='store_true')
    args = ap.parse_args()
    if not 2 <= args.support_power <= args.reference_power <= 19 or args.maxiter < 1:
        ap.error('Require positive iterations and ordered quadrature powers within [2, 19]')
    args.output.mkdir()
    coefficients = {}
    for name,path in CHECKPOINTS.items():
        d = torch.load(path,map_location='cpu',weights_only=True)
        coefficients[name] = np.r_[float(d['pooled']),d['angular_coefficients'].numpy()]
    if args.uniform_orientation_only:
        tests = {species:uniform_orientation_case(species,coefficients,args.support_power,args.reference_power,args.maxiter)
                 for species in ('NH3','C2H6')}
        with (args.output/'summary.json').open('x') as f:
            json.dump(dict(status='complete',scope='pure orientation at fixed uniform three-dimensional atomic density',
                           support_count=2**args.support_power,reference_count_per_seed=2**args.reference_power,
                           results=tests), f, indent=2, allow_nan=False)
        return
    out = {}
    for species in ('NH3','C2H6'):
        cloud = RigidCloud(species,args.support_power,6019)
        predict, data = {}, {}
        for case in CASES:
            data[case],predict[case] = {},{}
            for name,coefficient in coefficients.items():
                starts = ('uniform',) if case=='zero' else ('uniform','external')
                predict[case][name],data[case][name] = solve(cloud,case,coefficient,starts,args.maxiter)
                print(species,case,name,[(k,v['status'],v['steps']) for k,v in data[case][name].items()],flush=True)
        np.savez_compressed(args.output/f'{species}_predictions.npz',
                            **{f'{c}_{m}_{s}':a for c,models in predict.items() for m,starts in models.items() for s,a in starts.items()})
        ref = independent_reference(species,args.reference_power)
        ref_fields = {case:.5*(ref[6021][case][0]+ref[6022][case][0]) for case in CASES}
        out[species] = {}
        for case in CASES:
            exact_p = softmax(-cloud.u[case])
            exact_field = cloud.field(exact_p)
            ref_orientation = .5*(ref[6021][case][1]+ref[6022][case][1])
            for name, starts in predict[case].items():
                for start, field in starts.items():
                    details = data[case][name][start]
                    details['vs_independent'] = errors(field,ref_fields[case],cloud.dv)
                    details['vs_exact_same_support'] = errors(field,exact_field,cloud.dv)
                    details['orientation_error_vs_independent'] = abs(details['orientation']-ref_orientation)
                    if case!='zero':
                        delta = field[...,0]-predict['zero'][name]['uniform'][...,0]
                        reference = ref_fields[case][...,0]-ref_fields['zero'][...,0]
                        details['density_response_relative_l2'] = float(np.linalg.norm(delta-reference)/np.linalg.norm(reference))
            out[species][case] = dict(models=data[case],reference_orientation=ref_orientation,
                                      independent_reference_replicate_error=errors(ref[6021][case][0],ref[6022][case][0],cloud.dv),
                                      finite_support_error=errors(exact_field,ref_fields[case],cloud.dv))
        np.savez_compressed(args.output/f'{species}_references.npz',**ref_fields)
    with (args.output/'summary.json').open('x') as f:
        json.dump(dict(status='complete',scope='one rigid molecule, constrained discrete support, independent reference integration',
                       box_A=BOX,spacing_A=SPACING,coefficients={k:v.tolist() for k,v in coefficients.items()},
                       support_count=2**args.support_power,reference_count_per_seed=2**args.reference_power,
                       support_seed=6019,reference_seeds=[6021,6022],results=out),f,indent=2,allow_nan=False)


if __name__ == '__main__':
    main()
