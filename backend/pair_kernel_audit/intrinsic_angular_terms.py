"""Fit explicit rotationally invariant angular terms to intrinsic residuals."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from scipy.optimize import nnls
from scipy.sparse import csr_matrix
import torch
from torch import nn

from kernel import ELEMENTS
from pair_kernel_audit import intrinsic_anchor_trial as anchor
from pair_kernel_audit import run as audit
from pair_kernel_audit.intrinsic_angular_trial import FIT, PROBES, cases_from_log
from train_xi_intrinsic_full import factored_anchor_jets
from xi_response import deposition_stencil


TAU = .001
GROUPS = (slice(0, 3), slice(3, 8), slice(8, 15))


def angular_energy(mass, marks, atoms):
    return torch.stack([(.5*marks[..., group].square().sum(-1)/(mass+TAU)).sum()/atoms
                        for group in GROUPS])


class AngularIntrinsic(nn.Module):
    def __init__(self, pooled, coefficients):
        super().__init__()
        self.register_buffer('pooled', torch.as_tensor(pooled, dtype=torch.float64))
        self.register_buffer('angular_coefficients', torch.as_tensor(coefficients, dtype=torch.float64))

    def forward(self, xi, dv):
        """Four-dimensional x,y,z,atom grid with the 129 original ξ channels."""
        mass = dv*xi[..., 0]
        marks = dv*xi[..., 66:81]
        return self.pooled*anchor.anchor_energy(mass, pooled=True) + self.angular_coefficients @ angular_energy(
            mass, marks, mass.shape[-1]).to(self.angular_coefficients)


def angular_jets(fields, atoms):
    den = fields[:, 0, :, 0] + TAU
    dm = fields[:, 1:, :, 0]
    marks, directions = fields[:, 0, :, 1:], fields[:, 1:, :, 1:]
    first, second = [], []
    for group in GROUPS:
        a = marks[..., group]/den[..., None]
        da = directions[..., group]
        first.append(((da*a[:, None]).sum(-1) - .5*a.square().sum(-1)[:, None]*dm).sum(-1)/atoms)
        b = (da-dm[..., None]*a[:, None])/den.sqrt()[:, None, :, None]
        second.append(torch.einsum('bpic,bqic->bpq', b, b)/atoms)
    return torch.stack(first, -1), torch.stack(second, -1)


def singleton_angular_jets(scalar, norms, atoms):
    """Exact projections when all angular channels at a site come from one frame."""
    den, dm = scalar[:, 0]+TAU, scalar[:, 1:]
    first = (dm * (.5*(1.-(TAU/den).square()))[:, None]) @ norms
    second = (dm[:, :, None]*dm[:, None, :]*(TAU**2/den.pow(3))[:, None, None]) @ norms
    return first/atoms, second/atoms


def projection(case, charges, states, probes, atom_limit, seed):
    with np.load(case['labels']) as data:
        w = data['weights'][states]
        u = data['probe_energy'].T[probes].transpose(0, 2, 1)
        dw = -w[..., None]*(u-np.einsum('bt,btp->bp',w,u)[:, None])
        coeff = np.concatenate((w[..., None], dw), 2)
        shape, steps = tuple(data['shape']), data['source_steps']
    with h5py.File(case['cache'], 'r') as h:
        np.testing.assert_array_equal(h['step'][:], steps)
        atoms = np.arange(h['positions'].shape[1])
        if atom_limit and len(atoms)>atom_limit:
            atoms = np.sort(np.random.default_rng(seed+305).choice(atoms, atom_limit, replace=False))
        positions = h['positions'][:, atoms, :]
        bins, mass = deposition_stencil(positions, h['cell'][0], shape, 'smoothstep')
        z = np.array([charges[s] for s in ELEMENTS])
        marks = np.einsum('taem,e->tam', h['neighbor_element_nlm'][:, atoms, :, 4, 1:16], z)
    frames, atom_count, stencil = bins.shape
    observations = np.flatnonzero(mass.ravel())//stencil
    mask = mass.ravel() != 0.
    keys, row = np.unique(bins.ravel()[mask]*atom_count+observations%atom_count, return_inverse=True)
    matrix = csr_matrix((mass.ravel()[mask], (row, observations)), shape=(len(keys), frames*atom_count))
    columns = np.column_stack((np.ones(frames*atom_count), marks.reshape(-1, 15)))
    result = (np.zeros((len(states), 2, 3)), np.zeros((len(states), 2, 2, 3)))

    def device(x):
        return torch.as_tensor(x, device='cuda')

    coef = device(coeff)
    multiplicity = np.diff(matrix.indptr)
    singleton = np.flatnonzero(multiplicity == 1)
    multiple = np.flatnonzero(multiplicity > 1)
    for start in range(0, len(singleton), 32768):
        offset = matrix.indptr[singleton[start:start+32768]]
        obs = matrix.indices[offset]
        payload = device(columns[obs, 1:])
        norms = torch.stack([payload[:, group].square().sum(-1) for group in GROUPS], -1)
        frame, site = device(obs//atom_count), device(matrix.data[offset])
        for lo in range(0, len(states), 32):
            t = coef[lo:lo+32, frame].transpose(1, 2)*site
            for dst, jet in zip(result, singleton_angular_jets(t, norms, atom_count)):
                dst[lo:lo+len(t)] += jet.cpu().numpy()
    for start in range(0, len(multiple), 8192):
        tile = matrix[multiple[start:start+8192]]
        ids, inverse = np.unique(tile.indices, return_inverse=True)
        deposit = torch.sparse_csr_tensor(device(tile.indptr), device(inverse.astype(np.int32)), device(tile.data),
                                         size=(tile.shape[0], len(ids)))
        payload, frame = device(columns[ids]), device(ids//atom_count)
        for lo in range(0, len(states), 12):
            t = coef[lo:lo+12]
            values = t[:, frame].transpose(0, 1).reshape(len(ids), -1)
            field = deposit @ (values[..., None]*payload[:, None]).flatten(1)
            field = field.reshape(tile.shape[0], len(t), 3, 16).permute(1, 2, 0, 3)
            for dst, jet in zip(result, angular_jets(field, atom_count)):
                dst[lo:lo+len(t)] += jet.cpu().numpy()
    return result


def training_entries(cases, count):
    for index, case in enumerate(cases):
        rng = np.random.default_rng(713+index)
        with np.load(case['labels']) as data:
            family = data['candidate_family'][data['state_indices']]
        states = np.concatenate([rng.choice(np.flatnonzero(family==name), count, replace=False)
                                 for name in ('angular_only', 'spatial_and_angular')])
        pairs = {f'{l}/random': np.stack([32+rng.choice(np.arange(l*l,(l+1)**2), 2, replace=False)
                                         for _ in states]) for l in (1, 2, 3)}
        yield case, 'train', states, pairs


def evaluate_case(case, role, rows, pairs, charges, settings, index, pooled):
    states = np.tile(rows, len(pairs))
    probes = np.concatenate(list(pairs.values()))
    baseline = anchor.case_entry(case, role, charges, settings, index, selection=(states, probes),
                                 projector=factored_anchor_jets)
    seed = anchor.SEED+10000*('train', 'validation', 'test').index(role)+index
    jets = projection(case, charges, states, probes, settings.atoms, seed)
    for k, name in enumerate(pairs):
        sl = slice(k*len(rows),(k+1)*len(rows))
        entropy = tuple(x[sl] for x in baseline['estimates']['pooled'])
        yield name, dict(truth=tuple(x[sl] for x in baseline['truth']), entropy=entropy,
                         estimates={'base': tuple(pooled*x for x in entropy)},
                         features=tuple(x[sl] for x in jets), species=case['species'])


def refit_features(entry, with_angle):
    return tuple(np.concatenate((e[..., None], a), axis=-1) if with_angle else e[..., None]
                 for e, a in zip(entry['entropy'], entry['features']))


def fitting_system(groups, feature_fn, offset_fn):
    design, response = [], []
    for entries in groups.values():
        t, b, x = ([audit.groups(*a(e)) for e in entries] for a in (lambda e: e['truth'], offset_fn, feature_fn))
        for j in range(3):
            truth, baseline, matrix = (np.concatenate([e[j] for e in part]) for part in (t,b,x))
            scale = max(np.linalg.norm(truth),1e-15)
            design.append(matrix/scale)
            response.append((truth-baseline)/scale)
    return np.concatenate(design), np.concatenate(response)


def fit_coefficients(groups, feature_fn=lambda e: e['features'], offset_fn=lambda e: e['estimates']['base']):
    design, response = fitting_system(groups, feature_fn, offset_fn)
    norm = np.maximum(np.linalg.norm(design, axis=0),1e-15)
    return nnls(design/norm, response)[0]/norm


def evaluate_groups(groups, coefficients, refit):
    result = {}
    for name, rows in groups.items():
        for e in rows:
            e['estimates']['angular'] = tuple(b+x@coefficients for b,x in zip(e['estimates']['base'],e['features']))
            for variant, weights in refit.items():
                e['estimates'][variant] = tuple(matrix@weights for matrix in refit_features(e, variant=='joint_refit'))
        cohorts = {'all': rows}
        for species in sorted({r['species'] for r in rows}):
            cohorts[species] = [e for e in rows if e['species']==species]
        result[name] = {g: {v:anchor.evaluate(items,v,1.) for v in ('base','angular',*refit)}
                        for g,items in cohorts.items()}
    return result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--atoms', type=int, default=128)
    p.add_argument('--joint-refit', action='store_true', help='Also refit the pooled weight with and without angular features')
    args = p.parse_args()
    if args.atoms < 1:
        p.error('Use a positive atom sample count')
    args.output.mkdir()
    torch.cuda.set_per_process_memory_fraction(.25)
    splits = json.loads(audit.MANIFEST.read_text())['splits']
    charges = json.loads(audit.CHARGES.read_text())
    pooled = json.loads(FIT.read_text())['scales']['pooled']
    settings = SimpleNamespace(atoms=args.atoms, device='cuda')
    entries = {r:{} for r in ('train','validation','test')}
    counts = {r:0 for r in entries}

    def collect(source):
        for i, (case, role, rows, pairs) in enumerate(source):
            for name, entry in evaluate_case(case,role,rows,pairs,charges,settings,i,pooled):
                entries[role].setdefault(name,[]).append(entry)
            counts[role] += len(rows)
            audit.emit(stage='scored_case', role=role, case=case['case'], states=counts[role])

    collect(training_entries(dict(audit.cohort(splits))['train'],4))
    coefficients = fit_coefficients(entries['train'])
    refit = {}
    if args.joint_refit:
        def no_offset(entry):
            return tuple(np.zeros_like(x) for x in entry['truth'])

        refit['pooled_refit'] = fit_coefficients(entries['train'], lambda e: refit_features(e, False), no_offset)
        refit['joint_refit'] = fit_coefficients(entries['train'], lambda e: refit_features(e, True), no_offset)
    audit.emit(stage='coefficients', value=coefficients.tolist())
    audit.emit(stage='refit', coefficients={name: value.tolist() for name,value in refit.items()})
    collect(cases_from_log(PROBES,splits))
    scores = {role:evaluate_groups(groups,coefficients,refit) for role,groups in entries.items()}
    model = AngularIntrinsic(pooled,coefficients)
    torch.save(model.state_dict(),args.output/'model.pt')
    if refit:
        joint = refit['joint_refit']
        torch.save(AngularIntrinsic(joint[0], joint[1:]).state_dict(),args.output/'model_joint_refit.pt')
    tmp = args.output/'summary.pending'
    with tmp.open('x') as f:
        json.dump(dict(status='complete',states=counts,atoms=args.atoms,tau=TAU,pooled=pooled,
                       coefficients=coefficients.tolist(),scores=scores,
                       refit_coefficients={name: value.tolist() for name,value in refit.items()},
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30),f,indent=2,allow_nan=False)
    tmp.rename(args.output/'summary.json')


if __name__ == '__main__':
    main()
