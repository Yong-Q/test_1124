"""Direct intrinsic derivative tests of two scalar functionals of the current ξ."""
import argparse
from itertools import product
import json
from pathlib import Path
from time import monotonic

import h5py
import numpy as np
from scipy.sparse import csr_matrix
import torch

from pair_kernel_audit import run as audit
from pair_kernel_audit.report_density_metrics import intrinsic_targets
from portable_io import read_grid_inputs
from xi_response import deposition_stencil


SEED = 20261118


def anchor_energy(mass, pooled):
    """Scalar beta F_int on site masses ξ[..., 0]*dV, with axes x, y, z, atom."""
    total = mass
    if pooled:
        for axis in range(3):
            total = total + total.roll(1, axis) + total.roll(-1, axis)
    return (mass * total.clamp_min(torch.finfo(mass.dtype).tiny).log()).sum() / mass.shape[-1]


def neighborhood(keys, atoms, shape):
    voxel, atom = keys // atoms, keys % atoms
    grid = np.array(np.unravel_index(voxel, shape)).T
    rows, cols = [], []
    for delta in product((-1, 0, 1), repeat=3):
        shifted = ((grid + delta) % shape).T
        query = np.ravel_multi_index(shifted, shape) * atoms + atom
        index = np.searchsorted(keys, query).clip(max=len(keys)-1)
        match = keys[index] == query
        rows.append(np.flatnonzero(match))
        cols.append(index[match])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(keys), len(keys)))


def occupancy(voxels, mass, shape):
    frames, atoms, stencil = voxels.shape
    active = np.flatnonzero(mass)
    keys, index = np.unique(voxels.ravel()[active] * atoms + (active // stencil) % atoms,
                            return_inverse=True)
    counts = csr_matrix((mass.ravel()[active], (index, active // (stencil * atoms))),
                        shape=(len(keys), frames))
    return counts, neighborhood(keys, atoms, shape)


def log_jets(a, b):
    mean, dm = a[:, 0], a[:, 1:]
    pool = b[:, 0].clamp_min(torch.finfo(b.dtype).tiny)
    ratio = b[:, 1:] / pool[:, None]
    first = (dm * pool.log()[:, None] + mean[:, None] * ratio).sum(-1)
    cross = dm @ ratio.transpose(1, 2)
    second = cross + cross.transpose(1, 2) - (mean[:, None] * ratio) @ ratio.transpose(1, 2)
    return first, second


def anchor_jets(counts, neighbors, weights, tangent, atoms, block=32768, device='cpu'):
    """Derivatives of the same scalar along two supplied linear ξ directions."""
    states, frames, p = tangent.shape
    coeff = np.concatenate((weights[..., None], tangent), axis=2).transpose(1, 0, 2).reshape(frames, -1)

    def tensor(values):
        return torch.as_tensor(values, device=device)

    def sparse(matrix):
        return torch.sparse_csr_tensor(tensor(matrix.indptr), tensor(matrix.indices), tensor(matrix.data),
                                      size=matrix.shape)

    field = sparse(counts) @ tensor(coeff)
    pooled = sparse(neighbors) @ field
    result = {name: (field.new_zeros((states, p)), field.new_zeros((states, p, p)))
              for name in ('voxel', 'pooled')}
    for lo in range(0, len(field), block):
        a = field[lo:lo+block].reshape(-1, states, p+1).permute(1, 2, 0)
        n = pooled[lo:lo+block].reshape(-1, states, p+1).permute(1, 2, 0)
        for name, b in (('voxel', a), ('pooled', n)):
            first, second = result[name]
            grad, curve = log_jets(a, b)
            first += grad
            second += curve
    return {name: tuple(value.cpu().numpy() / atoms for value in values) for name, values in result.items()}


def case_entry(case, role, charges, args, index, *, selection=None, projector=anchor_jets):
    started = monotonic()
    seed = SEED + 10000 * ('train', 'validation', 'test').index(role) + index
    with np.load(case['labels']) as data:
        if str(data['deposition']) != 'smoothstep':
            raise ValueError('Expected a 0.5 Å smoothstep trajectory')
        rows, probes = audit.select_states(data, seed, args.per_family) if selection is None else selection
        shape = tuple(data['shape'])
        steps = data['source_steps']
        weights = data['weights'][rows]
        energy = data['probe_energy'].T[probes].transpose(0, 2, 1)
        tangent = -weights[..., None] * (energy - np.einsum('bt,btp->bp', weights, energy)[:, None])
        truth = intrinsic_targets(data, rows, probes)
    if 'xi_inputs' in case:
        bins, mass, _, _ = read_grid_inputs(case, shape, steps, charges)
    else:
        with h5py.File(case['cache'], 'r') as handle:
            np.testing.assert_array_equal(handle['step'][:], steps)
            bins, mass = deposition_stencil(handle['positions'][:], handle['cell'][0], shape, 'smoothstep')
    atoms = np.arange(bins.shape[1])
    if args.atoms and len(atoms) > args.atoms:
        atoms = np.sort(np.random.default_rng(seed + 305).choice(atoms, args.atoms, replace=False))
    predictions = {name: (np.zeros_like(truth[0]), np.zeros_like(truth[1])) for name in ('voxel', 'pooled')}
    for ids in np.array_split(atoms, (len(atoms) + 63) // 64):
        counts, kernel = occupancy(bins[:, ids], mass[:, ids], shape)
        for name, jets in projector(counts, kernel, weights, tangent, len(atoms), device=args.device).items():
            for total, value in zip(predictions[name], jets):
                total += value
    audit.emit(stage='case', role=role, case=case['case'], atoms=len(atoms), rows=rows.tolist(),
               probes=probes.tolist(), seconds=round(monotonic()-started, 2))
    return dict(truth=truth, estimates=predictions, role=role, species=case['species'])


def fit_scales(entries):
    target = [np.concatenate(col) for col in zip(*(audit.groups(*e['truth']) for e in entries))]
    rms = np.array([max(float(np.sqrt(np.mean(y*y))), 1e-15) for y in target])
    fit = {}
    for name in ('voxel', 'pooled'):
        features = [np.concatenate(col) for col in zip(*(audit.groups(*e['estimates'][name]) for e in entries))]
        xy = sum(np.mean(x*y) / s**2 for x, y, s in zip(features, target, rms))
        xx = sum(np.mean(x*x) / s**2 for x, s in zip(features, rms))
        fit[name] = max(0., float(xy / xx)) if xx else 0.
    return fit


def evaluate(entries, name, scale):
    targets = [np.concatenate(col) for col in zip(*(audit.groups(*e['truth']) for e in entries))]
    predict = [scale*np.concatenate(col) for col in zip(*(audit.groups(*e['estimates'][name]) for e in entries))]
    names = ('intrinsic_gradient', 'omega_curvature_diag', 'omega_curvature_offdiag')
    result = {}
    for tag, y, p in zip(names, targets, predict):
        rms = float(np.sqrt(np.mean(y*y)))
        result[tag] = dict(**audit.measure(y, p), reference_rms=rms)
        result[tag]['rmse_over_reference_rms'] = result[tag]['rmse'] / rms if rms else None
    gradient_mask = np.abs(targets[0]) > 1e-10
    result['projected_gradient_sign_agreement'] = float(np.mean(
        np.sign(targets[0][gradient_mask]) == np.sign(predict[0][gradient_mask]))) if gradient_mask.any() else None
    negative = sum(int(np.sum(np.linalg.eigvalsh(scale*e['estimates'][name][1])[:, 0]
                              < -1e-8 - 1e-6*np.linalg.norm(e['truth'][1], axis=(1, 2)))) for e in entries)
    result['negative_curvature_states'] = negative
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-family', type=int, default=4)
    p.add_argument('--atoms', type=int, default=128, help='Atoms sampled per trajectory, 0 for all')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--full', action='store_true', help='Include all trajectories in each split')
    args = p.parse_args()
    if args.per_family < 1 or args.atoms < 0:
        p.error('Positive sample size and nonnegative atom count required')
    args.output.mkdir()
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.25)
    charges = json.loads(audit.CHARGES.read_text())
    splits = json.loads(audit.MANIFEST.read_text())['splits']
    cases = splits if args.full else dict(audit.cohort(splits))
    scores, counts = {}, {}
    for role, cohort in cases.items():
        entries = [case_entry(case, role, charges, args, index) for index, case in enumerate(cohort)]
        if role == 'train':
            scales = fit_scales(entries)
        by_species = {'all': entries}
        for species in sorted({entry['species'] for entry in entries}):
            by_species[species] = [entry for entry in entries if entry['species'] == species]
        counts[role] = sum(len(r['truth'][0]) for r in entries)
        scores[role] = {group: {f'{name}_{fit}': evaluate(rows, name, scale)
                                for name in scales for fit, scale in (('fixed', 1.), ('fit', scales[name]))}
                        for group, rows in by_species.items()}
        audit.emit(stage='split_complete', role=role, scores=scores[role]['all'])
    temporary = args.output / 'summary.pending'
    with temporary.open('x') as handle:
        json.dump(dict(status='complete', states=counts, seed=SEED, atom_limit=args.atoms,
                       scales=scales, scores=scores), handle, indent=2, allow_nan=False)
    temporary.rename(args.output / 'summary.json')


if __name__ == '__main__':
    main()
