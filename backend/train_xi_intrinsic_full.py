"""Train the ξ-only intrinsic anchor functional on the complete 0.5 Å split."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from pair_kernel_audit import intrinsic_anchor_trial as anchor
from pair_kernel_audit import run as audit
from train_xi_balanced_trial import stratified_probes
from train_xi_full import save_state


BRANCHES = ('voxel', 'pooled')
METRICS = ('intrinsic_gradient', 'omega_curvature_diag', 'omega_curvature_offdiag')
EVAL_EPOCHS = {1, 5, 10, 20, 35, 50, 75, 100}


class IntrinsicAnchorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficients = nn.Parameter(torch.zeros(2, dtype=torch.float64))

    def forward(self, mass):
        return self.coefficients @ torch.stack((anchor.anchor_energy(mass, pooled=False),
                                                anchor.anchor_energy(mass, pooled=True)))


def factored_anchor_jets(counts, neighbors, weights, tangent, atoms, block=32768, device='cuda',
                         include_pooled=True):
    """Sum one-frame sites analytically, then project only sites with overlapping frames."""
    states, frames, p = tangent.shape
    safe_weights = np.maximum(weights, np.finfo(weights.dtype).tiny)
    result = {}

    def sparse(matrix):
        return torch.sparse_csr_tensor(*(torch.as_tensor(x, device=device)
                                         for x in (matrix.indptr, matrix.indices, matrix.data)),
                                       size=matrix.shape)

    branches = [('voxel', counts)]
    if include_pooled:
        branches.append(('pooled', (neighbors @ counts).tocsr()))
    for name, pool in branches:
        multiplicity = np.diff(pool.indptr)
        single = np.flatnonzero(multiplicity == 1)
        values = np.asarray(counts[single].sum(axis=1)).ravel()
        at = pool.indptr[single]
        frames_at_site = pool.indices[at]
        mass = np.bincount(frames_at_site, weights=values, minlength=frames)
        log_mass = np.bincount(frames_at_site, weights=values*np.log(pool.data[at]), minlength=frames)
        grad = np.einsum('btp,bt->bp', tangent, mass*(np.log(safe_weights)+1)+log_mass)
        curve = np.einsum('btp,btq,t->bpq', tangent/safe_weights[..., None], tangent, mass)
        mixed = np.flatnonzero(multiplicity > 1)
        coeff = np.concatenate((weights[..., None], tangent), 2)
        for lo in range(0, len(mixed), block):
            sites = mixed[lo:lo+block]
            own = sparse(counts[sites])
            nearby = own if name == 'voxel' else sparse(pool[sites])
            states_per_block = max(1, min(32, 384 // (p + 1)))
            for start in range(0, states, states_per_block):
                stop = min(start+states_per_block, states)
                inputs = torch.as_tensor(coeff[start:stop].transpose(1, 0, 2).reshape(frames, -1), device=device)
                a = (own @ inputs).reshape(-1, stop-start, p+1).permute(1, 2, 0)
                b = a if name == 'voxel' else (nearby @ inputs).reshape(-1, stop-start, p+1).permute(1, 2, 0)
                g, h = anchor.log_jets(a, b)
                grad[start:stop] += g.cpu().numpy()
                curve[start:stop] += h.cpu().numpy()
        result[name] = grad/atoms, curve/atoms
    return result


def prepare(role, cases, cache, charges):
    entries = []
    for index, case in enumerate(cases):
        path = cache / f"{case['case']}.npz"
        rows = np.arange(case['states'])
        seed = anchor.SEED + 10000 * ('train', 'validation', 'test').index(role) + index
        probes = stratified_probes(np.random.default_rng(seed), len(rows), phase=index)
        if not path.exists():
            entry = anchor.case_entry(case, role, charges, SimpleNamespace(atoms=0, device='cuda'), index,
                                      selection=(rows, probes), projector=factored_anchor_jets)
            pending = path.with_suffix('.pending')
            with pending.open('wb') as handle:
                np.savez(handle, probes=probes, targets_first=entry['truth'][0], targets_second=entry['truth'][1],
                         **{f'{name}_{k}': values[k] for name, values in entry['estimates'].items() for k in (0, 1)})
            os.replace(pending, path)
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved['probes'], probes)
            truth = saved['targets_first'], saved['targets_second']
            estimates = {n: (saved[f'{n}_0'], saved[f'{n}_1']) for n in BRANCHES}
        arrays = (*truth, *(value for branch in estimates.values() for value in branch))
        if any(len(x) != len(rows) or not np.isfinite(x).all() for x in arrays):
            raise ValueError(f"Nonfinite or incomplete projections for {case['case']}")
        entries.append(dict(truth=truth, estimates=estimates, role=role, species=case['species']))
        audit.emit(stage='prepared', role=role, trajectories=index+1, states=sum(len(x['truth'][0]) for x in entries))
    return entries


def train_tensors(entries):
    y = tuple(torch.as_tensor(np.concatenate([x['truth'][k] for x in entries]), device='cuda') for k in (0, 1))
    features = tuple(torch.as_tensor(np.stack([np.concatenate([x['estimates'][n][k] for x in entries])
                                               for n in BRANCHES], -1), device='cuda') for k in (0, 1))
    panels = (y[0], y[1].diagonal(dim1=1, dim2=2), y[1][:, 0, 1])
    scales = torch.stack([v.square().mean().sqrt().clamp_min(1e-15) for v in panels])
    return y, features, scales


def score(entries, coefficients):
    evaluated = []
    for x in entries:
        blend = tuple(sum(coefficients[i]*x['estimates'][name][k] for i, name in enumerate(BRANCHES))
                      for k in (0, 1))
        evaluated.append(dict(x, estimates={**x['estimates'], 'joint': blend}))
    cohorts = {'all': evaluated}
    for species in sorted({x['species'] for x in evaluated}):
        cohorts[species] = [x for x in evaluated if x['species'] == species]
    return {group: {name: anchor.evaluate(rows, branch, 1.) for name, branch in
                    (('model', 'joint'), ('voxel_fixed', 'voxel'), ('pooled_fixed', 'pooled'))}
            for group, rows in cohorts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--memory-fraction', type=float, default=.38)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch) < 1 or not 0 < args.memory_fraction <= .5:
        parser.error('Positive epochs/batch and memory fraction up to 0.5 required')
    args.output.mkdir(exist_ok=args.resume)
    cache = args.output / 'projections'
    cache.mkdir(exist_ok=args.resume)
    checkpoint = args.output / 'checkpoint.pt'
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    manifest = audit.MANIFEST.read_bytes()
    splits = json.loads(manifest)['splits']
    charges = json.loads(audit.CHARGES.read_text())
    counts = {role: sum(x['states'] for x in rows) for role, rows in splits.items()}
    identity = dict(manifest_sha256=hashlib.sha256(manifest).hexdigest(), method='intrinsic_anchor_joint_v1',
                    epochs=args.epochs, batch=args.batch, seed=anchor.SEED, charges=charges)
    model = IntrinsicAnchorModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    progress = dict(epoch=0, steps=0, history=[], best=None)
    if args.resume:
        stored = torch.load(checkpoint, map_location='cuda', weights_only=False)
        if stored['identity'] != identity:
            raise ValueError('Resume configuration differs')
        model.load_state_dict(stored['model'])
        optimizer.load_state_dict(stored['optimizer'])
        progress = stored['progress']
    else:
        save_state(checkpoint, model, optimizer, progress, identity)
    audit.emit(stage='starting', device=torch.cuda.get_device_name(), states=counts, epochs=args.epochs,
               batch=args.batch, initialization='zero coefficients', atom_limit=0)
    train = prepare('train', splits['train'], cache, charges)
    val = prepare('validation', splits['validation'], cache, charges)
    y, features, scales = train_tensors(train)
    norm = scales.cpu().numpy()
    for epoch in range(progress['epoch'] + 1, args.epochs + 1):
        order = np.random.default_rng(anchor.SEED + epoch).permutation(len(y[0]))
        for lo in range(0, len(order), args.batch):
            batch = torch.as_tensor(order[lo:lo+args.batch], device='cuda')
            first, second = (x[batch] @ model.coefficients - yy[batch] for x, yy in zip(features, y))
            errors = (first, second.diagonal(dim1=1, dim2=2), second[:, 0, 1])
            loss = sum((e/s).square().mean() for e, s in zip(errors, scales))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.coefficients.clamp_min_(0.)
            progress['steps'] += 1
        progress['epoch'] = epoch
        coeff = model.coefficients.detach().cpu().tolist()
        if not np.isfinite(coeff).all():
            raise FloatingPointError('Nonfinite intrinsic coefficient')
        if epoch in EVAL_EPOCHS or epoch == args.epochs:
            validation = score(val, coeff)
            valid = validation['all']['model']
            loss = sum((valid[name]['rmse']/scale)**2 for name, scale in zip(METRICS, norm))
            progress['history'].append(dict(epoch=epoch, coefficients=coeff, validation=validation,
                                            validation_loss=float(loss)))
            if progress['best'] is None or loss < progress['best']['loss']:
                progress['best'] = dict(epoch=epoch, coefficients=coeff, loss=float(loss))
            audit.emit(stage='validation', epoch=epoch, scores=valid)
        save_state(checkpoint, model, optimizer, progress, identity)
        audit.emit(stage='epoch_complete', epoch=epoch, steps=progress['steps'], coefficients=coeff)
    best = progress['best']
    test = prepare('test', splits['test'], cache, charges)
    matched = score(test, best['coefficients'])
    best_weights = {'coefficients': torch.tensor(best['coefficients'], dtype=torch.float64)}
    torch.save(best_weights, args.output / 'trained.pt')
    temporary = args.output / 'summary.pending'
    with temporary.open('w') as handle:
        json.dump(dict(status='complete', states=counts, training_rms=norm.tolist(),
                       selected=best, test=matched, train=score(train, best['coefficients']),
                       history=progress['history'], steps=progress['steps'], configuration=identity),
                  handle, indent=2, allow_nan=False)
    os.replace(temporary, args.output / 'summary.json')
    audit.emit(stage='complete', selected=best, test=matched['all'])


if __name__ == '__main__':
    main()
