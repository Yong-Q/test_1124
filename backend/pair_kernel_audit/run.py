"""Matched 0.5 Å trial of local and nonlocal residual corrections."""
import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from scipy.sparse import coo_matrix
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel import ELEMENTS
from pair_kernel_audit.kernels import RADII, local_jets, pair_jets, real_kernel
from model import XiGrandPotential
from portable_io import ResponseWeights, read_grid_inputs
from train_xi_full import SparseTrajectory, process_batch
from xi_response import deposition_stencil

MANIFEST = Path('xi_smooth05_accel_157349/training_split.json')
CHARGES = Path('charges_nuclear.json')
SEED = 735
BLOCK = 1024
RIDGETERM = .03


def emit(**entry):
    print(json.dumps(entry, allow_nan=False), flush=True)


def selected(rows, prefix):
    return next(row for row in rows if row['case'].startswith(prefix))


def cohort(splits):
    train = [selected(splits['train'], name) for name in ('H2O-T986.4-r00', 'CH4-T290.5-r00',
             'C2H4-T430.4-r00', 'CO2-T463.6-r00', 'H2O-T986.4-r03')]
    test = [selected(splits['test'], name) for name in ('NH3-T618.2-r00', 'C2H6-T465.4-r00')]
    return [('train', train), ('validation', splits['validation']), ('test', test)]


def select_states(data, seed, per_family):
    rng = np.random.default_rng(seed)
    families = data['candidate_family'][data['state_indices']]
    rows = np.concatenate([rng.choice(np.flatnonzero(families == family), per_family, replace=False)
                           for family in ('original_spatial', 'angular_only', 'spatial_and_angular')])
    rng.shuffle(rows)
    probes = np.empty((len(rows), 2), dtype=int)
    for i in range(len(probes)):
        if i % 3 == 0:
            probes[i] = rng.choice(32, 2, replace=False)
        elif i % 3 == 1:
            probes[i] = 32+rng.choice(16, 2, replace=False)
        else:
            probes[i] = rng.integers([0, 32], [32, 48])
    return rows, probes


def new_tensor(x):
    return torch.as_tensor(x, device='cuda', dtype=torch.float32)


def aggregate_inputs(case, shape, source_steps, charges):
    """Linear projection of xi onto its anchor and all 16 n=4 angular components."""
    if 'xi_inputs' in case:
        stencil, mass, full_marks, cell = read_grid_inputs(case, shape, source_steps, charges)
        atoms = stencil.shape[1]
        marks = np.concatenate((full_marks[..., :1], full_marks[..., 65:81]), axis=-1).reshape(-1, 17)
        del full_marks
    else:
        with h5py.File(case['cache'], 'r') as h:
            np.testing.assert_array_equal(h['step'][:], source_steps)
            positions = h['positions'][:]
            cell = h['cell'][0]
            frames, atoms, _ = positions.shape
            stencil, mass = deposition_stencil(positions, cell, shape, 'smoothstep')
            marks = np.ones((frames*atoms, 17), dtype=np.float32)
            z = np.array([charges[name] for name in ELEMENTS], dtype=np.float32)
            for lo in range(0, frames, 8):
                hi = min(lo+8, frames)
                raw = h['neighbor_element_nlm'][lo:hi, :, :, 4, :]
                marks[lo*atoms:hi*atoms, 1:] = np.einsum('taem,e->tam', raw, z).reshape(-1, 16)
    volume = float(np.linalg.det(cell)/np.prod(shape))
    columns = np.repeat(np.arange(frames*atoms, dtype=np.int32), stencil.shape[-1])
    matrix = coo_matrix((mass.ravel().astype(np.float32)/volume, (stencil.ravel(), columns)),
                        shape=(np.prod(shape), len(marks))).tocsr()
    matrix.eliminate_zeros()
    return matrix, marks, atoms, tuple(np.diag(cell)/shape)


def build_aggregate(case, shape, source_steps, charges):
    matrix, marks, atoms, spacing = aggregate_inputs(case, shape, source_steps, charges)
    matrix_cuda = torch.sparse_csr_tensor(torch.as_tensor(matrix.indptr, device='cuda'),
                                          torch.as_tensor(matrix.indices, device='cuda'),
                                          new_tensor(matrix.data), size=matrix.shape)
    return matrix_cuda, new_tensor(marks), atoms, spacing


def extract_features(case, shape, steps, charges, coeff, dv, block):
    matrix, marks, atoms, spacing = build_aggregate(case, shape, steps, charges)
    spectra = [torch.fft.rfftn(real_kernel(shape, spacing, radius, torch.float32, 'cuda')).real
               for radius in RADII]
    features = {name: ([], []) for name in ('local', 'pair')}
    for lo in range(0, len(coeff), block):
        batch = coeff[lo:lo+block]
        frame_weight = batch.transpose(0, 1).reshape(batch.shape[1], -1)
        products = frame_weight.repeat_interleave(atoms, dim=0)[:, :, None]*marks[:, None]
        fields = (matrix @ products.flatten(1)).T.reshape(len(batch), batch.shape[-1], 17, *shape)
        for name, derivatives in [('local', local_jets(fields, dv)), ('pair', pair_jets(fields, spectra, dv))]:
            for output, items in zip(features[name], derivatives):
                output.append(items.cpu().numpy().astype(np.float64))
    return {key: tuple(np.concatenate(values) for values in result) for key, result in features.items()}


def load(case, role, charges, count, seed, batch, block, feature_extractor=extract_features, *, selection=None):
    started = time.monotonic()
    with np.load(case['labels']) as data:
        if str(data['deposition']) != 'smoothstep':
            raise ValueError('Expected 0.5 Å smooth labels')
        shape = tuple(data['shape'])
        steps = data['source_steps']
        rows, probes = select_states(data, seed, count) if selection is None else selection
        response = ResponseWeights(data)
        coeff = new_tensor(np.concatenate((response.weights[rows, :, None],
                                           response.select(rows, probes)), axis=2))
        c1 = data['c1_projected'][rows[:, None], probes]
        c2 = data['c2_projected'][rows[:, None, None], probes[:, :, None], probes[:, None, :]]
        features = feature_extractor(case, shape, steps, charges, coeff, float(data['voxel_volume']), block)
        source = SparseTrajectory(case, data, charges, torch.device('cuda'))
        candidates = data['state_indices'][rows].tolist()
    targets = new_tensor(c1), new_tensor(c2)
    batches = [(coeff[i:i+batch], tuple(y[i:i+batch] for y in targets))
               for i in range(0, len(coeff), batch)]
    emit(stage='features', role=role, case=case['case'], seconds=round(time.monotonic()-started, 2),
         rows=rows.tolist(), candidates=candidates, probes=probes.tolist())
    return dict(case=case['case'], role=role, features=features, c1=c1, c2=c2, source=source,
                batches=batches, rows=rows, probes=probes)


@torch.no_grad()
def base_predictions(model, entry, scales):
    first, second = [], []
    for coeff, targets in entry['batches']:
        _, (c1, c2) = process_batch(model, entry['source'], coeff, targets, scales, BLOCK, None,
                                    return_predictions=True)
        first.append(c1.cpu().numpy().astype(np.float64))
        second.append(c2.cpu().numpy().astype(np.float64))
    entry['base'] = np.concatenate(first), np.concatenate(second)


def groups(first, second):
    suffix = first.shape[2:]
    return (first.reshape(-1, *suffix),
            second[:, (0, 1), (0, 1)].reshape(-1, *suffix), second[:, 0, 1])


def fit_residual(entries, branch):
    fit_coefficients = []
    fit_targets = []
    for group in range(3):
        x = np.concatenate([groups(*entry['features'][branch])[group] for entry in entries])
        t = np.concatenate([groups(entry['c1'], entry['c2'])[group] for entry in entries])
        b = np.concatenate([groups(*entry['base'])[group] for entry in entries])
        scale = max(np.sqrt(np.mean(t*t)), 1e-12)*np.sqrt(len(t))
        fit_coefficients.append(x/scale)
        fit_targets.append((t-b)/scale)
    matrix = np.concatenate(fit_coefficients)
    residual = np.concatenate(fit_targets)
    column_scale = np.maximum(np.linalg.norm(matrix, axis=0), 1e-12)
    standardized = matrix/column_scale
    beta = np.linalg.solve(standardized.T @ standardized+RIDGETERM*np.eye(matrix.shape[-1]),
                           standardized.T @ residual)
    return beta/column_scale


def measure(expected, prediction):
    sse = np.square(expected-prediction).sum()
    sst = np.square(expected-expected.mean()).sum()
    return dict(r2=float(1-sse/sst) if sst > 0 else None,
                rmse=float(np.sqrt(sse/len(expected))), count=len(expected))


def evaluate(entries, branch, beta):
    y, p = [], []
    for entry in entries:
        y.append(groups(entry['c1'], entry['c2']))
        predicted = entry['base']
        if beta is not None:
            predicted = tuple(base+feat@beta for base, feat in zip(predicted, entry['features'][branch]))
        p.append(groups(*predicted))
    out = {}
    for i, tag in enumerate(('c1', 'c2_diag', 'c2_offdiag')):
        out[tag] = measure(np.concatenate([a[i] for a in y]), np.concatenate([a[i] for a in p]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-family', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch', type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir()
    torch.cuda.set_per_process_memory_fraction(.38)
    torch.manual_seed(SEED)
    manifest = json.loads(MANIFEST.read_text())
    charges = json.loads(CHARGES.read_text())
    sections = cohort(manifest['splits'])
    model = XiGrandPotential().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train = [load(row, 'train', charges, args.per_family, SEED+i, args.batch, 2)
             for i, row in enumerate(sections[0][1])]
    scales = [max(np.sqrt(np.mean(np.concatenate([r[key].ravel() for r in train])**2)), 1e-12)
              for key in ('c1', 'c2')]
    batches = [(r['source'], coeff, t) for r in train for coeff, t in r['batches']]
    for epoch in range(args.epochs):
        for i in np.random.default_rng(SEED+epoch).permutation(len(batches)):
            source, coeff, t = batches[i]
            process_batch(model, source, coeff, t, scales, BLOCK, optimizer)
        emit(stage='train_local', epoch=epoch+1, steps=(epoch+1)*len(batches))
    del batches, source
    for r in train:
        base_predictions(model, r, scales)
        del r['source'], r['batches']
    torch.save(model.state_dict(), args.output/'local_model.pt')
    fits = {name: fit_residual(train, name) for name in ('local', 'pair')}
    fits['base'] = None
    sets = {'train': train}
    for role, cases in sections[1:]:
        sets[role] = []
        for i, row in enumerate(cases):
            entry = load(row, role, charges, args.per_family, SEED+100*(1 if role=='validation' else 2)+i,
                         args.batch, 2)
            base_predictions(model, entry, scales)
            del entry['source'], entry['batches']
            sets[role].append(entry)
    summary = dict(status='complete', seed=SEED, radii_A=RADII, regularization=RIDGETERM,
                   training_epochs=args.epochs, learning_rate=3e-4, batch=args.batch,
                   base_label_scales=scales, states={role: sum(len(r['c1']) for r in cases)
                                                    for role, cases in sets.items()},
                   scores={}, by_case={}, fit={k: v.tolist() for k, v in fits.items() if v is not None})
    for role, cases in sets.items():
        summary['scores'][role] = {name: evaluate(cases, name, beta) for name, beta in fits.items()}
        for row in cases:
            summary['by_case'][row['case']] = {name: evaluate([row], name, beta) for name, beta in fits.items()}
            np.savez_compressed(args.output/(row['case']+'.npz'), targets_c1=row['c1'], targets_c2=row['c2'],
                                probes=row['probes'], state_rows=row['rows'], base_c1=row['base'][0],
                                base_c2=row['base'][1], local_c1=row['features']['local'][0],
                                local_c2=row['features']['local'][1], pair_c1=row['features']['pair'][0],
                                pair_c2=row['features']['pair'][1])
        emit(role=role, scores=summary['scores'][role])
    with (args.output/'summary.json').open('x') as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
