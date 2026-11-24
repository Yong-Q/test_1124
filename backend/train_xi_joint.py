"""Trajectory-stratified baseline with joint c1/c2 supervision in xi space."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix
import torch

from kernel import ELEMENTS
from model import XiGrandPotential
from portable_io import read_grid_inputs
from xi_response import deposition_stencil


def sparse_observations(case, shape, source_steps, charges, deposition='hard'):
    """Group occupied voxel-atom pairs and keep one mark per frame atom."""
    if 'xi_inputs' in case:
        if deposition != 'smoothstep':
            raise ValueError('Portable input requires smoothstep deposition')
        bins, weights, marks, _ = read_grid_inputs(case, shape, source_steps, charges)
        return group_observations(bins, weights, marks)
    with h5py.File(case['cache'], 'r') as h:
        if not np.array_equal(h['step'][:], source_steps):
            raise ValueError('Label and cache step mismatch for ' + case['case'])
        atoms = len(h['elements'])
        nframes = len(source_steps)
        cell = h['cell'][0]
        bins, weights = deposition_stencil(h['positions'][:],cell,shape,deposition)
        nr, nm = h['neighbor_element_nlm'].shape[-2:]
        marks = np.ones((nframes, atoms, 1 + nr * nm), dtype=np.float32)
        z = np.array([charges[e] for e in ELEMENTS], dtype=np.float32)
        for start in range(0, nframes, 8):
            stop = min(start+8, nframes)
            raw = h['neighbor_element_nlm'][start:stop]
            marks[start:stop, :, 1:] = np.einsum('taenm,e->tanm', raw, z).reshape(stop-start, atoms, nr*nm)
    return group_observations(bins, weights, marks)


def group_observations(bins, weights, marks):
    atoms = bins.shape[1]
    active = np.flatnonzero(weights)
    observation = active//weights.shape[-1]
    mass = weights.ravel()[active].astype(np.float32)
    keys = bins.ravel()[active].astype(np.int64)*atoms + observation%atoms
    _, group = np.unique(keys, return_inverse=True)
    return group, marks.reshape(-1, marks.shape[-1]), atoms, observation, mass


def sparse_frames(case, shape, source_steps, charges):
    """Historical hard-deposition pilot input."""
    group,marks,atoms,_,_ = sparse_observations(case,shape,source_steps,charges)
    return group,marks,atoms


def lift_sparse(group, marks, atoms, coefficients, dv):
    """Exact projection of full atom-voxel statistics onto occupied pairs."""
    n = len(group)
    occupied = int(group.max()) + 1
    return np.stack([
        csr_matrix((np.repeat(row, atoms).astype(np.float32) / dv,
                    (group, np.arange(n))), shape=(occupied, n)) @ marks
        for row in coefficients
    ])


def contractions(model, values, dv, parameter_graph):
    """First derivatives and selected Hessian of a common local energy."""
    amplitude = values.new_zeros(len(values)-1, requires_grad=True)
    x = values[0] + (amplitude[:, None, None] * values[1:]).sum(0)
    energy = model.excess_density(x).sum() * dv
    c1, = torch.autograd.grad(-energy, amplitude, create_graph=True)
    c2 = torch.stack([torch.autograd.grad(a, amplitude, create_graph=parameter_graph,
                                          retain_graph=True)[0] for a in c1])
    return c1, c2


def joint_step(model, values, dv, target, scales, device, block, optimizer=None):
    """Accumulate whole-state derivatives with bounded GPU graph memory."""
    def chunks():
        for offset in range(0, values.shape[1], block):
            yield torch.from_numpy(values[:, offset:offset+block]).to(device)

    p = values.shape[0]-1
    pred1 = torch.zeros(p, device=device)
    pred2 = torch.zeros((p,p), device=device)
    for chunk in chunks():
        c1, c2 = contractions(model, chunk, dv, False)
        pred1 += c1.detach()
        pred2 += c2.detach()
    pred = [pred1, pred2]
    error = [a-b for a,b in zip(pred, target)]
    metrics = [float((e / s).square().mean()) for e,s in zip(error, scales)]
    if not np.isfinite(metrics).all():
        raise FloatingPointError('Nonfinite c1/c2 loss')
    if optimizer is not None:
        multipliers = [2*e / (s*s*e.numel()) for e,s in zip(error, scales)]
        optimizer.zero_grad(set_to_none=True)
        for chunk in chunks():
            parts = contractions(model, chunk, dv, True)
            sum((d*y).sum() for d,y in zip(multipliers, parts)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
    return metrics + [float((b/s).square().mean()) for b,s in zip(target,scales)]


def selected_cases(model, cases, charges, scales, device, args, seed, split, optimizer=None):
    rng = np.random.default_rng(seed)
    records = []
    order = rng.permutation(len(cases))
    for k in order:
        case = cases[k]
        with np.load(case['labels']) as data:
            if 'deposition' in data and str(data['deposition']) != 'hard':
                raise ValueError('Use train_xi_full.py for smooth-deposition labels')
            family = data['candidate_family'][data['state_indices']]
            state_rows = [int(rng.choice(np.flatnonzero(family == f))) for f in
                          ('original_spatial', 'angular_only', 'spatial_and_angular')
                          if np.any(family == f)]
            shape = tuple(data['shape'])
            source_steps = data['source_steps']
            dv = float(data['voxel_volume'])
            weights = data['weights'][state_rows]
            directions = data['dweights'][state_rows]
            c1 = data['c1_projected'][state_rows]
            c2 = data['c2_projected'][state_rows]
        groups, marks, atoms = sparse_frames(case, shape, source_steps, charges)
        for i, state_row in enumerate(state_rows):
            probes = np.array([rng.integers(0,32), rng.integers(33,48)])
            coefficients = np.vstack([weights[i], directions[i][:,probes].T])
            values = lift_sparse(groups, marks, atoms, coefficients, dv)
            target = [torch.as_tensor(a, device=device, dtype=torch.float32) for a in
                      (c1[i,probes], c2[i][np.ix_(probes,probes)])]
            losses = joint_step(model, values, dv, target, scales, device, args.block, optimizer)
            row = dict(split=split, case=case['case'], family=str(family[state_row]),
                       label_row=state_row, probes=probes.tolist(), c1_mse=losses[0],
                       c2_mse=losses[1], c1_zero_mse=losses[2], c2_zero_mse=losses[3])
            print(json.dumps(row), flush=True)
            records.append(row)
            del values
    return records


def label_scales(cases):
    squares = np.zeros(2)
    counts = np.zeros(2)
    for case in cases:
        with np.load(case['labels']) as data:
            for i,k in enumerate(('c1_projected', 'c2_projected')):
                x = data[k]
                squares[i] += np.einsum('i,i->', x.ravel(), x.ravel())
                counts[i] += x.size
    return np.maximum(np.sqrt(squares/counts), 1e-6).tolist()


def metrics(records):
    keys = ('c1_mse','c2_mse','c1_zero_mse','c2_zero_mse')
    return {key: float(np.mean([row[key] for row in records])) for key in keys}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--charges', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--block', type=int, default=16384)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    manifest = json.loads(Path(args.manifest).read_text())
    split = manifest['splits']
    charges = json.loads(Path(args.charges).read_text())
    torch.manual_seed(617)
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('Expected a GPU allocation')
    model = XiGrandPotential().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scales = label_scales(split['train'])
    print(json.dumps(dict(status='started', gpu=torch.cuda.get_device_name(), scales=scales,
                          cases={key:len(value) for key,value in split.items()})), flush=True)
    torch.save(model.state_dict(), out/'initial.pt')
    best = float('inf')
    report = []
    for epoch in range(args.epochs):
        fit = selected_cases(model, split['train'], charges, scales, device, args,
                             617+epoch, 'train', optimizer)
        val = selected_cases(model, split['validation'], charges, scales, device, args,
                             101, 'validation')
        scores = dict(epoch=epoch+1, train=metrics(fit), validation=metrics(val))
        report.append(scores)
        val_loss = scores['validation']['c1_mse'] + scores['validation']['c2_mse']
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), out/'best.pt')
        print(json.dumps(scores), flush=True)
    model.load_state_dict(torch.load(out/'best.pt', map_location=device, weights_only=True))
    test = selected_cases(model, split['test'], charges, scales, device, args, 203, 'test')
    result = dict(status='complete', holdout_species=manifest['holdout_species'],
                  coverage='one accepted state per perturbation family and two selected probes per case per epoch',
                  scales=scales, epochs=report, test=metrics(test),
                  test_by_species={species:metrics([r for r in test if r['case'].split('-')[0]==species])
                                   for species in manifest['holdout_species']})
    (out/'summary.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
