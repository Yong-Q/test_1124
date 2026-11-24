"""GPU ablation for probe sampling and loss scaling on smooth 0.5 Å xi."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from model import XiGrandPotential
from pair_kernel_audit import run as audit
from portable_io import ResponseWeights
from train_xi_full import SparseTrajectory, process_batch, sample_probes
from train_xi_joint import label_scales


SEED = 617
BLOCK = 1024
BATCH = 12
BRANCHES = ('uniform', 'balanced')
GROUPS = ('c1_space', 'c1_angular', 'c2_diag_space', 'c2_diag_angular',
          'c2_off_space_space', 'c2_off_space_angular', 'c2_off_angular_angular')


def stratified_probes(rng, count, phase):
    probes = np.empty((count, 2), dtype=int)
    for i in range(count):
        family = (i + phase) % 3
        if family == 0:
            probes[i] = rng.choice(32, 2, replace=False)
        elif family == 2:
            probes[i] = 32 + rng.choice(16, 2, replace=False)
        else:
            probes[i] = rng.integers([0, 32], [32, 48])
    return probes


def group_masks(probes):
    angular = probes >= 32
    degree = angular.sum(1)
    diagonal = np.eye(2, dtype=bool)[None, :, :]
    return {
        'c1_space': ~angular, 'c1_angular': angular,
        'c2_diag_space': diagonal & ~angular[:, :, None],
        'c2_diag_angular': diagonal & angular[:, :, None],
        'c2_off_space_space': ~diagonal & (degree == 0)[:, None, None],
        'c2_off_space_angular': ~diagonal & (degree == 1)[:, None, None],
        'c2_off_angular_angular': ~diagonal & (degree == 2)[:, None, None],
    }


def weighted_scales(probes, scales):
    masks = group_masks(probes)
    result = []
    for names in (GROUPS[:2], GROUPS[2:]):
        weights = np.empty(masks[names[0]].shape)
        active = sum(bool(masks[name].any()) for name in names)
        for name in names:
            mask = masks[name]
            weights[mask] = scales[name] * np.sqrt(active * mask.mean())
        result.append(weights)
    return tuple(result)


def select_rows(case, seed):
    with np.load(case['labels']) as data:
        rows, _ = audit.select_states(data, seed, per_family=8)
    return rows


def proportional_rows(cases, total, seed):
    populations = np.array([case['states'] for case in cases])
    if not 1 <= total <= populations.sum():
        raise ValueError('Training sample size must be within the training split')
    exact = populations * total / populations.sum()
    counts = exact.astype(int)
    order = np.argsort(-(exact - counts), kind='stable')
    counts[order[:total - counts.sum()]] += 1
    rng = np.random.default_rng(seed)
    return {case['case']: rng.choice(n, count, replace=False)
            for case, n, count in zip(cases, populations, counts) if count}


def group_rms(cases, rows_by_case):
    i, j = np.triu_indices(48, k=1)
    families = (i >= 32).astype(int) + (j >= 32)
    sums, counts = {}, {}
    for case in cases:
        rows = rows_by_case[case['case']]
        with np.load(case['labels']) as data:
            c1 = data['c1_projected'][rows]
            c2 = data['c2_projected'][rows]
        diagonal = np.diagonal(c2, axis1=1, axis2=2)
        for group, values in zip(GROUPS,
                                 (c1[:, :32], c1[:, 32:], diagonal[:, :32], diagonal[:, 32:],
                                  *(c2[:, i[families == k], j[families == k]] for k in range(3)))):
            sums[group] = sums.get(group, 0.) + np.square(values, dtype=np.float64).sum()
            counts[group] = counts.get(group, 0) + values.size
    return {name: max(float(np.sqrt(total / counts[name])), 1e-6) for name, total in sums.items()}


def tensor(x):
    return torch.as_tensor(x, device='cuda', dtype=torch.float32)


def load(case, charges):
    with np.load(case['labels']) as data:
        return (SparseTrajectory(case, data, charges, torch.device('cuda')), ResponseWeights(data),
                data['c1_projected'], data['c2_projected'])


def make_batch(response, c1, c2, rows, probes):
    coeff = np.concatenate((response.weights[rows, :, None], response.select(rows, probes)), axis=2)
    targets = c1[rows[:, None], probes], c2[rows[:, None, None], probes[:, :, None], probes[:, None, :]]
    return tensor(coeff), tuple(map(tensor, targets)), targets


def evaluate(models, cases, charges, rows_by_case, scales, batch):
    scores = {branch: {} for branch in models}
    for index, case in enumerate(cases):
        source, response, c1, c2 = load(case, charges)
        rows = rows_by_case[case['case']]
        probes = stratified_probes(np.random.default_rng(SEED + 6000 + index), len(rows), index)
        for lo in range(0, len(rows), batch):
            subset = slice(lo, lo + batch)
            pair = probes[subset]
            coeff, targets, labels = make_batch(response, c1, c2, rows[subset], pair)
            masks = group_masks(pair)
            masks['c1'] = np.ones_like(labels[0], dtype=bool)
            masks['c2_diag'] = masks['c2_diag_space'] | masks['c2_diag_angular']
            masks['c2_offdiag'] = ~masks['c2_diag']
            for branch, model in models.items():
                _, predictions = process_batch(model, source, coeff, targets, scales, BLOCK, None,
                                               return_predictions=True)
                values = tuple(x.cpu().numpy() for x in predictions)
                for group, mask in masks.items():
                    channel = int(group.startswith('c2'))
                    y, p = labels[channel][mask], values[channel][mask]
                    stats = (y.size, y.sum(dtype=np.float64), np.square(y, dtype=np.float64).sum(),
                             np.square(y - p, dtype=np.float64).sum())
                    for cohort in ('all', case['species']):
                        key = scores[branch].setdefault(cohort, {})
                        key[group] = key.get(group, np.zeros(4)) + stats
        del source, response
    for branch in scores.values():
        for cohort in branch.values():
            for group, (n, total, square, error) in cohort.items():
                variance = square - total**2 / n
                cohort[group] = dict(count=int(n), rmse=float(np.sqrt(error / n)),
                                     r2=float(1 - error / variance) if variance > 0 else None)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train-states', type=int, help='Use a proportional sample of the full training split')
    parser.add_argument('--batch', type=int, default=BATCH)
    parser.add_argument('--memory-fraction', type=float, default=.25)
    args = parser.parse_args()
    if min(args.epochs, args.batch) < 1 or not 0 < args.memory_fraction <= .5:
        parser.error('Use positive epochs/batch and a memory fraction within (0, 0.5]')
    args.output.mkdir()
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.manual_seed(SEED)
    splits = json.loads(audit.MANIFEST.read_text())['splits']
    charges = json.loads(audit.CHARGES.read_text())
    train = dict(audit.cohort(splits))['train']
    evaluation = {'train': train, 'validation': splits['validation'], 'test': splits['test']}
    if args.train_states is not None:
        evaluation['test'] = dict(audit.cohort(splits))['test']
    eval_rows = {case['case']: select_rows(case, SEED + 100 * split + index)
            for split, cases in enumerate(evaluation.values()) for index, case in enumerate(cases)}
    train_rows = eval_rows
    if args.train_states is not None:
        train_rows = proportional_rows(splits['train'], args.train_states, SEED + 701)
        train = [case for case in splits['train'] if case['case'] in train_rows]
        # Evaluate the frozen training fit on 24 actual training rows per representative trajectory.
        for index, case in enumerate(evaluation['train']):
            name = case['case']
            selected = train_rows.get(name, np.empty(0, dtype=int))
            eval_rows[name] = np.random.default_rng(SEED + 800 + index).choice(
                selected, min(24, len(selected)), replace=False)
        evaluation['train'] = [case for case in evaluation['train'] if len(eval_rows[case['case']])]
    uniform = label_scales(splits['train'])
    balanced = group_rms(train, train_rows)
    models = {name: XiGrandPotential().cuda() for name in BRANCHES}
    models['balanced'].load_state_dict(models['uniform'].state_dict())
    optimizers = {key: torch.optim.AdamW(model.parameters(), lr=3e-4) for key, model in models.items()}
    training_count = sum(len(train_rows[case['case']]) for case in train)
    audit.emit(stage='started', device=torch.cuda.get_device_name(), epochs=args.epochs,
               batch=args.batch, memory_fraction=args.memory_fraction, training_states=training_count,
               uniform_scales=uniform, balanced_scales=balanced,
               eval_rows={k: v.tolist() for k, v in eval_rows.items()})
    history = []
    updates = 0
    for epoch in range(1, args.epochs + 1):
        rng = np.random.default_rng(SEED + epoch)
        trained = 0
        for index in rng.permutation(len(train)):
            started = monotonic()
            case = train[index]
            source, response, c1, c2 = load(case, charges)
            state = rng.permutation(train_rows[case['case']])
            pairs = {'uniform': sample_probes(rng, len(state)),
                     'balanced': stratified_probes(rng, len(state), epoch + index)}
            for lo in range(0, len(state), args.batch):
                subset = slice(lo, lo + args.batch)
                for branch, model in models.items():
                    pair = pairs[branch][subset]
                    coeff, target, _ = make_batch(response, c1, c2, state[subset], pair)
                    scale = uniform if branch == 'uniform' else tuple(map(tensor, weighted_scales(pair, balanced)))
                    process_batch(model, source, coeff, target, scale, BLOCK, optimizers[branch])
                updates += 1
            del source, response
            trained += len(state)
            if epoch == 1:
                audit.emit(stage='trained_case', epoch=epoch, case=case['case'],
                           trained_states=trained, seconds=round(monotonic() - started, 2),
                           peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        audit.emit(stage='epoch_complete', epoch=epoch, updates_per_model=updates)
        if epoch in {1, 2, 5, 10, 20, 35, args.epochs}:
            sections = evaluation if epoch == args.epochs else {k: evaluation[k] for k in ('train', 'validation')}
            report = {role: evaluate(models, cases, charges, eval_rows, uniform, args.batch)
                      for role, cases in sections.items()}
            history.append(dict(epoch=epoch, scores=report))
            audit.emit(stage='evaluation', **history[-1])
            torch.save({branch: model.state_dict() for branch, model in models.items()},
                       args.output / f'epoch_{epoch:02d}.pt')
    temporary = args.output / 'summary.pending'
    with temporary.open('x') as handle:
        json.dump(dict(epochs=args.epochs, training_states=training_count, updates_per_model=updates,
                       states={k: sum(len(eval_rows[c['case']]) for c in v) for k, v in evaluation.items()},
                       peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                       history=history), handle, indent=2, allow_nan=False)
    temporary.rename(args.output / 'summary.json')


if __name__ == '__main__':
    main()
