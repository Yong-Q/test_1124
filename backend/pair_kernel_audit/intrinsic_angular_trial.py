"""Match the angular audit states and directions with frozen intrinsic entropy models."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pair_kernel_audit import intrinsic_anchor_trial as anchor
from pair_kernel_audit import run as audit
from train_xi_intrinsic_full import factored_anchor_jets


PROBES = Path('logs/xi_angular_probe_gpu2_20261119_091611.out')
FIT = Path('evidence/xi_intrinsic_anchor_smoke_20261118_190024/summary.json')


def cases_from_log(path, splits):
    expected = {(role, c['case']): c for role in ('validation', 'test') for c in splits[role]}
    seen = set()
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row['stage'] != 'case':
                continue
            role, name = row['role'], row['case']
            key = (role, name)
            if key not in expected or key in seen:
                raise ValueError(f'Expected an independent validation or test trajectory: {key}')
            seen.add(key)
            states = np.asarray(row['rows'])
            pairs = {k: np.asarray(v) for k, v in row['pairs'].items()}
            if len(pairs) != 6 or any(p.shape != (len(states), 2) for p in pairs.values()):
                raise ValueError('Expected two selections for each of l=1,2,3')
            yield expected[key], role, states, pairs


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--probes-log', type=Path, default=PROBES)
    p.add_argument('--fit-summary', type=Path, default=FIT)
    p.add_argument('--atoms', type=int, default=128, help='Atom sample count, or zero for all')
    args = p.parse_args()
    if args.atoms < 0:
        p.error('--atoms must be nonnegative')
    args.output.mkdir()
    torch.cuda.set_per_process_memory_fraction(.25)
    manifest = json.loads(audit.MANIFEST.read_text())['splits']
    charges = json.loads(audit.CHARGES.read_text())
    scale = json.loads(args.fit_summary.read_text())['scales']['pooled']
    settings = SimpleNamespace(atoms=args.atoms, device='cuda')
    entries = {'validation': {}, 'test': {}}
    counts = {role: 0 for role in entries}
    for i, (case, role, rows, pairs) in enumerate(cases_from_log(args.probes_log, manifest)):
        probes = np.concatenate(list(pairs.values()))
        states = np.tile(rows, len(pairs))
        entry = anchor.case_entry(case, role, charges, settings, i, selection=(states, probes),
                                  projector=factored_anchor_jets)
        for k, name in enumerate(pairs):
            sl = slice(k*len(rows), (k+1)*len(rows))
            estimates = {key: tuple(x[sl] for x in pair) for key, pair in entry['estimates'].items()}
            entries[role].setdefault(name, []).append(dict(truth=tuple(x[sl] for x in entry['truth']),
                                                          estimates=estimates, species=case['species']))
        counts[role] += len(rows)
    scores = {}
    for role, directions in entries.items():
        if not counts[role]:
            raise ValueError(f'Missing {role} cases in angular audit')
        scores[role] = {}
        for name, cases in directions.items():
            cohorts = {'all': cases}
            for species in sorted({case['species'] for case in cases}):
                cohorts[species] = [case for case in cases if case['species']==species]
            scores[role][name] = {cohort: {key: anchor.evaluate(rows, model, value) for key,model,value in
                                           (('voxel_fixed','voxel',1.), ('pooled_fixed','pooled',1.),
                                            ('pooled_train_fit','pooled',scale))}
                                  for cohort, rows in cohorts.items()}
        audit.emit(stage='split_complete', role=role, states=counts[role])
    path = args.output/'summary.pending'
    with path.open('x') as f:
        json.dump(dict(status='complete', states=counts, atoms=args.atoms, probe_log=str(args.probes_log),
                       fit_summary=str(args.fit_summary), pooled_coefficient=scale, scores=scores,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30), f, indent=2, allow_nan=False)
    path.rename(args.output/'summary.json')


if __name__ == '__main__':
    main()
