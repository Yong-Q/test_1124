"""Report projected derivative metrics, including pooled 2x2 c2 scores."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


MANIFEST = Path('xi_smooth05_accel_157349/training_split.json')


def intrinsic_targets(data, states, probes):
    weights = data['weights'][states]
    energies = data['probe_energy']
    tangent = energies.T[probes].transpose(0, 2, 1)
    centered = tangent - np.einsum('bt,btp->bp', weights, tangent)[:, None]
    amplitude = data['coefficients'][data['state_indices'][states]]
    external = amplitude @ energies.T
    first = np.einsum('bt,bt,btp->bp', external, weights, centered)
    second = np.einsum('bt,btp,btq->bpq', weights, centered, centered)
    return first, second


def label_squares(log_rows, manifest, active_splits, intrinsic=False):
    metadata = {(role, row['case']): row for role, cases in manifest.items() for row in cases}
    stats = {}
    for row in log_rows:
        role = row.get('role')
        if row['stage'] != 'features' or role not in active_splits:
            continue
        case = metadata[(role, row['case'])]
        states, probes = np.asarray(row['rows']), np.asarray(row['probes'])
        with np.load(case['labels']) as data:
            if intrinsic:
                c1, c2 = intrinsic_targets(data, states, probes)
                values = (c1, np.diagonal(c2, axis1=1, axis2=2), c2[:, 0, 1])
            else:
                c2 = data['c2_projected']
                diagonal = c2[states[:, None], probes, probes].astype(np.float64)
                cross = c2[states, probes[:, 0], probes[:, 1]].astype(np.float64)
                np.testing.assert_allclose(cross, c2[states, probes[:, 1], probes[:, 0]], rtol=1e-8, atol=1e-8)
                values = (diagonal, cross)
        moments = np.array([[y.size, y.sum(), np.square(y).sum()] for y in values])
        for group in ('all', case['species']):
            key = (role, group)
            stats[key] = stats.get(key, np.zeros_like(moments)) + moments
    return stats


def pooled_c2(scores, moments):
    errors = 0.
    for role, (count, total, squares), weight in zip(('c2_diag', 'c2_offdiag'), moments, (1, 2)):
        group = scores[role]
        if count != group['count']:
            raise ValueError(f'Logged labels do not match the count for {role}')
        sst = squares - total**2 / count
        sse = count * group['rmse']**2
        if group['r2'] is not None and not math.isclose(1 - sse / sst, group['r2'], abs_tol=1e-8):
            raise ValueError(f'Logged labels do not match the variance for {role}')
        errors += weight * sse
    n, total, squares = moments[0] + 2 * moments[1]
    variance = squares - total**2 / n
    return {'count': int(n), 'rmse': math.sqrt(errors / n),
            'r2': 1 - errors / variance if variance > 0 else None}


def intrinsic_metrics(scores, moments):
    # The analytic ideal derivatives cancel in the residual. Only the reference
    # variance changes when moving from excess correlations to intrinsic derivatives.
    first, diagonal, cross = moments
    pooled = diagonal + 2 * cross
    c2_sse = scores['c2_diag']['count'] * scores['c2_diag']['rmse']**2
    c2_sse += 2 * scores['c2_offdiag']['count'] * scores['c2_offdiag']['rmse']**2
    result = {}
    for name, stats in zip(('c1', 'c2_pooled', 'c2_diag', 'c2_offdiag'), (first, pooled, diagonal, cross)):
        n, total, square = stats
        if name == 'c2_pooled':
            sse = c2_sse
        else:
            if n != scores[name]['count']:
                raise ValueError(f'Logged intrinsic targets do not match the count for {name}')
            sse = n * scores[name]['rmse']**2
        variance = square - total**2 / n
        result[name] = dict(count=int(n), rmse=math.sqrt(sse / n),
                            r2=1 - sse / variance if variance > 0 else None,
                            reference_rms=math.sqrt(square / n),
                            rmse_over_reference_rms=math.sqrt(sse / square) if square > 0 else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--summary', type=Path, help='Include final species-specific metrics when available')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--intrinsic', action='store_true', help='Score total intrinsic derivatives and grand-potential curvature')
    args = parser.parse_args()
    with args.log.open() as handle:
        rows = [json.loads(line) for line in handle if line.endswith('\n')]
    if args.summary:
        scores = json.loads(args.summary.read_text())['scores']
    else:
        scores = {row['role']: {'all': row['scores']} for row in rows if row['stage'] == 'split_complete'}
    if not scores:
        parser.error('No completed split found')
    stats = label_squares(rows, json.loads(MANIFEST.read_text())['splits'], scores, args.intrinsic)
    metrics = []
    for role, groups in scores.items():
        for group, variants in groups.items():
            for model, values in variants.items():
                moments = stats[(role, group)]
                scores_by_metric = (intrinsic_metrics(values, moments) if args.intrinsic
                                    else {**values, 'c2_pooled': pooled_c2(values, moments)})
                for name in ('c1', 'c2_pooled', 'c2_diag', 'c2_offdiag'):
                    metrics.append(dict(split=role, group=group, model=model, metric=name,
                                        **scores_by_metric[name], negative_curvature_states=values['curvature']['negative'],
                                        states=values['curvature']['states']))
    if args.output:
        with args.output.open('x', newline='') as handle:
            write_csv(metrics, handle)
    else:
        write_csv(metrics, sys.stdout)


def write_csv(metrics, handle):
    writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
    writer.writeheader()
    writer.writerows(metrics)


if __name__ == '__main__':
    main()
