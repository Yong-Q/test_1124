"""Generate 3D per-atom external backgrounds on one prepared training case."""

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import h5py
import numpy as np
from scipy.sparse import csr_matrix, vstack

from atomic_backgrounds import backgrounds, temperature_from_case
from kernel import box_lengths
from xi_response import deposition_stencil, occupied_counts, projected_target_blocks, state_weights

PROBE_COUNT = 152


def occupations(positions, elements, cell, shape):
    frames = []
    for xyz in positions:
        bins, mass = deposition_stencil(xyz, cell, shape, 'smoothstep')
        used = mass > 0
        columns = bins * 4 + elements[:, None]
        frames.append(csr_matrix((mass[used], (np.zeros(used.sum(), dtype=np.int64),
                                               columns[used])),
                                 shape=(1, int(np.prod(shape)) * 4)))
    compact, _ = occupied_counts(vstack(frames, format='csr'))
    np.testing.assert_allclose(np.asarray(compact.sum(axis=1)).ravel(),
                               len(elements), rtol=1e-12)
    return compact.T.tocsr()


def project(counts, probes, weights, external_energy):
    extended = np.column_stack((probes, external_energy))
    centered = extended - weights @ extended
    dw = -weights[:, None] * centered
    mean = counts @ weights
    log_mean = np.log(mean, out=np.zeros_like(mean), where=mean > 0)
    inverse = np.divide(1., mean, out=np.zeros_like(mean), where=mean > 0)
    # Contract in frame space before multiplying the 153 tangents. This uses
    # the same ideal Hessian D.T @ diag(1/mean) @ D as preprocessing.
    frame_gram = (counts.T @ counts.multiply(inverse[:, None]).tocsr()).toarray()
    c1 = (external_energy + counts.T @ log_mean) @ dw
    total = (centered * weights[:, None]).T @ centered
    c2 = dw.T @ frame_gram @ dw - total
    return c1, (c2 + c2.T) * .5


def initialize_workers(counts, probes, weights, energy):
    global context
    context = counts, probes, weights, energy


def project_row(row):
    counts, probes, weights, energy = context
    return project(counts, probes, weights[row], energy[:, row])


def _relative_paths(case, original, output):
    case = dict(case)
    for key in ('cache', 'labels'):
        case[key] = os.path.relpath((original / case[key]).resolve(), output)
    if 'additional_labels' in case:
        value = case['additional_labels']
        paths = [value] if isinstance(value, str) else value
        case['additional_labels'] = [
            os.path.relpath((original / path).resolve(), output) for path in paths
        ]
    return case


def run(manifest_path, case_id, count, output, seed, workers):
    if min(count, workers) < 1 or seed < 0:
        raise ValueError('Use positive count and workers, and a nonnegative seed')
    manifest_path, output = manifest_path.resolve(), output.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('probe_count') != PROBE_COUNT:
        raise ValueError('Expected a 152-probe preprocessing manifest')
    choices = [row for row in manifest['splits']['train'] if row['case'] == case_id]
    if len(choices) != 1 or choices[0]['species'] in manifest['holdout_species']:
        raise ValueError('Select exactly one training case; held-out molecules cannot be augmented')
    case = choices[0]
    base = manifest_path.parent
    started = perf_counter()
    output.mkdir(parents=True)
    with np.load(base / case['labels']) as labels, h5py.File(base / case['cache'], 'r') as cache:
        shape = tuple(int(x) for x in labels['shape'])
        steps, probes = labels['source_steps'], labels['probe_energy']
        dv = float(labels['voxel_volume'])
        if str(labels['deposition']) != 'smoothstep' or probes.shape != (len(steps), PROBE_COUNT):
            raise ValueError('Expected smoothstep deposition and complete probe energies')
        np.testing.assert_array_equal(cache['step'][:], steps)
        cell = cache['cell'][0]
        np.testing.assert_allclose(cache['cell'][:], np.broadcast_to(cell, cache['cell'].shape))
        positions, elements = cache['positions'][:], cache['elements'][:]
    lengths = box_lengths(cell)
    np.testing.assert_allclose(dv, lengths.prod() / np.prod(shape))
    temp = temperature_from_case(case_id)
    fields, energy, weights, draws = backgrounds(
        positions, lengths, temp, count, np.random.default_rng(seed))
    independent, quality = state_weights(energy)
    np.testing.assert_allclose(weights, independent, rtol=1e-12, atol=0.)
    if not np.all(quality['accepted']):
        raise ValueError('Retained state failed ESS')
    elapsed_fields = perf_counter() - started
    print(f'Accepted {count} 3D backgrounds from {draws} draws in {elapsed_fields:.1f} s',
          flush=True, file=sys.stderr)
    counts = occupations(positions, elements, cell, shape)
    if workers == 1:
        computed = [project(counts, probes, w, energy[:, row]) for row, w in enumerate(weights)]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, count), initializer=initialize_workers,
                                 initargs=(counts, probes, weights, energy)) as executor:
            computed = list(executor.map(project_row, range(count), chunksize=2))
    first, second = map(np.stack, zip(*computed))
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise FloatingPointError('Nonfinite probe labels')
    # Independent production formula on the first and last state.
    verified = []
    for row in sorted({0, count - 1}):
        block_size = 8192
        blocks = (counts[i:i + block_size] for i in range(0, counts.shape[0], block_size))
        extended = np.column_stack((probes, energy[:, row]))
        ref1, ref2, _ = projected_target_blocks(blocks, extended, weights[row], energy[:, row])
        np.testing.assert_allclose(first[row], ref1, atol=1e-7, rtol=5e-9)
        np.testing.assert_allclose(second[row], ref2, atol=3e-6, rtol=5e-9)
        verified.append(dict(row=row, c1_max_abs=float(np.abs(first[row] - ref1).max()),
                             c2_max_abs=float(np.abs(second[row] - ref2).max())))
    part = output / 'labels.pending'
    with part.open('xb') as handle:
        np.savez_compressed(handle, source_steps=steps, state_indices=np.arange(count), shape=shape,
                            voxel_volume=dv, deposition='smoothstep', weights=weights, probe_energy=probes,
                            c1_projected=first[:, :-1].astype(np.float32),
                            c2_projected=second[:, :-1, :-1].astype(np.float32),
                            c1_combined=first[:, -1], c2_combined=second[:, -1, -1],
                            background_energy=energy,
                            background_family=np.full(count, 'cosine_cuboid_3d'), **quality)
    destination = output / 'xi_response_labels.npz'
    part.rename(destination)
    (output / 'background_fields.json').write_text(json.dumps({
        'temperature_K': temp, 'box_A': lengths.tolist(), 'seed': seed,
        'energy_units_before_RT_scaling': 'kJ/mol', 'fields': fields,
    }, indent=2)+'\n')
    manifest['splits'] = {key: [_relative_paths(c, base, output) for c in entries]
                          for key, entries in manifest['splits'].items()}
    target, = [c for c in manifest['splits']['train'] if c['case'] == case_id]
    paths = target.setdefault('additional_labels', [])
    paths.append(destination.name)
    target['states'] += count
    manifest['background'] = 'S7_cosines_plus_periodic_cuboids'
    (output / 'training_split.json').write_text(json.dumps(manifest, indent=2)+'\n')
    record = dict(status='passed', case=case_id, model_input='preprocessed_atomic_cache',
                  background='cosine_cuboid_3d', seed=seed, temperature_K=temp,
                  states=count, attempts=draws, frames=len(steps), atoms=len(elements),
                  grid=list(shape), box_A=lengths.tolist(), probes=PROBE_COUNT,
                  c2_shape=[count, PROBE_COUNT, PROBE_COUNT], verified=verified,
                  min_ess=float(quality['ess'].min()), min_block_ess=float(quality['block_ess'].min()),
                  fields_seconds=elapsed_fields, wall_seconds=perf_counter()-started)
    (output / 'summary.json').write_text(json.dumps(record, indent=2)+'\n')
    return record
