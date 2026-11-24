"""Rebuild the 0.5 Å smooth dataset with zero-centered Gaussian neighbor marks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kernel import Basis, combine, element_field, harmonics
from preprocess import digest, write_json
from xi_response import run_case

BASELINE = ROOT / 'xi_smooth05_accel_157349'
FAMILIES = ('original_spatial', 'angular_only', 'spatial_and_angular')


@dataclass(frozen=True)
class ZeroCenteredBasis(Basis):
    @property
    def widths(self):
        return np.geomspace(0.5, 2.5, self.radials)

    def values(self, vectors):
        r = np.linalg.norm(vectors, axis=-1)
        active = r < self.cutoff
        u = r[active] / self.cutoff
        out = np.zeros((len(r), self.channels))
        if active.any():
            envelope = (1 - u)**3 * (1 + 3*u + 6*u**2 + 10*u**3 + 15*u**4 + 21*u**5)
            radial = np.exp(-0.5 * (r[active, None] / self.widths)**2) * envelope[:, None]
            angular = harmonics(vectors[active])
            for ell in range(1, 4):
                angular[:, ell**2:(ell+1)**2] *= u[:, None]**ell
            out[active] = (radial[:, :, None] * angular[:, None, :]).reshape(-1, self.channels)
        return out


def rebuild_cache(old_cache, geometry, basis, charges, frame_limit):
    with h5py.File(old_cache, 'r') as old:
        if old.attrs['status'] != 'complete':
            raise ValueError(f'Incomplete source cache {old_cache}')
        steps = old['step'][:frame_limit]
        elements = old['elements'][:]
        atoms = np.arange(len(elements))
        zero = basis.values(np.zeros((1, 3)))[0]
        pending = geometry / 'atomic_frames.pending.h5'
        with h5py.File(pending, 'x') as new:
            new.attrs.update(old.attrs)
            new.attrs['status'] = 'writing'
            new.attrs['radial_basis'] = 'zero_centered_gaussian'
            new.attrs['radial_widths_A'] = basis.widths
            new.attrs['cutoff_A'] = basis.cutoff
            new.attrs['charges_complete'] = True
            for name in ('atom_ids', 'elements'):
                new.create_dataset(name, data=old[name][:])
            for name in ('step', 'source_indices', 'positions', 'cell', 'external_basis_energy'):
                if name in old:
                    new.create_dataset(name, data=old[name][:frame_limit])
            block = min(64, len(atoms))
            marks = new.create_dataset('neighbor_element_nlm',
                shape=(len(steps), len(atoms), 4, basis.radials, 16),
                dtype='f4', chunks=(1, block, 4, basis.radials, 16), compression='lzf')
            combined = new.create_dataset('atomic_xi_nlm',
                shape=(len(steps), len(atoms), basis.radials, 16),
                dtype='f4', chunks=(1, block, basis.radials, 16), compression='lzf')
            for t, (positions, cell) in enumerate(zip(new['positions'], new['cell'])):
                field = element_field(positions, positions, elements, cell, basis, chunk=64)
                field[atoms, elements] -= zero
                if not np.isfinite(field).all():
                    raise ValueError(f'Nonfinite marks in {geometry.name}, frame {t}')
                marks[t] = field.reshape(marks.shape[1:])
                combined[t] = combine(field, charges).reshape(combined.shape[1:])
                if t == 0 or (t + 1) % 50 == 0:
                    print(json.dumps(dict(stage='geometry', case=geometry.name,
                                          frames_done=t+1, frames=len(steps))), flush=True)
            new.attrs['status'] = 'complete'
    cache = geometry / 'atomic_frames.h5'
    pending.rename(cache)
    return cache, steps


def reprocess_case(task):
    entry, output, charges, smoke = task
    case = entry['case']
    source_cache = Path(entry['cache'])
    source_geom = source_cache.parent
    geometry = output / 'geometry' / case
    geometry.mkdir()
    old_meta = json.loads((source_geom/'summary.json').read_text())
    basis = ZeroCenteredBasis(old_meta['cutoff_A'], old_meta['radials'])
    # The spatial probe profiles retain the original radial basis as fixed external fields.
    (geometry/'external_basis.json').write_bytes((source_geom/'external_basis.json').read_bytes())
    cache, steps = rebuild_cache(source_cache, geometry, basis, charges, 72 if smoke else None)
    old_labels_path = ROOT / entry['labels']
    with np.load(old_labels_path) as old_labels:
        np.testing.assert_array_equal(steps, old_labels['source_steps'][:len(steps)])
        # Keep spatial amplitudes. Generate angular and mixed amplitudes at the original
        # perturbation strength levels using the newly computed angular probe energies.
        physical = int(old_labels['physical_probe_count'])
        spatial = old_labels['candidate_family'] == 'original_spatial'
        amplitudes = old_labels['coefficients'][spatial, :physical]
        if smoke:
            amplitudes = amplitudes[np.argsort(-old_labels['ess'][spatial], kind='stable')[:8]]
        additional = 16 if smoke else int((~spatial).sum())
        np.savez(geometry/'three_dimensional_perturbations.npz', source_steps=steps,
                 coefficients=amplitudes)
        write_json(geometry/'summary.json', dict(old_meta, status='complete',
                   selected_frames=len(steps), basis='zero_centered_gaussian',
                   radial_widths_A=basis.widths.tolist(), cache_source=str(source_cache),
                   charge_weights=charges, combined_xi_available=True,
                   frame_cache_sha256=digest(cache)))
        report = run_case((geometry, output, charges, 0.5, additional, 'smoothstep', None))
        labels_path = output/case/'xi_response_labels.npz'
        with np.load(labels_path) as labels:
            physical = int(labels['physical_probe_count'])
            np.testing.assert_array_equal(labels['shape'], old_labels['shape'])
            np.testing.assert_array_equal(labels['source_steps'], steps)
            np.testing.assert_array_equal(labels['angular_channels'], old_labels['angular_channels'])
            np.testing.assert_array_equal(labels['coefficients'][:len(amplitudes), :physical], amplitudes)
            np.testing.assert_allclose(labels['probe_energy'][:, :physical],
                                       old_labels['probe_energy'][:len(steps), :physical],
                                       rtol=1e-11, atol=1e-10)
            angular_difference = np.max(np.abs(labels['probe_energy'][:, physical:]
                                        - old_labels['probe_energy'][:len(steps), physical:]))
            if not np.isfinite(angular_difference) or angular_difference <= 1e-10:
                raise ValueError(f'New descriptor did not change angular probes for {case}')
            np.testing.assert_allclose(labels['weights'].sum(axis=1), 1., rtol=1e-12)
            if not smoke:
                np.testing.assert_array_equal(labels['candidate_family'], old_labels['candidate_family'])
            if smoke and any(report['accepted_by_family'].get(name, 0) == 0 for name in FAMILIES):
                raise ValueError('Smoke case must retain a label in each perturbation family')
    return dict(report, labels=str(labels_path), angular_energy_max_change=float(angular_difference))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    output = args.output.resolve()
    baseline_manifest = BASELINE/'training_split.json'
    split = json.loads(baseline_manifest.read_text())
    if args.smoke:
        split['splits'] = {'train': [next(row for row in split['splits']['train']
            if row['case'] == 'H2O-T986.4-r08-rho0.0922')]}
    entries = [row for rows in split['splits'].values() for row in rows]
    holdout = {'NH3', 'C2H6'}
    for role, rows in split['splits'].items():
        for row in rows:
            if (row['species'] in holdout) != (role == 'test'):
                raise ValueError(f'Unexpected split for {row["case"]}')
    if len({row['case'] for row in entries}) != len(entries):
        raise ValueError('Trajectory assigned more than once')
    output.mkdir()
    (output/'geometry').mkdir()
    charges = json.loads((ROOT/'charges_nuclear.json').read_text())
    write_json(output/'configuration.json', dict(baseline=str(baseline_manifest),
               basis='zero_centered_gaussian', radial_widths_A=ZeroCenteredBasis().widths.tolist(),
               cutoff_A=5., max_spacing_A=0.5, deposition='smoothstep',
               spatial_probe_basis='original_fixed_profiles', angular_probe_marks='new_basis',
               spatial_candidate_source=str(BASELINE), angular_amplitudes='rescaled_to_probe_energy_std',
               smoke=args.smoke, charge_weights=charges))
    with ProcessPoolExecutor(max_workers=min(args.workers, len(entries))) as pool:
        tasks = [(entry, output, charges, args.smoke) for entry in entries]
        by_case = {report['case']: report for report in pool.map(reprocess_case, tasks)}
    for role, rows in split['splits'].items():
        for row in rows:
            report = by_case[row['case']]
            row.update(labels=report['labels'], cache=report['cache'], states=report['states'])
    families = {role: {name: sum(by_case[row['case']]['accepted_by_family'].get(name, 0)
                                  for row in rows) for name in FAMILIES}
                for role, rows in split['splits'].items()}
    write_json(output/'summary.json', dict(status='complete', cases=list(by_case.values()),
               accepted_by_split_and_family=families))
    if not args.smoke:
        split.update(source_job=int(os.environ['SLURM_JOB_ID']), reference_candidates=None,
                     spatial_candidate_source=str(BASELINE), radial_basis='zero_centered_gaussian')
        write_json(output/'training_split.json', split)
    print(json.dumps(families), flush=True)


if __name__ == '__main__':
    main()
