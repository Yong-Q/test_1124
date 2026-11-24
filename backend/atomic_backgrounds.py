"""Production 3D atomic field generator, from xi_paper_3d_backgrounds/reprocess.py."""

import re

import numpy as np
from scipy.special import expit

from xi_response import state_weights


GAS_CONSTANT = 0.00831446261815324

SPREADS = (.12, .22, .35, .50, .70, .95, 1.25)

def temperature_from_case(name):
    match = re.search(r'[-_]T(\d+(?:\.\d+)?)(?:K$|-)', name)
    if match is None or float(match[1]) <= 0.:
        raise ValueError(f'Missing positive temperature in trajectory name {name}')
    return float(match[1])

def draw_field(rng, lengths):
    amplitudes = rng.uniform(-4.2, 4.2, (3, 4))
    amplitudes[np.arange(4)[None, :] >= rng.integers(1, 5, (3, 1))] = 0.
    wells = int(rng.integers(1, 9))
    return dict(amplitudes=amplitudes, wave_numbers=rng.integers(1, 5, (3, 4)),
                phases=rng.uniform(0., 2 * np.pi, (3, 4)),
                depths=rng.uniform(-16.7, 16.7, wells), centers=rng.uniform(0., lengths, (wells, 3)),
                half_widths=rng.uniform(.5, 5., (wells, 3)), edge_widths=rng.uniform(.05, 1., wells))

def atomic_potential(xyz, lengths, field):
    """Evaluate unscaled kJ/mol at every atom's actual position under PBC."""
    potential = np.zeros(xyz.shape[:-1], dtype=float)
    for axis in range(3):
        for amplitude, wave, phase in zip(field['amplitudes'][axis], field['wave_numbers'][axis],
                                           field['phases'][axis]):
            if amplitude != 0.:
                potential += amplitude * np.cos((2 * np.pi * wave / lengths[axis]) * xyz[..., axis] + phase)
    for depth, center, half_width, edge in zip(field['depths'], field['centers'],
                                              field['half_widths'], field['edge_widths']):
        window = np.ones_like(potential)
        for axis, length in enumerate(lengths):
            delta = xyz[..., axis] - center[axis]
            distance = np.abs(delta - length * np.rint(delta / length))
            window *= expit((2. / edge) * (half_width[axis] - distance))
        potential += depth * window
    return potential

def backgrounds(xyz, lengths, temperature, count, rng):
    """Draw and screen in small batches; keep exactly count accepted backgrounds."""
    fields, energies, weights = [], [], []
    candidates = 0
    while len(fields) < count and candidates < 12 * count:
        size = min(16, count - len(fields))
        batch = [draw_field(rng, lengths) for _ in range(size)]
        energy = np.column_stack([atomic_potential(xyz, lengths, field).sum(axis=1) for field in batch])
        energy /= GAS_CONSTANT * temperature
        energy -= energy.mean(axis=0)
        spread = energy.std(axis=0)
        scale = np.minimum(1., rng.choice(SPREADS, size=size) / np.maximum(spread, 1.e-12))
        energy *= scale
        w, quality = state_weights(energy)
        for j in np.flatnonzero(quality['accepted'] & (spread > 1.e-10)):
            fields.append({k: v.tolist() for k, v in batch[j].items()} | {'scale': float(scale[j])})
            energies.append(energy[:, j])
            weights.append(w[j])
        candidates += size
    if len(fields) != count:
        raise RuntimeError(f'Accepted only {len(fields)} of {count} requested backgrounds in {candidates} draws')
    return fields, np.stack(energies, axis=1), np.stack(weights), candidates
