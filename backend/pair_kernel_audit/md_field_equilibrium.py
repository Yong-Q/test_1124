"""Compare independent external-field MD with constrained ξ minimization.

The archived fields depend only on z. The trial minimizes the laterally
translation-invariant sector, with independent rigid orientations and phases.
"""
import argparse
import gzip
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
from scipy.special import softmax, xlogy
from scipy.stats import qmc

from pair_kernel_audit import newton_density_trial as solver
from pair_kernel_audit.rigid_equilibrium_trial import geometry, rotations_from_unit_cube


KB = 8.617333262145e-5
SPECIES = ('NH3', 'C2H6', 'C2H4', 'CH4')
TYPES = {6: 1, 1: 2, 7: 3}
ARCHIVE = Path('../zenodo_23901522_missing_20261115')


def field_input(path, length):
    """Integrate the force constants in the actual production input (metal units)."""
    source = path.read_text()
    required = ('units metal', 'boundary p p p', 'fix ext all addforce 0.0 0.0 v_fz',
                'variable fz atom "v_fw+v_fg"')
    if not all(line in source for line in required):
        raise ValueError(f'Unexpected field definition in {path}')
    find = lambda name: re.search(rf'^variable {name} atom "(.*)"$', source, re.M)[1]
    center, = re.fullmatch(r'z-([\d.]+)', find('zc')).groups()
    fw = find('fw')
    if fw == '0.0':
        half, ramp, wall, exponent = 0.,1.,0.,3.
    else:
        half, ramp = re.fullmatch(r'\(v_za-([\d.]+)\)/([\d.]+)', find('u')).groups()
        wall, exponent = re.fullmatch(r'-([\d.]+)\*v_sgn\*v_uc\^([\d.]+)\*v_gt0\*v_lt1',fw).groups()
    pattern = r'([\d.]+)\*\(z-([\d.]+)\)/([\d.]+)\*exp\(-\(\(z-\2\)\^2\)/\(2\*\3\)\)'
    fg = find('fg')
    gaussian = (np.empty((0,3)) if fg=='0.0' else
                np.array([re.fullmatch(pattern,term).groups() for term in fg.split('+')],dtype=float))
    temp = re.findall(r'^fix 1 all nvt temp ([\d.]+) \1 ', source, re.M)
    density = re.search(r'^read_data .*/r05_([\d.]+)\.lammps$', source, re.M)
    if not temp or density is None or (float(wall)==0 and not len(gaussian)) or abs(2 * float(center) - length) > 1e-4:
        raise ValueError(f'Inconsistent box or temperature metadata in {path}')
    return (float(temp[-1]), np.array([center,half,ramp,wall,exponent],dtype=float), gaussian), float(density[1])


def potential(z, length, temp, wall, gaussian):
    center,half,ramp,force = wall[:4]
    power = wall[4]+1 if len(wall)==5 else 4.
    displacement = np.clip((np.abs(z % length - center) - half) / ramp, 0., 1.)
    delta = (z % length)[..., None] - gaussian[:, 1]
    return (force * ramp / power * displacement**power +
            np.sum(gaussian[:, 0] * np.exp(-.5 * delta**2 / gaussian[:, 2]), axis=-1)) / (KB * temp)


def stencil(z, length, nz):
    u = z % length / length * nz - .5
    low = np.floor(u).astype(int)
    fraction = u - low
    upper = fraction**2 * (3 - 2 * fraction)
    return np.stack((low % nz, (low + 1) % nz), -1), np.stack((1 - upper, upper), -1)


def rigid_template(species):
    if species == 'CH4':
        directions = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        return np.vstack((np.zeros((1, 3)), 1.087 * directions / np.sqrt(3))), np.array([6,1,1,1,1])
    if species != 'C2H4':
        return geometry(species)
    c = .5 * 1.339
    x = c + 1.087 * np.cos(np.deg2rad(58.7))
    y = 1.087 * np.sin(np.deg2rad(58.7))
    return np.array([[-c, 0, 0], [c, 0, 0], [-x, -y, 0], [-x, y, 0], [x, -y, 0], [x, y, 0]]), np.array([6,6,1,1,1,1])


def refine(lift, weights, slope, scale, max_steps=40000):
    """Move mass between simplex coordinates with an exact convex line search."""
    columns = lift.tocsc()
    w = weights.copy()
    mass = lift @ w
    floor = solver.FLOOR
    trace = []
    target = 1e-6
    for step in range(max_steps):
        if step % 1000 == 0:
            mass = lift @ w
            padded = mass + floor
            trace.append(float(scale * (padded @ np.log(padded)) + slope @ w))
        gradient = lift.T @ (scale * (1 + np.log(np.maximum(mass, 0.) + floor))) + slope
        enter = int(np.argmin(gradient))
        gap = float(w @ gradient - gradient[enter])
        if gap <= target:
            break
        leave = int(np.argmax(np.where(w > 0, gradient, -np.inf)))
        e = slice(columns.indptr[enter], columns.indptr[enter + 1])
        l = slice(columns.indptr[leave], columns.indptr[leave + 1])
        sites = np.union1d(columns.indices[e], columns.indices[l])
        direction = np.zeros(len(sites))
        direction[np.searchsorted(sites, columns.indices[e])] += columns.data[e]
        direction[np.searchsorted(sites, columns.indices[l])] -= columns.data[l]

        def derivative(a):
            trial = np.maximum(mass[sites] + a * direction, 0.) + floor
            return float(scale * (direction @ (1 + np.log(trial))) + slope[enter] - slope[leave])

        amount = w[leave]
        if derivative(amount) > 0:
            amount = brentq(derivative, 0., amount, xtol=1e-16)
        w[enter] += amount
        w[leave] -= amount
        mass[sites] += amount * direction
    np.testing.assert_allclose(w.sum(), 1., atol=1e-12)
    padded = (lift @ w) + floor
    gradient = lift.T @ (scale * (1 + np.log(padded))) + slope
    gap = float(w @ gradient - gradient.min())
    trace.append(float(scale * (padded @ np.log(padded)) + slope @ w))
    return w, dict(status='stationary' if gap <= target else 'residual_above_tolerance',
                   stationarity_gap=gap, tolerance=target, omega_trace=trace, steps=step + 1)


def predict(species, lengths, energy_config, alpha, beta, start):
    relative, charges = rigid_template(species)
    length = lengths[2]
    nz = int(np.ceil(length / .5))
    spacing = length / nz
    centers = (np.arange(nz) + .5) * spacing
    samples = qmc.Sobol(4, scramble=True, seed=20261119).random_base2(6)
    orientations = rotations_from_unit_cube(samples[:, 1:])
    offsets = orientations[:, 2] @ relative.T + spacing * (samples[:, :1] - .5)
    z = (centers[:, None, None] + offsets[None]).reshape(-1, len(relative))
    natoms = len(relative)
    bins, weights = stencil(z, length, nz)
    rows = (bins * natoms + np.arange(natoms)[None, :, None]).ravel()
    cols = np.broadcast_to(np.arange(len(z))[:, None, None], bins.shape).ravel()
    lift = csr_matrix((weights.ravel(), (rows, cols)), shape=(nz * natoms, len(z)))
    lift.eliminate_zeros()
    external = potential(z, length, *energy_config).sum(axis=1)
    # Lateral phases average to the same additive constant for each rigid configuration.
    phase = xlogy(weights, weights).sum((1, 2)) / natoms
    uniform = np.full(len(z), 1. / len(z))
    field_guess = softmax(-external)
    initial = field_guess if start == 'external' else uniform

    def sites(w):
        return (lift @ w).reshape(nz, natoms)

    def profiles(w):
        site = sites(w)
        return np.stack([site[:, charges == element].sum(1) for element in TYPES], axis=1)

    result = {'uniform': profiles(uniform), 'ideal_rigid': profiles(field_guess)}
    records = {}
    for name, correction in (('voxel', 0.), ('voxel_phase', beta)):
        slope = external + correction * phase

        def objective(w):
            f, df, hessian = solver.density_jet(sites(w)[None, None], 'voxel')
            return (alpha * f + slope @ w, alpha * (lift.T @ df.ravel()) + slope,
                    lambda d: alpha * (lift.T @ hessian(sites(d)[None, None]).ravel()))

        w, info = solver.newton(objective, initial, 120)
        result[f'{name}_newton'] = profiles(w)
        w, final = refine(lift, w, slope, alpha / natoms)
        result[name], records[name] = profiles(w), dict(newton=info, refinement=final)
    return result, records


def md_profiles(path, lengths, nz, stride=2):
    """Average a fixed, predeclared set of production frames, with 20 time blocks."""
    profiles = []
    with gzip.open(path, 'rt') as stream:
        frame = 0
        while stream.readline():
            step = int(stream.readline())
            if stream.readline().strip() != 'ITEM: NUMBER OF ATOMS':
                raise ValueError('Unexpected atom count header')
            n = int(stream.readline())
            if not stream.readline().startswith('ITEM: BOX BOUNDS pp pp pp'):
                raise ValueError('Expected a periodic orthogonal box')
            bounds = np.array([stream.readline().split() for _ in range(3)], dtype=float)
            np.testing.assert_allclose(bounds[:, 1] - bounds[:, 0], lengths, rtol=1e-7)
            if stream.readline().strip() != 'ITEM: ATOMS id mol type x y z fx fy fz':
                raise ValueError('Unexpected trajectory columns')
            text = ''.join(stream.readline() for _ in range(n))
            if frame > 0 and frame % stride == 0:
                values = np.fromstring(text, sep=' ').reshape(n, 9)
                bins, mass = stencil(values[:, 5] - bounds[2, 0], lengths[2], nz)
                profiles.append(np.stack([np.bincount(bins[values[:, 2] == kind].ravel(),
                                weights=mass[values[:, 2] == kind].ravel(), minlength=nz)
                                for kind in TYPES.values()], axis=1))
            if step != frame * 1000:
                raise ValueError(f'Unexpected step {step} at frame {frame}')
            frame += 1
    if frame != 2001:
        raise ValueError(f'Incomplete production trajectory: {frame} frames')
    blocks = np.array([x.mean(0) for x in np.array_split(profiles, 20)])
    return blocks.mean(0), blocks.std(0, ddof=1) / np.sqrt(len(blocks)), len(profiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--new-trajectories', type=Path, required=True)
    parser.add_argument('--start', choices=('external', 'uniform'), default='external')
    args = parser.parse_args()
    args.output.mkdir()
    fit = json.loads(args.fit.read_text())
    summary = {}
    for species in SPECIES:
        case = f'{species}_r05_field'
        metadata = ARCHIVE / 'energy_deliverables/extracted'
        source = metadata / f'inputs/{case}_in.les_md'
        if species == 'CH4':
            center = float(re.search(r'^variable zc atom "z-([\d.]+)"$', source.read_text(), re.M)[1])
            lengths = np.full(3, 2 * center)
        else:
            config = json.loads((metadata / f'field/{case}_field_config.json').read_text())
            lengths = np.full(3, config['box_length_z_A'])
        force, molecular_density = field_input(source, lengths[2])
        ncopies = round(molecular_density * np.prod(lengths))
        _, charges = rigid_template(species)
        totals = ncopies * np.array([np.count_nonzero(charges == element) for element in TYPES])
        profiles, stats = predict(species, lengths, force, fit['alpha'], fit['beta'], args.start)
        nz = len(profiles['uniform'])
        voxel_slab = np.prod(lengths) / nz
        densities = {name: p * ncopies / voxel_slab for name, p in profiles.items()}
        for name, profile in densities.items():
            np.testing.assert_allclose(profile.sum(0) * voxel_slab, totals, rtol=1e-12, atol=1e-8, err_msg=name)
        np.savez(args.output / f'{case}_predictions.npz', **densities)
        reference_root = args.new_trajectories if species == 'CH4' else ARCHIVE / 'files/trajectories'
        trajectory = reference_root / f'{case}.dump.gz'
        counts, uncertainty, frames = md_profiles(trajectory, lengths, nz)
        np.testing.assert_allclose(counts.sum(0), totals, rtol=1e-12, atol=1e-8)
        reference = counts / voxel_slab
        rmse = {}
        for name, p in densities.items():
            difference = p - reference
            rmse[name] = dict(relative_l2=float(np.linalg.norm(difference) / np.linalg.norm(reference)),
                              rmse_per_A3=float(np.sqrt(np.mean(difference[:, totals > 0]**2))))
        np.savez(args.output / f'{case}_density.npz', z_A=(np.arange(nz) + .5) * lengths[2] / nz,
                 md=reference, md_block_sem=uncertainty / voxel_slab, **densities)
        summary[case] = dict(trajectory=str(trajectory.resolve()), force_input=str(source.resolve()),
                             temperature_K=force[0], box_A=lengths.tolist(), frames=frames, rmse=rmse, iteration=stats)
        print(case, {k: v['relative_l2'] for k,v in rmse.items()},
              {k: (v['refinement']['status'], v['refinement']['stationarity_gap']) for k,v in stats.items()}, flush=True)
    with (args.output / 'summary.json').open('x') as stream:
        json.dump(dict(status='complete', fit=str(args.fit.resolve()), scope=__doc__.strip(), initial=args.start,
                       density_unit='atoms/angstrom^3', element_order=['C','H','N'], cases=summary), stream,
                  indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
