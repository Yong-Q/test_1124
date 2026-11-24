"""Check fixed-N axial susceptibility of the frozen atomic density functional."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import null_space
from scipy.special import softmax

from pair_kernel_audit.md_field_equilibrium import stencil
from pair_kernel_audit.prl_density_coupling import AxialProjection
from pair_kernel_audit.same_fluid_bulk_transfer import LOW_K
from pair_kernel_audit.zenodo_external_fields import curve

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'evidence/xi_same_bulk_transfer_158913'
BLIND = ROOT / 'evidence/xi_bulk_molecule_holdout_158918'
FIELDS = ROOT / 'evidence/zenodo_field_info_23901522_20261119/verified_scaling/fields.json'
REF = ROOT / 'external_fields_23901522_158562/reference'
SPLIT = ROOT / 'xi_smooth05_accel_157349/training_split.json'


def load_profiles(path, skip=0):
    with h5py.File(path, 'r') as h:
        position = h['positions'][skip:, :, 2]
        length = np.diag(h['cell'][0])
        elements = h['elements'][:]
    shape = np.ceil(length / .5).astype(int)
    spacing = length / shape
    n = shape[2]
    slab = np.prod(length) / n
    indices, weights = stencil(position, length[2], n)
    keys = indices + np.arange(len(position))[:, None, None] * n
    species_density = {
        int(element): np.bincount(keys[:, elements == element].ravel(),
                                 weights=weights[:, elements == element].ravel(),
                                 minlength=len(position) * n).reshape(-1, n) / slab
        for element in np.unique(elements)
    }
    rho = sum(species_density.values())
    np.testing.assert_allclose(rho.sum(1) * slab, len(elements), rtol=1e-12)
    return shape, spacing, slab, rho, species_density


def project(values, modes):
    return np.fft.irfft(np.fft.rfft(values) * modes, n=len(values))


def difference(predicted, reference):
    size = np.linalg.norm(reference)
    return {
        'relative_l2': float(np.linalg.norm(predicted - reference) / size),
        'gain_along_reference': float(predicted @ reference / size**2),
        'cosine': float(predicted @ reference / (size * np.linalg.norm(predicted))),
    }


def uniform_covariance(rho, slab):
    # Average over grid translations since the equilibrium reference is homogeneous.
    modes = np.fft.rfft(rho, axis=1)
    power = np.mean(abs(modes - modes.mean(0))**2, axis=0) * slab / rho.shape[1]
    power[0] = 0.
    return power


def tangent_model(operator, coefficients, mean, force, tangent):
    _, _, pieces = operator.jets(mean, value=True)
    hessian = np.einsum('k,kij->ij', coefficients, pieces)
    restricted = tangent.T @ hessian @ tangent
    slope = -tangent @ np.linalg.solve(restricted, tangent.T @ force)
    residual = tangent.T @ (hessian @ slope + force)
    assert np.linalg.norm(residual) < 1e-6 * np.linalg.norm(force)
    assert abs(slope.sum()) < 1e-10 * max(1., np.linalg.norm(slope))
    return slope, float(np.linalg.eigvalsh(restricted)[0])


def audit_bulk(path, configs, models):
    shape, spacing, slab, rho, _ = load_profiles(path)
    n = shape[2]
    z = (np.arange(n) + .5) * spacing[2]
    low = (2 * np.pi * np.fft.rfftfreq(n, spacing[2]) <= LOW_K)
    low[0] = False
    op = AxialProjection(shape, spacing)
    tangent = null_space(np.ones((1, n)))
    uniform = np.full(n, rho.mean())
    covariance = uniform_covariance(rho, slab)
    half = [uniform_covariance(part, slab) for part in np.array_split(rho, 2)]
    output = {}
    for name, config in configs.items():
        field, _ = curve(config, z * config['box_length_z_A'] / (n * spacing[2]))
        forces = {'all': field - field.mean(), 'low_k': project(field, low)}
        results = {}
        for label, force in forces.items():
            response = -project(force, covariance)
            halves = [-project(force, sample) for sample in half]
            scores = {}
            for model, coefficients in models.items():
                prediction, eigmin = tangent_model(op, coefficients, uniform, force, tangent)
                scores[model] = {
                    **difference(prediction, response),
                    'min_total_curvature_per_density': eigmin,
                }
            results[label] = dict(models=scores, half_split=difference(*halves),
                                  response_rms=float(np.sqrt(np.mean(response**2))))
        output[name] = results
    return dict(source=str(path), frames=len(rho), number_density=float(rho.mean()), cases=output)


def audit_external(name, config, models, saved):
    shape, spacing, slab, rho, channels = load_profiles(REF / name / 'source.h5', skip=1)
    n = shape[2]
    z = (np.arange(n) + .5) * spacing[2]
    field, _ = curve(config, z)
    force = field - field.mean()
    mean = rho.mean(0)
    energies = slab * (rho @ force)
    centered = energies - energies.mean()
    slope = -(centered @ (rho - mean)) / len(rho)
    per_element = {element: -(centered @ (density - density.mean(0))) / len(rho)
                   for element, density in channels.items()}
    np.testing.assert_allclose(sum(per_element.values()), slope, atol=2e-14)
    epsilon = .05 / centered.std()
    weights = [softmax(-sign * epsilon * centered) for sign in (1, -1)]
    secant = ((weights[0] - weights[1]) @ rho) / (2 * epsilon)
    halves = []
    for densities, raw_energy in zip(np.array_split(rho, 2), np.array_split(energies, 2)):
        halves.append(-((raw_energy - raw_energy.mean()) @ densities) / len(raw_energy))
    numerical_check = difference(secant, slope)
    if numerical_check['relative_l2'] > .01:
        raise ValueError(f'Finite-difference cross-check failed on {name}: {numerical_check}')
    op = AxialProjection(shape, spacing)
    # A locally counterbalanced field can make the measured density stationary in the
    # model. Its linear response tests Hessian at matched density, apart from c1 bias.
    scores = {}
    valid = mean > 2 * np.finfo(float).eps
    eigenvectors = null_space(np.ones((1, valid.sum())))
    occupied_tangent = np.zeros((n, eigenvectors.shape[1]))
    occupied_tangent[valid] = eigenvectors
    with np.load(saved / f'{name}.npz') as field_run:
        np.testing.assert_allclose(field_run['md'], mean, rtol=5e-6, atol=1e-9)
        for model, coefficients in models.items():
            prediction, eigmin = tangent_model(op, coefficients, mean, force, occupied_tangent)
            stored = field_run[model]
            scores[model] = {
                'matched_density_c2_response': difference(prediction, slope),
                'min_total_curvature_per_density': eigmin,
                'finite_field_profile': difference(stored, mean),
                'finite_field_density_change': difference(stored - mean.mean(), mean - mean.mean()),
            }
    return dict(frames=len(rho), occupied_bins=int(valid.sum()), bins=int(n),
                beta_field_rms=float(np.sqrt(np.mean(force**2))),
                epsilon=float(epsilon), covariance_vs_finite_difference=numerical_check,
                half_split=difference(*halves), models=scores,
                reference_element_response_l2={e: float(np.linalg.norm(a)) for e, a in per_element.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads((BASE / 'summary.json').read_text())
    split = json.loads(SPLIT.read_text())['splits']
    configs = json.loads(FIELDS.read_text())
    known = previous['coefficients']
    shared = next(iter(known.values()))['shared_all_axial_modes']
    assert all(x['shared_all_axial_modes'] == shared for x in known.values())
    bulk = {}
    external = {}
    for species in sorted({c['species'] for c in configs.values()}):
        cases = {n: c for n, c in configs.items() if c['species'] == species}
        if species in known:
            candidates = [c for c in split['validation'] if c['species'] == species]
            assert len(candidates) == 1 and candidates[0]['case'] in previous['bulk_validation']
            selected = candidates[0]
            models = known[species]
            saved = BASE
        else:
            candidates = [c for c in split['test'] if c['species'] == species]
            config = next(iter(cases.values()))
            with h5py.File(REF / f'{species}_r05_field' / 'source.h5', 'r') as h:
                density = len(h['elements']) / np.linalg.det(h['cell'][0])
            def density_gap(candidate):
                with h5py.File(candidate['cache'], 'r') as h:
                    return abs(len(h['elements']) / np.linalg.det(h['cell'][0]) - density)
            selected = min(candidates, key=density_gap)
            models = {'shared_all_axial': shared}
            saved = BLIND
        bulk[species] = audit_bulk(Path(selected['cache']), cases, models)
        for name, config in cases.items():
            external[name] = audit_external(name, config, models, saved)
        print(species, selected['case'], len(cases), flush=True)
    result = dict(status='complete', weak_bulk=bulk, independent_external=external,
                  coefficients_from=str(BASE / 'summary.json'), field_configs=str(FIELDS),
                  scope='canonical axial total atomic density, zero mean forcing, model total intrinsic Hessian')
    serialized = json.dumps(result, indent=2, allow_nan=False)
    with args.output.open('x') as stream:
        stream.write(serialized + '\n')


if __name__ == '__main__':
    main()
