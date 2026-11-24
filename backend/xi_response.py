"""Trajectory estimates of directional excess derivatives with respect to full xi."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import product
import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix, vstack

from kernel import Basis, ELEMENTS, box_lengths, grid
from preprocess import external_basis, weight_quality, write_json


def state_weights(energy):
    """Energy axes are frame, state; returned weights have axes state, frame."""
    log_weights = -energy.T
    weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights, weight_quality(weights)


def additional_amplitudes(probes, physical_count, count, case):
    """Generate angular-only and mixed candidates with comparable energy spread."""
    if not 0 < physical_count < probes.shape[1] or count < 1:
        raise ValueError('Physical and angular probes and a positive count are required')
    tag = ('nuclear-angle-v1:' + case).encode()
    seed = int.from_bytes(hashlib.sha256(tag).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=(count, probes.shape[1]))
    coefficients[:count//2, :physical_count] = 0.
    spread = (probes @ coefficients.T).std(axis=0)
    if not np.isfinite(spread).all() or np.any(spread < 1e-12):
        raise ValueError('Angular probes have insufficient variation for reweighting')
    levels = np.array([.15, .25, .4, .55, .75, 1., 1.25, 1.5])
    strengths = np.resize(levels, count)
    rng.shuffle(strengths)
    return coefficients * (strengths / spread)[:, None]


def projected_targets(counts, probes, weights, energy):
    """Return contractions of c1/c2 with D = d<xi>/d(probe amplitude).

    Counts are the species voxel occupations of each frame, used solely to
    subtract the ideal reference. Energy and probes are beta-scaled.
    """
    return projected_target_blocks([counts.T],probes,weights,energy)


def projected_target_blocks(blocks, probes, weights, energy):
    centered = probes - weights @ probes
    dw = -weights[:, None] * centered
    c1 = energy @ dw
    c2 = -(centered * weights[:,None]).T @ centered
    for block in blocks:
        mean = np.asarray(block @ weights)
        d = np.asarray(block @ dw)
        active = mean > 0
        if not active.all():
            mean,d = mean[active],d[active]
        c1 += np.log(mean) @ d
        c2 += d.T @ (d / mean[:,None])
    return c1, c2, dw


def occupied_counts(counts):
    """Remove unobserved species/voxel columns before forming responses."""
    columns, remapped = np.unique(counts.indices,return_inverse=True)
    compact = csr_matrix((counts.data,remapped.astype(counts.indices.dtype),counts.indptr),
                         shape=(counts.shape[0],len(columns)))
    return compact,columns


def voxel_indices(positions, cell, shape):
    lengths = box_lengths(cell)
    scaled = (positions % lengths) / lengths
    indices = np.minimum((scaled * shape).astype(int), np.array(shape) - 1)
    return np.ravel_multi_index(np.moveaxis(indices, -1, 0), shape)


def deposition_stencil(positions, cell, shape, mode='hard'):
    """Return periodic voxel IDs and conservative weights per atom."""
    if mode == 'hard':
        bins = voxel_indices(positions, cell, shape)
        return bins[..., None], np.ones(bins.shape + (1,))
    if mode != 'smoothstep':
        raise ValueError('Unknown deposition mode: ' + mode)
    lengths = box_lengths(cell)
    coordinate = (positions % lengths) / lengths * shape - .5
    lower = np.floor(coordinate).astype(np.int64)
    fraction = coordinate - lower
    upper_weight = fraction**2 * (3. - 2.*fraction)
    shape = np.asarray(shape)
    offsets = np.array(list(product((0,1), repeat=3)))
    nodes = (lower[..., None, :] + offsets) % shape
    weights = np.prod(np.where(offsets, upper_weight[..., None, :],
                                1.-upper_weight[..., None, :]), axis=-1)
    return np.ravel_multi_index(np.moveaxis(nodes, -1, 0), tuple(shape)), weights


def voxel_centers(indices, shape, lengths):
    coord = np.stack(np.unravel_index(indices,shape),axis=-1)
    return (coord+.5)*(lengths/shape)


def occupied_probe_basis(indices, shape, cell, spec, basis):
    lengths = box_lengths(cell)
    xyz = voxel_centers(indices,shape,lengths)
    out = np.zeros((len(indices),len(spec['radial'])))
    for k,(center,n,h) in enumerate(zip(spec['centers_fractional'],spec['radial'],spec['harmonic'])):
        for lo in range(0,len(indices),65536):
            sl = slice(lo,lo+65536)
            v = xyz[sl]-np.asarray(center)*lengths
            v -= lengths*np.rint(v/lengths)
            mask = np.sum(v*v,axis=1)<basis.cutoff**2
            if mask.any():
                column = out[sl,k]
                column[mask] = basis.values(v[mask])[:,n*16+h]
    return out


def marks_at(h, frame, atom_rows, charges):
    a = h['neighbor_element_nlm'][frame, atom_rows]
    element_codes = np.unique(h['elements'][:])
    missing = [ELEMENTS[e] for e in element_codes if charges.get(ELEMENTS[e]) is None]
    if missing:
        raise ValueError('Missing Zi for ' + ', '.join(missing))
    z = [charges.get(e) or 0. for e in ELEMENTS]
    return np.einsum('aenm,e->anm', a, z).reshape(len(atom_rows), -1)


def field_block(cache, coefficients, charges, shape, start, stop, deposition='hard'):
    """Lift signed or probability frame coefficients into full xi[x,i,A].

    Use state weights for the mean or one saved dweight direction for D.
    """
    with h5py.File(cache, 'r') as h:
        frames, atoms, _, nr, nm = h['neighbor_element_nlm'].shape
        coefficients = np.asarray(coefficients)
        if coefficients.shape != (frames,) or not np.isfinite(coefficients).all():
            raise ValueError('Expected one finite coefficient per source frame')
        if not 0 <= start < stop <= np.prod(shape):
            raise ValueError('Spatial block outside voxel grid')
        dv = np.linalg.det(h['cell'][0]) / np.prod(shape)
        out = np.zeros((stop-start, atoms, 1+nr*nm))
        for t, coefficient in enumerate(coefficients):
            if coefficient == 0:
                continue
            bins, stencil = deposition_stencil(h['positions'][t],h['cell'][t],shape,deposition)
            rows, corners = np.where((bins >= start) & (bins < stop) & (stencil > 0))
            if len(rows):
                ids = bins[rows,corners]-start
                mass = coefficient/dv*stencil[rows,corners]
                unique, inverse = np.unique(rows,return_inverse=True)
                marks = marks_at(h,t,unique,charges)
                np.add.at(out[...,0],(ids,rows),mass)
                np.add.at(out[...,1:],(ids,rows),mass[:,None]*marks[inverse])
        return out


def run_case(task):
    source, output, charges, spacing, additional, deposition, reference, *partition = task
    source = Path(source)
    old = json.loads((source/'summary.json').read_text())
    spec = json.loads((source/'external_basis.json').read_text())
    basis = Basis(spec['cutoff_A'], spec['radials'])
    with np.load(source/'three_dimensional_perturbations.npz') as data:
        amplitudes = data['coefficients']
        steps = data['source_steps']
    cache = source/'atomic_frames.h5'
    with h5py.File(cache, 'r') as h:
        if h.attrs['status'] != 'complete' or not np.array_equal(h['step'][:], steps):
            raise ValueError('Frame cache and perturbation steps do not match')
        cell = h['cell'][0]
        if not np.allclose(h['cell'][:], cell):
            raise ValueError('Fixed cell required for conjugate voxel potentials')
        lengths = box_lengths(cell)
        shape = tuple(np.maximum(1,np.ceil(lengths/spacing).astype(int)))
        num_voxels = int(np.prod(shape))
        dv = np.prod(lengths)/num_voxels
        nframe,natom,_ = h['positions'].shape
        elements = h['elements'][:]
        parts = []
        for lo in range(0,nframe,8):
            bins,stencil = deposition_stencil(h['positions'][lo:lo+8],cell,shape,deposition)
            nf,_,nc = bins.shape
            part = csr_matrix((stencil.ravel(),(np.repeat(np.arange(nf),natom*nc),
                               (bins*4+elements[None,:,None]).ravel())),shape=(nf,num_voxels*4))
            part.eliminate_zeros()
            parts.append(part)
        counts = vstack(parts,format='csr')
        del parts
        np.testing.assert_allclose(np.asarray(counts.sum(axis=1)).ravel(),natom,rtol=1e-12)
        counts, columns = occupied_counts(counts)
        if deposition=='smoothstep':
            probe_grid_indices,inverse = np.unique(columns//4,return_inverse=True)
            grid_basis = occupied_probe_basis(probe_grid_indices,shape,cell,spec,basis)
        else:
            xyz,_,_ = grid(cell,spacing)
            probe_grid_indices = np.arange(num_voxels)
            grid_basis = external_basis(xyz,cell,spec,basis)
            inverse = columns//4
        physical_probes = np.column_stack([counts @ v[inverse] for v in grid_basis.T])
        probes = physical_probes
        angular_channels = []
        angular_profiles = np.empty((len(grid_basis),0))
        if all(charges.get(ELEMENTS[e]) is not None for e in np.unique(elements)):
            angular_channels = [(k % basis.radials)*16+k for k in range(16)]
            if additional or reference:
                angular_channels = [16*(basis.radials//2)+k for k in range(16)]
            angular_profiles = grid_basis[:, :len(angular_channels)]
            angular = np.empty((nframe, len(angular_channels)))
            all_atoms = np.arange(natom)
            for t in range(nframe):
                marks = marks_at(h, t, all_atoms, charges)
                bins,stencil = deposition_stencil(h['positions'][t],cell,shape,deposition)
                grid_rows = np.searchsorted(probe_grid_indices,bins)
                grid_rows[stencil==0] = 0
                if not np.array_equal(probe_grid_indices[grid_rows][stencil>0],bins[stencil>0]):
                    raise ValueError('Active voxel missing from probe table')
                profiles = np.einsum('ac,acp->ap',stencil,angular_profiles[grid_rows])
                angular[t] = np.sum(marks[:,angular_channels]*profiles,axis=0)
            probes = np.column_stack((physical_probes, angular))

    # The voxel potential is exactly conjugate to the deposited xi. Recompute
    # frame weights because the old energies used continuous atomic positions.
    base_count, physical_count = amplitudes.shape
    families = np.full(base_count + additional, 'original_spatial', dtype='U20')
    if additional and not reference:
        if not angular_channels:
            raise ValueError('Every element in the trajectory needs a nuclear Z for angular perturbations')
        zeros = np.zeros((base_count, len(angular_channels)))
        new = additional_amplitudes(probes, physical_count, additional, source.name)
        amplitudes = np.vstack((np.column_stack((amplitudes, zeros)), new))
        families[base_count:base_count+additional//2] = 'angular_only'
        families[base_count+additional//2:] = 'spatial_and_angular'
    if reference:
        with np.load(Path(reference)/source.name/'xi_response_labels.npz') as ref:
            np.testing.assert_array_equal(ref['source_steps'],steps)
            np.testing.assert_array_equal(ref['angular_channels'],angular_channels)
            amplitudes = ref['coefficients']
            families = ref['candidate_family']
        if amplitudes.shape[1]!=probes.shape[1] or len(families)!=len(amplitudes):
            raise ValueError('Reference candidates do not match the probe layout')
    energy = probes[:, :amplitudes.shape[1]] @ amplitudes.T
    weights, quality = state_weights(energy)
    accepted = np.flatnonzero(quality['accepted'])
    if not len(accepted):
        raise ValueError('No states passed the voxel reweighting overlap thresholds')
    if partition:
        shard_index,shard_count = partition
        accepted = np.array_split(accepted,shard_count)[shard_index]
    transposed = counts.T.tocsr()
    del counts
    blocks = [transposed[lo:lo+8192] for lo in range(0,transposed.shape[0],8192)]
    del transposed
    p = probes.shape[1]
    c1 = np.empty((len(accepted),p))
    c2 = np.empty((len(accepted),p,p))
    dw = np.empty((len(accepted),nframe,p))
    print(json.dumps(dict(stage='response_blocks',case=source.name,states=len(accepted),
                          occupied_voxels=len(probe_grid_indices),response_blocks=len(blocks))),flush=True)
    for row,s in enumerate(accepted):
        c1[row],c2[row],dw[row] = projected_target_blocks(blocks,probes,weights[s],energy[:,s])
    if not all(np.isfinite(x).all() for x in [c1, c2, dw]):
        raise ValueError('Nonfinite response label')
    out = Path(output)/source.name
    out.mkdir(exist_ok=False)
    with (out/'labels.pending').open('xb') as f:
        np.savez_compressed(f, state_indices=accepted, source_steps=steps, shape=shape,
                            voxel_volume=dv, deposition=deposition, weights=weights[accepted], dweights=dw,
                            c1_projected=c1, c2_projected=c2, probe_grid_indices=probe_grid_indices,
                            probe_energy=probes, physical_grid_basis=grid_basis, coefficients=amplitudes,
                            candidate_family=families, physical_probe_count=physical_count,
                            angular_channels=angular_channels, angular_profiles=angular_profiles, **quality)
    os.replace(out/'labels.pending', out/'xi_response_labels.npz')
    report = dict(case=source.name, cache=str(cache.resolve()), cache_sha256=old['frame_cache_sha256'],
                  shape=list(map(int, shape)), voxel_volume=dv, candidates=len(amplitudes),
                  states=len(accepted), direct_angular_probes=len(angular_channels),
                  charge_weights=charges, probe_count=probes.shape[1],
                  target_type='projected excess xi derivatives', source_frames=nframe,
                  additional_angular_candidates=additional, deposition=deposition,
                  reference_candidates=None if not reference else str(Path(reference).resolve()),
                  accepted_by_family={f: int(np.sum(families[accepted] == f)) for f in np.unique(families)},
                  old_weights='recomputed for voxel-conjugate potential')
    write_json(out/'summary.json', report)
    print(json.dumps(report), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--charges', required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--spacing', type=float, default=2.)
    p.add_argument('--extra-angular', type=int, default=0)
    p.add_argument('--extra-water-angular', type=int)
    p.add_argument('--deposition', choices=['hard','smoothstep'], default='hard')
    p.add_argument('--reference-candidates')
    args = p.parse_args()
    if not np.isfinite(args.spacing) or args.spacing<=0 or args.workers<1:
        p.error('Positive finite spacing and at least one worker required')
    source = Path(args.source)
    source_summary = json.loads((source/'summary.json').read_text())
    if source_summary['status'] != 'complete':
        raise ValueError('Expected completed source preprocessing')
    charges = json.loads(Path(args.charges).read_text())
    water = args.extra_angular if args.extra_water_angular is None else args.extra_water_angular
    if min(args.extra_angular, water) < 0:
        p.error('Additional candidate counts must be nonnegative')
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    cases = sorted((r['case'] for r in source_summary['cases']), key=lambda c: (not c.startswith('H2O'), c))
    write_json(out/'configuration.json', vars(args))
    with ProcessPoolExecutor(max_workers=min(args.workers, len(cases))) as pool:
        results = list(pool.map(run_case, [(source/c, out, charges, args.spacing,
                                           water if c.startswith('H2O') else args.extra_angular,
                                           args.deposition,args.reference_candidates)
                                          for c in cases]))
    write_json(out/'summary.json', dict(status='complete', cases=results,
                                       accepted_states=sum(r['states'] for r in results)))


if __name__ == '__main__':
    main()
