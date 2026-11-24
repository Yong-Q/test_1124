"""Verify production labels and three model families on low-density Xi tangents."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'backend')]
from atomic_backgrounds import GAS_CONSTANT, atomic_potential
from demo import Fields, derivative_check
from experiments.xi_all_projection_loss.train import case_labels
from experiments.xi_total_supervised_50.train import TotalCase
from models.cnn3d.model import components as cnn
from models.neighborhood_mlp.model import components as mlp
from models.stable_kernel.model import make_model
from portable_io import ResponseWeights
from production import dataset_inputs
from xi_response import deposition_stencil, state_weights


def reconstruct_fields(archive, shape, dv):
    edge = 5
    positions, cell, elements = archive['positions'][:], archive['cell'][0], archive['elements'][:]
    mass_by_voxel = np.zeros(np.prod(shape))
    for xyz in positions:
        bins, mass = deposition_stencil(xyz, cell, shape, 'smoothstep')
        np.add.at(mass_by_voxel, bins, mass)
    center = np.unravel_index(mass_by_voxel.argmax(), shape)
    axes = [np.arange(q-edge//2, q-edge//2+edge) % n for q, n in zip(center, shape)]
    coordinates = np.stack(np.meshgrid(*axes,indexing='ij'), -1).reshape(-1,3)
    voxels = np.ravel_multi_index(coordinates.T, shape)
    lookup = np.full(int(np.prod(shape)), -1, np.int32)
    lookup[voxels] = np.arange(len(voxels))
    fields = np.zeros((len(positions), edge**3, 133), np.float32)
    marks = archive['atomic_xi_nlm']
    for t, xyz in enumerate(positions):
        bins, mass = deposition_stencil(xyz, cell, shape, 'smoothstep')
        atoms, corners = np.nonzero((lookup[bins] >= 0) & (mass > 0))
        if not len(atoms):
            continue
        rows, reverse = np.unique(atoms, return_inverse=True)
        mark = marks[t, rows].reshape(len(rows), 128)[reverse]
        grid = lookup[bins[atoms, corners]]
        values = mass[atoms, corners] / dv
        np.add.at(fields[t, :, 0], grid, values)
        np.add.at(fields[t, :, 1:129], grid, values[:,None] * mark)
        np.add.at(fields[t], (grid, 129 + elements[atoms]), values)
    np.testing.assert_allclose(fields[...,0], fields[...,129:].sum(-1), atol=4e-6)
    return dict(xi_frames=fields, crop_shape=np.full(3, edge), voxel_volume_A3=dv,
                spacing_A=cell.diagonal()/shape), positions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('generated', type=Path)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    args=parser.parse_args()
    directory=args.generated.resolve()
    record=json.loads((directory/'summary.json').read_text())
    specification=json.loads(Path(__file__).with_name('case.json').read_text())
    for key in ('case','frames','atoms','probes','grid','seed'):
        if record[key]!=specification[key]:
            raise ValueError(f'The generated result does not match the pinned case field {key}')
    settings=json.loads((directory/'background_fields.json').read_text())
    manifest,_=dataset_inputs(directory/'training_split.json')
    case,=[row for row in manifest['splits']['train'] if row['case']==record['case']]
    torch.set_num_threads(2)
    with np.load(directory/'xi_response_labels.npz') as shard:
        energy,probes,weights=[shard[key] for key in ('background_energy','probe_energy','weights')]
        shape=tuple(int(n) for n in shard['shape'])
        dv=float(shard['voxel_volume'])
        for key, expected in specification['input_array_sha256'].items():
            found=hashlib.sha256(np.ascontiguousarray(shard[key]).tobytes()).hexdigest()
            if found!=expected:
                raise ValueError(f'Preprocessed input fingerprint differs for {key}')
        regenerated,quality=state_weights(energy)
        np.testing.assert_allclose(weights, regenerated, atol=0., rtol=1e-12)
        assert np.all(quality['accepted'])
        assert shard['c1_projected'].shape==(record['states'],152)
        assert shard['c2_projected'].shape==(record['states'],152,152)
        with h5py.File(case['cache'],'r') as cache:
            np.testing.assert_array_equal(shard['source_steps'], cache['step'][:])
            local_fields,positions=reconstruct_fields(cache,shape,dv)
        physical=[]
        lengths=np.asarray(settings['box_A'])
        thermal=GAS_CONSTANT*settings['temperature_K']
        for row in sorted({0,record['states']-1}):
            field={key:np.asarray(value) for key,value in settings['fields'][row].items()}
            sampled=atomic_potential(positions,lengths,field).sum(axis=1)/thermal
            expected=(sampled-sampled.mean())*field['scale']
            np.testing.assert_allclose(expected,energy[:,row],atol=2e-10)
            probe_xyz=positions[0,:64]
            np.testing.assert_allclose(atomic_potential(probe_xyz,lengths,field),
                                       atomic_potential(probe_xyz+lengths,lengths,field),atol=2e-10)
            physical.append(dict(row=row,energy_max_abs=float(np.abs(expected-energy[:,row]).max())))
        with case_labels(case) as labels:
            assert labels['weights'].shape==(case['states'],record['frames'])
            row=case['states']-1
            np.testing.assert_array_equal(labels['c1_projected'][row],shard['c1_projected'][-1])
            np.testing.assert_array_equal(labels['c2_projected'][row],shard['c2_projected'][-1])
            centered=probes-weights[-1]@probes
            directional=ResponseWeights(labels).select(np.array([row]),np.arange(152)[None])[0]
            np.testing.assert_allclose(directional,-weights[-1,:,None]*centered,atol=2e-11)
            total=TotalCase(labels)['c2_projected'][[row]][0]
            np.testing.assert_allclose(total,-(centered*weights[-1,:,None]).T@centered,atol=2e-7,rtol=1e-8)
    device=torch.device(args.device)
    torch.manual_seed(617)
    models={'fft-excess': mlp('fft',False)[0](), 'cnn-single-excess': cnn('single',False)[0](),
            'stable-bounded': make_model()}
    with torch.no_grad():
        models['stable-bounded'].environment_weights[0]=.001
    source=Fields(local_fields,case,device,False,False)
    coeff=source.tensor(np.column_stack((weights[-1],directional)))[None]
    checks={key:derivative_check(model.to(device),source,coeff,key=='stable-bounded')
            for key,model in models.items()}
    print(json.dumps(dict(status='passed',case=case['case'],device=args.device,
                          samples=record['states'],physical_energy=physical,manifest_readback=True,
                          model_validation_scope='two-direction derivatives on a 5^3 periodic crop',
                          model_checks=checks),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
