"""Extract compact, path-free training examples from published preprocessing."""

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.fft import rfftn
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from xi_response import deposition_stencil, projected_target_blocks


def profiles(x, phases, spans):
    p = np.empty((len(x),24))
    for d in range(3):
        z=x[:,d,None]
        lo,hi,v1,v2=spans[d].T
        p[:,8*d:8*d+4]=np.sin(2*np.pi*z*np.arange(1,5)+phases[d])
        p[:,8*d+4:8*d+8]=np.where((z>lo)&(z<hi),v1+(v2-v1)*(z-lo)/(hi-lo),0)
    return p


def extract(entry, bulk_path, split, output):
    with np.load(entry['labels']) as data:
        shape=np.asarray(data['shape'])
        steps=data['source_steps']
        phases,spans=data['profile_phases'],data['profile_spans']
        layout=data['angular_spatial_profile']
        archived_probe=data['probe_energy']
        archived_weights=data['weights'][0]
        np.testing.assert_allclose(archived_weights, np.full(len(steps),1/len(steps)))
        archived_y1=data['c1_projected'][0]
        archived_y2=data['c2_projected'][0]
    edge=5
    start=shape//2-edge//2
    coord=np.stack(np.meshgrid(*(np.arange(v,v+edge) for v in start),indexing='ij'),-1).reshape(-1,3)
    voxel_ids=np.ravel_multi_index(coord.T,shape)
    lookup=np.full(int(np.prod(shape)),-1,np.int32)
    lookup[voxel_ids]=np.arange(len(coord))
    with h5py.File(entry['cache'],'r') as h:
        np.testing.assert_array_equal(steps,h['step'][:])
        cell=h['cell'][0]
        spacing=cell.diagonal()/shape
        dv=float(np.prod(spacing))
        elements=h['elements'][:]
        codes,counts=np.unique(elements,return_counts=True)
        moments=np.zeros((4,8))
        raw=np.zeros((len(steps),len(coord),133),np.float32)
        marks=h['atomic_xi_nlm']
        for t in range(len(steps)):
            m=marks[t].reshape(-1,8,16)
            np.add.at(moments,elements,m[:,:,0])
            bins,mass=deposition_stencil(h['positions'][t],cell,shape,'smoothstep')
            grid=lookup[bins]
            atom,slot=np.nonzero((grid>=0)&(mass>0))
            row=grid[atom,slot]
            value=mass[atom,slot]/dv
            np.add.at(raw[t,:,0],row,value)
            np.add.at(raw[t,:,1:129],row,value[:,None]*m[atom].reshape(-1,128))
            np.add.at(raw[t],(row,129+elements[atom]),value)
    np.testing.assert_allclose(raw[:,:,0],raw[:,:,129:].sum(-1),atol=4e-6)
    p=profiles((coord+.5)/shape,phases,spans)
    local_probes=np.concatenate((raw[:,:,0]@p,
                 np.einsum('tqnh,qh->tnh',raw[:,:,1:129].reshape(len(steps),edge**3,8,16),
                           p[:,layout],optimize=True).reshape(len(steps),128)),axis=-1)*dv
    local_probes=local_probes.astype(np.float64)
    potential=local_probes[:,0]*(.2/local_probes[:,0].std())
    weights=np.exp(potential.min()-potential)
    weights/=weights.sum()
    counts_matrix=csr_matrix(raw[:,:,129:].astype(np.float64).reshape(len(steps),-1).T*dv)
    y1,y2,dw=projected_target_blocks([counts_matrix],local_probes,weights,potential)
    centered=local_probes-weights@local_probes
    total_y1=potential@dw
    total_y2=-(centered*weights[:,None]).T@centered
    with np.load(bulk_path) as b:
        np.testing.assert_array_equal(b['source_steps'],steps)
        np.testing.assert_array_equal(b['element_codes'],codes)
        available=np.flatnonzero(b['reliable'].any(-1))
        ranks=np.linspace(0,len(available)-1,32).round().astype(int)
        chosen=available[np.argsort(np.linalg.norm(b['wavevector_inv_A'][available],axis=1))[ranks]]
        bulk={k:b[k][chosen] for k in ('mode_indices','eigenvectors','eigenvalues','reliable','wavevector_inv_A')}
        bulk['rho_element']=b['rho_element']
        bulk['inputs']=np.column_stack((np.ones(len(codes)),np.eye(4)[codes],moments[codes]/(len(steps)*counts[:,None])))
    output.mkdir(exist_ok=True)
    np.savez_compressed(output/'input.npz',xi_frames=raw,weights=weights,probe_energy=local_probes,
                        beta_external=potential,spacing_A=spacing,voxel_volume_A3=dv,
                        crop_shape=np.full(3,edge),source_steps=steps,source_grid_shape=shape,**{f'bulk_{k}':v for k,v in bulk.items()})
    np.savez_compressed(output/'reference.npz',c1_excess=y1,c2_excess=y2,
                        y1_intrinsic=total_y1,y2_intrinsic=total_y2)
    np.savez_compressed(output/'archived_full_box.npz',probe_energy=archived_probe,
                        weights=archived_weights,c1_excess=archived_y1,c2_excess=archived_y2)
    (output/'case.json').write_text(json.dumps(dict(schema='xi-preprocessed-v1',case=entry['case'],
        species=entry['species'],split=split,frames=len(steps),probes=152,source='xi_pnas_probes152/run_159379',
        bulk_source='xi_prl_bulk_coupling/bulk_elements/run_159428',bulk_modes=32,
        fields='conservative smoothstep deposition using saved atom-centered Xi, cropped for local testing'),
        indent=2)+'\n')
    print(entry['case'],output,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original_project',type=Path)
    args=parser.parse_args()
    source=args.original_project.resolve()
    manifest=json.loads((source/'experiments/xi_pnas_probes152/run_159379/training_split.json').read_text())['splits']
    bulk=json.loads((source/'experiments/xi_prl_bulk_coupling/bulk_elements/run_159428/bulk_labels_index.json').read_text())['splits']
    cases=(('train','CH4-T290.5-r00-rho0.6700','ch4'),('test','NH3-T618.2-r03-rho0.4533','nh3'))
    for split,case,key in cases:
        entry=next(row for row in manifest[split] if row['case']==case)
        extract(entry,bulk[split][case],split,ROOT/'cases'/key)


if __name__=='__main__':
    main()
