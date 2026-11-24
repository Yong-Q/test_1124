"""Per-atom 3D preprocessing, independently reweightable frame cache, no FFT."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import h5py
import numpy as np
from kernel import Basis, box_lengths, element_field, combine, grid, ELEMENTS


def write_json(path,data):
    with Path(path).open('x') as f:json.dump(data,f,indent=2,allow_nan=False)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def deposit_atomic(positions,marks,cell,shape):
    """Small/reference deposition. Production uses spatial chunks (ensemble_xi)."""
    lengths=box_lengths(cell);n=len(positions);dv=np.prod(lengths)/np.prod(shape)
    index=np.floor((positions%lengths)/lengths*np.array(shape)).astype(int)
    flat=np.ravel_multi_index(index.T,shape)
    out=np.zeros((int(np.prod(shape)),n,marks.shape[1]+1))
    out[flat,np.arange(n),0]=1/dv;out[flat,np.arange(n),1:]=marks/dv
    return out


def perturbations(energies,count,seed):
    """Dimensionless atom-additive 3D potentials; importance weights on frames."""
    if count<2 or len(energies)<20:raise ValueError('>=20 selected frames and >=2 states required')
    rng=np.random.default_rng(seed);f,k=energies.shape
    coefficients=rng.normal(size=(count,k));coefficients[0]=0
    centered=energies-energies.mean(0)
    spread=np.std(centered@coefficients.T,axis=0)
    strengths=np.resize(np.array([.1,.2,.35,.5,.75,1.,1.5,2.]),count)
    coefficients[1:]*=(strengths[1:]/np.maximum(spread[1:],1e-12))[:,None]
    logw=-(centered@coefficients.T).T;logw-=logw.max(1,keepdims=True)
    weights=np.exp(logw);weights/=weights.sum(1,keepdims=True)
    return dict(coefficients=coefficients,weights=weights,**weight_quality(weights))


def weight_quality(weights):
    f=weights.shape[1];ess=1/np.sum(weights**2,axis=1)
    groups=np.array_split(np.arange(f),min(20,f));blocks=np.stack([weights[:,g].sum(1) for g in groups],1)
    block_ess=1/np.sum(blocks**2,axis=1)
    accepted=(ess>=max(min(50,f),.3*f))&(block_ess>=min(10,len(groups)))
    return dict(ess=ess,block_ess=block_ess,accepted=accepted)


def external_basis(positions,cell,spec,basis):
    lengths=box_lengths(cell);columns=[]
    for center,n,h in zip(spec['centers_fractional'],spec['radial'],spec['harmonic']):
        v=positions-np.asarray(center)*lengths;v-=lengths*np.rint(v/lengths)
        columns.append(basis.values(v)[:,n*16+h])
    return np.stack(columns,1)


def prepare_case(task):
    case=task['case'];out=Path(task['output'])/case;out.mkdir(exist_ok=False)
    start=time.time();meta=json.load(open(task['metadata']));source=meta['source_h5']
    cfg=Basis(task['cutoff'],task['radials']);seed=int(hashlib.sha256(case.encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed)
    spec={'centers_fractional':rng.random((32,3)).tolist(),'radial':(np.arange(32)%cfg.radials).tolist(),'harmonic':(np.arange(32)%16).tolist(),
          'definition':'betaV(x)=sum_k coefficient[k]*regular_gaussian_realSH(x-center[k]); U=sum over ATOMS betaV(r_i)',
          'cutoff_A':cfg.cutoff,'radials':cfg.radials}
    charges=json.load(open(task['charges']))
    with h5py.File(source,'r') as src:
        nframe,natom,_=src['positions'].shape;indices=np.arange(1,nframe,task['stride'])
        if task.get('max_frames'):indices=indices[:task['max_frames']]
        if len(indices)<20:raise ValueError('Too few source frames for declared overlap checks')
        elements=src['elements'][:].astype(int);atom_ids=src['atom_ids'][:];steps=src['step'][indices]
        if len(np.unique(atom_ids))!=natom or len(np.unique(steps))!=len(steps):raise ValueError('Nonunique atom/frame identity')
        lengths=box_lengths(src['cell'][indices[0]])
        if 2*cfg.cutoff>=lengths.min():raise ValueError('Box too short for requested physical cutoff')
        complete=all(charges.get(ELEMENTS[e]) is not None for e in np.unique(elements))
        path=out/'atomic_frames.pending.h5';energies=np.zeros((len(indices),32));maxmark=0.
        with h5py.File(path,'x') as dst:
            dst.attrs['schema']='xi_atomic_3d_v1';dst.attrs['status']='writing'
            dst.attrs['meaning']='i=actual source atom ID; neighbor element axis retained to change Zi without geometry recomputation'
            dst.create_dataset('atom_ids',data=atom_ids);dst.create_dataset('elements',data=elements)
            dst.create_dataset('source_indices',data=indices);dst.create_dataset('step',data=steps)
            dst.create_dataset('positions',shape=(len(indices),natom,3),dtype='f8')
            dst.create_dataset('cell',shape=(len(indices),3,3),dtype='f8')
            ds=dst.create_dataset('neighbor_element_nlm',shape=(len(indices),natom,4,cfg.radials,16),dtype='f4',chunks=(1,min(64,natom),4,cfg.radials,16),compression='lzf')
            combined=dst.create_dataset('atomic_xi_nlm',shape=(len(indices),natom,cfg.radials,16),dtype='f4',chunks=(1,min(64,natom),cfg.radials,16),compression='lzf') if complete else None
            zero=cfg.values(np.zeros((1,3)))[0]
            for row,index in enumerate(indices):
                p=src['positions'][index];cell=src['cell'][index]
                if not np.allclose(cell,np.diag(lengths),rtol=1e-8,atol=1e-8):raise ValueError('Changing volume needs its own ensemble/reference treatment')
                field=element_field(p,p,elements,cell,cfg,chunk=16)
                field[np.arange(natom),elements]-=zero # exclude only the center atom itself
                if not np.isfinite(field).all():raise ValueError('Nonfinite 3D atomic descriptor')
                ds[row]=field.reshape(natom,4,cfg.radials,16)
                if combined is not None:combined[row]=combine(field,charges).reshape(natom,cfg.radials,16)
                dst['positions'][row]=p;dst['cell'][row]=cell
                energies[row]=external_basis(p,cell,spec,cfg).sum(0)
                maxmark=max(maxmark,float(np.max(np.abs(field))))
                if row%10==0:
                    dst.flush();print(json.dumps({'case':case,'frame':row+1,'frames':len(indices),'elapsed_s':time.time()-start}),flush=True)
            dst.create_dataset('external_basis_energy',data=energies)
            dst.attrs['status']='complete';dst.attrs['charges_complete']=complete
        with h5py.File(path,'r') as check:
            if check['step'].shape!=(len(indices),) or check.attrs['status']!='complete':raise ValueError('Published frame cache verification failed')
        os.replace(path,out/'atomic_frames.h5')
    states=perturbations(energies,task['perturbations'],seed+1)
    np.savez_compressed(out/'three_dimensional_perturbations.npz',**states,source_steps=steps)
    write_json(out/'external_basis.json',spec)
    # Preserve reusable original-state frame weights, explicitly without copying projected c1/c2 labels.
    legacy=[];weights=[];lookup={int(s):j for j,s in enumerate(steps)}
    for entry in meta.get('atomic_labels',[]):
        file=entry.get('original_weight_file');dataset=entry.get('weights_dataset')
        if not file or not dataset:continue
        with h5py.File(file,'r') as h:
            available=h['source_step'][:];w=h[dataset][:]
        bystep={int(s):j for j,s in enumerate(available)}
        if any(int(s) not in bystep for s in steps):raise ValueError('Original state weights missing selected source steps')
        selected=w[np.array([bystep[int(s)] for s in steps])]
        if selected.ndim!=1 or not np.isfinite(selected).all() or np.any(selected<0) or selected.sum()<=0:raise ValueError('Invalid original frame weights')
        selected=selected/selected.sum();weights.append(selected);legacy.append({'id':entry['id'],'weight_file':file,'dataset':dataset,'parameters':entry['parameters']})
    if weights:
        a=np.array(weights);np.savez_compressed(out/'original_atomic_state_weights.npz',weights=a,source_steps=steps,**weight_quality(a))
    write_json(out/'original_atomic_states.json',legacy)
    report={'case':case,'status':'complete','source_h5':source,'source_metadata':task['metadata'],'source_metadata_sha256':digest(task['metadata']),
            'expected_source_h5_sha256':meta.get('source_h5_sha256'),'input_identity_check':'metadata hash and dataset/step/atom identities checked; full large source H5 hash not re-read',
            'source_frames':nframe,'selected_frames':len(indices),'frame_stride':task['stride'],'atoms':natom,'cutoff_A':cfg.cutoff,'radials':cfg.radials,'lmax':3,
            'preserved_axes':['frame','atom_i','neighbor_element','n','l,m'],'charge_weights':charges,'combined_xi_available':complete,
            'perturbation_candidates_including_unperturbed':task['perturbations'],'perturbations_accepted':int(states['accepted'].sum()),
            'original_atomic_states_preserved':len(legacy),'elapsed_s':time.time()-start,'max_abs_mark':maxmark,
            'limitations':['new perturbations are correlated reweighted samples, not independent MD','no new 3D c1/c2 labels claimed','frame/block ESS does not prove equilibration','finite cutoff needs target-specific convergence tests','coarse/strided postprocessing is exploratory'],
            'frame_cache_sha256':digest(out/'atomic_frames.h5')}
    write_json(out/'summary.json',report)
    return report


def ensemble_xi(cache,weights,charges,shape,start=0,stop=None):
    """Spatial block of xi[x, atomic i, 1+n*16], exact voxel number conservation.

    No dense global x*i tensor is stored. Whole-multiplet standardization must
    be fitted on training sources only before use in a learned model.
    """
    with h5py.File(cache,'r') as h:
        f,n,_,r,_=h['neighbor_element_nlm'].shape
        w=np.asarray(weights,float)
        if w.shape!=(f,) or np.any(w<0) or not np.isfinite(w).all() or not np.isclose(w.sum(),1.):raise ValueError('Aligned normalized frame weights required')
        total=int(np.prod(shape));stop=total if stop is None else stop
        if not 0<=start<stop<=total:raise ValueError('Invalid spatial block')
        out=np.zeros((stop-start,n,1+16*r));e=h['elements'][:]
        for t in range(f):
            cell=h['cell'][t];lengths=box_lengths(cell);dv=np.prod(lengths)/total
            index=np.floor((h['positions'][t]%lengths)/lengths*np.array(shape)).astype(int)
            bins=np.ravel_multi_index(index.T,shape);keep=np.flatnonzero((bins>=start)&(bins<stop))
            if not len(keep):continue
            field=h['neighbor_element_nlm'][t,keep].reshape(len(keep),4,-1)
            marks=combine(field,charges)
            out[bins[keep]-start,keep,0]+=w[t]/dv
            out[bins[keep]-start,keep,1:]+=w[t]*marks/dv
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--charges',required=True);p.add_argument('--workers',type=int,default=48)
    p.add_argument('--stride',type=int,default=5);p.add_argument('--perturbations',type=int,default=257)
    p.add_argument('--radials',type=int,default=8);p.add_argument('--cutoff',type=float,default=5.)
    p.add_argument('--max-frames',type=int);p.add_argument('--case-prefix',default='')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    manifest=json.load(open(args.manifest));entries=[x for x in manifest['cases'] if x['case'].startswith(args.case_prefix)]
    if not entries:raise ValueError('No eligible cases')
    write_json(out/'configuration.json',vars(args))
    tasks=[dict(vars(args),case=x['case'],metadata=x['metadata']) for x in entries]
    results=[];errors=[]
    workers=min(args.workers,int(os.environ.get('SLURM_CPUS_PER_TASK',args.workers)),len(tasks))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(prepare_case,t):t['case'] for t in tasks}
        for future in as_completed(futures):
            try:results.append(future.result())
            except Exception:
                failure={'case':futures[future],'traceback':traceback.format_exc()};errors.append(failure)
                write_json(out/f"failure_{futures[future]}.json",failure);print(json.dumps(failure),flush=True)
    write_json(out/'summary.json',{'status':'failed' if errors else 'complete','completed_cases':len(results),'failed_cases':len(errors),'cases':results,'failures':errors})
    if errors:raise RuntimeError(f'{len(errors)} cases failed; preserved failure evidence')


if __name__=='__main__':main()
