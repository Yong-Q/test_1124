"""Validate compact preprocessed inputs and generate new reweighted states."""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from xi_response import projected_target_blocks

ROOT = Path(__file__).resolve().parent
PROBES = 152
LABEL_KEYS = ('c1_excess','c2_excess','y1_intrinsic','y2_intrinsic')


def load(case, states=None, row=0):
    path=ROOT/'cases'/case if case in ('ch4','nh3') else Path(case)
    info=json.loads((path/'case.json').read_text())
    with np.load(path/'input.npz') as saved:
        data={k:saved[k] for k in saved.files}
    with np.load(states or path/'reference.npz') as saved:
        try:
            ref={k:saved[k] if states is None else saved[k][row] for k in LABEL_KEYS}
            if states is not None:
                data['weights']=saved['weights'][row]
                data['beta_external']=saved['beta_external'][row]
        except IndexError as exc:
            raise ValueError('State row is outside the prepared file') from exc
    frames, cells, channels=data['xi_frames'].shape
    if (channels!=133 or cells!=int(np.prod(data['crop_shape']))
            or data['weights'].shape!=(frames,) or data['probe_energy'].shape!=(frames,PROBES)
            or data['beta_external'].shape!=(frames,) or info['schema']!='xi-preprocessed-v1'):
        raise ValueError('Incorrect prepared input shape or schema')
    if frames!=info['frames'] or np.any(data['weights']<=0) or not np.isclose(sum(data['weights']),1):
        raise ValueError('Frame count or probability normalization is inconsistent')
    if not all(np.isfinite(x).all() for x in (*ref.values(),data['xi_frames'],data['probe_energy'])):
        raise ValueError('Inputs and labels must be finite')
    np.testing.assert_allclose(data['xi_frames'][...,0],data['xi_frames'][...,129:].sum(-1),atol=4e-6)
    for name, array in ref.items():
        if array.shape!=((PROBES,PROBES) if '2' in name else (PROBES,)):
            raise ValueError(f'Incorrect label shape: {name}')
    return data,ref,info


def occupation(data):
    values=data['xi_frames'][...,129:].astype(np.float64)
    return csr_matrix(values.reshape(len(values),-1).T*float(data['voxel_volume_A3']))


def references(data, weight, energy, counts=None):
    counts=occupation(data) if counts is None else counts
    probe=data['probe_energy']
    c1,c2,dw=projected_target_blocks([counts],probe,weight,energy)
    x=probe-weight@probe
    return (c1,c2,energy@dw,-(x*weight[:,None]).T@x),dw


def verify(case, states=None, row=0):
    data, expected, meta=load(case,states,row)
    found,dw=references(data,data['weights'],data['beta_external'])
    errors={}
    for key,value in zip(LABEL_KEYS,found):
        np.testing.assert_allclose(value,expected[key],atol=2e-9,rtol=3e-9)
        errors[key]=float(np.max(np.abs(value-expected[key])))
    np.testing.assert_allclose(dw.sum(0),0,atol=1e-11)
    return dict(case=meta['case'],split=meta['split'],frames=meta['frames'],probes=PROBES,
                max_reference_error=errors, status='passed')


def perturb(case, count, output, seed=617):
    data,_,info=load(case)
    if info['split']!='train':
        raise ValueError('Perturbation fixture must belong to the training split')
    if count<1:
        raise ValueError('Expected a positive requested count')
    rng=np.random.default_rng(seed)
    probe=data['probe_energy']
    occupation_matrix=occupation(data)
    rows={k:[] for k in ('coefficients','weights','beta_external','ess','block_ess',*LABEL_KEYS)}
    attempts=0
    while len(rows['weights'])<count and attempts<30*count:
        index=attempts
        attempts+=1
        coefficient=rng.normal(size=PROBES)
        if index%3==0:
            coefficient[24:]=0
        elif index%3==1:
            coefficient[:24]=0
        energy=probe@coefficient
        if energy.std()<1e-12:
            continue
        coefficient*=rng.uniform(.1,.8)/energy.std()
        energy=probe@coefficient
        w=np.exp(energy.min()-energy)
        w/=w.sum()
        ess=1/np.sum(w*w)
        block_ess=1/np.sum(np.asarray([b.sum() for b in np.array_split(w,20)])**2)
        if ess<max(min(50,len(w)),.30*len(w)) or block_ess<10:
            continue
        labels,_=references(data,w,energy,occupation_matrix)
        for key,val in zip(rows,(coefficient,w,energy,ess,block_ess,*labels)):
            rows[key].append(val)
    if len(rows['weights'])!=count:
        raise RuntimeError(f'Generated only {len(rows["weights"])} accepted states')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream,**rows)
    return dict(status='passed',case=info['case'], states=count,attempts=attempts,
                min_ess=min(rows['ess']),min_block_ess=min(rows['block_ess']),
                output=str(output),first=verify(case,output),last=verify(case,output,count-1))
