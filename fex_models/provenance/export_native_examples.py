"""Freeze small, exact contractions from the native preprocessed feature banks."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fit_grid3d import model_state
from fit_short import NearCase, load_state, prediction
from fuse import Bulk, FULL, PURE, pooled_shell
from references import entries
from short_kernel import ShortSpatial
from train import MANIFEST
from train_rows import GridFunctional, frame_weight_gradient
from voxel_loss import Voxels, voxel_gradient


CASES = ('CH4-T290.5-r00-rho0.6700','C2H6-T465.4-r00-rho0.6700')
KEYS = ('local','radial','angular','short')
DRAWS = 2


def export_row(data,k,x):
    if isinstance(x,torch.Tensor):
        x=x.detach().cpu().numpy()
    data[k]=np.asarray(x)


def fields(case,data,name):
    for k in ('wave','angular_wave','basis','short_kernels','mode_weight','volume'):
        export_row(data,f'{name}_{k}',getattr(case,k))


def pack(data,prefix,state):
    x,d,r,dt,an,s=state
    for key,base,tangent in zip(KEYS,(x,r,an[0],s[0]),(d.transpose(0,1),dt,an[1],s[1])):
        export_row(data,f'{prefix}_{key}',torch.cat((base.unsqueeze(0),tangent)))


def bulk_lift(w,phase):
    q=np.sqrt(2)*torch.cat((phase.real,phase.imag),dim=1)
    q=q-torch.einsum('t,tpa->pa',w,q)
    scale=q.shape[1]
    cov=torch.einsum('t,tpa,tpb->ab',w,q,q)/scale
    dual=q@torch.linalg.inv(cov)
    gram=torch.einsum('tpa,tpb->tab',dual,dual)/scale
    third=torch.einsum('t,tpa,tpb,tpc->cab',w,dual,dual,q)/scale
    response=gram-torch.einsum('t,tab->ab',w,gram)
    response-=torch.einsum('ta,abc->tbc',dual.mean(1),third)
    return w[:,None,None]*response


def c1_lift(q,m,w):
    root=m.sqrt()
    x=(q-m)/root
    p=x.T*w.sqrt()
    cov=p@p.T
    ridge=.015*(cov.diagonal().sum()/64).clamp_min(.05)
    cov+=ridge*torch.eye(len(m),device=q.device)
    b=(x-w@x).T*w
    return (torch.linalg.solve(cov,b)/root[:,None]).T.contiguous()


def run(a):
    a.out.mkdir(parents=True,exist_ok=True)
    src=Path(__file__).resolve().parents[3]
    research=src/'xi_sognet_atomic/xi_readout_1000'
    g=json.loads((FULL/'geometry_71_manifest.json').read_text())['splits']
    geom={r['case']:r for group in g.values() for r in group}
    refs=json.loads((a.references/'manifest.json').read_text())
    rby={r['case']:r for r in refs['cases']}
    cat={r['case']:r for r in entries()}
    c2={r['case']:r for f in PURE.glob('bulk_161755_*/manifest.json')
        for r in json.loads(f.read_text())['results'] if r['centers']}
    old={r['case']:r for group in json.loads(MANIFEST.read_text())['splits'].values() for r in group}
    scopes=json.loads((research/'short_spatial/run_161416/selection.json').read_text())
    scale=torch.from_numpy(np.load(research/'xi_only_coupling/run_161288/scale.npy'))
    torch.manual_seed(617)
    ppt=ShortSpatial(scale.clone(),2).cuda().eval().requires_grad_(False)
    torch.manual_seed(617)
    recent=GridFunctional(scale.clone(),2).cuda().eval().requires_grad_(False)
    rng=np.random.default_rng(617)

    for i,case in enumerate(CASES):
        print(json.dumps(dict(stage='export_start',case=case)),flush=True)
        split=('train','test')[i]
        role=('train','holdout')[i]
        data={'feature_scale':scale.numpy(),'case_id':case,'role':role,'format_version':1}
        legacy_rows=scopes[split][case][:1]
        legacy=NearCase(old[case],legacy_rows,'cuda',96).move('cuda')
        x,d,r,rt,ang,near=load_state(legacy,0)
        select=np.array([0,12,24,25,28,33])
        dsmall=(x,d[:,select],r,rt[select],(ang[0],ang[1][select]),(near[0],near[1][select]))
        pack(data,'ppt',dsmall)
        fields(legacy,data,'ppt')
        export_row(data,'ppt_y1',legacy.t1[0][select])
        export_row(data,'ppt_y2',legacy.t2[0][select][:,select])
        xp,yp=prediction(ppt,legacy,dsmall,False)
        export_row(data,'ppt_native_c1',xp)
        export_row(data,'ppt_native_c2',yp)
        export_row(data,'ppt_probe_indices',select)
        export_row(data,'ppt_source_step_count',len(legacy.observable))
        export_row(data,'ppt_state_id',legacy_rows[0])
        del legacy,x,d,r,rt,ang,near,dsmall
        torch.cuda.empty_cache()

        grid=Voxels(cat[case],rby[case],geom[case]['cache'],'cuda')
        sid=1
        good=grid.valid[0]&grid.valid[sid]
        cells=np.flatnonzero(good.any(1))
        cells=np.sort(rng.choice(cells,6,replace=False))
        cols=(cells[:,None]*grid.ne+np.arange(grid.ne))[good[cells]]
        index=torch.as_tensor(cols,device='cuda')
        q=grid.counts.index_select(1,index)
        fields(grid.case,data,'grid')
        with np.load(rby[case]['path']) as f:
            export_row(data,'grid_flat_voxel',f['voxel_indices'][cols//grid.ne])
            export_row(data,'grid_xyz_A',f['voxel_xyz'][cols//grid.ne])
            export_row(data,'grid_element',f['element_codes'][cols%grid.ne])
            export_row(data,'grid_frames',len(f['steps']))
        for tag,stateid in (('base',0),('weak',sid)):
            prefix=f'grid_{tag}'
            w=grid.weights[stateid]
            m=w@q
            tangent=c1_lift(q,m,w)
            state=model_state(grid.case,torch.cat((w[:,None],tangent),1))
            pack(data,prefix,state)
            export_row(data,prefix+'_label',grid.ref[stateid,cols])
            export_row(data,prefix+'_native_c1',voxel_gradient(q,m,w,frame_weight_gradient(
                recent,grid.case,w,False)))
            export_row(data,prefix+'_state_id',stateid)
        del grid
        torch.cuda.empty_cache()

        bulk=Bulk(c2[case],geom[case],'cuda')
        gframe=frame_weight_gradient(recent,bulk.case,bulk.w,False)
        directions,corrections,check=[],[],[]
        keep=bulk.mask[:2]
        for s in keep:
            x=bulk.phase.index_select(1,torch.as_tensor(np.flatnonzero(bulk.live==s),device='cuda'))
            d,g0=pooled_shell(bulk.w,x,gframe,rng,DRAWS)
            w64,x64,g64=bulk.w.double(),x.to(torch.complex128),gframe.double()
            b=bulk_lift(w64,x64)
            _,exact=pooled_shell(w64,x64,g64,np.random.default_rng(0),DRAWS)
            torch.testing.assert_close(torch.einsum('t,tab->ab',g64,b),exact,atol=2.e-8,rtol=2.e-8)
            directions.append(d)
            corrections.append(b.flatten(1).to(bulk.w))
            check.append(exact.to(g0))
            print(json.dumps(dict(stage='corrected_bulk_lift',case=case,shell=int(s),
                                  raw_float32_delta=float((exact-g0).abs().max()))),flush=True)
        w=bulk.w[:,None]
        pack(data,'bulk',model_state(bulk.case,torch.cat((w,*directions,*corrections),1)))
        state=model_state(bulk.case,torch.cat((w,*directions),1))
        _,pred=prediction(recent,bulk.case,state,False)
        vals=[]
        for j,g0 in enumerate(check):
            vals.append(torch.stack([(pred[t:t+bulk.ne,t:t+bulk.ne]-g0)/bulk.v
                          for t in range(j*DRAWS*bulk.ne,(j+1)*DRAWS*bulk.ne,bulk.ne)]))
        for k,v in (('bulk_native_c2',torch.stack(vals)),
                    ('bulk_reference',bulk.reference[:len(keep)]),
                    ('bulk_mode_counts',bulk.count[:len(keep)]),
                    ('bulk_density_scale',bulk.v),('bulk_shell_indices',keep),
                    ('bulk_draws',DRAWS)):
            export_row(data,k,v)
        del bulk
        file=a.out/(case.split('-')[0].lower()+'_prepared.npz')
        with file.with_suffix('.partial').open('xb') as f:
            np.savez_compressed(f,**data)
        file.with_suffix('.partial').rename(file)
        print(json.dumps(dict(stage='saved',case=case,name=file.name,bytes=file.stat().st_size)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--references',type=Path,required=True)
    run(p.parse_args())
