"""Nonuniform C1 EL fitting with a uniform, 3D bulk C2 regularizer."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch

from compare import SpectralCase
from fit_grid3d import model_state
from fit_regional import External,FULL,pooled_shell
from fit_short import prediction
from pilot import factors,pooled,requested_centers
from train_rows import GridFunctional,frame_weight_gradient


HERE=Path(__file__).resolve().parent
PURE=FULL/'real_grid/pooled_reference/pure_bulk'


def multiplicities(shape,length,width,kmax,nshell):
    """Exact count of reciprocal points of the full 3D periodic voxel grid."""
    axes=[2*np.pi*fn(n,d=L/n) for n,L,fn in zip(
          shape,length,(np.fft.fftfreq,np.fft.fftfreq,np.fft.rfftfreq))]
    mult=np.full(len(axes[-1]),2,dtype=int)
    mult[0]=1
    if shape[-1]%2==0:
        mult[-1]=1
    mag=(axes[0][:,None,None]**2+axes[1][None,:,None]**2+axes[2][None,None,:]**2)**.5
    indices=np.minimum((mag/width).astype(int),nshell-1)
    valid=(mag>0)&(mag<=kmax)
    return np.bincount(indices[valid],weights=np.broadcast_to(mult,mag.shape)[valid],minlength=nshell)


class Bulk:
    def __init__(self,row,geometry,device):
        self.case=SpectralCase(dict(row,cache=geometry['cache']),[],device,
                               spacing=None,frames_only=True).move(device)
        with np.load(row['archive']) as a,h5py.File(geometry['cache']) as h:
            np.testing.assert_array_equal(a['steps'],self.case.steps)
            shape=a['shape'];length=a['length'];n=a['densities']
            spectrum=a['physical_C2_shell'];valid=a['stable_shells']
            modes=a['mode_index'];self.live=a['mode_bin']
            radial_width=float(a['shell_width_inv_A']); cutoff=float(a['max_k_inv_A'])
            codes,e,natom=np.unique(h['elements'][:],return_inverse=True,return_counts=True)
            np.testing.assert_array_equal(codes,a['element_codes'])
            np.testing.assert_allclose(n,natom/np.prod(length))
            pos=h['positions'][:]
        self.grid_count=int(np.prod(shape))
        self.ne=len(n)
        self.mask=np.flatnonzero(valid)
        if self.mask.size<1:
            raise ValueError('No supported pure-phase band')
        self.v=torch.as_tensor(np.sqrt(n[:,None]*n[None,:]),device=device,dtype=torch.float32)
        self.reference=torch.as_tensor(spectrum[valid],dtype=torch.float32,device=device)
        k_count=multiplicities(shape,length,radial_width,cutoff,len(valid))
        self.count=torch.as_tensor(k_count[valid],dtype=torch.float32,device=device)
        self.w=torch.full((len(self.case.steps),),1/len(self.case.steps),dtype=torch.float32,device=device)
        phase,power=factors(pos,length,shape,e,natom,modes)
        # Independently check the saved bulk reference in the same native modes
        # before any prediction reads it. There is only one physical bulk state.
        _,actual,_=pooled(phase,power,self.live,natom,np.full(len(pos),1/len(pos)))
        np.testing.assert_allclose(actual[valid]/np.sqrt(n[:,None]*n[None,:]),spectrum[valid],
                                   atol=1e-4,rtol=3e-4)
        self.phase=torch.as_tensor(phase,device=device)
        self.device=device

    def loss(self,net,rng,training=True,draws=2,report_spectrum=False):
        # In a uniform bulk state all centers are equivalent by translation.
        # Sampling their actual positions multiplies real-space pair loss and
        # its normalization by the same count.
        selected=rng.choice(self.grid_count,requested_centers(self.grid_count),replace=False)
        if len(selected)==0:
            raise ValueError('No valid center on the 3D grid')
        g=frame_weight_gradient(net,self.case,self.w,create_graph=training)
        directions=[];corrections=[]
        for s in self.mask:
            p=torch.as_tensor(np.flatnonzero(self.live==s),device=self.device)
            d,g0=pooled_shell(self.w,self.phase.index_select(1,p),g,rng,draws)
            directions.append(d)
            corrections.append(g0)
        coeff=torch.cat((self.w[:,None],*directions),1)
        _,full=prediction(net,self.case,model_state(self.case,coeff),training)
        preds=[]
        for j,g0 in enumerate(corrections):
            current=[]
            for r in range(draws):
                start=(j*draws+r)*self.ne
                sl=slice(start,start+self.ne)
                current.append((full[sl,sl]-g0)/self.v)
            preds.append(torch.stack(current))
        prefix=torch.stack(preds,dim=1)
        err=prefix-self.reference
        squared=err[0]*err[1] if training else err.mean(0).square()
        scale=len(selected)*self.count[:,None,None]
        loss=(scale*squared).sum()/(scale*self.reference.square()).sum().clamp_min(1e-12)
        info=dict(centers=len(selected),retained_shells=len(self.mask),
                  represented_rows=len(selected)*self.ne,
                  mc_delta=float((scale*(err[0]-err[1]).detach().square()).sum().sqrt()/
                                 (scale*self.reference.square()).sum().sqrt().clamp_min(1e-12)))
        if report_spectrum:
            info['predicted']=prefix.detach().mean(0)
            info['reference']=self.reference
            info['mode_counts']=self.count
        return loss,info


def step(net,external,bulk,parameters,opt,weight,indices,rng,primary):
    """Apply the derivatives of both losses in a single optimizer update."""
    weight_c1,weight_c2=(1.,weight) if primary=='c1' else (weight,1.)
    g1=[None]*len(parameters)
    values=[]
    for i in indices:
        first,mu=external.at(net,int(i))
        partial=torch.autograd.grad(weight_c1*first,parameters,allow_unused=True)
        for k,v in enumerate(partial):
            if v is not None:
                contribution=v.detach()/len(indices)
                g1[k]=contribution if g1[k] is None else g1[k]+contribution
        values.append(float(first.detach()))
        del first,partial
    second,c2info=bulk.loss(net,rng)
    g2=torch.autograd.grad(weight_c2*second,parameters,allow_unused=True)
    def norm(seq):
        return torch.stack([x.detach().square().sum() for x in seq if x is not None]).sum().sqrt()
    ratio=float(norm(g2)/norm(g1).clamp_min(1e-20))
    opt.zero_grad(set_to_none=True)
    for a,b,p in zip(g1,g2,parameters):
        p.grad=a if b is None else (b if a is None else a+b)
    value=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
    opt.step()
    return dict(c1_loss=float(np.mean(values)),bulk_c2_loss=float(second.detach()),primary=primary,
        weight_c1=weight_c1,weight_c2=weight_c2,bulk_to_inhom_gradient_ratio=ratio,
        gradient_norm=float(value),
        c1_states=len(indices),**c2info)


def main(out,batch,weight,primary,full_epoch):
    device='cuda'
    free,total=torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(20*2**30,free-2*2**30)/total)
    files=[PURE/f'bulk_161755_{i}'/'manifest.json' for i in range(2)]
    if not full_epoch:
        files=[PURE/'bulk_161753/manifest.json']
    pure=[r for path in files for r in json.loads(path.read_text())['results']
          if r['centers']>0]
    geometry=json.loads((FULL/'geometry_71_manifest.json').read_text())
    source=next(x for x in pure if x['case']=='CH4-T290.5-r00-rho0.6700')
    assert source['split']=='train'
    geo=next(x for x in geometry['splits']['train'] if x['case']==source['case'])
    ext=json.loads((FULL/'external_c1_calibration/external_c1_train_manifest.json').read_text())
    real=next(x for x in ext['training_external_c1'] if x['case']=='CH4_r05_V2gauss')
    assert real['purpose']=='auxiliary_c1'
    out.mkdir()
    torch.manual_seed(617)
    scale=HERE.parents[3]/'xi_only_coupling/run_161288/scale.npy'
    net=GridFunctional(torch.from_numpy(np.load(scale)),2).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.002)
    params=tuple(net.parameters())
    rng=np.random.default_rng(617)
    rows=[]
    if full_epoch:
        c1_train=ext['training_external_c1']
        c2_train=[v for v in pure if v['split']=='train']
        geo_by_case={v['case']:v for v in geometry['splits']['train']}
        assert len(c1_train)==7 and len(c2_train)>=20
        subsets=np.array_split(rng.permutation(len(c2_train)),len(c1_train))
        for i,number in enumerate(rng.permutation(len(c1_train))):
            e=c1_train[number]
            external=External(e,device)
            order=rng.permutation(len(external.weights))
            blocks=[order[k:k+batch] for k in range(0,len(order),batch)]
            for j,ids in zip(subsets[i],np.array_split(np.arange(len(blocks)),len(subsets[i]))):
                source=c2_train[j]
                bulk=Bulk(source,geo_by_case[source['case']],device)
                for block in ids:
                    result=step(net,external,bulk,params,opt,weight,blocks[block],rng,primary)
                    record=dict(c1_case=e['case'],bulk_case=source['case'],**result)
                    rows.append(record)
                    print(json.dumps(record,allow_nan=False),flush=True)
                del bulk
            del external
        assert sum(r['c1_states'] for r in rows)==sum(e['states'] for e in c1_train)
        assert set(v['case'] for v in c2_train)=={r['bulk_case'] for r in rows}
    else:
        external=External(real,device)
        bulk=Bulk(source,geo,device)
        order=rng.choice(len(external.weights),2*batch,replace=False)
        for indices in np.split(order,2):
            record=step(net,external,bulk,params,opt,weight,indices,rng,primary)
            rows.append(record)
            print(json.dumps(record,allow_nan=False),flush=True)
    torch.save(net.state_dict(),out/('epoch1.pt' if full_epoch else 'smoke.pt'))
    (out/'summary.json').write_text(json.dumps(dict(full_epoch=full_epoch,primary=primary,
        parameters=sum(v.numel() for v in params),batch=batch,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,updates=rows),
        indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--lambda-aux',type=float,default=.01)
    p.add_argument('--primary',choices=('c1','c2'),default='c2')
    p.add_argument('--full-epoch',action='store_true')
    a=p.parse_args()
    if not 1<=a.batch<=100 or not np.isfinite(a.lambda_aux) or a.lambda_aux<=0:
        p.error('Expected a valid batch and positive auxiliary weight')
    main(a.out,a.batch,a.lambda_aux,a.primary,a.full_epoch)
