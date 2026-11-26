"""Matched multi-epoch comparison of MD training with and without gated peer guidance."""
import argparse
from copy import deepcopy
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from fuse import FULL
from references import entries
from run_lines import WARMUP_CASE, bulk_rows, emit
from train_mutual import check_peer_gradient, fit, holdout, initialized, role_manifest


BULK_CASES = ('CH4-T290.5-r05-rho0.3089','H2O-T986.4-r05-rho0.3089',
              'CO2-T463.6-r05-rho0.3089','C2H4-T430.4-r01-rho0.5978')
EXTERNAL_CASES = ('CH4_r05_V2gauss','H2O_r05_field','CO2_r05_V1well','C2H4_r05_V2gauss')


def add_gradient(model,loss,limit,protect=False,retain_graph=False):
    params=tuple(model.parameters())
    main=[p.grad for p in params]
    extra=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=retain_graph)
    device=params[0].device
    def flatten(parts):
        return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                          for p,g in zip(params,parts)])
    g=flatten(main)
    h=flatten(extra)
    dot=torch.dot(g,h)
    cos=dot/(g.norm()*h.norm()).clamp_min(1.e-20)
    if protect:
        h=h-(dot/g.square().sum().clamp_min(1.e-20)).clamp_max(0.)*g
    w=(limit*g.norm()/h.norm().clamp_min(1.e-20)).clamp_max(1.)
    for p,old,add,part in zip(params,main,extra,h.split([p.numel() for p in params])):
        if old is not None or add is not None:
            p.grad=(w*part).view_as(p) if old is None else old+(w*part).view_as(p)
    assert g.device==device
    return float(cos),float(w)


def teaching(first,second,a,b,target,enabled):
    signal=target.square().mean().clamp_min(1.e-4)
    err1=(a-target).square().mean()/signal
    err2=(b-target).square().mean()/signal
    e1,e2=float(err1.detach()),float(err2.detach())
    # A peer must improve on both the other prediction and zero response.
    # Holdout labels never enter these example weights.
    compare=max(e1+e2,1.e-8)
    w1=max(0.,1.-e1)*max(0.,(e2-e1)/compare)
    w2=max(0.,1.-e2)*max(0.,(e1-e2)/compare)
    instruct1=enabled and w1>0.
    instruct2=enabled and w2>0.

    _,ref_scale=add_gradient(second,.05*err2,.25,retain_graph=instruct1)
    c1_cos=c2_cos=None
    if instruct1:
        aux2=.1*w1*(b-a.detach()).square().mean()/signal
        c2_cos,_=add_gradient(second,aux2,.25,protect=True)
    if instruct2:
        aux1=.1*w2*(a-b.detach()).square().mean()
        c1_cos,_=add_gradient(first,aux1,.25,protect=True)
    return dict(md_c1_relative=e1**.5,md_c2_relative=e2**.5,c2_md_scale=ref_scale,
                trust_c1_to_c2=w1 if instruct1 else 0.,trust_c2_to_c1=w2 if instruct2 else 0.,
                cosine_c1_peer=c1_cos,cosine_c2_peer=c2_cos)


def verify_projection():
    model=torch.nn.Linear(2,1,bias=False).double()
    gradient=torch.tensor([[2.,0.]],dtype=torch.float64)
    model.weight.grad=gradient.clone()
    conflict=torch.tensor([[-3.,4.]],dtype=torch.float64)
    cosine,scale=add_gradient(model,(model.weight*conflict).sum(),.25,protect=True)
    assert cosine<0.
    torch.testing.assert_close(model.weight.grad,torch.tensor([[2.,.5]],dtype=torch.float64))
    assert scale==.125
    emit(stage='auxiliary_gradient_projection',test='passed')


def publish(path,data):
    pending=path.with_suffix('.partial')
    pending.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    pending.rename(path)


def run_epoch(label,epoch,models,opts,groups,rows,refs,train_geo,pure,args,output):
    emit(stage='ablation_epoch_start',arm=label,epoch=epoch,states=sum(map(len,groups.values())))
    outcome=fit(models,opts,groups,rows,refs,train_geo,pure,np.random.default_rng(930+epoch),args,
                couple=partial(teaching,enabled=label=='gated_peer'))
    checkpoint=output/f'{label}_epoch{epoch}.pt'
    pending=checkpoint.with_suffix('.partial')
    torch.save(dict(models=[m.state_dict() for m in models],optimizers=[o.state_dict() for o in opts]),
               pending)
    pending.rename(checkpoint)
    publish(output/f'{label}_epoch{epoch}.json',outcome)
    transfers=outcome['pairs']
    emit(stage='ablation_epoch_end',arm=label,epoch=epoch,states=outcome['states'],
         seconds=outcome['seconds'],mean_c1_EL_mse=outcome['mean_c1_EL_mse'],
         mean_bulk_c2_loss=outcome['mean_bulk_c2_loss'],peak_gpu_GiB=outcome['peak_gpu_GiB'],
         c1_to_c2=sum(p['trust_c1_to_c2']>0 for p in transfers),
         c2_to_c1=sum(p['trust_c2_to_c1']>0 for p in transfers))


def main(a):
    verify_projection()
    check_peer_gradient()
    free,total=torch.cuda.mem_get_info()
    if free<5*2**30:
        raise RuntimeError('CUDA needs at least 5 GiB free at launch')
    torch.cuda.set_per_process_memory_fraction(min(20*2**30,free-2*2**30)/total)
    refs,manifest=role_manifest(a.references)
    rows={r['case']:r for r in entries()}
    all_states=json.loads(a.selection.read_text())
    order=np.random.default_rng(62026)
    groups={}
    for name in (*BULK_CASES,*EXTERNAL_CASES):
        states=np.asarray(all_states[name])
        assert states[0]==0 and rows[name]['split']=='train'
        selected=np.sort(order.choice(states[1:],95,replace=False)).tolist()
        groups[name]=[0,*selected]
    assert sum(map(len,groups.values()))==768 and not set(groups)&set(manifest['validation_states'])
    g=json.loads((FULL/'geometry_71_manifest.json').read_text())['splits']
    train_geo={r['case']:r for r in g['train']}
    held={r['case']:r for split in ('validation_mid','test') for r in g[split]}
    chosen=set(BULK_CASES)|{WARMUP_CASE,'H2O-T986.4-r02-rho0.5256'}
    pure=[r for r in bulk_rows() if r['split']=='train' and r['case'] in chosen]
    assert {r['case'] for r in pure}==chosen
    a.out.mkdir()
    publish(a.out/'selection.json',dict(c1=groups,c2=sorted(chosen)))
    start,opt=initialized()
    run_epoch('warmup',1,start,opt,groups,rows,refs,train_geo,pure,a,a.out)
    weights=[deepcopy(m.state_dict()) for m in start]
    state=[deepcopy(o.state_dict()) for o in opt]
    del start,opt
    for arm in ('no_peer','gated_peer'):
        models,opts=initialized()
        for m,o,model_state,opt_state in zip(models,opts,weights,state):
            m.load_state_dict(model_state)
            o.load_state_dict(deepcopy(opt_state))
        for epoch in range(2,a.epochs+1):
            run_epoch(arm,epoch,models,opts,groups,rows,refs,train_geo,pure,a,a.out)
        emit(stage='ablation_holdout',arm=arm)
        scores=holdout(models,held,rows,refs,manifest['validation_states'])
        publish(a.out/f'{arm}_holdout.json',scores)
        del models,opts
        torch.cuda.empty_cache()
    emit(stage='ablation_complete')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--references',required=True,type=Path)
    p.add_argument('--selection',required=True,type=Path)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if a.batch<1 or a.epochs<2:
        p.error('Require a positive batch and at least two epochs per arm')
    if a.check_only:
        verify_projection()
        check_peer_gradient()
    else:
        main(a)
