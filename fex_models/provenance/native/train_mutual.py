"""Two independent excess functionals with externally verified response transfer."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from diagnose_holdout import bulk_metrics
from fuse import Bulk, FULL
from references import LIMIT, entries
from run_lines import WARMUP_CASE, bulk_rows, emit, model_derivative, remove_offsets
from voxel_loss import Voxels, check_frame_adjoint, network, role_manifest, test_small


def response_pair(dataset,state,rng):
    valid = dataset.valid[0] & dataset.valid[state]
    sites = np.flatnonzero(valid.any(axis=1))
    sites = np.sort(rng.choice(sites,min(LIMIT,len(sites)),replace=False))
    if not len(sites):
        raise ValueError(f'{dataset.name}: no common occupied voxel for state {state}')
    cols = (sites[:,None]*dataset.ne+np.arange(dataset.ne))[valid[sites]]
    index = torch.as_tensor(cols,device=dataset.device)
    element = index % dataset.ne
    target = torch.as_tensor(dataset.ref[state,cols]-dataset.ref[0,cols],device=dataset.device)
    return index,element,remove_offsets(target,element),len(sites)


def response(model,dataset,state,columns,element):
    start = model_derivative(model,dataset,0,columns)
    end = model_derivative(model,dataset,state,columns)
    return remove_offsets(end-start,element)


def mutual_terms(first,second,reference):
    signal = reference.square().mean().clamp_min(1.e-4)
    e1 = (first-reference).square().mean()/signal
    e2 = (second-reference).square().mean()/signal
    # Labels from the task train split, so a peer contributes only where it
    # outperforms the other model, continuously downweighted by its own error.
    a,b = e1.detach(),e2.detach()
    denominator = (a+b).clamp_min(1.e-8)
    trust2 = ((a-b)/denominator).clamp_min(0.)/(1.+b)
    trust1 = ((b-a)/denominator).clamp_min(0.)/(1.+a)
    to_first = trust2*(first-second.detach()).square().mean()
    to_second = trust1*(second-first.detach()).square().mean()/signal
    return e2,to_first,to_second,dict(md_c1_relative=float(a.sqrt()),md_c2_relative=float(b.sqrt()),
                                    trust_c2_to_c1=float(trust2),trust_c1_to_c2=float(trust1))


def check_peer_gradient():
    reference = torch.tensor([1.,-1.],dtype=torch.float64)
    for a,b in ((.7,.3),(.3,.7)):
        x=torch.tensor(a,requires_grad=True,dtype=torch.float64)
        y=torch.tensor(b,requires_grad=True,dtype=torch.float64)
        measured,to_x,to_y,stats=mutual_terms(x*reference,y*reference,reference)
        gx,leak_to_y=torch.autograd.grad(to_x,(x,y),allow_unused=True,retain_graph=True)
        leak_to_x,gy=torch.autograd.grad(to_y,(x,y),allow_unused=True,retain_graph=True)
        assert leak_to_x is None and leak_to_y is None
        assert float(gx)<0. if b>a else float(gy)<0.
        gm,=torch.autograd.grad(measured,y)
        assert float(gm)<0.
    emit(stage='two_way_stop_gradient',test='passed')


def add_capped_gradient(loss,model,ratio):
    params=tuple(model.parameters())
    original=torch.nn.utils.clip_grad_norm_(params,float('inf'),error_if_nonfinite=True)
    added=torch.autograd.grad(loss,params,allow_unused=True)
    norm=torch.stack([g.detach().square().sum() for g in added if g is not None]).sum().sqrt()
    fraction=float((ratio*original/norm.clamp_min(1.e-20)).clamp_max(1.))
    for p,g in zip(params,added):
        if g is not None:
            step=fraction*g
            p.grad=step if p.grad is None else p.grad+step
    return fraction


def fit(models,optimizers,groups,data,refs,geom,pure,rng,args,smoke=False,couple=None):
    first,second=models
    opt1,opt2=optimizers
    keys=rng.permutation(list(groups))
    order=rng.permutation(len(pure))
    total=sum((len(ids)+args.batch-1)//args.batch for ids in groups.values())
    assert total>=len(pure)
    active,bulk=None,None
    updates,states,points=0,0,0
    first_sum,c2_loss=0.,[]
    transfers=[]
    used_bulk=set()
    started=perf_counter()
    for name in keys:
        row=data[name]
        dataset=Voxels(row,refs[name],row['xi'] if row['kind']=='external' else geom[name]['cache'],'cuda')
        ids=rng.permutation(groups[name])
        for lo in range(0,len(ids),args.batch):
            batch=ids[lo:lo+args.batch]
            opt1.zero_grad(set_to_none=True)
            opt2.zero_grad(set_to_none=True)
            for idx in batch:
                loss,n,*rest=dataset.at(first,int(idx),rng)
                (loss/len(batch)).backward()
                first_sum+=float(loss.detach())
                points+=n
                states+=1
                del loss,rest

            choice=int(order[updates*len(pure)//total])
            if active!=choice:
                active=choice
                bulk=None
                entry=pure[active]
                bulk=Bulk(entry,geom[entry['case']],'cuda')
            h,_=bulk.loss(second,rng)
            h.backward()
            c2_loss.append(float(h.detach()))
            used_bulk.add(pure[active]['case'])
            del h

            candidate=next((int(s) for s in batch if s>0),None)
            # One pair per source, so a full trial covers all field families.
            if lo==0 and candidate is not None:
                cols,element,target,n=response_pair(dataset,candidate,rng)
                a=response(first,dataset,candidate,cols,element)
                b=response(second,dataset,candidate,cols,element)
                if couple is None:
                    direct,to_first,to_second,diagnostics=mutual_terms(a,b,target)
                    factor2=add_capped_gradient(.05*direct+.10*to_second,second,.25)
                    factor1=add_capped_gradient(.10*to_first,first,.25)
                    diagnostics.update(scale_c1_aux=factor1,scale_c2_aux=factor2)
                    del direct,to_first,to_second
                else:
                    diagnostics=couple(first,second,a,b,target)
                line=dict(case=name,state=candidate,grid_points=n,**diagnostics)
                transfers.append(line)
                if smoke or len(transfers)<=2 or len(transfers)%10==0:
                    emit(stage='paired_update',**line)
                del a,b

            for model,opt in zip(models,optimizers):
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                opt.step()
            updates+=1
            if smoke or updates<=3 or updates%20==0:
                emit(stage='update',trial='smoke' if smoke else 'full',updates=updates,total=total,
                     c1_case=name,c2_case=pure[active]['case'],states=states,c1_voxels=points,
                     mean_c1_EL_mse=first_sum/states,mean_bulk_c2_loss=float(np.mean(c2_loss)))
        del dataset
    assert states==sum(map(len,groups.values())) and states<=points<=LIMIT*states
    assert used_bulk=={r['case'] for r in pure}
    assert transfers
    return dict(states=states,voxels=points,updates=updates,bulk_sources=len(used_bulk),
                seconds=perf_counter()-started,mean_c1_EL_mse=first_sum/states,
                mean_bulk_c2_loss=float(np.mean(c2_loss)),pairs=transfers,
                peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30)


def holdout(models,geom,records,refs,holdouts):
    first,second=(m.eval().requires_grad_(False) for m in models)
    result={'c2_bulk':[],'c1_and_transfer':[]}
    for row in bulk_rows():
        name=row['case']
        if row['split']!='validation' and name not in ('C2H6-T465.4-r06-rho0.2367',
                                                       'NH3-T618.2-r04-rho0.3811'):
            continue
        case=Bulk(row,geom[name],'cuda')
        _,info=case.loss(second,np.random.default_rng(117),training=False,draws=8,report_spectrum=True)
        found=dict(case=name,split=row['split'],**bulk_metrics(info))
        result['c2_bulk'].append(found)
        emit(stage='c2_holdout',**found)
        del case
    rng=np.random.default_rng(417)
    names=('C2H4-T430.4-r05-rho0.3089','C2H4_r05_V1well','NH3_r05_field','C2H6_r05_field')
    for name in names:
        row=records[name]
        dataset=Voxels(row,refs[name],row['xi'] if row['kind']=='external' else geom[name]['cache'],'cuda')
        states=rng.choice(holdouts[name][1:],8,replace=False)
        for s in states:
            s=int(s)
            _,_,_,prediction,target,offset,codes=dataset.at(first,s,rng)
            _,index=torch.unique(codes,sorted=True,return_inverse=True)
            delta=prediction+offset[index]-target
            signal=remove_offsets(target,codes)
            cols,element,target_delta,_=response_pair(dataset,s,rng)
            d1=response(first,dataset,s,cols,element)
            d2=response(second,dataset,s,cols,element)
            found=dict(case=name,state=s,rmse=float(delta.square().mean().sqrt()),
                       r2=float(1.-delta.square().sum()/signal.square().sum().clamp_min(1.e-12)),
                       md_c1_delta_rmse=float((d1-target_delta).square().mean().sqrt()),
                       md_c2_delta_rmse=float((d2-target_delta).square().mean().sqrt()),
                       reference_delta_rms=float(target_delta.square().mean().sqrt()))
            result['c1_and_transfer'].append(found)
        emit(stage='c1_holdout',case=name,states=len(states),
             mean_rmse=float(np.mean([r['rmse'] for r in result['c1_and_transfer'] if r['case']==name])))
        del dataset
    assert len(result['c2_bulk'])==6 and len(result['c1_and_transfer'])==32
    return result


def initialized():
    models=[]
    for seed in (617,618):
        torch.manual_seed(seed)
        models.append(network('cuda'))
    first,second=models
    assert not ({id(p) for p in first.parameters()}&{id(p) for p in second.parameters()})
    return models,[torch.optim.AdamW(m.parameters(),lr=.002) for m in models]


def main(a):
    check_peer_gradient()
    check_frame_adjoint()
    test_small()
    free,total=torch.cuda.mem_get_info()
    if free<5*2**30:
        raise RuntimeError('CUDA needs at least 5 GiB free at launch')
    torch.cuda.set_per_process_memory_fraction(min(20*2**30,free-2*2**30)/total)
    refs,manifest=role_manifest(a.references)
    records={r['case']:r for r in entries()}
    groups=json.loads(a.selection.read_text())
    geom=json.loads((FULL/'geometry_71_manifest.json').read_text())['splits']
    training={r['case']:r for r in geom['train']}
    pure=[r for r in bulk_rows() if r['split']=='train']
    assert len(pure)==41
    assert len(groups)==49 and sum(map(len,groups.values()))==10000
    assert all(records[key]['split']=='train' for key in groups)
    assert not set(groups)&set(manifest['validation_states'])
    a.out.mkdir()
    (a.out/'selection.json').write_text(json.dumps(groups,indent=2)+'\n')

    if a.resume_weights is None:
        smoke='C2H4-T430.4-r01-rho0.5978'
        smodel,sopt=initialized()
        test=fit(smodel,sopt,{smoke:groups[smoke][:a.batch]},records,refs,training,
                [r for r in pure if r['case']==WARMUP_CASE],np.random.default_rng(617),a,smoke=True)
        emit(stage='large_box_smoke_complete',peak_gpu_GiB=test['peak_gpu_GiB'])
        del smodel,sopt
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    models,opts=initialized()
    labels=('c1','c2')
    if a.resume_weights is not None:
        for label,model in zip(labels,models):
            path=a.resume_weights/f'{label}_epoch{a.start_epoch-1}.pt'
            model.load_state_dict(torch.load(path,weights_only=True,map_location='cuda'))
    emit(stage='training_ready',c1_states=10000,bulk_c2_sources=41,batch=a.batch,
         start_epoch=a.start_epoch,last_epoch=a.epochs,
         optimizer_initialization='fresh AdamW',checkpoint_source=str(a.resume_weights) if a.resume_weights else None)
    for epoch in range(a.start_epoch,a.epochs+1):
        emit(stage='epoch_start',epoch=epoch)
        train=fit(models,opts,groups,records,refs,training,pure,np.random.default_rng(616+epoch),a)
        for label,model,opt in zip(labels,models,opts):
            for name,state in ((f'{label}_epoch{epoch}.pt',model.state_dict()),
                               (f'{label}_optimizer_epoch{epoch}.pt',opt.state_dict())):
                path=a.out/name
                pending=path.with_suffix('.partial')
                torch.save(state,pending)
                pending.rename(path)
        metrics=a.out/('training.json' if a.epochs==1 else f'training_epoch{epoch}.json')
        pending=metrics.with_suffix('.partial')
        pending.write_text(json.dumps(train,indent=2,allow_nan=False)+'\n')
        pending.rename(metrics)
        emit(stage='epoch_complete',epoch=epoch,**{k:v for k,v in train.items() if k!='pairs'},
             pairs=len(train['pairs']))
    valid={r['case']:r for split in ('validation_mid','test') for r in geom[split]}
    results=holdout(models,valid,records,refs,manifest['validation_states'])
    (a.out/'holdout.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--references',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--epochs',type=int,default=1,help='Final epoch number')
    p.add_argument('--start-epoch',type=int,default=1)
    p.add_argument('--resume-weights',type=Path)
    a=p.parse_args()
    if a.batch<1 or not 1<=a.start_epoch<=a.epochs:
        p.error('Require a positive batch and 1 <= start-epoch <= epochs')
    if (a.start_epoch>1)!=bool(a.resume_weights):
        p.error('Continuation requires --resume-weights')
    main(a)
