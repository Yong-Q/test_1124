"""Fit separate excess functionals to C1 or C2, with optional frozen response guidance."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from diagnose_holdout import bulk_metrics
from fuse import Bulk, FULL, PURE
from references import LIMIT, entries
from train_rows import frame_weight_gradient
from voxel_loss import Voxels, check_frame_adjoint, network, role_manifest, test_small, voxel_gradient


WARMUP_CASE = 'C2H4-source239-C2H4-PRL-T430.3-r00-0.000291'


def model_derivative(model, dataset, state, columns):
    w = dataset.weights[state]
    q = dataset.counts.index_select(1,columns)
    mean = w@q
    gradient = frame_weight_gradient(model,dataset.case,w,model.training)
    return voxel_gradient(q,mean,w,gradient)


def remove_offsets(values,element):
    n = int(element.max()) + 1
    counts = torch.bincount(element,minlength=n).to(values)
    sums = values.new_zeros(n).scatter_add_(0,element,values)
    return values - (sums/counts.clamp_min(1))[element]


def paired_change(model,teacher,dataset,state,rng):
    """Compare C1 increments at equal grid sites along an actual weak-field path."""
    eligible = dataset.valid[0] & dataset.valid[state]
    sites = np.flatnonzero(eligible.any(axis=1))
    sites = np.sort(rng.choice(sites,min(LIMIT,len(sites)),replace=False))
    flat = sites[:,None]*dataset.ne+np.arange(dataset.ne)
    ids = flat[eligible[sites]]
    if not len(ids):
        raise ValueError(f'{dataset.name}: no common occupied sites at state {state}')
    col = torch.as_tensor(ids,device=dataset.device)
    element = col % dataset.ne
    target = torch.as_tensor(dataset.ref[state,ids]-dataset.ref[0,ids],device=dataset.device)
    target = remove_offsets(target,element)

    def change(net):
        initial = model_derivative(net,dataset,0,col)
        perturbed = model_derivative(net,dataset,state,col)
        return remove_offsets(perturbed-initial,element)

    teacher_delta = change(teacher).detach()
    student_delta = teacher_delta if model is teacher else change(model)
    guide = (student_delta-teacher_delta).square().mean()
    ref_rms = target.square().mean().sqrt()
    teacher_rmse = (teacher_delta-target).square().mean().sqrt()
    student_rmse = (student_delta-target).square().mean().sqrt()
    return guide,dict(site_count=len(sites),teacher_rmse=float(teacher_rmse),
                      student_rmse=float(student_rmse.detach()),ref_rms=float(ref_rms))


def c1_epoch(model,optimizer,groups,rows,refs,geom,device,rng,args,teacher):
    value,voxels,states,updates,paired = 0.,0,0,0,[]
    for name in rng.permutation(list(groups)):
        row = rows[name]
        cache = row['xi'] if row['kind']=='external' else geom[name]['cache']
        dataset = Voxels(row,refs[name],cache,device)
        order = rng.permutation(groups[name])
        for start in range(0,len(order),args.batch):
            batch = order[start:start+args.batch]
            optimizer.zero_grad(set_to_none=True)
            for state in batch:
                loss,count,*rest = dataset.at(model,int(state),rng)
                (loss/len(batch)).backward()
                value += float(loss.detach())
                states += 1
                voxels += count
                del loss,rest
            if teacher is not None and start==0:
                candidate = next((int(s) for s in batch if s>0),None)
                if candidate is not None:
                    guide,report = paired_change(model,teacher,dataset,candidate,rng)
                    # The nonuniform MD labels set confidence in transferring
                    # a teacher extrapolation. The teacher is kept fixed.
                    trust = 1./(1.+(report['teacher_rmse']/max(report['ref_rms'],1.e-5))**2)
                    (args.guide_weight*trust*guide).backward()
                    paired.append(dict(case=name,state=candidate,weight=args.guide_weight*trust,
                                       guide_mse=float(guide.detach()),**report))
                    del guide
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            if updates==1 or updates%32==0:
                emit(stage='update',line='c1',updates=updates,case=name,
                     states=states,voxels=voxels,mean_c1_EL_mse=value/states)
        del dataset
    assert states==sum(map(len,groups.values()))
    assert 0<voxels<=LIMIT*states
    return dict(states=states,voxels=voxels,updates=updates,
                mean_c1_EL_mse=value/states,paired=paired)


def c2_epoch(model,optimizer,rows,geom,device,rng,budget):
    order = rng.permutation(len(rows))
    # Match the update budget of the joint run across the same bulk cases.
    group_sizes = np.array_split(np.arange(budget),len(rows))
    losses = []
    for j,indices in zip(order,group_sizes):
        row = rows[j]
        case = Bulk(row,geom[row['case']],device)
        for _ in indices:
            optimizer.zero_grad(set_to_none=True)
            value,_ = case.loss(model,rng)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(value.detach()))
            if len(losses)==1 or len(losses)%32==0:
                emit(stage='update',line='c2',updates=len(losses),case=row['case'],
                     mean_bulk_c2_loss=float(np.mean(losses)))
        del case
    assert len(losses)==budget and all(len(ids) for ids in group_sizes)
    return dict(sources=len(rows),updates=budget,mean_bulk_c2_loss=float(np.mean(losses)))


def emit(**data):
    print(json.dumps(data,allow_nan=False),flush=True)


def save(model,optimizer,rng,epoch,out):
    for file,obj in ((f'epoch{epoch}.pt',model.state_dict()),
                     (f'optimizer_epoch{epoch}.pt',dict(optimizer=optimizer.state_dict(),
                                                      numpy_rng=rng.bit_generator.state))):
        final=out/file
        pending=final.with_suffix('.partial')
        torch.save(obj,pending)
        pending.rename(final)


def bulk_rows():
    return [r for p in sorted(PURE.glob('bulk_161755_*/manifest.json'))
            for r in json.loads(p.read_text())['results'] if r['centers']>0]


def evaluate_c2(model,geom,out,device):
    model.eval().requires_grad_(False)
    scores = []
    for row in bulk_rows():
        if row['split']=='train':
            continue
        case=Bulk(row,geom[row['case']],device)
        _,m=case.loss(model,np.random.default_rng(617),training=False,draws=8,report_spectrum=True)
        score=dict(case=row['case'],split=row['split'],**bulk_metrics(m))
        emit(stage='c2_evaluation',**score)
        scores.append(score)
        del case
    assert len(scores)==29
    (out/'summary.json').write_text(json.dumps(scores,indent=2,allow_nan=False)+'\n')


def evaluate_transfer(net,references,geo,out):
    refs,manifest=role_manifest(references)
    records={r['case']:r for r in entries() if r['split']!='train'}
    names=('C2H4-T430.4-r05-rho0.3089','C2H4_r05_V1well','NH3_r05_field','C2H6_r05_field')
    rng=np.random.default_rng(417)
    result=[]
    net.eval().requires_grad_(False)
    for name in names:
        r=records[name]
        grid=Voxels(r,refs[name],r['xi'] if r['kind']=='external' else geo[name]['cache'],'cuda')
        ids=np.asarray(manifest['validation_states'][name][1:])
        for state in rng.choice(ids,8,replace=False):
            _,values=paired_change(net,net,grid,int(state),rng)
            report=dict(case=name,state=int(state),**values)
            result.append(report)
            emit(stage='teacher_transfer',**report)
        del grid
    assert len(result)==32
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def main(a):
    free,total=torch.cuda.mem_get_info()
    if free<5*2**30:
        raise RuntimeError('Insufficient CUDA memory at startup')
    torch.cuda.set_per_process_memory_fraction(min(20*2**30,free-2*2**30)/total)
    device='cuda'
    splits=json.loads((FULL/'geometry_71_manifest.json').read_text())['splits']
    train_geo={r['case']:r for r in splits['train']}
    if a.line=='c1':
        test_small()
        check_frame_adjoint()
    torch.manual_seed(617)
    rng=np.random.default_rng(617)
    model=network(device)
    a.out.mkdir()
    if a.evaluate or a.check_transfer:
        model.load_state_dict(torch.load(a.evaluate or a.check_transfer,weights_only=True,map_location='cuda'))
        held={r['case']:r for tag in ('validation_mid','test') for r in splits[tag]}
        if a.check_transfer:
            evaluate_transfer(model,a.references,held,a.out)
        else:
            evaluate_c2(model,held,a.out,device)
        return
    opt=torch.optim.AdamW(model.parameters(),lr=.002)
    if a.resume:
        model.load_state_dict(torch.load(a.resume/f'epoch{a.start_epoch-1}.pt',
                                         weights_only=True,map_location='cuda'))
        state=torch.load(a.resume/f'optimizer_epoch{a.start_epoch-1}.pt',
                         weights_only=True,map_location='cuda')
        opt.load_state_dict(state['optimizer'])
        rng.bit_generator.state=state['numpy_rng']
    teacher=None
    if a.line=='c1':
        refs,manifest=role_manifest(a.references)
        raw={r['case']:r for r in entries() if r['split']=='train'}
        groups=json.loads(a.selection.read_text())
        if a.smoke:
            groups={WARMUP_CASE:groups[WARMUP_CASE][:a.batch]}
        assert set(groups)<=set(raw) and not set(groups)&set(manifest['validation_states'])
        assert a.smoke or (len(groups)==49 and sum(map(len,groups.values()))==10000)
        if a.teacher:
            teacher=network(device).eval().requires_grad_(False)
            teacher.load_state_dict(torch.load(a.teacher,weights_only=True,map_location='cuda'))
            assert not ({id(p) for p in teacher.parameters()} & {id(p) for p in model.parameters()})
        emit(stage='training_ready',line=a.line,sources=len(groups),
             states=sum(map(len,groups.values())),epochs=a.epochs,teacher=str(a.teacher) if teacher else None)
    else:
        rows=[r for r in bulk_rows() if r['split']=='train']
        if a.smoke:
            rows=[r for r in rows if r['case']==WARMUP_CASE]
        assert len(rows)==(1 if a.smoke else 41) and all(r['case'] in train_geo for r in rows)
        emit(stage='training_ready',line=a.line,sources=len(rows),epochs=a.epochs)
    history=[]
    for epoch in range(a.start_epoch,a.start_epoch+a.epochs):
        start=perf_counter()
        if a.line=='c1':
            result=c1_epoch(model,opt,groups,raw,refs,train_geo,device,rng,a,teacher)
        else:
            result=c2_epoch(model,opt,rows,train_geo,device,rng,3 if a.smoke else a.updates)
        save(model,opt,rng,epoch,a.out)
        info=dict(epoch=epoch,line=a.line,seconds=perf_counter()-start,
                  peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,**result)
        history.append(info)
        emit(stage='epoch_end',**{k:v for k,v in info.items() if k!='paired'},
             paired_states=len(info.get('paired',[])))
        (a.out/f'epoch{epoch}_metrics.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
    (a.out/'summary.json').write_text(json.dumps(history,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--line',choices=('c1','c2'),required=True)
    p.add_argument('--references',type=Path)
    p.add_argument('--selection',type=Path)
    p.add_argument('--teacher',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--evaluate',type=Path)
    p.add_argument('--check-transfer',type=Path)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--updates',type=int,default=343)
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--start-epoch',type=int,default=1)
    p.add_argument('--guide-weight',type=float,default=.1)
    a=p.parse_args()
    if a.line=='c1' and (a.references is None or a.selection is None):
        p.error('C1 requires its reference manifest and its selected training states')
    if a.check_transfer and (a.line!='c2' or a.references is None):
        p.error('Transfer checking requires the C2 checkpoint and held-out grid references')
    if (a.evaluate and a.line!='c2') or (a.teacher and a.line!='c1'):
        p.error('Use --evaluate for C2 or --teacher for C1')
    if (a.start_epoch>1)!=bool(a.resume) or min(a.batch,a.epochs,a.start_epoch)<1:
        p.error('Provide --resume when continuing a positive epoch number')
    main(a)
