"""Use python -m fex_models to check data, train, or evaluate a released model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .core.model import energy
from .data import Case
from .training import c1_observables, c2_loss, metrics, model_set, step_grid, step_ppt


ROOT=Path(__file__).resolve().parent
MODES=('ppt','c1','c2','guided','mutual','gated','no-peer')
DEFAULT_TRAIN=ROOT/'examples/ch4_prepared.json'
DEFAULT_VALIDATION=ROOT/'examples/c2h6_prepared.json'


def load_models(path,models,device):
    weights=torch.load(path,weights_only=True,map_location=device)
    states=weights.get('models',weights)
    if len(models)==1 and set(models).isdisjoint(states):
        states={next(iter(models)):states}
    for key,net in models.items():
        net.load_state_dict(states[key],strict=True)


def evaluate(case,models,arrays,prefix):
    by={}
    def save(key,value):
        arrays[prefix+'_'+key]=value.detach().cpu().numpy()

    def free_energy(key,tag,meta):
        base=tuple(x[0] for x in case.fields(tag))
        save(key,energy(models['ppt' if tag=='ppt' else 'c2' if tag=='bulk' else 'c1'],
                        base,case.meta(meta)))

    for model in models.values():
        model.eval()
    if 'ppt' in models:
        y,h=case.ppt(models['ppt'],False)
        by['ppt_c1']=metrics(y,case.d['ppt_y1'])
        by['ppt_c2']=metrics(h,case.d['ppt_y2'])
        save('ppt_c1',y)
        save('ppt_c2',h)
        free_energy('ppt_beta_fex','ppt','ppt')
    if 'c1' in models:
        _,a,target,block=c1_observables(case,models['c1'],False)
        for i,k in enumerate(('base','weak')):
            by['grid_c1_'+k]=metrics(block[i],case.d[f'grid_{k}_label'],case.d['grid_element'])
            save(f'grid_c1_{k}',block[i])
            save(f'grid_beta_mu_{k}',block[i+2])
            free_energy(f'grid_beta_fex_{k}',f'grid_{k}','grid')
        by['grid_c1_delta']=metrics(a,target)
        save('grid_c1_delta',a)
    if 'c2' in models:
        raw=case.c2(models['c2'],False)
        by['bulk_c2']=metrics(raw.mean(1),case.d['bulk_reference'])
        by['bulk_c2']['relative_rmse']=float(c2_loss(case,raw,False).sqrt())
        save('bulk_c2',raw.mean(1))
        save('bulk_c2_draws',raw)
        free_energy('bulk_beta_fex','bulk','grid')
    return dict(case=case.name,role=case.role,metrics=by)


def check(cases,device):
    for c in cases:
        model=model_set('ppt',c.scale,device)['ppt'].eval()
        y,h=c.ppt(model,False)
        fresh=model_set('c2',c.scale,device)['c2'].eval()
        all_values=[(y,'ppt_native_c1'),(h,'ppt_native_c2'),
                    (c.c1(fresh,'base',False),'grid_base_native_c1'),
                    (c.c1(fresh,'weak',False),'grid_weak_native_c1'),
                    (c.c2(fresh,False),'bulk_native_c2')]
        errors={}
        for observed,key in all_values:
            expected=c.d[key]
            diff=(observed-expected).double()
            absolute=float(diff.abs().max())
            relative=float(diff.norm()/expected.double().norm().clamp_min(1.e-10))
            if relative>7.e-4 or absolute>2.e-3:
                raise AssertionError(f'{c.name} {key}: absolute {absolute:.6g}, relative {relative:.6g}')
            errors[key]=dict(max_abs=absolute,relative_l2=relative)
        print(json.dumps(dict(case=c.name,role=c.role,check='native derivative parity',
                              comparisons=len(all_values),errors=errors),allow_nan=False),flush=True)


def check_roles(train,validation):
    if any(c.role!='train' for c in train):
        raise ValueError('Cases marked holdout can only be used for evaluation')
    if any(c.role!='holdout' for c in validation):
        raise ValueError('Validation cases must have the holdout role')
    if {c.name for c in train} & {c.name for c in validation}:
        raise ValueError('Training and validation must use distinct source trajectories')


def main(a):
    if a.device=='cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Use --device cpu or install a CUDA PyTorch wheel.')
        free,total=torch.cuda.mem_get_info()
        cap=min(a.memory_gib*2**30,free-512*2**20)
        if cap<=0:
            raise RuntimeError('Insufficient CUDA memory')
        torch.cuda.set_per_process_memory_fraction(cap/total)
    if a.command=='check':
        paths=a.case or [DEFAULT_TRAIN,DEFAULT_VALIDATION]
        check([Case(p,a.device) for p in paths],a.device)
        return
    validation=[Case(p,a.device) for p in (a.validation_case or [DEFAULT_VALIDATION])]
    train=[] if a.command=='evaluate' else [Case(p,a.device) for p in (a.train_case or [DEFAULT_TRAIN])]
    check_roles(train,validation)
    scale=(train or validation)[0].scale
    for c in train+validation:
        torch.testing.assert_close(c.scale,scale,rtol=0,atol=0)
    models=model_set(a.model,scale,a.device)
    records=[]
    if a.command=='evaluate':
        load_models(a.weights,models,a.device)
    else:
        if a.model=='guided':
            if a.teacher is None:
                raise ValueError('The guided dataset requires --teacher with a C2 checkpoint')
            load_models(a.teacher,{'c2':models['c2']},a.device)
            models['c2'].eval().requires_grad_(False)
        updated={k:m for k,m in models.items() if any(p.requires_grad for p in m.parameters())}
        lr=a.lr if a.lr is not None else (.005 if a.model=='ppt' else .002)
        optim={k:torch.optim.AdamW(m.parameters(),lr=lr) for k,m in updated.items()}
        rng=np.random.default_rng(617)
        for epoch in range(1,a.epochs+1):
            for m in updated.values():
                m.train()
            for i in rng.permutation(len(train)):
                c=train[i]
                for o in optim.values():
                    o.zero_grad(set_to_none=True)
                scores=step_ppt(c,models['ppt']) if a.model=='ppt' else step_grid(c,models,a.model)
                for k,o in optim.items():
                    magnitude=torch.nn.utils.clip_grad_norm_(updated[k].parameters(),10.,error_if_nonfinite=True)
                    scores[f'{k}_gradient_norm']=float(magnitude)
                    o.step()
                records.append(dict(epoch=epoch,case=c.name,**scores))
    arrays={}
    result=dict(model=a.model,device=a.device,training=records,
                validation=[evaluate(c,models,arrays,f'case_{i}') for i,c in enumerate(validation)])
    output=a.output
    output.mkdir(parents=True)
    if a.command=='train':
        torch.save(dict(mode=a.model,models={k:v.state_dict() for k,v in models.items()}),output/'weights.pt')
    np.savez_compressed(output/'validation_predictions.npz',**arrays)
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    cmd=p.add_subparsers(dest='command',required=True)
    def common(q):
        q.add_argument('--device',choices=('cpu','cuda'),default='cpu')
        q.add_argument('--memory-gib',type=float,default=8.)
    check_parser=cmd.add_parser('check',help='Compare real test packets with saved native derivatives')
    common(check_parser)
    check_parser.add_argument('--case',type=Path,action='append')
    for name in ('train','evaluate'):
        q=cmd.add_parser(name)
        common(q)
        q.add_argument('--model',required=True,choices=MODES)
        q.add_argument('--output',type=Path,required=True)
        q.add_argument('--validation-case',type=Path,action='append')
        if name=='train':
            q.add_argument('--train-case',type=Path,action='append')
            q.add_argument('--epochs',type=int,default=1)
            q.add_argument('--lr',type=float)
            q.add_argument('--teacher',type=Path)
        else:
            q.add_argument('--weights',type=Path,required=True)
    args=p.parse_args()
    if args.memory_gib<=0 or (args.command=='train' and (args.epochs<1 or (args.lr is not None and args.lr<=0))):
        p.error('Require positive memory, epochs and learning rate')
    if args.command!='check' and args.output.exists():
        p.error('Select a new output directory')
    main(args)
