"""Train on matched Xi probe labels with optional short spatial coupling."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch

from angular_model import AngularSpatial
from atom_functional import structure_factors
from curvature_mu.fit import bounded_backward
from fit_angular import AngularCase, compute, select_training
from kernel import box_lengths
from relative import projected_loss
from short_kernel import SMOOTH, ShortSpatial, project, setup, short_modes
from train import MANIFEST, PROBES
from xi_readout_1000.fit import grouped_report

HERE = Path(__file__).resolve().parent
ORDERS = {'radial_shells':0,'directional_shells':2}


class NearCase(AngularCase):
    def __init__(self,entry,rows,device,quota):
        super().__init__(entry,rows,device)
        then = perf_counter()
        with h5py.File(MANIFEST.parent/entry['cache'],'r') as h:
            lengths = box_lengths(h['cell'][0])
            vectors, importance = short_modes(lengths,quota)
            basis, self.short_kernels = setup(vectors,importance,device)
            wave = torch.as_tensor(vectors,dtype=torch.float32,device=device)
            smooth = torch.exp(-.5*SMOOTH**2*wave.square().sum(-1))
            self.short_fields = torch.empty((len(self.observable),len(wave)*8*3),dtype=torch.complex64)
            for i in range(0,len(self.observable),8):
                sl = slice(i,i+8)
                positions = torch.as_tensor(h['positions'][sl],device=device,dtype=torch.float32)
                marks = self.observable[sl].reshape(len(positions),self.atoms,8,16)[...,:9].flatten(2)
                modes = structure_factors(positions,marks,wave).reshape(len(positions),len(wave),8,9)
                value = project(modes,basis)*smooth[None,:,None,None]
                self.short_fields[sl] = value.flatten(1).cpu()
        print(json.dumps(dict(stage='short_field',case=self.name,modes=len(wave),seconds=perf_counter()-then)),flush=True)

    def move(self,device):
        super().move(device)
        self.short_fields = self.short_fields.to(device)
        self.short_kernels = self.short_kernels.to(device)
        return self

    def short_state(self,i):
        row = [self.rows[i]]
        weights = np.c_[self.response.weights[row][0],self.response.select(row,PROBES[None])[0]]
        coef = torch.as_tensor(weights,device=self.short_fields.device,dtype=self.short_fields.dtype)
        packet = (coef.T@self.short_fields).reshape(1+len(PROBES),len(self.short_kernels),8,3)
        return packet[0],packet[1:]


def prediction(model,case,state,training):
    x,d,a,b,angular,short = state
    c1,c2 = compute(model,case,x,d,a,b,angular,training)
    if isinstance(model,ShortSpatial):
        n1,n2 = model.short_jets(*short,case.short_kernels,case.volume)
        c1,c2 = c1+n1,c2+n2
    return c1,c2


def load_state(case,i):
    x,d = case.state(i)
    a,b = case.spectral_state(i)
    angular = case.angular_state(i)
    near = case.short_state(i) if isinstance(case,NearCase) else None
    return x,d,a,b,angular,near


@torch.no_grad()
def evaluate(models,cases,device):
    values = {tag:{} for tag in models}
    for case in cases:
        case.move(device)
        collection = defaultdict(list)
        for i in range(len(case.rows)):
            state = load_state(case,i)
            for tag,model in models.items():
                collection[tag].append(prediction(model,case,state,False))
            del state
        for tag,items in collection.items():
            p,q = (torch.stack(v).cpu() for v in zip(*items))
            values[tag][case.name] = (p,q,case.t1.cpu(),case.t2.cpu())
        case.move('cpu')
    return {tag:grouped_report(data) for tag,data in values.items()}


def objective(metrics):
    return metrics['c2']['relative_rmse']**2+.1*metrics['c1']['relative_rmse']**2


def compact(scores):
    return {tag:{k:{stat:v[k][stat] for stat in ('rmse','r2','relative_rmse')}
                 for k in ('c1','c2','l0')} for tag,v in scores.items()}


def main(args):
    splits = ('train','validation_mid','test')
    original = json.loads(MANIFEST.read_text())['splits']
    entries = {e['case']:e for cases in original.values() for e in cases}
    fix = HERE.parent/'conditional_radial/run_161286/selection.json'
    chosen = {k:v for k,v in json.loads(fix.read_text()).items() if k in splits}
    if args.train_states!=192:
        chosen['train'] = select_training(original['train'],args.train_states)
    used = [(MANIFEST.parent/entries[x]['cache']).resolve() for k in splits for x in chosen[k]]
    assert len(used)==len(set(used)) and all(p.is_file() for p in used)
    assert not any('external_fields_23901522_' in str(p) for p in used)
    assert {'NH3','C2H6'}.isdisjoint(entries[name]['species'] for name in chosen['train'])
    assert sum(map(len,chosen['train'].values()))==args.train_states
    tags = ('baseline',*ORDERS) if args.model=='compare' else (args.model,)
    shell_lr = args.shell_lr
    if args.pilot:
        data = json.loads(args.pilot.read_text())
        tags = (data['selected_model'],)
        shell_lr = data.get('shell_lr',.005)
    args.out.mkdir()
    (args.out/'selection.json').write_text(json.dumps(chosen,indent=2)+'\n')
    device = 'cuda'
    free,cap = torch.cuda.mem_get_info()
    assert free>3*2**30,'GPU free memory is insufficient to start'
    torch.cuda.set_per_process_memory_fraction(min(20*2**30,free-2*2**30)/cap)
    scale = torch.from_numpy(np.load(HERE.parent/'xi_only_coupling/run_161288/scale.npy'))
    models,optim = {},{}
    for tag in tags:
        torch.manual_seed(617)
        install = AngularSpatial(scale.clone()) if tag=='baseline' else ShortSpatial(scale.clone(),ORDERS[tag])
        models[tag] = install.to(device)
        groups = models[tag].parameters()
        if isinstance(install,ShortSpatial) and shell_lr!=.005:
            groups = [
                {'params':[p for n,p in install.named_parameters() if n!='shell_weight']},
                {'params':[install.shell_weight],'lr':shell_lr},
            ]
        optim[tag] = torch.optim.AdamW(groups,lr=.005)
    parameters = {m:sum(p.numel() for p in n.parameters()) for m,n in models.items()}
    print(json.dumps(dict(stage='configuration',states=args.train_states,epochs=args.epochs,parameters=parameters,
                          trajectory_counts={k:len(chosen[k]) for k in splits},quota=args.quota,
                          gpu=torch.cuda.get_device_name(),source_pilot=str(args.pilot),shell_lr=shell_lr)),flush=True)
    global_started = perf_counter()
    cls = AngularCase if tags==('baseline',) else NearCase
    extra = [] if cls is AngularCase else [args.quota]
    panel = {p:[cls(entries[c],rows,device,*extra).move('cpu')
               for c,rows in chosen[p].items()] for p in splits}
    torch.cuda.empty_cache()
    best,history = {},[]
    rng = np.random.default_rng(617)
    for epoch in range(1,args.epochs+1):
        started = perf_counter()
        seen = 0
        for j in rng.permutation(len(panel['train'])):
            case = panel['train'][j].move(device)
            for i in rng.permutation(len(case.rows)):
                state = load_state(case,i)
                p,q = case.t1[i],case.t2[i]
                log = {}
                for tag,model in models.items():
                    y1,y2 = prediction(model,case,state,True)
                    l1 = projected_loss(y1,p,p,1.,'state')
                    l2 = projected_loss(y2,q,q,1.,'state')
                    optim[tag].zero_grad(set_to_none=True)
                    bounded_backward(l2,l1,None,model)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                    optim[tag].step()
                    if seen==0:
                        log[tag] = [float(l1.detach()),float(l2.detach())]
                    del y1,y2,l1,l2
                seen += 1
                del state
                if seen==1 or seen%250==0:
                    torch.cuda.synchronize()
                    print(json.dumps(dict(stage='progress',epoch=epoch,states=seen,first_step_loss=log,
                                          seconds=perf_counter()-started,
                                          peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)),flush=True)
            del p,q
            case.move('cpu')
        scores = evaluate(models,panel['validation_mid'],device)
        record = dict(stage='epoch',epoch=epoch,seconds=perf_counter()-started,validation=compact(scores))
        for tag,score in scores.items():
            grade = objective(score)
            if tag not in best or grade<best[tag][0]:
                best[tag] = grade,epoch,{k:v.detach().clone() for k,v in models[tag].state_dict().items()}
                torch.save(best[tag][2],args.out/f'{tag}_epoch_{epoch:02d}.pt')
        history.append(record)
        print(json.dumps(record,allow_nan=False),flush=True)
        if len(tags)==1:
            tag, = tags
            torch.save(dict(model=models[tag].state_dict(),optimizer=optim[tag].state_dict(),
                            epoch=epoch,variant=tag,generator=rng.bit_generator.state,
                            train_states=args.train_states),args.out/f'resume_epoch_{epoch:02d}.pt')
    for tag,(_,_,weights) in best.items():
        models[tag].load_state_dict(weights)
        torch.save(weights,args.out/f'{tag}.pt')
    winner = min(best,key=lambda tag:best[tag][0])
    # Holdout evaluation occurs after selecting a configuration with independent validation.
    scores = {key:evaluate(models,panel[key],device) for key in ('validation_mid','test','train')}
    result = dict(selected_model=winner,selection_metric='relative_c2_rmse_squared_plus_0.1_relative_c1_rmse_squared',
                  train_states=args.train_states,epochs=args.epochs,parameters=parameters,history=history,shell_lr=shell_lr,
                  best_epochs={k:v[1] for k,v in best.items()},scores=scores,
                  seconds=perf_counter()-global_started,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
    target = args.out/'summary.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    temp.replace(target)
    print(json.dumps(dict(stage='complete',selected_model=winner,best_epochs=result['best_epochs'],
                          independent=compact(scores['validation_mid']),holdout=compact(scores['test']))),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--train-states',type=int,default=192)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--model',choices=('compare','baseline',*ORDERS),default='compare')
    parser.add_argument('--pilot',type=Path)
    parser.add_argument('--quota',type=int,default=96)
    parser.add_argument('--shell-lr',type=float,default=.005)
    main(parser.parse_args())
