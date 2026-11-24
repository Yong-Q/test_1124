"""Batched full-state training of the local xi functional."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model import InvariantStandardization, XiGrandPotential
from portable_io import ResponseWeights
from train_xi_joint import label_scales, sparse_observations


def invariants(marks):
    blocks = [marks[..., 0]]
    for ell in range(1, 4):
        a = marks[..., ell*ell:(ell+1)**2]
        blocks.append((a @ a.transpose(-1, -2)).flatten(-2))
    return torch.cat(blocks, -1)


def local_derivatives(model, fields):
    """Exact directional derivatives for F = rho E(mark / rho).

    fields has shape [state, cell-atom, 1+directions, features]. Its first
    column is the mean, the remainder are the tangent directions.
    """
    mean, tangent = fields[:, :, 0], fields[:, :, 1:]
    rho = mean[..., 0]
    drho = tangent[..., 0]
    denominator = rho.clamp_min(1e-30)
    marks = (mean[..., 1:] / denominator[..., None]).unflatten(-1, (model.radials, 16))
    dm = ((tangent[..., 1:] - drho[..., None] * marks.flatten(-2)[..., None, :])
          / denominator[..., None, None]).unflatten(-1, (model.radials, 16))
    first = [dm[..., 0]]
    second = [torch.zeros_like(dm[..., :, None, :, 0] * dm[..., None, :, :, 0])]
    for ell in range(1, 4):
        s = slice(ell*ell, (ell+1)**2)
        a, da = marks[..., s], dm[..., s]
        dgram = da @ a[..., None, :, :].transpose(-1, -2)
        d2gram = da[..., :, None, :, :] @ da[..., None, :, :, :].transpose(-1, -2)
        first.append((dgram + dgram.transpose(-1, -2)).flatten(-2))
        second.append((d2gram + d2gram.transpose(-1, -2)).flatten(-2))
    energy, grad, hess = invariants(marks), torch.cat(first,-1), torch.cat(second,-1)
    for layer in model.readout:
        if isinstance(layer, torch.nn.Linear):
            energy = layer(energy)
            grad = F.linear(grad, layer.weight)
            hess = F.linear(hess, layer.weight)
        elif isinstance(layer, InvariantStandardization):
            energy = layer(energy)
            grad = grad / layer.scale
            hess = hess / layer.scale
        elif isinstance(layer, torch.nn.Softplus):
            slope = energy.sigmoid()
            curvature = slope * (1-slope)
            hess = (slope[...,None,None,:]*hess
                    + curvature[...,None,None,:]*grad[..., :,None,:]*grad[...,None,:,:])
            grad = slope[...,None,:] * grad
            energy = layer(energy)
        else:
            raise TypeError('Unsupported energy layer')
    return (-(rho[...,None]*grad.squeeze(-1)+drho*energy),
            -rho[...,None,None]*hess.squeeze(-1))


class SparseTrajectory:
    def __init__(self, case, data, charges, device):
        self.dv = float(data['voxel_volume'])
        mode = str(data['deposition']) if 'deposition' in data else 'hard'
        group,marks,atoms,observation,mass = sparse_observations(
            case,tuple(data['shape']),data['source_steps'],charges,mode)
        multiplicity = np.bincount(group)
        singleton = multiplicity[group] == 1
        self.frames = len(data['source_steps'])
        indices = np.arange(len(group))
        # Each singleton contributes linearly; combine its stencil mass
        # with other singletons from the same frame atom before readout.
        single_mass = np.bincount(observation[singleton],weights=mass[singleton],minlength=len(marks))
        single_obs = np.flatnonzero(single_mass)
        self.single_mass = torch.as_tensor(single_mass[single_obs],dtype=torch.float32,device=device)
        self.single_frame = torch.as_tensor(single_obs//atoms, device=device)
        raw = torch.as_tensor(marks[single_obs,1:], device=device)
        self.single_marks = invariants(raw.unflatten(-1, (8,16)))
        del raw
        where = indices[~singleton]
        order = np.argsort(group[where], kind='stable')
        where = where[order]
        self.multi_frame = torch.as_tensor(observation[where]//atoms, device=device)
        used, indices = np.unique(observation[where],return_inverse=True)
        self.multi_marks = torch.as_tensor(marks[used],device=device)
        self.multi_observation = torch.as_tensor(indices,device=device)
        self.multi_mass = torch.as_tensor(mass[where],device=device)
        sizes = multiplicity[multiplicity > 1]
        self.offsets = np.r_[0, sizes.cumsum()]
        self.multi_group = torch.as_tensor(np.repeat(np.arange(len(sizes)), sizes), device=device)

    def chunks(self, coeff, block):
        for lo in range(0,len(self.offsets)-1,block):
            hi = min(lo+block,len(self.offsets)-1)
            section = slice(self.offsets[lo],self.offsets[hi])
            frame = self.multi_frame[section]
            group = self.multi_group[section]-lo
            obs = self.multi_marks[self.multi_observation[section]]
            obs = obs*self.multi_mass[section,None]
            result = obs.new_empty((coeff.shape[0],hi-lo,coeff.shape[2],obs.shape[1]))
            for k in range(coeff.shape[2]):
                weighted = (coeff[:,frame,k,None]/self.dv)*obs
                dest = weighted.new_zeros((coeff.shape[0],hi-lo,obs.shape[1]))
                dest.index_add_(1,group,weighted)
                result[:,:,k] = dest
            yield result

    def singleton_slices(self):
        for lo in range(0,len(self.single_frame),131072):
            yield slice(lo,lo+131072)


def process_batch(model, source, coeff, targets, scales, block, optimizer, *,
                  return_predictions=False, additive_jets=None):
    with torch.no_grad():
        singleton = coeff.new_zeros(source.frames)
        for s in source.singleton_slices():
            value = model.readout(source.single_marks[s]).flatten()*source.single_mass[s]
            singleton.index_add_(0,source.single_frame[s],value)
        pred1 = -torch.einsum('btp,t->bp', coeff[:,:,1:], singleton)
        p = pred1.shape[-1]
        pred2 = pred1.new_zeros((len(pred1),p,p))
        for fields in source.chunks(coeff,block):
            d1,d2 = local_derivatives(model,fields)
            pred1 += d1.sum(1)*source.dv
            pred2 += d2.sum(1)*source.dv
        if additive_jets is not None:
            weights, first, second = additive_jets
            pred1 += first @ weights
            pred2 += second @ weights
        errors = [a-b for a,b in zip((pred1,pred2),targets)]
        losses = [(e/s).square().flatten(1).mean(1) for e,s in zip(errors,scales)]
        zeros = [(t/s).square().flatten(1).mean(1) for t,s in zip(targets,scales)]
        means = [(t/s).flatten(1).mean(1) for t,s in zip(targets,scales)]
        scores = torch.stack(losses+zeros+means,dim=1).double().sum(0).cpu().numpy()
    if not np.isfinite(scores).all():
        raise FloatingPointError('Nonfinite derivative loss')
    if optimizer is not None:
        grad1,grad2 = [2*e/(s*s*e.numel()) for e,s in zip(errors,scales)]
        frame_grad = -torch.einsum('bp,btp->t', grad1, coeff[:,:,1:])
        optimizer.zero_grad(set_to_none=True)
        if additive_jets is not None:
            ((first @ weights * grad1).sum() + (second @ weights * grad2).sum()).backward()
        for s in source.singleton_slices():
            e = model.readout(source.single_marks[s]).flatten()
            (e*source.single_mass[s]*frame_grad[source.single_frame[s]]).sum().backward()
        for fields in source.chunks(coeff,block):
            d1,d2 = local_derivatives(model,fields)
            loss = ((d1*grad1[:,None]).sum()+(d2*grad2[:,None]).sum())*source.dv
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
        optimizer.step()
    if return_predictions:
        return scores, (pred1, pred2)
    return scores


def sample_probes(rng, count):
    """Cover all 48 probes and within/between-family c2 pairs across states."""
    return np.stack([rng.choice(48, 2, replace=False) for _ in range(count)])


def cases_in_order(splits):
    order = np.random.default_rng(617).permutation(len(splits['train']))
    return ([('train', splits['train'][i]) for i in order]
            + [(role, row) for role in ('validation','test') for row in splits[role]])


def save_state(path, model, optimizer, progress, identity):
    temp = path.with_suffix('.pending')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    progress=progress,identity=identity), temp)
    os.replace(temp,path)


def direct_metrics(sums, n, scales):
    mse, mean_square, mean = np.asarray(sums, dtype=np.float64).reshape(3,2)/n
    variance = mean_square-mean**2
    return {tag: dict(rmse=float(np.sqrt(mse[i])*scales[i]),
                      r2=float(1-mse[i]/variance[i]) if variance[i]>0 else None)
            for i, tag in enumerate(('c1','c2'))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--charges',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--block',type=int,default=1024)
    p.add_argument('--memory-fraction',type=float,default=.38)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--max-batches', type=int, help='Stop after saving this many batches in this invocation')
    args = p.parse_args()
    if args.batch<1 or args.block<1 or not 0<args.memory_fraction<=1:
        p.error('Positive batch/block and memory fraction in (0,1] required')
    if args.max_batches is not None and args.max_batches < 1:
        p.error('--max-batches must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('Expected a CUDA GPU')
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device=0)
    torch.manual_seed(617)
    out = Path(args.output)
    out.mkdir(exist_ok=args.resume)
    manifest = json.loads(Path(args.manifest).read_text())
    charges = json.loads(Path(args.charges).read_text())
    identity = dict(manifest=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                    charges=charges, batch=args.batch, algorithm='exact_local_jets_v2')
    splitting = manifest['splits']
    model = XiGrandPotential().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
    checkpoint = out/'checkpoint.pt'
    if args.resume:
        cached = torch.load(checkpoint,map_location=device,weights_only=False)
        if cached['identity']!=identity:
            raise ValueError('Resume configuration changed')
        model.load_state_dict(cached['model'])
        optimizer.load_state_dict(cached['optimizer'])
        progress = cached['progress']
    else:
        progress = {'case_index':0,'offset':0,'scales':label_scales(splitting['train']),
                    'sums':{role:[0.]*6 for role in splitting},
                    'test_species':{species:{'sums':[0.]*6,'count':0}
                                    for species in manifest['holdout_species']},
                    'counts':{role:0 for role in splitting}}
        save_state(checkpoint,model,optimizer,progress,identity)
    print(json.dumps(dict(status='started', batch=args.batch, memory_fraction=args.memory_fraction,
                          gpu=torch.cuda.get_device_name(), position=progress['case_index'])),flush=True)
    batches_done = 0
    for index,(role,case) in enumerate(cases_in_order(splitting)):
        if index<progress['case_index']:
            continue
        if role!='train' and not (out/'trained.pt').exists():
            torch.save(model.state_dict(),out/'trained.pt')
        rng = np.random.default_rng(4000+index)
        with np.load(case['labels']) as data:
            deposition = str(data['deposition']) if 'deposition' in data else 'hard'
            if deposition != manifest.get('deposition', deposition):
                raise ValueError('Manifest and label deposition differ for '+case['case'])
            response = ResponseWeights(data)
            weights = response.weights
            c1,c2 = data['c1_projected'],data['c2_projected']
            if len(weights) != case['states'] or c1.shape != (len(weights),48) or c2.shape != (len(weights),48,48):
                raise ValueError('Unexpected label count or shape for '+case['case'])
            source = SparseTrajectory(case,data,charges,device)
        order = rng.permutation(len(weights))
        probes = sample_probes(rng,len(weights))
        for lo in range(progress['offset'],len(weights),args.batch):
            rows = order[lo:lo+args.batch]
            pair = probes[rows]
            coeff = np.concatenate((weights[rows,:,None],
                                    response.select(rows, pair)),axis=2)
            t1 = c1[rows[:,None],pair]
            t2 = c2[rows[:,None,None],pair[:,:,None],pair[:,None,:]]
            tensor = lambda x: torch.as_tensor(x,dtype=torch.float32,device=device)
            scores = process_batch(model,source,tensor(coeff),(tensor(t1),tensor(t2)),
                                   progress['scales'],args.block,optimizer if role=='train' else None)
            progress['sums'][role] = (np.array(progress['sums'][role])+scores).tolist()
            progress['counts'][role] += len(rows)
            if role=='test':
                summary = progress['test_species'][case['species']]
                summary['sums'] = (np.array(summary['sums'])+scores).tolist()
                summary['count'] += len(rows)
            progress['offset'] = lo+len(rows)
            save_state(checkpoint,model,optimizer,progress,identity)
            batches_done += 1
            if lo==0 or (lo//args.batch)%20==0 or progress['offset']==len(weights):
                print(json.dumps(dict(split=role, case=case['case'], done_in_case=progress['offset'],
                                      total=len(weights), total_by_split=progress['counts'],
                                      losses=(scores[:4]/len(rows)).tolist(),
                                      metrics=direct_metrics(progress['sums'][role],
                                                             progress['counts'][role],progress['scales']),
                                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)),flush=True)
            if batches_done == args.max_batches:
                print(json.dumps(dict(status='stopped_after_batches', batches=batches_done,
                                      deposition=deposition, states=progress['counts'])),flush=True)
                return
        progress['case_index'] = index+1
        progress['offset'] = 0
        save_state(checkpoint,model,optimizer,progress,identity)
        del source
    expected = {key:sum(row['states'] for row in rows) for key,rows in splitting.items()}
    if expected!=progress['counts']:
        raise RuntimeError('Incomplete full-state coverage')
    result = dict(status='complete', states=expected, holdout_species=manifest['holdout_species'],
                  probes_per_state=2, loss_normalization=progress['scales'],
                  split_metrics={key:direct_metrics(value,expected[key],progress['scales'])
                                 for key,value in progress['sums'].items()},
                  test_by_species={key:direct_metrics(value['sums'],value['count'],progress['scales'])
                                   for key,value in progress['test_species'].items()})
    with (out/'summary.json').open('x') as f:
        json.dump(result,f,indent=2)


if __name__=='__main__':
    main()
