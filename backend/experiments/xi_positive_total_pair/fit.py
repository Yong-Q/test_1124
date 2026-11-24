"""Fit total curvature with a positive base and a bounded environmental correction."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
import torch

from experiments.xi_all_projection_loss.train import KEYS, Metrics, case_labels, dataset, emit
from experiments.xi_continuous_pair_mlp.direct import DirectPairSource
from experiments.xi_conv3d_152_sampled_c2.projection import stratified_probes
from experiments.xi_direct_pair_frame_cache.cache import FrameCache
from experiments.xi_direct_pair_frame_cache.fit_kernel_coefficients import basis_operators, features
from experiments.xi_positive_total_pair.operators import (
    CORRECTION_BOUND, IDEAL_FLOOR, NEURONS, EnvironmentBasis, IdealProjection, StableTotalPair, positive_transform,
)
from portable_io import ResponseWeights


def solve(gram, rhs, sizes, environment, channels=(0,)):
    transform, count = positive_transform(sizes,environment,channels)
    g, b = transform.T @ gram @ transform,transform.T @ rhs
    # Scale a group by its representative sensitivity. Keep a ridge on weak
    # coefficients so an unobserved channel cannot acquire a large weight.
    diagonal = np.diag(g)
    standard = np.sqrt(max(np.max(diagonal),1.e-30))
    scale = np.sqrt(np.maximum(diagonal,(standard*1.e-3)**2))
    normal = g/np.outer(scale,scale)+1.e-4*np.eye(len(scale))
    matrix = np.linalg.cholesky((normal+normal.T)*.5).T
    target = np.linalg.solve(matrix.T,b/scale)
    lower = np.r_[np.zeros(count),np.full(len(scale)-count,-np.inf)]
    fit = lsq_linear(matrix,target,bounds=(lower,np.inf),method='bvls',tol=1.e-9)
    if not fit.success:
        raise RuntimeError(f'Positive total-energy fit failed: {fit.message}')
    fitted = transform @ (fit.x/scale)
    used = np.abs(fitted[sum(sizes):]).reshape(len(channels),len(NEURONS)).sum(1)
    if np.any(used > CORRECTION_BOUND*fitted[list(channels)]+1.e-9):
        raise ValueError('Environment curvature budget was exceeded')
    return fitted


def curvature_ratio(predicted, target, columns):
    p = -predicted[columns][:,columns]
    t = -target[columns][:,columns]
    eig,v = torch.linalg.eigh((t+t.mT)*.5)
    keep = eig > eig[-1]*1.e-8
    q = v[:,keep]/eig[keep].sqrt()
    matrix = q.mT @ p @ q
    values = torch.linalg.eigvalsh((matrix+matrix.mT)*.5)
    return np.array([float(values[0]),float(values[-1]),int((values < -1.e-5).sum())])


def three_point(model, species_id):
    """Change one density and record the mixed curvature at two other sites."""
    fields = torch.zeros((9**3,1,129),device='cuda',dtype=torch.float64)
    fields[...,0] = .15
    species = torch.tensor([species_id],device='cuda')
    sites = torch.tensor([4*81+4*9+i for i in (4,3,5)],device='cuda')

    def energy(delta):
        change = fields.new_zeros(len(fields)).index_add(0,sites,delta)
        altered = torch.cat((fields[...,:1]+change[:,None,None],fields[...,1:]),-1)
        return model.total_density(altered,species).sum()*.125

    before,after = (float(torch.autograd.functional.hessian(energy,fields.new_tensor([a,0,0]))[1,2])
                    for a in (0.,.1))
    return dict(before=before,after=after,change=after-before)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference_run',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--channels',nargs='+',type=int,choices=(0,1,2,3,4,13,14,15,16),default=(0,))
    parser.add_argument('--train-microbatch',type=int,default=4)
    args = parser.parse_args()
    config = json.loads((args.reference_run/'summary.json').read_text())['configuration']
    raw = Path(config['manifest']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config['manifest_sha256']:
        raise ValueError('The manifest differs from the reference run')
    splits = dataset(json.loads(raw),smoke=False)
    rows_by_split = json.loads((args.reference_run/'selected_rows.json').read_text())
    probes = config['probes']
    free,total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction((free-2*2**30)/total)
    torch.manual_seed(config['seed'])
    if len(set(args.channels)) != len(args.channels):
        parser.error('Choose each environment channel only once')
    if args.train_microbatch < 1:
        parser.error('The training microbatch must be positive')
    model = StableTotalPair(channels=args.channels).cuda()
    parameters = (model.scalar_pair,model.angular_pair,model.reference,model.environment_weights)
    sizes = [v.numel() for v in parameters]
    width = sum(sizes)
    grams = [torch.zeros((width,width),dtype=torch.float64,device='cuda') for _ in KEYS]
    rhs = [torch.zeros(width,dtype=torch.float64,device='cuda') for _ in KEYS]
    square = np.zeros(2)
    counts = np.zeros(2)
    rng = np.random.default_rng(config['seed'])
    args.output.mkdir()

    def batches(role):
        for case in splits[role]:
            if case['case'] not in rows_by_split[role]:
                continue
            rows = np.asarray(rows_by_split[role][case['case']])
            with case_labels(case) as data:
                source = DirectPairSource(case,data,config['charges'],torch.device('cuda'),8)
                cache = FrameCache.build(model,source)
                bank,reference = basis_operators(model,cache)
                ideal = IdealProjection(cache.occupancy.counts,source.dv)
                del cache
                torch.cuda.empty_cache()
                environment = EnvironmentBasis(model,source)
                response = ResponseWeights(data)
                labels = [data[k] for k in KEYS]
            microbatch = args.train_microbatch if role == 'train' else 4
            for offset in range(0,len(rows),microbatch):
                ids = rows[offset:offset+microbatch]
                directions = response.select(ids,np.broadcast_to(np.arange(probes),(len(ids),probes)))
                coeff = torch.as_tensor(np.concatenate((response.weights[ids,:,None],directions),-1),
                                        dtype=torch.float64,device='cuda')
                if role == 'train':
                    columns,inverse = map(np.stack,zip(*(stratified_probes(rng,24) for _ in ids)))
                else:
                    columns = np.broadcast_to(np.arange(probes),(len(ids),probes)).copy()
                    inverse = None
                index = torch.as_tensor(columns,device='cuda')
                selected = coeff[:,:,1:].gather(2,index[:,None,:].expand(-1,coeff.shape[1],-1))
                id_proj = ideal.project(coeff, selected)
                pair = features(bank,reference,coeff.float(),index)
                local = environment.project(coeff,selected)
                design = tuple(torch.cat((x.double(),y),-1) for x,y in zip(pair,local))
                target = (labels[0][ids],labels[1][ids[:,None,None],columns[:,:,None],columns[:,None,:]])
                target = tuple(torch.as_tensor(t,dtype=torch.float64,device='cuda') for t in target)
                yield case, design, id_proj, target, inverse
            emit(stage='case_ready',split=role,case=case['case'],states=len(rows),environment_storage=environment.storage)
            del source,bank,reference,ideal,environment

    for _,design,ideal,target,inverse in batches('train'):
        weights = (1.,np.sqrt(inverse))
        for i,(x,h,y,w) in enumerate(zip(design,ideal,target,weights)):
            scale = torch.as_tensor(w,dtype=torch.float64,device='cuda')
            x = (x*scale[...,None]).reshape(-1,width)
            total_target = (y+h)*scale
            shifted = (total_target-IDEAL_FLOOR*h*scale).flatten()
            grams[i] += x.mT @ x
            rhs[i] += x.mT @ shifted
            square[i] += float(total_target.square().sum())
            counts[i] += np.square(np.broadcast_to(w,h.shape)).sum()
    scales_total = np.sqrt(square/counts).tolist()
    g = sum(gram.cpu().numpy()/scale for gram,scale in zip(grams,square))
    h = sum(v.cpu().numpy()/scale for v,scale in zip(rhs,square))
    fits = {name: solve(g,h,sizes[:-1],flag,args.channels) for name,flag in (('base',False),('bounded_mlp',True))}
    predictor = {name:torch.as_tensor(w,dtype=torch.float64,device='cuda') for name,w in fits.items()}
    emit(stage='fit_complete', total_rms=scales_total, ideal_floor=IDEAL_FLOOR,
         correction_bound=CORRECTION_BOUND, correction=fits['bounded_mlp'][-sizes[-1]:].tolist(),
         environment_channels=args.channels, corrected_channel_weights=fits['bounded_mlp'][list(args.channels)].tolist())
    (args.output/'model_config.json').write_text(json.dumps({'environment_channels':args.channels,
                                                              'ideal_floor':IDEAL_FLOOR},indent=2)+'\n')
    for name,w in fits.items():
        for p,v in zip(parameters,np.split(w,np.cumsum(sizes)[:-1])):
            p.copy_(torch.as_tensor(v,device=p.device,dtype=p.dtype))
        pending = args.output/f'{name}.pending'
        torch.save(model.state_dict(),pending)
        pending.rename(args.output/f'{name}.pt')

    results, curvatures = {}, {name:[] for name in fits}
    for split in ('validation','test'):
        if not rows_by_split[split]:
            continue
        meters = {name:{k:Metrics(probes) for k in ('total','excess')} for name in fits}
        for case,x,ideal,target,_ in batches(split):
            truth = tuple(y+h for y,h in zip(target,ideal))
            for name,w in predictor.items():
                total = tuple(z@w+IDEAL_FLOOR*h for z,h in zip(x,ideal))
                excess = tuple(y-h for y,h in zip(total,ideal))
                meters[name]['total'].update(tuple(y.cpu().numpy() for y in truth),total)
                meters[name]['excess'].update(tuple(y.cpu().numpy() for y in target),excess)
                if split=='test':
                    columns,_ = stratified_probes(np.random.default_rng(617),24)
                    selected = torch.as_tensor(columns,device='cuda')
                    curvatures[name].append(curvature_ratio(total[1][0],truth[1][0],selected))
        results[split] = {name:{k:metric.summary(scales_total if k=='total' else config['scales'])
                                for k,metric in values.items()} for name,values in meters.items()}
    point_response = {}
    for name,w in fits.items():
        dense = StableTotalPair((9,9,9),(.5,.5,.5),channels=args.channels).cuda().double()
        dense.load_state_dict(torch.load(args.output/f'{name}.pt',map_location='cpu',weights_only=True))
        point_response[name] = {species:three_point(dense,i) for i,species in ((0,'C'),(1,'H'),(3,'O'))}
        del dense
    curvature = {name:{'states':len(values),'minimum_ratio':float(np.min(np.asarray(values)[:,0])),
                       'maximum_ratio':float(np.max(np.asarray(values)[:,1])),
                       'states_with_negative_modes':int(np.count_nonzero(np.asarray(values)[:,2]))}
                 for name,values in curvatures.items()}
    pending = args.output/'summary.pending'
    pending.write_text(json.dumps(dict(metrics=results,curvature=curvature,ideal_floor=IDEAL_FLOOR,
                                      correction_bound=CORRECTION_BOUND,environment_channels=args.channels,
                                      three_point=point_response),indent=2,allow_nan=False)+'\n')
    pending.rename(args.output/'summary.json')
    emit(stage='complete',summary=str(args.output/'summary.json'),curvature=curvature)


if __name__=='__main__':
    main()
