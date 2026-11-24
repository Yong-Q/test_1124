"""Fit independent coefficients of one scalar pair functional with fixed radial mixing."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.xi_all_projection_loss import train as driver
from experiments.xi_all_projection_loss.train import emit
from experiments.xi_continuous_pair_mlp.direct import DirectPair, DirectPairSource
from experiments.xi_conv3d_152_sampled_c2.projection import stratified_probes
from experiments.xi_conv3d_full_radial.network import ORDERS, SCALARS
from experiments.xi_direct_pair_frame_cache.cache import FrameCache
from experiments.xi_direct_pair_frame_cache.train import batch
from portable_io import ResponseWeights


def basis_operators(model, cache):
    radial = model.spatial[:SCALARS]
    angular = model.spatial[SCALARS:].reshape(len(ORDERS), model.rank, -1)
    i, j = model.triangle
    left = torch.cat((radial, angular[:, i].flatten(0, 1)))
    right = torch.cat((radial, angular[:, j].flatten(0, 1)))
    short = cache.blend(left, right, cache.short_i, cache.short_j)
    short[SCALARS:] *= 2
    short_terms = torch.einsum('ck,ckij->cij', short, cache.short_grams)
    long = cache.blend(model.long_mix, model.long_mix, cache.long_i, cache.long_j)
    long_terms = torch.einsum('ck,ckij->cij', long, cache.long_grams)
    terms = torch.cat((short_terms[:SCALARS], long_terms, short_terms[SCALARS:]))
    reference = torch.cat((-(radial@cache.short_dc)[:, None]*cache.short_mass,
                           -(model.long_mix@cache.long_dc)[:, None]*cache.long_mass))
    # Total and per-element atom counts are conserved. Centered long-range
    # densities also integrate to zero. Their affine terms are gauge modes.
    reference[:5].zero_()
    reference[SCALARS:].zero_()
    return terms, reference


def features(bank, reference, coeff, indices):
    weights, directions = coeff[:, :, 0], coeff[:, :, 1:]
    weights = torch.cat((weights, weights.sum(1, keepdim=True)), 1)
    directions = torch.cat((directions, directions.sum(1, keepdim=True)), 1)
    grad = torch.einsum('bf,kft->btk', weights, bank)
    first = -(directions.mT @ torch.cat((grad, reference.T.expand(len(coeff), -1, -1)), -1))
    selected = directions.gather(2, indices[:, None, :].expand(-1, directions.shape[1], -1))
    curve = -(selected[:, None].mT @ (bank[None] @ selected[:, None])).movedim(1, -1)
    second = torch.cat((curve, curve.new_zeros((*curve.shape[:-1], len(reference)))), -1)
    return first, second


def positive_kernel_fit(gram, rhs, scalar_count, angular_count, reference_count):
    """Parameterize each angular quadratic as a nonnegative sum of four squares."""
    from scipy.optimize import lsq_linear

    blocks = angular_count // 3
    positive_count = scalar_count + blocks*4
    transform = np.zeros((scalar_count+angular_count+reference_count,positive_count+reference_count))
    transform[:scalar_count,:scalar_count] = np.eye(scalar_count)
    directions = np.array([[1.,0.],[0.,1.],[2**-.5,2**-.5],[2**-.5,-2**-.5]])
    square = np.stack((.5*directions[:,0]**2,directions[:,0]*directions[:,1],.5*directions[:,1]**2))
    for block in range(blocks):
        r,c = scalar_count+3*block,scalar_count+4*block
        transform[r:r+3,c:c+4] = square
    transform[-reference_count:,-reference_count:] = np.eye(reference_count)

    reduced = transform.T @ gram @ transform
    targets = transform.T @ rhs
    scale = np.sqrt(np.maximum(np.diag(reduced),1.e-30))
    normalized = reduced / np.outer(scale,scale)
    eigen, vectors = np.linalg.eigh((normalized+normalized.T)*.5)
    keep = eigen > eigen.max()*1.e-9
    design = eigen[keep,None]**.5*vectors[:,keep].T
    observations = (vectors[:,keep].T @ (targets/scale))/eigen[keep]**.5
    lower = np.r_[np.zeros(positive_count),np.full(reference_count,-np.inf)]
    # A small ridge in normalized coordinates removes unresolved directions
    # without altering the nonnegative-square constraint.
    design = np.vstack((design, np.sqrt(1.e-8)*np.eye(len(scale))))
    observations = np.r_[observations, np.zeros(len(scale))]
    fit = lsq_linear(design, observations, bounds=(lower, np.inf), method='bvls', tol=1.e-9)
    if not fit.success:
        raise RuntimeError(f'Positive quadratic fit failed: {fit.message}')
    coefficients = fit.x/scale
    if np.any(coefficients[:positive_count] < -1.e-10):
        raise ValueError('A squared kernel term has a negative weight')
    result = transform @ coefficients
    return result, int(keep.sum()), coefficients[:positive_count]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference_run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--positive-kernel', action='store_true')
    args = parser.parse_args()
    config = json.loads((args.reference_run/'summary.json').read_text())['configuration']
    sample_rows = json.loads((args.reference_run/'selected_rows.json').read_text())
    splits = driver.dataset(json.loads(Path(config['manifest']).read_text()), smoke=False)
    selection = {case: np.asarray(rows) for split in sample_rows.values() for case, rows in split.items()}
    scales = config['scales']
    free,total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(free, 20*2**30)/total)
    torch.manual_seed(config['seed'])
    model = DirectPair().cuda()
    sizes = [p.numel() for p in (model.scalar_pair, model.angular_pair, model.reference)]
    size = sum(sizes)
    nstates = sum(map(len,sample_rows['train'].values()))
    probes = config['probes']
    gram = torch.zeros((size,size),dtype=torch.float64,device='cuda')
    rhs = torch.zeros(size,dtype=torch.float64,device='cuda')
    args.output.mkdir()

    for case in splits['train']:
        ids = selection[case['case']]
        with driver.case_labels(case) as data:
            source = DirectPairSource(case,data,config['charges'],torch.device('cuda'),8)
            cache = FrameCache.build(model, source)
            bank, reference = basis_operators(model, cache)
            response = ResponseWeights(data)
            labels = tuple(data[key] for key in driver.KEYS)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
            f'{config["seed"]}:c2:{case["case"]}'.encode()).digest()[:8], 'little'))
        for start in range(0,len(ids),16):
            rows = ids[start:start+16]
            directions = response.select(rows,np.broadcast_to(np.arange(probes),(len(rows),probes)))
            coeff = torch.as_tensor(np.concatenate((response.weights[rows,:,None], directions),-1),
                                    dtype=torch.float32,device='cuda')
            columns, inverse = map(np.stack,zip(*(stratified_probes(rng,24) for _ in rows)))
            first, second = features(bank, reference, coeff, torch.as_tensor(columns,device='cuda'))
            if start == 0:
                initial = torch.cat((model.scalar_pair, model.angular_pair, model.reference))
                predicted = cache.project(cache.operator(model), coeff, torch.as_tensor(columns, device='cuda'))
                for values, expected in zip((first, second), predicted):
                    torch.testing.assert_close(values @ initial, expected, atol=1.e-4, rtol=2.e-3)
            y1 = labels[0][rows]
            y2 = labels[1][rows[:,None,None],columns[:,:,None],columns[:,None,:]]
            weights = (1/np.sqrt(nstates*probes), np.sqrt(inverse)/(np.sqrt(nstates)*probes))
            for x, y, w, scale in zip((first,second),(y1,y2),weights,scales):
                multiplier = torch.as_tensor(w/scale,device='cuda',dtype=torch.float64)
                design = (x.double()*multiplier[...,None]).reshape(-1,size)
                target = (torch.as_tensor(y,device='cuda',dtype=torch.float64)*multiplier).flatten()
                gram += design.mT @ design
                rhs += design.mT @ target
        emit(stage='fit_case',case=case['case'],states=len(ids))
        del source, cache, bank, reference

    positive_weights = None
    if args.positive_kernel:
        normal, target = gram.cpu().numpy(), rhs.cpu().numpy()
        np.savez(args.output/'normal_equations.npz', gram=normal, rhs=target)
        coefficients, rank, positive_weights = positive_kernel_fit(normal,target,*sizes)
    else:
        norms = gram.diag().clamp_min(1.e-30).sqrt().cpu().numpy()
        normal = gram.cpu().numpy() / np.outer(norms,norms)
        solution, _, rank, _ = np.linalg.lstsq(normal, rhs.cpu().numpy()/norms,rcond=1.e-9)
        coefficients = solution/norms
    for parameter, values in zip((model.scalar_pair,model.angular_pair,model.reference),
                                 np.split(coefficients,np.cumsum(sizes)[:-1])):
        parameter.copy_(torch.as_tensor(values,dtype=parameter.dtype,device=parameter.device))
    emit(stage='fit_complete', states=nstates, coefficients=size, rank=int(rank),positive_kernel=args.positive_kernel)
    settings = SimpleNamespace(batch=16,microbatch=16,block=config['block'])
    metrics = {role: driver.run_split(model,splits[role],role,probes,config['charges'],scales,settings,
                                       selections=selection,optimizer=None,epoch=1,directory=args.output,
                                       source_factory=DirectPairSource,batch_fn=batch)
               for role in ('validation','test')}
    torch.save(model.state_dict(),args.output/'trained.pt')
    pending = args.output/'summary.pending'
    pending.write_text(json.dumps(dict(fitted_states=nstates,coefficients=size,rank=int(rank),metrics=metrics,
                                      positive_kernel=args.positive_kernel,
                                      square_weights=None if positive_weights is None else positive_weights.tolist()),
                                  indent=2,allow_nan=False)+'\n')
    pending.rename(args.output/'summary.json')
    emit(stage='complete',summary=str(args.output/'summary.json'))


if __name__=='__main__':
    main()
