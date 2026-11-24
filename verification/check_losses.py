"""Compare the report's scalar losses with the gradients used by training."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from scipy.optimize import lsq_linear
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'backend')]
from demo import example, stable_design, stable_fit
from experiments.xi_conv3d_152_sampled_c2.projection import sampled_pass, stratified_probes
from experiments.xi_conv3d_sixlayer.bulk import predicted_hessian
from experiments.xi_positive_total_pair.operators import CORRECTION_BOUND, IDEAL_FLOOR, NEURONS
from models.cnn3d.model import components as cnn
from models.neighborhood_mlp.model import components as mlp
from models.stable_kernel.model import CHANNELS, make_model


def report_probabilities(pick):
    sizes=np.array([24,8,24,40,56])
    draws=np.array([6,2,4,5,7])
    bands=(np.arange(152)[24:]).reshape(8,16)
    layer=np.zeros(152,dtype=int)
    for ell in range(4):
        layer[bands[:,ell**2:(ell+1)**2].ravel()]=ell+1
    chosen=layer[pick]
    p=draws[chosen]/sizes[chosen]
    pair=np.outer(p,p)
    for i in range(len(pick)):
        level=chosen[i]
        for j in range(len(pick)):
            if i==j:
                pair[i,j]=p[i]
            elif level==chosen[j]:
                pair[i,j]*=(1-1/draws[level])/(1-1/sizes[level])
    return 1/pair


def el_loss(source,w,g,total):
    counts=source.local_occupancy.counts.to_dense()
    w=w.reshape(-1)
    w=w/w.sum()
    mean=counts@w
    centered=counts-mean[:,None]
    second=counts.square()@w
    variance=centered.square()@w
    active=variance>8*torch.finfo(w.dtype).eps*second
    n=max(int(active.sum()),1)
    covariance=lambda value: centered[active]@(w*value)
    log=torch.where(mean>0,mean.clamp_min(torch.finfo(mean.dtype).tiny).log(),0.)
    r0=counts.T@log-w.log()
    drive=g.reshape(-1)-w.log() if total else g.reshape(-1)+r0
    scale=((covariance(r0).square()/variance[active]).sum()/n).clamp_min(1e-6)
    return (covariance(drive).square()/variance[active]).sum()/(n*scale)


def explicit_bulk(model,bulk,config):
    if config['family']=='cnn3d':
        # predicted_hessian includes the identity once for the total reference.
        value=predicted_hessian(bulk,model,bulk.short_modes,
                                bulk.long_modes if config['variant']=='dual' else None)
        if config['output']=='intrinsic':
            value=value-torch.eye(len(bulk.rho),device=bulk.rho.device)
    else:
        weights=model.spatial[:13]
        x0=(bulk.rho@bulk.inputs)*weights.sum(-1)
        short=bulk.inputs[None]*(bulk.short_modes@weights.T)[:,None,:]
        long=bulk.inputs[None,:,1:5]*(bulk.long_modes@model.long_mix.T)[:,None,:]
        d=bulk.rho.sqrt()[None,:,None]*torch.cat((short,long),dim=-1)
        def energy(x):
            z=torch.cat((x[:13],x.new_zeros(30),x[13:]))
            return model.energy_density(z[None]).sum()
        h=torch.autograd.functional.hessian(
            energy,torch.cat((x0,x0.new_zeros(4))),create_graph=True)
        value=d@h@d.mT
        if config['output']=='excess':
            value=value+torch.eye(len(bulk.rho),device=bulk.rho.device)
    chosen=bulk.basis.mH@value.to(bulk.basis.dtype)@bulk.basis
    return (chosen-bulk.targets)[bulk.pair_mask].abs().square().sum()/bulk.targets.square().sum()


def snapshot(model):
    return torch.cat([torch.zeros_like(p).flatten() if p.grad is None else p.grad.detach().flatten()
                      for p in model.parameters()])


def network(config,model,source,coeff,targets,pick,inverse,bulk_type):
    total=config['output']=='intrinsic'
    bulk=source.bulk()
    bulk.subtract_ideal_identity=total
    scales=[float(v.square().mean().sqrt().clamp_min(1e-12)) for v in targets]
    native, native_el=sampled_pass(model,source,coeff,targets,scales,1,pick,
                                   inverse_inclusion=inverse)
    native_bulk=bulk_type.bulk_loss(bulk,model)
    native_bulk.backward()
    observed=snapshot(model)
    model.zero_grad(set_to_none=True)

    w=coeff[:,:,0].detach().requires_grad_()
    energy=model.tile_energy(*next(source.chunks(model,w[...,None]))).sum()*source.dv
    g,=torch.autograd.grad(energy,w,create_graph=True)
    first=-(coeff[:,:,1:].mT@g.unsqueeze(-1)).squeeze(-1)
    _,second=model.volume_jets(*next(source.chunks(model,coeff[:,:,np.r_[0,pick+1]])))
    second=second*source.dv
    torch.testing.assert_close(first,native.c1,rtol=3e-4,atol=2e-7)
    torch.testing.assert_close(second,native.c2,rtol=3e-4,atol=2e-7)
    one=(first-targets[0]).square().mean()/scales[0]**2
    small=targets[1][:,pick][:,:,pick]
    two=((second-small).square()*source.tensor(inverse)).sum()/(152**2*scales[1]**2)
    el=el_loss(source,w.detach(),g,total)
    physical_bulk=explicit_bulk(model,bulk,config)
    torch.testing.assert_close(el,native_el,rtol=3e-5,atol=1e-6)
    torch.testing.assert_close(physical_bulk,native_bulk,rtol=3e-5,atol=1e-6)
    (one+two+el+physical_bulk).backward()
    expected=snapshot(model)
    absolute=float((expected-observed).abs().max())
    relative=float((expected-observed).norm()/expected.norm().clamp_min(1e-12))
    if not (np.isfinite(relative) and relative<3e-4):
        raise AssertionError(f'Parameter gradient disagrees: {relative:.6g}')
    names=('c1','c2','local_el','bulk')
    return dict(losses={key:float(x.detach()) for key,x in zip(names,(one,two,el,physical_bulk))},
                parameter_gradient_relative_l2=relative,parameter_gradient_max_abs=absolute)


def stable(config,model,source,coeff,targets,pick,inverse):
    design,ideal=stable_design(model,source,coeff,pick)
    target=(targets[0][0],targets[1][0][pick][:,pick])
    measured={}
    def capture(a,b,*args,**kwargs):
        fit=lsq_linear(a,b,*args,**kwargs)
        d=a@fit.x-b
        gradient=a.T@d
        active=fit.active_mask==-1
        violation=np.where(active,np.minimum(gradient,0.),gradient)
        measured.update(ridge=float(1e-4*(fit.x@fit.x)), objective_change=float(d@d-b@b),
                        kkt_absolute=float(np.linalg.norm(violation,np.inf)))
        assert fit.success and measured['kkt_absolute']<5e-6
        return fit
    with patch('experiments.xi_positive_total_pair.fit.lsq_linear',side_effect=capture):
        parameters=stable_fit(model,source,coeff,targets,pick,inverse,config['variant']=='bounded')
    params=design[0].new_tensor(parameters)
    fit_losses=[]
    initial=[]
    for d,y,i,w in zip(design,target,ideal,(1.,inverse)):
        w=d.new_tensor(w)
        denominator=(w*y.square()).sum()
        fit_losses.append(float((w*(d@params+IDEAL_FLOOR*i-y).square()).sum()/denominator))
        initial.append(float((w*(IDEAL_FLOOR*i-y).square()).sum()/denominator))
    direct=sum(fit_losses)+measured['ridge']-sum(initial)
    np.testing.assert_allclose(direct,measured['objective_change'],atol=2e-7,rtol=2e-6)
    assert direct<0.
    gains=model.environment_weights.detach().reshape(len(CHANNELS),len(NEURONS)).abs().sum(1)
    bound=CORRECTION_BOUND*model.scalar_pair.detach()[list(CHANNELS)]
    assert torch.all(gains<=bound+1e-7)
    return dict(losses=dict(c1=fit_losses[0],c2=fit_losses[1],ridge=measured['ridge']),
                objective_reduction=float(-direct),solver_kkt_violation=measured['kkt_absolute'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--variant',nargs='+')
    args=parser.parse_args()
    device=torch.device(args.device)
    torch.set_num_threads(2)
    variants=json.loads((ROOT/'variants.json').read_text())
    pick,inverse=stratified_probes(np.random.default_rng(617),24)
    np.testing.assert_allclose(inverse,report_probabilities(pick),atol=1e-12)
    for name in args.variant or variants:
        config=variants[name]
        source,coeff,targets,info=example('ch4',None,0,config,device)
        torch.manual_seed(617)
        if config['family']=='stable_kernel':
            result=stable(config,make_model().to(device),source,coeff,targets,pick,inverse)
        else:
            factory=cnn if config['family']=='cnn3d' else mlp
            model,bulk,*_=factory(config['variant'],config['output']=='intrinsic')
            result=network(config,model().to(device),source,coeff,targets,pick,inverse,bulk)
        print(json.dumps(dict(status='passed',variant=name,device=args.device,
                              case=info['case'],**result),allow_nan=False),flush=True)


if __name__=='__main__':
    main()
