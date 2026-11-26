"""The PPT projected fit and separate C1/C2 optimization with optional mutual supervision."""
import math
import torch

from .core.model import GridFunctional, ShortSpatial


def center(values,element):
    _,inv=torch.unique(element,sorted=True,return_inverse=True)
    count=torch.bincount(inv).to(values)
    means=values.new_zeros(len(count)).scatter_add_(0,inv,values)/count
    return values-means[inv],means


def c1_observables(case,model,training):
    y0,y1=(case.c1(model,key,training) for key in ('base','weak'))
    t0,t1,_=case.targets()
    element=case.d['grid_element']
    p0,mu0=center(t0-y0,element)
    p1,mu1=center(t1-y1,element)
    d1,_=center(y1-y0,element)
    dr,_=center(t1-t0,element)
    return (p0.square().mean()+p1.square().mean())/2,d1,dr,(y0,y1,mu0,mu1)


def c2_loss(case,prediction,train):
    ref=case.d['bulk_reference']
    weights=case.d['bulk_mode_counts'][:,None,None]
    err=prediction-ref[:,None]
    square=err[:,0]*err[:,1] if train else err.mean(1).square()
    return (weights*square).sum()/(weights*ref.square()).sum().clamp_min(1.e-12)


def gradient_add(model,loss,cap=.25,maximum=1.,protect=False,retain_graph=False):
    params=tuple(model.parameters())
    primary=[p.grad for p in params]
    aux=torch.autograd.grad(loss,params,retain_graph=retain_graph,allow_unused=True)
    g=torch.cat([(torch.zeros_like(p) if d is None else d).detach().flatten()
                 for p,d in zip(params,primary)])
    h=torch.cat([(torch.zeros_like(p) if d is None else d).detach().flatten()
                 for p,d in zip(params,aux)])
    if protect:
        h=h-(torch.dot(g,h)/g.square().sum().clamp_min(1.e-20)).clamp_max(0.)*g
    scale=(cap*g.norm()/h.norm().clamp_min(1.e-20)).clamp_max(maximum)
    for p,old,extra,b in zip(params,primary,aux,h.split([p.numel() for p in params])):
        if old is not None or extra is not None:
            p.grad=scale*b.view_as(p) if old is None else old+scale*b.view_as(p)
    return float(scale)


def model_set(mode,scale,device):
    table={'ppt':('ppt',),'c1':('c1',),'c2':('c2',),'guided':('c1','c2'),
           'mutual':('c1','c2'),'gated':('c1','c2'),'no-peer':('c1','c2')}
    models={}
    for i,key in enumerate(table[mode]):
        torch.manual_seed(617+i)
        constructor=ShortSpatial if key=='ppt' else GridFunctional
        models[key]=constructor(scale.detach().cpu().clone(),2).to(device)
    return models


def step_ppt(case,model):
    first,second=case.ppt(model,True)
    u,v=case.d['ppt_y1'],case.d['ppt_y2']
    l1=(first-u).square().sum()/u.square().sum().clamp_min(1e-30)
    l2=(second-v).square().sum()/v.square().sum().clamp_min(1e-30)
    l2.backward(retain_graph=True)
    weight=gradient_add(model,l1,maximum=.10)
    return dict(c1_loss=float(l1.detach()),c2_loss=float(l2.detach()),c1_weight=weight)


def step_grid(case,models,mode):
    stats={}
    if 'c1' in models:
        first=models['c1']
        el,a,target,_=c1_observables(case,first,True)
        el.backward(retain_graph=mode in ('guided','mutual','gated'))
        stats['c1_EL_mse']=float(el.detach())
    if 'c2' in models and mode!='guided':
        second=models['c2']
        h=c2_loss(case,case.c2(second,True),True)
        h.backward()
        stats['bulk_c2_loss']=float(h.detach())
    if mode not in ('c1','c2'):
        second=models['c2']
        _,b,_,_=c1_observables(case,second,mode!='guided')
        signal=target.square().mean().clamp_min(1.e-4)
        err1=(a-target).square().mean()/signal
        err2=(b-target).square().mean()/signal
        expected,observed=float(err1.detach()),float(err2.detach())
        total=max(expected+observed,1.e-8)
        stat1=0.
        stat2=0.
        if mode=='guided':
            guide=1./(1.+observed)
            (.1*guide*(a-b.detach()).square().mean()).backward()
            stat2=guide
        elif mode=='mutual':
            stat1=max(0.,(observed-expected)/total)/(1.+expected)
            stat2=max(0.,(expected-observed)/total)/(1.+observed)
            gradient_add(second,.05*err2+.1*stat1*(b-a.detach()).square().mean()/signal)
            gradient_add(first,.1*stat2*(a-b.detach()).square().mean())
        else:
            if mode=='gated':
                stat1=max(0.,1.-expected)*max(0.,(observed-expected)/total)
                stat2=max(0.,1.-observed)*max(0.,(expected-observed)/total)
            gradient_add(second,.05*err2,retain_graph=stat1>0)
            if stat1>0:
                gradient_add(second,.1*stat1*(b-a.detach()).square().mean()/signal,protect=True)
            if stat2>0:
                gradient_add(first,.1*stat2*(a-b.detach()).square().mean(),protect=True)
        stats.update(md_c1_relative_rmse=math.sqrt(expected),md_c2_relative_rmse=math.sqrt(observed),
                     c1_to_c2=stat1,c2_to_c1=stat2)
    return stats


def metrics(pred,ref,element=None):
    residual=pred-ref
    centered=ref-ref.mean() if element is None else center(ref,element)[0]
    if element is not None:
        residual=center(residual,element)[0]
    s2=float(centered.double().square().sum())
    e2=float(residual.double().square().sum())
    return dict(rmse=math.sqrt(e2/ref.numel()),r2=1-e2/s2 if s2>0 else None)
