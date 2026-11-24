"""Reference variational solver: optimize Omega[xi; v] on a realizable frame hull.

A common probability distribution over complete atomic configurations enforces
realizability, atomic identity, nonnegative density and particle conservation.
The scalar network sees only xi (and fixed external/state conditions).
Dense frame_fields are for small/reference solves; they are not a scalable
production implementation for an entire 400-frame, full-box atomic field.
"""
import json
import torch


def volumes(dv,frame_fields):
    out=torch.as_tensor(dv,dtype=frame_fields.dtype,device=frame_fields.device)
    if out.ndim==0:out=out.expand(frame_fields.shape[1])
    if out.shape!=(frame_fields.shape[1],) or not bool((out>0).all()):raise ValueError('Positive spatial integration volumes required')
    return out


def external_initial_guess(frame_fields,dv,external,reference_weights=None):
    """w0 proportional to reference w * exp(-Delta beta U), not a fitted xi."""
    vol=volumes(dv,frame_fields)
    if frame_fields.ndim!=4 or external.shape!=frame_fields.shape[1:]:raise ValueError('Frame/atomic external dimensions mismatch')
    energy=torch.einsum('tqic,qic,q->t',frame_fields,external,vol)
    ref=torch.ones_like(energy) if reference_weights is None else reference_weights.to(energy)
    if ref.shape!=energy.shape or bool((ref<0).any()) or not bool(torch.isfinite(ref).all()) or not bool(ref.sum()>0):raise ValueError('Valid reference weights required')
    return torch.softmax(torch.log(ref)-energy,0)


def simplex_project(value):
    """Euclidean projection onto nonnegative weights summing to one."""
    sorted_values=torch.sort(value,descending=True).values
    cumulative=sorted_values.cumsum(0)-1
    rank=torch.arange(1,len(value)+1,dtype=value.dtype,device=value.device)
    active=sorted_values-cumulative/rank>0
    last=torch.nonzero(active)[-1,0]
    return (value-cumulative[last]/rank[last]).clamp_min(0)


def relax(model,frame_fields,dv,external,chemical,species,initial_weights=None,max_steps=500,tolerance=1e-8):
    """Projected-gradient minimization with Armijo backtracking.

    Result is stationary within the supplied convex hull, not evidence of the
    unrestricted physical equilibrium of a trained molecular model.
    """
    if frame_fields.ndim!=4 or not bool(torch.isfinite(frame_fields).all()):raise ValueError('Finite frame_fields[T,x,i,C] required')
    frames=frame_fields.detach();vol=volumes(dv,frames)
    counts=torch.einsum('tqi,q->ti',frames[...,0],vol)
    if not bool(torch.allclose(counts,torch.ones_like(counts),atol=1e-7,rtol=1e-7)):raise ValueError('Each frame must conserve each atomic i')
    if bool((frames[...,0]<0).any()):raise ValueError('Negative number density')
    w=external_initial_guess(frames,dv,external) if initial_weights is None else simplex_project(initial_weights.to(frames))
    if w.shape!=(len(frames),):raise ValueError('Weight dimension mismatch')
    history=[];status='max_steps';gap=float('inf');gradient=None

    def evaluate(weight,need_gradient=False):
        xi=torch.einsum('t,tqic->qic',weight,frames).detach().requires_grad_(need_gradient)
        parts=model(xi,dv,external,chemical,species)
        grad=torch.autograd.grad(parts['omega'],xi)[0] if need_gradient else None
        return xi,parts,grad

    for step in range(max_steps+1):
        xi,parts,gradient=evaluate(w,True);energy=float(parts['omega'].detach());history.append(energy)
        gw=torch.einsum('tqic,qic->t',frames,gradient)
        # Simplex KKT: g_t >= lambda, equality on positive-weight frames.
        gap=float((w*gw).sum()-gw.min())
        if gap<=tolerance:status='converged';break
        if step==max_steps:break
        direction=simplex_project(w-gw)-w
        slope=float((gw*direction).sum())
        if slope>=0:status='no_descent_direction';break
        length=1.;accepted=False
        for _ in range(60):
            trial=w+length*direction
            _,trial_parts,_=evaluate(trial)
            trial_energy=float(trial_parts['omega'].detach())
            if trial_energy<=energy+1e-4*length*slope:
                w=trial.detach();accepted=True;break
            length*=.5
        if not accepted:status='line_search_failed';break
    # Return both deltaOmega/dxi (discrete derivative includes dV) and the
    # projected stationarity gap; deltaOmega/dxi need not vanish under constraints.
    return {'status':status,'weights':w,'xi':xi.detach(),'gradient_xi':gradient.detach(),
            'stationarity_gap':gap,'history':history,'steps':len(history)-1,
            'omega':float(parts['omega'].detach()),'scope':'stationary only within mixtures of the supplied complete atomic frames'}


def demo():
    from model import XiGrandPotential
    frames=torch.zeros(3,3,1,49,dtype=torch.double)
    for t in range(3):frames[t,t,0,0]=1.
    external=torch.zeros(3,1,49,dtype=torch.double);external[:,0,0]=torch.tensor([0.,.7,-.4])
    model=XiGrandPotential(radials=3,width=8).double()
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
    result=relax(model,frames,1.,external,torch.zeros(1,dtype=torch.double),torch.tensor([1]),initial_weights=torch.tensor([.8,.1,.1],dtype=torch.double))
    exact=torch.softmax(-external[:,0,0],0)
    print(json.dumps({'case':'analytic ideal atomic reference; zero excess head, not trained real fluid','status':result['status'],
                      'weights':result['weights'].tolist(),'analytic_weights':exact.tolist(),'max_abs_error':float((result['weights']-exact).abs().max()),
                      'stationarity_gap':result['stationarity_gap'],'steps':result['steps'],'omega':result['omega'],
                      'initial_omega':result['history'][0],'monotone_descent':all(b<=a+1e-12 for a,b in zip(result['history'],result['history'][1:]))}))
    if result['status']!='converged' or float((result['weights']-exact).abs().max())>1e-6:raise RuntimeError('Reference equilibrium test failed')


if __name__=='__main__':demo()
