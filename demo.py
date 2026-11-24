"""Direct CPU/CUDA training and derivative checks on compact preprocessed examples."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.xi_conv3d_152_sampled_c2.projection import sampled_pass, stratified_probes
from experiments.xi_conv3d_local_el.train import LocalOccupancy
from experiments.xi_conv3d_multiscale.network import (
    LONG_CUTOFF_A, LONG_WIDTHS_A, SHORT_CUTOFF_A, SHORT_WIDTHS_A, radial_bank,
)
from experiments.xi_positive_total_pair.fit import solve
from experiments.xi_positive_total_pair.operators import IDEAL_FLOOR, NEURONS
from experiments.xi_sparse_xi_neighbors.source import radial_stencil
from experiments.xi_total_supervised_50.train import TotalOccupancy
from models.cnn3d.model import components as cnn
from models.neighborhood_mlp.model import components as mlp
from models.stable_kernel.model import CHANNELS, make_model
from prepared import load, references


def bank(shape, spacing, long, sparse):
    widths,cutoff,radii=(LONG_WIDTHS_A,LONG_CUTOFF_A,(4.5,6.5)) if long else (
                              SHORT_WIDTHS_A,SHORT_CUTOFF_A,(.65,2.5))
    if not sparse:
        return radial_bank(shape,spacing,widths,cutoff,long)
    result=np.zeros((len(widths),*shape))
    offsets=np.stack(np.meshgrid(*([np.arange(-1,2)]*3),indexing='ij'),-1).reshape(-1,3)
    for steps,values in radial_stencil(np.asarray(spacing),long,radii,widths,cutoff):
        location=tuple((offsets*steps%shape).T)
        for grid,value in zip(result,values):
            np.add.at(grid,location,value.ravel())
    return torch.from_numpy(result)


class Fields:
    def __init__(self,data,info,device,sparse,total):
        self.data,self.local_case,self.device,self.sparse=data,info['case'],device,sparse
        self.shape=tuple(int(x) for x in data['crop_shape'])
        self.spacing=tuple(float(x) for x in data['spacing_A'])
        self.dv=float(data['voxel_volume_A3'])
        self.frame=self.tensor(data['xi_frames'])
        counts=self.frame[...,129:].flatten(1).T*self.dv
        self.local_occupancy=(TotalOccupancy if total else LocalOccupancy)(counts.to_sparse_coo())
        self.transfers={}
        self.bulk_transfers={}

    def tensor(self,value):
        return torch.as_tensor(value,dtype=torch.float32,device=self.device)

    def transfer(self, long):
        if long not in self.transfers:
            basis=bank(self.shape,self.spacing,long,self.sparse).to(self.frame)
            self.transfers[long]=torch.fft.rfftn(basis,dim=(-3,-2,-1)).real
        return self.transfers[long]

    def filtered(self,model,coeff):
        raw=torch.einsum('btp,tvc->bpvc',coeff,self.frame)
        def convolve(value,mix,is_long):
            b,p,n,c=value.shape
            field=value.movedim(-1,2).reshape(b,p,c,*self.shape)
            fft=torch.fft.rfftn(field,dim=(-3,-2,-1))
            basis=self.transfer(is_long)
            gain=(mix@basis.flatten(1)).reshape(c,*basis.shape[-3:])
            result=torch.fft.irfftn(fft*gain,s=self.shape,dim=(-3,-2,-1))
            return result.flatten(-3).permute(0,3,1,2)
        density=raw[...,129:]
        short=convolve(model.encode(raw[...,:129],density),model.short_weights(),False)
        if not model.multiscale:
            return short
        long=convolve(density-density.mean(-2,keepdim=True),model.long_mix,True)
        return torch.cat((short,long),-1)

    def chunks(self,model,coeff):
        values=self.filtered(model,coeff)
        halo=len(model.layers)-1
        if halo==0:
            yield values,self.shape
            return
        parts=[torch.arange(-halo,n+halo,device=self.device)%n for n in self.shape]
        x,y,z=torch.meshgrid(*parts,indexing='ij')
        _,ny,nz=self.shape
        sites=((x*ny+y)*nz+z).flatten()
        yield values.index_select(1,sites),tuple(n+2*halo for n in self.shape)

    def bulk(self):
        group=self.data
        wave=group['bulk_wavevector_inv_A']
        shape=tuple(int(x) for x in group['source_grid_shape'])
        indices=group['bulk_mode_indices']
        selection=tuple((indices[:,d]%n) for d,n in enumerate(shape))
        if not self.bulk_transfers:
            for long in (False,True):
                basis=bank(shape,self.spacing,long,self.sparse)
                self.bulk_transfers[long]=torch.fft.rfftn(basis,dim=(-3,-2,-1)).real[(slice(None),*selection)].T.to(self.frame)
        keep=group['bulk_reliable']
        inverse=np.divide(1.,group['bulk_eigenvalues'],where=keep,
                          out=np.zeros_like(group['bulk_eigenvalues']))
        return SimpleNamespace(rho=self.tensor(group['bulk_rho_element']),
            inputs=self.tensor(group['bulk_inputs']),
            shells=self.tensor(np.column_stack((np.ones(len(wave)),np.cos(wave*np.array(self.spacing)).mean(-1)))),
            targets=torch.diag_embed(self.tensor(inverse)),
            basis=torch.as_tensor(group['bulk_eigenvectors'],dtype=torch.complex64,device=self.device),
            pair_mask=torch.as_tensor(keep[:,:,None]&keep[:,None,:],device=self.device),
            short_modes=self.bulk_transfers[False],long_modes=self.bulk_transfers[True])


def example(case,states,row,config,device):
    data,ref,info=load(case,states,row)
    targets,dw=references(data,data['weights'],data['beta_external'])
    total=config['output']=='intrinsic'
    source=Fields(data,info,device,config['variant']=='sparse',total)
    coeff=source.tensor(np.column_stack((data['weights'],dw)))[None]
    chosen=tuple(source.tensor(ref[key])[None] for key in (
        ('y1_intrinsic','y2_intrinsic') if total else ('c1_excess','c2_excess')))
    return source,coeff,chosen,info


def stable_design(model,source,coeff,indices):
    with torch.no_grad():
        values=source.filtered(model,coeff)[0]
        mean,dirs=values[:,0],values[:,1:]
        s,a,l=model.blocks(mean)
        ds,da,dl=model.blocks(dirs)
        x=torch.cat((s,l),-1)
        dx=torch.cat((ds,dl),-1)
        first=[-(x[:,None,:]*dx).sum(0),dx.sum(0)]
        d=dx[:,indices,:]
        second=[-torch.einsum('vkc,vjc->kjc',d,d),x.new_zeros(len(indices),len(indices),x.shape[-1])]
        af,ah=[],[]
        for m,t in zip(a,da):
            for i,j in ((0,0),(0,1),(1,1)):
                af.append(-(t[:,:,i]*m[:,None,j]+t[:,:,j]*m[:,None,i]).sum((0,2)))
                u,v=t[:,indices,i],t[:,indices,j]
                h=torch.einsum('vkm,vjm->kj',u,v)
                ah.append(-h-h.T)
        d1=[first[0],torch.stack(af,-1),first[1]]
        d2=[second[0],torch.stack(ah,-1),second[1]]
        env1,env2=[],[]
        for ch in CHANNELS:
            for scale,bias in NEURONS:
                p=(x[:,ch]/scale+bias).sigmoid()
                p0=x.new_tensor(bias).sigmoid()
                env1.append(-(dx[:,:,ch]*(4*scale*(p-p0))[:,None]).sum(0))
                env2.append(-torch.einsum('vk,vj,v->kj',d[:,:,ch],d[:,:,ch],4*p*(1-p)))
        d1.append(torch.stack(env1,-1))
        d2.append(torch.stack(env2,-1))
        dens=source.frame[...,129:].flatten(1)
        r=coeff[0,:,0]@dens
        dr=coeff[0,:,1:].T@dens
        active=r>0
        ideal1=-(dr[:,active]@r[active].log())*source.dv
        ideal2=-(dr[indices][:,active]/r[active])@dr[indices][:,active].T*source.dv
        return tuple(torch.cat(d,-1).double()*source.dv for d in (d1,d2)),(ideal1.double(),ideal2.double())


def stable_fit(model, source, coeff, target, pick, inverse, bounded):
    design,ideal=stable_design(model,source,coeff,pick)
    reference=(target[0][0],target[1][0][pick][:,pick])
    widths=[model.scalar_pair.numel(),model.angular_pair.numel(),model.reference.numel()]
    matrices,vectors=[],[]
    for x,ref,id_term,w in zip(design,reference,ideal,(1.,inverse)):
        w=torch.as_tensor(w,device=x.device,dtype=x.dtype).sqrt()
        a=(x*w[...,None]).reshape(-1,x.shape[-1])
        scale=(ref.double()*w).square().sum().clamp_min(1e-20)
        b=((ref.double()-IDEAL_FLOOR*id_term)*w).flatten()
        matrices.append((a.T@a/scale).cpu().numpy())
        vectors.append((a.T@b/scale).cpu().numpy())
    params=solve(sum(matrices),sum(vectors),widths,bounded,CHANNELS)
    with torch.no_grad():
        for par,val in zip((model.scalar_pair,model.angular_pair,model.reference,model.environment_weights),
                           np.split(params,np.cumsum(widths))):
            par.copy_(par.new_tensor(val))
    return params.tolist()


def score(model,source,coeff,targets,stable,pick):
    with torch.no_grad():
        if stable:
            design,ideal=stable_design(model,source,coeff,pick)
            params=torch.cat((model.scalar_pair,model.angular_pair,model.reference,model.environment_weights)).double()
            pred=[d@params+IDEAL_FLOOR*i for d,i in zip(design,ideal)]
        else:
            index=np.r_[0,np.asarray(pick)+1]
            p1,p2=model.volume_jets(*next(source.chunks(model,coeff[:,:,index])))
            # Use the full set for first-order only.
            w=coeff[:,:,0].detach().requires_grad_()
            with torch.enable_grad():
                f=model.tile_energy(*next(source.chunks(model,w[...,None])))*source.dv
                byframe,=torch.autograd.grad(f.sum(),w)
            pred=[-(coeff[0,:,1:].T@byframe[0]),p2[0]*source.dv]
        ref=[targets[0][0],targets[1][0][pick][:,pick]]
        return [float((a-b).square().mean().sqrt()) for a,b in zip(pred,ref)]


def derivative_check(model,source,coeff,stable):
    """Compare analytic two-direction projections against scalar-energy autodiff."""
    indices=np.array([0,31])
    selected=coeff[:,:,np.r_[0,indices+1]]
    if stable:
        d,id_terms=stable_design(model,source,coeff,indices)
        params=torch.cat((model.scalar_pair,model.angular_pair,model.reference,model.environment_weights)).detach().double()
        expected=(d[0][indices]@params+IDEAL_FLOOR*id_terms[0][indices],
                  d[1]@params+IDEAL_FLOOR*id_terms[1])
    else:
        with torch.no_grad():
            expected=tuple(x[0]*source.dv for x in model.volume_jets(*next(source.chunks(model,selected))))
    def energy(a):
        mixture=selected[:,:,:1]+selected[:,:,1:]@a[:,None]
        if stable:
            field=source.filtered(model,mixture)[0,:,0]
            _,ang,long=model.blocks(field)
            scalar=model.scalar_features(model.angular_invariants(field))
            value=model.energy_density(field)
            gains=model.environment_weights.reshape(len(CHANNELS),len(NEURONS))
            for index,channel in enumerate(CHANNELS):
                q=scalar[:,channel]
                for gain,(scale,bias) in zip(gains[index],NEURONS):
                    z=q/scale
                    origin=z.new_tensor(bias)
                    value=value+gain*4*scale**2*(torch.nn.functional.softplus(z+origin)
                            -torch.nn.functional.softplus(origin)-origin.sigmoid()*z)
            raw=torch.einsum('bt,tvc->bvc',mixture[:,:,0],source.frame)[0,:,129:]
            ideal=raw.clamp_min(1e-12)
            return (value.sum()+IDEAL_FLOOR*(raw*(ideal.log()-1)).sum())*source.dv
        fields,shape=next(source.chunks(model,mixture))
        return model.tile_energy(fields,shape).sum()*source.dv
    zero=coeff.new_zeros(2,requires_grad=True)
    g,=torch.autograd.grad(energy(zero),zero)
    h=torch.autograd.functional.hessian(energy,zero)
    for calc,ref in zip((-g,-h),expected):
        torch.testing.assert_close(calc.double(),ref.double(),rtol=3e-3,atol=3e-6)
    return dict(c1_max_abs=float((g+expected[0]).abs().max()),
                c2_max_abs=float((h+expected[1]).abs().max()))


def run(args,config):
    device=torch.device(args.device)
    if device.type=='cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA was requested, but no CUDA device is visible')
        free,total=torch.cuda.mem_get_info()
        budget=min(args.memory_gib*2**30,free-512*2**20)
        if budget<2**30:
            raise ValueError('At least 1 GiB of GPU memory is needed')
        torch.cuda.set_per_process_memory_fraction(min(budget/total,.95))
    torch.manual_seed(617)
    source,coeff,target,info=example(args.case,args.states,args.row,config,device)
    if info['split']!='train':
        raise ValueError('The NH3 fixture is evaluation only')
    stable=config['family']=='stable_kernel'
    if stable:
        model=make_model().to(device)
    else:
        factory=cnn if config['family']=='cnn3d' else mlp
        model_type,bulk_type,_,_=factory(config['variant'],config['output']=='intrinsic')
        model=model_type().to(device)
        bulk=source.bulk()
    pick,inverse=stratified_probes(np.random.default_rng(617),24)
    before=score(model,source,coeff,target,stable,pick)
    parameters=[p.detach().clone() for p in model.parameters()]
    if stable:
        stable_fit(model,source,coeff,target,pick,inverse,config['variant']=='bounded')
        updates=1
    else:
        def bulk_loss():
            # PairBulk inspects this flag, while the CNN subclass overrides its implementation.
            bulk.subtract_ideal_identity=(config['output']=='intrinsic')
            return bulk_type.bulk_loss(bulk,model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
        scales=[float(v.square().mean().sqrt().clamp_min(1e-12)) for v in target]
        for _ in range(args.epochs):
            optimizer.zero_grad(set_to_none=True)
            sampled_pass(model,source,coeff,target,scales,1,pick,inverse_inclusion=inverse)
            bulk_loss().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
            optimizer.step()
        updates=args.epochs
    after=score(model,source,coeff,target,stable,pick)
    held,held_coeff,held_y,_=example('nh3',None,0,config,device)
    test=score(model,held,held_coeff,held_y,stable,pick)
    check=derivative_check(model,source,coeff,stable)
    changed=any(not torch.equal(before,after) for before,after in zip(parameters,model.parameters()))
    if not changed or not np.isfinite([*before,*after,*test]).all():
        raise RuntimeError('Training failed to update parameters or produced nonfinite results')
    args.output.mkdir(parents=True)
    torch.save(model.state_dict(),args.output/'weights.pt')
    result=dict(status='passed',variant=args.variant,device=str(device),updates=updates,
                fit_rmse_before=before,fit_rmse_after=after,nh3_rmse=test,
                train_probes_c1=152,train_probes_c2=24,derivative_check=check)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
