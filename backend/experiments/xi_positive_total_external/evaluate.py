"""Solve the fixed-particle axial sector on independent molecular quadrature."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, softmax
from scipy.stats import qmc
import torch

from experiments.xi_all_projection_loss.train import emit
from experiments.xi_conv3d_full_radial.network import ORDERS, SCALARS
from experiments.xi_conv3d_multiscale.network import (
    LONG_CUTOFF_A, LONG_WIDTHS_A, SHORT_CUTOFF_A, SHORT_WIDTHS_A, radial_bank,
)
from experiments.xi_positive_total_pair.operators import CORRECTION_BOUND, IDEAL_FLOOR, NEURONS, StableTotalPair
from experiments.xi_zero_center_radial.reprocess import ZeroCenteredBasis
from pair_kernel_audit.check_c2_response import FIELDS, REF
from pair_kernel_audit.md_field_equilibrium import rigid_template, stencil
from pair_kernel_audit.newton_density_trial import newton
from pair_kernel_audit.rigid_equilibrium_trial import rotations_from_unit_cube
from pair_kernel_audit.zenodo_external_fields import curve


def molecule(species):
    if species == 'CO2':
        return np.array([[0.,0.,0.],[0.,0.,1.16],[0.,0.,-1.16]]),np.array([6,8,8])
    if species == 'H2O':
        x,z = .9572*np.sin(np.deg2rad(52.26)),.9572*np.cos(np.deg2rad(52.26))
        return np.array([[0.,0.,0.],[x,0.,z],[-x,0.,z]]),np.array([8,1,1])
    return rigid_template(species)


def support(config, model):
    length = config['box_length_z_A']
    n = int(np.ceil(length/.5))
    spacing = length/n
    z = (np.arange(n)+.5)*spacing
    volume = length**2*spacing / config['number_of_molecules']
    vectors,charges = molecule(config['species'])
    element = np.array([{6:0,1:1,7:2,8:3}[int(q)] for q in charges])
    sample = qmc.Sobol(4,scramble=True,seed=20261119).random_base2(5)
    rotations = rotations_from_unit_cube(sample[:,1:])
    xyz = np.einsum('rij,aj->rai',rotations,vectors)
    k,atoms,_ = xyz.shape
    i,j = np.where(~np.eye(atoms,dtype=bool))
    dr = xyz[:,j]-xyz[:,i]
    marks = ZeroCenteredBasis().values(dr.reshape(-1,3)).reshape(k,atoms,atoms-1,128)
    marks = np.einsum('rakf,ak->raf',marks,charges[j].reshape(atoms,-1))
    onehot = np.broadcast_to(np.eye(4)[element],(k,atoms,4))
    encoded = model.encode(torch.from_numpy(np.concatenate((np.ones((k,atoms,1)),marks),-1)),
                           torch.tensor(onehot.copy())).detach().numpy()
    position = (z[:,None,None]+xyz[None,:,:,2]+spacing*(sample[None,:,0,None]-.5)).reshape(-1,atoms)
    sites,weights = stencil(position,length,n)
    count = len(position)
    design = np.zeros((n,count,model.short_channels))
    indexed = np.broadcast_to(np.arange(count)[:,None,None],sites.shape)
    packed = np.tile(encoded,(n,1,1))
    np.add.at(design,(sites.ravel(),indexed.ravel()),
              (weights[...,None]*packed[:,:,None,:]).reshape(-1,model.short_channels)/volume)
    raw = design.transpose(2,0,1)
    density = raw[1:5]
    potential,_ = curve(config,z)
    external = volume*(density.sum(0).T @ potential)

    def filtered(fields, mix, widths, cutoff, long):
        axial_bank = getattr(model, 'axial_kernel_bank', None)
        if axial_bank is None:
            bank = radial_bank((n,n,n),(spacing,)*3,widths,cutoff,long=long).sum((1,2)).numpy()
        else:
            bank = axial_bank(n, spacing, widths, cutoff, long)
        transfer = np.fft.rfft(mix @ bank,axis=1)
        return np.fft.irfft(np.fft.rfft(fields,axis=1)*transfer[...,None],n=n,axis=1)

    short = filtered(raw,model.short_weights().detach().numpy(),SHORT_WIDTHS_A,SHORT_CUTOFF_A,False)
    if not model.multiscale:
        return z,volume,density,short,external
    long = filtered(density-density.mean(1,keepdims=True),model.long_mix.detach().numpy(),LONG_WIDTHS_A,LONG_CUTOFF_A,True)
    return z,volume,density,np.concatenate((short,long)),external


class Objective:
    def __init__(self, data, model):
        _,self.volume,self.density,field,self.external = data
        self.mass = self.density.reshape(-1,len(self.external))
        scalar = np.concatenate((field[:SCALARS],field[-4:]))
        weights = model.scalar_pair.detach().numpy()
        if np.any(weights<0):
            raise ValueError('Negative scalar square weight in stable model')
        positive = weights>0.
        blocks = [scalar[positive]*np.sqrt(weights[positive,None,None])]
        offset = SCALARS
        for order, coefficients in zip(ORDERS,model.angular_pair.detach().numpy().reshape(-1,3)):
            a,b,c = coefficients
            eigen,vectors = np.linalg.eigh([[2*a,b],[b,2*c]])
            if eigen[0] < -1.e-9*max(1.,eigen[-1]):
                raise ValueError('Indefinite angular block in stable model')
            keep = eigen>0.
            channels = field[offset:offset+2*order].reshape(2,order,*field.shape[1:])
            if np.any(keep):
                mixed = np.einsum('ij,imzk->jmzk',vectors[:,keep],channels)
                blocks.append((mixed*np.sqrt(eigen[keep,None,None,None])).reshape(-1,*field.shape[1:]))
            offset += 2*order
        self.square = np.concatenate(blocks).reshape(-1,len(self.external))
        self.affine = self.external-self.volume*np.einsum('c,czn->n',model.reference.detach().numpy(),scalar)
        environment = model.environment_weights.detach().numpy().reshape(-1,len(NEURONS))
        if any(np.abs(gain).sum()>CORRECTION_BOUND*weights[channel]+1.e-6
               for channel,gain in zip(model.channels,environment)):
            raise ValueError('Environment bound exceeded')
        self.environment = [(scalar[channel],gain,scale,bias) for channel,gains in zip(model.channels,environment)
                            for gain,(scale,bias) in zip(gains,NEURONS) if gain != 0.]

    def __call__(self,w):
        rho = np.maximum(self.mass@w,0.)+1.e-15
        square = self.square@w
        log = np.log(rho)
        volume = self.volume
        energy = (self.affine@w)+volume*(.5*(square@square)+IDEAL_FLOOR*(rho@(log-1)))
        gradient = self.affine+volume*(self.square.T@square+IDEAL_FLOOR*(self.mass.T@log))
        curvatures = []
        for matrix,weight,scale,bias in self.environment:
            x = (matrix@w)/scale
            s = expit(x+bias)
            s0 = expit(bias)
            energy += 4*volume*weight*scale**2*np.sum(np.logaddexp(0.,x+bias)-np.logaddexp(0.,bias)-s0*x)
            gradient += 4*volume*weight*scale*(matrix.T@(s-s0))
            curvatures.append((matrix,4*volume*weight*s*(1-s)))

        def action(direction):
            value = volume*(self.square.T@(self.square@direction)
                            +IDEAL_FLOOR*self.mass.T@((self.mass@direction)/rho))
            for matrix,k in curvatures:
                value += matrix.T@(k*(matrix@direction))
            return value

        return float(energy),gradient,action


def refine(objective,w,steps=2500):
    """Correct active-set stalls by transferring weight along feasible pairs."""
    w = w.copy()
    for iteration in range(steps):
        energy,g,_ = objective(w)
        if w@g-g.min() <= 1.e-7:
            break
        j = int(g.argmin())
        k = int(np.argmax(np.where(w>0,g,-np.inf)))
        if j == k or g[k] <= g[j]:
            break
        def gradient(amount):
            trial = w.copy()
            trial[j] += amount
            trial[k] -= amount
            return objective(trial)[1]
        def derivative(amount):
            value = gradient(amount)
            return value[j]-value[k]
        amount = w[k]
        if derivative(amount)>0:
            amount = brentq(derivative,0.,amount,xtol=1.e-14)
        w[j] += amount
        w[k] -= amount
    energy,g,_ = objective(w)
    return w, iteration+1, float(w@g-g.min()), energy


def run_case(task):
    name,config,checkpoint,channels,output = task
    started = perf_counter()
    model = StableTotalPair(channels=channels).double()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    z,volume,density,_,external = data = support(config,model)
    objective = Objective(data,model)
    initial = softmax(-external)
    w,info = newton(objective,initial,60)
    w,iterations,gap,final = refine(objective,w)
    np.testing.assert_allclose(w.sum(),1.,atol=1.e-10)
    cal = np.einsum('ezj,j->ze',density,w)
    start = np.einsum('ezj,j->ze',density,initial)
    # The reference trajectory is first opened after solving the model.
    with np.load(REF/name/'reference_density.npz') as result:
        np.testing.assert_allclose(z,result['z_A'],rtol=1.e-7,atol=1.e-9)
        observed = result['density_z']
    np.testing.assert_allclose(cal.sum(0),observed.sum(0),rtol=1.e-7,atol=1.e-8)
    def metric(x,y):
        return dict(relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(y)),rmse=float(np.sqrt(np.mean((x-y)**2))))
    score = dict(case=name,species=config['species'],converged=gap<=1.e-7,stationarity_gap=gap,
                 objective=final,initial_objective=info['omega_initial'],steps=iterations+len(info['trace'])-1,
                 negative_curvature_events=info['negative_curvature_events'],
                 total_density=metric(cal.sum(1),observed.sum(1)),initial_density=metric(start.sum(1),observed.sum(1)),
                 element_density={e:metric(cal[:,j],observed[:,j]) for j,e in enumerate('CHNO') if observed[:,j].any()},
                 seconds=perf_counter()-started)
    np.savez_compressed(output/f'{name}.npz',z_A=z,model_density=cal,reference_density=observed,initial_density=start)
    emit(stage='case',**score)
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--case',action='append')
    parser.add_argument('--checkpoint',choices=('bounded_mlp','base'),default='bounded_mlp')
    args = parser.parse_args()
    meta = args.run/'model_config.json'
    if not meta.is_file(): meta = args.run/'summary.json'
    channels = json.loads(meta.read_text())['environment_channels']
    fields = json.loads(FIELDS.read_text())
    names = args.case or list(fields)
    if not args.case and len(names)!=17:
        raise ValueError('Expected all 17 verified independent external fields')
    checkpoint = args.run/f'{args.checkpoint}.pt'
    if not checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {checkpoint}')
    args.output.mkdir()
    tasks = [(key,fields[key],checkpoint,channels,args.output) for key in names]
    with ProcessPoolExecutor(max_workers=min(args.workers,len(tasks))) as pool:
        scores = list(pool.map(run_case,tasks))
    result = dict(model=str(checkpoint),cases=len(scores),
                  converged=sum(s['converged'] for s in scores),results=scores,
                  mean_density_relative_l2=float(np.mean([s['total_density']['relative_l2'] for s in scores])),
                  scope='fixed-N axial profiles; independent rigid orientations; reference data used after minimization')
    pending = args.output/'summary.pending'
    pending.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    pending.rename(args.output/'summary.json')
    emit(stage='complete',cases=len(scores),converged=result['converged'])


if __name__=='__main__':
    main()
