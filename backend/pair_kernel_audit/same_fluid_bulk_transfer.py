"""Test new external fields on same-fluid models trained on homogeneous curvature only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re

import h5py
import numpy as np

from pair_kernel_audit.md_field_equilibrium import stencil
from pair_kernel_audit.prl_density_coupling import AxialProjection, RHO0, compare, fit, periodic_kernels
from pair_kernel_audit.run import MANIFEST
from pair_kernel_audit.zenodo_external_fields import curve


ATOMS_PER_MOLECULE = {'CH4':5,'H2O':3,'C2H4':6,'CO2':3}
LOW_K = 1.7


def bulk_rows(case):
    with h5py.File(case['cache'],'r') as h:
        position = h['positions'][:]
        lengths = np.diag(h['cell'][0])
    nframes,natom,_ = position.shape
    shape = np.ceil(lengths/.5).astype(int)
    spacing = lengths/shape
    kernels = periodic_kernels(spacing)
    rho = natom/np.prod(lengths)
    rows = []
    for axis,n in enumerate(shape):
        points,weights = stencil(position[:,:,axis],lengths[axis],n)
        keys = points+np.arange(nframes)[:,None,None]*n
        count = np.bincount(keys.ravel(),weights=weights.ravel(),minlength=nframes*n).reshape(nframes,n)
        amp = np.fft.rfft(count,axis=1)
        amp -= amp.mean(0)
        power = abs(amp)**2/natom
        blocks = np.array([samples.mean(0) for samples in np.array_split(power,10)])
        uncertainty = blocks.std(0,ddof=1)/np.sqrt(len(blocks))
        s = power.mean(0)
        k = 2*np.pi*np.arange(n//2+1)/lengths[axis]
        use = (k>0)&(s>.04)&(uncertainty<=.35*s)
        response = []
        for coords,y in kernels:
            marginal = np.bincount(coords[:,axis]%n,weights=y,minlength=n)
            response.append(np.fft.rfft(marginal).real[use])
        transfer = np.stack(response,axis=1)
        matrix = np.column_stack((np.ones(use.sum()),transfer**2,rho/RHO0*transfer**2,
                                  rho/RHO0*(1-transfer)**2,(rho/RHO0)**2*transfer**2))
        rows.append((matrix,1/s[use],k[use]))
    matrix,target,k = [np.concatenate(p) for p in zip(*rows)]
    return dict(case=case['case'],species=case['species'],density=float(rho),
                matrix=matrix,target=target,k=k)


def learn(entries,low=False):
    def data(row):
        select = row['k']<=LOW_K if low else np.ones(len(row['k']),dtype=bool)
        return row['matrix'][select],row['target'][select]
    return fit([data(row) for row in entries],[])[0]


def score_bulk(entry,coef):
    error = entry['matrix']@coef-entry['target']
    low = entry['k']<=LOW_K
    return dict(all_modes=len(error),low_modes=int(low.sum()),
                rmse=float(np.sqrt(np.mean(error**2))),
                low_k_rmse=float(np.sqrt(np.mean(error[low]**2))))


def test_profile(task):
    folder,name,config,models,output = task
    species = config['species']
    with h5py.File(folder/'reference'/name/'source.h5','r') as h:
        lengths = np.diag(h['cell'][0])
    shape = np.ceil(lengths/.5).astype(int)
    spacing = lengths/shape
    n = shape[2]
    z = (np.arange(n)+.5)*spacing[2]
    field,_ = curve(config,z)
    total = config['number_of_molecules']*ATOMS_PER_MOLECULE[species]*n/np.prod(lengths)
    op = AxialProjection(shape,spacing)
    neutral = np.full(n,total/n)
    predicted = {}
    for label,coef in models.items():
        guess,init,info = compare(op,dict(rho=neutral,field=field),coef,total_from_metadata=total)
        predicted[label] = (guess,info)
    with np.load(folder/'reference'/name/'reference_density.npz') as f:
        rho = f['density_z'].sum(1)
    np.testing.assert_allclose(rho.sum(),total,rtol=1e-12)
    scores = {}
    for label,(calc,info) in {**predicted,'external_guess':(init,{})}.items():
        delta = calc-rho
        scores[label] = dict(relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(rho)),
                             mae_atoms_per_A3=float(np.abs(delta).mean()),
                             stationary=info.get('solver_success'),
                             stationarity_gap=info.get('stationarity_gap'))
    np.savez_compressed(output/f'{name}.npz',z_A=z,md=rho,external_guess=init,
                         **{label:pair[0] for label,pair in predicted.items()})
    return name,scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--postprocess',type=Path,required=True)
    p.add_argument('--fields',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=9)
    args = p.parse_args()
    args.output.mkdir()
    layouts = json.loads(MANIFEST.read_text())['splits']
    fields = {n:x for n,x in json.loads(args.fields.read_text()).items() if x['species'] in ATOMS_PER_MOLECULE}
    temperature = {x['species']:x['temperature_K'] for x in fields.values()}

    def eligible(case):
        species = case['species']
        match = re.search(r'[_-]T(\d+(?:\.\d+)?)',case['case'])
        return species in temperature and match and abs(float(match[1])-temperature[species])<.75

    bulk_train = [x for x in layouts['train'] if eligible(x)]
    bulk_val = [x for x in layouts['validation'] if eligible(x)]
    if {x['species'] for x in bulk_train} != set(ATOMS_PER_MOLECULE):
        raise ValueError('Expected a same-temperature homogeneous training set for every species')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(bulk_rows,bulk_train))
        truth = list(pool.map(bulk_rows,bulk_val))
    shared = learn(rows)
    models = {}
    for species in ATOMS_PER_MOLECULE:
        subset = [x for x in rows if x['species']==species]
        models[species] = dict(same_fluid_low_k=learn(subset,low=True),same_fluid_all_axial_modes=learn(subset),
                               shared_all_axial_modes=shared)
        print('bulk_fit',species,len(subset),sum(len(x['target']) for x in subset),flush=True)
    bulk_validation = {x['case']:{label:score_bulk(x,coef) for label,coef in models[x['species']].items()}
                       for x in truth}
    result = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [(args.postprocess,n,c,models[c['species']],args.output) for n,c in fields.items()]
        for name,scores in pool.map(test_profile,tasks):
            result[name] = scores
            print(name,{k:(v['relative_l2'],v['stationarity_gap']) for k,v in scores.items()},flush=True)
    with (args.output/'summary.json').open('x') as out:
        json.dump(dict(status='complete',scope='fixed-N atomic total density, homogeneous positive-k curvature only',
                       reweighting=False,external_used_in_training=False,absolute_bulk_mu_available=False,
                       excluded_species=['NH3','C2H6'],
                       bulk_training={x['case']:dict(modes=len(x['target']),density=x['density']) for x in rows},
                       coefficients={s:{k:v.tolist() for k,v in c.items()} for s,c in models.items()},
                       bulk_validation=bulk_validation,external_tests=result),out,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
