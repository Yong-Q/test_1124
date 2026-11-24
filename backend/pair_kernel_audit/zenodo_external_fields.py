"""Validate archived external-force metadata and reconstruct correctly scaled potentials."""
import argparse
import csv
import io
import json
from pathlib import Path
import tarfile

import h5py
import numpy as np

from pair_kernel_audit.md_field_equilibrium import KB, field_input, potential


def curve(config,z):
    """Return β V and the force in eV/Å from the published dimensionless parameters."""
    length = float(config['box_length_z_A'])
    value = np.zeros_like(z,dtype=float)
    derivative = np.zeros_like(value)
    if config['well'] is not None:
        well = config['well']
        x = z-length/2
        ramp = (1-well['w'])*length/2
        r = np.maximum((np.abs(x)-well['w']*length/2)/ramp,0.)
        value += well['s']*r**well['p']
        derivative += (well['s']*well['p']/ramp)*r**(well['p']-1)*np.sign(x)
    for gaussian in config['gaussians']:
        sigma = length*gaussian['sigma_frac']
        distance = z-length*gaussian['mu_frac']
        v = gaussian['h']*np.exp(-.5*(distance/sigma)**2)
        value += v
        derivative -= v*distance/sigma**2
    return value,-config['kT_eV']*derivative


def validate(archive,postprocess,originals):
    names = {x['case'] for x in json.loads((postprocess/'summary.json').read_text())['cases']}
    configs, audit, data = {},{},{}
    with tarfile.open(archive) as f:
        members = {x.name.removeprefix('./'):x for x in f.getmembers() if x.isfile()}
        provided = {n.split('/')[0] for n in members if n.endswith('/field_config.json')}
        if provided != names:
            raise ValueError(f'Mismatch between metadata and trajectories: {provided ^ names}')
        for name in sorted(names):
            config = json.load(f.extractfile(members[f'{name}/field_config.json']))
            raw = f.extractfile(members[f'{name}/Vext_z.csv']).read()
            columns = np.genfromtxt(io.BytesIO(raw),delimiter=',',names=True)
            z = columns['z_A']
            with h5py.File(postprocess/'reference'/name/'source.h5','r') as h:
                length = float(h['cell'][0,2,2])
                count = len(np.unique(h['molecule_ids'][:]))
            if config['sim_id']!=name or config['direction']!='z' or config['atom_scope']!='all atoms in the box (field depends only on z)':
                raise ValueError(f'Unexpected force scope or sim_id for {name}')
            np.testing.assert_allclose(config['kT_eV'],KB*config['temperature_K'],rtol=2e-7)
            np.testing.assert_allclose([config['box_length_z_A'],z[-1]],length,rtol=1e-8)
            if z[0]!=0 or not np.all(np.diff(z)>0):
                raise ValueError(f'Unexpected grid for {name}')
            v,force = curve(config,z)
            kt = config['kT_eV']
            archived_force = -np.gradient(kt*v,z)
            np.testing.assert_allclose(archived_force,columns['Fz_eV_per_A'],
                                       rtol=2e-6,atol=3e-8,err_msg=name)
            np.testing.assert_allclose(v*kt,columns['Vext_kT'],rtol=2e-6,atol=3e-8,err_msg=name)
            np.testing.assert_allclose(v*kt**2,columns['Vext_eV'],rtol=2e-6,atol=3e-8,err_msg=name)
            record = dict(points=len(z),csv_force_difference_max_abs=float(np.abs(archived_force-columns['Fz_eV_per_A']).max()),
                          force_analytic_vs_sampled_max_abs=float(np.abs(force-columns['Fz_eV_per_A']).max()),
                          archived_columns='Vext_kT stored eV; Vext_eV stored eV multiplied by kT')
            original = next((folder/f'{name}_in.les_md' for folder in originals
                             if (folder/f'{name}_in.les_md').exists()),None)
            if original is not None:
                old,old_density = field_input(original,length)
                compare = potential(z[:-1],length,*old)
                np.testing.assert_allclose(v[:-1],compare,atol=5e-4,rtol=5e-4,err_msg=name)
                np.testing.assert_allclose(round(old_density*length**3),count)
                record['original_lammps_max_betaV_difference'] = float(np.abs(v[:-1]-compare).max())
            configs[name] = {**config,'number_of_molecules':count}
            audit[name] = record
            data[name] = (z,v,kt*v,force)
    return configs,audit,data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--postprocess',type=Path,required=True)
    parser.add_argument('--originals',type=Path,nargs='*',default=[])
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir()
    configs,audit,data = validate(args.archive,args.postprocess,args.originals)
    with (args.output/'fields.json').open('x') as f:
        json.dump(configs,f,indent=2,allow_nan=False)
    with (args.output/'verification.json').open('x') as f:
        json.dump(dict(status='complete',record=23901522,cases=audit),f,indent=2,allow_nan=False)
    with (args.output/'corrected_curves.csv').open('x') as f:
        writer = csv.writer(f)
        writer.writerow(('sim_id','z_A','beta_V','V_eV','Fz_eV_per_A'))
        for name,columns in data.items():
            writer.writerows((name,*r) for r in zip(*columns))
    print('Verified',len(audit),'curves, including',
          sum('original_lammps_max_betaV_difference' in x for x in audit.values()),'against original LAMMPS inputs')


if __name__=='__main__':
    main()
