"""Evaluate the FFT neighborhood MLP on independent external-field profiles."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import softmax
import torch

from experiments.xi_all_projection_loss.train import emit
from experiments.xi_continuous_pair_mlp.network import PairMLP
from experiments.xi_positive_total_external.evaluate import FIELDS, REF, support
from pair_kernel_audit.newton_density_trial import newton


class AxialEnergy:
    def __init__(self, data, model):
        _, self.volume, density, fields, external = data
        self.n = len(external)
        self.mass = torch.from_numpy(density.reshape(-1, self.n).copy())
        self.fields = torch.from_numpy(fields.reshape(-1, self.n).copy())
        self.channels = fields.shape[0]
        self.external = torch.from_numpy(external)
        self.model = model

    def __call__(self, weights):
        w = torch.from_numpy(weights).requires_grad_()
        field = (self.fields @ w).reshape(self.channels, -1).T
        rho = (self.mass @ w).clamp_min(1.e-15)
        intrinsic = self.model.energy_density(field).sum()
        if not getattr(self.model, 'outputs_total', False):
            intrinsic = intrinsic + (rho * (rho.log()-1.)).sum()
        energy = self.volume * intrinsic + self.external @ w
        gradient, = torch.autograd.grad(energy, w, create_graph=True)

        def hessian_action(direction):
            value, = torch.autograd.grad(gradient, w, grad_outputs=torch.from_numpy(direction),
                                         retain_graph=True)
            return value.numpy()

        return float(energy.detach()), gradient.detach().numpy(), hessian_action


def check_derivatives(objective, initial):
    direction = np.random.default_rng(617).normal(size=len(initial))
    direction -= direction.mean()
    direction /= np.linalg.norm(direction)
    interior = np.full_like(initial, 1./len(initial))
    step = 1.e-5
    _, gradient, hessian = objective(interior)
    above, g_above, _ = objective(interior + step*direction)
    below, g_below, _ = objective(interior - step*direction)
    np.testing.assert_allclose((above-below)/(2*step), gradient@direction, rtol=2.e-5, atol=1.e-7)
    np.testing.assert_allclose((g_above-g_below)/(2*step), hessian(direction), rtol=2.e-5, atol=1.e-7)


def profile_error(predicted, reference):
    diff = predicted-reference
    return dict(rmse=float(np.sqrt(np.mean(diff**2))),
                relative_l2=float(np.linalg.norm(diff)/np.linalg.norm(reference)))


def run_case(task):
    name, config, checkpoint, output, steps, model_factory = task
    began = perf_counter()
    model = model_factory().double()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval().requires_grad_(False)
    z, _, density, _, external = data = support(config, model)
    objective = AxialEnergy(data, model)
    initial = softmax(-external)
    check_derivatives(objective, initial)
    weights, history = newton(objective, initial, steps)
    np.testing.assert_allclose(weights.sum(), 1., atol=1.e-9)
    if weights.min() < -1.e-12:
        raise ValueError(f'Negative state probability in {name}')
    start = np.einsum('ezj,j->ze', density, initial)
    final = np.einsum('ezj,j->ze', density, weights)
    # The reference MD profile is opened only after the solver returns.
    with np.load(REF/name/'reference_density.npz') as record:
        np.testing.assert_allclose(z, record['z_A'], rtol=1.e-7, atol=1.e-9)
        ref = record['density_z']
    np.testing.assert_allclose(final.sum(0), ref.sum(0), rtol=1.e-7, atol=1.e-8)
    errors = {}
    for label, values in (('initial', start), ('iterated', final)):
        channels = {e: profile_error(values[:, j], ref[:, j])
                    for j, e in enumerate('CHNO') if ref[:, j].any()}
        errors[label] = dict(total=profile_error(values.sum(1), ref.sum(1)), elements=channels)
    np.savez_compressed(output/f'{name}.npz', z_A=z, beta_V_particle_state=external,
                        initial_atom_density=start, iterated_atom_density=final, reference_atom_density=ref,
                        energy_trace=np.asarray(history['trace']))
    result = dict(case=name, species=config['species'], held_out=config['species'] in ('NH3', 'C2H6'),
                  steps=len(history['trace'])-1, seconds=perf_counter()-began,
                  solver=history, errors=errors)
    emit(stage='case', **{**result, 'solver': {k: v for k, v in history.items() if k != 'trace'}})
    return result


def main(model_factory=PairMLP, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--case', action='append')
    args = parser.parse_args(argv)
    fields = json.loads(FIELDS.read_text())
    names = args.case or list(fields)
    if not args.case and len(names) != 17:
        raise ValueError('Expected 17 external trajectories')
    args.output.mkdir()
    tasks = [(name, fields[name], args.checkpoint, args.output, args.steps, model_factory) for name in names]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        cases = list(pool.map(run_case, tasks))
    groups = {}
    for label, items in (
        ('all', cases),
        ('seen_species', [r for r in cases if not r['held_out']]),
        ('held_out_species', [r for r in cases if r['held_out']]),
    ):
        if items:
            groups[label] = dict(cases=len(items),
                                 converged=sum(r['solver']['status'] == 'stationary' for r in items),
                                 improved=sum(r['errors']['iterated']['total']['rmse'] <
                                              r['errors']['initial']['total']['rmse'] for r in items),
                                 mean_total_relative_l2={stage:
                                     float(np.mean([r['errors'][stage]['total']['relative_l2'] for r in items]))
                                     for stage in ('initial', 'iterated')})
    result = dict(checkpoint=str(args.checkpoint), scope='fixed-N axial profiles with 3D orientations',
                  groups=groups, cases=cases)
    pending = args.output/'summary.pending'
    pending.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    pending.rename(args.output/'summary.json')
    emit(stage='complete', groups=groups)


if __name__ == '__main__':
    main()
