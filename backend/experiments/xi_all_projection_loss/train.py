"""Fit and audit all projected excess derivatives of the local XiGrandPotential."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch
from torch.nn.functional import linear

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import XiGrandPotential
from portable_io import ResponseWeights
from train_xi_full import invariants
from experiments.xi_all_projection_loss.projection_trajectory import ProjectionTrajectory

SEED = 617
KEYS = ('c1_projected', 'c2_projected')


def full_jets(model, fields):
    """Local -dF_ex and -d²F_ex in every supplied direction.

    Contract the second derivative of each Gram block with the readout gradient
    before forming probe pairs. This avoids [state, cell, P, P, invariants].
    """
    mean, tangents = fields[:, :, 0], fields[:, :, 1:]
    rho = mean[..., 0]
    drho = tangents[..., 0]
    denominator = rho.clamp_min(1e-30)
    marks = (mean[..., 1:] / denominator[..., None]).unflatten(-1, (model.radials, 16))
    dm = ((tangents[..., 1:] - drho[..., None] * marks.flatten(-2)[..., None, :])
          / denominator[..., None, None]).unflatten(-1, (model.radials, 16))
    d_invariants = [dm[..., 0]]
    angular_tangents = []
    for ell in range(1, 4):
        band = slice(ell * ell, (ell + 1) ** 2)
        da = dm[..., band]
        gram = da @ marks[..., None, :, band].transpose(-1, -2)
        d_invariants.append((gram + gram.transpose(-1, -2)).flatten(-2))
        angular_tangents.append(da)

    first, activation1, second, activation2, final = model.readout
    z1 = first(invariants(marks))
    z2 = second(activation1(z1))
    value = final(activation2(z2)).squeeze(-1)
    s1 = torch.where(z1 > activation1.threshold, 1., z1.sigmoid())
    s2 = torch.where(z2 > activation2.threshold, 1., z2.sigmoid())
    dz1 = linear(torch.cat(d_invariants, dim=-1), first.weight)
    dz2 = linear(s1.unsqueeze(-2) * dz1, second.weight)
    adjoint2 = final.weight[0]
    adjoint1 = (adjoint2 * s2) @ second.weight
    grad = (adjoint1 * s1) @ first.weight
    hess = dz1 @ (dz1 * (adjoint1 * s1 * (1. - s1)).unsqueeze(-2)).transpose(-1, -2)
    hess = hess + dz2 @ (dz2 * (adjoint2 * s2 * (1. - s2)).unsqueeze(-2)).transpose(-1, -2)
    angular_grad = grad[..., model.radials:].unflatten(-1, (3, model.radials, model.radials))
    for ell, da in enumerate(angular_tangents):
        gram_grad = angular_grad[..., ell, :, :]
        sym = (gram_grad + gram_grad.transpose(-1, -2)).unsqueeze(-3)
        hess = hess + da.flatten(-2) @ (sym @ da).flatten(-2).transpose(-1, -2)
    directional = (dz2 * (adjoint2 * s2).unsqueeze(-2)).sum(-1)
    return -(rho[..., None] * directional + drho * value[..., None]), -rho[..., None, None] * hess


def singleton_energy(model, source, section):
    return model.readout(source.single_marks[section]).flatten() * source.single_mass[section]


def batch(model, source, coeff, targets, scales, block, update_size, correction=None):
    coupled = None
    if correction is not None:
        with torch.set_grad_enabled(update_size is not None):
            coupled = correction.projected(model, coeff)
    with torch.no_grad():
        singleton = coeff.new_zeros(source.frames)
        for section in source.singleton_slices():
            singleton.index_add_(0, source.single_frame[section],
                                 singleton_energy(model, source, section))
        first = -torch.einsum('btp,t->bp', coeff[:, :, 1:], singleton)
        second = first.new_zeros(targets[1].shape)
        for fields in source.chunks(coeff, block):
            d1, d2 = full_jets(model, fields)
            first += d1.sum(1) * source.dv
            second += d2.sum(1) * source.dv
        if coupled is not None:
            first += coupled[0]
            second += coupled[1]
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise FloatingPointError('Nonfinite projected derivative')
    if update_size is not None:
        fraction = coeff.shape[0] / update_size
        grad1, grad2 = [2. * fraction * (pred - ref) / (scale**2 * ref.numel())
                        for pred, ref, scale in zip((first, second), targets, scales)]
        if coupled is not None:
            ((coupled[0] * grad1).sum() + (coupled[1] * grad2).sum()).backward()
        frame_grad = -torch.einsum('bp,btp->t', grad1, coeff[:, :, 1:])
        for section in source.singleton_slices():
            (singleton_energy(model, source, section) *
             frame_grad[source.single_frame[section]]).sum().backward()
        for fields in source.chunks(coeff, block):
            d1, d2 = full_jets(model, fields)
            ((d1 * grad1[:, None]).sum() * source.dv +
             (d2 * grad2[:, None]).sum() * source.dv).backward()
    return first, second


class Metrics:
    """Aggregate error and target moments without storing predictions."""

    def __init__(self, probes):
        self.n = 0
        self.moments = [np.zeros((3, probes)), np.zeros((3, probes, probes))]

    def update(self, targets, predictions):
        self.n += len(targets[0])
        for totals, target, pred in zip(self.moments, targets, predictions):
            ref = target.astype(np.float64)
            totals[0] += np.square(pred.detach().cpu().numpy() - ref).sum(0)
            totals[1] += ref.sum(0)
            totals[2] += np.square(ref).sum(0)

    @staticmethod
    def score(totals, n, scale):
        mse, mean, ms = totals / n
        variance = np.maximum(ms - mean**2, 0.)
        r2 = np.divide(mse, variance, out=np.zeros_like(variance), where=variance > 0)
        return dict(mse=mse, rmse=np.sqrt(mse), normalized_mse=mse / scale**2,
                    r2=np.where(variance > 0, 1. - r2, np.nan))

    def summary(self, scales):
        first, second = self.moments
        probes = first.shape[1]
        diagonal = np.eye(probes, dtype=bool)
        groups = [("c1", first, scales[0]), ("c2", second.reshape(3, -1), scales[1]),
                  ('c2_diag', second[:, diagonal], scales[1]),
                  ('c2_offdiag', second[:, ~diagonal], scales[1])]
        result = {}
        for key, values, scale in groups:
            metrics = self.score(values.sum(axis=1), self.n * values.shape[1], scale)
            result[key] = {name: float(x) if np.isfinite(x) else None for name, x in metrics.items()}
        result['loss'] = result['c1']['normalized_mse'] + result['c2']['normalized_mse']
        result['states'] = self.n
        return result

    def per_projection(self, scales):
        return {f'{key}_{metric}': value
                for key, totals, scale in zip(('c1', 'c2'), self.moments, scales)
                for metric, value in self.score(totals, self.n, scale).items()}


def dataset(manifest, smoke):
    splits = manifest['splits']
    heldout = {'NH3', 'C2H6'}
    if set(splits) != {'train', 'validation', 'test'} or set(manifest['holdout_species']) != heldout:
        raise ValueError('Expected the NH3/C2H6 molecular holdout')
    seen = set()
    for role, cases in splits.items():
        for case in cases:
            paths = [case['cache'], case['labels']]
            if 'additional_labels' in case:
                if role != 'train':
                    raise ValueError('Additional backgrounds are restricted to training')
                paths.append(case['additional_labels'])
            for value in (case['case'], *(str(Path(path).resolve()) for path in paths)):
                if value in seen:
                    raise ValueError('Trajectory reused across manifest entries: ' + value)
                seen.add(value)
            if (case['species'] in heldout) != (role == 'test'):
                raise ValueError('Unexpected molecule for ' + role + ': ' + case['species'])
    if smoke:
        splits = {role: cases[:1] if role != 'test' else
                  [next(case for case in cases if case['species'] == name) for name in sorted(heldout)]
                  for role, cases in splits.items()}
    return splits


@contextmanager
def case_labels(case):
    with np.load(case['labels']) as base:
        if 'additional_labels' not in case:
            yield base
            return
        with np.load(case['additional_labels']) as added:
            shared = ('shape', 'source_steps', 'voxel_volume', 'deposition', 'probe_energy')
            data = {key: base[key] for key in shared}
            if 'dweights' in base or 'dweights' in added:
                raise ValueError('Augmented 152-probe labels must reconstruct response weights')
            for key, values in data.items():
                np.testing.assert_array_equal(values, added[key], err_msg=f'Probe shards differ in {key}')
            for key in ('weights', *KEYS):
                data[key] = np.concatenate((base[key], added[key]))
            yield data


def take_rows(case, smoke, smoke_states=3):
    if smoke:
        return np.linspace(0, case['states'] - 1, min(case['states'], smoke_states), dtype=int)
    return np.arange(case['states'])


def select_rows(splits, args):
    selections = {}
    for role, cases in splits.items():
        if role == 'train':
            cap = args.train_states
        elif role == 'validation':
            cap = args.validation_states
        else:
            cap = args.test_states_per_species
        groups = [cases] if role != 'test' else [
            [c for c in cases if c['species'] == species] for species in ('NH3', 'C2H6')]
        for group in groups:
            counts = np.array([c['states'] for c in group])
            if args.smoke or cap is None:
                selections.update((c['case'], take_rows(c, args.smoke, args.smoke_states)) for c in group)
                continue
            if cap > counts.sum():
                raise ValueError(f'Requested {cap} {role} states, only {counts.sum()} available')
            ideal = counts * cap / counts.sum()
            quotas = ideal.astype(int)
            extra = cap - int(quotas.sum())
            order = np.argsort(quotas - ideal, kind='stable')
            quotas[order[:extra]] += 1
            for case, quota in zip(group, quotas):
                seed = hashlib.sha256((str(SEED) + ':' + case['case']).encode()).digest()[:8]
                rng = np.random.default_rng(int.from_bytes(seed, 'little'))
                selections[case['case']] = np.sort(rng.choice(case['states'], int(quota), replace=False))
    return selections


def scales_from_train(cases, probes, selections):
    squares = np.zeros(2)
    counts = np.zeros(2)
    for case in cases:
        with case_labels(case) as data:
            for i, key in enumerate(KEYS):
                array = data[key]
                if array.shape != (case['states'],) + (probes,) * (i + 1):
                    raise ValueError(f'Unexpected {key} dimensions: {case["case"]}')
                selected = array[selections[case['case']]].astype(np.float64)
                squares[i] += np.square(selected).sum()
                counts[i] += selected.size
    scales = np.maximum(np.sqrt(squares / counts), 1e-12)
    if not np.isfinite(scales).all():
        raise ValueError('Invalid training label scales')
    return scales


def emit(**record):
    print(json.dumps(record, allow_nan=False), flush=True)


def run_split(model, cases, role, probes, charges, scales, args, selections, optimizer, epoch,
              directory=None, case_factory=None, source_factory=ProjectionTrajectory, batch_fn=batch,
              after_case=None, case_index_offset=0):
    device = next(model.parameters()).device
    direction_ids = np.arange(probes)
    metrics = {'all': Metrics(probes)}
    bulk_scores = {}
    model.train(optimizer is not None)
    for index, case in enumerate(cases, start=case_index_offset):
        case_started = monotonic()
        with case_labels(case) as data:
            if str(data['deposition']) != 'smoothstep':
                raise ValueError('Expected smoothstep deposition')
            source = source_factory(case, data, charges, device, args.block)
            correction = case_factory(case, data, device) if case_factory else None
            bulk_loss = getattr(correction, 'bulk_loss', None)
            response = ResponseWeights(data)
            labels = tuple(data[key] for key in KEYS)
        if len(response.weights) != case['states'] or any(
                a.shape != (case['states'],) + (probes,) * (i + 1) for i, a in enumerate(labels)):
            raise ValueError('Unexpected label or weight dimensions: ' + case['case'])
        rows = selections[case['case']].copy()
        if optimizer is not None:
            np.random.default_rng(SEED + epoch * 1000 + index).shuffle(rows)
        by_case = Metrics(probes)
        by_species = metrics.setdefault(case['species'], Metrics(probes))
        to_device = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)
        updates = 0
        for lo in range(0, len(rows), args.batch):
            update_started = monotonic()
            group = rows[lo:lo + args.batch]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(group), args.microbatch):
                selected = group[offset:offset + args.microbatch]
                directions = np.broadcast_to(direction_ids, (len(selected), probes))
                coeff = np.concatenate((response.weights[selected, :, None],
                                        response.select(selected, directions)), axis=-1)
                targets = tuple(y[selected] for y in labels)
                predicted = batch_fn(model, source, to_device(coeff), tuple(map(to_device, targets)),
                                  scales, args.block, len(group) if optimizer is not None else None, correction=correction)
                by_case.update(targets, predicted)
            if optimizer is not None:
                if bulk_loss is not None:
                    bulk_loss(model).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                if updates == 1:
                    emit(stage='first_update', case=case['case'], epoch=epoch, states=len(group), microbatch=args.microbatch,
                         seconds=monotonic() - update_started,
                         peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30 if device.type == 'cuda' else None)
        for totals in (by_species, metrics['all']):
            totals.n += by_case.n
            for m, added in zip(totals.moments, by_case.moments):
                m += added
        case_bulk = {}
        if bulk_loss is not None:
            with torch.no_grad():
                loss = float(bulk_loss(model))
            case_bulk = dict(bulk_relative_mse=loss, bulk_modes=len(correction.targets))
            for key in ('all', case['species']):
                bulk_scores.setdefault(key, []).append((loss, len(correction.targets)))
        emit(stage='case', split=role, case=case['case'], epoch=epoch, updates=updates, **case_bulk,
             seconds=monotonic() - case_started, **by_case.summary(scales))
        if directory is not None:
            np.savez(directory / f'{role}_case_{index:03}.npz',
                     case=case['case'], species=case['species'], **by_case.per_projection(scales))
        if after_case is not None:
            after_case(index, case['case'])
        del source, correction, bulk_loss, response, labels
    if directory is not None:
        for name, totals in metrics.items():
            np.savez(directory / f'{role}_{name}.npz', **totals.per_projection(scales))
    summaries = {name: totals.summary(scales) for name, totals in metrics.items()}
    for name, values in bulk_scores.items():
        losses, counts = zip(*values)
        summaries[name]['bulk_relative_mse'] = float(np.average(losses, weights=counts))
        summaries[name]['bulk_modes'] = sum(counts)
    return summaries


def main(model_factory=XiGrandPotential, case_factory=None, argv=None, optimizer_factory=None,
         source_factory=ProjectionTrajectory, batch_fn=batch, save_case_checkpoints=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'experiments/xi_zero_center_radial/run_159106/training_split.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--microbatch', type=int, default=2)
    parser.add_argument('--block', type=int, default=128)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--memory-fraction', type=float, default=.45)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-states', type=int, default=3)
    parser.add_argument('--train-states', type=int)
    parser.add_argument('--validation-states', type=int)
    parser.add_argument('--test-states-per-species', type=int)
    parser.add_argument('--skip-train-eval', action='store_true', help='Skip frozen rescoring of the training split')
    parser.add_argument('--skip-validation', action='store_true', help='Skip the validation split')
    parser.add_argument('--resume-checkpoint', type=Path, help='Resume a single epoch at a saved trajectory boundary')
    args = parser.parse_args(argv)
    caps = (args.train_states, args.validation_states, args.test_states_per_species)
    if any(n is not None and n < 1 for n in caps):
        parser.error('State caps must be positive')
    if args.smoke_states < 1:
        parser.error('--smoke-states must be positive')
    if args.smoke and any(n is not None for n in caps):
        parser.error('Smoke mode and state caps are mutually exclusive')
    if min(args.epochs, args.batch, args.microbatch, args.block) < 1 or not 0 < args.memory_fraction <= .5:
        parser.error('Positive epochs, batch, microbatch and block, and memory fraction in (0, 0.5] required')
    manifest_bytes = args.manifest.read_bytes()
    splits = dataset(json.loads(manifest_bytes), args.smoke)
    if args.skip_validation:
        splits['validation'] = []
    selections = select_rows(splits, args)
    splits = {role: [c for c in cases if selections[c['case']].size]
              for role, cases in splits.items()}
    selected_rows = {role: {c['case']: selections[c['case']].tolist() for c in cases}
                     for role, cases in splits.items()}
    checkpoint, completed = None, 0
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location='cpu', weights_only=True)
        prior = checkpoint['configuration']
        previous_rows = json.loads(args.resume_checkpoint.with_name('selected_rows.json').read_text())
        completed = checkpoint['completed_cases']
        if (args.epochs != 1 or checkpoint['epoch'] != 1 or prior['manifest_sha256'] != manifest_hash
                or prior['batch'] != args.batch or selected_rows != previous_rows or not 0 < completed < len(splits['train'])
                or splits['train'][completed-1]['case'] != checkpoint['last_case']):
            parser.error('Checkpoint must match the manifest, sample selections, batch and trajectory order')
    with np.load(splits['train'][0]['labels']) as data:
        probes = data['c1_projected'].shape[1]
    if probes < 2:
        raise ValueError('At least two probes required')
    charges = json.loads((ROOT / 'charges_nuclear.json').read_text())
    args.output.mkdir()
    with (args.output / 'selected_rows.json').open('x') as handle:
        json.dump(selected_rows, handle)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.manual_seed(SEED)
    model = model_factory().to(args.device)
    optimizer = (optimizer_factory(model) if optimizer_factory else
                 torch.optim.AdamW(model.parameters(), lr=3e-4))
    if checkpoint is None:
        scales = scales_from_train(splits['train'], probes, selections)
    else:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['configuration']['charges'] != charges:
            raise ValueError('Checkpoint and current element charges differ')
        scales = np.asarray(checkpoint['configuration']['scales'])
    configuration = dict(manifest=str(args.manifest.resolve()),
                         manifest_sha256=manifest_hash,
                         probes=probes, scales=scales.tolist(), charges=charges, seed=SEED,
                         parameters=sum(p.numel() for p in model.parameters()),
                         smoke=args.smoke, epochs=args.epochs, batch=args.batch, microbatch=args.microbatch,
                         skip_train_eval=args.skip_train_eval,
                         block=args.block, assembly=source_factory.__name__,
                         model=model_factory.__name__,
                         bulk_loss_weight=float(bool(case_factory and getattr(case_factory, 'bulk_supervision', True))),
                         states={role: sum(selections[c['case']].size for c in cases)
                                 for role, cases in splits.items()},
                         device=args.device, memory_fraction=args.memory_fraction)
    if checkpoint is not None:
        configuration['resumed_checkpoint'] = str(args.resume_checkpoint.resolve())
        configuration['resumed_training_trajectories'] = completed
        configuration['resumed_training_states'] = sum(len(selections[c['case']]) for c in splits['train'][:completed])
    emit(stage='starting', **configuration)
    history = []
    def after_case(index, name):
        pending = args.output / 'latest_training.pending'
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                        completed_cases=index + 1, last_case=name,
                        configuration=configuration, complete_epoch_history=history), pending)
        pending.replace(args.output / 'latest_training.pt')

    for epoch in range(1, args.epochs + 1):
        resume_offset = {'case_index_offset': completed} if completed else {}
        history.append(run_split(model, splits['train'][completed:], 'train_updates', probes, charges, scales,
                                 args, selections=selections, optimizer=optimizer, epoch=epoch,
                                 case_factory=case_factory, source_factory=source_factory, batch_fn=batch_fn,
                                 after_case=after_case if save_case_checkpoints else None, **resume_offset))
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                        configuration=configuration, training_history=history),
                   args.output / f'epoch_{epoch:03}.pt')
        emit(stage='epoch', epoch=epoch, **history[-1]['all'])
    # Freeze one final model when scoring every split including training.
    eval_roles = ('validation', 'test') if args.skip_train_eval else ('train', 'validation', 'test')
    results = {role: run_split(model, splits[role], role, probes, charges, scales, args, selections=selections,
                               optimizer=None, epoch=args.epochs, directory=args.output, case_factory=case_factory,
                               source_factory=source_factory, batch_fn=batch_fn)
               for role in eval_roles if splits[role]}
    torch.save(model.state_dict(), args.output / 'trained.pt')
    peak_gib = torch.cuda.max_memory_reserved() / 2**30 if args.device == 'cuda' else None
    with (args.output / 'summary.json').open('x') as handle:
        json.dump(dict(configuration=configuration, training_history=history, metrics=results,
                       peak_reserved_gib=peak_gib), handle, indent=2, allow_nan=False)
    emit(stage='complete', peak_reserved_gib=peak_gib)


if __name__ == '__main__':
    main()
