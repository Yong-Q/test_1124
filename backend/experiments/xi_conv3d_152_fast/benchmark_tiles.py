"""Compare tile sizes on one reweighted training state, without an optimizer step."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.xi_all_projection_loss.train import (
    KEYS, SEED, ResponseWeights, case_labels, dataset, emit,
)
from experiments.xi_conv3d_local_el.train import (
    LocalELMultiscaleXi, LocalELXi, local_backward,
)
from experiments.xi_conv3d_152_fast.source import GroupedLocalELSource
from experiments.xi_conv3d_152_fast.train import MANIFEST
from experiments.xi_conv3d_sixlayer.train import pass_once


def state(manifest):
    cases = dataset(json.loads(manifest.read_bytes()), smoke=False)['train']
    case = next(c for c in cases if 'additional_labels' in c)
    with np.load(case['labels']) as base:
        row = len(base['weights'])
    with np.load(case['additional_labels']) as added:
        family = str(added['background_family'][added['state_indices'][0]])
    with case_labels(case) as data:
        response = ResponseWeights(data)
        directions = np.arange(data[KEYS[0]].shape[1])[None, :]
        if directions.shape[1] != 152:
            raise ValueError('The benchmark expects all 152 probes')
        rows = np.array([row])
        coeff = np.concatenate((response.weights[rows, :, None],
                                response.select(rows, directions)), axis=-1)
        targets = tuple(data[key][rows] for key in KEYS)
        geometry = {key: data[key] for key in ('voxel_volume', 'shape', 'source_steps')}
    return case, row, family, geometry, coeff, targets


def timed(label, block, operation):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    value = operation()
    torch.cuda.synchronize()
    emit(stage=label, block=block, seconds=perf_counter() - start,
         peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
         peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    return value


def gradients(model):
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten()
                      for p in model.parameters()]).detach().cpu()


def run_tile(model_class, block, case, geometry, charges, coeff, targets, scales, region, fft_group):
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    model = model_class().cuda()
    source = GroupedLocalELSource(case, geometry, charges, coeff.device, block)
    source.region = region
    source.directions_per_fft = fft_group
    emit(stage='tile_start', block=block)
    local_loss = timed('local_el', block, lambda: local_backward(model, source, coeff, 1))
    local_grad = gradients(model)
    model.zero_grad(set_to_none=True)
    predicted = timed('projection_forward_and_backward', block,
                      lambda: pass_once(model, source, coeff, targets, scales, 1))
    return dict(local_el=local_loss.detach().cpu(), local_gradient=local_grad,
                c1=predicted[0].cpu(), c2=predicted[1].cpu(),
                projection_gradient=gradients(model))


def compare(reference, candidate, reference_block, block):
    for key, first in reference.items():
        second = candidate[key]
        delta = (second.double() - first.double()).flatten()
        norm = torch.linalg.vector_norm(first.double().flatten()).clamp_min(1e-12)
        emit(stage='tile_difference', reference_block=reference_block, block=block,
             quantity=key, max_abs=float(delta.abs().max()),
             relative_l2=float(torch.linalg.vector_norm(delta) / norm))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--variant', choices=('single', 'multiscale'), default='single')
    parser.add_argument('--tiles', nargs='+', type=int, default=(8, 10, 12))
    parser.add_argument('--region', type=int, help='Compare the same smaller cube for quick timing')
    parser.add_argument('--fft-group', type=int, default=4)
    parser.add_argument('--memory-fraction', type=float, default=.45)
    parser.add_argument('--memory-gib', type=float, help='Absolute allocator cap for cross-GPU tests')
    args = parser.parse_args()
    if min(*args.tiles, args.fft_group, args.region if args.region is not None else 1) < 1 or not 0 < args.memory_fraction <= .5:
        parser.error('Tile, region and group sizes must be positive and memory fraction in (0, .5]')
    if not torch.cuda.is_available():
        parser.error('A CUDA GPU is required for timing')
    free, total = torch.cuda.mem_get_info()
    if args.memory_gib is not None and not 0 < args.memory_gib <= 23:
        parser.error('The allocator cap must be in (0, 23] GiB for a 24 GiB card')
    requested = args.memory_gib * 2**30 if args.memory_gib else args.memory_fraction * total
    budget = min(requested, free - 2 * 2**30, total - 1 * 2**30)
    if budget < 4 * 2**30:
        parser.error('At least 4 GiB of allocator budget is required after reserving headroom')
    torch.cuda.set_per_process_memory_fraction(budget / total)
    case, row, family, geometry, coeff, targets = state(args.manifest)
    coeff = torch.as_tensor(coeff, device='cuda', dtype=torch.float32)
    targets = tuple(torch.as_tensor(t, device='cuda', dtype=torch.float32) for t in targets)
    scales = [t.double().square().mean().sqrt().clamp_min(1e-12).item() for t in targets]
    charges = json.loads((ROOT / 'charges_nuclear.json').read_text())
    model = LocalELXi if args.variant == 'single' else LocalELMultiscaleXi
    emit(stage='benchmark_start', case=case['case'], row=row, family=family,
         manifest=str(args.manifest), probes=152, tiles=args.tiles, variant=args.variant,
         shape=geometry['shape'].tolist(), region=args.region, fft_group=args.fft_group,
         gpu=torch.cuda.get_device_name(), memory_budget_gib=budget / 2**30,
         available_gib=free / 2**30)
    results = {}
    for block in args.tiles:
        try:
            results[block] = run_tile(model, block, case, geometry, charges, coeff, targets, scales,
                                      args.region, args.fft_group)
        except RuntimeError as exc:
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and 'CUFFT_ALLOC_FAILED' not in str(exc):
                raise
            emit(stage='tile_memory_limit', block=block, detail=str(exc))
    if not results:
        raise SystemExit('No tile size fit the configured memory limit')
    reference_block = next(iter(results))
    for block, result in results.items():
        if block != reference_block:
            compare(results[reference_block], result, reference_block, block)
    emit(stage='benchmark_complete', blocks=list(results))


if __name__ == '__main__':
    main()
