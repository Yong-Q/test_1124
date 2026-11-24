"""GPU trials for three equivalent operator changes; no optimizer step."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.xi_all_projection_loss.train import emit
from experiments.xi_conv3d_152_fast.benchmark_tiles import MANIFEST, state
from experiments.xi_conv3d_152_fft_trials.operators import TrialSource, boxes
from experiments.xi_conv3d_local_el.train import LocalELMultiscaleXi
from experiments.xi_conv3d_multiscale.network import radial_bank
from experiments.xi_conv3d_sixlayer.network import HALO, valid_adjoint, valid_conv


def measure(label, run):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    result = run()
    torch.cuda.synchronize()
    emit(stage=label, seconds=perf_counter() - start,
         peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
         peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    return result


def compare(label, reference, candidate, atol=2e-6, rtol=2e-4):
    delta = (reference - candidate).flatten()
    scale = torch.linalg.vector_norm(reference.flatten()).clamp_min(1e-20)
    emit(stage='equivalence', quantity=label, max_abs=float(delta.abs().max()),
         relative_l2=float(torch.linalg.vector_norm(delta) / scale))
    torch.testing.assert_close(candidate, reference, rtol=rtol, atol=atol)


@torch.no_grad()
def first_layer(model, source, coeff, region):
    tiles = list(boxes(region, source.block, HALO))
    lower = np.full(3, -HALO)
    upper = np.full(3, region + HALO)
    shape = tuple(upper - lower)
    emit(stage='first_layer_setup', region=region, tiles=len(tiles), projections=coeff.shape[-1] - 1)
    for name, long, halo in (('short', False, source.short_halo), ('long', True, source.long_halo)):
        # Build kernel spectra and run a tiny untimed first group for every shape.
        for lo, hi in tiles:
            source.filtered_tile(model, coeff[..., :1], lo, hi, halo, long)
        source.global_crop(model, coeff[..., :1], lower, upper, long)
        reference = measure(name + '_tiled', lambda: [source.filtered_tile(model, coeff, lo, hi, halo, long)
                                                     for lo, hi in tiles])
        shared = measure(name + '_shared', lambda: [source.shared_tile(model, coeff, lo, hi, halo, long)
                                                   for lo, hi in tiles])
        for i, (base, new) in enumerate(zip(reference, shared)):
            compare(f'{name}_shared_{i}', base, new)
        del shared, base, new
        torch.cuda.empty_cache()
        volume = measure(name + '_global', lambda: source.global_crop(model, coeff, lower, upper, long))
        volume = volume.unflatten(1, shape)
        for i, (base, (lo, hi)) in enumerate(zip(reference, tiles)):
            crop = tuple(slice(a, b) for a, b in zip(lo - lower, hi - lower))
            compare(f'{name}_global_{i}', base, volume[(slice(None), *crop)].flatten(1, 3))
        del reference, base, volume
        torch.cuda.empty_cache()


def projected_derivatives(model, source, coeff):
    # A corner tile spans periodic boundaries in all three Cartesian directions.
    lo, hi = next(boxes(4, 4, HALO))
    coeff = coeff[..., :9]
    params = tuple(model.parameters())
    reference = None
    for mode in ('tiled', 'shared', 'global'):
        fields = []
        for long, halo in ((False, source.short_halo), (True, source.long_halo)):
            if mode == 'global':
                field = source.global_crop(model, coeff, lo, hi, long)
            else:
                op = source.filtered_tile if mode == 'tiled' else source.shared_tile
                field = op(model, coeff, lo, hi, halo, long)
            fields.append(field)
        c1, c2 = model.volume_jets(torch.cat(fields, -1), tuple(hi - lo))
        gradients = torch.autograd.grad(c1.square().sum() + c2.square().sum(), params, allow_unused=True)
        vector = torch.cat([g.detach().flatten() for g in gradients if g is not None])
        result = (c1.detach(), c2.detach(), vector)
        if reference is None:
            reference = result
        else:
            for name, baseline, candidate in zip(('c1', 'c2', 'parameter_gradient'), reference, result):
                compare(mode + '_' + name, baseline, candidate)
    torch.cuda.empty_cache()


def stencils():
    for name, operation, shape in (('valid_conv', valid_conv, (1, 18, 18, 18, 152, 64)),
                                   ('valid_adjoint', valid_adjoint, (1, 16, 16, 16, 64))):
        x = torch.randn(shape, device='cuda', requires_grad=True)
        coeff = torch.randn((64, 2), device='cuda', requires_grad=True)

        def once(op):
            y = op(x, coeff)
            gradients = torch.autograd.grad(y.square().mean(), (x, coeff))
            return y.detach(), *gradients

        compiled = torch.compile(operation, fullgraph=True)
        once(operation)
        measure(name + '_compile_warmup', lambda: once(compiled))
        baseline = measure(name + '_eager', lambda: once(operation))
        candidate = measure(name + '_fused', lambda: once(compiled))
        for i, (base, new) in enumerate(zip(baseline, candidate)):
            compare(f'{name}_{i}', base, new, atol=2.e-5)


@torch.no_grad()
def scales():
    unit = torch.randn((40, 40, 40), device='cuda')
    # The repeated small field is the same periodic input on 20 and 200 Å boxes.
    large = unit.repeat(10, 10, 10)
    for long, cutoff, width in ((False, 5., .9), (True, 12., 4.)):
        result = []
        for x in (unit, large):
            kernel = radial_bank(x.shape, (.5, .5, .5), (width,), cutoff, long=long)[0]
            spectrum = torch.fft.rfftn(x)
            spectrum *= torch.fft.rfftn(kernel.to(x)).real
            y = torch.fft.irfftn(spectrum, s=x.shape)
            result.append(y[:40, :40, :40].clone())
        compare(f'20A_200A_{"long" if long else "short"}', *result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=('fft', 'stencil', 'scale'), required=True)
    parser.add_argument('--region', type=int, default=16)
    args = parser.parse_args()
    free, total = torch.cuda.mem_get_info()
    budget = min(23 * 2**30, free - 2 * 2**30, total - 2**30)
    if args.region < 1 or budget < 4 * 2**30:
        parser.error('Positive region size and at least 4 GiB available required')
    torch.cuda.set_per_process_memory_fraction(budget / total)
    torch.manual_seed(617)
    emit(stage='start', part=args.part, gpu=torch.cuda.get_device_name(), budget_gib=budget / 2**30)
    if args.part == 'fft':
        case, _, _, geometry, coeff, _ = state(MANIFEST)
        charges = json.loads((ROOT / 'charges_nuclear.json').read_text())
        model = LocalELMultiscaleXi().cuda()
        source = TrialSource(case, geometry, charges, torch.device('cuda'), 8)
        source.directions_per_fft = 4
        weights = torch.as_tensor(coeff, device='cuda', dtype=torch.float32)
        first_layer(model, source, weights, min(args.region, *source.shape))
        projected_derivatives(model, source, weights)
    elif args.part == 'stencil':
        stencils()
    else:
        scales()
    emit(stage='complete', part=args.part)


if __name__ == '__main__':
    main()
