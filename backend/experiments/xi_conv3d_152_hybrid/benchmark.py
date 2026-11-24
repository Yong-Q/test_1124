"""Verify complete projected derivatives, then time a whole 152-probe state."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.xi_all_projection_loss.train import emit
from experiments.xi_conv3d_152_fast.benchmark_tiles import MANIFEST, gradients, state
from experiments.xi_conv3d_152_fast.source import GroupedLocalELSource
from experiments.xi_conv3d_152_fft_trials.benchmark import compare, measure
from experiments.xi_conv3d_152_hybrid.source import HybridSource
from experiments.xi_conv3d_local_el import train as local_el
from experiments.xi_conv3d_sixlayer import network
from experiments.xi_conv3d_sixlayer.train import pass_once


def enable_fused_stencils():
    """Use the tested equivalent operators in this process only."""
    network.valid_conv = torch.compile(network.valid_conv, fullgraph=True, dynamic=True)
    network.valid_adjoint = torch.compile(network.valid_adjoint, fullgraph=True, dynamic=True)
    local_el.valid_conv = network.valid_conv


def run(source_class, case, geometry, charges, coeff, targets, scales, block, region, label):
    torch.cuda.empty_cache()
    torch.manual_seed(617)
    model = local_el.LocalELMultiscaleXi().cuda()
    source = source_class(case, geometry, charges, coeff.device, block)
    source.region = region
    source.directions_per_fft = 8
    emit(stage='configuration', label=label, block=block, region=region)
    local_loss = measure(label + '_el', lambda: local_el.local_backward(model, source, coeff, 1))
    local_grad = gradients(model)
    model.zero_grad(set_to_none=True)
    first, second = measure(label + '_projected', lambda: pass_once(model, source, coeff, targets, scales, 1))
    return dict(el=local_loss.detach().cpu(), el_gradient=local_grad, c1=first.cpu(), c2=second.cpu(),
                projected_gradient=gradients(model))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--block', type=int, default=12)
    parser.add_argument('--region', type=int, help='Limit central grid for an optional short timing run')
    parser.add_argument('--no-fusion', action='store_true')
    args = parser.parse_args()

    free, total = torch.cuda.mem_get_info()
    budget = min(23 * 2**30, free - 2 * 2**30, total - 2**30)
    if budget < 4 * 2**30 or args.block < 4 or (args.region is not None and args.region < 1):
        parser.error('At least 4 GiB of allocator budget and a block size of at least 4 are required')
    torch.cuda.set_per_process_memory_fraction(budget / total)
    case, row, family, geom, coeff, targets = state(args.manifest)
    coeff = torch.as_tensor(coeff, dtype=torch.float32, device='cuda')
    targets = tuple(torch.as_tensor(y, dtype=torch.float32, device='cuda') for y in targets)
    scales = [y.double().square().mean().sqrt().clamp_min(1.e-12).item() for y in targets]
    charges = json.loads((ROOT / 'charges_nuclear.json').read_text())
    emit(stage='starting', manifest=str(args.manifest), case=case['case'], row=row, family=family,
         probes=152, shape=geom['shape'].tolist(), allocator_gib=budget/2**30,
         gpu=torch.cuda.get_device_name())
    common = (case, geom, charges, coeff, targets, scales)

    reference = run(GroupedLocalELSource, *common, 4, 8, 'verify_reference')
    if not args.no_fusion:
        enable_fused_stencils()
    candidate = run(HybridSource, *common, 4, 8, 'verify_hybrid')
    for key, expected in reference.items():
        compare(key, expected, candidate[key], atol=2.e-7)
    emit(stage='verification_complete', projections=152)
    if not args.verify_only:
        for block in sorted({4, 8, 10, args.block} & set(range(4, args.block+1)), reverse=True):
            try:
                run(HybridSource, *common, block, args.region, 'hybrid')
            except RuntimeError as exc:
                if not isinstance(exc, torch.cuda.OutOfMemoryError) and 'CUFFT_ALLOC_FAILED' not in str(exc):
                    raise
                emit(stage='memory_limit', block=block, detail=str(exc))
            else:
                emit(stage='complete', block=block, scope='whole_box' if args.region is None else 'region')
                break
        else:
            raise SystemExit('All attempted block sizes reached the memory limit')


if __name__ == '__main__':
    main()
