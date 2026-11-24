"""Train against the 152-probe split with the validated hybrid operator path."""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.xi_all_projection_loss.train import emit, main
from experiments.xi_conv3d.bulk import INDEX
from experiments.xi_conv3d_152_hybrid.benchmark import enable_fused_stencils
from experiments.xi_conv3d_152_hybrid.source import HybridSource
from experiments.xi_conv3d_local_el.train import LocalELMultiscaleXi, LocalELXi, batch
from experiments.xi_conv3d_sixlayer.bulk import SixLayerMultiscaleBulk, SixLayerSingleScaleBulk

LATEST = ROOT / 'experiments/xi_paper_3d_backgrounds/run_159573/training_split_3d_eval.json'


def entry():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--manifest', type=Path, default=LATEST)
    parser.add_argument('--variant', choices=('single', 'multiscale'), default='multiscale')
    parser.add_argument('--block', type=int, default=10)
    parser.add_argument('--fft-group', type=int, default=8)
    parser.add_argument('--memory-gib', type=float, default=23.)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--no-fusion', action='store_true')
    args, remaining = parser.parse_known_args()
    if '--help' not in remaining and '-h' not in remaining:
        if not args.manifest.is_file():
            parser.error(f'No published preprocessed split at {args.manifest}')
        if json.loads(args.manifest.read_text()).get('probe_count') != 152:
            parser.error('A 152-probe manifest is required')
    if args.block < 4 or args.fft_group < 1 or not 0 < args.memory_gib <= 23.:
        parser.error('Require block >= 4, a positive FFT group, and memory budget in (0, 23] GiB')
    model, bulk = {'single': (LocalELXi, SixLayerSingleScaleBulk),
                   'multiscale': (LocalELMultiscaleXi, SixLayerMultiscaleBulk)}[args.variant]
    groups = json.loads(INDEX.read_text())['splits']
    labels = {case: path for split in groups.values() for case, path in split.items()}

    def make_model():
        if args.device == 'cuda':
            free, total = torch.cuda.mem_get_info()
            budget = min(args.memory_gib * 2**30, free - 2 * 2**30, total - 2**30)
            if budget < 4 * 2**30:
                raise RuntimeError('Insufficient CUDA memory after reserving headroom')
            torch.cuda.set_per_process_memory_fraction(budget / total)
            emit(stage='hybrid_allocator', effective_gib=budget/2**30, fft_group=args.fft_group)
            if not args.no_fusion:
                enable_fused_stencils()
        return model()

    def prepare(case, data, device):
        return bulk(case, data, device, labels)

    def source_factory(*inputs):
        source = HybridSource(*inputs)
        source.directions_per_fft = args.fft_group
        return source

    main(model_factory=make_model, case_factory=prepare, source_factory=source_factory, batch_fn=batch,
         save_case_checkpoints=True, argv=['--microbatch', '1', *remaining, '--manifest', str(args.manifest),
                                          '--device', args.device, '--block', str(args.block)])


if __name__ == '__main__':
    entry()
