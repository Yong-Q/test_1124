"""Train the existing local-EL CNN with bounded FFT groups for 152 probes."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.xi_all_projection_loss.train import main
from experiments.xi_conv3d.bulk import INDEX
from experiments.xi_conv3d_local_el.train import LocalELMultiscaleXi, LocalELXi, batch
from experiments.xi_conv3d_sixlayer.bulk import SixLayerMultiscaleBulk, SixLayerSingleScaleBulk
from experiments.xi_conv3d_152_fast.source import GroupedLocalELSource

MANIFEST = ROOT / 'experiments/xi_pnas_probes152_augmented/run_159403/training_split.json'


def entry():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--variant', choices=('single', 'multiscale'), default='single')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--fft-group', type=int, default=8)
    parser.add_argument('--block', type=int, default=8)
    parser.add_argument('--microbatch', type=int, default=1)
    args, remaining = parser.parse_known_args()
    if min(args.fft_group, args.block, args.microbatch) < 1:
        parser.error('FFT group, block and microbatch must be positive')
    if json.loads(args.manifest.read_text()).get('probe_count') != 152:
        parser.error('A 152-probe training manifest is required')

    model, bulk = {'single': (LocalELXi, SixLayerSingleScaleBulk),
                   'multiscale': (LocalELMultiscaleXi, SixLayerMultiscaleBulk)}[args.variant]
    groups = json.loads(INDEX.read_text())['splits']
    labels = {case: path for split in groups.values() for case, path in split.items()}

    def prepare(case, data, device):
        return bulk(case, data, device, labels)

    def source(*inputs):
        trajectory = GroupedLocalELSource(*inputs)
        trajectory.directions_per_fft = args.fft_group
        return trajectory

    main(model_factory=model, case_factory=prepare, source_factory=source, batch_fn=batch,
         save_case_checkpoints=True, argv=[*remaining, '--manifest', str(args.manifest),
                                          '--block', str(args.block), '--microbatch', str(args.microbatch)])


if __name__ == '__main__':
    entry()
