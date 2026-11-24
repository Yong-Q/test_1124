"""Compare the FFT MLP and two CNNs with identical samples and probe selection."""

import argparse
import sys

from experiments.xi_all_projection_loss import train as driver
from experiments.xi_continuous_pair_mlp.train import PairBulk, PairMLP
from experiments.xi_conv3d_152_hybrid import train as hybrid
from experiments.xi_conv3d_152_sampled_c2.projection import SampledMetrics, batch_factory
from experiments.xi_conv3d_152_shared_region.source import SharedRegionSource
from experiments.xi_fft_neighborhood_10k.train import with_curvature


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('architecture', choices=('fft', 'single', 'multiscale'))
    args, remaining = parser.parse_known_args()
    driver.Metrics = SampledMetrics
    hybrid.HybridSource = SharedRegionSource
    hybrid.batch = with_curvature(batch_factory)(24, stratified=True)
    block = 8
    variant = args.architecture
    if args.architecture == 'fft':
        hybrid.LocalELMultiscaleXi = PairMLP
        hybrid.SixLayerMultiscaleBulk = PairBulk
        variant = 'multiscale'
        block = 16
    driver.emit(stage='comparison_configuration', architecture=args.architecture, block=block,
                train_curvature_probes=24, test_curvature_probes=152)
    sys.argv[1:] = [*remaining, '--variant', variant, '--block', str(block)]
    hybrid.entry()


if __name__ == '__main__':
    main()
