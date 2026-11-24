"""Use total-output checkpoints in the common external validation."""

import argparse

from experiments.xi_architecture_50.iterate import SingleAxial
from experiments.xi_continuous_pair_mlp.network import PairMLP
from experiments.xi_fft_neighborhood_10k.iterate import main as evaluate


class TotalSingleAxial(SingleAxial):
    outputs_total = True


class TotalPairAxial(PairMLP):
    outputs_total = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('architecture',choices=('single','fft'))
    args, remaining = parser.parse_known_args()
    model = TotalSingleAxial if args.architecture=='single' else TotalPairAxial
    evaluate(model,remaining)
