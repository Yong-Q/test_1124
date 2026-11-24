"""Use the same independent fixed-particle external cases for all three models."""

import argparse

import torch
from torch.nn import functional as F

from experiments.xi_continuous_pair_mlp.network import PairMLP
from experiments.xi_conv3d_local_el.train import LocalELMultiscaleXi, LocalELXi
from experiments.xi_fft_neighborhood_10k.iterate import main as evaluate


class AxialCNN:
    def energy_density(self, fields):
        h = self.angular_invariants(fields)
        for index, layer in enumerate(self.layers):
            z = F.linear(h, layer.weight)
            if index:
                weights = self.context[index-1]
                # The four transverse neighbors have the same value in the axial sector.
                neighbors = 4*z + torch.roll(z, 1, 0) + torch.roll(z, -1, 0)
                z = z*weights[:, 0] + neighbors*(weights[:, 1]/6.)
            h = F.softplus(z+layer.bias)-F.softplus(layer.bias)
        return self.output(h).squeeze(-1)


class SingleAxial(AxialCNN, LocalELXi):
    pass


class MultiscaleAxial(AxialCNN, LocalELMultiscaleXi):
    pass


def main():
    models = dict(fft=PairMLP, single=SingleAxial, multiscale=MultiscaleAxial)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('architecture', choices=models)
    args, remaining = parser.parse_known_args()
    evaluate(models[args.architecture], remaining)


if __name__ == '__main__':
    main()
