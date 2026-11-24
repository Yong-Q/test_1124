"""Evaluate the trained sparse Xi model with its exact axial stencil."""

import numpy as np

from experiments.xi_continuous_pair_mlp.network import PairMLP
from experiments.xi_fft_neighborhood_10k.iterate import main as evaluate
from experiments.xi_sparse_xi_neighbors.source import radial_stencil


class SparseAxialPair(PairMLP):
    @staticmethod
    def axial_kernel_bank(n, spacing, widths, cutoff, long):
        radii = (4.5, 6.5) if long else (.65, 2.5)
        stencil = radial_stencil(np.full(3, spacing), long, radii, widths, cutoff)
        bank = np.zeros((len(widths), n))
        for step, kernel in stencil:
            ids = np.arange(-1, 2) * step[2] % n
            for row, values in zip(bank, kernel.sum((1, 2))):
                np.add.at(row, ids, values)
        return bank


if __name__ == '__main__':
    evaluate(SparseAxialPair)
