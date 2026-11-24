"""Train the sparse Xi neighborhood on the matched 50-state split."""

import sys

from experiments.xi_all_projection_loss import train as driver
from experiments.xi_continuous_pair_mlp.train import PairBulk, PairMLP
from experiments.xi_conv3d_152_hybrid import train as hybrid
from experiments.xi_conv3d_152_sampled_c2.projection import SampledMetrics, batch_factory
from experiments.xi_fft_neighborhood_10k.train import with_curvature
from experiments.xi_sparse_xi_neighbors.source import SparseXiSource


if __name__ == '__main__':
    driver.Metrics = SampledMetrics
    hybrid.HybridSource = SparseXiSource
    hybrid.LocalELMultiscaleXi = PairMLP
    # The FFT bulk loss evaluates the dense radial kernel, while this model
    # uses a sparse stencil. Supply the matching Fourier symbol for the loss.
    from experiments.xi_sparse_xi_neighbors.bulk import SparseXiBulk
    hybrid.SixLayerMultiscaleBulk = SparseXiBulk
    hybrid.batch = with_curvature(batch_factory)(24, stratified=True)
    sys.argv[1:] = [*sys.argv[1:], '--variant', 'multiscale', '--block', '16', '--no-fusion']
    hybrid.entry()
