from experiments.xi_continuous_pair_mlp.network import PairMLP
from experiments.xi_continuous_pair_mlp.train import PairBulk
from experiments.xi_sparse_xi_neighbors.bulk import SparseXiBulk
from experiments.xi_sparse_xi_neighbors.source import SparseXiSource


def components(variant, total):
    from experiments.xi_conv3d_152_shared_region.source import SharedRegionSource
    from experiments.xi_total_supervised_50.train import TotalPairBulk

    if variant == 'sparse':
        return PairMLP, SparseXiBulk, SparseXiSource, 16
    return PairMLP, TotalPairBulk if total else PairBulk, SharedRegionSource, 16
