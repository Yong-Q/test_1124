from experiments.xi_conv3d_152_shared_region.source import SharedRegionSource
from experiments.xi_conv3d_local_el.train import LocalELMultiscaleXi, LocalELXi
from experiments.xi_conv3d_sixlayer.bulk import SixLayerMultiscaleBulk, SixLayerSingleScaleBulk
from experiments.xi_total_supervised_50.train import TotalSingleBulk


def components(variant, total):
    if variant == 'dual':
        return LocalELMultiscaleXi, SixLayerMultiscaleBulk, SharedRegionSource, 8
    return LocalELXi, TotalSingleBulk if total else SixLayerSingleScaleBulk, SharedRegionSource, 8
