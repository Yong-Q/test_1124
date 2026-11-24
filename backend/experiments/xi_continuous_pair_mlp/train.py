"""Train the pair-kernel model with the existing projection, EL and bulk losses."""

import argparse
import sys

import torch

from experiments.xi_all_projection_loss import train as driver
from experiments.xi_conv3d_152_hybrid import train as hybrid
from experiments.xi_conv3d_152_sampled_c2.projection import SampledMetrics, batch_factory
from experiments.xi_conv3d_152_shared_region.source import SharedRegionSource
from experiments.xi_conv3d_multiscale.bulk import MultiscaleBulk
from experiments.xi_conv3d_sixlayer.bulk import match
from experiments.xi_conv3d_sixlayer.network import softplus_slope
from experiments.xi_continuous_pair_mlp.direct import DirectPair, DirectPairSource
from experiments.xi_continuous_pair_mlp.network import PairMLP, SCALARS


class PairBulk(MultiscaleBulk):
    def bulk_loss(self, model):
        short_weights = model.spatial[:SCALARS]
        short_mean = (self.rho @ self.inputs) * short_weights.sum(1)
        features = torch.cat((short_mean, short_mean.new_zeros(model.features-SCALARS)))
        short = self.inputs[None] * (self.short_modes @ short_weights.T)[:, None, :]
        long = self.inputs[None, :, 1:5] * (self.long_modes @ model.long_mix.T)[:, None, :]
        directions = torch.cat((short, long), -1) * self.rho.sqrt()[None, :, None]
        hessian = (directions * model.scalar_pair) @ directions.mT
        if isinstance(model, PairMLP) and not isinstance(model, DirectPair):
            layer = model.layers[0]
            weights = torch.cat((layer.weight[:, :SCALARS], layer.weight[:, -model.long_channels:]), -1)
            hidden = directions @ weights.T
            slope = softplus_slope(layer(features))
            curvature = model.output.weight[0] * slope * (1-slope)
            hessian = hessian + (hidden * curvature) @ hidden.mT
        identity = torch.eye(len(self.rho), device=hessian.device, dtype=hessian.dtype)
        if getattr(self, 'subtract_ideal_identity', False):
            return match(self, hessian)
        return match(self, hessian + identity)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--c2-probes', type=int, default=48)
    parser.add_argument('--direct', action='store_true')
    parser.add_argument('--stratified', action='store_true')
    args, remaining = parser.parse_known_args()
    if not 2 <= args.c2_probes <= 152:
        parser.error('c2-probes must be in [2, 152]')
    if args.stratified and args.c2_probes not in (8, 16, 24):
        parser.error('Stratified sampling requires 8, 16 or 24 directions')
    driver.Metrics = SampledMetrics
    hybrid.HybridSource = DirectPairSource if args.direct else SharedRegionSource
    hybrid.LocalELMultiscaleXi = DirectPair if args.direct else PairMLP
    hybrid.SixLayerMultiscaleBulk = PairBulk
    hybrid.batch = batch_factory(args.c2_probes, stratified=args.stratified)
    driver.emit(stage='pair_mlp', direct=args.direct, radial_rank=PairMLP.rank, c2_directions=args.c2_probes,
                full_eval_directions=152, stratified=args.stratified)
    sys.argv[1:] = ['--block', '8', '--no-fusion', *remaining, '--variant', 'multiscale']
    hybrid.entry()


if __name__ == '__main__':
    main()
