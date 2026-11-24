"""Train the same 50-state architectures directly on the total intrinsic functional."""

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
import sys

import numpy as np
import torch

from experiments.xi_all_projection_loss import train as driver
from experiments.xi_continuous_pair_mlp.train import PairBulk, PairMLP
from experiments.xi_conv3d_152_hybrid import train as hybrid
from experiments.xi_conv3d_152_sampled_c2.projection import SampledMetrics, batch_factory
from experiments.xi_conv3d_152_shared_region.source import SharedRegionSource
from experiments.xi_conv3d_local_el.train import LocalOccupancy
from experiments.xi_conv3d_sixlayer.bulk import SixLayerSingleScaleBulk, match, predicted_hessian
from experiments.xi_fft_neighborhood_10k.train import curvature_summary
from experiments.xi_positive_total_pair.operators import IdealProjection
from portable_io import ResponseWeights


class TotalLabels:
    def __init__(self, response, probes, order):
        self.response = response
        self.probes = probes
        self.order = order
        self.shape = (len(response.weights),) + (probes,)*order

    def __getitem__(self, rows):
        rows = np.atleast_1d(rows)
        weights = self.response.weights[rows].astype(np.float64)
        directions = np.broadcast_to(np.arange(self.probes), (len(rows), self.probes))
        dw = self.response.select(rows, directions).astype(np.float64)
        if self.order == 1:
            log_weights = np.log(weights, where=weights>0, out=np.zeros_like(weights))
            return -np.einsum('bf,bfp->bp', log_weights, dw)
        normalized = np.divide(dw, np.sqrt(weights)[...,None], where=weights[...,None]>0,
                               out=np.zeros_like(dw))
        return -(normalized.transpose(0,2,1) @ normalized)


class TotalCase(Mapping):
    def __init__(self, data):
        self.data = data
        response = ResponseWeights(data)
        self.labels = {key:TotalLabels(response, data[key].shape[1], index)
                       for index,key in enumerate(driver.KEYS, 1)}

    def __getitem__(self, key):
        return self.labels[key] if key in self.labels else self.data[key]

    def __iter__(self):
        return iter(self.data.keys())

    def __len__(self):
        return len(self.data.keys())


class TotalOccupancy(LocalOccupancy):
    @torch.no_grad()
    def loss_and_tangent(self, weights, total, return_tangent=True):
        weights = weights/weights.sum(-1,keepdim=True)
        mean = (self.counts @ weights.T).T
        log_mean = mean.clamp_min(torch.finfo(mean.dtype).tiny).log()
        ideal = (self.transpose @ log_mean.T).T
        # The inherited EL residual adds this ideal term analytically.
        return super().loss_and_tangent(weights, total-ideal, return_tangent)


class TotalSingleBulk(SixLayerSingleScaleBulk):
    def bulk_loss(self, model):
        with_identity = predicted_hessian(self, model, self.short_modes)
        return match(self, with_identity-torch.eye(len(self.rho),device=self.rho.device))


class TotalPairBulk(PairBulk):
    subtract_ideal_identity = True


def measured_batch():
    original = batch_factory(24, stratified=True)

    def batch(model, source, coeff, targets, scales, block, update_size, correction=None):
        if update_size is not None and not hasattr(source,'local_occupancy'):
            source.local_occupancy = TotalOccupancy.from_source(source)
        result = original(model, source, coeff, targets, scales, block, update_size, correction)
        if update_size is None:
            if not hasattr(source,'ideal_curvature'):
                occupancy = LocalOccupancy.from_source(source)
                source.ideal_curvature = IdealProjection(occupancy.counts,source.dv)
            coeff = coeff.double()
            _,ideal = source.ideal_curvature.project(coeff,coeff[:,:,1:])
            p,y = [x.detach().cpu().double() for x in (result[1],targets[1])]
            ideal = ideal.cpu()
            driver.emit(stage='held_out_curvature',case=source.local_case,probes=p.shape[-1],
                        total=curvature_summary(p,y),
                        excess=curvature_summary(p-ideal,y-ideal))
        return result

    return batch


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('architecture',choices=('single','fft'))
    args, remaining = parser.parse_known_args()
    original_labels = driver.case_labels

    @contextmanager
    def total_labels(case):
        with original_labels(case) as data:
            yield TotalCase(data)

    driver.case_labels = total_labels
    driver.Metrics = SampledMetrics
    hybrid.HybridSource = SharedRegionSource
    hybrid.batch = measured_batch()
    hybrid.SixLayerSingleScaleBulk = TotalSingleBulk
    hybrid.SixLayerMultiscaleBulk = TotalPairBulk
    hybrid.LocalELMultiscaleXi = PairMLP
    variant, block = ('single',8) if args.architecture=='single' else ('multiscale',16)
    driver.emit(stage='objective',network_output='beta_F_total',metrics='signed_total_derivatives',
                architecture=args.architecture,probe_count=152,train_c2_probes=24)
    sys.argv[1:] = [*remaining,'--variant',variant,'--block',str(block),'--no-fusion']
    hybrid.entry()


if __name__ == '__main__':
    main()
