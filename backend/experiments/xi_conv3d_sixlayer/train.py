"""Train both six-layer volumetric CNNs on the same full projection objective."""

import argparse
import gc
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.xi_all_projection_loss.train import emit, main
from experiments.xi_conv3d.bulk import INDEX
from experiments.xi_conv3d_sixlayer.bulk import SixLayerBulk, SixLayerMultiscaleBulk, SixLayerSingleScaleBulk
from experiments.xi_conv3d_sixlayer.network import SixLayerXi, SixLayerMultiscaleXi, SixLayerSingleScaleXi
from experiments.xi_conv3d_sixlayer.source import SixLayerSource, SixLayerAxisSource


def pass_once(model, source, coeff, targets, scales, update_size):
    with torch.no_grad():
        first,second = (torch.zeros_like(t) for t in targets)
        for field,shape in source.chunks(model,coeff):
            d1,d2 = model.volume_jets(field,shape)
            first += d1*source.dv
            second += d2*source.dv
        if not (torch.isfinite(first).all() and torch.isfinite(second).all()):
            raise FloatingPointError('Nonfinite six-layer derivative projections')
    if update_size is not None:
        fraction = len(coeff)/update_size
        g1,g2 = [2.*fraction*(p-t)/(s*s*t.numel())
                 for p,t,s in zip((first,second),targets,scales)]
        for field,shape in source.chunks(model,coeff):
            d1,d2 = model.volume_jets(field,shape)
            ((d1*g1).sum()*source.dv+(d2*g2).sum()*source.dv).backward()
    return first,second


def batch(model,source,coeff,targets,scales,block,update_size,correction=None):
    params = tuple(model.parameters())
    old = [None if p.grad is None else p.grad.detach().clone() for p in params]
    while True:
        try:
            return pass_once(model,source,coeff,targets,scales,update_size)
        except RuntimeError as exc:
            is_oom = isinstance(exc,torch.cuda.OutOfMemoryError) or 'CUFFT_ALLOC_FAILED' in str(exc)
            if not is_oom or source.block<=4:
                raise
        for p,previous in zip(params,old):
            p.grad=None if previous is None else previous.clone()
        previous=source.block
        source.block=max(4,source.block-2)
        source.transfer_cache.clear()
        source.observation_cache.clear()
        source.observation_cache_bytes=0
        gc.collect()
        torch.cuda.empty_cache()
        emit(stage='retry_spatial_tile', previous=previous, current=source.block)


def entry():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--variant', choices=('axis','single','multiscale'), required=True)
    selected,rest=parser.parse_known_args()
    groups=json.loads(INDEX.read_text())['splits']
    labels={case:path for split in groups.values() for case,path in split.items()}
    model,case,source = {
        'axis': (SixLayerXi,SixLayerBulk,SixLayerAxisSource),
        'single': (SixLayerSingleScaleXi,SixLayerSingleScaleBulk,SixLayerSource),
        'multiscale': (SixLayerMultiscaleXi,SixLayerMultiscaleBulk,SixLayerSource),
    }[selected.variant]

    def prepare(entry,data,device):
        return case(entry,data,device,labels)
    main(model_factory=model,case_factory=prepare,source_factory=source,batch_fn=batch,
         save_case_checkpoints=True,argv=rest)


if __name__ == '__main__':
    entry()
