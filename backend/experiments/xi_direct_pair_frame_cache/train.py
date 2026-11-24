"""Train the direct 24-probe kernel using a bounded per-trajectory frame cache."""

import argparse
from functools import partial
import hashlib
import json
from time import perf_counter

import numpy as np
import torch

from experiments.xi_all_projection_loss import train as driver
from experiments.xi_continuous_pair_mlp.direct import DirectPair, DirectPairSource
from experiments.xi_continuous_pair_mlp.train import PairBulk
from experiments.xi_conv3d.bulk import INDEX
from experiments.xi_conv3d_152_hybrid.train import LATEST
from experiments.xi_conv3d_152_sampled_c2.projection import SampledMetrics, stratified_probes
from experiments.xi_direct_pair_frame_cache.cache import FrameCache


def batch(model, source, coeff, targets, scales, block, update_size, correction=None, *, include_el=True):
    if not hasattr(source, 'frame_cache'):
        started = perf_counter()
        driver.emit(stage='frame_cache_start', case=source.local_case, shape=source.shape.tolist())
        torch.cuda.reset_peak_memory_stats()
        source.frame_cache = FrameCache.build(model, source)
        torch.cuda.synchronize()
        driver.emit(stage='frame_cache_ready', case=source.local_case, seconds=perf_counter()-started,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        token = hashlib.sha256(f'{driver.SEED}:c2:{source.local_case}'.encode()).digest()[:8]
        source.probe_sampler = np.random.default_rng(int.from_bytes(token, 'little'))
    cache = source.frame_cache
    if update_size is None:
        with torch.no_grad():
            if not hasattr(source, 'frozen_operator'):
                source.frozen_operator = cache.operator(model)
            first, second, _ = cache.project(source.frozen_operator, coeff)
        return first, second

    draws = [stratified_probes(source.probe_sampler, 24) for _ in coeff]
    ids, inverse = (np.stack(part) for part in zip(*draws))
    loss, prediction, el = cache.loss(model, coeff, targets, scales, ids, inverse, include_el=include_el)
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Nonfinite loss in {source.local_case}')
    (loss*(len(coeff)/update_size)).backward()
    count = getattr(source, 'local_measurements', 0)
    if el is not None and count % 50 == 0:
        driver.emit(stage='local_el', case=source.local_case, state_offset=count,
                    local_el_loss=float(el), weight=1., sampled_c2_directions=ids[0].tolist())
    source.local_measurements = count + len(coeff)
    return prediction


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--memory-gib', type=float, help='Optional ceiling below the currently free GPU memory')
    parser.add_argument('--projection-only', action='store_true', help='Train with the C1 and C2 losses only')
    args, remaining = parser.parse_known_args()
    if args.memory_gib is not None and args.memory_gib <= 0:
        parser.error('Positive GPU allocator budget required')

    def make_model():
        free, total = torch.cuda.mem_get_info()
        budget = free if args.memory_gib is None else min(args.memory_gib*2**30, free)
        if budget < 10*2**30:
            raise RuntimeError('At least 10 GiB of allocator budget is required')
        torch.cuda.set_per_process_memory_fraction(budget/total)
        driver.emit(stage='frame_cache_training', gpu=torch.cuda.get_device_name(),
                    allocator_gib=budget/2**30, loss='c1+c2' if args.projection_only else 'c1+c2+local_el+bulk', sampled_c2=24,
                    eval_directions=152)
        return DirectPair()

    bulk = {case: path for split in json.loads(INDEX.read_text())['splits'].values()
            for case, path in split.items()}

    def prepare(case, data, device):
        return PairBulk(case, data, device, bulk)

    driver.Metrics = SampledMetrics
    driver.main(model_factory=make_model, case_factory=None if args.projection_only else prepare,
                source_factory=DirectPairSource, batch_fn=partial(batch, include_el=not args.projection_only),
                save_case_checkpoints=True,
                argv=['--manifest', str(LATEST), '--block', '8', '--microbatch', '1', *remaining])


if __name__ == '__main__':
    main()
