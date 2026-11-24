"""Full-grid training on the published 152-probe preprocessing format."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from experiments.xi_all_projection_loss import train as driver
from experiments.xi_conv3d_152_sampled_c2.projection import SampledMetrics, batch_factory
from experiments.xi_fft_neighborhood_10k.train import with_curvature
from experiments.xi_total_supervised_50.train import TotalCase, measured_batch
from models.cnn3d.model import components as cnn
from models.neighborhood_mlp.model import components as mlp
from models.stable_kernel.model import CHANNELS


def dataset_inputs(path, bulk_index=None):
    manifest = json.loads(path.read_text())
    if manifest.get('probe_count') != 152:
        raise ValueError('Expected a manifest with 152 probes')
    for cases in manifest['splits'].values():
        for case in cases:
            for key in ('cache','labels','additional_labels'):
                if key not in case:
                    continue
                file=(path.parent/case[key]).resolve()
                if not file.is_file():
                    raise FileNotFoundError(file)
                case[key]=str(file)
    driver.dataset(manifest, smoke=False)
    if bulk_index is None:
        return manifest, {}
    groups=json.loads(bulk_index.read_text())['splits']
    required={(role,c['case']) for role,cases in manifest['splits'].items() for c in cases}
    labels={}
    for role,name in required:
        file=(bulk_index.parent/groups[role][name]).resolve()
        if not file.is_file():
            raise FileNotFoundError(file)
        labels[name]=str(file)
    return manifest, labels


def train(args, config):
    stable=config['family']=='stable_kernel'
    if not stable and args.bulk_index is None:
        raise ValueError('--bulk-index is required for FFT, sparse and CNN training')
    manifest, labels=dataset_inputs(args.manifest, args.bulk_index)
    if args.check_only:
        return dict(status='input_manifest_verified',variant=args.variant,
                    counts={k:len(v) for k,v in manifest['splits'].items()})
    if args.device!='cuda' or not torch.cuda.is_available():
        raise ValueError('Full-grid training requires CUDA. Use --case ch4 for the CPU example.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='xi-config-',dir=args.output.parent) as temp:
        directory=Path(temp)
        path=directory/'manifest.json'
        path.write_text(json.dumps(manifest))
        if stable:
            from experiments.xi_positive_total_pair.fit import main
            selection=SimpleNamespace(train_states=args.train_states,validation_states=None,
                                     test_states_per_species=args.test_states_per_species,
                                     smoke=False,smoke_states=3)
            subsets=driver.select_rows(manifest['splits'],selection)
            (directory/'selected_rows.json').write_text(json.dumps({role:{
                c['case']:subsets[c['case']].tolist() for c in group if subsets[c['case']].size}
                for role,group in manifest['splits'].items()}))
            charges=json.loads((Path(__file__).resolve().parent/'backend/charges_nuclear.json').read_text())
            (directory/'summary.json').write_text(json.dumps({'configuration':dict(
                manifest=str(path),manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                probes=152,seed=driver.SEED,charges=charges)}))
            sys.argv=[sys.argv[0],str(directory),str(args.output),'--channels',*map(str,CHANNELS),
                      '--train-microbatch',str(args.microbatch)]
            main()
            return dict(status='trained', output=str(args.output),checkpoint=f"{'base' if config['variant']=='base' else 'bounded_mlp'}.pt")
        total=config['output']=='intrinsic'
        factory=cnn if config['family']=='cnn3d' else mlp
        model, bulk, source, block=factory(config['variant'],total)
        original=driver.case_labels

        @contextmanager
        def label_adapter(case):
            with original(case) as data:
                yield TotalCase(data) if total else data

        def make_model():
            free,all_memory=torch.cuda.mem_get_info()
            budget=min(args.memory_gib*2**30,free-2*2**30,all_memory-2**30)
            if budget<4*2**30:
                raise ValueError('Insufficient CUDA memory for requested configuration')
            torch.cuda.set_per_process_memory_fraction(budget/all_memory)
            return model()

        def source_factory(*inputs):
            obj=source(*inputs)
            obj.directions_per_fft=args.fft_group
            return obj

        params=['--manifest',str(path),'--output',str(args.output),'--block',str(block),
                '--batch',str(args.batch),'--microbatch',str(args.microbatch),
                '--epochs',str(args.epochs),'--skip-train-eval']
        for key in ('train_states','test_states_per_species'):
            val=getattr(args,key)
            if val is not None:
                params.extend(['--'+key.replace('_','-'),str(val)])
        driver.case_labels=label_adapter
        driver.Metrics=SampledMetrics
        batch=measured_batch() if total else with_curvature(batch_factory)(24,stratified=True)
        driver.main(model_factory=make_model,case_factory=lambda c,d,dev:bulk(c,d,dev,labels),
                    source_factory=source_factory,batch_fn=batch,save_case_checkpoints=True,argv=params)
        return dict(status='trained',output=str(args.output))
