"""Run release checks and a real optimizer update for every training mode."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import numpy as np
import torch

from . import __main__ as cli


def hashes(path):
    for piece in json.loads(path.read_text())['parts']:
        p=path.parent/piece['file']
        with p.open('rb') as f:
            check=hashlib.sha256()
            while data:=f.read(1<<20):
                check.update(data)
        if check.hexdigest()!=piece['sha256'] or p.stat().st_size!=piece['bytes']:
            raise ValueError(f'Invalid case shard {p}')


def main(a):
    torch.set_num_threads(1)
    a.out.mkdir(parents=True)
    paths=(cli.DEFAULT_TRAIN,cli.DEFAULT_VALIDATION)
    for path in paths:
        hashes(path)
    evidence=[]
    for dev in a.device:
        tag=a.out/dev
        cli.main(SimpleNamespace(command='check',device=dev,memory_gib=8.,case=list(paths)))
        for mode in ('ppt','c1','c2','guided','mutual','gated','no-peer'):
            destination=tag/mode
            cli.main(SimpleNamespace(command='train',device=dev,memory_gib=8.,
                train_case=[paths[0]],validation_case=[paths[1]],model=mode,epochs=1,lr=None,
                teacher=tag/'c2/weights.pt' if mode=='guided' else None,output=destination))
            out=json.loads((destination/'summary.json').read_text())
            train=out['training']
            assert len(train)==1 and out['validation'][0]['role']=='holdout'
            grad={k:v for k,v in train[0].items() if k.endswith('_gradient_norm')}
            assert grad and all(np.isfinite(v) and v>0 for v in grad.values())
            with np.load(destination/'validation_predictions.npz',allow_pickle=False) as pred:
                assert len(pred.files) and all(np.isfinite(pred[k]).all() for k in pred.files)
            evidence.append(dict(device=dev,model=mode,updated_branches=sorted(grad),
                                 loss_terms=sorted(k for k in train[0] if 'loss' in k or 'mse' in k)))
        cli.main(SimpleNamespace(command='evaluate',device=dev,memory_gib=8.,
            validation_case=[paths[1]],model='mutual',weights=tag/'mutual/weights.pt',
            output=tag/'reload_mutual'))
        final=json.loads((tag/'reload_mutual/summary.json').read_text())
        prior=json.loads((tag/'mutual/summary.json').read_text())
        assert final['validation']==prior['validation']
    summary=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                cuda_device=torch.cuda.get_device_name() if 'cuda' in a.device else None,
                cases=2,derivative_parity_comparisons_per_device=10,updates=evidence,
                checkpoint_reload='passed on all requested devices')
    (a.out/'verified.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(stage='verified',cases=2,configuration_runs=len(evidence),
                          devices=a.device),allow_nan=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--device',choices=('cpu','cuda'),action='append',required=True)
    main(p.parse_args())
