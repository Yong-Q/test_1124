#!/usr/bin/env python3
"""Train, verify or generate 3D Xi perturbation inputs from one entry point."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'backend'))
VARIANTS=json.loads((ROOT/'variants.json').read_text())


def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('list',help='Print configurations and target definitions')
    sub.add_parser('doctor',help='Print program version, dependencies and CUDA availability')
    check=sub.add_parser('validate',help='Recalculate the reference labels')
    check.add_argument('--case',default='both',help='ch4, nh3, both, or a prepared case directory')
    check.add_argument('--states',type=Path)
    check.add_argument('--row',type=int,default=0)
    generate=sub.add_parser('perturb',help='Apply atomic cosines and periodic 3D wells to prepared data')
    generate.add_argument('--manifest',type=Path,required=True)
    generate.add_argument('--case',required=True,help='Training case identifier from the 152-probe manifest')
    generate.add_argument('--workers',type=int,default=1)
    generate.add_argument('--count',type=int,default=300)
    generate.add_argument('--seed',type=int,default=617)
    generate.add_argument('--output',type=Path,required=True)
    train=sub.add_parser('train',help='Train on a compact case or a full preprocessing manifest')
    train.add_argument('variant',choices=VARIANTS)
    selection=train.add_mutually_exclusive_group(required=True)
    selection.add_argument('--case',help='ch4 or a prepared training-case directory')
    selection.add_argument('--manifest',type=Path)
    train.add_argument('--bulk-index',type=Path)
    train.add_argument('--output',type=Path,required=True)
    train.add_argument('--states',type=Path)
    train.add_argument('--row',type=int,default=0)
    train.add_argument('--device',choices=('cpu','cuda'))
    train.add_argument('--epochs',type=int,default=1)
    train.add_argument('--batch',type=int,default=4)
    train.add_argument('--microbatch',type=int,default=1)
    train.add_argument('--fft-group',type=int,default=8)
    train.add_argument('--memory-gib',type=float,default=20.)
    train.add_argument('--threads',type=int,default=2)
    train.add_argument('--train-states',type=int)
    train.add_argument('--test-states-per-species',type=int)
    train.add_argument('--skip-validation',action='store_true',help='Skip the intermediate full-grid validation split')
    train.add_argument('--check-only',action='store_true')
    return parser


def main():
    parser=cli()
    args=parser.parse_args()
    if getattr(args,'row',0)<0:
        parser.error('--row must be nonnegative')
    if args.action=='list':
        result=VARIANTS
    elif args.action=='doctor':
        import platform
        import numpy
        import scipy
        import h5py
        import torch
        result=dict(code_version=(ROOT/'VERSION').read_text().strip(), python=platform.python_version(),
                    numpy=numpy.__version__,scipy=scipy.__version__,h5py=h5py.__version__,
                    torch=torch.__version__,torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available())
    elif args.action=='validate':
        from prepared import verify
        if args.states and args.case=='both':
            parser.error('--states requires a particular case')
        cases=('ch4','nh3') if args.case=='both' else (args.case,)
        result=[verify(case,args.states,args.row) for case in cases]
    elif args.action=='perturb':
        from preprocessing.atomic import run
        result=run(args.manifest,args.case,args.count,args.output,args.seed,args.workers)
    else:
        if min(args.epochs,args.batch,args.microbatch,args.fft_group,args.threads)<1:
            parser.error('Epochs, batch sizes, FFT group and threads must be positive')
        if not 0<args.memory_gib<=23:
            parser.error('--memory-gib must be in (0, 23]')
        if args.states and args.manifest:
            parser.error('--states is a compact-case option')
        args.output=args.output.resolve()
        if args.output.exists() and not args.check_only:
            parser.error(f'Output already exists: {args.output}')
        os.environ.setdefault('OMP_NUM_THREADS',str(args.threads))
        import torch
        torch.set_num_threads(args.threads)
        args.device=args.device or ('cpu' if args.case else 'cuda')
        config=VARIANTS[args.variant]
        if args.manifest:
            from production import train
            args.manifest=args.manifest.resolve()
            args.bulk_index=args.bulk_index.resolve() if args.bulk_index else None
            result=train(args,config)
        elif args.check_only:
            from prepared import verify
            result=verify(args.case,args.states,args.row)
        else:
            from demo import run
            result=run(args,config)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
