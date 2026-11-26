"""Losslessly split native NPZ exports along their field prefixes for Git hosting."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def group(key):
    if key.startswith('grid_weak_'):
        return 'grid_weak'
    if key.startswith('grid_'):
        return 'grid_base'
    if key.startswith('bulk_'):
        return 'bulk'
    return 'ppt'


def main(a):
    with np.load(a.source,allow_pickle=False) as src:
        pieces={}
        for k in src.files:
            pieces.setdefault(group(k),{})[k]=src[k]
    index=[]
    for name,arrays in sorted(pieces.items()):
        f=a.manifest.with_name(f'{a.manifest.stem}.{name}.npz')
        with f.open('xb') as stream:
            np.savez_compressed(stream,**arrays)
        with f.open('rb') as stream:
            h=sha256()
            while data:=stream.read(1<<20):
                h.update(data)
        index.append(dict(file=f.name,sha256=h.hexdigest(),bytes=f.stat().st_size))
    with a.manifest.open('x') as stream:
        json.dump(dict(format_version=1,parts=index),stream,indent=2)
        stream.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('manifest',type=Path)
    main(p.parse_args())
