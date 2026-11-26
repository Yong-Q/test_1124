"""Loader for release input/output packets containing preprocessed Xi contractions."""
import json
from pathlib import Path

import numpy as np
import torch

from .core.model import FIELDS,META,first_order,jets


def read_packet(path):
    if path.suffix=='.json':
        manifest=json.loads(path.read_text())
        if manifest['format_version']!=1 or not manifest['parts']:
            raise ValueError(f'{path}: unsupported examples manifest')
        parts=[path.parent/entry['file'] for entry in manifest['parts']]
    else:
        parts=[path]
    fields={}
    for part in parts:
        with np.load(part,allow_pickle=False) as source:
            if set(source.files)&set(fields):
                raise ValueError(f'{part}: duplicate fields in this case')
            fields.update({k:source[k] for k in source.files})
    return fields


class Case:
    def __init__(self,path:Path,device='cpu'):
        d=read_packet(path)
        if int(d['format_version'])!=1:
            raise ValueError(f'{path}: unsupported input format')
        self.name=str(d['case_id'])
        self.role=str(d['role'])
        if self.role not in ('train','holdout'):
            raise ValueError(f'{path}: unknown case role {self.role}')
        self.d={k:torch.as_tensor(x,device=device) for k,x in d.items()
                if x.dtype.kind in 'biufc'}
        self.scale=self.d['feature_scale']
        if self.scale.shape!=(116,) or (self.scale<=0).any():
            raise ValueError(f'{path}: expected 116 positive feature scales')
        ne=self.d['bulk_reference'].shape[-1]
        self.shells=len(self.d['bulk_reference'])
        self.ne=ne
        self.draws=int(self.d['bulk_draws'])
        if self.draws!=2 or self.d['bulk_reference'].shape!=(self.shells,ne,ne):
            raise ValueError(f'{path}: expected two bulk draws and square element blocks')
        for name,n in (('ppt',len(self.d['ppt_y1'])),('grid_base',len(self.d['grid_element'])),
                       ('grid_weak',len(self.d['grid_element'])),
                       ('bulk',self.shells*ne*(self.draws+ne))):
            fields=self.fields(name)
            if len({x.shape[0] for x in fields})!=1 or fields[0].shape[0]!=1+n:
                raise ValueError(f'{path}: inconsistent derivative axes for {name}')
            if fields[0].shape[-2:]!=(8,16) or any(not torch.isfinite(x).all() for x in fields):
                raise ValueError(f'{path}: invalid 8 by 16 Xi fields in {name}')

    def fields(self,tag):
        return tuple(self.d[tag+'_'+k] for k in FIELDS)

    def meta(self,tag):
        return tuple(float(self.d[tag+'_'+k]) if k in ('mode_weight','volume') else
                     self.d[tag+'_'+k] for k in META)

    def ppt(self,net,training):
        return jets(net,self.fields('ppt'),self.meta('ppt'),training)

    def c1(self,net,tag,training):
        return first_order(net,self.fields('grid_'+tag),self.meta('grid'),training)

    def c2(self,net,training):
        packed=self.fields('bulk')
        meta=self.meta('grid')
        shell_count,ne,draws=self.shells,self.ne,self.draws
        pivot=1+shell_count*ne*draws
        _,affine=jets(net,tuple(x[:pivot] for x in packed),meta,training)
        nonlinear=first_order(net,tuple(torch.cat((x[:1],x[pivot:])) for x in packed),
                              meta,training).reshape(shell_count,ne,ne)
        blocks=[]
        for s in range(shell_count):
            slot=s*ne*draws
            blocks.append(torch.stack([affine[k:k+ne,k:k+ne]
                                       for k in range(slot,slot+ne*draws,ne)]))
        return (torch.stack(blocks)+nonlinear[:,None])/self.d['bulk_density_scale']

    def targets(self):
        return (self.d['grid_base_label'],self.d['grid_weak_label'],
                self.d['bulk_reference'])
