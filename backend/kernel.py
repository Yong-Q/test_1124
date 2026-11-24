"""Continuous real-space elemental Gaussian/SH fields; no Fourier projection."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import sph_harm

ELEMENTS=('C','H','N','O')


def box_lengths(cell):
    a=np.asarray(cell,float)
    if a.shape!=(3,3) or not np.isfinite(a).all() or not np.allclose(a,np.diag(np.diag(a)),atol=1e-10) or np.any(np.diag(a)<=0):
        raise ValueError('Only positive orthorhombic cells supported; triclinic requires a separate neighbor backend')
    return np.diag(a)


def grid(cell,spacing):
    lengths=box_lengths(cell)
    if not np.isfinite(spacing) or spacing<=0:raise ValueError('Positive spacing required')
    shape=tuple(np.maximum(1,np.ceil(lengths/spacing).astype(int)))
    axes=[(np.arange(n)+.5)*a/n for n,a in zip(shape,lengths)]
    q=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
    return q,shape,float(np.prod(lengths)/len(q))


def harmonics(vectors):
    v=np.asarray(vectors,float);r=np.linalg.norm(v,axis=-1)
    theta=np.arccos(np.clip(np.divide(v[:,2],r,out=np.ones_like(r),where=r>0),-1,1))
    phi=np.arctan2(v[:,1],v[:,0]);out=np.zeros((len(v),16))
    for l in range(4):
        for m in range(-l,l+1):
            z=sph_harm(abs(m),l,phi,theta)
            out[:,l*l+m+l]=z.real if m==0 else np.sqrt(2)*(-1.)**abs(m)*(z.real if m>0 else z.imag)
    return out


@dataclass(frozen=True)
class Basis:
    cutoff:float=5.
    radials:int=8

    def __post_init__(self):
        if not np.isfinite(self.cutoff) or self.cutoff<=0 or self.radials<2:raise ValueError('Positive cutoff and >=2 radial functions required')

    @property
    def channels(self):return self.radials*16

    def values(self,vectors):
        r=np.linalg.norm(vectors,axis=-1);t=np.minimum(r/self.cutoff,1.)
        centers=np.linspace(0,self.cutoff,self.radials);sigma=self.cutoff/(self.radials-1)
        envelope=(1-28*t**6+48*t**7-21*t**8)*(r<self.cutoff)
        radial=np.exp(-.5*((r[:,None]-centers)/sigma)**2)*envelope[:,None]
        y=harmonics(vectors)
        # Regular solid-harmonic factor makes the field continuous at r=0.
        for l in range(1,4):y[:,l*l:(l+1)**2]*=t[:,None]**l
        return (radial[:,:,None]*y[:,None,:]).reshape(len(r),-1)


def element_field(queries,positions,elements,cell,basis=Basis(),chunk=32):
    p=np.asarray(positions,float);q=np.asarray(queries,float);e=np.asarray(elements)
    if p.ndim!=2 or p.shape[1]!=3 or q.ndim!=2 or q.shape[1]!=3 or e.shape!=(len(p),):raise ValueError('Coordinates/elements shape mismatch')
    if not np.isfinite(p).all() or not np.isfinite(q).all() or not np.isin(e,np.arange(4)).all():raise ValueError('Invalid coordinates or element codes')
    lengths=None if cell is None else box_lengths(cell)
    if lengths is not None:
        if 2*basis.cutoff>=lengths.min():raise ValueError('cutoff must be below half the shortest box length')
        p=p%lengths;q=q%lengths
    tree=cKDTree(p,boxsize=lengths);out=np.zeros((len(q),4,basis.channels))
    for start in range(0,len(q),chunk):
        end=min(start+chunk,len(q));neighbors=tree.query_ball_point(q[start:end],basis.cutoff)
        sizes=np.array([len(x) for x in neighbors]);ii=np.repeat(np.arange(start,end),sizes)
        if not len(ii):continue
        jj=np.concatenate(neighbors).astype(int);v=p[jj]-q[ii]
        if lengths is not None:v-=lengths*np.rint(v/lengths)
        k=basis.values(v)
        np.add.at(out,(ii,e[jj]),k)
    return out


def combine(field,charges):
    z=[]
    for i,symbol in enumerate(ELEMENTS):
        value=charges.get(symbol)
        if value is None and np.any(field[:,i]!=0):raise ValueError('Missing charge weight for '+symbol)
        value=0. if value is None else float(value)
        if not np.isfinite(value):raise ValueError('Nonfinite weight')
        z.append(value)
    return np.einsum('qec,e->qc',field,z)


def voxel_density(positions,elements,cell,shape):
    lengths=box_lengths(cell);index=np.floor((positions%lengths)/lengths*np.array(shape)).astype(int)
    flat=np.ravel_multi_index(index.T,shape);count=np.zeros((int(np.prod(shape)),4))
    np.add.at(count,(flat,np.asarray(elements,int)),1.)
    return count/(np.prod(lengths)/len(count))
