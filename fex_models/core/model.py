"""The frozen directional-shell SOG architectures and their observable derivatives."""
import torch

from .short_kernel import ShortSpatial


FIELDS=('local','radial','angular','short')
META=('wave','angular_wave','basis','short_kernels','mode_weight','volume')


class GridFunctional(ShortSpatial):
    # Cartesian squared modulus has smooth higher derivatives at zero.
    def spatial_energy(self,spectrum,wave,volume):
        x=self.weighted_spectrum(spectrum)
        return ((x.real.square()+x.imag.square())*self.transfer(wave)).sum()/volume

    def angular_energy(self,spectrum,basis,magnitude,volume,mode_weight):
        x=self.angular_features(spectrum,basis)
        return ((x.real.square()+x.imag.square())*self.angular_weights(magnitude)*mode_weight).sum()/volume


def energy(model,base,meta):
    x,r,a,s=base
    wave,angular_wave,basis,kernels,mode_weight,volume=meta
    return model(x,r,wave,a,basis,angular_wave,volume,mode_weight,s,kernels)


def first_order(model,fields,meta,training):
    """The negative chain-rule derivative over explicitly supplied tangents."""
    base=tuple(f[0].detach().requires_grad_() for f in fields)
    with torch.enable_grad():
        grad=torch.autograd.grad(energy(model,base,meta),base,create_graph=training)
    terms=[(f[1:].flatten(1)@g.flatten().conj()).real for f,g in zip(fields,grad)]
    return -sum(terms)


def jets(model,fields,meta,training):
    x,r,a,s=fields
    wave,angular_wave,basis,kernels,mode_weight,volume=meta
    c1,c2=model.jets(x[0],x[1:].transpose(0,1),training)
    values=(model.spatial_jets(r[0],r[1:],wave,volume),
            model.angular_jets(a[0],a[1:],basis,angular_wave,volume,mode_weight),
            model.short_jets(s[0],s[1:],kernels,volume))
    for y,h in values:
        c1,c2=c1+y,c2+h
    return c1,c2
