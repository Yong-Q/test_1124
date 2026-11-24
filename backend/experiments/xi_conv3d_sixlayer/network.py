"""Six sequential, trainable 3D spatial layers with exact projected jets."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.xi_conv3d_full_radial.network import (
    FullRadialXi, RADIALS, SCALARS, GRAM,
)
from experiments.xi_conv3d_multiscale.network import (
    SHORT_CUTOFF_A, SHORT_WIDTHS_A, LONG_CUTOFF_A, LONG_WIDTHS_A, radial_bank,
)


DEPTH = 6
WIDTH = 64
HALO = DEPTH - 1
CENTER = (slice(None), slice(1, -1), slice(1, -1), slice(1, -1))


def softplus_slope(z):
    return torch.where(z > 20., 1., z.sigmoid())


def valid_conv(x, coeff):
    middle = x[CENTER]
    neighbors = (x[:, :-2, 1:-1, 1:-1] + x[:, 2:, 1:-1, 1:-1] +
                 x[:, 1:-1, :-2, 1:-1] + x[:, 1:-1, 2:, 1:-1] +
                 x[:, 1:-1, 1:-1, :-2] + x[:, 1:-1, 1:-1, 2:])
    return middle * coeff[:, 0] + neighbors * (coeff[:, 1] / 6.)


def valid_adjoint(x, coeff):
    mid = F.pad(x, (0, 0, 1, 1, 1, 1, 1, 1))
    neighbors = ((0,0,0,2,1,1,1,1), (0,0,2,0,1,1,1,1),
                 (0,0,1,1,0,2,1,1), (0,0,1,1,2,0,1,1),
                 (0,0,1,1,1,1,0,2), (0,0,1,1,1,1,2,0))
    n = sum(F.pad(x, padding) for padding in neighbors)
    return mid * coeff[:, 0] + n * (coeff[:, 1] / 6.)


def periodic_conv(x, coeff):
    n = sum(torch.roll(x, shift, dim) for dim in (1, 2, 3) for shift in (-1, 1))
    return x * coeff[:, 0] + n * (coeff[:, 1] / 6.)


def overlap(d, weighted):
    x = d.movedim(-2, 1).flatten(2)
    y = weighted.movedim(-2, 1).flatten(2)
    return x @ y.transpose(-1, -2)


class SixLayerXi(FullRadialXi):
    """The first convolution acts on all radial and angular input channels."""

    multiscale = False
    spherical = False

    def __init__(self, shape=None, spacing=None):
        super().__init__(shape, long_channels=4 if self.multiscale else 0)
        del self.readout
        if self.multiscale:
            self.long_mix = nn.Parameter(torch.randn(4, len(LONG_WIDTHS_A)) * .1)
        if self.spherical:
            if (shape is None) != (spacing is None):
                raise ValueError('Specify grid shape and spacing together for evaluation')
            if shape is not None:
                settings = [('short',SHORT_WIDTHS_A,SHORT_CUTOFF_A)]
                if self.multiscale:
                    settings.append(('long',LONG_WIDTHS_A,LONG_CUTOFF_A))
                for name,sigma,radius in settings:
                    grid = radial_bank(shape,spacing,sigma,radius,long=name=='long')
                    real = torch.fft.rfftn(grid,dim=(-3,-2,-1)).real
                    self.register_buffer(name+'_transfer',real,persistent=False)
        self.layers = nn.ModuleList(nn.Linear(a, WIDTH) for a in [self.features] + [WIDTH]*HALO)
        kernels = torch.randn(HALO, WIDTH, 2)*.04
        kernels[...,0] += .9
        kernels[...,1] += .1
        self.context = nn.Parameter(kernels)
        self.output = nn.Linear(WIDTH, 1)

    def dense_first(self, xi, species):
        if self.shape is None or math.prod(self.shape) != xi.shape[0]:
            raise ValueError('Set the three-dimensional periodic grid')
        rho = xi[...,0] @ F.one_hot(species, num_classes=4).to(xi)
        encoded = self.encode(xi.sum(1), rho)
        if not self.spherical:
            return self.convolve_dense(encoded, self.shape).reshape(-1, self.short_channels)

        def conv(values, weights, basis):
            fields = values.T.reshape(-1, *self.shape)
            spectrum = torch.fft.rfftn(fields, dim=(-3,-2,-1))
            transfer = (weights @ basis.to(weights).flatten(1)).reshape(spectrum.shape)
            return torch.fft.irfftn(transfer * spectrum, s=self.shape, dim=(-3,-2,-1)).flatten(1).T

        short = conv(encoded, self.short_weights(), self.short_transfer)
        if not self.multiscale:
            return short
        long = conv(rho-rho.mean(0), self.long_mix, self.long_transfer)
        return torch.cat((short,long), dim=-1)

    def excess_density(self, xi, species):
        fields = self.dense_first(xi, species)
        h = self.angular_invariants(fields).reshape(1,*self.shape,self.features)
        for i, layer in enumerate(self.layers):
            z = F.linear(h, layer.weight)
            if i:
                z = periodic_conv(z, self.context[i-1])
            h = F.softplus(z + layer.bias) - F.softplus(layer.bias)
        return F.linear(h,self.output.weight).flatten()

    def volume_jets(self, fields, shape):
        """Return the central tile's projected derivatives without a local P² tensor."""
        mean = fields[:,:,0]
        tangent = fields[:,:,1:]
        _, angular, _ = self.blocks(mean)
        scalar, tangents, tail = self.blocks(tangent)
        i, j = self.triangle
        contributions = []
        for a, da in zip(angular, tangents):
            u = da @ a.unsqueeze(-3).transpose(-1,-2)
            contributions.append((u+u.transpose(-1,-2))[...,i,j])
        b, _, p, _ = tangent.shape
        h = self.angular_invariants(mean).reshape(b,*shape,self.features)
        dh = torch.cat((scalar,*contributions,tail), dim=-1).reshape(b,*shape,p,self.features)
        dzs, slopes = [], []
        for level, layer in enumerate(self.layers):
            z = F.linear(h,layer.weight)
            dz = F.linear(dh,layer.weight)
            if level:
                c = self.context[level-1]
                z,dz = valid_conv(z,c),valid_conv(dz,c)
            z = z + layer.bias
            slope = softplus_slope(z)
            h = F.softplus(z)-F.softplus(layer.bias)
            dh = dz*slope.unsqueeze(-2)
            dzs.append(dz)
            slopes.append(slope)

        first = (dh @ self.output.weight[0]).sum((1,2,3))
        adj = self.output.weight[0]
        second = fields.new_zeros((b,p,p))
        for level in range(DEPTH-1,-1,-1):
            slope = slopes[level]
            dz = dzs[level]
            g = adj * slope
            second = second + overlap(dz, dz*(g*(1.-slope)).unsqueeze(-2))
            if level:
                g = valid_adjoint(g,self.context[level-1])
            adj = F.linear(g,self.layers[level].weight.T)

        grad = adj.flatten(1,3)
        for l, da in enumerate(tangents):
            picked = grad[..., SCALARS+l*GRAM:SCALARS+(l+1)*GRAM]
            matrix = picked.new_zeros((*picked.shape[:-1],RADIALS,RADIALS))
            matrix[...,i,j] = picked
            action = (matrix+matrix.transpose(-1,-2)).unsqueeze(-3) @ da
            second = second + overlap(da.flatten(-2), action.flatten(-2))
        return -first, -second


class SixLayerMultiscaleXi(SixLayerXi):
    multiscale = True
    spherical = True


class SixLayerSingleScaleXi(SixLayerMultiscaleXi):
    """The corresponding random initial model with the long branch removed."""

    def __init__(self, shape=None, spacing=None):
        super().__init__(shape,spacing)
        first = self.layers[0]
        self.multiscale = False
        self.features -= self.long_channels
        self.long_channels = 0
        del self.long_mix
        if hasattr(self,'long_transfer'):
            del self.long_transfer
        # Both runs use the same seed. Keep the dual model's random
        # initialization on every common weight, with no trained checkpoint.
        self.layers[0] = nn.Linear(self.features,first.out_features)
        with torch.no_grad():
            self.layers[0].weight.copy_(first.weight[:,:self.features])
            self.layers[0].bias.copy_(first.bias)
