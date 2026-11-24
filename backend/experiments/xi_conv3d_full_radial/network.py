"""CNNs with every radial and spherical harmonic channel entering convolution."""

import math

import torch
from torch import nn

from experiments.xi_conv3d.network import ConvXi
from experiments.xi_conv3d_multiscale.network import (
    SHORT_WIDTHS_A, LONG_WIDTHS_A, SHORT_CUTOFF_A, LONG_CUTOFF_A, radial_bank,
)


RADIALS = 8
SCALARS = 13
ORDERS = (3, 5, 7)
ANGLES = tuple(RADIALS * m for m in ORDERS)
GROUPS = SCALARS + 3 * RADIALS
GRAM = RADIALS * (RADIALS + 1) // 2
INVARIANTS = SCALARS + 3 * GRAM


class FullRadialXi(ConvXi):
    """Keep n and m through convolution, then use full radial Gram invariants."""

    short_channels = 133

    def __init__(self, shape=None, long_channels=0):
        nn.Module.__init__(self)
        self.shape = shape
        self.long_channels = long_channels
        self.features = INVARIANTS + long_channels
        initial = torch.randn(GROUPS, 4) * .13
        initial[:, 0] += .5
        self.spatial = nn.Parameter(initial)
        self.readout = nn.Sequential(nn.Linear(self.features, 64), nn.Softplus(),
                                     nn.Linear(64, 64), nn.Softplus(), nn.Linear(64, 1))
        self.register_buffer('multiplicity', torch.tensor(
            [1] * SCALARS + [m for m in ORDERS for _ in range(RADIALS)]), persistent=False)
        self.register_buffer('triangle', torch.triu_indices(RADIALS, RADIALS), persistent=False)

    def encode(self, fields, density):
        marks = fields[..., 1:].unflatten(-1, (8, 16))
        scalars = torch.cat((fields[..., :1], density, marks[..., 0]), dim=-1)
        angle = [marks[..., l*l:(l+1)**2].flatten(-2) for l in range(1, 4)]
        return torch.cat((scalars, *angle), dim=-1)

    def short_weights(self):
        return self.spatial.repeat_interleave(self.multiplicity, dim=0,
                                              output_size=self.short_channels)

    def _stencil_weights(self, index):
        return self.short_weights()[:, index]

    def blocks(self, field):
        sizes = (SCALARS, *ANGLES, self.long_channels)
        scalar, *middle, tail = field.split(sizes, dim=-1)
        angular = tuple(a.unflatten(-1, (RADIALS, m)) for a, m in zip(middle, ORDERS))
        return scalar, angular, tail

    def angular_invariants(self, field):
        scalar, angles, tail = self.blocks(field)
        i, j = self.triangle
        grams = [(a @ a.transpose(-1, -2))[..., i, j] for a in angles]
        return torch.cat((scalar, *grams, tail), dim=-1)

    def feature_energy(self, field):
        features = self.angular_invariants(field)
        return (self.readout(features) - self.readout(features.new_zeros(self.features))).squeeze(-1)

    def jets(self, fields):
        mean = fields[:, :, 0]
        tangent = fields[:, :, 1:]
        _, angular, _ = self.blocks(mean)
        scalar, tangents, tail = self.blocks(tangent)
        i, j = self.triangle
        grads = []
        for a, da in zip(angular, tangents):
            term = da @ a.unsqueeze(-3).transpose(-1,-2)
            grads.append((term + term.transpose(-1,-2))[..., i, j])
        d = torch.cat((scalar, *grads, tail), dim=-1)
        first, hess, gradient = self.invariant_jets(self.angular_invariants(mean), d)
        for q, da in enumerate(tangents):
            selected = gradient[..., SCALARS + q*GRAM:SCALARS + (q+1)*GRAM]
            g = selected.new_zeros((*selected.shape[:-1], RADIALS, RADIALS))
            g[..., i, j] = selected
            s = (g + g.transpose(-1,-2)).unsqueeze(-3)
            hess = hess + da.flatten(-2) @ (s @ da).flatten(-2).transpose(-1,-2)
        return -first, -hess


class MultiscaleFullRadialXi(FullRadialXi):
    def __init__(self, shape=None, spacing=None):
        super().__init__(shape, long_channels=4)
        self.long_mix = nn.Parameter(torch.randn(4, len(LONG_WIDTHS_A)) * .1)
        if (shape is None) != (spacing is None):
            raise ValueError('Specify both shape and spacing for dense evaluation')
        if shape is not None:
            for name, widths, cutoff in (('short', SHORT_WIDTHS_A, SHORT_CUTOFF_A),
                                         ('long', LONG_WIDTHS_A, LONG_CUTOFF_A)):
                bank = radial_bank(shape, spacing, widths, cutoff, long=name == 'long')
                transfer = torch.fft.rfftn(bank, dim=(-3,-2,-1)).real
                self.register_buffer(name+'_transfer', transfer, persistent=False)

    def excess_density(self, xi, species):
        if self.shape is None or math.prod(self.shape) != xi.shape[0]:
            raise ValueError('Set the three-dimensional periodic grid')
        density = xi[..., 0] @ torch.nn.functional.one_hot(species, num_classes=4).to(xi)
        encoded = self.encode(xi.sum(1), density)
        centered = density - density.mean(0)

        def convolve(values, weights, basis):
            shaped = values.T.reshape(-1, *self.shape)
            spectrum = torch.fft.rfftn(shaped, dim=(-3,-2,-1))
            transfer = (weights @ basis.to(weights).flatten(1)).reshape(spectrum.shape)
            value = torch.fft.irfftn(transfer * spectrum, s=self.shape, dim=(-3,-2,-1))
            return value.flatten(1).T

        short = convolve(encoded, self.short_weights(), self.short_transfer)
        long = convolve(centered, self.long_mix, self.long_transfer)
        return self.feature_energy(torch.cat((short, long), dim=-1))
