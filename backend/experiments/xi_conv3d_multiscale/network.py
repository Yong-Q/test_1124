"""Dense two-scale CNN prototype with physical-radius periodic convolutions."""

import math

import torch
from torch import nn

from experiments.xi_conv3d.network import ANGULAR, FEATURES, SCALARS, ConvXi

SHORT_WIDTHS_A = (.5, .9, 1.5, 2.5)
LONG_WIDTHS_A = (4., 6.)
SHORT_CUTOFF_A = 5.
LONG_CUTOFF_A = 12.


def radial_bank(shape, spacing, widths, cutoff, long=False):
    """Mass-normalized, periodized kernels on the sampled orthorhombic grid."""
    limits = [math.ceil(cutoff / d) for d in spacing]
    indices = torch.meshgrid(*(torch.arange(-l, l + 1) for l in limits), indexing='ij')
    r = sum((i.double() * d).square() for i, d in zip(indices, spacing)).sqrt()
    u = (r / cutoff).clamp(max=1.)
    envelope = (1. - u).pow(3) * (1. + 3*u + 6*u**2 + 10*u**3 + 15*u**4 + 21*u**5)
    if long:
        turn = ((r - 4.) / 2.).clamp(0., 1.)
        envelope *= turn**3 * (10. - 15.*turn + 6.*turn.square())
    where = tuple((i % n).flatten() for i, n in zip(indices, shape))
    result = []
    for sigma in widths:
        weight = envelope * (-.5 * (r / sigma).square()).exp()
        if not bool((weight.sum() > 0.).item()):
            raise ValueError('No nonzero radial kernel samples')
        kernel = torch.zeros(shape, dtype=torch.float64)
        kernel.index_put_(where, (weight / weight.sum()).flatten(), accumulate=True)
        result.append(kernel)
    return torch.stack(result)


class MultiscaleConvXi(ConvXi):
    def __init__(self, shape=None, spacing=None):
        super().__init__(shape)
        if (shape is None) != (spacing is None):
            raise ValueError('Supply both the grid shape and spacing for dense evaluation')
        self.spacing = None if spacing is None else tuple(spacing)
        if shape is not None:
            short = radial_bank(shape, spacing, SHORT_WIDTHS_A, SHORT_CUTOFF_A)
            long = radial_bank(shape, spacing, LONG_WIDTHS_A, LONG_CUTOFF_A, long=True)
            self.register_buffer('short_transfer', torch.fft.rfftn(short, dim=(-3, -2, -1)).real,
                                 persistent=False)
            self.register_buffer('long_transfer', torch.fft.rfftn(long, dim=(-3, -2, -1)).real,
                                 persistent=False)
        self.register_buffer('multiplicity', torch.tensor([1] * SCALARS + list(ANGULAR)),
                             persistent=False)
        self.long_mix = nn.Parameter(torch.randn(4, len(LONG_WIDTHS_A)) * .1)
        self.readout = nn.Sequential(nn.Linear(FEATURES + 4, 64), nn.Softplus(),
                                     nn.Linear(64, 64), nn.Softplus(), nn.Linear(64, 1))

    def short_weights(self):
        return self.spatial.repeat_interleave(self.multiplicity, dim=0,
                                              output_size=SCALARS + sum(ANGULAR))

    def jets(self, fields):
        mean = fields[:, :, 0]
        tangent = fields[:, :, 1:]
        scalar, *dangle, long_d = tangent.split((SCALARS, *ANGULAR, 4), dim=-1)
        _, *angle, long = mean.split((SCALARS, *ANGULAR, 4), dim=-1)
        angular = [2. * (a.unsqueeze(-2) * da).sum(-1, keepdim=True)
                   for a, da in zip(angle, dangle)]
        d = torch.cat((scalar, *angular, long_d), dim=-1)
        invariant = torch.cat((super()._logits(mean[..., :-4]), long), dim=-1)
        first, hessian, gradient = self.invariant_jets(invariant, d)
        for i, da in enumerate(dangle):
            scale = 2. * gradient[..., SCALARS + i]
            hessian = hessian + scale[..., None, None] * (da @ da.transpose(-1, -2))
        return -first, -hessian

    def filtered_branches(self, marks, element_density):
        def convolve(fields, weights, basis):
            spectrum = torch.fft.rfftn(fields, dim=(-3, -2, -1))
            transfer = (weights @ basis.to(weights).flatten(1)).reshape(-1, *spectrum.shape[-3:])
            return torch.fft.irfftn(transfer * spectrum, s=self.shape, dim=(-3, -2, -1))

        short_input = self.encode(marks, element_density).T.reshape(-1, *self.shape)
        weights = self.short_weights()
        short = convolve(short_input, weights, self.short_transfer)
        centered = element_density - element_density.mean(0)
        long = convolve(centered.T.reshape(4, *self.shape), self.long_mix, self.long_transfer)
        return short.movedim(0, -1).reshape(-1, short.shape[0]), long.flatten(1).T

    def excess_density(self, xi, species):
        if self.shape is None or math.prod(self.shape) != xi.shape[0]:
            raise ValueError('xi does not match the periodic grid shape')
        density = xi[..., 0] @ torch.nn.functional.one_hot(species, num_classes=4).to(xi)
        short, long = self.filtered_branches(xi.sum(1), density)
        features = torch.cat((super()._logits(short), long), dim=-1)
        return (self.readout(features) - self.readout(features.new_zeros(FEATURES + 4))).squeeze(-1)
