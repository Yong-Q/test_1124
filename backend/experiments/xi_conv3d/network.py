"""Learned 3D convolutions of scalar and real spherical-harmonic voxel fields."""

import math

import torch
from torch import nn
from torch.nn import functional as F



STEPS = (1, 3, 6)
AXES = (0, 1, 2)
SCALARS = 8
RADIAL_OUTPUTS = 2
ANGULAR = tuple((2 * l + 1) for l in range(1, 4) for _ in range(RADIAL_OUTPUTS))
FEATURES = SCALARS + len(ANGULAR)


class ConvXi(nn.Module):
    """Trainable separable spatial convolution followed by nonlinear 1×1 readout."""

    radials = 8

    def __init__(self, shape=None):
        super().__init__()
        self.shape = shape
        self.scalar_mix = nn.Linear(13, SCALARS, bias=False)
        self.angular_mix = nn.Parameter(torch.empty(3, RADIAL_OUTPUTS, 8))
        nn.init.uniform_(self.angular_mix, -.35, .35)
        initial = torch.randn(FEATURES, 1 + len(STEPS)) * .13
        initial[:, 0] += .5
        self.spatial = nn.Parameter(initial)
        self.readout = nn.Sequential(nn.Linear(FEATURES, 64), nn.Softplus(),
                                     nn.Linear(64, 64), nn.Softplus(), nn.Linear(64, 1))

    def encode(self, marks, element_density):
        # All n,l,m enter; weights for a given l and n are shared across m.
        channels = marks[..., 1:].unflatten(-1, (8, 16))
        scalar = self.scalar_mix(torch.cat((marks[..., :1], element_density, channels[..., 0]), dim=-1))
        angle = [torch.einsum('cn,...nm->...cm', w, channels[..., l*l:(l+1)**2]).flatten(-2)
                 for l, w in enumerate(self.angular_mix, 1)]
        return torch.cat((scalar, *angle), dim=-1)

    def _stencil_weights(self, index):
        w = self.spatial[:, index]
        return torch.cat((w[:SCALARS], *(part.expand(m) for part, m in zip(w[SCALARS:], ANGULAR))))

    def convolve_dense(self, encoded, shape):
        field = encoded.reshape(*shape, -1)
        value = field * self._stencil_weights(0)
        for shell, step in enumerate(STEPS, 1):
            neighbors = sum(torch.roll(field, shift, axis) for axis in AXES for shift in (-step, step))
            value = value + neighbors * (self._stencil_weights(shell) / 6.)
        return value

    def convolve_sparse(self, lifted, center, neighbors):
        zero = lifted.new_zeros((len(lifted), 1, *lifted.shape[2:]))
        padded = torch.cat((zero, lifted), dim=1)
        value = padded[:, center] * self._stencil_weights(0)
        for shell, indices in enumerate(neighbors, 1):
            nearby = padded[:, indices].sum(dim=1)
            value = value + nearby * (self._stencil_weights(shell) / 6.)
        return value

    def _logits(self, field):
        scalar, *angle = field.split((SCALARS, *ANGULAR), dim=-1)
        return torch.cat((scalar, *(a.square().sum(-1, keepdim=True) for a in angle)), dim=-1)

    def feature_energy(self, field):
        features = self._logits(field)
        return (self.readout(features) - self.readout(features.new_zeros(FEATURES))).squeeze(-1)

    def invariant_jets(self, features, d):
        """Chain-rule jets of the pointwise readout and its input gradient."""
        first, activation1, second, activation2, final = self.readout
        z1 = first(features)
        z2 = second(activation1(z1))
        s1 = torch.where(z1 > activation1.threshold, 1., z1.sigmoid())
        s2 = torch.where(z2 > activation2.threshold, 1., z2.sigmoid())
        dz1 = F.linear(d, first.weight)
        dz2 = F.linear(s1.unsqueeze(-2) * dz1, second.weight)
        adj2 = final.weight[0] * s2
        adj1 = adj2 @ second.weight
        first_dir = (dz2 * adj2.unsqueeze(-2)).sum(-1)
        curvature_1 = adj1 * s1 * (1. - s1)
        curvature_2 = adj2 * (1. - s2)
        hessian = (dz1 * curvature_1.unsqueeze(-2)) @ dz1.transpose(-1, -2)
        hessian = hessian + (dz2 * curvature_2.unsqueeze(-2)) @ dz2.transpose(-1, -2)
        gradient = (adj1 * s1) @ first.weight
        return first_dir, hessian, gradient

    def jets(self, fields):
        """Exact first and second negative energy derivatives in a direction tile."""
        mean = fields[:, :, 0]
        tangent = fields[:, :, 1:]
        scalar, *dangle = tangent.split((SCALARS, *ANGULAR), dim=-1)
        _, *angle = mean.split((SCALARS, *ANGULAR), dim=-1)
        d = torch.cat((scalar, *(2. * (a.unsqueeze(-2) * da).sum(-1, keepdim=True)
                                 for a, da in zip(angle, dangle))), dim=-1)
        first, hessian, gradient = self.invariant_jets(self._logits(mean), d)
        for i, da in enumerate(dangle):
            scale = (2. * gradient[..., SCALARS + i])[..., None, None]
            hessian = hessian + scale * (da @ da.transpose(-1, -2))
        return -first, -hessian

    def excess_density(self, xi, species):
        if self.shape is None or math.prod(self.shape) != xi.shape[0]:
            raise ValueError('Set the periodic three-dimensional grid shape')
        # Element grouping is a fixed projection of xi's density channel.
        groups = F.one_hot(species, num_classes=4).to(xi)
        element_density = xi[..., 0] @ groups
        encoded = self.encode(xi.sum(1), element_density)
        return self.feature_energy(self.convolve_dense(encoded, self.shape)).reshape(-1)

    def forward(self, xi, dv, external, chemical, species, thermal_volume=None):
        rho = xi[..., 0]
        volume = torch.as_tensor(dv, dtype=xi.dtype, device=xi.device)
        ext = (volume * (xi * external).sum(dim=(1, 2))).sum()
        chem = (volume * (rho * chemical).sum(dim=1)).sum()
        elements = rho @ F.one_hot(species, num_classes=4).to(xi)
        thermal = xi.new_ones(4) if thermal_volume is None else thermal_volume
        logarg = (elements * thermal).clamp_min(torch.finfo(xi.dtype).tiny)
        ideal = (volume * (elements * (logarg.log() - 1.)).sum(1)).sum()
        excess = (volume * self.excess_density(xi, species)).sum()
        return dict(ideal=ideal, excess=excess, external=ext, chemical=chem,
                    omega=ideal+excess+ext-chem)
