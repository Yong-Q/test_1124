"""Continuous radial pair interactions between spherical harmonic fields."""

import torch
from torch import nn
from torch.nn import functional as F

from experiments.xi_conv3d_full_radial.network import ORDERS, SCALARS
from experiments.xi_conv3d_sixlayer.network import SixLayerMultiscaleXi, overlap, softplus_slope


class PairMLP(SixLayerMultiscaleXi):
    """Contract matching m components within each l to form scalar pair terms."""

    rank = 2
    short_channels = SCALARS + 2 * sum(ORDERS)

    def __init__(self, shape=None, spacing=None):
        super().__init__(shape, spacing)
        self.register_buffer('radial_mix', torch.tensor([[1.] * 8, [-1.] * 4 + [1.] * 4]) / 8)
        self.register_buffer('multiplicity', torch.tensor([1] * SCALARS + [m for m in ORDERS for _ in range(self.rank)]),
                             persistent=False)
        self.register_buffer('triangle', torch.triu_indices(self.rank, self.rank), persistent=False)
        self.gram_size = self.triangle.shape[1]
        self.features = SCALARS + len(ORDERS)*self.gram_size + self.long_channels
        self.spatial = nn.Parameter(self.spatial[:len(self.multiplicity)].detach().clone())
        self.layers = nn.ModuleList([nn.Linear(self.features, 32)])
        self.output = nn.Linear(32, 1, bias=False)
        del self.context
        self.angular_pair = nn.Parameter(torch.full((len(ORDERS)*self.gram_size,), .01))
        self.scalar_pair = nn.Parameter(torch.full((SCALARS+self.long_channels,), .01))

    def encode(self, fields, density):
        marks = fields[..., 1:].unflatten(-1, (8, 16))
        scalars = torch.cat((fields[..., :1], density, marks[..., 0]), -1)
        angular = [ (self.radial_mix @ marks[..., l*l:(l+1)**2]).flatten(-2)
                   for l in range(1, 4)]
        return torch.cat((scalars, *angular), -1)

    def blocks(self, field):
        scalar, *angular, long = field.split((SCALARS, *(self.rank*m for m in ORDERS), self.long_channels), -1)
        return scalar, tuple(a.unflatten(-1, (self.rank, m)) for a, m in zip(angular, ORDERS)), long

    def scalar_features(self, features):
        return torch.cat((features[..., :SCALARS], features[..., -self.long_channels:]), -1)

    def energy_density(self, fields):
        features = self.angular_invariants(fields)
        layer = self.layers[0]
        linear = self.output(F.softplus(layer(features)) - F.softplus(layer.bias)).squeeze(-1)
        angular = features[..., SCALARS:-self.long_channels] @ self.angular_pair
        scalar = .5 * (self.scalar_features(features).square() @ self.scalar_pair)
        return linear + angular + scalar

    def excess_density(self, xi, species):
        return self.energy_density(self.dense_first(xi, species))

    def tile_energy(self, fields, shape):
        return self.energy_density(fields[:, :, 0]).sum(1)

    def volume_jets(self, fields, shape):
        mean, tangent = fields[:, :, 0], fields[:, :, 1:]
        _, angles, _ = self.blocks(mean)
        scalar, angle_tangents, long = self.blocks(tangent)
        i, j = self.triangle
        gram_tangents = []
        for a, da in zip(angles, angle_tangents):
            product = da @ a.unsqueeze(-3).transpose(-1, -2)
            gram_tangents.append((product + product.transpose(-1, -2))[..., i, j])

        features = self.angular_invariants(mean)
        tangents = torch.cat((scalar, *gram_tangents, long), -1)
        layer = self.layers[0]
        slopes = softplus_slope(layer(features))
        readout = slopes * self.output.weight[0]
        first_gradient = F.linear(readout, layer.weight.T)
        angular_gradient = first_gradient[..., SCALARS:-self.long_channels] + self.angular_pair
        scalar_tangents = self.scalar_features(tangents)
        first = (tangents * first_gradient.unsqueeze(-2)).sum((1, 3))
        first += (torch.cat(gram_tangents, -1) * self.angular_pair).sum((1, 3))
        first += (scalar_tangents * (self.scalar_features(features)*self.scalar_pair).unsqueeze(-2)).sum((1, 3))

        hidden_tangents = F.linear(tangents, layer.weight)
        second = overlap(hidden_tangents, hidden_tangents * (readout*(1-slopes)).unsqueeze(-2))
        second = second + overlap(scalar_tangents, scalar_tangents * self.scalar_pair)
        for da, gradient in zip(angle_tangents, angular_gradient.split(self.gram_size, -1)):
            matrix = gradient.new_zeros((*gradient.shape[:-1], self.rank, self.rank))
            matrix[..., i, j] = gradient
            action = (matrix + matrix.transpose(-1, -2)).unsqueeze(-3) @ da
            second = second + overlap(da.flatten(-2), action.flatten(-2))
        return -first, -second
