"""Periodic free energy with longitudinal and transverse Xi responses."""
import torch
from torch import nn

from backend.kernel import harmonics
from .spatial import SpatialXi


def angular_basis(wavevectors, device):
    """Use the real spherical-harmonic convention of the Xi preprocessing."""
    y = torch.from_numpy(harmonics(wavevectors)).to(device=device)
    basis = []
    for ell in (1, 2, 3):
        block = y[:, ell**2:(ell+1)**2]
        basis.append((1j**ell)*(block/block.norm(dim=-1, keepdim=True)))
    return torch.cat(basis, dim=-1)


class AngularSpatial(SpatialXi):
    def __init__(self, scale, width=32, context_width=24):
        super().__init__(scale)
        if width != 32:
            self.local = nn.Linear(116, width)
            self.output = nn.Linear(width, 1, bias=False)
            with torch.no_grad():
                self.local.weight.mul_(scale.to(self.local.weight))
        if context_width != 24:
            self.context = nn.Linear(116, context_width)
            self.gate = nn.Linear(context_width, 8)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        self.angular_embedding = nn.Parameter(torch.empty(3, 8, 8))
        for block in self.angular_embedding:
            nn.init.orthogonal_(block)
        # Per l: longitudinal, transverse, and scalar-angular interaction.
        self.angular_mix = nn.Parameter(torch.zeros(3, 3, 8, 4))

    def forward(self, xi, spectrum, magnitude, angular_spectrum, basis,
                angular_magnitude, volume, mode_weight):
        return (self.value(xi) + self.spatial_energy(spectrum, magnitude, volume) +
                self.angular_energy(angular_spectrum, basis, angular_magnitude, volume, mode_weight))

    def angular_features(self, spectrum, basis):
        scalar = self.weighted_spectrum(spectrum[..., 0])
        features = []
        for row, ell in enumerate((1, 2, 3)):
            block = spectrum[..., ell**2:(ell+1)**2]
            embedded = self.angular_embedding[row].to(block.dtype) @ block
            longitudinal = (embedded * basis[:, None, ell**2-1:(ell+1)**2-1]).sum(-1)
            features.extend((longitudinal, embedded.flatten(-2),
                             scalar + longitudinal, scalar - longitudinal))
        return torch.cat(features, dim=-1)

    def angular_weights(self, magnitude):
        gaussian = torch.exp(-.5*(magnitude[:, None]*self.log_sigma.exp()[None])**2)
        weights = torch.einsum('ks,ltcs->kltc', gaussian, self.angular_mix)
        columns = []
        for row, ell in enumerate((1, 2, 3)):
            longitudinal, transverse, cross = weights[:, row].unbind(1)
            columns.extend((longitudinal-transverse,
                            transverse.repeat_interleave(2*ell+1, dim=-1),
                            .5*cross, -.5*cross))
        return torch.cat(columns, dim=-1)

    def angular_energy(self, spectrum, basis, magnitude, volume, mode_weight):
        features = self.angular_features(spectrum, basis)
        weights = self.angular_weights(magnitude)*mode_weight
        return (features.abs().square()*weights).sum()/volume

    def angular_jets(self, spectrum, tangents, basis, magnitude, volume, mode_weight):
        features = self.angular_features(spectrum, basis)
        directions = self.angular_features(tangents, basis)
        weights = self.angular_weights(magnitude)*mode_weight
        left = directions.flatten(1).conj()
        first = -2*(left @ (weights*features).flatten()).real/volume
        second = -2*(left @ (weights*directions).flatten(1).mT).real/volume
        return first, second
