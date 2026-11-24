"""Assemble every response direction together with grouped matrix products."""

import numpy as np
import torch

from train_xi_full import SparseTrajectory


class ProjectionTrajectory(SparseTrajectory):
    def __init__(self, case, data, charges, device, block):
        super().__init__(case, data, charges, device)
        sizes = np.diff(self.offsets)
        powers = np.ceil(np.log2(sizes)).astype(int)
        self.tiles = []
        for power in np.unique(powers):
            groups = np.flatnonzero(powers == power)
            for start in range(0, len(groups), block):
                selected = groups[start:start + block]
                columns = np.arange(sizes[selected].max())
                active = columns < sizes[selected, None]
                rows = self.offsets[selected, None] + columns * active
                indices = torch.as_tensor(rows, device=device)
                observations = self.multi_observation[indices]
                frames = self.multi_frame[indices]
                mass = self.multi_mass[indices] * torch.as_tensor(active, device=device)
                self.tiles.append((observations, frames, mass / self.dv))

    def chunks(self, coeff, block):
        for observations, frames, mass in self.tiles:
            marks = self.multi_marks[observations] * mass[..., None]
            weights = coeff[:, frames].permute(1, 0, 3, 2).flatten(1, 2)
            fields = torch.bmm(weights, marks)
            yield fields.unflatten(1, (coeff.shape[0], coeff.shape[2])).transpose(0, 1)
