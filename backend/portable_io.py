"""Load serialized ξ inputs and reconstruct probe tangents on demand."""
import h5py
import numpy as np


def read_grid_inputs(case, shape, steps, charges):
    with h5py.File(case['xi_inputs'], 'r') as handle:
        if (not np.array_equal(handle.attrs['shape'], shape)
                or not np.array_equal(handle['step'][:], steps)
                or dict(zip('CHNO', handle.attrs['nuclear_weights'])) != charges):
            raise ValueError('Portable ξ input metadata mismatch for ' + case['case'])
        return tuple(handle[key][:] for key in ('voxel', 'mass', 'marks', 'cell'))


class ResponseWeights:
    def __init__(self, data):
        self.weights = data['weights']
        self.derivatives = data['dweights'] if 'dweights' in data else None
        self.probes = data['probe_energy'] if self.derivatives is None else None

    def select(self, rows, directions):
        if self.derivatives is not None:
            return np.take_along_axis(self.derivatives[rows], directions[:, None, :], axis=2)
        weights = self.weights[rows]
        centered = self.probes.T[directions].transpose(0, 2, 1)
        centered -= np.einsum('bt,btp->bp', weights, centered)[:, None, :]
        return -weights[:, :, None] * centered
