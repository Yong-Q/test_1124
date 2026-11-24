"""Fit the CNN's homogeneous element Hessian to the independent bulk matrices."""

from pathlib import Path

import h5py
import numpy as np
import torch

from experiments.xi_conv3d.network import FEATURES, SCALARS, STEPS


INDEX = (Path(__file__).resolve().parents[1]
         / 'xi_prl_bulk_coupling/bulk_elements/run_159428/bulk_labels_index.json')


class BulkCase:
    def __init__(self, case, data, device, labels):
        with np.load(labels[case['case']]) as reference, h5py.File(case['cache'], 'r') as h:
            steps = data['source_steps']
            np.testing.assert_array_equal(reference['source_steps'], steps)
            np.testing.assert_array_equal(reference['shape'], data['shape'])
            elements = h['elements'][:]
            codes = reference['element_codes']
            rho = reference['rho_element']
            moments = np.zeros((len(codes), 8), dtype=np.float64)
            marks = h['atomic_xi_nlm']
            for lo in range(0, len(marks), 20):
                values = marks[lo:lo+20, :, :, 0]
                for e, code in enumerate(codes):
                    moments[e] += values[:, elements == code].sum(axis=(0, 1), dtype=np.float64)
            moments /= (len(marks) * reference['element_counts'][:, None])
            self.inputs = self.tensor(np.column_stack((np.ones(len(codes)), np.eye(4)[codes], moments)),
                                      device)
            resolved = reference['reliable'].any(-1)
            keep = reference['reliable'][resolved]
            eig = reference['eigenvalues'][resolved]
            self.targets = torch.diag_embed(self.tensor(np.divide(
                1., eig, out=np.zeros_like(eig), where=keep), device))
            self.basis = self.tensor(reference['eigenvectors'][resolved], device, torch.complex64)
            self.pair_mask = self.tensor(keep[..., :, None] & keep[..., None, :], device, torch.bool)
            spacing = reference['lengths_A'] / reference['shape']
            phase = reference['wavevector_inv_A'][resolved] * spacing
            self.shells = self.tensor(np.column_stack(
                (np.ones(len(phase)), *[np.cos(step * phase).mean(-1) for step in STEPS])), device)
            self.rho = self.tensor(rho, device)

    @staticmethod
    def tensor(value, device, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=device)

    def bulk_loss(self, model):
        mixed = model.scalar_mix(self.inputs)
        homogeneous = self.rho @ mixed
        weights = model.spatial[:SCALARS]
        mean = torch.cat((homogeneous * weights.sum(1), homogeneous.new_zeros(FEATURES-SCALARS)))
        directions = torch.eye(FEATURES, dtype=mean.dtype, device=mean.device)[:SCALARS]
        _, curvature, _ = model.invariant_jets(mean, directions)
        transfer = self.shells @ weights.T
        d = (self.rho.sqrt()[:, None] * mixed)[None] * transfer[:, None, :]
        h = torch.einsum('kec,cd,kfd->kef', d, curvature, d)
        h = h + torch.eye(len(self.rho), device=h.device, dtype=h.dtype)
        pred = self.basis.mH @ h.to(self.basis.dtype) @ self.basis
        return ((pred - self.targets) * self.pair_mask).abs().square().sum() / self.targets.square().sum()
