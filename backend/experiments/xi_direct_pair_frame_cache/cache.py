"""Exact frame-space coefficients for the trainable direct radial pair kernel."""

import math

import numpy as np
import torch
from torch.nn.functional import one_hot

from experiments.xi_conv3d_full_radial.network import ORDERS, SCALARS
from experiments.xi_conv3d_local_el.train import LocalOccupancy
from experiments.xi_conv3d_152_sampled_c2.projection import SampledPrediction


def basis_products(basis, shape, dv):
    i, j = torch.triu_indices(len(basis), len(basis), device=basis.device)
    weights = basis[i]*basis[j]*(dv/math.prod(shape))
    weights[..., 1:(-1 if shape[-1] % 2 == 0 else None)] *= 2
    return weights.flatten(1), i, j


def gram(left, right, products):
    """Real, symmetrized Parseval contraction, streamed over frequencies."""
    frames = len(left)
    accumulated = products.new_zeros((len(products), frames, frames))
    for offset in range(0, left.shape[1], 32768):
        indices = slice(offset, offset+32768)
        a, b = left[:, indices], right[:, indices]
        weights = products[:, indices].unsqueeze(1)
        accumulated += (a.real.contiguous().unsqueeze(0) @ (b.real.unsqueeze(0)*weights).mT
                        + a.imag.contiguous().unsqueeze(0) @ (b.imag.unsqueeze(0)*weights).mT)
    return (accumulated + accumulated.mT)*.5


class FrameCache:
    @classmethod
    @torch.no_grad()
    def build(cls, model, source):
        instance = cls()
        shape = tuple(int(v) for v in source.shape)
        cells = math.prod(shape)
        frames = source.frames
        raw_voxels = torch.as_tensor(np.repeat(source.voxels, source.count_per_voxel),
                                     device=source.marks.device)
        free, _ = torch.cuda.mem_get_info()
        fft_bytes = min(4*2**30, free//12)
        frame_block = min(frames, 256, max(1, fft_bytes//(cells*source.marks.element_size())))
        deposition = []
        for lo in range(0, frames, frame_block):
            hi = min(lo+frame_block, frames)
            keep = ((source.frame_index >= lo) & (source.frame_index < hi)).nonzero().flatten()
            indices = (source.frame_index[keep]-lo)*cells + raw_voxels[keep]
            deposition.append((lo, hi, keep, indices))
        atoms = source.observations
        species = one_hot(source.elements, 4).to(source.marks)
        per_atom_species = species.repeat(frames, 1)
        per_atom = model.encode(torch.cat((torch.ones_like(source.marks[:, :1]), source.marks), -1),
                                per_atom_species)
        elements = species[atoms % source.atoms]
        encoded = per_atom.index_select(0, atoms)
        del per_atom, per_atom_species

        def spectrum(values, center=None):
            frequencies = math.prod(shape[:-1])*(shape[-1]//2+1)
            total = torch.empty((frames+1, frequencies), device=values.device,
                                dtype=torch.complex64 if values.dtype == torch.float32 else torch.complex128)
            for lo, hi, keep, indices in deposition:
                dense = values.new_zeros((hi-lo)*cells)
                dense.scatter_add_(0, indices, values[keep]*source.mass[keep])
                total[lo:hi] = torch.fft.rfftn(dense.reshape(hi-lo, *shape), dim=(-3, -2, -1)).flatten(1)
            if center is not None:
                total[:-1, 0] -= center*cells
            # Store the common field as a separate row to avoid cancellation
            # when contracting large, nearly identical frame fields with D.
            total[-1] = total[:-1].mean(0)
            total[:-1].sub_(total[-1:])
            return total

        short_basis = source.kernel_transfer(shape, False, encoded)
        long_basis = source.kernel_transfer(shape, True, encoded)
        short, instance.short_i, instance.short_j = basis_products(short_basis, shape, source.dv)
        long, instance.long_i, instance.long_j = basis_products(long_basis, shape, source.dv)
        instance.short_dc, instance.long_dc = short_basis[:, 0, 0, 0], long_basis[:, 0, 0, 0]
        short_grams, short_mass, long_mass = [], [], []
        for c in range(SCALARS):
            actual = spectrum(encoded[:, c])
            short_grams.append(gram(actual, actual, short))
            short_mass.append(actual[:, 0].real.clone()*source.dv)
            del actual
        start = SCALARS
        for order in ORDERS:
            parts = []
            for m in range(order):
                left, right = (spectrum(encoded[:, start+r*order+m]) for r in range(2))
                contribution = torch.stack((gram(left, left, short), gram(left, right, short),
                                            gram(right, right, short)))
                parts.append(contribution)
                del left, right
            short_grams.extend(torch.stack(parts).sum(0))
            start += 2*order
        long_grams = []
        for c in range(model.long_channels):
            actual = spectrum(elements[:, c], source.count_density[c])
            long_grams.append(gram(actual, actual, long))
            long_mass.append(actual[:, 0].real.clone()*source.dv)
            del actual
        instance.short_grams = torch.stack(short_grams)
        instance.long_grams = torch.stack(long_grams)
        instance.short_mass, instance.long_mass = torch.stack(short_mass), torch.stack(long_mass)
        instance.occupancy = LocalOccupancy.from_source(source)
        return instance

    @staticmethod
    def blend(left, right, i, j):
        return left[:, i]*right[:, j] + (i != j)*(left[:, j]*right[:, i])

    def operator(self, model):
        angular = model.spatial[SCALARS:].reshape(len(ORDERS), 2, -1)
        i, j = model.triangle
        spatial = model.spatial[:SCALARS]
        left = torch.cat((spatial, angular[:, i].flatten(0, 1)))
        right = torch.cat((spatial, angular[:, j].flatten(0, 1)))
        scale = torch.cat((model.scalar_pair[:SCALARS], 2*model.angular_pair))
        short = self.blend(left, right, self.short_i, self.short_j)*scale[:, None]
        long = self.blend(model.long_mix, model.long_mix, self.long_i, self.long_j)
        long = long*model.scalar_pair[SCALARS:, None]
        hessian = short.flatten() @ self.short_grams.flatten(0, 1).flatten(1)
        hessian = hessian + long.flatten() @ self.long_grams.flatten(0, 1).flatten(1)
        constant = -(model.reference[:SCALARS]*(spatial@self.short_dc)) @ self.short_mass
        constant -= (model.reference[SCALARS:]*(model.long_mix@self.long_dc)) @ self.long_mass
        return hessian.reshape(self.short_grams.shape[-2:]), constant

    def project(self, operator, coeff, columns=None):
        hessian, constant = operator
        weights, directions = coeff[:, :, 0], coeff[:, :, 1:]
        weights_aug = torch.cat((weights, weights.sum(1, keepdim=True)), 1)
        directions_aug = torch.cat((directions, directions.sum(1, keepdim=True)), 1)
        gradient_aug = weights_aug @ hessian + constant
        frame_gradient = gradient_aug[:, :-1] + gradient_aug[:, -1:]
        first = -(gradient_aug[:, None, :] @ directions_aug).squeeze(1)
        if columns is None:
            selected = directions_aug
        elif columns.ndim == 1:
            selected = directions_aug.index_select(-1, columns)
        else:
            selected = directions_aug.gather(2, columns[:, None, :].expand(-1, directions_aug.shape[1], -1))
        second = -(selected.mT @ (hessian @ selected))
        return first, second, frame_gradient

    def loss(self, model, coeff, targets, scales, ids, inverse, *, include_el=True):
        columns = torch.as_tensor(ids, device=coeff.device, dtype=torch.long)
        inverse_weights = torch.as_tensor(inverse, device=coeff.device, dtype=coeff.dtype)
        first, second, frame_gradient = self.project(self.operator(model), coeff, columns)
        if columns.ndim == 1:
            target2 = targets[1].index_select(1, columns).index_select(2, columns)
        else:
            batches = torch.arange(len(coeff), device=coeff.device)[:, None, None]
            target2 = targets[1][batches, columns[:, :, None], columns[:, None, :]]
        first_loss = ((first-targets[0])/scales[0]).square().mean()
        second_loss = (((second-target2)/scales[1]).square()*inverse_weights).sum()/targets[1].numel()
        loss, el = first_loss+second_loss, None
        if include_el:
            el, el_tangent = self.occupancy.loss_and_tangent(coeff[:, :, 0], frame_gradient)
            loss = loss+((frame_gradient-frame_gradient.detach())*el_tangent).sum()+el
        return loss, SampledPrediction(first, second, ids, inverse), el
