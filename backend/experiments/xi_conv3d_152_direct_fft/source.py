"""Exact short-kernel gradient with a bounded, tile-local spectrum cache."""

import math

import torch

from experiments.xi_conv3d_152_tile_fft.source import TileFFTSource


def lift(part, tiers, channels, shape):
    b, _, p = part.shape
    field = part.new_zeros((math.prod(shape), b*p, channels))
    for destinations, frame_ids, encoded in tiers:
        directions = part[:, frame_ids].permute(1, 0, 3, 2).flatten(1, 2)
        field.index_copy_(0, destinations, torch.bmm(directions, encoded))
    return field.permute(1, 2, 0).reshape(b*p, channels, *shape)


def mix_adjoint(spectrum, cotangent, basis, shape):
    """Adjoint of irfftn(rfftn(x) * (mix @ basis)) for real basis spectra."""
    cross = (spectrum.conj() * torch.fft.rfftn(cotangent, dim=(-3, -2, -1))).real.sum(0)
    # rFFT stores only one element of the conjugate pair on interior planes.
    cross[..., 1:(-1 if shape[-1] % 2 == 0 else None)] *= 2
    return (cross.flatten(1) @ basis.flatten(1).T) / math.prod(shape)


class ShortKernelFFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, mix, basis, tiers, shape, window, directions_per_fft, cache_bytes):
        ctx.save_for_backward(coeff, basis)
        ctx.tiers, ctx.shape, ctx.window = tiers, shape, window
        ctx.group, ctx.channels = directions_per_fft, len(mix)
        ctx.spectra = {}
        b = coeff.shape[0]
        transfer = (mix @ basis.flatten(1)).reshape(len(mix), *basis.shape[-3:])
        outputs = []
        for group, part in enumerate(coeff.split(directions_per_fft, -1)):
            field = lift(part, tiers, len(mix), shape)
            chunks = []
            for start in range(0, len(mix), 32):
                end = min(start+32, len(mix))
                spectrum = torch.fft.rfftn(field[:, start:end].contiguous(), dim=(-3, -2, -1))
                size = spectrum.numel()*spectrum.element_size()
                if size <= cache_bytes:
                    ctx.spectra[group, start] = spectrum
                    cache_bytes -= size
                result = torch.fft.irfftn(spectrum*transfer[start:end], s=shape, dim=(-3, -2, -1))
                chunks.append(result[(..., *window)].flatten(2).transpose(1, 2))
            outputs.append(torch.cat(chunks, -1).reshape(b, part.shape[-1], -1, len(mix)))
        return torch.cat(outputs, dim=1).transpose(1, 2)

    @staticmethod
    def backward(ctx, output_grad):
        coeff, basis = ctx.saved_tensors
        shape, channels = ctx.shape, ctx.channels
        grad_mix = basis.new_zeros((channels, len(basis)))
        crop = tuple(s.stop-s.start for s in ctx.window)
        coeff_parts = coeff.split(ctx.group, -1)
        grad_parts = output_grad.split(ctx.group, 2)
        for group, (part, grad) in enumerate(zip(coeff_parts, grad_parts)):
            field = None
            for start in range(0, channels, 32):
                end = min(start+32, channels)
                spectrum = ctx.spectra.get((group, start))
                if spectrum is None:
                    if field is None:
                        field = lift(part, ctx.tiers, channels, shape)
                    spectrum = torch.fft.rfftn(field[:, start:end].contiguous(), dim=(-3, -2, -1))
                co = grad[..., start:end].permute(0, 2, 3, 1).reshape(-1, end-start, *crop)
                padded = grad.new_zeros((co.shape[0], end-start, *shape))
                padded[(..., *ctx.window)] = co
                grad_mix[start:end] += mix_adjoint(spectrum, padded, basis, shape)
        return None, grad_mix, None, None, None, None, None, None


class DirectFFTSource(TileFFTSource):
    cache_bytes = 0

    def shared_tile(self, model, coeff, lower, upper, halo, long):
        # The EL frame-weight derivative and other differentiable inputs use
        # the original autograd path, which returns their gradients as well.
        if long or coeff.requires_grad or not torch.is_grad_enabled() or not model.spatial.requires_grad:
            return super().shared_tile(model, coeff, lower, upper, halo, long)
        shape, tiers = self.prepare(model, coeff, lower, upper, halo, False)
        basis = self.kernel_transfer(shape, False, coeff)
        window = tuple(slice(h, h+n) for h, n in zip(halo, upper-lower))
        return ShortKernelFFT.apply(coeff, model.short_weights(), basis, tiers, shape, window,
                                    self.directions_per_fft, self.cache_bytes)
