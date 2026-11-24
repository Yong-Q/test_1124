"""Periodic real-space pair kernels and exact derivatives of scalar energies."""
import numpy as np
import torch


MULTIPLETS = (slice(0, 1), slice(1, 2), slice(2, 5), slice(5, 10), slice(10, 17))
RADII = (2., 4., 6.)


def real_kernel(shape, spacing, radius, dtype, device):
    if radius >= min(n*h for n, h in zip(shape, spacing))/2:
        raise ValueError('Pair cutoff must be less than half the shortest periodic side')
    axes = [torch.arange(n, dtype=dtype, device=device) for n in shape]
    squared = [torch.minimum(x, n-x).square()*h*h for x, n, h in zip(axes, shape, spacing)]
    distance = (squared[0][:, None, None]+squared[1][None, :, None]+squared[2][None, None, :]).sqrt()
    t = distance/radius
    kernel = (1-t).clamp_min(0).pow(4)*(1+4*t.clamp_max(1))
    kernel[0, 0, 0] = 0
    return kernel


def cyclic_convolution(field, kernel):
    """Apply the sampled real-space kernel with no spectral truncation."""
    shape = kernel.shape
    return torch.fft.irfftn(torch.fft.rfftn(field, dim=(-3, -2, -1))*
                           torch.fft.rfftn(kernel), s=shape, dim=(-3, -2, -1))


def pair_energy(field, kernel, dv):
    return .5*dv**2*(field*cyclic_convolution(field, kernel)).sum()


def pair_jets(fields, spectra, dv):
    """Return minus gradient and Hessian projections along the supplied directions.

    Input [state, 1+directions, component, x, y, z]; each output column is
    one complete multiplet and one radial kernel.
    """
    shape = fields.shape[-3:]
    p = fields.shape[1]-1
    # Parseval with the multiplicity of each stored frequency in a real FFT.
    multiplicity = fields.new_full((shape[-1]//2+1,), 2)
    multiplicity[0] = 1
    if shape[-1] % 2 == 0:
        multiplicity[-1] = 1
    scale = -dv**2/np.prod(shape)
    first, second = [], []
    for group in MULTIPLETS:
        fourier = torch.fft.rfftn(fields[:, :, group].contiguous(), dim=(-3, -2, -1)).flatten(-3)
        features = torch.stack([s.flatten() for s in spectra])*scale
        features = (features.reshape(len(spectra), *spectra[0].shape)*multiplicity).flatten(1)
        for k in range(p):
            dot = (fourier[:, 0].conj()*fourier[:, k+1]).real.sum(dim=1)
            first.append(dot@features.T)
            for j in range(p):
                dot = (fourier[:, k+1].conj()*fourier[:, j+1]).real.sum(dim=1)
                second.append(dot@features.T)
    batch = fields.shape[0]
    return (torch.stack(first, dim=1).reshape(batch, len(MULTIPLETS), p, -1)
            .transpose(1, 2).flatten(2),
            torch.stack(second, dim=1).reshape(batch, len(MULTIPLETS), p, p, -1)
            .permute(0, 2, 3, 1, 4).flatten(3))


def local_jets(fields, dv):
    """Three local polynomial powers per multiplet, matched to 15 pair features."""
    first, second = [], []
    for group in MULTIPLETS:
        mean = fields[:, 0, group].flatten(-3)
        tangent = fields[:, 1:, group].flatten(-3)
        norm = mean.square().sum(1)
        dot = (tangent*mean[:, None]).sum(2)
        dd = torch.einsum('bkim,bjim->bkjm', tangent, tangent)
        for power in (1, 2, 3):
            factor = norm.pow(power-1)
            first.append(-dv*(dot*factor[:, None]).sum(-1))
            curvature = dd*factor[:, None, None]
            if power > 1:
                curvature = curvature + 2*(power-1)*norm.pow(power-2)[:, None, None]*dot[:, :, None]*dot[:, None]
            second.append(-dv*curvature.sum(-1))
    return torch.stack(first, dim=-1), torch.stack(second, dim=-1)
