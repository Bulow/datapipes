import datapipes.utils.import_resource
import torch
from typing import Tuple, Literal, Optional, Callable
import math
import einops

def _load_stbn() -> torch.Tensor:
    return datapipes.utils.import_resource.load_tensor("stbn.pt").to("cuda")

_stbn_base = _load_stbn()

def stbn_like(frames: torch.Tensor) -> torch.Tensor:
    assert frames.ndim == 4 and frames.shape[1] == 1, f"Cannot generate stbn with shape={[s for s in frames.shape]}. Only (n 1 h w) noise is shipped with this library. For other shapes, see: https://github.com/NVIDIA-RTX/STBN/"
    shape: Tuple[int] = frames.shape
    stbn = _stbn_base.to(device=frames.device, dtype=frames.dtype)
    repeats = [math.ceil(s / n) for s, n in zip(shape, _stbn_base.shape)]
    n, c, h, w = repeats
    noise = einops.repeat(stbn, "n c h w -> (rn n) (rc c) (rh h) (rw w)", rn=n, rc=c, rh=h, rw=w)
    return noise[tuple(slice(0, s) for s in shape)]

import torch

def pink_like(x):
    """
    Generate pink noise with the same shape, dtype, and device as x.
    Assumes the FIRST dimension is the time dimension.
    """
    n = x.shape[0]
    device = x.device
    dtype = x.dtype

    # Frequency bins
    freqs = torch.fft.rfftfreq(n, d=1.0).to(device)
    freqs[0] = freqs[1]  # avoid divide-by-zero

    # Scale shape: (freqs, 1, 1, ...)
    scale = (1.0 / torch.sqrt(freqs)).view(
        -1, *([1] * (x.ndim - 1))
    )

    # White noise in frequency domain
    real = torch.randn(freqs.numel(), *x.shape[1:], device=device, dtype=dtype)
    imag = torch.randn(freqs.numel(), *x.shape[1:], device=device, dtype=dtype)
    spectrum = torch.complex(real, imag) * scale

    # Back to time domain (along dim=0)
    noise = torch.fft.irfft(spectrum, n=n, dim=0)

    # Normalize along time dimension
    noise = noise / noise.std(dim=0, keepdim=True)

    return noise


def exponential_like(tensor, rate=1.0):
    """
    Generate exponential noise with the same shape, dtype, and device as `tensor`.
    """
    dist = torch.distributions.Exponential(rate)
    return dist.sample(tensor.shape).to(device=tensor.device, dtype=tensor.dtype)



def gaussian_like(tensor, mean=0.0, std=1.0):
    """
    Generate Gaussian noise with the same shape, dtype, and device as `tensor`.
    """
    return torch.randn_like(tensor) * std + mean



def uniform_like(tensor, low=0.0, high=1.0):
    """
    Generate uniform noise with the same shape, dtype, and device as `tensor`.
    """
    return torch.rand_like(tensor) * (high - low) + low


def laplace_like(tensor, loc=0.0, scale=1.0):
    """
    Generate Laplace noise with the same shape, dtype, and device as `tensor`.
    """
    dist = torch.distributions.Laplace(loc, scale)
    return dist.sample(tensor.shape).to(device=tensor.device, dtype=tensor.dtype)



def rayleigh_like(tensor, sigma=1.0):
    """
    Generate Rayleigh noise with the same shape, dtype, and device as `tensor`.
    X = sigma * sqrt(-2 * log(1 - U))
    """
    u = torch.rand_like(tensor)
    return sigma * torch.sqrt(-2.0 * torch.log(1.0 - u))



def poisson_like(tensor, rate=1.0):
    """
    Generate Poisson noise with the same shape, dtype, and device as `tensor`.
    """
    return torch.poisson(torch.full_like(tensor, rate))


def multiplicative_noise_op(noise_like_func: Callable[[torch.Tensor], torch.Tensor], gain: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def _mul_noise(frames: torch.Tensor) -> torch.Tensor:
        raw_noise = (noise_like_func(frames) - 0.5) * 2 # Clip space [-1..1]
        gain_noise = 1 + (raw_noise * gain)

        return frames * (gain_noise)
    return _mul_noise