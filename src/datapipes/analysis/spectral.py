import torch
import torch.fft as fft
import torch.nn.functional as F

import torch
from typing import Literal

window_name_type = Literal["hann2d", "raised_cosine", "uniform"]
default_window: window_name_type = "hann2d"

def get_window_hann2d(h: int, w: int, device="cuda") -> torch.Tensor:
    hann_h = torch.hann_window(h, periodic=True, device=device)
    hann_w = torch.hann_window(w, periodic=True, device=device)
    window2d = torch.outer(hann_h, hann_w)
    
    return window2d.unsqueeze(0)

def get_window_raised_cosine(h: int, w: int, device="cuda") -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, steps=h, device=device)
    x = torch.linspace(-1.0, 1.0, steps=w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt(xx**2 + yy**2)  # 0 at center, ~sqrt(2) at corners

    # Normalize radius so that radius <= 1 inside the unit circle
    # Values outside (corners) will be > 1.
    window = torch.zeros_like(radius, device=device)
    inside = radius <= 1.0
    r_inside = radius[inside]
    # Raised cosine: 1 at r=0, 0 at r=1
    window[inside] = 0.5 * (1.0 + torch.cos(r_inside * torch.pi))
    # Outside the unit circle: already 0

    return window.unsqueeze(0)

def window_2d_like(tensor: torch.Tensor, window_type: window_name_type=default_window) -> torch.Tensor:
    assert tensor.ndim >= 2, f"tensor must be at least 2D, got shape={tensor.shape}"
    h, w = tensor.shape[-2:]
    match window_type:
        case "hann2d":
            return get_window_hann2d(h, w, device=tensor.device)
        case "raised_cosine":
            return get_window_raised_cosine(h, w, device=tensor.device)
        case "uniform":
            return torch.ones(size=(1, h, w), device=tensor.device)
        case _:
            raise ValueError(f"Unsupported window type: \"{window_type}\"")





def center_crop(frames: torch.Tensor) -> torch.Tensor:
    # Ensure frame has shape (H, W)
    # original_ndim = frame.ndim
    # if frame.ndim == 4 and frame.shape[0] == 1:
    #     frame = frame[0]

    # if frame.ndim == 3 and frame.shape[0] == 1:
    #     frame_2d = frame[0]
    # elif frame.ndim == 2:
    #     frame_2d = frame
    # else:
    #     raise ValueError(
    #         f"Expected frame shape (1, H, W) or (H, W), got {tuple(frame.shape)}"
    #     )

    H, W = frames.shape[-2:]

    crop_h = crop_w = min(H, W)

    # Center crop
    start_y = (H - crop_h) // 2
    start_x = (W - crop_w) // 2
    end_y = start_y + crop_h
    end_x = start_x + crop_w
    crop = frames[..., start_y:end_y, start_x:end_x]
    return crop

def apply_window(tensor: torch.Tensor, window_type: window_name_type=default_window) -> torch.Tensor:
    cropped = center_crop(tensor)
    window = window_2d_like(cropped, window_type=window_type)
    return cropped * window

def windowed_psd(frame: torch.Tensor, window_type: window_name_type=default_window) -> torch.Tensor:
    frame = frame - frame.mean()
    windowed = apply_window(frame, window_type=window_type)
    
    # 2D FFT
    Ff = torch.fft.fft2(windowed)          # shape: (1, H, W)

    # Power spectrum = |F|^2
    psd = (Ff.real**2 + Ff.imag**2)  # or: psd = F.abs()**2

    # Normalize by number of pixels (one common convention)
    psd = psd / (windowed.shape[-1] * windowed.shape[-2])

    # Shift zero frequency to center
    psd = torch.fft.fftshift(psd, dim=(-2, -1))
    return psd


def windowed_fft2(frame: torch.Tensor, window_type: window_name_type=default_window) -> torch.Tensor:
    # frame = frame - frame.mean()
    windowed = apply_window(frame, window_type=window_type)

    # Compute 2D FFT using the windowed crop
    fft = torch.fft.fft2(windowed)
    fft_shifted = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shifted)
    while magnitude.ndim < original_ndim:
        magnitude = magnitude.unsqueeze(0)
    return magnitude

def windowed_fft2_log_magnitude(frame: torch.Tensor, window_type: window_name_type=default_window) -> torch.Tensor:
    # frame = frame - frame.mean()
    windowed = apply_window(frame, window_type=window_type)


    # Compute 2D FFT using the windowed crop
    fft = torch.fft.fft2(windowed)
    fft_shifted = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shifted)
    log_magnitude = torch.log1p(magnitude)
    while log_magnitude.ndim < original_ndim:
        log_magnitude = log_magnitude.unsqueeze(0)
    return log_magnitude


def radial_profile(mag2d: torch.Tensor) -> torch.Tensor:
    """
    Radial average of a 2D magnitude spectrum.
    mag2d: (H, W)
    returns: (R,) where R ≈ max radius
    """
    H, W = mag2d.shape[-2:]
    mag2d = mag2d[..., :, :]

    device = mag2d.device

    yy = torch.arange(H, device=device) - H // 2
    xx = torch.arange(W, device=device) - W // 2
    Y, X = torch.meshgrid(yy, xx, indexing='ij')
    R = torch.sqrt(X.float()**2 + Y.float()**2)

    r_int = R.long().view(-1)
    vals = mag2d.view(-1)

    nbins = int(r_int.max().item()) + 1
    sums = torch.bincount(r_int, weights=vals, minlength=nbins)
    counts = torch.bincount(r_int, minlength=nbins).clamp_min(1)

    return sums / counts


def resample_tiled_noise(base_noise: torch.Tensor,
                         out_h: int,
                         out_w: int,
                         scale: float) -> torch.Tensor:
    """
    Resample a base noise tile to (out_h, out_w) using periodic tiling
    and a continuous scale factor.

    base_noise: (H0, W0)
    scale     : spatial scale ( >0 ). scale>1 → zoom in; scale<1 → zoom out
    """
    device = base_noise.device
    H0, W0 = base_noise.shape

    base = base_noise.unsqueeze(0).unsqueeze(0)  # [1,1,H0,W0]

    # coordinates in output space
    yy = torch.arange(out_h, device=device)
    xx = torch.arange(out_w, device=device)
    Y, X = torch.meshgrid(yy, xx, indexing='ij')

    # Map to input coordinates (continuous), then wrap to tile
    Y_in = (Y.float() / scale) % 128
    X_in = (X.float() / scale) % 128

    # Normalize to [-1, 1] for grid_sample (align_corners=True)
    Y_n = ((Y_in / 128) * 2) - 1
    X_n = ((X_in / 128) * 2) - 1

    grid = torch.stack((X_n, Y_n), dim=-1).unsqueeze(0)  # [1,H,W,2]

    out = F.grid_sample(
        base, grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    return out[0, 0]  # (out_h, out_w)


def resample_noise_inverse_spectrum(
    base_noise: torch.Tensor,
    signal: torch.Tensor,
    n_scales: int = 25,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
    eps: float = 1e-6,
):
    """
    Find a spatial scale of the base_noise tile whose spectrum best matches
    the *inverse* spectral shape of the given signal, using resizing only.

    base_noise : (H0, W0) blue-noise tile (assumed tileable)
    signal     : (H, W)   2D signal of interest
    returns    : (H, W)   resampled noise
    """
    device = base_noise.device
    base_noise = base_noise.float()
    signal = signal.float().to(device)

    H, W = signal.shape

    # --- Target inverse radial spectrum from signal ---
    Sf = fft.fftshift(fft.fft2(signal))
    mag_S = torch.abs(Sf)
    prof_S = radial_profile(mag_S)

    inv_prof = 1.0 / (prof_S + eps)
    inv_prof = inv_prof / (inv_prof.max() + eps)  # normalize to [0,1]

    # --- Search over scales ---
    scales = torch.logspace(
        torch.log10(torch.tensor(min_scale)),
        torch.log10(torch.tensor(max_scale)),
        steps=n_scales
    )

    best_loss = None
    best_noise = None
    best_scale = None

    for s in scales:
        s = s.item()

        # Resample base noise with this scale (spatial operation only)
        n_resampled = resample_tiled_noise(base_noise, H, W, scale=s)

        # Spectrum of resampled noise
        Nf = fft.fftshift(fft.fft2(n_resampled))
        mag_N = torch.abs(Nf)
        prof_N = radial_profile(mag_N)

        # Match lengths
        L = min(len(inv_prof), len(prof_N))
        pN = prof_N[:L]
        t  = inv_prof[:L]

        # Normalize noise profile
        pN = pN / (pN.max() + eps)

        # L2 loss between radial profiles
        loss = torch.mean((pN - t) ** 2)

        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_noise = n_resampled
            best_scale = s

    print(f"best_scale: {best_scale}")

    # Optional: normalize output to [0,1]
    out = best_noise
    out = out - out.min()
    out = out / (out.max() + eps)

    return out


import torch


def filter_frame_radial_lowpass(
    frame: torch.Tensor,
    cutoff_radius: float = 0.5,
    transition_width: float = 0.1,
) -> torch.Tensor:
    """
    Apply a smooth radial low-pass filter in the frequency domain and return
    the filtered frame in image space.

    Parameters
    ----------
    frame : torch.Tensor
        Input frame of shape (1, H, W) or (H, W), real-valued.
    cutoff_radius : float, optional
        Normalized cutoff radius in [0, 1].
        0 -> only DC; 1 -> keep almost all frequencies.
    transition_width : float, optional
        Width of the cosine roll-off (normalized radius units).
        Frequencies in [cutoff_radius - transition_width/2,
                        cutoff_radius + transition_width/2]
        are smoothly attenuated from 1 -> 0.

    Returns
    -------
    filtered_frame : torch.Tensor
        Filtered frame with same shape as input.
    """

    # Handle shape (1, H, W) vs (H, W)
    add_channel_dim = False
    if frame.ndim == 3 and frame.shape[0] == 1:
        frame_2d = frame[0]
        add_channel_dim = True
    elif frame.ndim == 2:
        frame_2d = frame
    else:
        raise ValueError(
            f"Expected frame shape (1, H, W) or (H, W), got {tuple(frame.shape)}"
        )

    frame_2d = frame_2d.to(dtype=torch.float32)
    device = frame_2d.device
    H, W = frame_2d.shape

    # Forward FFT (frequency domain)
    fft = torch.fft.fft2(frame_2d)
    fft_shifted = torch.fft.fftshift(fft)

    # --- Build smooth radial low-pass mask ---
    # Coordinate grid in [-1, 1] (arbitrary normalized units)
    y = torch.linspace(-1.0, 1.0, steps=H, device=device)
    x = torch.linspace(-1.0, 1.0, steps=W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt(xx**2 + yy**2)

    # Normalize radius to [0, 1] (1 ~ farthest corner)
    radius = radius / (radius.max() + 1e-8)

    # Clamp parameters
    cutoff_radius = float(cutoff_radius)
    transition_width = max(float(transition_width), 1e-6)  # avoid div by zero

    cutoff_radius = max(0.0, min(1.0, cutoff_radius))
    transition_width = max(0.0, min(1.0, transition_width))

    r1 = cutoff_radius - transition_width / 2.0
    r2 = cutoff_radius + transition_width / 2.0

    # Ensure sensible ordering
    r1 = max(0.0, r1)
    r2 = min(1.0, r2)
    if r2 <= r1:  # degenerate case => hard cutoff
        mask = (radius <= cutoff_radius).float()
    else:
        mask = torch.zeros_like(radius)

        # Passband: radius <= r1 -> 1
        pass_region = radius <= r1
        mask[pass_region] = 1.0

        # Stopband: radius >= r2 -> 0 (already default)

        # Transition band: r1 < radius < r2
        trans_region = (radius > r1) & (radius < r2)
        t = (radius[trans_region] - r1) / (r2 - r1)  # t in (0, 1)
        # Raised cosine from 1 -> 0
        mask[trans_region] = 0.5 * (1.0 + torch.cos(torch.pi * t))

    # Apply mask in frequency domain
    filtered_fft_shifted = fft_shifted * mask

    # Inverse FFT, back to image domain
    filtered_fft = torch.fft.ifftshift(filtered_fft_shifted)
    filtered_spatial = torch.fft.ifft2(filtered_fft)

    # Take real part (input is real so imag part is numerical noise)
    filtered_2d = filtered_spatial.real

    # Match original shape
    if add_channel_dim:
        filtered_frame = filtered_2d.unsqueeze(0)  # (1, H, W)
    else:
        filtered_frame = filtered_2d  # (H, W)

    return filtered_frame
