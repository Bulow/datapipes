# import test_compression_method as test
import torch
import rich
from pathlib import Path
from datapipes.datapipe import DataPipe

from datapipes.manual_ops import with_manual_op

from datapipes import filters
# import test_compression_metrics as metrics

from datapipes.manual_ops import with_manual_op
from typing import Callable, Optional, Tuple

from datapipes.plotting import map01, plots
from datapipes.plotting.torch_colormap import TorchColormap
import matplotlib.pyplot as plt
from datapipes.utils import Slicer
from datapipes import sinks
import kornia
import einops

import logging
logger = logging.getLogger(__file__)

from datapipes.plotting import plot, crop_to_common_size

def cc_get_hist(frames: torch.Tensor, num_bins=256, min_val=0, max_val=0) -> torch.Tensor:
    frames = map01(frames)
    hist_original = torch.histc(frames.reshape(-1).to("cuda"), bins=num_bins, min=min_val, max=max_val)

    eps = 1e-12
    normalized_hist = (hist_original + eps) / (hist_original.sum() + eps)
    plt.figure(figsize=(50, 10))
    plt.plot(normalized_hist.cpu().numpy())

    return normalized_hist
    

def _get_hist(flat_frames: torch.Tensor) -> torch.Tensor:
    
    if flat_frames.dtype != torch.uint8:
        raise ValueError("Input tensor must be uint8")
    if flat_frames.dim() != 2:
        raise ValueError("Input tensor must be 2D")

    B, X = flat_frames.shape

    # indices must be long
    idx = flat_frames.to(torch.long)

    # output histogram
    hist = torch.zeros((B, 256), device=flat_frames.device, dtype=torch.int64)

    # source tensor must match idx shape
    src = torch.ones((B, X), device=flat_frames.device, dtype=hist.dtype)
    # print(f"{hist.shape = }, {src.shape = }, {idx.shape = }")
    # print(f"{hist.dtype = }, {src.dtype = }, {idx.dtype = }")
    # print(f"{type(hist) = }, {type(src) = }, {type(idx) = }")
    # print(f"{idx.max(1).values = }, {idx.min(1).values}")
    # print(src[0])
    hist.scatter_add_(1, idx.to(torch.int64), src.to(torch.int64))
    # print(hist)

    eps = 1e-12
    normalized_hist = (hist + eps) / (hist.sum(-1, keepdim=True) + eps)
    # plt.figure(figsize=(50, 10))
    # plt.plot(normalized_hist.cpu().numpy())
    return normalized_hist

# def _get_hist(frames: torch.Tensor, num_bins=256, min_val=0, max_val=0) -> torch.Tensor:
#     frames = map01(frames)
#     hist_original = torch.histc(frames.reshape(-1).to("cuda"), bins=num_bins, min=min_val, max=max_val)

#     values, counts = einops.rearrange((frames * 255.0).to(torch.uint8), "t c h w -> (c h w) t").unique(return_counts=True, dim=1)

#     print(f"{values.shape = }, {counts.shape = }")
#     raise RuntimeError()


#     eps = 1e-12
#     normalized_hist = (hist_original + eps) / (hist_original.sum() + eps)
#     # plt.figure(figsize=(50, 10))
#     # plt.plot(normalized_hist.cpu().numpy())

#     return normalized_hist

def _get_mask(data: torch.Tensor, upper_quantile_q_value: float=0.9) -> torch.Tensor:
    num_hist_bins = 256

    b, c, h, w = data.shape
    flat_frames = einops.rearrange(data, "t c h w -> t (c h w)").contiguous()
    hist = _get_hist((flat_frames * 255.0).to(torch.uint8))

    max_idx = torch.max(hist, dim=1).indices.unsqueeze(-1)

    upper_quantile = torch.quantile(flat_frames, q=upper_quantile_q_value, dim=1).unsqueeze(-1)

    upper_quantile_idx = torch.ceil(upper_quantile * num_hist_bins).to(torch.int)

    idx = torch.arange(256, device=hist.device).unsqueeze(0)
    roi_mask = (idx > max_idx) & (idx < upper_quantile_idx)


    float_hist = hist.to(torch.float32)
    float_hist.masked_fill_(~roi_mask, value=float("inf"))

    cutoff_idx = torch.min(float_hist, dim=1).indices

    cutoff_val = cutoff_idx / float(num_hist_bins)

    mask = (flat_frames > cutoff_val.unsqueeze(-1)).to(torch.uint8)
    mask = einops.rearrange(mask, "b (c h w) -> b c h w", c=c, h=h, w=w)

    return mask

def _clean_mask(mask: torch.Tensor) -> torch.Tensor:
    opened = kornia.morphology.opening(mask.to(torch.float32), torch.ones(size=(8, 8), dtype=torch.float32, device="cuda"), engine="convolution")
    dilated = kornia.morphology.dilation(opened, torch.ones(size=(8, 8), dtype=torch.float32, device="cuda"), engine="convolution").to(torch.uint8)
    opened_mask = (dilated * mask)
    closed = kornia.morphology.closing(opened_mask.to(torch.float32), torch.ones(size=(2, 2), dtype=torch.float32, device="cuda"), engine="convolution").to(torch.uint8)
    
    bc = kornia.filters.median_blur(map01(opened_mask.to(torch.float32)), kernel_size=3)

    mb = kornia.filters.box_blur(bc.to(torch.float32), kernel_size=7, border_type="constant")**2

    _m, _c, _o, _bc = crop_to_common_size((mb > 0.5), closed > 0, opened_mask > 0, bc > 0)

    t = torch.zeros_like(_c).to(torch.uint8)
    t[~_m] = _bc[~_m].to(torch.uint8)

    t[_m] = _o[_m].to(torch.uint8)

    smooth_mask = kornia.filters.box_blur(t.to(torch.float32), kernel_size=3)
    smooth_mask = (smooth_mask > 0.5).to(torch.uint8)
    
    # smoothed = kornia.filters.box_blur(t.to(torch.float32), kernel_size=7, separable=True, border_type="constant")
    # t = (smoothed > 0.5).to(torch.uint8)
    return smooth_mask

def get_hand_mask(frames: torch.Tensor) -> torch.Tensor:
    # Get mask based on boolean and morphological manipulation of std and mean
    # frames = torch.log(map01(frames) + 1e-5)
    # logger.info(f"{frames.shape = }")
    while frames.ndim < 5:
        frames = frames.unsqueeze(0)

    # Get normalized temporal std and mean 
    std, m = torch.std_mean(frames, dim=1)
    
    std = map01(std)
    m = map01(m)

    # Compute masks by thresholding with values based on histograms
    std_mask = _get_mask(std)
    m_mask = _get_mask(torch.sqrt(m+1e-6))

    # Union mask
    combined_mask = (m_mask | std_mask).to(torch.uint8)

    blurred_mask = kornia.filters.box_blur(input=map01(combined_mask.to(torch.float32)), kernel_size=3, separable=True)
    deblurred_mask = (blurred_mask > 0.5).to(torch.uint8)
    deblurred_mask, combined_mask = plots._pad_to_largest(deblurred_mask, combined_mask)
    mask = (deblurred_mask * combined_mask).to(torch.uint8)

    mask = _clean_mask(mask)
    return mask#.squeeze(0)


def apply_mask(mask: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    # mask = mask[0]
    def _apply_mask(frames: torch.Tensor) -> torch.Tensor:
        return frames * (mask.to(dtype=frames.dtype, device = frames.device))
    return with_manual_op(_apply_mask)

def get_mask_op(n_frames_per_mask: int=256):
    def _get_mask_op(frames: torch.Tensor) -> torch.Tensor:
        max_length = (len(frames) // n_frames_per_mask) * n_frames_per_mask
        frames = frames[:max_length]
        input_frames = einops.rearrange(frames, "(b t) 1 h w -> b t 1 h w", t=n_frames_per_mask)
        return get_hand_mask(input_frames)
    return with_manual_op(_get_mask_op, Slicer[::n_frames_per_mask, :, :, :])

def segment_datapipe(dp: DataPipe, idx: slice=slice(None), n_frames_per_mask: int=256) -> torch.Tensor:
    return sinks.accumulate(dp | get_mask_op(n_frames_per_mask=n_frames_per_mask), idx, batch_size=1, progress_bar=True)

def render_pretty_mask(image: torch.Tensor, mask: torch.Tensor, cmap: Optional[str]="viridis", mask_color_rgb01: Optional[Tuple[float]]=(0.7, 0.1, 0.3)) -> torch.Tensor:
    image = plots.qtile(image, quantile=(0.05, 0.95))
    cmapped = TorchColormap.apply(map01(image), cmap_name=cmap)
    # cmap = TorchColormap(cmap_name=cmap)
    # img.plots(cmapped)
    m = mask[0]
    r, g, b = [c for c in cmapped]

    r[m == 0] = mask_color_rgb01[0]
    g[m == 0] = mask_color_rgb01[1]
    b[m == 0] = mask_color_rgb01[2]
    pretty_masked = torch.stack([r, g, b])

    return pretty_masked
