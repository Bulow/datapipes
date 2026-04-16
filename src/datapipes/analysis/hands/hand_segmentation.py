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

    while data.ndim < 4:
        data = data.unsqueeze(0)

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


def crop_wrist(
    mask: torch.Tensor,
    radial_wrist_point: torch.Tensor,
    ulnar_wrist_point: torch.Tensor,
    index_metacarp_point: torch.Tensor,
) -> torch.Tensor:
    """
    Zero everything on the proximal side of the infinite line through the wrist points.

    The distal side is chosen as the side containing `index_metacarp_point`.
    """
    if mask.ndim == 3:
        mask_in = mask.unsqueeze(0)
        squeeze_batch = True
    elif mask.ndim == 4:
        mask_in = mask
        squeeze_batch = False
    else:
        raise ValueError(f"Expected mask shape (c, h, w) or (b, c, h, w), got {tuple(mask.shape)}")

    batch_size, _, h, w = mask_in.shape

    def _prepare_points(points: torch.Tensor, name: str) -> torch.Tensor:
        points = points.to(device=mask.device, dtype=torch.float32)
        if points.shape == (2,):
            points = points.unsqueeze(0)
        elif points.ndim != 2 or points.shape[-1] != 2:
            raise ValueError(f"{name} must have shape (2,) or (b, 2), got {tuple(points.shape)}")

        if points.shape[0] == 1:
            points = points.expand(batch_size, -1)
        elif points.shape[0] != batch_size:
            raise ValueError(f"{name} batch dimension must match mask batch size {batch_size}, got {points.shape[0]}")
        return points

    radial = _prepare_points(radial_wrist_point, "radial_wrist_point")
    ulnar = _prepare_points(ulnar_wrist_point, "ulnar_wrist_point")
    index_metacarp = _prepare_points(index_metacarp_point, "index_metacarp_point")

    line_vec = ulnar - radial
    if (line_vec == 0).all(dim=-1).any():
        raise ValueError("radial_wrist_point and ulnar_wrist_point must define a non-zero line")

    def signed_side(points_xy: torch.Tensor) -> torch.Tensor:
        rel = points_xy - radial
        return (line_vec[:, 0] * rel[:, 1]) - (line_vec[:, 1] * rel[:, 0])

    distal_sign = signed_side(index_metacarp)
    if torch.isclose(distal_sign, torch.zeros_like(distal_sign)).any():
        raise ValueError("index_metacarp_point must not lie on the wrist line")

    yy, xx = torch.meshgrid(
        torch.arange(h, device=mask_in.device, dtype=torch.float32),
        torch.arange(w, device=mask_in.device, dtype=torch.float32),
        indexing="ij",
    )
    grid_points = torch.stack((xx, yy), dim=-1).view(1, h * w, 2).expand(batch_size, -1, -1)
    rel = grid_points - radial.unsqueeze(1)
    signed_grid = ((line_vec[:, 0].unsqueeze(1) * rel[..., 1]) - (line_vec[:, 1].unsqueeze(1) * rel[..., 0])).view(batch_size, h, w)

    keep_mask = (signed_grid * distal_sign.view(batch_size, 1, 1)) >= 0
    cropped = mask_in * keep_mask.unsqueeze(1).to(dtype=mask_in.dtype)
    return cropped[0] if squeeze_batch else cropped


def crop_wrist_along_points(
    mask: torch.Tensor,
    wrist_cut_points: torch.Tensor,
    index_metacarp_point: torch.Tensor,
) -> torch.Tensor:
    """
    Zero everything on the proximal side of a wrist cut polyline.

    The kept side is chosen as the side containing `index_metacarp_point`.
    Supports masks with shape (c, h, w) or (b, c, h, w), and cut paths with
    shape (p, 2) or (b, p, 2).
    """
    if mask.ndim == 3:
        mask_in = mask.unsqueeze(0)
        squeeze_batch = True
    elif mask.ndim == 4:
        mask_in = mask
        squeeze_batch = False
    else:
        raise ValueError(f"Expected mask shape (c, h, w) or (b, c, h, w), got {tuple(mask.shape)}")

    batch_size, _, h, w = mask_in.shape

    wrist_cut_points = wrist_cut_points.to(device=mask_in.device, dtype=torch.float32)
    if wrist_cut_points.ndim == 2 and wrist_cut_points.shape[-1] == 2:
        wrist_cut_points = wrist_cut_points.unsqueeze(0)
    elif wrist_cut_points.ndim != 3 or wrist_cut_points.shape[-1] != 2:
        raise ValueError(f"wrist_cut_points must have shape (p, 2) or (b, p, 2), got {tuple(wrist_cut_points.shape)}")

    if wrist_cut_points.shape[1] < 2:
        raise ValueError("wrist_cut_points must contain at least two points")

    if wrist_cut_points.shape[0] == 1:
        wrist_cut_points = wrist_cut_points.expand(batch_size, -1, -1)
    elif wrist_cut_points.shape[0] != batch_size:
        raise ValueError(f"wrist_cut_points batch dimension must match mask batch size {batch_size}, got {wrist_cut_points.shape[0]}")

    index_metacarp = index_metacarp_point.to(device=mask_in.device, dtype=torch.float32)
    if index_metacarp.shape == (2,):
        index_metacarp = index_metacarp.unsqueeze(0)
    elif index_metacarp.ndim != 2 or index_metacarp.shape[-1] != 2:
        raise ValueError(f"index_metacarp_point must have shape (2,) or (b, 2), got {tuple(index_metacarp.shape)}")

    if index_metacarp.shape[0] == 1:
        index_metacarp = index_metacarp.expand(batch_size, -1)
    elif index_metacarp.shape[0] != batch_size:
        raise ValueError(f"index_metacarp_point batch dimension must match mask batch size {batch_size}, got {index_metacarp.shape[0]}")

    seg_start = wrist_cut_points[:, :-1, :]
    seg_stop = wrist_cut_points[:, 1:, :]
    seg_vec = seg_stop - seg_start
    seg_len2 = (seg_vec * seg_vec).sum(dim=-1)
    if (seg_len2 <= 0).all(dim=-1).any():
        raise ValueError("Each wrist cut path must include at least one non-zero segment")

    def _signed_side(points_xy: torch.Tensor) -> torch.Tensor:
        rel = points_xy.unsqueeze(1) - seg_start
        t = ((rel * seg_vec).sum(dim=-1) / seg_len2.clamp_min(1e-12)).clamp(0.0, 1.0)
        closest = seg_start + (t.unsqueeze(-1) * seg_vec)
        offset = points_xy.unsqueeze(1) - closest
        signed_dist = (seg_vec[..., 0] * offset[..., 1]) - (seg_vec[..., 1] * offset[..., 0])
        sq_dist = (offset * offset).sum(dim=-1)
        closest_seg_idx = sq_dist.argmin(dim=-1)
        return signed_dist.gather(1, closest_seg_idx.unsqueeze(-1)).squeeze(-1)

    distal_sign = _signed_side(index_metacarp)
    if torch.isclose(distal_sign, torch.zeros_like(distal_sign)).any():
        raise ValueError("index_metacarp_point must not lie on the wrist cut path")

    yy, xx = torch.meshgrid(
        torch.arange(h, device=mask_in.device, dtype=torch.float32),
        torch.arange(w, device=mask_in.device, dtype=torch.float32),
        indexing="ij",
    )
    grid_points = torch.stack((xx, yy), dim=-1).view(1, h * w, 2).expand(batch_size, -1, -1)

    rel = grid_points.unsqueeze(2) - seg_start.unsqueeze(1)
    t = ((rel * seg_vec.unsqueeze(1)).sum(dim=-1) / seg_len2.unsqueeze(1).clamp_min(1e-12)).clamp(0.0, 1.0)
    closest = seg_start.unsqueeze(1) + (t.unsqueeze(-1) * seg_vec.unsqueeze(1))
    offset = grid_points.unsqueeze(2) - closest
    signed_dist = (seg_vec[:, None, :, 0] * offset[..., 1]) - (seg_vec[:, None, :, 1] * offset[..., 0])
    sq_dist = (offset * offset).sum(dim=-1)
    closest_seg_idx = sq_dist.argmin(dim=-1, keepdim=True)
    signed_grid = signed_dist.gather(2, closest_seg_idx).squeeze(-1).view(batch_size, h, w)

    keep_mask = (signed_grid * distal_sign.view(batch_size, 1, 1)) >= 0
    cropped = mask_in * keep_mask.unsqueeze(1).to(dtype=mask_in.dtype)
    return cropped[0] if squeeze_batch else cropped

def get_hand_mask(frames: torch.Tensor) -> torch.Tensor:
    # Get mask based on boolean and morphological manipulation of std and mean
    # frames = torch.log(map01(frames) + 1e-5)
    # logger.info(f"{frames.shape = }")
    while frames.ndim < 5:
        frames = frames.unsqueeze(0)

    # Get normalized temporal std and mean 
    std, m = torch.std_mean(frames, dim=1)
    
    return get_hand_mask_from_std_mean(std=std, mean=m)

def get_hand_mask_from_std_mean(std: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    # Get mask based on boolean and morphological manipulation of std and mean
    
    std = map01(std)
    m = map01(mean)

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
