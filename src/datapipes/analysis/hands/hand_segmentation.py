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

from datapipes.plotting import plot, crop_to_common_size

def _get_hist(frames: torch.Tensor, num_bins=256, min_val=0, max_val=0) -> torch.Tensor:
    frames = map01(frames)
    hist_original = torch.histc(frames.reshape(-1).to("cuda"), bins=num_bins, min=min_val, max=max_val)

    eps = 1e-12
    normalized_hist = (hist_original + eps) / (hist_original.sum() + eps)
    # plt.figure(figsize=(50, 10))
    # plt.plot(normalized_hist.cpu().numpy())

    return normalized_hist

def _get_mask(data: torch.Tensor, num_hist_bins: int=256, upper_quantile_q_value: float=0.9) -> torch.Tensor:
    hist = _get_hist(data, num_bins=num_hist_bins)
    _, max_idx = torch.max(hist, dim=0)
    # print(max_val, max_idx)

    upper_quantile = torch.quantile(data, q=upper_quantile_q_value)
    upper_quantile_idx = torch.ceil(upper_quantile * num_hist_bins).to(torch.int).item()

    border_region = hist[max_idx:upper_quantile_idx]

    # plt.plot(border_region.cpu().numpy())

    _, cutoff_idx = torch.min(border_region, dim=0)

    cutoff_idx = max_idx + cutoff_idx
    cutoff_val = cutoff_idx / float(num_hist_bins)
    # print(cutoff_val, cutoff_idx)


    mask = (data > cutoff_val).to(torch.uint8)
    # test.plot(mask)
    return mask

def _clean_mask(mask: torch.Tensor) -> torch.Tensor:
    opened = kornia.morphology.opening(mask.to(torch.float32), torch.ones(size=(8, 8), dtype=torch.float32, device="cuda"), engine="convolution")
    dilated = kornia.morphology.dilation(opened, torch.ones(size=(8, 8), dtype=torch.float32, device="cuda"), engine="convolution").to(torch.uint8)
    opened_mask = (dilated * mask)
    closed = kornia.morphology.closing(opened_mask.to(torch.float32), torch.ones(size=(2, 2), dtype=torch.float32, device="cuda"), engine="convolution").to(torch.uint8)
    
    bc = kornia.filters.median_blur(map01(opened_mask.to(torch.float32)), kernel_size=11)

    mb = kornia.filters.box_blur(bc.to(torch.float32), kernel_size=31, border_type="constant")**2

    _m, _c, _o, _bc = crop_to_common_size((mb > 0.5), closed > 0, opened_mask > 0, bc > 0)

    t = torch.zeros_like(_c).to(torch.uint8)
    t[~_m] = _bc[~_m].to(torch.uint8)

    t[_m] = _o[_m].to(torch.uint8)
    
    return t

def get_hand_mask(frames: torch.Tensor) -> torch.Tensor:
    # Get mask based on boolean and morphological manipulation of std and mean

    # Get normalized temporal std and mean 
    std, m = torch.std_mean(frames, dim=0)
    std = map01(std)
    m = map01(m)

    # Compute masks by thresholding with values based on histograms
    std_mask = _get_mask(std)
    m_mask = _get_mask(torch.sqrt(m+1e-6))

    # Union mask
    combined_mask = (m_mask | std_mask).to(torch.uint8)

    # Denoise
    blurred_mask = filters.blurs.uniform_disk_blur(map01(combined_mask.to(torch.float32)), kernel_size=3)
    deblurred_mask = (blurred_mask > 0.5).to(torch.uint8)
    deblurred_mask, combined_mask = plots._pad_to_largest(deblurred_mask, combined_mask)
    mask = (deblurred_mask * combined_mask).to(torch.uint8)

    mask = _clean_mask(mask)
    return mask.squeeze(0)


def apply_mask(mask: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    # mask = mask[0]
    def _apply_mask(frames: torch.Tensor) -> torch.Tensor:
        return frames * (mask.to(dtype=frames.dtype, device = frames.device))
    return with_manual_op(_apply_mask)

def get_mask_op(n_frames_per_mask: int=256):
    return with_manual_op(get_hand_mask, Slicer[::n_frames_per_mask, :, :, :])

def segment_datapipe(dp: DataPipe, idx: slice=slice(None), n_frames_per_mask: int=256) -> torch.Tensor:
    return sinks.accumulate(dp | get_mask_op(n_frames_per_mask=n_frames_per_mask), idx, batch_size=1, progressbar=True)

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
