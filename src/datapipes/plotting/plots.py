from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
from icecream import ic
from torchvision.transforms.functional import to_pil_image
import torch
import einops
import numpy as np

from imageio import imwrite

from datapipes.sic import sic

from pathlib import Path

import math

from datapipes.plotting.torch_colormap import TorchColormap

from typing import Optional, Tuple, List, Literal, Callable, Any

import plotly.express as px

def animate(frames: torch.Tensor, cmap: str = "viridis"):
    fig = px.imshow(
        frames.squeeze(1).cpu().numpy(), 
        animation_frame=0, 
        binary_string=True, 
        labels=dict(animation_frame="frame"), contrast_rescaling="minmax", 
        color_continuous_scale=cmap
    )
    fig.show()

def _map01_torch(frames: torch.Tensor, eps=1e-8):
    '''
    Map values in `frames` to `[0, 1]`
    
    Where the minimum value is mapped to `0` and the maximum value is mapped to `1`
    '''
    # if frames.device == "meta":
    #     return frames

    min_val = frames.min()
    max_val = frames.max()

    # if (min_val == max_val):
    #     return frames
    
    return (frames.to(torch.float32) - min_val) / ((max_val - min_val) + eps)

def map01(frames: torch.Tensor|np.ndarray):
    '''
    Map values in `frames` to `[0, 1]`
    
    Where the minimum value is mapped to `0` and the maximum value is mapped to `1`
    '''
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    
    return _map01_torch(frames)

def _clip_space(frames: torch.Tensor|np.ndarray) -> torch.Tensor:
    '''
    Map to [-1:1]
    '''
    return (_map01_torch(frames) - 0.5) * 2

def _plot(frame, return_image=False, show_image=True):
    '''
    Display `frame` as an image
    '''
    image = to_pil_image(map01(frame))
    if show_image:
        display(image)
    if return_image:
        return image

def _grid_reshape(batch):
    n = batch.shape[0]  # number of images to show
    grid_cols = math.ceil(math.sqrt(n))
    grid_rows = math.ceil(n / grid_cols)
    pad = grid_rows * grid_cols - n
    if pad > 0:
        pad_imgs = (torch.ones_like(batch[0]) * batch.max()).expand(pad, -1, -1, -1)
        imgs = torch.cat([batch, pad_imgs], dim=0)
    else:
        imgs = batch
    grid = einops.rearrange(imgs, "(r c) n h w -> n (r h) (c w)", r=grid_rows, c=grid_cols)
    return grid

def _plot_grid(batch, return_image=False, show_image=True):
    return _plot(frame=_grid_reshape(batch), return_image=return_image, show_image=show_image)

# @torch.no_grad()
def qtile(tensor: torch.Tensor, quantile: tuple[float], output_bytes=False):
    def get_qtile(tensor: torch.Tensor, min_max_quantiles: tuple=(0.05, 0.95)):
        lower, upper = min_max_quantiles
        tensor = tensor.to(torch.float32)
        min_max = torch.Tensor([lower, upper]).to(tensor)
        q = torch.quantile(tensor, min_max)       
        return (q[0], q[1])

    dev = tensor.device
    tensor.to("cuda")
    quantile = get_qtile(tensor, quantile)
    tensor = tensor.clip(*quantile)
    if output_bytes:
        tensor = (((tensor - quantile[0]) / (quantile[1] - quantile[0])) * 255).to(torch.uint8)
    return tensor.to(dev)

def _pad_to_largest(*tensors: torch.Tensor) -> Tuple[torch.Tensor]:
    # Pad tensors to match the largest frame size
    max_height = max([tensor.shape[-2] for tensor in tensors])
    max_width = max([tensor.shape[-1] for tensor in tensors])

    padded_tensors = []
    for tensor in tensors:
        height, width = tensor.shape[-2], tensor.shape[-1]
        pad_top = (max_height - height) // 2
        pad_bottom = max_height - height - pad_top
        pad_left = (max_width - width) // 2
        pad_right = max_width - width - pad_left
        padded_tensors.append(torch.nn.functional.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom)))
    assert all(t.shape[-2:] == t[0].shape[-2:] for t in padded_tensors), f"All padded tensors must have the same height and width, got shapes: ({", ".join([f"{t.shape}" for t in padded_tensors])})"
    return padded_tensors

    
def plot_raw(
        *tensors: torch.Tensor, 
        map01_individually=True, 
        quantiles: Optional[tuple[float]]=None, 
        cmap: Optional[str|TorchColormap]=None, 
        mode: Literal["grid", "horizontal", "vertical"]="grid",
        return_image=False, 
        show_image=True
    ):

    # Convert all to torch.Tensor
    tensors = [torch.from_numpy(t) if isinstance(t, np.ndarray) else t for t in tensors]

    # Ensure all have a channel dimension
    tensors = [t.unsqueeze(0) if t.ndim == 2 else t for t in tensors]

    # Ensure all have a batch dimension
    tensors = [t.unsqueeze(0) if len(t.shape) == 3 else t for t in tensors]
    
    # Clip all to quantiles
    if quantiles is not None:
        tensors = [qtile(t, quantiles) for t in tensors]
    
    # Remap to [0..1]
    if map01_individually:
        tensors = [map01(t) for t in tensors]

    # Pad all to larges (H W) size
    tensors = _pad_to_largest(*tensors)
    
    # Apply color map to all
    if cmap is not None:
        if isinstance(cmap, str):
            cmap = TorchColormap(cmap)
        tensors = [cmap(t) for t in tensors]

    tensors = [t.unsqueeze(0) if len(t.shape)==3 else t for t in tensors]
    
    match mode:
        case "grid":
            tensors = torch.cat(tensors)
            _plot_grid(tensors, return_image=return_image, show_image=show_image)
        case "horizontal":
            # _plot(einops.rearrange(tensors, "n c h w -> c h (w n)"))
            tensors = torch.cat(tensors, dim=-1)[0]
            _plot(tensors)
        case "vertical":
            tensors = torch.cat(tensors, dim=-2)[0]
            print(tensors.shape)
            _plot(tensors)
            # _plot(einops.rearrange(tensors, "n c h w -> c (h n) w"))

def plot(
        *tensors: torch.Tensor, 
        mode: Literal["grid", "horizontal", "vertical"]="grid",
    ):
    plot_raw(*tensors, quantiles=(0.02, 0.98), cmap="viridis", mode=mode)


def transpose(frames):
    return einops.rearrange(frames, "f c w h -> f c h w")


def plot_transpose(
        *tensors: torch.Tensor, 
        mode: Literal["grid", "horizontal", "vertical"]="grid",
    ):
    tensors = [t.unsqueeze(0) if t.ndim == 3 else t for t in tensors]
    plot_raw(*[transpose(t) for t in tensors], quantiles=(0.02, 0.98), cmap="viridis", mode=mode)

def plot_T(
        *tensors: torch.Tensor, 
        mode: Literal["grid", "horizontal", "vertical"]="grid",
    ):
    plot_transpose(*tensors, mode=mode)

#%%
def crop_to_common_size(*tensors: torch.Tensor) -> Tuple[torch.Tensor]:
    # Each tensor should have dimensions (N, C, H, W) or (C, H, W) or (H, W)

    # Get minimum height and width
    min_height = min(tensor.shape[-2] for tensor in tensors)
    min_width = min(tensor.shape[-1] for tensor in tensors)

    # Center-crop all tensors to common size
    cropped_tensors = []
    for tensor in tensors:
        height, width = tensor.shape[-2], tensor.shape[-1]
        top = (height - min_height) // 2
        left = (width - min_width) // 2
        cropped_tensors.append(tensor[..., top:top + min_height, left:left + min_width])

    return cropped_tensors

def per_frame_mean(frames: torch.Tensor):
    return frames.reshape((frames.shape[0], -1)).mean(-1)

def plot_1D(frames: torch.Tensor):
    plt.plot(per_frame_mean(frames).cpu().numpy())