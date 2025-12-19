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
from datapipes.plotting.interactive_plots import imshow_default as plotly_imshow_default
from datapipes.plotting.interactive_plots import animate as plotly_animate
from typing import Optional, Tuple, List, Literal, Callable, Any, Dict
import plotly
import plotly.express as px
import plotly.graph_objects as go
import warnings

plot_output_backend_type = Literal[
    "ipython", 
    "plotly", 
    "plotly_animate", 
    "return_pil", 
    "ipython_and_return_pil",
    "return_tensor"
]
default_plot_output_backend: plot_output_backend_type = "ipython"

def backend_needs_preprocessing(backend: plot_output_backend_type) -> bool:
    need_preprocessing: Tuple[plot_output_backend_type] = ("ipython", "return_pil", "ipython_and_return_pil")
    return backend in need_preprocessing

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

def _plot(frame, backend: plot_output_backend_type):
    '''
    Display `frame` as an image
    '''

    match backend:
        case "ipython":
            display(to_pil_image(map01(frame)))
        case "plotly":
            return plotly_imshow_default(frame)
        case "plotly_animate":
            return plotly_animate(frames=frame)
        case "return_pil":
            return to_pil_image(map01(frame))
        case "ipython_and_return_pil":
            image = to_pil_image(map01(frame))
            display(image)
            return image
        case "return_tensor":
            t: torch.Tensor = torch.from_numpy(frame) if isinstance(frame, np.ndarray) else frame
            assert(isinstance(t, torch.Tensor))
            return t
        case _:
            raise ValueError(f"Unsupported backed: {backend}")


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

def _plot_grid(batch, backend: plot_output_backend_type = default_plot_output_backend):
    return _plot(frame=_grid_reshape(batch), backend=backend)

# @torch.no_grad()
def qtile(tensor: torch.Tensor, quantile: tuple[float]=(0.02, 0.98), output_bytes=False):
    def get_qtile(tensor: torch.Tensor, min_max_quantiles: tuple=(0.05, 0.95)):
        lower, upper = min_max_quantiles
        tensor = tensor.to(torch.float32)
        min_max = torch.Tensor([lower, upper]).to(tensor)
        try:
            q = torch.quantile(tensor, min_max)
        except RuntimeError as re:
            warnings.warn(message=f"Encountered {type(re).__name__} while getting quantiles: \"{re}\"\n\t Will use min and max values instead")
            return (min_max[0], min_max[1])
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
        padded_tensors.append(torch.nn.functional.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=tensor.min()))
    assert all(t.shape[-2:] == t[0].shape[-2:] for t in padded_tensors), f"All padded tensors must have the same height and width, got shapes: ({", ".join([f"{t.shape}" for t in padded_tensors])})"
    return padded_tensors

    
def plot_raw(
        *tensors: torch.Tensor, 
        map01_individually=True, 
        quantiles: Optional[tuple[float]]=None, 
        cmap: Optional[str|TorchColormap]=None, 
        mode: Literal["grid", "horizontal", "vertical", "animate"]="grid",
        backend: plot_output_backend_type = default_plot_output_backend
    ):
    if mode == "animate" or backend == "plotly_animate":
        mode = "animate"
        backend = "plotly_animate"
        

    # Convert all to torch.Tensor
    tensors = [torch.from_numpy(t) if isinstance(t, np.ndarray) else t for t in tensors]

    # Ensure all have a channel dimension
    tensors = [t.unsqueeze(0) if t.ndim == 2 else t for t in tensors]

    # Ensure all have a batch dimension
    tensors = [t.unsqueeze(0) if len(t.shape) == 3 else t for t in tensors]

    # Add empty third channel to tensors with 2 channels (useful for plotting fields of 2D vectors)
    tensors = [torch.cat([t, torch.zeros_like(t[..., 1, :, :].unsqueeze(-3))], dim=-3) if t.shape[-3] == 2 else t for t in tensors]
    
    # Clip all to quantiles
    if quantiles is not None:
        tensors = [qtile(t, quantiles) for t in tensors]
    
    # Remap to [0..1]
    if map01_individually:
        tensors = [map01(t) for t in tensors]

    # Pad all to larges (H W) size
    tensors = _pad_to_largest(*tensors if not isinstance(tensors, torch.Tensor) else tensors)


    if backend_needs_preprocessing(backend):
        # Apply color map to all
        if cmap is not None:
            if isinstance(cmap, str):
                cmap = TorchColormap(cmap)
            tensors = [cmap(t) for t in tensors]

    
    tensors = [t.unsqueeze(0) if len(t.shape)==3 else t for t in tensors]
    
    match mode:
        case "grid":
            tensors = torch.cat(tensors)
            return _plot_grid(tensors, backend=backend)
        case "horizontal":
            # _plot(einops.rearrange(tensors, "n c h w -> c h (w n)"))
            tensors = torch.cat(tensors, dim=-1)[0]
            return _plot(tensors, backend=backend)
        case "vertical":
            tensors = torch.cat(tensors, dim=-2)[0]
            print(tensors.shape)
            return _plot(tensors, backend=backend)
            # _plot(einops.rearrange(tensors, "n c h w -> c (h n) w"))
        case "animate":
            raise NotImplementedError(f"backend={backend}")
            # return _plot(tensors, backend=backend)
        case _:
            raise ValueError(f"Unsupported mode: {mode}")

def plot(
        *tensors: torch.Tensor, 
        mode: Literal["grid", "horizontal", "vertical"]="grid",
        backend: plot_output_backend_type = default_plot_output_backend
    ):
    return plot_raw(*tensors, quantiles=(0.02, 0.98), cmap="viridis", mode=mode, backend=backend)


def transpose(frames):
    return einops.rearrange(frames, "f c w h -> f c h w")


def plot_transpose(
        *tensors: torch.Tensor, 
        mode: Literal["grid", "horizontal", "vertical"]="grid",
        backend: plot_output_backend_type = default_plot_output_backend
    ):
    tensors = [t.unsqueeze(0) if t.ndim == 3 else t for t in tensors]
    return plot_raw(*[transpose(t) for t in tensors], quantiles=(0.02, 0.98), cmap="viridis", mode=mode, backend=backend)

def plot_T(
        *tensors: torch.Tensor, 
        mode: Literal["grid", "horizontal", "vertical"]="grid",
        backend: plot_output_backend_type = default_plot_output_backend
    ):
    return plot_transpose(*tensors, mode=mode, backend=backend)

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

