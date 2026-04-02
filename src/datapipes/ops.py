
from typing import Tuple, Callable, Protocol
from datapipes.datasets.dataset_source import DatasetSource
from datapipes.manual_ops import with_manual_op
from typing import Literal, Callable
import math
import torch
import torch.nn.functional as F
import numpy as np
from datapipes.sic import sic
import einops
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List, Iterator
# from datapipes.utils import introspection
# from datapipes.nd_windows import NdValidWindow, NdAutoUnpaddingWindow
# from datapipes import subbatching

import warnings

from datapipes.manual_ops import with_manual_op, with_manual_unpad

from datapipes.plotting.torch_colormap import TorchColormap
from datapipes.plotting import plot
# def roi(frames: torch.Tensor) -> torch.Tensor:



class Ops:
    
    @staticmethod
    def to(*args, **kwargs):
        '''
        Convert to GPU and/or dtype
        '''
        def to(frames: torch.Tensor):
            if args:
                frames = frames.to(*args, **kwargs)
            return frames
        return with_manual_op(to)
    
    @staticmethod
    def bytes_to_float01_gpu(frames: torch.ByteTensor) -> torch.FloatTensor:
        return frames.to(device="cuda", dtype=torch.float32, non_blocking=True) / 255.0
    
    @staticmethod
    def float01_to_bytes_cpu(frames: torch.FloatTensor) -> torch.ByteTensor:
        return (frames * 255.0).to("cpu", dtype=torch.uint8, non_blocking=True)

    @staticmethod
    def numpy(frames: torch.Tensor) -> np.ndarray:
        return frames.to("cpu", non_blocking=True).numpy()
    
    @staticmethod
    def cpu(frames: torch.Tensor) -> torch.Tensor:
        return frames.to("cpu", non_blocking=True)
    
    @staticmethod
    def gpu(frames: torch.Tensor) -> torch.Tensor:
        return frames.to("cuda", non_blocking=True)
    
    @staticmethod
    def pytorch(frames: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(frames)
    
    @staticmethod
    def remove_channels(frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 and frames.shape[1] != 1:
            raise ValueError(f"Frames must have shape (N 1 H W), got {frames.shape}")
        return frames.squeeze(1)

    @staticmethod
    def subtract_mean(frames: torch.Tensor) -> torch.Tensor:
        return frames - frames.mean()
    
    @staticmethod
    def roi(left, top, width, height):
        def _roi(frames):
            if frames.ndim == 4:  # Handle batch of frames (N, C, H, W)
                return frames[:, :, top:top + height, left:left + width]
            if frames.ndim == 3 and frames.shape[0] == 1:
                frames = frames.squeeze(0)
            if frames.ndim == 2:  # Handle single frame (H, W)
                return frames[top:top + height, left:left + width]
            else:
                raise ValueError("Unsupported frame dimensions. Expected (N, C, H, W) or (H, W).", sic(frames))
        return with_manual_op(_roi, equivalent_slicing_op=(
            slice(None),
            slice(None),
            slice(top, top + height),
            slice(left, left + width),
        ))
    

    def visualize_roi(
        frame: torch.Tensor, 
        left,
        top,
        width,
        height,
        visualize_roi: bool=True,
        opacity: float=0.8
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        
        top -= height // 2
        left -= width // 2

        roi_func = Ops.roi(
            left=left,
            top=top,
            width=width,
            height=height, 
        )

        if visualize_roi:
            frame = TorchColormap.apply(Ops.map01(Ops.qtile(frame)))

            if frame.ndim == 3:
                frame = frame.unsqueeze(0)
            roi = torch.zeros_like(frame)
            roi[..., top:top+height, left:left+width] = roi_func(frame)
            plot((frame * opacity) + roi)

        return roi_func

    @staticmethod
    def log(frames: torch.Tensor, eps=1e-6) -> torch.Tensor:
        return torch.log(frames + eps)
    
    @staticmethod
    def log1p(frames: torch.Tensor) -> torch.Tensor:
        return torch.log1p(frames)
    
    @staticmethod
    def sqrt(frames: torch.Tensor, eps=1e-6) -> torch.Tensor:
        return torch.sqrt(torch.clamp_min(frames, min=0) + eps)
    
    
    @staticmethod
    def py_to_matlab(frames: torch.Tensor|np.ndarray) -> np.ndarray:
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        
        if frames.ndim > 4:
            raise ValueError(f"Shape must be one of [(n 1 h w), (1 h w), (n h w), (h w)], got {frames.shape}")
        
        if frames.ndim == 4:
            frames = einops.rearrange(frames, "n 1 h w -> w h n")
        elif (frames.ndim == 3 and frames.shape[0] == 1):
            frames = einops.rearrange(frames, "1 h w -> w h")
        elif (frames.ndim == 3 and frames.shape[0] > 1):
            frames = einops.rearrange(frames, "n h w -> w h n")
        elif (frames.ndim == 2):
            frames = einops.rearrange(frames, "h w -> w h")
        
        
        fortran_frames: np.ndarray = np.asfortranarray(frames)
        return fortran_frames
    
    @staticmethod
    def matlab_to_py(frames: np.ndarray) -> torch.Tensor:
        # TODO: Support more shapes
        frames = einops.rearrange(frames, "w h n -> n 1 h w")
        frames = torch.from_numpy(frames, device="cuda") 
        return frames
    
    @staticmethod
    def apply_mask(mask: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        def _apply_mask(frames: torch.Tensor) -> torch.Tensor:
            return frames * (mask.to(dtype=frames.dtype, device = frames.device))
        return with_manual_op(_apply_mask)
    
    @staticmethod
    def resample(scale_factor: float, mode: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"]="bicubic") -> Callable[[torch.Tensor], torch.Tensor]:
        def _resample(frames: torch.Tensor) -> torch.Tensor:
            return F.interpolate(frames, scale_factor=scale_factor, mode=mode, align_corners=False)
        return with_manual_op(_resample, equivalent_slicing_op=(
            slice(None),
            slice(None),
            slice(None, None, int(round(1 / scale_factor))), 
            slice(None, None, int(round(1 / scale_factor)))
            )
        )
    

    @staticmethod
    def temporal_diff(d_index: int=1):
        '''
        Convert to GPU and/or dtype
        '''
        def _temporal_diff(frames: torch.Tensor):
            # print(f"{frames.shape = }")
            # frames = frames.to(torch.float32) / 255.0
            # print(f"{frames.shape = }, {frames.dtype = }, {frames.device = }")
            
            # print(f"{frames[d_index:].shape = }, {frames[:-d_index].shape = }")
            return (frames[d_index:] - frames[:-d_index])
        # return with_manual_op(_temporal_diff, equivalent_slicing_op=((slice(0, - (d_index)), slice(None), slice(None), slice(None))))
        # return _temporal_diff
        
        # return with_manual_unpad(_temporal_diff, padding=1)
        return with_manual_op(_temporal_diff, equivalent_slicing_op=(slice(0, -d_index), slice(None), slice(None), slice(None)))

    

    @staticmethod
    def map01(frames: torch.Tensor, dim: Optional[int]=None, eps=1e-8) -> torch.Tensor:
        '''
        Map values in `frames` to `[0, 1]`
        
        Where the minimum value is mapped to `0` and the maximum value is mapped to `1`
        '''
        # if frames.device == "meta":
        #     return frames
        
        if dim is None:
            min_val = frames.min()
            max_val = frames.max()
        else:
            min_val = frames.min(dim=dim, keepdim=True).values
            max_val = frames.max(dim=dim, keepdim=True).values

        # if (min_val == max_val):
        #     return frames
        
        return (frames.to(torch.float32) - min_val) / ((max_val - min_val) + eps)
    
    @staticmethod
    def map_values(frames: torch.Tensor, min_val: float, max_val: float, eps: float=1e-6) -> torch.Tensor:
        return (frames.to(torch.float32) - min_val) / ((max_val - min_val) + eps)

    @staticmethod
    def normalize(frames: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        frames = frames.to(torch.float32)
        mean = frames.mean()
        std = frames.std()

        return (frames - mean) / (std + eps)

    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
    def crop_to_common_length(*tensors: torch.Tensor) -> Tuple[torch.Tensor]:
        # Each tensor should have dimensions (N, C, H, W) or (C, H, W) or (H, W)

        # Get minimum height and width
        min_length = min(tensor.shape[0] for tensor in tensors)

        # Center-crop all tensors to common size
        cropped_tensors = []
        for tensor in tensors:
            length = tensor.shape[0]
            start = (length - min_length) // 2
            cropped_tensors.append(tensor[start:start + min_length])

        return cropped_tensors
    
    def pad_to_largest(*tensors: torch.Tensor) -> Tuple[torch.Tensor]:
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
        if not all(t.shape[-2:] == padded_tensors[0].shape[-2:] for t in padded_tensors[1:]):
            raise RuntimeError(f"All padded tensors must have the same height and width, got shapes: ({", ".join([f"{t.shape}" for t in padded_tensors])})")
        return padded_tensors

class PassthroughOp(Protocol):

    def __init__(self):
        self.reset_state()

    def reset_state(self) -> None:
        ...

    def __call__(self, frames: torch.Tensor) -> torch.Tensor: 
        ...

class CountFrames(PassthroughOp):
    def reset_state(self):
        self.count = 0
        
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        self.count += frames.shape[0]
        return frames

