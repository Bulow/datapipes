
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

from datapipes.manual_ops import with_manual_op


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
        return frames.to("cuda", dtype=torch.float32) / 255.0
    
    @staticmethod
    def float01_to_bytes_cpu(frames: torch.FloatTensor) -> torch.ByteTensor:
        return (frames * 255.0).to("cpu", dtype=torch.uint8)

    @staticmethod
    def numpy(frames: torch.Tensor) -> np.ndarray:
        return frames.cpu().numpy()
    
    @staticmethod
    def cpu(frames: torch.Tensor) -> torch.Tensor:
        return frames.cpu()
    
    @staticmethod
    def gpu(frames: torch.Tensor) -> torch.Tensor:
        return frames.to("cuda")
    
    @staticmethod
    def pytorch(frames: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(frames)
    
    @staticmethod
    def remove_channels(frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 and frames.shape[1] != 1:
            raise ValueError(f"Frames must have shape (N 1 H W), got {frames.shape}")
        return frames.squeeze(1)
    
    @staticmethod
    def roi(left, top, width, height):
        def roi(frames):
            if frames.ndim == 4:  # Handle batch of frames (N, C, H, W)
                return frames[:, :, top:top + height, left:left + width]
            if frames.ndim == 3 and frames.shape[0] == 1:
                frames = frames.squeeze(0)
            if frames.ndim == 2:  # Handle single frame (H, W)
                return frames[top:top + height, left:left + width]
            else:
                raise ValueError("Unsupported frame dimensions. Expected (N, C, H, W) or (H, W).", sic(frames))
        return roi

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
        if (frames.ndim == 4):
            frames = einops.rearrange(frames, "n 1 h w -> w h n")
        elif (frames.ndim == 3 and frames.shape[0] == 1):
            frames = einops.rearrange(frames, "1 h w -> w h")
        elif (frames.ndim == 3 and frames.shape[0] > 1):
            frames = einops.rearrange(frames, "n h w -> w h n")
        elif (frames.ndim == 2):
            frames = einops.rearrange(frames, "h w -> w h")
        else:
            raise ValueError(f"Shape must be one of [(n 1 h w), (1 h w), (n h w), (h w)], got {frames.shape}")
        return frames
    
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

