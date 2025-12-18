import torch
import math
import einops
from skimage.morphology import disk

import torch.nn.functional as F
from datapipes.sic import sic
from typing import Callable
from datapipes.analysis.noise import multiplicative_noise_op, stbn_like

gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to(*args, **kwargs):
    '''
    Convert to GPU and/or dtype
    '''
    def inner(frames: torch.Tensor):
        if args:
            frames = frames.to(*args, **kwargs)
        return frames
    return inner

def get_laplacian_kernel():
    '''
    [
        [sqrt2, 1, sqrt2],
        [1,     0,     1],
        [sqrt2, 1, sqrt2]
    ]
    '''
    with torch.no_grad():
        sqrt2 = math.sqrt(2)
        laplacian_kernel = torch.tensor([[
            [sqrt2, 1, sqrt2],
            [1,     0,     1],
            [sqrt2, 1, sqrt2]
        ]], dtype=torch.float32, device=gpu)

        laplacian_kernel[0, 1, 1] = -laplacian_kernel.sum()
        return laplacian_kernel

def get_moving_mean(window_size):
    def inner(frames):
        '''
        Compute moving mean of `frames`

        Args:
            `frames`: `Tensor` containing frames to compute moving mean of
            `kernel_time_dimension_length`: Length of the time dimension of the kernel used in the moving mean
        '''
        with torch.no_grad():
            mov_mean = frames.unfold(0, window_size, 1).mean(-1)
            return mov_mean
    return inner


@torch.no_grad
def laplacian_contrast(frames: torch.Tensor):
    '''
    Laplacian contrast

    Args:
        `frames`: `Tensor` containing frames to compute contrast of
    '''
    # laplacian_kernel = torch.tensor([[
    #     [0, 1, 0],
    #     [1, -4, 1],
    #     [0, 1, 0]
    # ]], dtype=torch.float32, device=gpu)
    # with torch.no_grad():
    laplacian_kernel = get_laplacian_kernel().to(frames)
    laplacian = F.conv2d(frames, laplacian_kernel.unsqueeze(0))
    contrast = torch.abs(laplacian)
    # contrast[torch.isnan(contrast)] = 0
    return contrast




def spatial_contrast(window_size=7, eps=1e-6):
    def spatial_contrast(frames: torch.Tensor):
        """
        Computes the spatial speckle contrast for each frame using convolution.

        Args:
            frames (torch.Tensor): Input tensor of shape (N, C, H, W) containing grayscale frames.
            window_size (int): The size of the local window to compute statistics.
            eps (float): A small constant to prevent division by zero.

        Returns:
            torch.Tensor: Tensor of shape (N, C, H, W) containing the speckle contrast for each pixel.
        """
        # with torch.no_grad():
        # sic(frames)
        # print(frames)

        # Create a convolutional kernel for computing local mean
        mean_kernel = torch.from_numpy(disk(window_size // 2)).to(frames.device, frames.dtype) # H W
        mean_kernel /= mean_kernel.sum()
        mean_kernel = einops.rearrange(mean_kernel, "H W -> 1 1 H W") # N C H W
        
        # sic(mean_kernel)
        # print(mean_kernel)

        # Compute local mean using convolution
        local_mean = F.conv2d(frames, mean_kernel)
        
        # Compute local squared mean using convolution
        local_squared_mean = F.conv2d(frames ** 2, mean_kernel)
        
        # Compute local variance and standard deviation
        local_variance = torch.abs(local_squared_mean - local_mean ** 2)
        local_std = torch.sqrt(local_variance + eps)
        
        # Compute speckle contrast
        contrast = local_std / (local_mean + eps)
        
        return contrast
    return spatial_contrast


def temporal_contrast(window_size=7, eps=1e-6):
    def temporal_contrast(frames: torch.Tensor):
        """
        Computes the local temporal speckle contrast for a sequence of frames using convolution.

        Args:
            frames (torch.Tensor): Tensor of shape (N, H, W) representing N frames.
            window_size (int): The size of the local window to compute statistics.
            eps (float): A small constant to prevent division by zero.

        Returns:
            torch.Tensor: Tensor of shape (N, H, W) containing the local temporal speckle contrast.
        """
        # with torch.no_grad():
        local_mean = frames.unfold(0, window_size, 1).mean(-1)
        local_squared_mean = (frames ** 2).unfold(0, window_size, 1).mean(-1)
        local_variance = torch.abs(local_squared_mean - local_mean ** 2)
        local_std = torch.sqrt(local_variance + eps)

        contrast = local_std / (local_mean + eps)

        return contrast
    return temporal_contrast

def noise_shaped_contrast(contrast_func: Callable[[torch.Tensor], torch.Tensor], gain: float=0.15):
    def _noisy_contrast(frames: torch.Tensor):
        noisy_frames = multiplicative_noise_op(stbn_like, gain=gain)(frames)
        noisy_contrast = contrast_func(noisy_frames)
        clean_contrast = contrast_func(frames)

        contrast_ratio = (clean_contrast / (noisy_contrast + 1e-6))

        contrast_ratio -= contrast_ratio.mean()
        contrast_ratio += clean_contrast.mean()

        return contrast_ratio
    return _noisy_contrast

def total_temporal_speckle_contrast(frames, eps=1e-6):
    """
    Computes the temporal speckle contrast for a sequence of frames.

    Args:
        frames (torch.Tensor): Tensor of shape (N, H, W) representing N frames.
        eps (float): A small constant to prevent division by zero.

    Returns:
        torch.Tensor: Tensor of shape (H, W) containing the temporal speckle contrast.
    """

    # Compute the mean and standard deviation over the time dimension (dim=0)
    # temporal_mean = frames.mean(dim=0)
    # temporal_std = frames.std(dim=0, unbiased=False)

    temporal_mean, temporal_std = torch.std_mean(frames, dim=0, unbiased=False, keepdim=True)

    # Calculate the temporal speckle contrast: K = σ / (μ + eps)
    contrast = temporal_std / (temporal_mean + eps)
    
    return contrast


def spatial_contrast_total_frame(frames: torch.Tensor) -> torch.Tensor:
    frames = einops.rearrange(frames, "n c h w -> n (c h w)")
    std, m = torch.std_mean(frames, dim=-1)
    return std / (m + 1e-6)

def cumulative_spatial_contrast(frames: torch.Tensor) -> torch.Tensor:
    csum = torch.cumsum(frames, dim=0)  / torch.arange(start=1, end=frames.shape[0] + 1, step=1, device=frames.device)
    c_contrast = spatial_contrast_total_frame(csum)
    return c_contrast
