import torch
import math
import einops
from skimage.morphology import disk

import torch.nn.functional as F
from datapipes.sic import sic
from datapipes.analysis.noise import multiplicative_noise_op, stbn_like
from datapipes.manual_ops import with_manual_op
from typing import Literal, Optional, Any, Callable

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
    # kernel = torch.full(size=(window_size, 1, 1), fill_value=1.0 / window_size, dtype=torch.float32, device="cuda")
    def inner(frames: torch.Tensor):
        '''
        Compute moving mean of `frames`

        Args:
            `frames`: `Tensor` containing frames to compute moving mean of
            `kernel_time_dimension_length`: Length of the time dimension of the kernel used in the moving mean
        '''
        with torch.no_grad():
            n, c, h, w = frames.shape
            # flat = frames.flatten(start_dim=2)
            # print(f"{flat.shape = }")
            # flat_mov_mean = F.conv1d(flat, kernel, groups=c, padding=0)
            # print(f"{flat_mov_mean.shape = }")
            # return einops.rearrange(flat_mov_mean, "t c (h w) -> t c h w", h=h, w=w)
            # # mov_mean = frames.unfold(0, window_size, 1)[:-1].mean(-1)
            # # return mov_mean
            # Reshape to (N, C_in, L) for conv1d where:
            # N = C*H*W independent temporal signals, C_in = 1, L = T
            x = frames.permute(1, 2, 3, 0).contiguous().view(-1, 1, n)

            # Box filter kernel for mean: shape (C_out=1, C_in=1, K=window)
            kernel = torch.full((1, 1, window_size), 1.0 / window_size, device=frames.device, dtype=frames.dtype)

            # Valid convolution: no padding => output length T-window+1
            y = F.conv1d(x, kernel, padding=0)

            # Reshape back to (T', C, H, W)
            T_out = y.shape[-1]
            out = y.view(c, h, w, T_out).permute(3, 0, 1, 2).contiguous()
            return out
    return with_manual_op(inner, equivalent_slicing_op=(slice(window_size // 2, - (window_size // 2)), slice(None), slice(None), slice(None)))


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


def spatial_contrast(window_size=7, eps=1e-6, kernel_type: Literal["disk", "square"] = "square"):
    """
    Computes the spatial speckle contrast for each frame using convolution.
    Args:
        frames (torch.Tensor): Input tensor of shape (N, C, H, W) containing grayscale frames.
        window_size (int): The size of the local window to compute statistics.
        eps (float): A small constant to prevent division by zero.
    Returns:
        torch.Tensor: Tensor of shape (N, C, H, W) containing the speckle contrast for each pixel.
    """
    def _spatial_contrast(frames: torch.Tensor):
        # Create a convolutional kernel for computing local mean
        match kernel_type:
            case "disk":
                mean_kernel = torch.from_numpy(disk(window_size // 2)).to(frames.device, frames.dtype) # H W
            
            case "square":
                mean_kernel = torch.ones(window_size, window_size).to(frames.device, frames.dtype)    
            
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
    return _spatial_contrast


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

def bfi(frames: torch.Tensor) -> torch.Tensor:
    return 1.0 / (frames ** 2)


def skewness(window_size: int, eps: float = 1e-12) -> Callable[[torch.Tensor], torch.Tensor]:
    def _skewness(x: torch.Tensor) -> torch.Tensor:
        """
        Compute local skewness over the frame dimension of an (n, c, h, w) tensor,
        using a sliding window and returning only valid positions.

        Args:
            x: Input tensor of shape (n, c, h, w)
            window_size: Temporal window size
            eps: Numerical stability constant

        Returns:
            Tensor of shape (n - window_size + 1, c, h, w)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x to have shape (n, c, h, w), got {tuple(x.shape)}")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if window_size > x.shape[0]:
            raise ValueError("window_size must be <= x.shape[0]")

        x = x.to(dtype=torch.float32)
        _, c, h, w = x.shape

        # (n, c, h, w) -> (1, c*h*w, n)
        x = einops.rearrange(x, "n c h w -> 1 (c h w) n")

        x2 = x * x
        x3 = x2 * x

        kernel = torch.ones((x.shape[1], 1, window_size), device=x.device, dtype=x.dtype)

        sum1 = F.conv1d(x,  kernel, groups=x.shape[1])
        sum2 = F.conv1d(x2, kernel, groups=x.shape[1])
        sum3 = F.conv1d(x3, kernel, groups=x.shape[1])

        mean = sum1 / window_size
        ex2 = sum2 / window_size
        ex3 = sum3 / window_size

        m2 = ex2 - mean.square()
        m3 = ex3 - 3 * mean * ex2 + 2 * mean.pow(3)

        skew = m3 / m2.clamp_min(eps).pow(1.5)

        # (1, c*h*w, n_valid) -> (n_valid, c, h, w)
        return einops.rearrange(skew, "1 (c h w) n -> n c h w", c=c, h=h, w=w)
    return _skewness

def kurtosis(window_size: int, eps: float = 1e-12) -> Callable[[torch.Tensor], torch.Tensor]:
    def _kurtosis(x: torch.Tensor) -> torch.Tensor:
        """
        Compute local kurtosis over the frame dimension of an (n, c, h, w) tensor,
        using a sliding window and returning only valid positions.

        Args:
            x: Input tensor of shape (n, c, h, w)
            window_size: Temporal window size
            eps: Numerical stability constant

        Returns:
            Tensor of shape (n - window_size + 1, c, h, w)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x to have shape (n, c, h, w), got {tuple(x.shape)}")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if window_size > x.shape[0]:
            raise ValueError("window_size must be <= x.shape[0]")

        x = x.to(dtype=torch.float32)
        _, c, h, w = x.shape

        # (n, c, h, w) -> (1, c*h*w, n)
        x = einops.rearrange(x, "n c h w -> 1 (c h w) n")

        x2 = x * x
        x3 = x2 * x
        x4 = x2 * x2

        kernel = torch.ones((x.shape[1], 1, window_size), device=x.device, dtype=x.dtype)

        sum1 = F.conv1d(x,  kernel, groups=x.shape[1])
        sum2 = F.conv1d(x2, kernel, groups=x.shape[1])
        sum3 = F.conv1d(x3, kernel, groups=x.shape[1])
        sum4 = F.conv1d(x4, kernel, groups=x.shape[1])

        ex1 = sum1 / window_size
        ex2 = sum2 / window_size
        ex3 = sum3 / window_size
        ex4 = sum4 / window_size

        m2 = ex2 - ex1.square()
        m4 = ex4 - 4 * ex1 * ex3 + 6 * ex1.square() * ex2 - 3 * ex1.pow(4)

        kurt = m4 / m2.clamp_min(eps).square()

        # (1, c*h*w, n_valid) -> (n_valid, c, h, w)
        return einops.rearrange(kurt, "1 (c h w) n -> n c h w", c=c, h=h, w=w)
    return _kurtosis