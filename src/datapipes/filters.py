#%%

import torch
import torch.nn.functional as F
from skimage.morphology import disk
import einops
from typing import Callable

from datapipes.plotting import img
from datapipes.sic import sic


class blurs:
    @staticmethod
    def uniform_disk_blur(frames: torch.Tensor, kernel_size: int=3) -> torch.Tensor:
        pad = kernel_size // 2
        padder = torch.nn.ReplicationPad2d(pad)
        if kernel_size == 3:
            kernel = torch.tensor([
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ], dtype=torch.float32, device="cuda")
        else:
            kernel = torch.from_numpy(disk(pad)).view(1, 1, 1, 1, kernel_size, kernel_size).to(frames.device, dtype=torch.float32)
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        if len(frames.shape) == 3:
            frames = frames.unsqueeze(0)
        fframe = padder(frames)
        frames = F.conv2d(frames, kernel)
        return frames
    
    @staticmethod
    def gaussian_blur(frames: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        n, c, h, w = frames.shape
        axis = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=frames.device, dtype=frames.dtype)
        gauss = torch.exp(-axis**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        kernel2d = gauss[:, None] * gauss[None, :]
        kernel = kernel2d.expand(c, 1, kernel_size, kernel_size)
        # sic(kernel)
        # img.imshow(kernel[0, 0])
        padding = kernel_size // 2
        padder = torch.nn.ReplicationPad2d(padding)
        blurred = padder(frames)
        blurred = F.conv2d(blurred, kernel, groups=c)
        return blurred

class gradients:
    @staticmethod
    def laplace_of_gaussians(frames: torch.tensor, window_size: int, sigma_low: float=2, sigma_high: float=3, normalize_input=True):
        frames = img.map01(frames) if normalize_input else frames
        return (blurs.gaussian_blur(frames, window_size, sigma_high) - blurs.gaussian_blur(frames, window_size, sigma_low))
    
    @staticmethod
    def sobel(frames: torch.Tensor) -> torch.Tensor:
        # Combine Sobel kernels into a single kernel of shape (2, 1, 3, 3)
        sobel_kernels = torch.stack([
            torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
            ], dtype=torch.float32, device=frames.device),
            torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
            ], dtype=torch.float32, device=frames.device)
        ], dim=0).unsqueeze(1)  # shape: (2, 1, 3, 3)

        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        n, c, h, w = frames.shape

        # Expand kernels for each channel
        kernel = sobel_kernels.repeat(c, 1, 1, 1)  # (2*c, 1, 3, 3)
        frames_reshaped = einops.rearrange(frames, 'n c h w -> (n c) 1 h w')
        grads = F.conv2d(frames_reshaped, kernel, groups=c)  # (n*c, 2, h, w)
        grads = einops.rearrange(grads, '(n c) d h w -> n c h w d', n=n, c=c)
        return grads

    @staticmethod
    def hessian_matrix(frames: torch.Tensor) -> torch.Tensor:
        """
        Compute the per-pixel 2×2 Hessian matrix of a single-channel image.

        Parameters
        ----------
        image : torch.Tensor
            Tensor of shape (H, W), (1, H, W), or (N, 1, H, W).

        Returns
        -------
        torch.Tensor
            Tensor of shape (..., H, W, 2, 2) containing
            [[I_xx, I_xy],
            [I_xy, I_yy]]  for each pixel.
        """
        # Ensure shape (N, 1, H, W)
        if frames.dim() == 2:
            frames = frames.unsqueeze(0).unsqueeze(0)
        elif frames.dim() == 3:
            frames = frames.unsqueeze(1)
        assert frames.dim() == 4, "expected (N, 1, H, W)"

        device, dtype = frames.device, frames.dtype

        # Second-derivative kernels (central differences, Δ = 1 pixel)
        k_xx = torch.tensor([[1, -2, 1],
                            [2, -4, 2],
                            [1, -2, 1]], dtype=dtype, device=device) / 4.0
        k_yy = k_xx.t()
        k_xy = torch.tensor([[1, 0, -1],
                            [0, 0,  0],
                            [-1, 0, 1]], dtype=dtype, device=device) / 4.0

        k_xx = k_xx.view(1, 1, 3, 3)
        k_yy = k_yy.view(1, 1, 3, 3)
        k_xy = k_xy.view(1, 1, 3, 3)

        # image = F.pad(image, [1, 1, 1, 1], mode="reflect")
        I_xx = F.conv2d(frames, k_xx, padding="valid")
        I_yy = F.conv2d(frames, k_yy, padding="valid")
        I_xy = F.conv2d(frames, k_xy, padding="valid")

        hess = torch.stack((torch.stack((I_xx, I_xy), dim=-1),
                            torch.stack((I_xy, I_yy), dim=-1)), dim=-2)
        return hess  # shape: (N, 1, H, W, 2, 2)

    @staticmethod
    def meijering_vesselness(frames: torch.Tensor) -> torch.Tensor: # , kernel_size: int = 9, sigma: float = 0.0
        # if sigma > 0:
        #     frames = blurs.gaussian_blur(frames, kernel_size=kernel_size, sigma=sigma)
        hess = gradients.hessian_matrix(frames)
        eigvals = torch.linalg.eigvalsh(hess)
        # λ1 <= λ2 by convention for eigvalsh
        eig1 = eigvals[..., 0]
        eig2 = eigvals[..., 1]

        # Meijering: vesselness = max(0, -λ2) if λ2 < 0 else 0 (for dark tubes on bright background)
        vesselness = torch.clamp(-eig2, max=0.0)
        vesselness = vesselness / (vesselness.max() + 1e-8)

        # img.plots(
        #     vesselness,
        #     ((eig1 < 0) * (eig2 < 0)).float(),
        #     vesselness + ((eig1 < 0) * (eig2 < 0)).float(),
        # )
        return vesselness
    
    @staticmethod
    def get_hessian_eigenvals(frames: torch.Tensor) -> torch.Tensor: 
        hess = gradients.hessian_matrix(frames)
        eigvals = torch.linalg.eigvalsh(hess)
        # λ1 <= λ2 by convention for eigvalsh
        eig1 = eigvals[..., 0]
        eig2 = eigvals[..., 1]
        return eig1, eig2
    
class patch_filters:
    
    @staticmethod
    def local_fn(fn_patches: Callable, kernel_size: int = 3, iterations: int = 1, sharpness: float=10.0) -> Callable[[torch.Tensor], torch.Tensor]:
        pad = kernel_size // 2
        # print(f"kernel_size: {kernel_size}")
        # print(f"pad: {pad}")
        erosion_mask_kernel = torch.from_numpy(disk(pad)).view(1, 1, 1, 1, kernel_size, kernel_size).to(torch.float32)
        padder = torch.nn.ReplicationPad2d(pad)
        def local_fn(tensor: torch.Tensor) -> torch.Tensor:
            _erosion_mask_kernel = erosion_mask_kernel.to(tensor.device)
            x = padder(tensor)
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            for _ in range(iterations):
                x = img.map01(x)
                patches = x.unfold(-2, kernel_size, 1) # Unfold height
                patches = patches.unfold(-2, kernel_size, 1) # Unfold width
                patches = patches * _erosion_mask_kernel
                patches = patches.contiguous().view(*patches.shape[:4], -1)
                x = fn_patches(patches)
            return x
        return local_fn

    @staticmethod
    def local_softmin_fn(kernel_size: int=3) -> Callable[[torch.Tensor], torch.Tensor]:
        def fn_patches(patches: torch.Tensor):
            weights = torch.softmax((1 - patches), dim=-1)
            return (patches * weights).sum(dim=-1)
        return patch_filters.local_fn(fn_patches, kernel_size=kernel_size)
    
    @staticmethod
    def remove_bg(kernel_size: int=3, sharpness: float=10.0) -> Callable[[torch.Tensor], torch.Tensor]:
        def fn_patches(patches: torch.Tensor):
            weights = F.softmin(patches * sharpness, dim=-1) 
            return -(patches * weights + 1e-5).sqrt().sum(dim=-1)
        return patch_filters.local_fn(fn_patches, kernel_size=kernel_size)
    
    @staticmethod
    def local_softmax_fn(kernel_size: int=3) -> Callable[[torch.Tensor], torch.Tensor]:
        def fn_patches(patches: torch.Tensor):
            weights = torch.softmax(patches, dim=-1)
            return (patches * weights).sum(dim=-1)
        return patch_filters.local_fn(fn_patches, kernel_size=kernel_size)

class metrics:
    @staticmethod
    def spatial_contrast_total_frame(frames: torch.Tensor, eps=0):
            # frames.to("cuda")
            frames = einops.rearrange(frames, "n c h w -> n c (h w)")
            std, m = torch.std_mean(frames, dim=2)
            return std / (m + eps)


# def get_sample_data() -> tuple["FlowMatchingDataset", torch.Tensor]:
#     from load_flow_matching_dataset import get_data_loader, FlowMatchingDataset
#     loader = get_data_loader()
#     ds: FlowMatchingDataset = loader.dataset
#     x1 = ds.flow_field[0].means[0:2].mean(0)
#     return ds, x1

# if __name__ == "__main__":
#     ds, x1 = get_sample_data()
#     img.plots(
#         x1,
#         meijering_vesselness_loss(x1)
#     )