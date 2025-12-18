import torch
import torch.nn.functional as F
from torch.nn import Parameter
from skimage.morphology import disk
import einops

from typing import Callable
import abc

import img
import filters

import matplotlib.pyplot as plt

class OptimizationFilter(torch.nn.Module, abc.ABC):
    def __init__(self, frames: torch.Tensor, *args, **kwargs):
        super().__init__()
        with torch.no_grad():
            frames = self.preprocess_input(frames)
        self.original: torch.Tensor
        self.register_buffer("original", frames.clone().detach().to("cuda"))

        self.setup_parameters(frames, *args, **kwargs)
        self.to("cuda", dtype=torch.float32)

    def setup_parameters(self, frames: torch.Tensor, *args, **kwargs):
        self.mutable_frames = Parameter(frames.clone().detach())

    def preprocess_input(self, raw_input_frames) -> torch.Tensor:
        return img.map01(raw_input_frames)
    
    def postprocess_output(self, raw_output_frames) -> torch.Tensor:
        return img.map01(raw_output_frames)
    
    def forward(self):
        return self.mutable_frames

    def train(self, steps, lr, override_loss_fn: Callable=None, plot_iterative_frame_contrast=True):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        loss_fn = override_loss_fn if override_loss_fn is not None else self.loss_fn
        one = torch.tensor(1, device="cuda")
        if plot_iterative_frame_contrast:
            step_frame_contrast = []
        for i in range(steps):
            i_frac = i / steps
            pred = self.forward()
            loss = loss_fn(pred, i_frac * one)
            if plot_iterative_frame_contrast:
                step_frame_contrast.append(filters.metrics.spatial_contrast_total_frame(pred).item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        output = self.forward().detach()
        if plot_iterative_frame_contrast:
            self.plot_iterative_frame_contrast(step_frame_contrast)
        return output
    
    def plot_iterative_frame_contrast(self, step_frame_contrast: list[float]):
        # fig = plt.figure()
        plt.plot(step_frame_contrast)
        plt.title("Spatial contrast of entire frame")
        plt.xlabel("Iteration")
        plt.ylabel("Contrast")

    def combine_losses(*losses: torch.Tensor):
        frame_losses = [frame for frame in losses if len(frame.shape) == 4]
        scalar_losses = [scalar for scalar in losses if len(scalar.shape) == 0]

        assert len(losses) == len(frame_losses) + len(scalar_losses), f"Losses must have len(shape)==0 or len(shape)==4"

        if frame_losses:
            frame_losses = img.crop_to_common_size(*frame_losses)
            frame_losses = torch.stack(frame_losses).reshape([len(frame_losses), -1]).mean(1)
        losses = [*frame_losses, *scalar_losses]

        loss = torch.stack(losses).sum()
        return loss

    @abc.abstractmethod
    def loss_fn(self, pred: torch.Tensor, i_frac):
        pass
    
    @classmethod
    def apply_filter(cls, frames: torch.Tensor, steps: int=50, lr: float=0.005) -> torch.Tensor:
        optimization_filter = cls(frames)
        filtered = optimization_filter.train(steps, lr, plot_iterative_frame_contrast=False)
        return optimization_filter.postprocess_output(filtered)
