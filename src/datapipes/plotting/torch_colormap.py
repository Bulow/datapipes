from torchvision.transforms import functional as vis_F
from matplotlib import colormaps
import torch
from einops import einops
import numpy as np

class TorchColormap:

    @classmethod
    def apply(cls, to: torch.Tensor, cmap_name: str="viridis", vmin: float=0, vmax: float=1, v_min_max:bool=False):
         cmap = cls(cmap_name, vmin, vmax, v_min_max)
         return cmap(to)
    
    
    @torch.no_grad()
    @classmethod
    def get_qtile(cls, tensor: torch.Tensor|np.ndarray, min_max_quantiles: tuple=(0.05, 0.95)):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor).to("cuda")
        lower, upper = min_max_quantiles
        tensor = tensor.to(torch.float32)
        min_max = torch.Tensor([lower, upper]).to(tensor)
        q = torch.quantile(tensor, min_max)
        q = q.cpu().numpy()
        
        return (q[0], q[1])

    def __init__(self, cmap_name: str, vmin: float=0, vmax: float=1, v_min_max:bool=False, device="cuda"):
        self.cmap_name = cmap_name
        self.vmin = vmin
        self.vmax = vmax
        self.v_min_max = v_min_max
        self.device = device

        cmap = colormaps[cmap_name]
        cmap_indices = torch.arange(0, 256, 1)
        
        self.cmap = torch.from_numpy(cmap(cmap_indices))[:, 0:3].to(device)

    @torch.no_grad
    def remap(self, t: torch.Tensor, vmin: float, vmax: float, eps=1e-8) -> torch.Tensor:
            assert vmin < vmax, f"vmin < vmax should be true. Got vmin={vmin}, vmax={vmax}"

            if (vmin == vmax):
                return t
            
            return (t.to(torch.float32) - vmin) / ((vmax - vmin) + eps)

    @torch.no_grad
    def __call__(self, t: torch.Tensor, vmin: float=None, vmax: float=None):
        in_dim = t.ndim
        if t.ndim == 2:
             t = t.unsqueeze(0)
        if t.ndim == 3:
             t = t.unsqueeze(0)
        if t.shape[-3] == 3:
             # Frame is already RGB. Let it pass through unchanged
             return t
        assert t.ndim == 4 and t.shape[1] == 1, f"t must have shape (n 1 h w), got {t.shape}"
        
        if self.v_min_max:
             vmin = t.min()
             vmax = t.max()
        
        if vmin is None:
            vmin = self.vmin
        if vmax is None:
            vmax = self.vmax

        device = t.device
        dtype = t.dtype

        if vmin != 0 or vmax != 1:
            t = self.remap(t=t, vmin=vmin, vmax=vmax)

        t = t.clip(0, 1)
        t_bytes = vis_F.convert_image_dtype(t.to("cuda"), torch.uint8).to(torch.long)
        t_bytes = t_bytes.squeeze(1)

        cmapped = self.cmap[t_bytes]
        cmapped = einops.rearrange(cmapped, "n h w c -> n c h w")
        if in_dim == 3 or (t.ndim == 4 and t.shape[0] == 1):
             cmapped = cmapped.squeeze(0)
        return cmapped.to(device=device, dtype=dtype)
    

