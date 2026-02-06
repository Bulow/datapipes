import torch
import einops
import torch.nn.functional as F

def project_marker_along_segment(start: torch.Tensor, stop: torch.Tensor, frac: float) -> torch.Tensor:
    return start + ((stop - start) * frac)

def normalize(vec: torch.Tensor) -> torch.Tensor:
        return vec / (vec * vec).sum(-1).sqrt()
    

def vec_len(vec: torch.Tensor) -> float:
    return (vec * vec).sum(-1).sqrt()

def avg_dir(start: torch.Tensor, stop0: torch.Tensor, stop1: torch.Tensor, normalize_input_dirs=False) -> torch.Tensor:
    dir0 = (stop0 - start)
    dir1 = (stop1 - start)

    if normalize_input_dirs:
        dir0 = normalize(dir0)
        dir1 = normalize(dir1)

    return ((dir0 + dir1) / 2)

def project_point_onto_line(origin: torch.Tensor, vec: torch.Tensor, dir: torch.Tensor):
    dir = normalize(dir)
    return origin + ((dir * vec).sum(-1) * dir)




def sample_line(img: torch.Tensor, start_point: torch.Tensor, end_point: torch.Tensor) -> torch.Tensor:
    """
    Sample pixels along a line segment from start_point to end_point.

    Args:
        img:        (H, W) tensor
        start_point:(2,) tensor [x, y] in pixel coordinates (float or int)
        end_point:  (2,) tensor [x, y] in pixel coordinates (float or int)

    Returns:
        (N,) tensor of sampled values along the line (bilinear interpolation).
        N is chosen as ceil(max(|dx|, |dy|)) + 1 (like a dense Bresenham-style sampling).
    """
    if img.ndim != 2:
        raise ValueError(f"img must be (H, W). Got {tuple(img.shape)}")

    device = img.device
    dtype = img.dtype

    sp = start_point.to(device=device, dtype=torch.float32).view(2)
    ep = end_point.to(device=device, dtype=torch.float32).view(2)

    H, W = img.shape

    dx = (ep[0] - sp[0]).abs()
    dy = (ep[1] - sp[1]).abs()
    n = int(torch.ceil(torch.maximum(dx, dy)).item()) + 1
    n = max(n, 1)

    t = torch.linspace(0.0, 1.0, steps=n, device=device, dtype=torch.float32)
    xs = sp[0] + t * (ep[0] - sp[0])
    ys = sp[1] + t * (ep[1] - sp[1])

    # Normalize to grid_sample coords in [-1, 1]
    # x: 0..W-1 -> -1..1, y: 0..H-1 -> -1..1
    xg = 2.0 * xs / (W - 1) - 1.0 if W > 1 else torch.zeros_like(xs)
    yg = 2.0 * ys / (H - 1) - 1.0 if H > 1 else torch.zeros_like(ys)

    # grid_sample expects grid as (N, H_out, W_out, 2) with last dim (x, y)
    grid = torch.stack([xg, yg], dim=-1).view(1, n, 1, 2)

    # input to grid_sample must be (N, C, H, W)
    inp = img.to(torch.float32).view(1, 1, H, W)

    # Sample; padding_mode="border" clamps outside points to border values.
    out = F.grid_sample(
        inp,
        grid,
        mode="nearest",
        padding_mode="border",
        align_corners=True,
    )  # (1, 1, n, 1)

    line = out.view(n).to(dtype)
    return line

def get_transitions(sampled_line: torch.Tensor):
        change = sampled_line[1:] ^ sampled_line[:-1]
        idx = torch.nonzero(change, as_tuple=False).flatten() + 1
        return idx
    
    

def get_transition_points_along_line(img: torch.Tensor, start_point: torch.Tensor, end_point: torch.Tensor) -> torch.Tensor:
        # if start_point.max() <= 1.0 and end_point.max() <= 1.0:
        #     px_scale = torch.tensor((img.shape[-1], img.shape[-2]), device=start_point.device)
        #     start_point *= px_scale
        #     end_point *= px_scale
        line_px = sample_line(img=img, start_point=start_point, end_point=end_point).to(torch.uint8)
        transition_fracs = get_transitions(line_px) / len(line_px)
        points = tuple(project_marker_along_segment(start=start_point, stop=end_point, frac=frac) for frac in transition_fracs)
        return points
