import torch
import einops
from typing import Optional, Literal
from datapipes import Ops

def defensive_gather(input: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """
    Minimal, explicit wrapper around torch.gather.

    Raises (on CPU or CUDA) for the same conditions that commonly cause CUDA
    device-side asserts / crashes:
        - device mismatch
        - index dtype not int64
        - rank/shape mismatch (non-gather dims)
        - out-of-bounds indices
    """
    if not isinstance(input, torch.Tensor) or not isinstance(index, torch.Tensor):
        raise TypeError("values and selected_indices must be torch.Tensors")

    # Normalize dim
    if dim < 0:
        dim = input.dim() + dim
    if dim < 0 or dim >= input.dim():
        raise ValueError(f"dim={dim} is invalid for values.ndim={input.dim()}")

    # Explicit preconditions (no auto-fixing)
    if index.device != input.device:
        raise ValueError(f"Device mismatch: values on {input.device}, indices on {index.device}")

    if index.dtype != torch.int64:
        raise TypeError(f"indices must be torch.int64 (LongTensor), got {index.dtype}")

    if index.dim() != input.dim():
        raise ValueError(
            f"Rank mismatch: values.ndim={input.dim()} vs indices.ndim={index.dim()} "
            f"(values.shape={tuple(input.shape)}, indices.shape={tuple(index.shape)})"
        )

    for d in range(input.dim()):
        if d == dim:
            continue
        if index.size(d) != input.size(d):
            raise ValueError(
                f"Shape mismatch at dim {d}: values.size({d})={input.size(d)} vs indices.size({d})={index.size(d)} "
                f"(values.shape={tuple(input.shape)}, indices.shape={tuple(index.shape)})"
            )

    dim_size = input.size(dim)
    if dim_size <= 0:
        raise ValueError(f"values.size({dim}) must be > 0, got {dim_size}")

    if index.numel() > 0:
        mn = int(index.min().item())
        mx = int(index.max().item())
        if mn < 0 or mx >= dim_size:
            raise IndexError(
                f"Out-of-bounds indices for dim={dim} (size={dim_size}): min={mn}, max={mx} "
                f"(valid range: 0..{dim_size-1})"
            )

    return torch.gather(input, dim=dim, index=index)




def defensive_scatter_reduce(
    input: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduce: Literal["sum", "prod", "mean", "amax", "amin"],
    *,
    include_self: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Minimal, explicit wrapper around torch.scatter_reduce_ / torch.scatter_reduce.

    Raises (on CPU or CUDA) for common conditions that can cause CUDA device-side asserts:
        - device mismatch
        - index dtype not int64
        - rank/shape mismatch between input/index/src (per scatter semantics)
        - out-of-bounds indices
        - invalid reduce
        - dtype issues for reductions like "mean"
    """
    if not isinstance(input, torch.Tensor) or not isinstance(index, torch.Tensor) or not isinstance(src, torch.Tensor):
        raise TypeError("input, index, and src must be torch.Tensors")

    # Normalize dim
    if dim < 0:
        dim = input.dim() + dim
    if dim < 0 or dim >= input.dim():
        raise ValueError(f"dim={dim} is invalid for input.ndim={input.dim()}")

    # Validate reduce
    valid_reduces = {"sum", "prod", "mean", "amax", "amin"}
    if reduce not in valid_reduces:
        raise ValueError(f"reduce must be one of {sorted(valid_reduces)}, got {reduce!r}")

    # Device checks
    dev = input.device
    if index.device != dev:
        raise ValueError(f"Device mismatch: input on {dev}, index on {index.device}")
    if src.device != dev:
        raise ValueError(f"Device mismatch: input on {dev}, src on {src.device}")
    if out is not None and out.device != dev:
        raise ValueError(f"Device mismatch: input on {dev}, out on {out.device}")

    # Dtype checks
    if index.dtype != torch.int64:
        raise TypeError(f"index must be torch.int64 (LongTensor), got {index.dtype}")

    if reduce == "mean" and input.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(f'reduce="mean" requires floating input dtype, got {input.dtype}')

    # Be explicit/defensive about dtype alignment (optional policy)
    if src.dtype != input.dtype:
        raise TypeError(f"src.dtype must match input.dtype (no auto-casting): input={input.dtype}, src={src.dtype}")
    if out is not None and out.dtype != input.dtype:
        raise TypeError(f"out.dtype must match input.dtype: out={out.dtype}, input={input.dtype}")

    # Rank checks
    if index.dim() != input.dim():
        raise ValueError(
            f"Rank mismatch: input.ndim={input.dim()} vs index.ndim={index.dim()} "
            f"(input.shape={tuple(input.shape)}, index.shape={tuple(index.shape)})"
        )
    if src.dim() != input.dim():
        raise ValueError(
            f"Rank mismatch: input.ndim={input.dim()} vs src.ndim={src.dim()} "
            f"(input.shape={tuple(input.shape)}, src.shape={tuple(src.shape)})"
        )
    if out is not None and out.dim() != input.dim():
        raise ValueError(
            f"Rank mismatch: input.ndim={input.dim()} vs out.ndim={out.dim()} "
            f"(input.shape={tuple(input.shape)}, out.shape={tuple(out.shape)})"
        )

    # Shape checks:
    #  - index and src must match everywhere
    #  - for d != dim, they must match input
    #  - for d == dim, size can differ; only index VALUES must be in-bounds
    for d in range(input.dim()):
        if index.size(d) != src.size(d):
            raise ValueError(
                f"Shape mismatch between index and src at dim {d}: "
                f"index.size({d})={index.size(d)} vs src.size({d})={src.size(d)} "
                f"(index.shape={tuple(index.shape)}, src.shape={tuple(src.shape)})"
            )
        if d != dim and index.size(d) != input.size(d):
            raise ValueError(
                f"Shape mismatch at dim {d}: input.size({d})={input.size(d)} vs index.size({d})={index.size(d)} "
                f"(input.shape={tuple(input.shape)}, index.shape={tuple(index.shape)})"
            )

    dim_size = input.size(dim)
    if dim_size <= 0:
        raise ValueError(f"input.size({dim}) must be > 0, got {dim_size}")

    # Out-of-bounds indices (the real critical safety check)
    if index.numel() > 0:
        mn = int(index.min().item())
        mx = int(index.max().item())
        if mn < 0 or mx >= dim_size:
            raise IndexError(
                f"Out-of-bounds indices for dim={dim} (size={dim_size}): min={mn}, max={mx} "
                f"(valid range: 0..{dim_size-1})"
            )

    # Execute
    if out is not None:
        out.scatter_reduce_(dim=dim, index=index, src=src, reduce=reduce, include_self=include_self)
        return out

    result = input.clone()
    result.scatter_reduce_(dim=dim, index=index, src=src, reduce=reduce, include_self=include_self)
    return result



def get_region_wise_reduction(frames: torch.Tensor, segmentation_map: torch.Tensor, reduction: Literal["sum", "prod", "mean", "amax", "amin"]):
    """
    Args:
        mask:  (N, H, W) uint8
        image: (N, 1, H, W) float32

    Returns:
        means: (N, S) float32 (where S = mask.max() + 1)
        
    """

    while segmentation_map.ndim < 4:
        segmentation_map = segmentation_map.unsqueeze(0)

    while frames.ndim < 4:
        frames = frames.unsqueeze(0)
    
    segmentation_map, frames = Ops.crop_to_common_size(segmentation_map, frames)

    assert segmentation_map.ndim == 4 and segmentation_map.shape[1] == 1
    assert frames.ndim == 4 and frames.shape[1] == 1
    if segmentation_map.shape[0] == 1:
        assert segmentation_map.shape[1:] == frames.shape[1:], f"{segmentation_map.shape = }, {frames.shape = }"
        segmentation_map = einops.repeat(segmentation_map, "n c h w -> (n r) c h w", r=frames.shape[0])
    else:
        assert segmentation_map.shape == frames.shape, f"{segmentation_map.shape = }, {frames.shape = }"

    N, _, H, W = segmentation_map.shape
    device = frames.device

    segmentation_map = segmentation_map.long()
    S = int(segmentation_map.max().item()) + 1

    mask_flat = segmentation_map.view(N, -1)
    image_flat = frames.view(N, -1)

    means = torch.zeros((N, S), device=device, dtype=frames.dtype)
    
    print(f"{means.shape = }, {mask_flat.shape = }, {image_flat.shape = }")
    means = defensive_scatter_reduce(
        input=means,
        dim=1,
        index=mask_flat,
        src=image_flat,
        reduce=reduction,
        include_self=False
    )
    means[..., 0] = 0

    return means



def get_region_wise_means(frames: torch.Tensor, segmentation_map: torch.Tensor):
    """
    Args:
        mask:  (N, H, W) uint8
        image: (N, 1, H, W) float32

    Returns:
        means: (N, S) float32 (where S = mask.max() + 1)
        
    """

    return get_region_wise_reduction(
        frames=frames,
        segmentation_map=segmentation_map,
        reduction="mean"
    )



def reconstruct_hands_from_region_values(region_wise_means: torch.Tensor, segmentation_map: torch.Tensor):
    """
    Args:
        mask:          (N, H, W) uint8 or long
        segment_means: (N, S) float32

    Returns:
        idealized_mask: (N, 1, H, W) float32
    """

    assert segmentation_map.ndim == 3
    assert region_wise_means.ndim == 2
    

    _, H, W = segmentation_map.shape
    N = region_wise_means.shape[0]

    segmentation_map = segmentation_map.long()

    masks = einops.repeat(segmentation_map, "1 h w -> r 1 h w", r=N)
    

    # Gather per-pixel means
    # mask: (N, 1, H, W)
    # segment_means: (N, S)
    # We expand segment_means to allow batch-wise gather
    means_expanded = region_wise_means.unsqueeze(-1).unsqueeze(-1)  # (N, S, 1, 1)

    idealized = defensive_gather(
        input=means_expanded.expand(-1, -1, H, W),  # (N, S, H, W)
        dim=1,
        index=masks                  # (N, 1, H, W)
    )

    return idealized
