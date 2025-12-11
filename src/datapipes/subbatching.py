#%%
import rich

from typing import Tuple, Optional, Iterator, Any, Callable, List
from tqdm import tqdm

import torch
from datapipes.nd_windows import NdValidWindow

# from datapipes.utils import introspection
# from datapipes.ops import Ops
from datapipes.nd_windows import NdValidWindow, NdAutoUnpaddingWindow
# from datapipes

def nd_subbatch_emit_indices(
    data: NdValidWindow,
    window_size: Tuple[int, ...],
    stride: Tuple[int, ...]=None,
) -> Iterator[Tuple[torch.Tensor, slice]]:
    """
    Yield sliding-window sub-batches of a tensor by chunking along every dimension.

    Args:
        tensor (torch.Tensor): Input tensor of shape (d1, d2, ..., dk)
        window_size (Tuple[int,...]): Window size for each dimension
        stride (Tuple[int,...]): Stride for each dimension

    Yields:
        torch.Tensor: A sub-tensor sliced according to window_size and stride.
                      Windows near the boundary may be smaller if the tensor
                      size is not a multiple of stride.
    """
    window_size = [w if w is not None else d for w, d in zip(window_size, data.shape)]
    if stride is None: stride = window_size

    assert len(data.shape) == len(window_size) == len(stride), \
        "window_size and stride must match tensor dimensions"

    dims = data.shape

    # Ranges of starting indices for each dimension
    index_ranges = [
        range(0, dims[i], stride[i]) for i in range(len(dims))
    ]

    def recurse(dim: int, current_starts):
        if dim == len(dims):
            # Build the slice tuple and yield the sub-tensor
            slc = tuple(
                slice(start, min(start + window_size[i], dims[i]))
                for i, start in enumerate(current_starts)
            )
            yield data[slc], slc
            return

        for start in index_ranges[dim]:
            yield from recurse(dim + 1, current_starts + [start])

    yield from recurse(0, [])

def nd_subbatch(
    data: NdValidWindow,
    window_size: Tuple[int, ...],
    stride: Tuple[int, ...]=None,
) -> Iterator[torch.Tensor]:
    for b, _ in nd_subbatch_emit_indices(data=data, window_size=window_size, stride=stride):
        yield b



