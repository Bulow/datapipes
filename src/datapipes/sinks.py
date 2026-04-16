#%%
from datapipes.datapipe import DataPipe
import torch
from pathlib import Path
import numpy as np
from datapipes.sic import sic
from typing import Optional, Tuple, Generator, Iterator
import inspect
from functools import partial
from typing import Literal, Callable, Iterable, Iterator, Any, Optional
import datapipes
from tqdm import tqdm
from datapipes.utils import SimpleTqdm
from datapipes.utils.slicer import _Slicer

def get_progress_bar() -> Callable:
    if datapipes.utils.running_under_matlab():
        return SimpleTqdm
    else:
        return tqdm

PRIMITIVES = (int, float, bool, str, type(None))

# def get_default_progress_bar() -> Callable:
#     if 

def safe_repr(value):
    # Primitives → print literal
    if isinstance(value, PRIMITIVES):
        return repr(value)

    # PyTorch tensor
    if torch is not None and isinstance(value, torch.Tensor):
        return f"<Tensor shape={tuple(value.shape)} dtype={value.dtype}>"
    
    # DataPipe
    if DataPipe is not None and isinstance(value, DataPipe):
        return f"<DataPipe shape={tuple(value.shape)}>"

    # NumPy array
    if isinstance(value, np.ndarray):
        return f"<ndarray shape={value.shape} dtype={value.dtype}>"

    # Generic array-like with .shape and .dtype
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return f"<{type(value).__name__} shape={value.shape} dtype={value.dtype}>"

    # Fallback
    return f"<{type(value).__name__}>"


def get_caller_signature():
    frame = inspect.currentframe().f_back.f_back
    code = frame.f_code
    func_name = code.co_name

    # args, varargs, varkw, locals_ = inspect.getargvalues(frame)
    # parts = []

    # # Positional args
    # for name in args:
    #     parts.append(safe_repr(locals_[name]))

    # # *args
    # if varargs:
    #     for value in locals_[varargs]:
    #         parts.append(safe_repr(value))

    # # **kwargs
    # if varkw:
    #     for key, value in locals_[varkw].items():
    #         parts.append(f"{key}={safe_repr(value)}")

    # return f"{func_name}({', '.join(parts)})"
    return f"{func_name}"

def subbatch_emit_indices(dp: DataPipe, idx: slice, batch_size: int=256, progress_bar: Callable[[Iterable, Optional[int], str], Iterator] = partial(tqdm, leave=False), pb_description: Optional[str]=None) -> Iterator[Tuple[torch.Tensor, slice]]:
    start = idx.start if idx.start is not None else 0
    stop = idx.stop if idx.stop is not None else len(dp)

    if progress_bar and pb_description is None:
        # caller_frame = inspect.currentframe().f_back
        # caller_name = caller_frame.f_code.co_name
        pb_description = get_caller_signature()

    pb = (lambda it: get_progress_bar()(it, desc=pb_description)) if progress_bar else (lambda it: it)
    for i in pb(range(start, stop, batch_size)):
        batch_stop = min(i + batch_size, stop)
        idx = slice(i, batch_stop)
        yield dp[i:batch_stop][:idx.stop - idx.start], idx 

def subbatch(dp: DataPipe, idx: slice, batch_size: int=256, progress_bar: Callable[[Iterable, int, str], Iterator] = tqdm, pb_description: Optional[str]=None) -> Iterator[torch.Tensor]:
    if progress_bar and pb_description is None:
        # caller_frame = inspect.currentframe().f_back
        # caller_name = caller_frame.f_code.co_name
        pb_description = get_caller_signature()
    
    for batch, _ in subbatch_emit_indices(dp=dp, idx=idx, batch_size=batch_size, progress_bar=progress_bar, pb_description=pb_description):
        yield batch

def accumulate(dp: DataPipe, idx: slice, batch_size: int=256, progress_bar: Callable[[Iterable, Optional[int], str], Iterator] = partial(tqdm, leave=False), destination_device="cpu") -> torch.Tensor:
    normalized_idx: slice = _Slicer.from_shape(shape=dp.shape)[idx]
    
    shape = tuple((slc.stop - slc.start) // slc.step for slc in normalized_idx)
    # print(f"{shape = }")
    out = torch.empty(size=shape, dtype=dp.dtype, device="cpu")
    for batch, batch_idx in subbatch_emit_indices(dp=dp, idx=idx, batch_size=batch_size, progress_bar=progress_bar):
        out[batch_idx] = batch
    return out

def list_accumulate(dp: DataPipe, idx: slice, batch_size: int=256, progress_bar: Callable[[Iterable, Optional[int], str], Iterator] = partial(tqdm, leave=False), destination_device="cpu") -> torch.Tensor:
    batches = [] # TODO: Write directly to an empty tensor to avoid cat
    for batch in subbatch(dp=dp, idx=idx, batch_size=batch_size, progress_bar=progress_bar):
        batches.append(batch.to("cpu", non_blocking=True))
    return torch.cat(batches, axis=0)

def sum(frames: DataPipe, idx: slice=slice(None), batch_size: int=128) -> torch.Tensor:
    total_sum = torch.zeros_like(frames[0]).to("cuda", torch.float32)

    for batch in subbatch(dp=frames, idx=idx, batch_size=batch_size, progressbar=True):
        total_sum += batch.sum(0)
    return total_sum

def mean(frames: DataPipe, idx: slice=slice(None), batch_size: int=128) -> torch.Tensor:
    total_sum = torch.zeros_like(frames[0]).to(device="cuda", dtype=torch.float32)

    n: int = 0

    for batch in subbatch(dp=frames, idx=idx, batch_size=batch_size, progress_bar=tqdm):
        batch = batch.to(device="cuda", dtype=torch.float32)
        total_sum += batch.sum(dim=0, keepdim=False)
        n += batch.shape[0]
    total_sum /= n
    return total_sum



def var_mean(frames: DataPipe, idx: slice = slice(None), batch_size: int = 128) -> torch.Tensor:
    total_sum = torch.zeros_like(frames[0]).to(device="cuda", dtype=torch.float32)
    total_sq_sum = torch.zeros_like(frames[0]).to(device="cuda", dtype=torch.float32)
    n: int = 0

    for batch in subbatch(dp=frames, idx=idx, batch_size=batch_size, progress_bar=tqdm):
        batch = batch.to(device="cuda", dtype=torch.float32)
        total_sum += batch.sum(dim=0, keepdim=False)
        total_sq_sum += (batch * batch).sum(dim=0, keepdim=False)
        n += batch.shape[0]


    _mean = total_sum / n
    _var = total_sq_sum / n - _mean * _mean
    return _var.clamp_min(0), _mean

def var(frames: DataPipe, idx: slice = slice(None), batch_size: int = 128) -> torch.Tensor:
    _var, _ = var_mean(frames=frames, idx=idx, batch_size=batch_size)
    return _var

def std_mean(frames: DataPipe, idx: slice = slice(None), batch_size: int = 128) -> torch.Tensor:
    _var, _mean = var_mean(frames=frames, idx=idx, batch_size=batch_size)
    _std = torch.sqrt(_var)
    return _std, _mean

def std(frames: DataPipe, idx: slice = slice(None), batch_size: int = 128) -> torch.Tensor:
    _std, _ = std_mean(frames=frames, idx=idx, batch_size=batch_size)
    return _std

def kurtosis(frames: DataPipe, idx: slice = slice(None), batch_size: int = 512) -> torch.Tensor:
    total_sum = torch.zeros_like(frames[0]).to(device="cuda", dtype=torch.float32)
    total_sq_sum = torch.zeros_like(frames[0]).to(device="cuda", dtype=torch.float32)
    total_cu_sum = torch.zeros_like(frames[0]).to(device="cuda", dtype=torch.float32)
    total_qt_sum = torch.zeros_like(frames[0]).to(device="cuda", dtype=torch.float32)
    n = 0

    for batch in subbatch(dp=frames, idx=idx, batch_size=batch_size, progress_bar=tqdm):
        batch = batch.to(device="cuda", dtype=torch.float32)
        total_sum += batch.sum(dim=0, keepdim=False)
        total_sq_sum += (batch ** 2).sum(dim=0, keepdim=False)
        total_cu_sum += (batch ** 3).sum(dim=0, keepdim=False)
        total_qt_sum += (batch ** 4).sum(dim=0, keepdim=False)
        n += batch.shape[0]

    mean = total_sum / n

    m2 = total_sq_sum / n - mean ** 2
    m4 = (
        total_qt_sum / n
        - 4 * mean * (total_cu_sum / n)
        + 6 * mean ** 2 * (total_sq_sum / n)
        - 3 * mean ** 4
    )

    return m4 / (m2.clamp_min(1e-12) ** 2)

def approximate_median_reservoir(
    frames: DataPipe,
    idx: slice = slice(None),
    batch_size: int = 512,
    sample_size: int = 8192,
) -> torch.Tensor:
    shape = frames[0].shape
    reservoir = None
    n_seen = 0

    for batch in subbatch(dp=frames, idx=idx, batch_size=batch_size, progress_bar=tqdm):
        batch = batch.to(device="cuda", dtype=torch.float32)
        flat_batch = batch.reshape(batch.shape[0], -1)

        if reservoir is None:
            take = min(sample_size, flat_batch.shape[0])
            reservoir = flat_batch[:take].clone()
            n_seen = take
            start = take
        else:
            start = 0

        for i in range(start, flat_batch.shape[0]):
            n_seen += 1
            j = torch.randint(0, n_seen, (1,), device=batch.device).item()
            if reservoir.shape[0] < sample_size:
                reservoir = torch.cat([reservoir, flat_batch[i:i+1]], dim=0)
            elif j < sample_size:
                reservoir[j] = flat_batch[i]

    return reservoir.median(dim=0).values.reshape(shape)

def approximate_median_hist(
    frames: DataPipe,
    idx: slice = slice(None),
    batch_size: int = 128,
    num_bins: int = 256,
) -> torch.Tensor:
    shape = frames[0].shape
    device = "cuda"

    running_min = torch.full_like(frames[0], float("inf"), device=device, dtype=torch.float32)
    running_max = torch.full_like(frames[0], float("-inf"), device=device, dtype=torch.float32)
    n = 0

    # Pass 1: find per-element min and max
    for batch in subbatch(dp=frames, idx=idx, batch_size=batch_size, progress_bar=tqdm):
        batch = batch.to(device=device, dtype=torch.float32)
        running_min = torch.minimum(running_min, batch.amin(dim=0))
        running_max = torch.maximum(running_max, batch.amax(dim=0))
        n += batch.shape[0]

    flat_min = running_min.reshape(-1)
    flat_max = running_max.reshape(-1)
    num_features = flat_min.shape[0]

    # Avoid zero-width ranges
    same = flat_max <= flat_min
    flat_max = torch.where(same, flat_min + 1.0, flat_max)

    hist = torch.zeros((num_features, num_bins), device=device, dtype=torch.int64)

    # Pass 2: build per-element histograms
    for batch in subbatch(dp=frames, idx=idx, batch_size=batch_size, progress_bar=tqdm):
        batch = batch.to(device=device, dtype=torch.float32)
        flat_batch = batch.reshape(batch.shape[0], -1)

        scaled = (flat_batch - flat_min.unsqueeze(0)) / (flat_max - flat_min).unsqueeze(0)
        bin_idx = torch.clamp((scaled * num_bins).long(), 0, num_bins - 1)

        for b in range(flat_batch.shape[0]):
            hist.scatter_add_(
                dim=1,
                index=bin_idx[b].unsqueeze(1),
                src=torch.ones((num_features, 1), device=device, dtype=torch.int64),
            )

    cdf = hist.cumsum(dim=1)
    target = (n - 1) // 2

    median_bin = (cdf > target).to(torch.int64).argmax(dim=1)

    bin_width = (flat_max - flat_min) / num_bins
    median = flat_min + (median_bin.to(torch.float32) + 0.5) * bin_width

    # Restore exact value for constant features
    median = torch.where(same, flat_min, median)

    return median.reshape(shape)

def covariance(frames: DataPipe, idx: slice = slice(None), batch_size: int = 128) -> torch.Tensor:
    first = frames[0].to(device="cuda", dtype=torch.float32)
    num_features = first.numel()

    total_sum = torch.zeros(num_features, device="cuda", dtype=torch.float32)
    total_outer = torch.zeros((num_features, num_features), device="cuda", dtype=torch.float32)
    n = 0

    for batch in subbatch(dp=frames, idx=idx, batch_size=batch_size, progress_bar=tqdm):
        batch = batch.to(device="cuda", dtype=torch.float32)
        batch = batch.reshape(batch.shape[0], -1)

        total_sum += batch.sum(dim=0)
        total_outer += batch.T @ batch
        n += batch.shape[0]

    mean = total_sum / n
    cov = total_outer / n - torch.outer(mean, mean)
    return cov

def skewness(frames: DataPipe, idx: slice = slice(None), batch_size: int = 128) -> torch.Tensor:
    total_sum = torch.zeros_like(frames[0]).to(device="cuda", dtype=torch.float32)
    total_sq_sum = torch.zeros_like(frames[0]).to(device="cuda", dtype=torch.float32)
    total_cu_sum = torch.zeros_like(frames[0]).to(device="cuda", dtype=torch.float32)
    n = 0

    for batch in subbatch(dp=frames, idx=idx, batch_size=batch_size, progress_bar=tqdm):
        batch = batch.to(device="cuda", dtype=torch.float32)
        total_sum += batch.sum(dim=0, keepdim=False)
        total_sq_sum += (batch ** 2).sum(dim=0, keepdim=False)
        total_cu_sum += (batch ** 3).sum(dim=0, keepdim=False)
        n += batch.shape[0]

    mean = total_sum / n

    m2 = total_sq_sum / n - mean ** 2
    m3 = (
        total_cu_sum / n
        - 3 * mean * (total_sq_sum / n)
        + 2 * mean ** 3
    )

    return m3 / (m2.clamp_min(1e-12) ** 1.5)

def approximate_mode(
    frames: DataPipe,
    idx: slice = slice(None),
    batch_size: int = 512,
    num_bins: int = 256,
) -> torch.Tensor:
    first = frames[0].to(device="cuda", dtype=torch.float32)
    shape = first.shape
    flat_size = first.numel()

    running_min = torch.full((flat_size,), float("inf"), device="cuda", dtype=torch.float32)
    running_max = torch.full((flat_size,), float("-inf"), device="cuda", dtype=torch.float32)

    # Pass 1: per-pixel min/max
    for batch in subbatch(dp=frames, idx=idx, batch_size=batch_size, progress_bar=tqdm):
        batch = batch.to(device="cuda", dtype=torch.float32)
        flat = batch.reshape(batch.shape[0], -1)
        running_min = torch.minimum(running_min, flat.amin(dim=0))
        running_max = torch.maximum(running_max, flat.amax(dim=0))

    same = running_max <= running_min
    running_max = torch.where(same, running_min + 1.0, running_max)

    counts = torch.zeros((flat_size, num_bins), device="cuda", dtype=torch.int32)

    # Pass 2: accumulate histogram counts
    for batch in subbatch(dp=frames, idx=idx, batch_size=batch_size, progress_bar=tqdm):
        batch = batch.to(device="cuda", dtype=torch.float32)
        flat = batch.reshape(batch.shape[0], -1)

        scaled = (flat - running_min.unsqueeze(0)) / (running_max - running_min).unsqueeze(0)
        bin_idx = torch.clamp((scaled * num_bins).long(), 0, num_bins - 1)

        for b in range(flat.shape[0]):
            counts.scatter_add_(
                dim=1,
                index=bin_idx[b].unsqueeze(1),
                src=torch.ones((flat_size, 1), device="cuda", dtype=torch.int32),
            )

    mode_bin = counts.argmax(dim=1)
    bin_width = (running_max - running_min) / num_bins
    mode = running_min + (mode_bin.to(torch.float32) + 0.5) * bin_width
    mode = torch.where(same, running_min, mode)

    return mode.reshape(shape)

def logsumexp(frames: DataPipe, idx: slice = slice(None), batch_size: int = 512) -> torch.Tensor:
    running_max = torch.full_like(frames[0], float("-inf"), device="cuda", dtype=torch.float32)
    running_sum = torch.zeros_like(frames[0], device="cuda", dtype=torch.float32)

    for batch in subbatch(dp=frames, idx=idx, batch_size=batch_size, progress_bar=tqdm):
        batch = batch.to(device="cuda", dtype=torch.float32)

        batch_max = batch.amax(dim=0)
        new_max = torch.maximum(running_max, batch_max)

        running_sum = (
            running_sum * torch.exp(running_max - new_max)
            + torch.exp(batch - new_max.unsqueeze(0)).sum(dim=0)
        )
        running_max = new_max

    return running_max + torch.log(running_sum)

from blake3 import blake3

def hash_frames(frames: DataPipe, batch_size=512, digest_length=32):
    hasher = blake3(max_threads=blake3.AUTO)
    base_str = f"shape={torch.Size(frames.shape)}, dtype={frames[0].dtype}"
    print(base_str)
    hasher.update(base_str.encode(encoding="utf-8"))
    for batch in subbatch(dp=frames, idx=slice(None), batch_size=batch_size, progressbar=True):
        hasher.update(batch.cpu().numpy())
    return hasher.digest(length=digest_length)



# %%
