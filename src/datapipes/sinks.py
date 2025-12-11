#%%
from src.datapipes.datapipe import DataPipe, FutureSlice
import torch
from pathlib import Path
import numpy as np
from datapipes.sic import sic
from tqdm import tqdm
from typing import Optional, Tuple, Generator, Iterator
from src.datapipes.datapipe import DataPipe
import inspect
import ast


PRIMITIVES = (int, float, bool, str, type(None))


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

    args, varargs, varkw, locals_ = inspect.getargvalues(frame)
    parts = []

    # Positional args
    for name in args:
        parts.append(safe_repr(locals_[name]))

    # *args
    if varargs:
        for value in locals_[varargs]:
            parts.append(safe_repr(value))

    # **kwargs
    if varkw:
        for key, value in locals_[varkw].items():
            parts.append(f"{key}={safe_repr(value)}")

    # return f"{func_name}({', '.join(parts)})"
    return f"{func_name}"

def subbatch_emit_indices(dp: DataPipe, idx: slice, batch_size: int=256, progressbar=False, pb_description: Optional[str]=None) -> Iterator[Tuple[torch.Tensor, slice]]:
    start = idx.start if idx.start is not None else 0
    stop = idx.stop if idx.stop is not None else len(dp)

    if progressbar and pb_description is None:
        # caller_frame = inspect.currentframe().f_back
        # caller_name = caller_frame.f_code.co_name
        pb_description = get_caller_signature()

    pb = (lambda it: tqdm(it, leave=False, desc=pb_description)) if progressbar else (lambda it: it)
    for i in pb(range(start, stop, batch_size)):
        batch_stop = min(i + batch_size, stop)
        yield dp[i:batch_stop], slice(i, batch_stop)

def subbatch(dp: DataPipe, idx: slice, batch_size: int=256, progressbar=False, pb_description: Optional[str]=None) -> Iterator[torch.Tensor]:
    if progressbar and pb_description is None:
        # caller_frame = inspect.currentframe().f_back
        # caller_name = caller_frame.f_code.co_name
        pb_description = get_caller_signature()
    
    for batch, _ in subbatch_emit_indices(dp=dp, idx=idx, batch_size=batch_size, progressbar=progressbar, pb_description=pb_description):
        yield batch

def accumulate(dp: DataPipe, idx: slice, batch_size: int=256, progressbar=True) -> torch.Tensor:
    batches = []
    for batch in subbatch(dp=dp, idx=idx, batch_size=batch_size, progressbar=progressbar):
        batches.append(batch)
    return torch.cat(batches, axis=0)

def sum(frames: DataPipe, idx: slice=slice(None), batch_size: int=512) -> torch.Tensor:
    total_sum = torch.zeros_like(frames[0]).to("cuda", torch.float32)
    frames.to("cuda")

    for batch in subbatch(dp=frames, idx=idx, batch_size=batch_size, progressbar=True):
        total_sum += batch.sum(0)
    return total_sum

def mean(frames: DataPipe, idx: slice=slice(None), batch_size: int=512) -> torch.Tensor:
    total_sum = torch.zeros_like(frames[0]).to("cuda", torch.float32)
    frames.to("cuda")

    for batch in subbatch(dp=frames, idx=idx, batch_size=batch_size, progressbar=True):
        total_sum += batch.sum(0)
    total_sum /= len(frames)
    return total_sum
