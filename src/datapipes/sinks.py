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
        yield dp[i:batch_stop], slice(i, batch_stop)

def subbatch(dp: DataPipe, idx: slice, batch_size: int=256, progress_bar: Callable[[Iterable, int, str], Iterator] = tqdm, pb_description: Optional[str]=None) -> Iterator[torch.Tensor]:
    if progress_bar and pb_description is None:
        # caller_frame = inspect.currentframe().f_back
        # caller_name = caller_frame.f_code.co_name
        pb_description = get_caller_signature()
    
    for batch, _ in subbatch_emit_indices(dp=dp, idx=idx, batch_size=batch_size, progress_bar=progress_bar, pb_description=pb_description):
        yield batch

def accumulate(dp: DataPipe, idx: slice, batch_size: int=256, progress_bar: Callable[[Iterable, Optional[int], str], Iterator] = partial(tqdm, leave=False), destination_device="cpu") -> torch.Tensor:
    batches = [] # TODO: Write directly to an empty tensor to avoid cat
    for batch in subbatch(dp=dp, idx=idx, batch_size=batch_size, progress_bar=progress_bar):
        batches.append(batch.to("cpu", non_blocking=True))
    return torch.cat(batches, axis=0)

def sum(frames: DataPipe, idx: slice=slice(None), batch_size: int=512) -> torch.Tensor:
    total_sum = torch.zeros_like(frames[0]).to("cuda", torch.float32)

    for batch in subbatch(dp=frames, idx=idx, batch_size=batch_size, progressbar=True):
        total_sum += batch.sum(0)
    return total_sum

def mean(frames: DataPipe, idx: slice=slice(None), batch_size: int=512) -> torch.Tensor:
    total_sum = torch.zeros_like(frames[0]).to("cuda", torch.float32)

    for batch in subbatch(dp=frames, idx=idx, batch_size=batch_size, progress_bar=True):
        total_sum += batch.sum(0)
    total_sum /= len(frames)
    return total_sum

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
