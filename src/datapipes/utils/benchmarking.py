import datapipes
from datapipes.datasets import DatasetSource
import torch
import numpy as np
from datapipes import datasets, sinks
from datapipes.plotting.plots import plot
from datapipes.datapipe import DataPipe
import time
import sys
from collections import deque
import warnings
import rich

class MultiBlockTimer:
    def __init__(self):
        self.total = 0.0
        self.n_blocks = 0
        self._start = None

    def __enter__(self):
        # Start timing this block
        self.n_blocks += 1
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop timing this block and accumulate
        end = time.perf_counter()
        self.total += end - self._start
        self._start = None
        # Do not suppress exceptions
        return False
    
def human_readable_filesize(size_bytes: int, dp=2):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024.0:
            return f"{size_bytes:.{dp}f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.{dp}f} PB"

def human_readable_time(secs: float) -> str:
    ms = int((secs - int(secs)) * 1000)
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}.{ms:02d}s"
    elif m:
        return f"{m}m {s}.{ms:02d}s"
    else:
        return f"{s}.{ms:02d}s"

def human_readable_time_us(us: float) -> str:
    rem_us = us % 1_000
    return f"{human_readable_time(secs=us / 1_000_000)} {rem_us}us"
    
def raw_frame_throughput(time_elapsed: float, ds: datasets.DatasetSource):
    n, c, h, w = ds.shape
    total_bytes = n * c * h * w
    bytes_per_second = round(total_bytes / time_elapsed)
    print(f"Raw frame throughput = {human_readable_filesize(bytes_per_second)}/s")

def get_logical_size(datapipe: DataPipe|DatasetSource) -> int:
    first_frame: torch.Tensor = datapipe[0]
    total_logical_bytes = first_frame.numel() * first_frame.element_size() * len(datapipe)
    return total_logical_bytes

def get_disk_size(datapipe: DataPipe|DatasetSource) -> int:
    return datapipe.path.stat().st_size

def _get_tensor_size_bytes(tensor: torch.Tensor) -> int:
    total_bytes = tensor.numel() * tensor.element_size()
    return total_bytes

def _get_ndarray_size_bytes(array: np.ndarray) -> int:
    total_bytes = array.size * array.itemsize
    return total_bytes

# def get_physical_size(datapipe: DataPipe) -> str:



def get_memory_size(datapipe: DataPipe|DatasetSource) -> int:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*torch.distributed.reduce_op.*",
            category=FutureWarning,
        )
        return _get_memory_size(datapipe)

def _get_memory_size(obj, seen=None):
    """
    Recursively computes the approximate memory footprint of a Python object.

    Parameters:
        obj: any Python object
        seen: set of object ids already counted (used internally)

    Returns:
        int: approximate size in bytes
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)
    size = sys.getsizeof(obj)

    # Handle containers
    if isinstance(obj, dict):
        size += sum(
            _get_memory_size(k, seen) + _get_memory_size(v, seen)
            for k, v in obj.items() if not (isinstance(k, str) and k.startswith("__"))
        )
    elif isinstance(obj, torch.Tensor):
        size += _get_tensor_size_bytes(obj)
    elif isinstance(obj, np.ndarray):
        size += _get_ndarray_size_bytes(obj)
    elif isinstance(obj, (list, tuple, set, frozenset, deque)):
        size += sum(_get_memory_size(i, seen) for i in obj)
    elif hasattr(obj, "__dict__"):
        size += _get_memory_size(obj.__dict__, seen)

    return size


def print_memory_stats(datapipe: DataPipe):
    if hasattr(datapipe, "path"):
        disk_size = human_readable_filesize(get_disk_size(datapipe))
    else:
        disk_size = "No file path associated with datapipe"
    ram_size = human_readable_filesize(get_memory_size(datapipe))
    logical_size = human_readable_filesize(get_logical_size(datapipe))

    # print(f"{disk_size = }, {ram_size = }, {logical_size = }")
    print(f"""
Memory stats:
    Disk size: {disk_size}
    RAM size: {ram_size}
    Logical size: {logical_size}
    -
    shape: {datapipe.shape}
    dtype: {str(datapipe[0].dtype)}
    device: {datapipe[0].device}
""")
    

def mean_output_benchmark(pt, desc: str="output") -> torch.Tensor:
    t = MultiBlockTimer()
    with t:
        m = sinks.mean(DataPipe(pt))

    print(human_readable_time(t.total))
    contrast_computation_bandwidth: float = (get_logical_size(pt) / t.total)
    print(f"""
Computed temporal_mean({desc}) of {len(pt)} frames (from raw lsci frames):
    Time: {human_readable_time(t.total)}
    Datapipe output computed at: {human_readable_filesize(int(contrast_computation_bandwidth))}/s
    """)
    print_memory_stats(pt)
    plot(m)
    return m