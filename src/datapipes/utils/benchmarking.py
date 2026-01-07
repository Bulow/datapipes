from datapipes import datasets
import time

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