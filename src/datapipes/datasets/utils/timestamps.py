import numpy as np
import torch
from typing import Dict, Tuple, Optional

camera_tick_frequencies: Dict[str, int] = {
    "acA2040-90umNIR": int(1e+9)
}

def human_readable_timestamp(ns: int) -> str:
    """
    Convert a duration given in nanoseconds into a human-readable string like:
      "2h 43m 12s 340ms 345us 129ns"

    Notes:
    - Units are: hours, minutes, seconds, milliseconds, microseconds, nanoseconds
    - Leading zero units are omitted.
    - If ns == 0, returns "0ns".
    - Accepts negative values and prefixes the result with '-'.
    """
    if isinstance(ns, torch.Tensor):
        ns = int(ns.item())
    if not isinstance(ns, int):
        raise TypeError("ns must be an int (nanoseconds)")

    sign = "-" if ns < 0 else ""
    ns = abs(ns)

    units = [
        ("h", 3_600_000_000_000),
        ("m", 60_000_000_000),
        ("s", 1_000_000_000),
        ("ms", 1_000_000),
        ("us", 1_000),
        ("ns", 1),
    ]

    parts = []
    for suffix, unit_size in units:
        value, ns = divmod(ns, unit_size)
        if value or parts or suffix == "ns":  # ensure we always emit something (at least ns)
            if value or (suffix == "ns" and len(parts) == 0):       # show 0 only if it's the final "ns"
                parts.append(f"{value}{suffix}")

    # If original was 0, parts will be ["0ns"] due to the logic above.
    return sign + " ".join(parts)


class Timestamps:
    def __init__(self, timestamps_tensor: torch.LongTensor, tick_frequency=1_000):
        if isinstance(timestamps_tensor, np.ndarray):
            timestamps_tensor = torch.from_numpy(timestamps_tensor)
        self.timestamps_tensor: torch.LongTensor = timestamps_tensor.to(torch.int64)
        self.tick_frequency = tick_frequency
        self.zero_time_ticks: torch.Tensor = self.timestamps_tensor[0]
        self.as_ns_factor = (1_000_000_000 / self.tick_frequency)

    def __len__(self):
        return len(self.timestamps_tensor)
    
    def __getitem__(self, idx: slice|int) -> torch.LongTensor:
        match idx:
            case slice():
                start = idx.start or 0
                stop = idx.stop or len(self)
            case int():
                start = idx
                stop = idx + 1
            case _:
                raise TypeError(f"Unsupported index type: {type(idx)}")
        
        abs_ticks = self.timestamps_tensor[slice(start, stop)] - self.zero_time_ticks
        time_s = abs_ticks * self.as_ns_factor
        return time_s.to(torch.int64)
        

