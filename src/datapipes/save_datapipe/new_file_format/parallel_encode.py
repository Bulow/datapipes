from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, Optional
import os
import numpy as np


def _encode_chunk(
    args: tuple[np.ndarray, Callable[[np.ndarray], Iterable[bytes]], int, int]
) -> list[bytes]:
    frames, encode_func, start, end = args
    out: list[bytes] = []

    for i in range(start, end):
        enc = encode_func(frames[i])  # typically a view; ok if encoder is read-only
        if isinstance(enc, (bytes, bytearray, memoryview)):
            out.append(bytes(enc))
        else:
            out.extend(bytes(part) for part in enc)

    return out


def encode_frames_threaded(
    frames: np.ndarray,
    encode_func: Callable[[np.ndarray], Iterable[bytes]],
    *,
    max_workers: Optional[int] = None,
    chunk_frames: int = 128,
    **kwargs,
) -> Iterator[bytes]:
    """
    High-throughput threaded encoder.
    Preserves frame order and yields a flat stream of bytes.
    """
    n = int(frames.shape[0])
    if n == 0:
        return iter(())

    chunk_frames = max(1, int(chunk_frames))
    workers = max_workers or (os.cpu_count() or 1)

    # Build tasks in order (start indices increasing).
    tasks = [
        (frames, encode_func, start, min(start + chunk_frames, n))
        for start in range(0, n, chunk_frames)
    ]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        # Results come back in task order, which matches frame order.
        for chunk_bytes in ex.map(_encode_chunk, tasks, chunksize=1):
            yield from chunk_bytes
