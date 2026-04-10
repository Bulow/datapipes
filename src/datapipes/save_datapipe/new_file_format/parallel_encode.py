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
    chunk_frames: int = 32,
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


def _decode_chunk(
    args: tuple[list[bytes], Callable[[bytes], np.ndarray], int, int]
) -> list[np.ndarray]:
    items, decode_func, start, end = args
    out: list[np.ndarray] = []

    for i in range(start, end):
        out.append(decode_func(items[i]))

    return out


def decode_frames_threaded(
    encoded: Iterable[bytes],
    decode_func: Callable[[bytes], np.ndarray],
    *,
    max_workers: Optional[int] = None,
    chunk_items: int = 32,
    **kwargs,
) -> Iterator[np.ndarray]:
    """
    High-throughput threaded decoder.
    Preserves encoded-item order and yields decoded frames lazily.

    Parameters
    ----------
    encoded:
        Iterable of encoded frame payloads, one item per frame.
    decode_func:
        Function that decodes one encoded payload into one frame array.
    max_workers:
        Number of worker threads. Defaults to os.cpu_count().
    chunk_items:
        Number of encoded items per submitted task.

    Yields
    ------
    np.ndarray
        Decoded frames in the same order as the input stream.
    """
    items = list(encoded)
    n = len(items)
    if n == 0:
        return iter(())

    chunk_items = max(1, int(chunk_items))
    workers = max_workers or (os.cpu_count() or 1)

    tasks = [
        (items, decode_func, start, min(start + chunk_items, n))
        for start in range(0, n, chunk_items)
    ]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        # Results come back in task order, which matches input order.
        for chunk_frames in ex.map(_decode_chunk, tasks, chunksize=1):
            yield from chunk_frames