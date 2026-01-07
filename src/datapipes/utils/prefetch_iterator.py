import threading
import queue
from typing import Generic, Iterable, Iterator, TypeVar, Union

T = TypeVar("T")


class PrefetchIterator(Generic[T]):
    """
    Wrap an iterator/iterable and prefetch items into a queue on a background thread.

    This provides an easy way to parallelize operations that iterate over a dataset. If you want to prefetch and cache an entire DatasetSource, consider using `CompressedCachedDataset` (for low RAM consumption and fast RAM->VRAM transfers) or `CachedDataset` (for fast CPU work with the raw frames stored in RAM)


    - Bounded queue provides backpressure (producer blocks when full).
    - Exceptions in the producer are re-raised in the consumer thread.
    - Supports clean shutdown via close() or context manager.
    """

    _SENTINEL = object()

    def __init__(
        self,
        source: Union[Iterable[T], Iterator[T]],
        *,
        max_prefetch: int = 3,
        daemon: bool = True,
    ) -> None:
        if max_prefetch < 1:
            raise ValueError("max_prefetch must be >= 1")

        self._it: Iterator[T] = iter(source)
        self._q: "queue.Queue[object]" = queue.Queue(maxsize=max_prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=daemon)

        self._started = False
        self._closed = False

    def _ensure_started(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def _put_blocking(self, item: object) -> None:
        """
        Put with periodic stop-checking so close() doesn't hang.
        """
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.1)
                return
            except queue.Full:
                continue
        # If we're stopping, don't block further.

    def _worker(self) -> None:
        try:
            for item in self._it:
                if self._stop.is_set():
                    break
                self._put_blocking(item)
        except BaseException as e:
            # Send exception to consumer
            self._put_blocking(e)
        finally:
            # Signal completion
            self._put_blocking(self._SENTINEL)

    def __iter__(self) -> "PrefetchIterator[T]":
        self._ensure_started()
        return self

    def __next__(self) -> T:
        self._ensure_started()
        if self._closed:
            raise StopIteration

        while True:
            obj = self._q.get()  # blocks until available
            if obj is self._SENTINEL:
                self._closed = True
                raise StopIteration
            if isinstance(obj, BaseException):
                self.close()
                raise obj
            return obj  # type: ignore[return-value]

    def close(self) -> None:
        """
        Stop producer and try to release consumer/producer promptly.
        Safe to call multiple times.
        """
        if self._closed and self._stop.is_set():
            return

        self._stop.set()

        # Try to nudge consumer(s) and allow worker to exit quickly.
        try:
            self._q.put_nowait(self._SENTINEL)
        except queue.Full:
            pass

        self._closed = True

    def __enter__(self) -> "PrefetchIterator[T]":
        self._ensure_started()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# --- Example usage ---
if __name__ == "__main__":
    import time

    def slow_numbers(n: int) -> Iterator[int]:
        for i in range(n):
            time.sleep(0.05)  # simulate slow production
            yield i

    for x in PrefetchIterator(slow_numbers(10), max_prefetch=3):
        # consumer can do work; production overlaps
        time.sleep(0.03)
        print(x)
