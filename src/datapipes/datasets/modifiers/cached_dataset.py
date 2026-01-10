from datapipes.datasets import DatasetSource
from typing import Any, Optional, Tuple
import threading
import torch
from pathlib import Path
from datapipes import datasets
import time


from datapipes.utils.benchmarking import human_readable_filesize, get_disk_size, get_logical_size

Index = Any

class CachedDataset(datasets.DatasetSource):
    """
    Cached DatasetSource wrapper with sequential background prefetch.

    """

    def __init__(
        self,
        # shape: Union[torch.Size, Tuple[int, int, int, int]],
        # dtype: torch.dtype,
        # fetch_fn: Callable[[slice], torch.Tensor],
        underlying_dataset: datasets.DatasetSource,
        prefetch_batchsize: int = 256,
        device: torch.device | str = "cpu",
        pin_memory: bool = True,
        daemon: bool = True,
    ) -> None:
        self._underlying_dataset: DatasetSource = underlying_dataset
        self._shape: torch.Size = torch.Size(self._underlying_dataset.shape)
        if len(self._shape) != 4:
            raise ValueError(f"Expected shape (n, c, h, w), got {tuple(self._shape)}")

        self._dtype = self._underlying_dataset[0].dtype
        self._device = torch.device(device)
        self._pin_memory: bool = pin_memory
        self._fetch_fn = self._underlying_dataset.__getitem__
        self._batch = int(prefetch_batchsize)
        if self._batch <= 0:
            raise ValueError("batch_frames must be > 0")

        self.storage_size = get_logical_size(self._underlying_dataset)

        
        self._tensors_allocated: bool = False
        self._cache: torch.Tensor
        # self._valid: torch.Tensor
        # Validity tracked per frame (dim 0). Keep on CPU for cheap synchronization.
        self._valid: torch.Tensor

        # Thread coordination
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._error: Optional[BaseException] = None
        self._prefetch_pos = 0  # next frame index to fetch

        self._thread = threading.Thread(target=self._prefetch_loop, daemon=daemon)
        self._thread.start()

    @property
    def timestamps(self) -> torch.LongTensor:
        return self._underlying_dataset.timestamps
    
    @property
    def path(self) -> Path:
        return Path(self._underlying_dataset.path)
    
    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def close(self) -> None:
        """Stop the background prefetch thread (best-effort)."""
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("Background prefetch thread failed") from self._error

    def _prefetch_loop(self) -> None:
        try:
            n, c, h, w = self._shape

            print(f"Prefetching {len(self)} frames ({human_readable_filesize(self.storage_size)}) in the background")
            start_time = time.perf_counter()
            

            self._valid = torch.zeros(self.shape[0], dtype=torch.bool, device="cpu", pin_memory=True)

            # Full cache allocation (RAM / chosen device)
            self._cache = torch.empty((n, c, h, w), dtype=self._dtype, device=self._device, pin_memory=self._pin_memory)

            
            with self._cv:
                self._tensors_allocated = True

            while not self._stop.is_set():
                with self._cv:
                    start = self._prefetch_pos
                    if start >= n:
                        stop_time = time.perf_counter()
                        prefetch_time: float = stop_time - start_time
                        
                        print(f"Finished prefetching {n} frames ({human_readable_filesize(self.storage_size)}) in {prefetch_time:.2f}s, averaging {human_readable_filesize(int(self.storage_size / prefetch_time))}/s")
                        self._cv.notify_all()
                        return
                    end = min(n, start + self._batch)
                    self._prefetch_pos = end

                # Fetch outside the lock
                sl = slice(start, end)  # contiguous along dim 0
                fetched = self._fetch_fn(sl)
                if not isinstance(fetched, torch.Tensor):
                    fetched = torch.as_tensor(fetched)
                fetched = fetched.to(dtype=self._dtype, device=self._device)

                # Store and mark valid
                self._cache[sl] = fetched
                with self._cv:
                    self._valid[start:end] = True
                    self._cv.notify_all()

        except BaseException as e:
            with self._cv:
                self._error = e
                self._cv.notify_all()

    def _normalize_slice_dim0(self, s: slice) -> tuple[int, int]:
        """
        Convert a dim-0 slice to concrete [start, stop) with step==1.
        Supports negative start/stop via slice.indices.
        """
        n = self._shape[0]
        start, stop, step = s.indices(n)
        if step != 1:
            raise ValueError("CachedTensor only supports contiguous slices along dim 0 (step must be 1).")
        return start, stop

    def _wait_until_valid_range(self, start: int, stop: int) -> None:
        """Block until all frames in [start, stop) are cached."""
        if stop <= start:
            return
        with self._cv:
            while True:
                self._raise_if_error()
                if self._stop.is_set():
                    raise RuntimeError("CachedTensor is closed/stopped while waiting for data.")
                if self._tensors_allocated and bool(self._valid[start:stop].all()):
                    return
                self._cv.wait(timeout=0.1)

    def block_until_fully_cached(self):
        self._wait_until_valid_range(0, len(self))

    def __getitem__(self, index: int|slice|Tuple) -> torch.Tensor:
        # Accept idx as slice or tuple whose first element is a slice
        if isinstance(index, tuple):
            if len(index) == 0 or not isinstance(index[0], slice):
                raise TypeError("First index must be a slice along dim 0, e.g. ct[a:b, ...].")
            dim0_slice = index[0]
        elif isinstance(index, slice):
            dim0_slice = index
        elif isinstance(index, int):
            dim0_slice = slice(index, index + 1)
        else:
            raise TypeError("Index must be a slice along dim 0, e.g. ct[a:b] or ct[a:b, ...].")

        start, stop = self._normalize_slice_dim0(dim0_slice)
        self._wait_until_valid_range(start, stop)

        return self._cache[index]

    def __len__(self) -> int:
        return self._shape[0]

    def __repr__(self) -> str:
        return f"CachedTensor(shape={tuple(self.shape)}, dtype={self.dtype}, batch_frames={self._batch})"
