from typing import Any, Optional, Tuple
import threading
import torch

from datapipes import datasets
from datapipes.datasets.utils.compressed_image_stream_tensor import CompressedImageStreamTensor
import h5py
import time

from tqdm.notebook import tqdm_notebook

from datapipes.utils.benchmarking import human_readable_filesize, get_disk_size, get_logical_size

from pathlib import Path

Index = Any

from datapipes.utils.benchmarking import human_readable_filesize

from datapipes.datasets.utils.background_progress_jupyter import ThreadSafeProgress

class CompressedCachedDataset(datasets.DatasetSource):
    """
    Compressed cached DatasetSource wrapper with sequential background prefetch.

    Note: By compressing frames cached in RAM, much less time is spent on expensive RAM->VRAM transfers when they are used for GPU computations compared to using CachedDataset which stores the frames at full size.
    """
    def __init__(
        self,
        underlying_compressed_ds: datasets.DatasetCompressedImageStreamHdf5,
        prefetch_batchsize: int = 512,
        device: torch.device | str = "cpu",
        daemon: bool = True,
        pin_memory: bool=True,
    ) -> None:
        self._underlying_dataset = underlying_compressed_ds
        # self._path = self.remote_compressed_ds.path

        self.storage_size: int = get_disk_size(self._underlying_dataset)
        self.logical_size: int = get_logical_size(self._underlying_dataset)


        

        self._tensors_allocated: bool = False
        self.remote_compressed_frames: h5py.Dataset = self._underlying_dataset.frames
        self.lengths: torch.Tensor = self._underlying_dataset.lengths[:].to(torch.int64)
        self.offsets: torch.Tensor = self._underlying_dataset.offsets[:].to(torch.int64)

        self._shape = torch.Size(underlying_compressed_ds.shape)
        if len(self._shape) != 4:
            raise ValueError(f"Expected shape (n, c, h, w), got {tuple(self._shape)}")

        self._dtype = torch.uint8
        self._device = torch.device(device)
        self._batch = int(prefetch_batchsize)
        if self._batch <= 0:
            raise ValueError("batch_frames must be > 0")

        n, c, h, w = self._shape

        self._pin_memory= pin_memory
        self._cache: torch.Tensor
        self.lazy_decoding_tensor: CompressedImageStreamTensor

        # Validity tracked per frame (dim 0). Keep on CPU for cheap synchronization.
        self._valid = torch.zeros(n, dtype=torch.bool, device="cpu")

        self._progress_bar: ThreadSafeProgress = ThreadSafeProgress(total_storage_size = self.storage_size, total_logical_size=self.logical_size, total_type_used_for_stats="storage", desc="Prefetching frames in the background", path=self.path).display()

        # Thread coordination
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._error: Optional[BaseException] = None
        self._prefetch_pos = 0  # next frame index to fetch
        self._cache_fill_pos = 0

        self._thread = threading.Thread(target=self._prefetch_loop, daemon=daemon)
        self._thread.start()

    def _fetch_compressed_frames(self, idx: slice) -> torch.Tensor:
        start = idx.start or 0
        stop = idx.stop or len(self)
        
        # TODO: remove
        lengths = self.lengths[start:stop]
        offsets = self.offsets[start:stop]

        # Read entire encoded batch in a single call and use views to split it
        memory_start_index = offsets[0]
        memory_stop_index = offsets[-1] + lengths[-1]
        
        raw_stream = torch.from_numpy(self.remote_compressed_frames[memory_start_index:memory_stop_index])
        return raw_stream
    
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
            start_time = time.perf_counter()
            n = self._shape[0]
            # print(f"Prefetching {n} compressed frames ({human_readable_filesize(len(self.remote_compressed_frames))})")
            self._progress_bar.set_status(f"Allocating memory ({human_readable_filesize(self.storage_size)})")
            

            # Allocate cache tensor:
            # Full cache allocation (RAM / chosen device)
            self._cache = torch.empty(self.remote_compressed_frames.shape, dtype=self._dtype, device=self._device, pin_memory=self._pin_memory)

            self.lazy_decoding_tensor: CompressedImageStreamTensor = CompressedImageStreamTensor(
                frames=self._cache,
                lengths=self.lengths,
                offsets=self.offsets,
                individual_frame_shape=self.shape[0:]
            )

            with self._cv:
                self._tensors_allocated = True
            
            self._progress_bar.set_status("Loading dataset into RAM. The dataset is ready to use.")

            while not self._stop.is_set():
                with self._cv:
                    start = self._prefetch_pos
                    if start >= n:
                        stop_time: float = time.perf_counter()
                        prefetch_time: float = stop_time - start_time
                        
                        # print(f"Finished prefetching {n} frames ({human_readable_filesize(self._ds_size)}) in {prefetch_time:.2f}s, averaging {human_readable_filesize(int(self._ds_size / prefetch_time))}/s")
                        self._progress_bar.set_status("Done")
                        self._progress_bar.close()
                        self._cv.notify_all()
                        return
                    end = min(n, start + self._batch)
                    self._prefetch_pos = end

                # Fetch outside the lock
                sl = slice(start, end)  # contiguous along dim 0
                fetched = self._fetch_compressed_frames(sl)
                # print(fetched.shape)
                if not isinstance(fetched, torch.Tensor):
                    fetched = torch.as_tensor(fetched)
                fetched = fetched.to(dtype=self._dtype, device=self._device)

                # Store and mark valid
                batch_end_pos = self._cache_fill_pos + len(fetched)
                self._cache[self._cache_fill_pos:batch_end_pos] = fetched
                self._cache_fill_pos = batch_end_pos
                # print(batch_end_pos)
                with self._cv:
                    self._valid[start:end] = True
                    self._cv.notify_all()
                time.sleep(0)
                self._progress_bar.report(n=len(fetched))

        except BaseException as e:
            with self._cv:
                self._error = e
                self._progress_bar.error(str(e))
                self._cv.notify_all()
                raise e

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

        return self.lazy_decoding_tensor[index]
    
    

    def __len__(self) -> int:
        return self._shape[0]

    def __repr__(self) -> str:
        return f"CachedTensor(shape={tuple(self.shape)}, dtype={self.dtype}, batch_frames={self._batch})"
