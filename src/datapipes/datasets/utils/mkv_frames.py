from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import threading

import av
import torch

from datapipes.utils import Slicer
import ast

class MkvFrames:
    """
    Lazy MKV frame reader using PyAV.

    This version avoids scanning the whole file up front.
    It is optimized for sequential access, but also supports random access.

    Returns:
        grayscale=True  -> (1, H, W) uint8
        grayscale=False -> (3, H, W) uint8
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        # shape: Optional[tuple[int, int, int, int]] = None,
        stream_index: int = 0,
        grayscale: bool = True,
        cache_size: int = 32,
    ) -> None:
        self.path = str(path)
        self.stream_index = stream_index
        self.grayscale = grayscale
        self.cache_size = cache_size

        self._container: Optional[av.container.input.InputContainer] = None
        self._stream: Optional[av.video.stream.VideoStream] = None
        self._decoder = None
        self._lock = threading.RLock()

        self._num_frames: Optional[int] = None
        self._shape: Optional[tuple[int, int, int, int]] = None

        # Sequential decode state
        self._current_index: int = -1
        self._eof: bool = False

        # Small frame cache: idx -> tensor
        self._cache: dict[int, torch.Tensor] = {}

        self._open()

    def _open(self) -> None:
        if self._container is None:
            self._container = av.open(self.path, mode="r")
            self._shape = ast.literal_eval(self._container.metadata["SHAPE"])
            self._stream = self._container.streams.video[self.stream_index]
            self._stream.thread_type = "AUTO"
            self._reset_decoder()

    @property
    def stream(self) -> av.video.stream.VideoStream:
        self._open()
        assert self._stream is not None
        return self._stream

    def _reset_decoder(self) -> None:
        assert self._container is not None
        self._decoder = self._container.decode(self.stream)
        self._current_index = -1
        self._eof = False

    def close(self) -> None:
        with self._lock:
            if self._container is not None:
                self._container.close()
                self._container = None
                self._stream = None
                self._decoder = None

    def __enter__(self) -> "MkvFrames":
        self._open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _frame_to_tensor(self, frame: av.VideoFrame) -> torch.Tensor:
        if self.grayscale:
            arr = frame.to_ndarray(format="gray")  # (H, W)
            return torch.from_numpy(arr).unsqueeze(0).contiguous()  # (1, H, W)
        else:
            arr = frame.to_ndarray(format="rgb24")  # (H, W, 3)
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)

    def _add_to_cache(self, idx: int, frame: torch.Tensor) -> None:
        self._cache[idx] = frame
        if len(self._cache) > self.cache_size:
            oldest = min(self._cache.keys())
            del self._cache[oldest]

    def __len__(self) -> int:
        return self._shape[0]


    def _decode_next(self) -> torch.Tensor:
        assert self._decoder is not None

        try:
            frame = next(self._decoder)
        except StopIteration:
            self._eof = True
            raise IndexError("frame index out of range")

        self._current_index += 1
        t = self._frame_to_tensor(frame)
        self._add_to_cache(self._current_index, t)
        return t

    def _seek_to_start(self) -> None:
        assert self._container is not None
        self._container.seek(0, backward=True, any_frame=False, stream=self.stream)
        self._reset_decoder()

    def _seek_near_frame(self, target_idx: int) -> None:
        """
        Approximate seek using average_rate/time_base metadata.
        Then decode forward until the requested frame index is reached.

        This is approximate; compressed video seeking lands on keyframes.
        """
        s = self.stream
        if s.average_rate is None or s.time_base is None:
            self._seek_to_start()
            return

        fps = float(s.average_rate)
        if fps <= 0:
            self._seek_to_start()
            return

        seconds = target_idx / fps
        pts = int(seconds / float(s.time_base))

        assert self._container is not None
        self._container.seek(pts, backward=True, any_frame=False, stream=s)
        self._reset_decoder()

    def _get_frame(self, idx: int) -> torch.Tensor:
        if not isinstance(idx, int):
            raise TypeError(f"Expected idx to be of type `int`, got `{type(idx)}`")

        if idx in self._cache:
            return self._cache[idx]

        with self._lock:
            self._open()

            if idx in self._cache:
                return self._cache[idx]

            # Fast path: sequential forward decode
            if not self._eof and idx == self._current_index + 1:
                return self._decode_next()

            if not self._eof and idx > self._current_index + 1 and self._current_index >= -1:
                while self._current_index < idx:
                    frame = self._decode_next()
                return frame

            # Backward/random access: seek approximately, then decode forward
            self._seek_near_frame(idx)

            while self._current_index < idx:
                frame = self._decode_next()

            return frame

    def _get_slice(self, s: slice) -> torch.Tensor:
        start, stop, step = s.indices(len(self))
        frames = [self._get_frame(i) for i in range(start, stop, step)]
        if not frames:
            c = 1 if self.grayscale else 3
            return torch.empty((0, c, 0, 0), dtype=torch.uint8)
        return torch.stack(frames, dim=0)

    def __getitem__(self, idx: int|slice|tuple[int|slice, ...]) -> torch.Tensor:
        idx = Slicer.normalize(idx, shape=self.shape)
        remaining = None
        if isinstance(idx, tuple):
            remaining = idx[1:]
            idx = idx[0]

        out = None
        if isinstance(idx, slice):
            out = self._get_slice(idx)
        elif isinstance(idx, int):
            out = self._get_frame(idx)
        else:
            raise TypeError(f"Expected idx to be of type `int`, got `{type(idx)}`")
        
        if remaining:
            return out[:, *remaining]
        else:
            return out
    
    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self._shape