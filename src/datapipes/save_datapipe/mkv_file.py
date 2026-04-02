from __future__ import annotations

import base64
import json
import threading
import zlib
from fractions import Fraction
from pathlib import Path
from typing import Dict, Optional, Union

import av
import numpy as np
import torch


def _encode_pts_table(pts_table: list[int]) -> str:
    """
    Store pts table compactly as compressed int64 bytes in base64.
    """
    arr = np.asarray(pts_table, dtype=np.int64)
    raw = arr.tobytes(order="C")
    compressed = zlib.compress(raw, level=9)
    return base64.b64encode(compressed).decode("ascii")


def _decode_pts_table(encoded: str) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(encoded.encode("ascii")))
    return np.frombuffer(raw, dtype=np.int64)


def write_ffv1_mkv_with_embedded_index(
    data,
    out_path: str | Path,
    *,
    fps: int = 60,
    batch_size: int = 512,
    overwrite: bool = False,
    stream_options: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Write FFV1-in-MKV and embed an exact per-frame packet-pts index.

    Input:
        data.shape == (N, C, H, W), with C in {1, 3}
        data batches must be uint8

    Embedded metadata key:
        container.metadata["DP_INDEX_V2"]

    Notes:
      - This records the *actual muxed packet pts* values, which are the correct
        units for container.seek(..., stream=video_stream).
      - For FFV1, one packet per frame is expected in presentation order.
    """
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".mkv":
        raise ValueError(f"Expected .mkv output, got {out_path.suffix}")

    if out_path.exists() and not overwrite:
        print(f"Skipping file: {out_path} already exists")
        return {}

    out_path.parent.mkdir(parents=True, exist_ok=True)

    N, C, H, W = data.shape
    if C not in (1, 3):
        raise ValueError(f"Expected input shape (N,1,H,W) or (N,3,H,W), got C={C}")

    container = av.open(
        str(out_path),
        mode="w",
        container_options={
            "reserve_index_space": str(1024 * 1024),
        },
    )

    stream = container.add_stream("ffv1", rate=fps)
    stream.width = W
    stream.height = H
    stream.pix_fmt = "gray" if C == 1 else "yuv444p"

    # Keep codec-side timestamps simple and monotonic.
    # Stream-side timestamps may still be rescaled by the muxer.
    stream.time_base = Fraction(1, fps)
    stream.options = stream_options or {"level": "3"}

    pts_table: list[int] = []
    frame_idx = 0

    # Initial metadata placeholder.
    container.metadata["DP_INDEX_V2"] = json.dumps(
        {
            "version": 2,
            "index_kind": "packet_pts_table",
            "codec": "ffv1",
            "fps": int(fps),
            "num_frames": int(N),
            "channels": int(C),
            "height": int(H),
            "width": int(W),
            "pix_fmt": stream.pix_fmt,
            "codec_time_base_num": int(stream.time_base.numerator),
            "codec_time_base_den": int(stream.time_base.denominator),
            "pts_table_codec": "zlib+base64+int64",
            "pts_table": "",
        },
        separators=(",", ":"),
    )

    for batch in data.batches_with_progressbar(batch_size=batch_size):
        if batch.ndim != 4:
            raise ValueError(f"Expected 4D batch (N,C,H,W), got {tuple(batch.shape)}")
        if batch.dtype != torch.uint8:
            raise TypeError(f"Expected uint8 batches, got {batch.dtype}")
        if batch.shape[1] != C:
            raise ValueError(
                f"Inconsistent channel count in batch. Expected {C}, got {batch.shape[1]}"
            )

        if C == 1:
            gray_np = batch[:, 0].cpu().numpy()  # (N,H,W)
            frames = [
                av.VideoFrame.from_ndarray(arr, format="gray")
                for arr in gray_np
            ]
        else:
            rgb_np = batch.permute(0, 2, 3, 1).cpu().numpy()  # (N,H,W,3)
            frames = [
                av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(
                    width=W,
                    height=H,
                    format="yuv444p",
                )
                for arr in rgb_np
            ]

        for vf in frames:
            # Frame pts are in codec time base.
            vf.pts = frame_idx
            vf.time_base = stream.time_base

            packets = stream.encode(vf)
            if len(packets) != 1:
                raise RuntimeError(
                    f"Expected exactly 1 packet per FFV1 frame, got {len(packets)} at frame {frame_idx}"
                )

            pkt = packets[0]
            if pkt.pts is None:
                raise RuntimeError(f"Encoded packet has no pts at frame {frame_idx}")

            pts_table.append(int(pkt.pts))
            container.mux(pkt)
            frame_idx += 1

    flush_packets = stream.encode()
    if len(flush_packets) != 0:
        # FFV1 should not produce delayed packets here in a way that breaks 1:1 indexing.
        raise RuntimeError(
            f"Expected no delayed packets on flush for FFV1, got {len(flush_packets)}"
        )

    if len(pts_table) != frame_idx:
        raise RuntimeError(
            f"PTS table length mismatch: len(pts_table)={len(pts_table)} frame_idx={frame_idx}"
        )

    embedded_index = {
        "version": 2,
        "index_kind": "packet_pts_table",
        "codec": "ffv1",
        "fps": int(fps),
        "num_frames": int(frame_idx),
        "channels": int(C),
        "height": int(H),
        "width": int(W),
        "pix_fmt": stream.pix_fmt,
        "codec_time_base_num": int(stream.time_base.numerator),
        "codec_time_base_den": int(stream.time_base.denominator),
        "pts_table_codec": "zlib+base64+int64",
        "pts_table": _encode_pts_table(pts_table),
    }
    container.metadata["DP_INDEX_V2"] = json.dumps(embedded_index, separators=(",", ":"))

    container.close()
    return embedded_index


def read_mkv_index_metadata(path: str | Path) -> dict:
    with av.open(str(path), mode="r") as container:
        raw = container.metadata.get("DP_INDEX_V2")
        if raw is None:
            raise KeyError("No embedded DP_INDEX_V2 metadata found")
        meta = json.loads(raw)
        if "pts_table" in meta and isinstance(meta["pts_table"], str) and meta["pts_table"]:
            meta["pts_table_array"] = _decode_pts_table(meta["pts_table"])
        return meta


class MkvFrames:
    """
    MKV frame reader for files written by write_ffv1_mkv_with_embedded_index().

    Uses an embedded per-frame packet-pts table so __getitem__(i) can seek using
    the correct stream timestamp units.

    Returned tensors:
      - grayscale source -> (1, H, W), uint8
      - color source     -> (3, H, W), uint8
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        stream_index: int = 0,
        cache_size: int = 64,
    ) -> None:
        self.path = str(path)
        self.stream_index = stream_index
        self.cache_size = cache_size

        self._container: Optional[av.container.input.InputContainer] = None
        self._stream: Optional[av.video.stream.VideoStream] = None
        self._lock = threading.RLock()

        self._index_meta: Optional[dict] = None
        self._pts_table: Optional[np.ndarray] = None
        self._shape: Optional[tuple[int, int, int, int]] = None
        self._cache: dict[int, torch.Tensor] = {}

    def _open(self) -> None:
        if self._container is None:
            self._container = av.open(self.path, mode="r")
            self._stream = self._container.streams.video[self.stream_index]
            self._stream.thread_type = "AUTO"

    @property
    def container(self) -> av.container.input.InputContainer:
        self._open()
        assert self._container is not None
        return self._container

    @property
    def stream(self) -> av.video.stream.VideoStream:
        self._open()
        assert self._stream is not None
        return self._stream

    def close(self) -> None:
        with self._lock:
            if self._container is not None:
                self._container.close()
                self._container = None
                self._stream = None

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

    def _load_index_meta(self) -> dict:
        if self._index_meta is not None:
            return self._index_meta

        self._open()
        raw = self.container.metadata.get("DP_INDEX_V2")
        if raw is None:
            raise KeyError("This MKV does not contain embedded 'DP_INDEX_V2' metadata.")

        meta = json.loads(raw)

        if meta.get("version") != 2:
            raise ValueError(f"Unsupported index version: {meta.get('version')}")
        if meta.get("index_kind") != "packet_pts_table":
            raise ValueError(f"Unsupported index kind: {meta.get('index_kind')}")

        required = ("num_frames", "channels", "height", "width", "pts_table")
        missing = [k for k in required if k not in meta]
        if missing:
            raise ValueError(f"Embedded index missing keys: {missing}")

        self._index_meta = meta
        self._pts_table = _decode_pts_table(meta["pts_table"])

        if len(self._pts_table) != int(meta["num_frames"]):
            raise ValueError(
                f"PTS table length {len(self._pts_table)} does not match num_frames {meta['num_frames']}"
            )

        return meta

    @property
    def index_meta(self) -> dict:
        return self._load_index_meta()

    @property
    def pts_table(self) -> np.ndarray:
        self._load_index_meta()
        assert self._pts_table is not None
        return self._pts_table

    @property
    def shape(self) -> tuple[int, int, int, int]:
        if self._shape is None:
            m = self.index_meta
            self._shape = (
                int(m["num_frames"]),
                int(m["channels"]),
                int(m["height"]),
                int(m["width"]),
            )
        return self._shape

    def __len__(self) -> int:
        return self.shape[0]

    def _normalize_index(self, idx: int) -> int:
        n = len(self)
        if idx < 0:
            idx += n
        if not (0 <= idx < n):
            raise IndexError(f"frame index out of range: {idx}")
        return idx

    def _frame_to_tensor(self, frame: av.VideoFrame) -> torch.Tensor:
        c = self.shape[1]
        if c == 1:
            arr = frame.to_ndarray(format="gray")
            return torch.from_numpy(arr).unsqueeze(0).contiguous()
        if c == 3:
            arr = frame.to_ndarray(format="rgb24")
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        raise ValueError(f"Unsupported channel count: {c}")

    def _add_to_cache(self, idx: int, frame: torch.Tensor) -> None:
        self._cache[idx] = frame
        if len(self._cache) > self.cache_size:
            oldest = min(self._cache.keys())
            del self._cache[oldest]

    def _target_pts(self, idx: int) -> int:
        return int(self.pts_table[idx])

    def _seek_and_decode_exact(self, idx: int) -> torch.Tensor:
        idx = self._normalize_index(idx)
        if idx in self._cache:
            return self._cache[idx]

        target_pts = self._target_pts(idx)

        with self._lock:
            self._open()

            if idx in self._cache:
                return self._cache[idx]

            self.container.seek(target_pts, backward=True, any_frame=False, stream=self.stream)

            for frame in self.container.decode(self.stream):
                if frame.pts is None:
                    continue
                if frame.pts < target_pts:
                    continue
                if frame.pts == target_pts:
                    out = self._frame_to_tensor(frame)
                    self._add_to_cache(idx, out)
                    return out
                if frame.pts > target_pts:
                    break

        raise IndexError(f"Could not decode exact frame {idx} (expected pts={target_pts})")

    def _get_slice_fast(self, s: slice) -> torch.Tensor:
        start, stop, step = s.indices(len(self))
        indices = list(range(start, stop, step))

        if not indices:
            _, c, h, w = 0, self.shape[1], self.shape[2], self.shape[3]
            return torch.empty((0, c, h, w), dtype=torch.uint8)

        if step != 1:
            return torch.stack([self._seek_and_decode_exact(i) for i in indices], dim=0)

        if all(i in self._cache for i in indices):
            return torch.stack([self._cache[i] for i in indices], dim=0)

        first = indices[0]
        last = indices[-1]
        first_pts = self._target_pts(first)
        last_pts = self._target_pts(last)

        with self._lock:
            self._open()
            self.container.seek(first_pts, backward=True, any_frame=False, stream=self.stream)

            got_first = False
            frames_by_index: dict[int, torch.Tensor] = {}

            for frame in self.container.decode(self.stream):
                pts = frame.pts
                if pts is None:
                    continue
                if pts < first_pts:
                    continue

                # Lock onto the first requested frame.
                if not got_first:
                    if pts != first_pts:
                        if pts > first_pts:
                            break
                        continue
                    got_first = True

                # We only care about frames whose pts are in our requested table range.
                if pts > last_pts:
                    break

                # Translate stream pts back to frame index.
                # Since the table is monotonic, use binary search.
                pos = int(np.searchsorted(self.pts_table, pts, side="left"))
                if pos >= len(self.pts_table):
                    continue
                if int(self.pts_table[pos]) != int(pts):
                    continue
                if pos < first or pos > last:
                    continue

                t = self._frame_to_tensor(frame)
                frames_by_index[pos] = t
                self._add_to_cache(pos, t)

                if len(frames_by_index) == (last - first + 1):
                    break

        # Fallback only for any missed frames.
        out = []
        for i in indices:
            if i in self._cache:
                out.append(self._cache[i])
            else:
                out.append(self._seek_and_decode_exact(i))

        return torch.stack(out, dim=0)

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        if isinstance(idx, slice):
            return self._get_slice_fast(idx)

        idx = self._normalize_index(idx)
        if idx in self._cache:
            return self._cache[idx]
        return self._seek_and_decode_exact(idx)