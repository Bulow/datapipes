import torch
import av
av.logging.set_level(av.logging.VERBOSE)
import numpy as np
from datapipes.datapipe import DataPipe
from pathlib import Path
# import test_compression_method as test
from typing import Dict, Tuple, Any, Callable, List, Optional, Literal

def _get_nv12(frames: torch.Tensor) -> torch.Tensor:
    assert frames.dtype == torch.uint8
    uv_shape = (*frames.shape[:-2], frames.shape[-2] // 2, frames.shape[-1])

    y_plane = frames.to("cuda")
    # UV plane (constant 128 for neutral chroma)
    
    uv_plane = torch.full(
        uv_shape,
        128,
        dtype=torch.uint8,
        device="cuda",
    )

    nv12 = torch.cat((y_plane, uv_plane), dim=-2)
    return nv12


def _get_tensor_raw_size(t: torch.Tensor) -> int:
    """Return raw size in bytes of tensor storage."""
    return int(t.nelement() * t.element_size())

def _datapipe_to_video(data: DataPipe, out_path: Path|str, fps: int = 60, codec: str="hevc_nvenc", stream_options: Optional[Dict]=None, overwrite: bool=False):
    """
    Encode a batch of grayscale frames stored as a CUDA tensor (N,1,H,W)
    into an H.264-in-MP4 file using NVENC via PyAV.

    frames: uint8 CUDA tensor, (N,1,H,W)
    out_path: output .mp4 filename
    fps: frames per second
    """
    assert len(data.shape) == 4
    assert data.shape[1] == 1, "Expect (N,1,H,W)"
    frames = data[0:1]
    assert frames.dtype == torch.uint8
    assert frames.is_cuda

    out_path = Path(out_path)
    assert out_path.suffix == ".mp4", f"file type must be .mp4 (got {out_path.suffix})"

    if not overwrite and out_path.exists():
        print(f"Skipping file: {out_path.as_posix()} already exists (Call with overwrite=True to disable this check)")
        return
    
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True)

    N, _, H, W = data.shape

    # Open MP4 container for writing
    container = av.open(out_path.as_posix(), mode="w")

    stream = container.add_stream(codec, rate=fps)
    stream.width = W
    stream.height = H
    stream.pix_fmt = "nv12"

    stream.options = stream_options or {
        "preset": "p7",        # p1..p7 (p7 = highest quality)
        "tune": "lossless",    # replaces old "lossless" preset
        "rc": "constqp",
        "qp": "0",             # QP=0 = mathematically lossless
    }

    max_frames = None
    for batch in data.batches_with_progressbar(batch_size=512, max_frames=max_frames):
        # for i in range(N):
        nv12_batch = _get_nv12(batch).cpu().numpy()
        for f in nv12_batch:

            video_frame = av.VideoFrame.from_ndarray(f[0], format="nv12")
            av.VideoStream
            video_frame.pict_type = 0

            # Encode and mux packets
            for packet in stream.encode(video_frame):
                container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()

    # in_total_size = _get_tensor_raw_size(raw[0]) * (max_frames or len(raw))
    # out_total_size = test.get_raw_file_size(out_path)

    # print(f"{[s for s in ((max_frames or len(raw)), *raw[0].shape)]} => [{test.human_readable_filesize(in_total_size)} -> {test.human_readable_filesize(out_total_size)} ({(out_total_size / in_total_size) * 100:.2f}%)] (-{test.human_readable_filesize(in_total_size - out_total_size)})")

# p = f"{str(Path("video_tests") / "vid.mp4")}"
# print(p)
# encode_tensor_to_mp4_nvenc(raw, p)

def datapipe_to_lossless_hevc_mp4(raw: DataPipe, out_path: str, fps: int = 60, overwrite: bool=False):
    codec = "hevc_nvenc"
    stream_options = {
        "preset": "p7",        # p1..p7 (p7 = highest quality)
        "tune": "lossless",    # replaces old "lossless" preset
        "rc": "constqp",
        "qp": "0",             # QP=0 = mathematically lossless
    }

    _datapipe_to_video(data=raw, out_path=out_path, fps=fps, codec=codec, stream_options=stream_options, overwrite=overwrite)

def datapipe_to_lossy_av1_mp4(raw: DataPipe, out_path: str, fps: int = 60, overwrite: bool=False, quality_preset: Literal["p1", "p2", "p3", "p4", "p5", "p6", "p7"] = "p7"):
    """
    p7 = highest quality
    
    """
    codec = "av1_nvenc"
    stream_options = {
        "preset": quality_preset,        # p1..p7 (p7 = highest quality)
        "b": "30M",          # target bitrate
        "maxrate": "60M",
        "bufsize": "16M",
        "rc": "vbr"
        # "tuning_info": "lossless",
        # "rc": "vbr"
    }
    # stream_options = {
    #     # Quality / rate control
    #     "b": "2M",          # target bitrate
    #     "maxrate": "2M",
    #     "bufsize": "4M",
    #     "rc": "vbr",        # rate control mode: cbr, vbr, constqp, etc.

    #     # Preset / speed vs quality
    #     "preset": "p5",     # typical NVENC presets: p1 (slow) ... p7 (fast)

    #     # Profile
    #     "profile": "main",  # or "high", etc., depending on GPU / SDK

    #     # Tuning
    #     # "aq": "1",        # adaptive quantization toggle, if supported
    #     # "multipass": "qres",  # example for some NVENC modes, depends on ffmpeg version
    # }

    _datapipe_to_video(data=raw, out_path=out_path, fps=fps, codec=codec, stream_options=stream_options, overwrite=overwrite)


    