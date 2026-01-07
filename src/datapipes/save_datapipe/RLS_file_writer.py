import numpy as np
import os
import torch
import einops
from pathlib import Path
from typing import Tuple
from datapipes.datapipe import DataPipe
import struct
from datapipes.ops import Ops
from datapipes.manual_ops import with_manual_op


import mmap

from contextlib import contextmanager
from contextlib import ExitStack
from dataclasses import dataclass


class RLS_Writer:
    def __init__(self, mm: mmap.mmap, dtype: type, n_elements: int, offset: int = 30*1024):
        
        self.mm: mmap.mmap = mm
        self.dtype: type = dtype
        self.n_elements: int = n_elements

        self.disk_array_view = np.ndarray(buffer=self.mm, shape=self.n_elements, dtype=self.dtype, offset=offset)
    
    def __set_item__(self, index: slice, data: np.ndarray):
        self.disk_array_view[index] = data

    def interleave_batch(self, frames: torch.Tensor, timestamps: torch.Tensor):
        if len(frames) != len(timestamps):
            raise ValueError(f"frames and timestamps must be the same length, got len(frames): {len(frames)}, len(timestamps): {len(timestamps)}")
        
        current_batch_length = len(frames)
        interleave_buffer = np.empty(current_batch_length, dtype=self.dtype)

        interleave_buffer[0:current_batch_length]["frames"] = frames
        interleave_buffer[0:current_batch_length]["timestamps"] = timestamps

        return np.ascontiguousarray(interleave_buffer[0:current_batch_length])

@contextmanager
def prepare_rls_file(data: DataPipe, path: Path|str, n_writers: int=1):
    if isinstance(path, str):
        path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = len(data)
    first_frame: torch.Tensor = data[0]
    frame_size_bytes = first_frame.nbytes

    header_offset = 30 * 1024

    data = data | with_manual_op(lambda f: einops.rearrange(f, "n c h w -> n c w h"))

    if first_frame.device.type != "cpu":
        data = data | Ops.cpu

    channels, size_first_dim, size_second_dim = data.shape[-3:]
    # first_frame.dtype.itemsize

    # bytes_per_pixel = torch.tensor([], dtype=first_frame.dtype).element_size()
    timestamp_size = torch.tensor([], dtype=torch.uint64).element_size()

    rec_dtype = np.dtype([
            ("frames", np.uint8, first_frame.shape),
            ("timestamps", np.uint64),
        ])

    # writer = RLS_FileWriter
    with open(path, 'w+b') as f:
        # with [mmap.mmap(f.fileno(), length=header_offset + frame_count * (frame_size_bytes + timestamp_size), access=mmap.ACCESS_WRITE) for _ in range(n_writers)] as writers:
        with ExitStack() as stack:
            writer_mmaps = [
                stack.enter_context(
                    mmap.mmap(
                        f.fileno(),
                        length=header_offset + frame_count * (frame_size_bytes + timestamp_size),
                        access=mmap.ACCESS_WRITE
                    )
                )
                for _ in range(n_writers)
            ]
            rec = writer_mmaps[0]
            rec.write(struct.pack('Q', size_first_dim))
            rec.write(struct.pack('Q', size_second_dim))
            rec.write(struct.pack('Q', frame_count))
            rec.write(struct.pack('Q', 0)) # sample_rate
            rec.write(struct.pack('Q', 1)) # version

            for mm in writer_mmaps:
                mm.seek(header_offset)

            # yield rec
            if n_writers == 1:
                rls_writer = RLS_Writer(mm=rec, dtype=rec_dtype, n_elements=len(data))
                yield rls_writer
            elif n_writers > 1:
                yield [RLS_Writer(mm=w, dtype=rec_dtype, n_elements=len(data)) for w in writer_mmaps]
            else:
                raise ValueError(f"n_writers must be >= 1, got: {n_writers}")

            for mm in writer_mmaps:
                mm.flush()


def write_datapipe_to_rls(data: DataPipe, path: Path|str, batch_size: int=256):
    if isinstance(path, str):
        path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = len(data)
    first_frame: torch.Tensor = data[0]
    frame_size_bytes = first_frame.nbytes

    header_offset = 30 * 1024

    if first_frame.device.type != "cpu":
        data = data | Ops.cpu()

    channels, height, width = data.shape[-3:]
    # first_frame.dtype.itemsize

    # bytes_per_pixel = torch.tensor([], dtype=first_frame.dtype).element_size()
    timestamp_size = torch.tensor([], dtype=torch.uint64).element_size()

    # writer = RLS_FileWriter
    with open(path, 'w+b') as f:
        with mmap.mmap(f.fileno(), length=header_offset + frame_count * (frame_size_bytes + timestamp_size), access=mmap.ACCESS_WRITE) as rec:
            rec.write(struct.pack('Q', width))
            rec.write(struct.pack('Q', height))
            rec.write(struct.pack('Q', frame_count))
            rec.write(struct.pack('Q', 0)) # sample_rate
            rec.write(struct.pack('Q', 1)) # version

            rec.seek(header_offset)

            rec_dtype = np.dtype([
                ("frames", np.uint8, first_frame.shape),
                ("timestamps", np.uint64),
            ])

            interleave_buffer = np.empty(batch_size, dtype=rec_dtype)

            for i, batch in enumerate(data.batches_with_progressbar(batch_size=batch_size)):
                current_batch_length = len(batch)
                
                # Get memory location and structure
                from_frame_index = i * batch_size
                # from_memory_index = header_offset + (from_frame_index * (frame_size_bytes + timestamp_size))
                # to_memory_index = from_memory_index + (current_batch_length * (frame_size_bytes + timestamp_size))

                # TODO: Use timestamps from metadata
                timestamps = np.arange(start=from_frame_index, stop=from_frame_index + current_batch_length)
                # ts = np.asarray(timestamps, dtype=np.dtype("<u8"))

                interleave_buffer[0:current_batch_length]["frames"] = batch
                interleave_buffer[0:current_batch_length]["timestamps"] = timestamps

                
                rec.write(np.ascontiguousarray(interleave_buffer[0:current_batch_length]).tobytes())
