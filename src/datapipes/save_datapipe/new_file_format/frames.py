from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Protocol, Tuple

import einops

# Base on h5py initially
import h5py as storage_backend

# import torch
import imagecodecs
import numpy as np

from datapipes.save_datapipe.new_file_format.parallel_encode import (
    encode_frames_threaded,
)

from datapipes.save_datapipe.new_file_format.codecs import get_encoder, get_decoder

from datapipes.save_datapipe.new_file_format.lazy_decoding_image_tensor import LazyDecodingImageTensor

from datapipes.datasets import DatasetSource

from datapipes.save_datapipe.new_file_format.wrapper_dataset import WrapperDataset

from dataclasses import dataclass

def _is_string_dtype(dt):
    # bytes (S), unicode (U), or object that may contain strings
    return dt.kind in ("S", "U") or dt.kind == "O"

def get_val(group: storage_backend.Group, name: str) -> Any:
    data = group[name]
    if isinstance(data, storage_backend.Dataset):
        # If data is a single value
        if data.shape == ():
            val = data[()]
            if hasattr(val, "item"):
                val = val.item()
            # decode if string
            if _is_string_dtype(data.dtype):
                try:
                    val = val.decode('utf-8')
                except Exception:
                    val = val.decode(errors='replace')
            return val
    
    # Passthrough
    return data

def encode_frame_by_frame(
    frames: np.ndarray, 
    encode_func: Callable[[np.ndarray], Iterable[bytes]],
    **kwargs,
) -> Iterable[bytes]:
    return [encode_func(frame) for frame in frames]

def decode_frame_by_frame(
    encoded_frames: Iterable[bytes],
    decode_func: Callable[[bytes], np.ndarray],
    **kwargs,
) -> np.ndarray:
    out = np.array([decode_func(encoded) for encoded in encoded_frames])
    if out.ndim == 3 and out.shape[0] > 1:
        out = einops.rearrange("n h w -> n 1 h w")
    return out

# def get_encoder(codec_name: str) -> Callable[[np.ndarray], Iterable[bytes]]:
#     match codec_name:
#         case "j2k":
#             return partial(encode_frames_threaded, encode_func=partial(imagecodecs.htj2k_encode, reversible=True))
#         case "jxl":
#             return partial(encode_frames_threaded, encode_func=partial(imagecodecs.jpegxl_encode, lossless=True))
#         case _:
#             raise NotImplementedError

# def get_decoder(codec_name: str) -> Callable[[Iterable[bytes]], np.ndarray]:
#     match codec_name:
#         case "j2k": # TODO: decode directly into assembled ndarray using views and the out kwarg
#             return partial(decode_frame_by_frame, decode_func=imagecodecs.htj2k_decode)
#         case "jxl":
#             return partial(decode_frame_by_frame, decode_func=imagecodecs.jpegxl_decode)
#         case _:
#             raise NotImplementedError
    # def f(): raise NotImplementedError
    # return f



FRAMES_FORMAT_VERSION: str = "2026.01.22.1"

@dataclass
class CustomEncoder:
    encoder: Callable[[np.ndarray], bytes]
    base_codec: str
    is_batched_encoder: bool=False

class Frames(Protocol):
    def __init__(self, group: storage_backend.Group):
        self.group: storage_backend.Group = group

        self.individual_frame_shape: Tuple[int, ...] 

        self.dtype: np.dtype

        self.frame_count: int = 0

        self._codec: str
        self._container: str = "h5"
        self._frames_type_id: str = "compressed_codestream"

        self.encoded_frames: storage_backend.Dataset
        self.frame_lengths_bytes: storage_backend.Dataset
        self.frame_start_memory_offsets: storage_backend.Dataset
        self.timestamps: storage_backend.Dataset

        self._current_frame_capacity: int
        self._current_codestream_capacity: int
        self._current_codestream_position: int
        self._encode_func: Callable[[np.ndarray], Iterable[bytes]]
        self._decode_func: Callable[[Iterable[bytes]], np.ndarray]

        self.parent = None

        self.lazy_decoder: LazyDecodingImageTensor

        self.frames_format_version: str = FRAMES_FORMAT_VERSION


    @classmethod
    def create(cls, group: storage_backend.Group, individual_frame_shape: Tuple[int, ...], dtype: np.dtype, codec: Literal["j2k", "jxl"]|tuple[Callable[[np.ndarray], bytes], str] = "j2k"):
        f = Frames(group)

        f.individual_frame_shape = individual_frame_shape

        f.dtype = np.array([], dtype=dtype).dtype # Convert dtype to np.dtype

        f.frame_count = 0

        f._container = "h5"
        f._frames_type_id = "compressed_codestream"

        f._current_frame_capacity = 1024
        f._current_codestream_capacity = 10**9
        f._current_codestream_position = 0
        
        if isinstance(codec, CustomEncoder):
            f._custom_encoder = codec
            f._codec = codec.base_codec
            f._encode_func = codec.encoder if codec.is_batched_encoder else (lambda f: encode_frames_threaded(f, codec.encoder))
        else:
            f._codec = codec
            f._encode_func = get_encoder(f._codec)

        f._decode_func = get_decoder(f._codec)

        f.init_group()

        # Must come after initializing group
        f.lazy_decoder = LazyDecodingImageTensor(
            frames=f.encoded_frames,
            lengths=f.frame_lengths_bytes,
            offsets=f.frame_start_memory_offsets,
            decode_func=f._decode_func,
            individual_frame_shape=f.shape[1:]
        )
        f.__getitem__ = f.lazy_decoder.__getitem__

        return f

    @classmethod
    def open(cls, group: storage_backend.Group):
        f = Frames(group)

        def load_str(name: str) -> str:
            return bytes(group[name][:]).decode("utf-8")

        # Load values from group
        n, c, h , w = tuple(int(d) for d in group["shape"])
        f.dtype = get_val(group, "dtype") #s- # TODO: Parse dtype instead of setting to str(dtype)
        f._codec = get_val(group, "codec") #s-
        f._frames_type_id = get_val(group, "frames_type_id") #s-
        f.frames_format_version = get_val(group, "frames_format_version") #s-

        f.encoded_frames = group["encoded_frames"]
        f.frame_lengths_bytes = group["frame_lengths_bytes"]
        f.frame_start_memory_offsets = group["frame_start_memory_offsets"]
        f.timestamps = group["timestamps"]

        f.individual_frame_shape = (c, h, w)
        # f.frame_count = len(f.frame_lengths_bytes)
        f.frame_count = f.frame_lengths_bytes[:].nonzero()[0].shape[0]

        f._container = "h5"
        f._frames_type_id = "compressed_codestream"

        f._current_frame_capacity = len(f.frame_lengths_bytes)
        f._current_codestream_capacity = len(f.encoded_frames)
        f._current_codestream_position = f._current_codestream_capacity
        f._encode_func = get_encoder(f._codec)
        f._decode_func = get_decoder(f._codec)

        f.lazy_decoder = LazyDecodingImageTensor(
            frames=f.encoded_frames,
            lengths=f.frame_lengths_bytes,
            offsets=f.frame_start_memory_offsets,
            decode_func=f._decode_func,
            individual_frame_shape=f.shape[1:]
        )

        return f
    
    def __getitem__(self, idx: int|slice|Tuple[slice|int, ...]):
        return self.lazy_decoder[idx]

    @property
    def path(self) -> Path:
        return Path(f"{self.group.file.filename}:{self.group.name}")

    # def __getitem__(self, idx: int|slice|Tuple[slice, ...]):
    #     return self.lazy_decoder[idx]

    def as_dataset(self) -> DatasetSource:
        return WrapperDataset(self)
        # raise NotImplementedError

    @classmethod
    def like_frame(cls, group: storage_backend.Group, frame: np.ndarray|Any, codec: Literal["j2k", "jxl"] = "j2k") -> "Frames":
        if frame.ndim != 3:
            raise ValueError(f"frame must have 3 dimensions, got {frame.ndim = }")
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        return Frames.create(group=group, individual_frame_shape=frame.shape, dtype=frame.dtype, codec=codec)

    def init_group(self):
        if "frames_type_id" in self.group.keys():
            self.load_frames() # TODO: Implement?
            return
        
        self.group["shape"] = self.shape # We technically write shape twice - the first time shape is (0 c h w) so we know the individual frame size if the recording fails without closing properly
        self.group["dtype"] = str(self.dtype)
        self.group["codec"] = self._codec
        self.group["frames_type_id"] = self._frames_type_id
        self.group["frames_format_version"] = self.frames_format_version

        self.encoded_frames = self.group.create_dataset(
            name="encoded_frames",
            shape=(self._current_codestream_capacity, ),
            maxshape=(None, ),
            dtype=np.uint8,
            chunks=True,
        )

        self.frame_lengths_bytes = self.group.create_dataset(
            name="frame_lengths_bytes",
            shape=(self._current_frame_capacity, ),
            maxshape=(None, ),
            dtype=np.uint64,
            chunks=True,
        )

        self.frame_start_memory_offsets = self.group.create_dataset(
            name="frame_start_memory_offsets",
            shape=(self._current_frame_capacity, ),
            maxshape=(None, ),
            dtype=np.uint64,
            chunks=True,
        )

        self.timestamps = self.group.create_dataset(
            name="timestamps",
            shape=(self._current_frame_capacity, ),
            maxshape=(None, ),
            dtype=np.uint64,
            chunks=True,
        )

    def set_timestamps(self, timestamps: np.ndarray):
        self.timestamps.resize(size=(len(timestamps), ))
        self.timestamps[0:len(timestamps)] = timestamps

    def add_frames(self, new_frames: np.ndarray, timestamps: Optional[np.ndarray]=None):
        # Ensure valid inputs
        if new_frames.shape[1:] != self.individual_frame_shape:
            raise ValueError(f"Expected shape {self.individual_frame_shape}, got {new_frames.shape = }")
        
        if new_frames.dtype != self.dtype:
            raise TypeError(f"Expected dtype {self.dtype}, got {new_frames.dtype = }")
        
        if (timestamps is not None) and timestamps.dtype != np.uint64:
            raise TypeError(f"Expected timestamps to have dtype {self.dtype}, got {timestamps.dtype = }")
        
        if (timestamps is not None) and len(new_frames) != len(timestamps):
            raise ValueError(f"new_frames and timestamps must have equal length, got {len(new_frames) = }, {len(timestamps) = }")

        if new_frames.ndim == 3:
            new_frames = einops.rearrange(new_frames, "c h w -> 1 c h w")

        # if new_frames.shape[1] == 3:
        #     new_frames = einops.rearrange(new_frames, "n c h w -> n h w")
        new_frames = np.ascontiguousarray(new_frames)

        # Encode data to be stored for batch
        encoded: Iterable[bytes] = list(self._encode_func(new_frames))
        # encoded = tuple(_encoded)
        # print(f"{len(encoded[0])}")
        flat_array = np.frombuffer(b"".join(encoded), dtype=np.uint8)
        # print(f"{len(encoded[0])}, {len(flat_array) = }")
        lengths = np.fromiter([len(f) for f in encoded], dtype=np.uint64)
        offsets = np.empty_like(lengths)
        offsets[0] = 0
        np.cumsum(lengths[:-1], out=offsets[1:])

        # Compute global indices to store the data at
        current_batch_frame_length = len(new_frames)
        current_batch_byte_length = len(flat_array)
        frame_indices = slice(self.frame_count, self.frame_count + current_batch_frame_length)
        byte_indices = slice(self._current_codestream_position, self._current_codestream_position + current_batch_byte_length)

        # Resize byte count oriented arrays if needed
        size_after_write = self._current_codestream_position + current_batch_byte_length
        if (self._current_codestream_capacity < size_after_write):
            self._current_codestream_capacity = size_after_write * 2
            self.encoded_frames.resize((self._current_codestream_capacity, ))

        # Resize frame count oriented arrays if needed
        frame_count_after_write = self.frame_count + current_batch_frame_length
        if (self._current_frame_capacity < frame_count_after_write):
            self._current_frame_capacity = frame_count_after_write * 2
            self.frame_lengths_bytes.resize((self._current_frame_capacity, ))
            self.frame_start_memory_offsets.resize((self._current_frame_capacity, ))
            if timestamps is not None:
                self.timestamps.resize((self._current_frame_capacity, ))

        # Write data
        self.frame_lengths_bytes[frame_indices] = lengths
        self.frame_start_memory_offsets[frame_indices] = offsets + self._current_codestream_position
        self.encoded_frames[byte_indices] = flat_array
        if timestamps is not None:
            self.timestamps[frame_indices] = timestamps

        # Update position
        self.frame_count += current_batch_frame_length
        self._current_codestream_position += current_batch_byte_length

    def close(self) -> None:
        if self.group.file.mode == "w":
            self.group["shape"][...] = self.shape

            # Resize arrays to final size
            self.frame_lengths_bytes.resize((self.frame_count, ))
            self.frame_start_memory_offsets.resize((self.frame_count, ))
            self.timestamps.resize((self.frame_count, ))
            self.encoded_frames.resize((self._current_codestream_position, ))

        # Close file if needed
        # if isinstance(self.group, storage_backend.File):
        #     self.group.close()

    def load_frames(self):
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, ...]:
        return (len(self), *self.individual_frame_shape)
    
    def __len__(self):
        return self.frame_count
    
    