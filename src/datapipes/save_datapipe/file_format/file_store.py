from typing import Callable, Any, Optional, Dict, Tuple, Iterable, Protocol, Literal
from dataclasses import dataclass, asdict, is_dataclass, field
import json
import numpy as np
from pathlib import Path

# Base on h5py initially
import h5py as storage_backend
# import torch
import imagecodecs

# filename.codec.container
# 
# brain.raw.h5
# brain.j2k.h5
# brain.jxl.h5

def encode_frame_by_frame(
    frames: np.ndarray, 
    encode_func: Callable[[np.ndarray], Iterable[bytes]]
) -> Iterable[bytes]:
    return [encode_func(frame) for frame in frames]

def decode_frame_by_frame(
    encoded_frames: Iterable[bytes],
    decode_func: Callable[[bytes], np.ndarray]
) -> np.ndarray:
    return np.array([decode_func(encoded) for encoded in encoded_frames])

def get_encoder(codec_name: str) -> Callable[[np.ndarray], Iterable[bytes]]:
    match codec_name:
        case "j2k":
            def encoder(frames: np.ndarray) -> Iterable[bytes]:
                return encode_frame_by_frame(frames, imagecodecs.htj2k_encode)
            return encoder
        case "jxl":
            def encoder(frames: np.ndarray) -> Iterable[bytes]:
                return encode_frame_by_frame(frames, imagecodecs.jpegxl_encode)
            return encoder
        case _:
            raise NotImplementedError

def get_decoder(codec_name: str) -> Callable[[Iterable[bytes]], np.ndarray]:
    match codec_name:
        case "j2k":
            def decoder(frames: np.ndarray) -> Iterable[bytes]:
                return decode_frame_by_frame(frames, imagecodecs.htj2k_decode)
            return decoder
        case "jxl":
            def decoder(frames: np.ndarray) -> Iterable[bytes]:
                return decode_frame_by_frame(frames, imagecodecs.jpegxl_decode)
            return decoder
        case _:
            raise NotImplementedError

class Frames(Protocol):
    def __init__(self, group: storage_backend.Group, individual_frame_shape: Tuple[int, ...], dtype: np.dtype):
        self.group: storage_backend.Group = group
        self.individual_frame_shape: Tuple[int, ...] = individual_frame_shape

        self.dtype: np.dtype = np.array([], dtype=dtype).dtype # Convert dtype to np.dtype

        self.frame_count: int = 0

        self._codec: str = "j2k"
        self._container: str = "h5"
        self._frames_type_id: str = "compressed_codestream"

        self.encoded_frames: storage_backend.Dataset
        self.frame_lengths_bytes: storage_backend.Dataset
        self.frame_start_memory_offsets: storage_backend.Dataset
        self.timestamps: storage_backend.Dataset

        self._current_frame_capacity: int = 1024
        self._current_codestream_capacity: int = 10**9
        self._current_codestream_position: int = 0
        self._encode_func: Callable[[np.ndarray], Iterable[bytes]] = get_encoder(self._codec)
        self._decode_func: Callable[[Iterable[bytes]], np.ndarray] = get_decoder(self._codec)

        self.init_group()

    @classmethod
    def like_frame(cls, group: storage_backend.Group, frame: np.ndarray|Any) -> "Frames":
        if frame.ndim != 3:
            raise ValueError(f"frame must have 3 dimensions, got {frame.ndim = }")
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        return cls(group=group, individual_frame_shape=frame.shape, dtype=frame.dtype)

    def init_group(self):
        if "frames_type_id" in self.group.keys():
            self.load_frames() # TODO: Implement?
            return
        
        self.group["shape"] = self.shape # We technically write shape twice - the first time shape is (0 c h w) so we know the individual frame size if the recording fails without closing properly
        self.group["dtype"] = str(self.dtype)
        self.group["encoder"] = self._codec
        self.group["frames_type_id"] = self._frames_type_id

        self.encoded_frames = self.group.create_dataset(
            name="encoded_frames",
            shape=(self._current_codestream_capacity, ),
            maxshape=(None, ),
            dtype=np.uint8,
            chunks=True,
        )

        self.frame_lengths_bytes = self.group.create_dataset(
            name="frame_lengths_bytes",
            shape=self._current_frame_capacity,
            maxshape=(None, ),
            dtype=np.uint64,
            chunks=True,
        )

        self.frame_start_memory_offsets = self.group.create_dataset(
            name="frame_start_memory_offsets",
            shape=self._current_frame_capacity,
            maxshape=(None, ),
            dtype=np.uint64,
            chunks=True,
        )

        self.timestamps = self.group.create_dataset(
            name="timestamps",
            shape=self._current_frame_capacity,
            maxshape=(None, ),
            dtype=np.uint64,
            chunks=True,
        )

    def add_frames(self, new_frames: np.ndarray, timestamps: np.ndarray):

        # Ensure valid inputs
        if new_frames.shape[1:] != self.individual_frame_shape:
            raise ValueError(f"Expected shape {self.individual_frame_shape}, got {new_frames.shape = }")
        
        if new_frames.dtype != self.dtype:
            raise TypeError(f"Expected dtype {self.dtype}, got {new_frames.dtype = }")
        
        if timestamps.dtype != np.uint64:
            raise TypeError(f"Expected timestamps to have dtype {self.dtype}, got {timestamps.dtype = }")
        
        if len(new_frames) != len(timestamps):
            raise ValueError(f"new_frames and timestamps must have equal length, got {len(new_frames) = }, {len(timestamps) = }")

        # Encode data to be stored for batch
        encoded: Iterable[bytes] = self._encode_func(new_frames)
        flat_array = np.frombuffer(b"".join(encoded), dtype=np.uint8)
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
        frame_count_after_write = self._current_frame_capacity + current_batch_frame_length
        if (self._current_frame_capacity < frame_count_after_write):
            self._current_frame_capacity = frame_count_after_write * 2
            self.frame_lengths_bytes.resize((self._current_frame_capacity, ))
            self.frame_start_memory_offsets.resize((self._current_frame_capacity, ))
            self.timestamps.resize((self._current_frame_capacity, ))

        # Write data
        self.frame_lengths_bytes[frame_indices] = lengths
        self.frame_start_memory_offsets[frame_indices] = offsets + self._current_codestream_position
        self.timestamps[frame_indices] = timestamps
        self.encoded_frames[byte_indices] = flat_array

        # Update position
        self.frame_count += current_batch_frame_length
        self._current_codestream_position += current_batch_byte_length



        # new_frames_idx = slice(self._current_codestream_position, self._current_codestream_position + )
        # self.encoded_frames[]
    def close(self) -> None:
        self.group["shape"][...] = self.shape

        # Resize arrays to final size
        self.frame_lengths_bytes.resize((self.frame_count, ))
        self.frame_start_memory_offsets.resize((self.frame_count, ))
        self.timestamps.resize((self.frame_count, ))
        self.encoded_frames.resize((self._current_codestream_position, ))

        # Close file if needed
        if isinstance(self.group, storage_backend.File):
            self.group.close()

    def load_frames(self):
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.frame_count, *self.individual_frame_shape)
    
    
class FileStore:
    def __init__(self, path: Path|str, metadata: object, individual_frame_shape: Tuple[int, ...], dtype: np.dtype, io_mode: Literal["r", "w"] = "w"):
        self.path: Path = Path(path)
        
        # TODO: Make metadata optional in case we're loading instead of writing
        if not is_dataclass(metadata):
            raise TypeError(f"metadata must be a dataclass, got {type(metadata) = }")
        self.metadata: object = metadata
        self.io_mode: Literal["r", "w"] = io_mode

        if self.io_mode == "w" and (not self.path.parent.exists()):
            self.path.parent.mkdir(parents=True)
        self._file = storage_backend.File(self.path, mode=self.io_mode)
        frames_group = self._file.create_group("frames")
        self.frames: Frames = Frames(frames_group, individual_frame_shape=individual_frame_shape, dtype=dtype)
        self._write_metadata()

    @classmethod
    def like_frame(cls, path: Path|str, metadata: object, frame: np.ndarray|Any, io_mode: Literal["r", "w"] = "r") -> "FileStore":
        if frame.ndim != 3:
            raise ValueError(f"frame must have 3 dimensions, got {frame.ndim = }")
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        return cls(path=path, metadata=metadata, individual_frame_shape=frame.shape, dtype=frame.dtype, io_mode=io_mode)

    def _write_metadata(self):
        value = json.dumps(asdict(self.metadata))
        key = "metadata"
        if key in self._file.keys():
            self._file[key][...] = value # Update existing
        else:
            self._file[key] = value # Create

    # @classmethod
    # def create(self, path: Path, metadata: object) -> "FileStore":

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.frames.shape
    
    @property
    def dtype(self) -> np.dtype:
        return self.frames.dtype

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        self.frames.close()
        self._write_metadata()
        self._file.close()

    