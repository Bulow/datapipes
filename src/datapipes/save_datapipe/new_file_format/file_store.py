import json
import os

# filename.codec.container
# 
# brain.raw.h5
# brain.j2k.h5
# brain.jxl.h5

from dataclasses import asdict, is_dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Protocol, Tuple, Dict

# Base on h5py initially
import h5py as storage_backend

# import torch
import numpy as np
from datapipes.save_datapipe.new_file_format.frames import Frames

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

def get_metadata(group: storage_backend.Group, name: str) -> Dict[str, Any]:
    json_str = get_val(group, name)
    return json.loads(json_str)
    
class FileStore:
    def __init__(self, path: Path|str, io_mode: Literal["r", "w"] = "r"):
        try:
            self.path: Path = Path(path)
            
            self.io_mode: Literal["r", "w"] = io_mode

            if self.io_mode == "w" and (not self.path.parent.exists()):
                self.path.parent.mkdir(parents=True)
            self._file = storage_backend.File(self.path, mode=self.io_mode)
            
            self.frames: Frames
            self.metadata: object
        
        except OSError:
            self.close()

    @classmethod
    def create(
        cls,
        path: Path|str, 
        metadata: object, 
        individual_frame_shape: Tuple[int, ...], 
        dtype: np.dtype, 
        codec: Literal["j2k", "jxl"] = "j2k"
    ) -> "FileStore":
        fs = FileStore(path=path, io_mode="w")

        if not is_dataclass(metadata):
            raise TypeError(f"metadata must be a dataclass, got {type(metadata) = }")
        fs.metadata = metadata
        fs._write_metadata()

        frames_group = fs._file.create_group("frames")
        fs.frames = Frames.create(frames_group, individual_frame_shape=individual_frame_shape, dtype=dtype, codec=codec)
        
        return fs
    

    @classmethod
    def open(
        cls,
        path: Path|str,
        io_mode: Literal["r", "w"] = "r", 
    ) -> "FileStore":
        fs = FileStore(path, io_mode=io_mode)
        
        frames_group = fs._file["frames"]
        fs.frames = Frames.open(frames_group)

        fs.metadata = get_metadata(fs._file, name="metadata") # get_val(fs._file, "metadata")

        return fs

    @classmethod
    def like_frame(cls, path: Path|str, metadata: object, frame: np.ndarray|Any, io_mode: Literal["r", "w"] = "r", codec: Literal["j2k", "jxl"] = "j2k") -> "FileStore":
        if frame.ndim != 3:
            raise ValueError(f"frame must have 3 dimensions, got {frame.ndim = }")
        if hasattr(frame, "cpu"):
            frame = frame.cpu()
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        return FileStore.create(path=path, metadata=metadata, individual_frame_shape=frame.shape, dtype=frame.dtype, codec=codec)

    def _write_metadata(self):
        value = json.dumps(asdict(self.metadata))
        key = "metadata"
        if key in self._file.keys():
            self._file[key][...] = value # Update existing
        else:
            self._file[key] = value # Create

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
        if self.io_mode == "w":
            self._write_metadata()
        self._file.close()

    