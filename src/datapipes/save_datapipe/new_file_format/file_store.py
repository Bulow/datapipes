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
from datapipes.save_datapipe.new_file_format.frames import Frames, CustomEncoder
from datapipes.save_datapipe.new_file_format.codecs import get_available_encoders as _get_available_encoders
from datapipes.datapipe import DataPipe
from datapipes.ops import Ops

def _is_string_dtype(dt):
    # bytes (S), unicode (U), or object that may contain strings
    return dt.kind in ("S", "U") or dt.kind == "O"

def get_val(group: storage_backend.Group, name: str) -> Any:
    data = group.get(name, "{}")
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
            
        except OSError as ex:
            print(ex)
            self.close()

    @classmethod
    def get_available_encoders(cls) -> Iterable[str]:
        return _get_available_encoders()
    
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

        if not is_dataclass(metadata) and not isinstance(metadata, dict) and metadata is not None:
            raise TypeError(f"metadata must be a None, a dataclass, or a dict. Got {type(metadata) = }")
        fs.metadata = metadata
        fs._write_metadata()

        frames_group = fs._file.create_group("frames")
        fs.frames = Frames.create(frames_group, individual_frame_shape=individual_frame_shape, dtype=dtype, codec=codec)
        fs.frames.parent = fs # Keep a reference to prevent GC of object owning file handle
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
        fs.frames.parent = fs # Keep a reference to prevent GC of object owning file handle

        fs.metadata = get_metadata(fs._file, name="metadata") # get_val(fs._file, "metadata")

        return fs

    @classmethod
    def like_frame(cls, path: Path|str, metadata: object, frame: np.ndarray|Any, codec: Literal["j2k", "jxl"] = "j2k") -> "FileStore":
        if frame.ndim != 3:
            raise ValueError(f"frame must have 3 dimensions, got {frame.ndim = }")
        if hasattr(frame, "cpu"):
            frame = frame.cpu()
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        return FileStore.create(path=path, metadata=metadata, individual_frame_shape=frame.shape, dtype=frame.dtype, codec=codec)

    def _write_metadata(self):
        if self.metadata is None:
            return
        if isinstance(self.metadata, dict):
            value = self.metadata
        else:
            value = json.dumps(asdict(self.metadata))
        if len(value) == 0:
            return

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
        print("closing")
        if hasattr(self, "frames"):
            self.frames.close()
        if self.io_mode == "w":
            self._write_metadata()
        self._file.close()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def datapipe_to_file_store(dp: DataPipe, out_path: Path|str, metadata: Optional[dict|object]=None, batch_size: int=256, codec: str|CustomEncoder="j2k", overwrite: bool=False):
    if not isinstance(dp, DataPipe):
        dp = DataPipe(dp)
    out_path = Path(out_path)
    if overwrite and out_path.exists():
        out_path.unlink()
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True)

    fs = FileStore.like_frame(
        path=out_path,
        metadata=metadata,
        frame=dp[0],
        codec=codec,
    )
    for batch in (dp | Ops.numpy).batches_with_progressbar(batch_size=batch_size):
        fs.frames.add_frames(batch)

def load_file_store_as_frames_dataset(path):
    fs = FileStore.open(path)
    return fs.frames.as_dataset()