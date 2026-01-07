from blake3 import blake3
import torch
from datapipes.datapipe import DataPipe
import torch
from pathlib import Path
import numpy as np
from datapipes.sic import sic
from tqdm import tqdm
from typing import Optional, Tuple, Generator, Iterator
import inspect
import base64
from datapipes.sinks import subbatch
from typing import Any, Dict
from dataclasses import dataclass, field
from tqdm import tqdm
from functools import partial
from typing import Literal, Callable, Iterable, Iterator, Any, Optional

import logging
logger = logging.getLogger(__name__)

@dataclass(frozen=True, kw_only=True)
class DeepHash:
    origin_path: Path|str = field(compare=False)
    frames_processed: int
    method: str
    digest_length: int
    frames_digest: bytes
    metadata_digest: bytes

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "origin_path": str(self.origin_path),
            "frames_processed": self.frames_processed,
            "method": self.method,
            "digest_length": self.digest_length,
            "frames_digest": base64.b64encode(self.frames_digest).decode("utf-8"),
            "metadata_digest": base64.b64encode(self.metadata_digest).decode("utf-8"),
        }

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "DeepHash":
        return cls(
            origin_path=Path(data["origin_path"]),
            frames_processed=data["frames_processed"],
            method=data["method"],
            digest_length=data["digest_length"],
            frames_digest=base64.b64decode(data["frames_digest"]),
            metadata_digest=base64.b64decode(data["metadata_digest"]),
        )
    
    

class DeepHasher:
    @staticmethod
    def from_datapipe(datapipe: DataPipe) -> "DeepHasher":
        if hasattr(datapipe._dataset, "path"):
            origin_path = datapipe._dataset._path
        else:
            origin_path = type(datapipe._dataset)
        dh = DeepHasher(shape=datapipe.shape, dtype=datapipe[0].dtype, origin_path=origin_path)
        return dh
    
    @staticmethod
    def digest_datapipe(datapipe: DataPipe, digest_length: int=64, progress_bar: Callable[[Iterable[torch.Tensor], int, str], Iterator[torch.Tensor]]=tqdm) -> DeepHash:
        dh = DeepHasher.from_datapipe(datapipe=datapipe)
        dh.ingest_datapipe(frames=datapipe, progress_bar=progress_bar)
        dh.ingest_metadata(datapipe.timestamps)
        return dh.digest(digest_length=digest_length)

    def __init__(self, shape: torch.Size, dtype: torch.dtype, origin_path: str=""):
        self.origin_path = origin_path

        # Frames
        self.frames_processed = 0
        self.individual_frame_shape = torch.Size(shape[-3:])
        self.frame_hasher = blake3(max_threads=blake3.AUTO)
        frames_base_info_str = f"shape={shape if isinstance(shape, torch.Size) else torch.Size(shape)}, dtype={dtype}"
        # print(frames_base_info_str)
        self.frame_hasher.update(frames_base_info_str.encode(encoding="utf-8"))

        # Metadata
        self.metadata_hasher = blake3()

    def ingest_frames(self, frames: torch.Tensor):
        if frames.ndim != 4:
            raise ValueError(f"frames must be (n b h w), got shape {frames.shape}")
        
        if frames.shape[1:] != self.individual_frame_shape:
            raise ValueError(f"Individual frames must be of shape {self.individual_frame_shape}, got shape {frames.shape}")

        self.frame_hasher.update(frames.contiguous().to(device="cpu", memory_format=torch.contiguous_format).numpy())

        self.frames_processed += frames.shape[0]

    def ingest_datapipe(self, frames: DataPipe, batch_size: int=512, progress_bar: Callable[[Iterable, int, str], Iterator] = tqdm, pb_description=None):
        for batch in subbatch(dp=frames, idx=slice(None), batch_size=batch_size, progress_bar=progress_bar, pb_description=pb_description):
            self.ingest_frames(batch)

    def ingest_metadata(self, *data: str|bytes|torch.Tensor|np.ndarray):
        def prep(data: str|bytes|torch.Tensor) -> bytes:
            match data:
                case str():
                    return data.encode(encoding="utf-8")
                case torch.Tensor():
                    return data.to(device="cpu", memory_format=torch.contiguous_format).numpy().tobytes()
                case np.ndarray():
                    return torch.from_numpy(data).to(device="cpu", memory_format=torch.contiguous_format).numpy().tobytes()
                case bytes():
                    return data
                case _:
                    raise TypeError(f"Unsupported type: {type(data)}")
        data = [prep(d) for d in data]
        self.metadata_hasher.update(b"".join(data))

    def digest(self, digest_length=64):
        frame_copy = self.frame_hasher.copy()
        frame_copy.update(f"frames_processed={self.frames_processed}".encode(encoding="utf-8"))
        return DeepHash(
            origin_path=self.origin_path,
            frames_digest=frame_copy.digest(length=digest_length),
            metadata_digest=self.metadata_hasher.digest(length=digest_length),
            method="blake3",
            digest_length=digest_length,
            frames_processed=self.frames_processed,
        )

def compare_hashes(source_hasher: DeepHasher, destination_hasher: DeepHasher, digest_length: int=64) -> bool:
    if source_hasher.origin_path == destination_hasher.origin_path:
        raise RuntimeError(f"Source and destination describe the same file: {source_hasher.origin_path=} {destination_hasher.origin_path=}")

    source_hash = source_hasher.digest(digest_length)
    destination_hash = destination_hasher.digest(digest_length)

    if not (isinstance(source_hash, DeepHash) and isinstance(destination_hash, DeepHash)):
        raise TypeError(f"{type(source_hash)=}, {type(destination_hash)=}")
    
    hashes_match = source_hash == destination_hash

    print(f"Hashes match: {hashes_match}")

    if not hashes_match:
        print(f"source: {sh}", f"destination: {wh}")
        raise RuntimeError(f"Deep hashes of source and destination datasets do not match.\n\n{source_hash=}, {destination_hash=}")
    
    return hashes_match


# def hash_frames(frames: DataPipe, batch_size=512, digest_length=32):
#     hasher = blake3(max_threads=blake3.AUTO)
#     base_str = f"shape={torch.Size(frames.shape)}, dtype={frames[0].dtype}"
#     print(base_str)
#     hasher.update(base_str.encode(encoding="utf-8"))
#     for batch in subbatch(dp=frames, idx=slice(None), batch_size=batch_size, progressbar=True):
#         hasher.update(batch.cpu().numpy())
#     return hasher.digest(length=digest_length)
