import torch
from torch.utils import dlpack as torch_dlpack

import rich
from datapipes.datasets import DatasetSource
import torch
from datapipes.datasets.utils.mkv_frames import MkvFrames
from typing import List
from pathlib import Path
import numpy as np

from dataclasses import dataclass, asdict
import json

@dataclass(kw_only=True)
class DatapipeMetadata:
    shape: tuple[int, int, int, int]
    dtype: str
    timestamps: list[int]

    def __post_init__(self):
        self.shape = tuple(self.shape)
        self.dtype = str(self.dtype)

        if isinstance(self.timestamps, torch.Tensor):
            self.timestamps = self.timestamps.cpu().numpy()
        self.timestamps = self.timestamps[:] # Coerce array materialization if it is a lazy container-like dataset
        if isinstance(self.timestamps, np.ndarray):
            self.timestamps = self.timestamps.tolist()

    def write(self, video_file_out_path: Path):
        
        with video_file_out_path.with_suffix(".json").open("w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def load(cls, video_file_out_path: Path) -> "DatapipeMetadata":
        with video_file_out_path.with_suffix(".json").open("r") as f:
            data = json.load(f)
        return cls(**data)

class DatasetMkv(DatasetSource):
    def __init__(self, path: Path|str):
        self._path = Path(path)
        self._metadata = DatapipeMetadata.load(self._path)
        self._frames = MkvFrames(self._path)#, shape=self._metadata.shape)

    @property
    def shape(self):
        return self._metadata.shape
    
    @property
    def timestamps(self) -> torch.LongTensor:
        return self._metadata.timestamps
    
    @property
    def path(self) -> Path:
        return self._path
    
    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, idx: int|slice|tuple) -> torch.Tensor:
        return self._frames[idx]

