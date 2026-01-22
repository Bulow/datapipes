from typing import Any, Optional, Tuple, Callable, Protocol
import torch
import numpy as np

from datapipes import datasets

from pathlib import Path

class _Wrappable(Protocol):
    @property
    def shape(self):
        raise NotImplementedError
    
    @property
    def path(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError

class WrapperDataset(datasets.DatasetSource):
    """
    Wraps anything with a __getitem__ method and presents itself as a dataset
    """
    def __init__(
        self,
        wrapped: _Wrappable,
        timestamps: Optional[torch.Tensor]=None
    ) -> None:
        self.wrapped = wrapped
        self._dtype = None
        self._timestamps = timestamps

    @property
    def timestamps(self) -> torch.LongTensor:
        if self._timestamps is not None:
            return self._timestamps
        elif hasattr(self.wrapped, "timestamps"):
            return self.wrapped.timestamps
        else:
            raise NotImplementedError
    
    @property
    def path(self) -> Path:
        return Path(self.wrapped.path)

    @property
    def shape(self) -> torch.Size:
        return self.wrapped.shape

    @property
    def dtype(self) -> torch.dtype:
        if self._dtype is None:
            self._dtype = self[0, 0, 0, 0].dtype
        return self._dtype

    def __getitem__(self, idx: int|slice|Tuple) -> torch.Tensor:
        out = self.wrapped[idx]

        # Return as a torch.Tensor
        if isinstance(out, torch.Tensor):
            return out
        elif isinstance(out, np.ndarray):
            return torch.from_numpy(out)
        else:
            raise TypeError(f"Output type not supported: {type(out) = }")

    def __len__(self) -> int:
        return self._shape[0]

    def __repr__(self) -> str:
        return f"WrapperDataset(wrapped=<{type(self.wrapped)}>, shape={tuple(self.shape)}, dtype={self.dtype})"
