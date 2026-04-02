#%% 
from typing import Protocol, Tuple, Sequence
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# class PytorchDatasetWrapper(Dataset):
#     def __init__(self, dataset: Dataset):
#         self.dataset = dataset

class TensorDataset(Dataset):

    def __init__(self, source_tensor: torch.Tensor):
        self._source_tensor = source_tensor
        self._path = None

    def __len__(self) -> int:
        return len(self._source_tensor)

    def __getitem__(self, index: int|slice|Tuple) -> torch.Tensor:
        return self._source_tensor[index]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._source_tensor.shape

    @property
    def timestamps(self) -> torch.LongTensor:
        return torch.arange(len(self._source_tensor))
    
    @property
    def path(self) -> Path:
        return self._path or f"memory:/torch.Tensor(shape={str(self._source_tensor.shape)}, dtype={self._source_tensor.dtype}, device={self._source_tensor.device})"