#%% 
from typing import Protocol, Tuple, Sequence
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# class PytorchDatasetWrapper(Dataset):
#     def __init__(self, dataset: Dataset):
#         self.dataset = dataset

class DatasetSource(Dataset):
    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index: int|slice|Tuple) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    @property
    def timestamps(self) -> torch.LongTensor:
        raise NotImplementedError()
    
    @property
    def path(self) -> Path:
        raise NotImplementedError()
    

    def as_pytorch_dataloader(self, batch_size: int=128, shuffle: bool=True) -> DataLoader:
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle)

class DatasetWithMetadata(Protocol):
    def get_metadata(self) -> dict:
        raise NotImplementedError()
    
    @property
    def frame_index_in_recording(self) -> Sequence:
        raise NotImplementedError()
    
    @property
    def timestamps(self) -> Sequence:
        raise NotImplementedError()
        
    


# class DatasetMetadata(Protocol):
#     per_frame_data: dict[str, list[any]|torch.Tensor]
#     recording_settings: dict[str, any]
#     experiment_protocol: dict[str, any]
#     data_analysis_protocol: dict[str, any]