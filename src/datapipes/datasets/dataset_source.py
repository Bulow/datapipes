#%% 
from typing import Protocol, Tuple, Sequence
import torch
from torch.utils.data import Dataset, DataLoader

# class PytorchDatasetWrapper(Dataset):
#     def __init__(self, dataset: Dataset):
#         self.dataset = dataset

class DatasetSource(Dataset):
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int|slice|Tuple) -> torch.Tensor:
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        ...

    def as_pytorch_dataloader(self, batch_size: int=128, shuffle: bool=True) -> DataLoader:
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle)

class DatasetWithMetadata(Protocol):
    def get_metadata(self) -> dict:
        ...
    
    @property
    def frame_index_in_recording(self) -> Sequence:
        ...
    
    @property
    def timestamps(self) -> Sequence:
        ...
        
    


# class DatasetMetadata(Protocol):
#     per_frame_data: dict[str, list[any]|torch.Tensor]
#     recording_settings: dict[str, any]
#     experiment_protocol: dict[str, any]
#     data_analysis_protocol: dict[str, any]