#%%
from pathlib import Path
import torch
from skimage.io import imread
from datapipes.datasets.dataset_source import DatasetSource
from typing import Tuple
from collections import namedtuple
import numpy as np
from datapipes.sic import sic
from typing import Callable

def get_frame_index_in_recording(filename: Path):
    return int(str(filename.stem).split('_')[-1].split('.')[0])

def get_file_names(path: Path|str, extension: str=".tiff", sort_by: Callable=get_frame_index_in_recording):
        if isinstance(path, str):
             path = Path(path)
        filenames = list(path.glob(f"*{extension}"))
        if sort_by is not None:
            filenames = sorted(filenames, key=sort_by)
        return filenames

def sort_by_frame_index_in_recording(filenames: list[Path]):
    return sorted(filenames, key=get_frame_index_in_recording)

def load_images_to_tensor(file_paths: list[Path]) -> torch.Tensor:
        return torch.stack([torch.tensor(imread(path), dtype=torch.uint8) for path in file_paths]).unsqueeze(1)

def load_image_folder_to_tensor(path: Path|str, extension: str=".tiff", max_images=None):
    file_names = get_file_names(path, extension)

    if max_images is not None:
        file_names = file_names[0:min(max_images, len(file_names))]

    return load_images_to_tensor(file_names)

def load_image_folder_to_tensor_batched(path: Path|str, chunk_size = 512, extension: str=".tiff", max_images=None):
    file_names = get_file_names(path, extension)
    if max_images is not None:
        file_names = file_names[0:min(max_images, len(file_names))]

    length = len(file_names)
    for i in range(0, length, chunk_size):
        if i + chunk_size > length:
            yield load_images_to_tensor(file_names[i:])
        else:
            yield load_images_to_tensor(file_names[i:i+chunk_size])

def get_metadata_from_fname(fname: Path) -> Tuple[str, str, str, str, str]:
    camera_model, camera_serial_number, timestamp_index = fname.stem.split("__")
    date, time, index = timestamp_index.split("_")
    return int(f"{date}_{time}"), int(index)

Metadata = namedtuple('Metadata', ['date_time', 'index'])

def get_metadata_from_filenames(filenames: list[Path]) -> Metadata:
    metadata = [get_metadata_from_fname(fname) for fname in filenames]
    date_time, index = zip(*metadata)
    return date_time, index


class DatasetImageFolder(DatasetSource):
    def __init__(self, path, extension=".tiff", limit_to_range: tuple[int, int]=None):
        self.folder = path
        self.extension = extension
        self.filenames = get_file_names(self.folder, self.extension)
        
        if limit_to_range is not None:
             start, stop = limit_to_range
             self.filenames = self.filenames[start:stop]

        self.frame_count = len(self.filenames)
        first_frame = load_image_folder_to_tensor(path, extension, 1)
        # sic(first_frame)
        _, _, self.height, self.width = first_frame.shape

        self.timestamps, self.frame_index_in_recording = get_metadata_from_filenames(self.filenames)
        
        # sic(self.timestamps, self.frame_index_in_recording)
    def __len__(self) -> int:
        return self.frame_count

    def __getitem__(self, index: int) -> torch.Tensor:
        return load_images_to_tensor(self.filenames[index])

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.frame_count, 1, self.height, self.width

    def get_range_metadata(self, index):
        return self.timestamps[index], self.frame_index_in_recording[index]

#%%
if __name__ == "__main__":
    path = R"G:\recordings\despeckler_data\cranial_window_ex0\despeckler_on\02"
    ds = DatasetImageFolder(path)
    from datapipes.datapipe import DataPipe
    dp = DataPipe(ds)
    import img

    img.plot(dp[0])

    #%%
    from datapipes.sinks import save_hdf5

    out_path = R"G:\Emil\test_image_folder_dataset_to_hdf5.hdf5"
    save_hdf5(dp, out_path, batch_size=8192, dtype=np.uint8, save_mean=True)
    # %%
    # sic(dp)
    # sic(ds.frame_index_in_recording[0:len(ds)])
    # sic(dp.frame_index_in_recording)