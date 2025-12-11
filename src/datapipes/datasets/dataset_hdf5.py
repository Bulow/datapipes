
import hdf5plugin #Loads compression filters. Do not remove even though it appears unused
import h5py

from collections import namedtuple
from datapipes.datasets.dataset_source import DatasetSource
from pathlib import Path
# from image_folder_dataset import load_image_folder_to_tensor, get_file_names, load_image_folder_to_tensor_batched
from typing import Tuple
import numpy as np
import torch
from tqdm import tqdm
import math

# def get_metadata_from_fname(fname: Path) -> Tuple[str, str, str, str, str]:
#     camera_model, camera_serial_number, timestamp_index = fname.stem.split("__")
#     date, time, index = timestamp_index.split("_")
#     return int(f"{date}_{time}"), int(index)

# Metadata = namedtuple('Metadata', ['date_time', 'index'])

# def get_metadata_from_filenames(filenames: list[Path]) -> Metadata:
#     metadata = [get_metadata_from_fname(fname) for fname in filenames]
#     date_time, index = zip(*metadata)
#     return Metadata(np.array(date_time), np.array(index))

def get_hdf5_paths_in_folder(folder: str): #TODO: add "recursively" arg
    folder = Path(folder)
    dataset_paths = {d.name:d for d in folder.glob("*.hdf5")}
    return dataset_paths

class LSCI_HDF5_FORMAT:
    frames = "frames"
    mean_frame = "mean_frame"

    metadata = "metadata"
    timestamps = "timestamps"
    frame_index_in_recording = "frame_index_in_recording"

    metadata__timestamps = f"{metadata}/{timestamps}"
    metadata__frame_index_in_recording = f"{metadata}/{frame_index_in_recording}"


class DatasetHDF5(DatasetSource):
    def __init__(self, path: Path|str, *, force_no_mean=False, max_frames=None):
        if isinstance(path, str):
             path = Path(path)
        self.path = path
        self.file = h5py.File(path, "r")
        self.frames, self.timestamps, self.frame_index_in_recording, self.mean_frame = self.load_hdf5_dataset(path)
        self.length = len(self.frames)
        if max_frames is not None:
            self.length = min(self.length, max_frames)
        if force_no_mean:
            self.mean = None
    
    def load_hdf5_dataset(self, path: Path|str):
        if isinstance(path, str):
            path = Path(path)

        frames = self.file[LSCI_HDF5_FORMAT.frames]

        try:
            mean_frame = torch.from_numpy(self.file[LSCI_HDF5_FORMAT.mean_frame][0])
        except KeyError:
            mean_frame = None

        try:
            timestamps = self.file[LSCI_HDF5_FORMAT.metadata__timestamps]
        except KeyError:
            timestamps = None

        try:
            frame_index_in_recording = self.file[LSCI_HDF5_FORMAT.metadata__frame_index_in_recording]
        except KeyError:
            frame_index_in_recording = None
        
        
        return frames, timestamps, frame_index_in_recording, mean_frame
            
    @property
    def shape(self):
        return self.length, *self.frames.shape[1:]

    def __del__(self):
        self.file.close()
        
    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def __getitem__(self, index) -> torch.Tensor:
        def validate_index(index):
            if isinstance(index, int):
                return index <= self.length
            elif isinstance(index, slice):
                return index.stop <= self.length
            elif isinstance(index, tuple):
                return validate_index(index[0])
        if not validate_index(index):
            raise IndexError(f"Index out of bounds: index={index}, length={self.length}")
        frames = torch.from_numpy(self.frames[index])
        if self.mean_frame is not None:
            frames = frames.to(torch.float32) / self.mean_frame
        return frames
    
    def get_range_metadata(self, index):
        return self.timestamps[index], self.frame_index_in_recording[index]

