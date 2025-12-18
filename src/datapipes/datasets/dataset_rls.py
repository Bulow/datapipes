#%%

import h5py
from collections import namedtuple
from datapipes.datasets.dataset_source import DatasetSource, DatasetWithMetadata
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from tqdm import tqdm
from datapipes.sic import sic
import struct
import mmap
import warnings
from collections.abc import Sequence
import einops


class DatasetRLS(DatasetSource, DatasetWithMetadata):
    def __init__(self, path: Path|str, rls_dtype=torch.uint8, max_frames=None, switch_wh_metadata_read_order: bool = False):
        self.rls_file_reader = RLS_FileReader(path, rls_dtype=rls_dtype, max_frames=max_frames, switch_wh_metadata_read_order=switch_wh_metadata_read_order)

    def __len__(self):
        return self.rls_file_reader.total_frame_count
    
    @property
    def shape(self):
        return self.rls_file_reader.total_frame_count, *self.rls_file_reader.get_frame_shape()
    
    def __getitem__(self, index) -> torch.Tensor:
        return self.rls_file_reader[index]
    
    @property
    def timestamps(self):
        timestamps = self.rls_file_reader.timestamps
        return timestamps.timestamps
    
    @property
    def frame_index_in_recording(self):
        return range(0, self.rls_file_reader.total_frame_count)
    
class CachedTimestampList(Sequence):
    def __init__(self, frame_count):
        self.timestamps = np.zeros((frame_count), dtype=np.uint64)
        self.timestamps_read = np.zeros((frame_count), dtype=np.uint8)

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, index):
        if np.any(self.timestamps_read[index] == 0):
            warnings.warn("All frames must be read before accessing timestamps, since RLS uses a discontiguous array-of-structs memory format")
        return self.timestamps[index]
    
    def __setitem__(self, index, value):
        self.timestamps[index] = value
        self.timestamps_read[index] = 1

    

class RLS_FileReader:
    def __init__(self, path: Path|str, rls_dtype=torch.uint8, max_frames=None, switch_wh_metadata_read_order: bool=False):
        if isinstance(path, str):
             path = Path(path)
        self.path = path
        self.rls_dtype = rls_dtype
        self.bytes_per_pixel = torch.tensor([], dtype=rls_dtype).element_size()

        # self.read_file_metadata()
        with open(self.path, 'rb') as f:
            self.size_first_dim = struct.unpack('Q', f.read(8))[0]
            self.size_second_dim = struct.unpack('Q', f.read(8))[0]
            
            self.total_frame_count = struct.unpack('Q', f.read(8))[0]
            self.sample_rate = struct.unpack('Q', f.read(8))[0]

            self.version = struct.unpack('Q', f.read(8))[0]

        if switch_wh_metadata_read_order:
            self.size_first_dim, self.size_second_dim = (self.size_second_dim, self.size_first_dim)

        self.total_frame_count = self.total_frame_count if max_frames is None else min(self.total_frame_count, max_frames)

        self.header_offset = 30 * 1024
        self.frame_size = self.size_first_dim * self.size_second_dim * self.bytes_per_pixel
        self.timestamp_size = 8

        self.timestamps: CachedTimestampList = CachedTimestampList(self.total_frame_count)

        self.file = open(self.path, 'rb')
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        

    # def read_file_metadata(self):
    #     '''
    #     Read the metadata from the `.rls` file and store it in member variables.
    #     '''
    #     with open(self.path, 'rb') as f:
    #         self.width = struct.unpack('Q', f.read(8))[0]
    #         self.height = struct.unpack('Q', f.read(8))[0]
    #         self.total_frame_count = struct.unpack('Q', f.read(8))[0]
    #         self.sample_rate = struct.unpack('Q', f.read(8))[0]

    #         self.version = struct.unpack('Q', f.read(8))[0]

    def get_location_of_first_byte_in_frame(self, frame_index):
        '''
        Get the location of the first byte of a frame in the `.rls` file.
        
        Assumes single channel image data

        Frame layout:
            `[timestamp (self.timestamp_size bytes)][frame data (width * height * bytes_per_pixel bytes)]`

        Args:
            `frame_index (int)`: The index of the frame to get the location of.
        '''
        if frame_index < 0 or frame_index >= self.total_frame_count:
            print("frame_index ({}) must be less than the total number of frames ({})".format(frame_index, self.total_frame_count))
            return -1        

        total_offset_from_file_start = self.header_offset + (self.timestamp_size + self.frame_size) * frame_index
        return total_offset_from_file_start
    
    def get_total_size_per_frame(self):
        '''
        Get the total size of a frame in bytes
        '''
        return self.timestamp_size + self.frame_size

    def get_frame_shape(self):
        '''
        Get the shape of a frame.
        '''
        return (1, self.size_first_dim, self.size_second_dim) # Assumes single channel image data
    
    def load_frames(self, from_frame_index: int, frame_count: int) -> torch.Tensor:
        '''
        Load a single frame from the memory-mapped `.rls` file.

        Args:
            `frame_index (int)`: The index of the frame to load.
        '''

        # Adjust frame_count if it exceeds the total number of frames
        frame_count = min(frame_count, self.total_frame_count - from_frame_index)

        # if from_frame_index < 0 or from_frame_index >= self.total_frame_count - frame_count:
        #     raise IndexError("from_frame_index ({}) must be between 0 and {}".format(from_frame_index, self.total_frame_count - frame_count - 1))


        # Get memory location and structure
        from_memory_index = self.get_location_of_first_byte_in_frame(from_frame_index)
        to_memory_index = from_memory_index + (self.get_total_size_per_frame() * frame_count)

        # Map to tensor
        data = self.mm[from_memory_index:to_memory_index]
        
        flat_tensor = torch.frombuffer(data, dtype=self.rls_dtype, count=frame_count * self.get_total_size_per_frame(), requires_grad=False).clone()
        #Get views that extract and remap frames into seperate tensors. Actual underlying memory is still flat_tensor, so torch.reshape doesn't work.
        frames = torch.as_strided(flat_tensor, size=(frame_count, 1, self.size_first_dim, self.size_second_dim), 
            stride=(                                                                    # Bytes to skip to get next index along dimension:
                self.timestamp_size + self.size_second_dim * self.size_first_dim * self.bytes_per_pixel,  # Frame+timestamp
                self.size_second_dim * self.size_first_dim * self.bytes_per_pixel,                        # Frame
                self.size_second_dim * self.bytes_per_pixel,                                     # Row
                self.bytes_per_pixel                                                    # Pixel
            ), 
            storage_offset=self.timestamp_size)
        
        # frames = einops.rearrange(frames, "f c w h -> f c h w")
        
        timestamps = torch.as_strided(flat_tensor, size=(frame_count, self.timestamp_size), 
            stride=(
                self.size_second_dim * self.size_first_dim * self.bytes_per_pixel,                        # Frame
                1                                                                       # Byte (self.timestamp_size=8 per timestamp)
            ), 
            storage_offset=0).view(torch.uint64).squeeze(1)                             # Reinterpret those 8 bytes as a uint64

        self.timestamps[from_frame_index:from_frame_index + frame_count] = timestamps
        return frames
       
    def __del__(self):
        self.file.close()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def __getitem__(self, index) -> torch.Tensor:
        # start, end = self.get_range_from_slice(index)
        # print(f"rls[{index}]")
        if isinstance(index, Tuple):
            roi = index[1:]
            index = index[0]
        else:
            roi = None
        if isinstance(index, int):
            start = index
            end = index + 1
        elif isinstance(index, slice):
            start = index.start if index.start is not None else 0
            end = index.stop if index.stop is not None else self.total_frame_count
        else:
            raise TypeError(f"Index must be an int or a slice, got {type(index)}")
        frames = self.load_frames(from_frame_index=start, frame_count=end - start)

        if roi is not None:
            return frames[:, *roi] # TODO: Apply roi before actually loading the data
        else:
            return frames


