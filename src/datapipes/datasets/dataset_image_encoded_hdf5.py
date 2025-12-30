#%%
from datapipes.datasets.dataset_source import DatasetSource
from pathlib import Path
import torch
import h5py
import einops
from typing import Any, List, Tuple
from pathlib import Path
from datapipes.save_datapipe.file_format import image_compression, metadata_utils, format_specification

#%%
class DatasetCompressedImageStreamHdf5(DatasetSource):
    def __init__(self, path: Path|str, *, max_frames=None):
        if isinstance(path, str):
             path = Path(path)

        self._path = path
        self.file = h5py.File(path, "r")

        self.file_structure: format_specification.LsciEncodedFramesH5 = metadata_utils.deserialize_hdf5(self.file, format_specification.LsciEncodedFramesH5)

        self.frames = self.file_structure.frames.encoded_frames.ds
        self.lengths = torch.from_numpy(self.file_structure.frames.frame_lengths_bytes.ds[:]).to(torch.int64)
        self.offsets = torch.from_numpy(self.file_structure.frames.frame_start_memory_offsets.ds[:]).to(torch.int64)

        self._shape = self.file_structure.frames.frame_parameters.shape
        self._length = self._shape[0]
        
        self._timestamps = torch.from_numpy(self.file_structure.metadata.timestamps[:])

        if max_frames is not None:
            self._length = min(self._length, max_frames)
    
    def get_user_metadata(self) -> format_specification.UserMetadata:
        return self.file_structure.metadata
    
    def get_frames_metadata(self) -> format_specification.FrameParameters:
        return self.file_structure.frames.frame_parameters
    
    @property
    def timestamps(self) -> torch.LongTensor:
        return self._timestamps
    
    @property
    def path(self) -> Path:
        return self._path
    
    @property
    def shape(self):
        return (self._length, *self._shape[1:])
    
    def close(self):
        self.file.close()

    def __del__(self):
        self.close()
        
    def __len__(self):
        return self._length

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def __getitem__(self, index) -> torch.Tensor:
        def validate_index(index):
            if isinstance(index, int):
                return index <= self._length
            elif isinstance(index, slice):
                return index.stop <= self._length
            elif isinstance(index, tuple):
                return validate_index(index[0])
            
        if not validate_index(index):
            raise IndexError(f"Index out of bounds: index={index}, length={len(self)}")
        
        return self.decode_range(index)
    
    
    def decode_range(self, index: int|slice) -> torch.Tensor:
        frame_index = index[0] if isinstance(index, tuple) else index
        if isinstance(frame_index, slice):
            start = 0 if frame_index.start is None else frame_index.start
            stop = self.frames.shape[0] if frame_index.stop is None else frame_index.stop
        elif isinstance(frame_index, int):
            start = frame_index
            stop = frame_index + 1
        else:
            raise Exception(f"Only int, slice, or tuple are accepted as indices. Got {type(index)}")
        
        lengths = self.lengths[start:stop]
        offsets = self.offsets[start:stop]

        # Read entire encoded batch in a single call and use views to split it
        memory_start_index = offsets[0]
        memory_stop_index = offsets[-1] + lengths[-1]
        raw_stream = self.frames[memory_start_index:memory_stop_index]
        relative_offsets = offsets - memory_start_index

        # Split
        frame_stream_views = []
        for ln, offs in zip(lengths, relative_offsets):
            frame_stream_views.append(raw_stream[offs:offs + ln])

        # TODO: Support block-based ROI-only decoding
        return image_compression.torch_decode(frame_stream_views, index if isinstance(index, tuple) else None)
    
    def get_range_metadata(self, index):
        return self.timestamps[index], self.frame_index_in_recording[index]