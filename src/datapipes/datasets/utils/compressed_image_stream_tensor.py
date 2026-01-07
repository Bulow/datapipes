#%%
import torch
from typing import Optional
from datapipes.save_datapipe.file_format import image_compression
#%%
class CompressedImageStreamTensor:
    def __init__(self, frames, lengths, offsets, individual_frame_shape: Optional[torch.Size]=None):
        self.frames = frames
        self.lengths = lengths
        self.offsets = offsets

        if individual_frame_shape is None:
            individual_frame_shape = self.decode_range(0).shape
            
        self._shape = torch.Size([len(self.lengths), *individual_frame_shape])

    @property
    def shape(self):
        # return (self._length, *self._shape[1:])
        return self._shape
    
    def __len__(self):
        return self._shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def __getitem__(self, index) -> torch.Tensor:
        def validate_index(index):
            if isinstance(index, int):
                return index < len(self)
            elif isinstance(index, slice):
                return index.stop is None or index.stop <= len(self)
            elif isinstance(index, tuple):
                return validate_index(index[0])
            
        if not validate_index(index):
            raise IndexError(f"Index out of bounds: index={index}, length={self._length}")
        
        return self.decode_range(index)
    
    
    def decode_range(self, index: int|slice) -> torch.Tensor:
        frame_index = index[0] if isinstance(index, tuple) else index
        if isinstance(frame_index, slice):
            start = frame_index.start or 0
            stop = frame_index.stop or len(self)
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

    def get_stored_size(self) -> int:
        return len(self.frames) + len(self) * (2 * 8) # lengths(8) and offsets(8)

    def get_logical_size(self) -> int:
        return len(self) * 1 * 512 * 1024