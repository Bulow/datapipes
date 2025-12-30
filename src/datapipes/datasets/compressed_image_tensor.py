#%%
from datapipes.datasets.dataset_source import DatasetSource
from pathlib import Path
import torch
import h5py
import einops
from typing import Any, List, Tuple
import numpy as np
from datapipes.save_datapipe.file_format import image_compression, metadata_utils, format_specification
import nvidia.nvimgcodec as nv
#%%
class CompressedImageTensor(DatasetSource):
    def __init__(self):
        self.frames = bytearray()
        self.lengths = []
        self.offsets = []

        self.batch_start_frame_index = 0
        self.batch_start_byte_index = 0

        self._jpeg2k_params = nv.Jpeg2kEncodeParams()
        
        # self._jpeg2k_params.num_resolutions = 4
        # jpeg2k_params.code_block_size = (block_size, block_size)
        self._jpeg2k_params.bitstream_type = nv.Jpeg2kBitstreamType.J2K
        # jpeg2k_params.prog_order = nv.Jpeg2kProgOrder.PCRL
        self._jpeg2k_params.ht = True

        self._individual_frame_shape = None


    def push(self, payload: torch.Tensor):       
        # current_array_size = self.frames.shape[0]
        # Encode batch
        if payload.ndim < 4:
            payload = payload.unsqueeze(0)
        
        if self._individual_frame_shape is None:
            self._individual_frame_shape = payload.shape[-3:]
        
        encoded = image_compression.torch_encode(frames=payload, codec="jpeg2k", params=nv.EncodeParams(
                quality_type = nv.QualityType.LOSSLESS,
                jpeg2k_encode_params=self._jpeg2k_params
            )
        )
        # import rich
        # rich.inspect(encoded[0])

        # Compute indices
        lengths = np.fromiter([len(f) for f in encoded], dtype=np.uint64)
        offsets = np.empty_like(lengths)
        offsets[0] = 0
        np.cumsum(lengths[:-1], out=offsets[1:])

        flat_array = bytes(b"".join([bytes(buf) for buf in encoded]))
        
        current_batch_frame_length = len(payload)
        current_batch_byte_length = len(flat_array)

        self.lengths[self.batch_start_frame_index:self.batch_start_frame_index + current_batch_frame_length] = lengths

        self.offsets[self.batch_start_frame_index:self.batch_start_frame_index + current_batch_frame_length] = offsets + self.batch_start_byte_index

        # size_after_write = self.batch_start_byte_index + current_batch_byte_length
        # if (current_array_size < size_after_write):
        #     current_array_size = size_after_write * 2
        #     self.frames.ds.resize((current_array_size, ))

        self.frames[self.batch_start_byte_index:self.batch_start_byte_index + current_batch_byte_length] = flat_array

        # Update position
        self.batch_start_frame_index += current_batch_frame_length
        self.batch_start_byte_index += current_batch_byte_length

        # self.frames.ds.resize((batch_start_byte_index, ))
        # print(f"frames size: {len(self.frames)}")

    @property
    def shape(self):
        # return (self._length, *self._shape[1:])
        return (len(self), *self._individual_frame_shape)
    
    def __len__(self):
        return self.batch_start_frame_index

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