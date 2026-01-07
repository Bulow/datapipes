import torch
from torch.utils import dlpack as torch_dlpack

# Monkey-patch torch.Tensor.__dlpack__ to fix positional/kwarg discrepancy bug
def _patched_dlpack(self, stream=None):
    # Works with both old and new call styles
    return torch_dlpack.to_dlpack(self)
torch.Tensor.__dlpack__ = _patched_dlpack

import rich
from datapipes.datasets import DatasetSource
import torch
import PyNvVideoCodec as nvc
from typing import List
from pathlib import Path

class DatasetVideoFile(DatasetSource):
    def __init__(self, path: Path|str):
        self._path = Path(path)

        self.decoder = nvc.SimpleDecoder(self._path.as_posix(), use_device_memory=True, output_color_type=nvc.OutputColorType.NATIVE)

        metadata = self.decoder.get_stream_metadata() 
        self._shape = (metadata.num_frames, 1, metadata.height, metadata.width)

    @property
    def shape(self):
        return self._shape
    
    @property
    def timestamps(self) -> torch.LongTensor:
        # TODO: Implement
        raise NotImplementedError()
    
    @property
    def path(self) -> Path:
        return self._path
    
    def __len__(self):
        return self.shape[0]
    
    def _to_tensor(self, decoded_frames: List[nvc.DecodedFrame]) -> torch.Tensor:
        tensor = torch.stack([torch.from_dlpack(f) for f in decoded_frames]) # Zero copy
        return tensor[:, :int(tensor.shape[-2] // 1.5), :] # NV12 -> (n h w)
    
    def __getitem__(self, idx: int|slice|tuple) -> torch.Tensor:
        remaining_dim_slices = None
        single_idx = False
        if isinstance(idx, tuple):
            remaining_dim_slices = idx[1:]
            idx = idx[0]
        if isinstance(idx, slice):
            idx = slice(idx.start or 0, idx.stop or len(self), idx.step or 1)
        if isinstance(idx, int):
            single_idx = True
            idx = slice(idx, idx + 1, 1)

        assert isinstance(idx, slice), f"idx must be slice at this point, got {type(idx)}"

        if idx.step == 1:
            # Get frames as consecutive batch
            length = idx.stop - idx.start
            self.decoder.seek_to_index(idx.start)
            frames = self.decoder.get_batch_frames(length)
        else:
            # Get frames using slice syntax
            frames = self.decoder[idx]
        
        t = self._to_tensor(frames).contiguous()
        
        if not single_idx:
            t = t.unsqueeze(1) # Decod
        
        return t if remaining_dim_slices is None else t[:, *remaining_dim_slices]

