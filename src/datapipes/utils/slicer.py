#%%
from collections.abc import Sequence
import torch
from typing import Optional, List, Tuple
import numpy as np
from numbers import Integral

from ndindex import ndindex as _ndindex
import numpy as np
class _Slicer:
    def __init__(self, shape: Tuple[int]=None):
        self.shape: Tuple[int] = shape

    def __getitem__(self, index: Tuple):
        return self.normalize(index, self.shape)
    
    def from_shape(self, shape: Tuple[int]) -> "_Slicer":
        return _Slicer(shape=shape)
    
    def normalize(self, index, shape: Optional[Tuple[int]]=None):
        
        if shape is None:
            normalized_index = _ndindex(index).raw
        else:
            if isinstance(shape, torch.Tensor):
                shape = shape.numpy()
                
            shape = [s.item() if isinstance(s, np.ndarray) else s for s in shape]
            normalized_index = _ndindex(index).expand(shape).raw
        
        slice_only_index = tuple((slice(i, i + 1) if isinstance(i, int) else i for i in normalized_index))
        return slice_only_index

Slicer = _Slicer()