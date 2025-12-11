#%%
from collections.abc import Sequence
import torch
from typing import Optional, List, Tuple
import numpy as np

from ndindex import ndindex as _ndindex
import numpy as np

from datapipes.utils import Slicer

#%%

class NdValidWindow():
    """
    A windowed view over a sequence, supporting slicing and indexing with bounds checking.

    ValidWindow allows you to create a view into a subsequence of a given sequence, defined by start and stop indices.
    It supports indexing, slicing, iteration, and can be nested (i.e., a ValidWindow of a ValidWindow will be flattened).
    The window ensures that all accesses are within valid bounds, raising IndexError for out-of-range operations.

    Attributes:
        data (Sequence): The underlying sequence being windowed.
        start (int): The starting index of the window (inclusive).
        stop (int): The ending index of the window (exclusive).
    """
    def __init__(self, data: torch.Tensor, bounds: Tuple[slice]):

        # Populate bounds slices
        self.data = data
        self._shape = data.shape
        # print(f"self.data.shape: {self.data.shape}")
        self.bounds = Slicer.normalize(bounds, shape=self.data.shape)
        self._shape = [b.stop - b.start for b in self.bounds]
    
    @staticmethod
    def get_padded(data: torch.Tensor, padding: Tuple[slice]):
        bounds = tuple([slice(p, -p) if p > 0 else slice(None) for p in padding])
        return NdValidWindow(data=data, bounds=bounds)
    
    @property
    def shape(self):
        return self._shape
    
    def __len__(self):
        return self._shape[0]
    
    def get_source_slices(self, index: int|slice|Tuple[slice]):
        # print(f"get_source_slices-index: {index}")
        index = Slicer.normalize(index=index, shape=self.shape)
        # print(f"\tget_source_slices-index: {index}")
        source_slices = []
        for i, (dim_idx, dim_bounds) in enumerate(zip(index, self.bounds)):
            size_idx = dim_idx.stop - dim_idx.start
            size_bounds = dim_bounds.stop - dim_bounds.start
            if size_idx > size_bounds:
                raise IndexError(f"Index out of bounds in dim={i}. index: {index}, bounds: {self.bounds}")
            source_slices.append(
                slice(
                    dim_idx.start + dim_bounds.start, 
                    dim_idx.stop + dim_bounds.start, 
                    dim_idx.step
                )
            )
        return tuple(source_slices)

    def __getitem__(self, index: int|slice|Tuple[slice]):
        source_index = self.get_source_slices(index)
        return self.data[source_index]
    

class NdAutoUnpaddingWindow(Sequence):
    """
    A view that automatically unpads indices when accessing elements of a ValidWindow.

    This is especially useful for fetching batches of data for use in convolutional operations with valid padding, where the outer elements are not used.

    Attributes:
        data_window (ValidWindow): The underlying ValidWindow instance.
        padding (int): The number of elements to unpad from each end. This corresponds to window_size // 2 for convolutional operations.
    Methods:
        __len__(): Returns the length of the unpadded window.
        __getitem__(index): Accesses an element or slice, automatically adjusting for padding.
        __repr__(): Returns a string representation of the UnPadded view.
    Raises:
        ValueError: If the provided data is not a ValidWindow or if unpadding would result in out-of-bounds access.

    """
    def __init__(self, data: NdValidWindow, padding: Tuple[int]):
        if not isinstance(data, NdValidWindow):
            raise ValueError("Unpadded can only be applied to ValidWindow instances")
        
        if len(data.shape) != len(padding):
            raise ValueError(f"len(padding)={len(padding)} must match the number of dimensions in data ({len(data.shape)})")
            
        self.data_window: NdValidWindow = data
        self.padding: Tuple[int] = padding

        self._unpadded_shape = [s + (2 * p) for s, p in zip(self.shape, self.padding)]

        # assert len(padding) == len(data.bounds), f"bounds[{len(self.data_window.bounds)}] and padding[{len(self.padding)}] have different number of dimensions."

        dim_invalid_checks = []
        for pad, bound, dim_shape in zip(self.padding, self.data_window.bounds, self.data_window.data.shape):
            check = bound.start - pad < 0 or bound.stop + pad > dim_shape
            dim_invalid_checks.append(check)
        if any(dim_invalid_checks):
            raise ValueError(f"Unpadded would result in out-of-bounds access, got padding={padding} for {data.bounds} with invalid dims: {dim_invalid_checks}")

        
    def __len__(self):
        return len(self.data_window)
    
    @property
    def shape(self):
        return self.data_window.shape
    
    @property
    def unpadded_shape(self):
        return self._unpadded_shape
    
    def __getitem__(self, index: int|slice|tuple[slice, ...]):
        source_slices_before_unpad = self.data_window.get_source_slices(index)

        unpadded_slices = tuple([slice(s.start - dim_padding, s.stop + dim_padding, s.step) for s, dim_padding in zip(source_slices_before_unpad, self.padding)])

        return self.data_window.data[unpadded_slices]
    
    def __repr__(self):
        return f"{type(self).__qualname__}(padding={self.padding}, window={self.data_window})"
    


        
# ndw = NdValidWindow(torch.ones([512, 1, 512, 512]), Slicer[10:, :, 0:64, 64:128])

# # print(ndw.prepare_idx(Slicer[5, 0, 10:, :]))
# # print(ndw.get_source_slices(Slicer[5, 0, 10:, :]))

# print(ndw[5:10].shape)

# ndw = NdValidWindow(torch.ones([512, 1, 512, 512]), Slicer[-200:-10, :, 10:64, 64:128])
# aunpad = NdAutoUnpaddingWindow(data=ndw, padding=(8, 0, 2, 2))

# # print(aunpad)

# t = torch.tensor(range(32))

# print(t)

# ndw = NdValidWindow(t, Slicer[4:-4])
# print(ndw[:])

# aunpad = NdAutoUnpaddingWindow(data=ndw, padding=(2, ))
# print(aunpad[:])


# ndw = NdValidWindow(torch.ones(size=(512, 1, 128, 128)), Slicer[10:-10, :, 4:-4, 4:-4])
# print(ndw[:].shape)

# aunpad = NdAutoUnpaddingWindow(data=ndw, padding=(6, 0, 3, 3))
# print(aunpad[:].shape)

#%%
