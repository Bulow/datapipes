#%%
from collections.abc import Sequence
from typing import Optional

# from __future__ import ValidWindow

class AutoUnpaddingWindow(Sequence):
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
    def __init__(self, data: "ValidWindow", padding: int):
        if not isinstance(data, ValidWindow):
            raise ValueError("UnPadded can only be applied to ValidWindow instances")
            
        self.data_window: ValidWindow = data
        self.padding: int = padding

        if self.data_window.start - padding < 0 or self.data_window.stop + padding > len(self.data_window.data):
            raise ValueError(f"UnPadded would result in out-of-bounds access, got padding={padding} for {data}")
        
    def __len__(self):
        return len(self.data_window)
    
    def __getitem__(self, index: int|slice):
        if isinstance(index, slice):
            start: int = index.start if index.start is not None else 0
            stop: int = index.stop if index.stop is not None else len(self)
        elif isinstance(index, int):
            if index < 0:
                index += len(self)
            start: int = index
            stop: int = index + 1
        else:
            raise TypeError(f"Index must be int or slice, got {type(index)}")
        
        return self.data_window.data[start + self.data_window.start - self.padding:stop + self.data_window.start + self.padding]
    
    def __repr__(self):
        return f"{type(self).__qualname__}(padding={self.padding}, window={self.data_window})"

class ValidWindow(Sequence):
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

    def __init__(self, data: Sequence, start: Optional[int], stop: Optional[int]):
        """
        Initialize a ValidWindow over a sequence.

        Args:
            data (Sequence): The sequence to window.
            start (Optional[int]): The starting index (inclusive).
            stop (Optional[int]): The ending index (exclusive).

        Raises:
            IndexError: If the window range is invalid.
        """
        self.data: Sequence = data
        self.start: int = start if start is not None else 0
        self.stop: int = stop if stop is not None else len(data)

        # Extract underlying data and adjust start/stop if data is itself a ValidWindow
        if isinstance(self.data, ValidWindow):
            self.start += self.data.start
            self.stop += self.data.start
            self.data = self.data.data

        # Ensure start and stop are within bounds
        if self.start < 0 or self.stop > len(data) or self.start > self.stop:
            raise IndexError("Invalid window range")

    
    def get_unpadding_window(self, padding: int):
        return AutoUnpaddingWindow(self, padding)

    @staticmethod
    def pad(data: Sequence, pad: int):
        """
        Create a ValidWindow with 'pad' elements excluded from both ends.

        Args:
            data (Sequence): The sequence to window.
            pad (int): Number of elements to exclude from both ends.

        Returns:
            ValidWindow: The padded window.
        """
        if pad <= 0:
            return ValidWindow(data, 0, len(data))
        return ValidWindow(data, pad, len(data) - pad)

    def __repr__(self):
        """
        Return a string representation of the ValidWindow.

        Returns:
            str: The string representation.
        """
        if hasattr(self.data, "shape") and len(self.data.shape) > 1:
            data_repr = f"data_shape={self.data.shape}"
        else:
            if len(self.data) > 10:
                data_repr = f"data=[{', '.join(map(str, self.data[:5]))}, ..., {', '.join(map(str, self.data[-3:]))}]"
            else:
                data_repr = repr(self.data)

        return f"{type(self).__qualname__}(start={self.start}, stop={self.stop}, length={len(self)} (of {len(self.data)}), {data_repr})"

    def __len__(self):
        """
        Return the length of the window.

        Returns:
            int: The number of elements in the window.
        """
        return self.stop - self.start

    def __getitem__(self, index: int|slice):
        """
        Get an item or slice from the window, with bounds checking.

        Args:
            index (int | slice): The index or slice to access.

        Returns:
            Any: The item or slice from the underlying sequence.

        Raises:
            IndexError: If the index or slice is out of range.
            TypeError: If the index is not int or slice.
        """
        if isinstance(index, int):
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError(f"Index out of range, got {index} for length {len(self)}")
            return self.data[self.start + index]
        elif isinstance(index, slice):
            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else len(self)
            if start < 0:
                start += len(self)
            if stop < 0:
                stop += len(self)
            if start < 0 or stop > len(self) or start > stop:
                raise IndexError(f"Slice out of range, got {index} for length {len(self)}")
            return self.data[self.start + start:self.start + stop]
        else:
            raise TypeError("Index must be int or slice")

    def __iter__(self):
        """
        Iterate over the elements in the window.

        Yields:
            Any: The next element in the window.
        """
        for i in range(len(self)):
            yield self[i]

if __name__ == "__main__":
    print("_" * 72)
    d = list(range(100))
    # vw = ValidWindow(d, 2, 8)
    # print(vw)          # ValidWindow(start=2, stop=8, length=6, data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(vw[0])       # 2
    # print(vw[1:4])     # [3, 4, 5]
    # print(len(vw))     # 6
    # print(list(vw))    # [2, 3, 4, 5, 6, 7]
    # for item in vw:
    #     print(item)    # 2 3 4 5 6 7 (each on a new line)

    vw = ValidWindow.pad(d, 2)
    # print(vw)          # ValidWindow(start=2, stop=8, length=6, data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(vw[0])       # 2
    # print(vw[1:4])     # [3, 4, 5]
    # print(len(vw))     # 6
    # print(list(vw))    # [2, 3, 4, 5, 6, 7]
    # for item in vw:
    #     print(item)    # 2 3 4 5 6 7 (each on a new line)

    vw = ValidWindow.pad(vw, 2)
    print(vw)          # ValidWindow(start=2, stop=8, length=6, data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(vw[0])       # 2
    print(vw[1:4])     # [3, 4, 5]
    print(len(vw))     # 6
    # print(list(vw))    # [2, 3, 4, 5, 6, 7]
    print()
    for item in vw:
        print(item, end=", ")    # 2 3 4 5 6 7 (each on a new line)
    print()
    up = vw.get_unpadding_window(2)
    print(vw[1:4])
    print(up[1:4])
    print(up)