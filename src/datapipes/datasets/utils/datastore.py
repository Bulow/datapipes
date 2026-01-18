from __future__ import annotations

import numpy as np
import torch
import h5py
from typing import Union, Any, Tuple, Dict, Optional, Literal, Protocol
from abc import ABC, abstractmethod

import numpy as np
from typing import Any

from pathlib import Path



class ArrayLike:
    """Thin wrapper around np.ndarray."""
    
    def __init__(self, data: Any, dtype: np.dtype, mode: Literal["read", "write"]="read"):
        try:
            self.data = np.asarray(data, dtype=dtype)
        except Exception as e:
            raise TypeError(
                f"data must be array-like and convertible to a NumPy array, "
                f"got {type(data).__name__}"
            ) from e

        # Reject scalars (0-D arrays)
        if self.data.ndim == 0:
            raise TypeError(
                f"data must be array-like (not a scalar), got {type(data).__name__}"
            )
        
        self._mode: Literal["read", "write"] = mode

    @property
    def shape(self) -> tuple:
        """Get array shape."""
        return self.data.shape
    
    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data.dtype)
    
    def __len__(self) -> int:
        """Get length of first dimension."""
        return len(self.data)
    
    def __getitem__(self, key):
        """Get item(s) from array."""
        return self.data[key]
    
    def __setitem__(self, key, value):
        """Set item(s) in array."""
        match self._mode:
            case "read":
                raise RuntimeError(f"Array is read-only. Init with mode=\"write\" to open in write mode")
            case "write":
                self.data[key] = value
            case _:
                raise ValueError(f"Unrecognized mode: {self._mode}")


class DataStore(ABC):
    """ABC for data storage."""

    def __init__(self, filepath: Path, mode: Literal["read", "write"] = "read"):
        """
        Initialize the datastore.
        """
        self.filepath = filepath
        self.mode = mode
        self.file = h5py.File(filepath, mode)
    
    @abstractmethod
    def create_group(self, name: str) -> DataStore:
        """Create a new group."""
        raise NotImplementedError
    
    @abstractmethod
    def get_group(self, name: str) -> DataStore:
        """Get an existing group."""
        raise NotImplementedError
    
    @abstractmethod
    def create_array(self, name: str, shape: Tuple[int, ...], dtype: np.dtype) -> ArrayLike:
        """Create a new array-like dataset, and return ArrayLike proxy."""
        raise NotImplementedError
    
    @abstractmethod
    def get_array_dataset(self, name: str, mode: Literal["read", "write"]) -> ArrayLike:
        """Get ArrayLike proxy from a dataset."""
        raise NotImplementedError
    
    @abstractmethod
    def resize_array_dataset(self, name: str, new_shape: Tuple[int, ...]) -> ArrayLike:
        """Resize array dataset and return an updated ArrayLike proxy."""
        raise NotImplementedError

    @abstractmethod
    def get_value(self, name: str) -> Any:
        """Read data from a dataset."""
        raise NotImplementedError

    @abstractmethod
    def set_value(self, name: str, value: Any):
        """Write primitive data to dataset."""
        raise NotImplementedError
    
    @abstractmethod
    def close(self):
        """Close the file."""
        raise NotImplementedError


class H5DataStore(DataStore):
    """Minimal wrapper around h5py for reading and writing HDF5 files."""
    
    def __init__(self, filepath: str, mode: str = 'r'):
        """
        Initialize the datastore.
        
        Args:
            filepath: Path to HDF5 file
            mode: File mode ('r', 'w', 'a')
        """
        self.filepath = filepath
        self.mode = mode
        self.file = h5py.File(filepath, mode)
    
    def create_group(self, name: str) -> DataStore:
        """Create a new group."""
        return self.file.create_group(name)
    
    def get_group(self, name: str) -> DataStore:
        """Get an existing group."""
        return self.file[name]

    def get_value(self, name: str) -> Any:
        return self.file[name]

    def set_value(self, name: str, value: Any):
        self.file[name] = value
    
    def create_array(self, name: str, data: ArrayLike):
        """Write data to a dataset."""
        self.file.create_dataset(name, data=data)
    
    def get_array_dataset(self, name: str, mode: Literal["read", "write"]) -> ArrayLike:
        """Read data from a dataset."""
        return self.file[name]
    
    def resize_array_dataset(self, name: str, new_shape: Tuple[int, ...]) -> ArrayLike:
        """Resize an existing dataset."""
        self.file[name].resize(new_shape)
    
    def close(self):
        """Close the file."""
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


        