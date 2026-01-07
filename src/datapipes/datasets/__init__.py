#%%
"""
Load datasets from various formats
"""
from datapipes.datasets.dataset_source import DatasetSource
# from .dataset_source import DatasetSource, DatasetWithMetadata

from datapipes.datasets.dataset_hdf5 import DatasetHDF5
from datapipes.datasets.dataset_rls import DatasetRLS
from datapipes.datasets.dataset_image_folder import DatasetImageFolder
from datapipes.datasets.dataset_zarr import DatasetZarr
from datapipes.datasets.dataset_image_encoded_hdf5 import DatasetCompressedImageStreamHdf5
from datapipes.datasets import DatasetCompressedImageStreamHdf5
from datapipes.datasets.dataset_video_file import DatasetVideoFile

from datapipes.sic import sic
from tqdm import tqdm

import fnmatch
from pathlib import Path
from typing import Callable, Dict, Iterable


from pathlib import Path

Handler = Callable[[Path], DatasetSource]

# TODO: Use glob instead
_dataset_extensions = {
    "*.rls": DatasetRLS,
    "*.j2k.h5": DatasetCompressedImageStreamHdf5,
    "*.hdf5": DatasetHDF5,
    "*.h5": DatasetHDF5,
    "*.zarr": DatasetZarr,
    "*.zarr.zip": DatasetZarr,
    "*.mp4": DatasetVideoFile,
}

# _extensions_from_dataset_class = {ds: ext for ext, ds in _dataset_extensions.items()}



def register_file_type(glob_pattern: str, handler: Handler) -> None:
    """Register/overwrite a handler for a glob pattern."""
    _dataset_extensions[glob_pattern] = handler

def load_dataset(
    path: Path|str,
    *args,
    **kwargs,
) -> DatasetSource:
    """
    Return the reader from the *longest matching* glob pattern.
    Longest-first ensures multi-suffix patterns like '*.tar.gz' beat '*.gz'.
    """
    p = Path(path)
    s = p.as_posix()

    for pattern in sorted(_dataset_extensions, key=len, reverse=True):
        if fnmatch.fnmatch(s, pattern):
            return _dataset_extensions[pattern](p, *args, **kwargs)

    raise ValueError(f"Unknown format. Got \"{p.name}\". Expected a path matching one of [{str.join(", ", _dataset_extensions.keys())}]")


__all__ = [
    "DatasetSource", 
    "DatasetHDF5", 
    "DatasetCompressedImageStreamHdf5", 
    "DatasetRLS", 
    "DatasetImageFolder", 
    "DatasetZarr", 
    "DatasetVideoFile",
    "load_dataset",
    "register_file_type", 
]
# %%



