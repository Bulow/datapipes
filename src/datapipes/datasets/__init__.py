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

from datapipes.datasets.modifiers.cached_dataset import CachedDataset
from datapipes.datasets.modifiers.compressed_cached_dataset import CompressedCachedDataset

from datapipes.sic import sic
from tqdm import tqdm

import fnmatch
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal, Optional


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

def _get_dataset_class_for_extension_pattern(path: Path|str) -> type:
    """
    Return the reader from the *longest matching* glob pattern.
    Longest-first ensures multi-suffix patterns like '*.j2k.h5' beat '*.h5'.
    """
    p = Path(path)
    s = p.as_posix()
    for pattern in sorted(_dataset_extensions, key=len, reverse=True):
        if fnmatch.fnmatch(s, pattern):
            return _dataset_extensions[pattern]
        
    raise ValueError(f"Unknown format. Got \"{p.name}\". Expected a path matching one of [{str.join(", ", _dataset_extensions.keys())}]")

cached_datasets: Dict[Path, DatasetSource] = {}

def clear_dataset_reuse_cache():
    for path, dataset in cached_datasets.items():
        if hasattr(dataset, "close") and callable(dataset.close):
            dataset.close()
    cached_datasets.clear()

# def reuse_if_cached(path: Path) -> DatasetSource:
#     if path in cached_datasets.keys():
#         return cached_datasets[path]
#     else:
#         return 

def load_dataset(
    path: Path|str,
    *args,
    cache_strategy: Literal["cache_raw", "cache_compressed", "cache_raw_reuse", "cache_compressed_reuse", "no_caching"]="no_caching",
    **kwargs,
) -> DatasetSource:
    
    path = Path(path)
    ds_class = _get_dataset_class_for_extension_pattern(path=path)

    ds = ds_class(path, *args, **kwargs)

    match cache_strategy:
        case "cache_raw":
            return CachedDataset(underlying_dataset=ds)
        case "cache_compressed":
            return CompressedCachedDataset(underlying_compressed_ds=ds)
        case "cache_raw_reuse":
            if path in cached_datasets.keys() and cached_datasets[path]._error is None:
                print(f"Reusing cached dataset: {path.name}")
                return cached_datasets[path]
            else:
                ds = CachedDataset(underlying_dataset=ds)
                cached_datasets[path] = ds
                return ds
        case "cache_compressed_reuse":
            if path in cached_datasets.keys() and cached_datasets[path]._error is None:
                print(f"Reusing cached dataset: {path.name}")
                return cached_datasets[path]
            else:
                ds = CompressedCachedDataset(underlying_compressed_ds=ds)
                cached_datasets[path] = ds
                return ds
        case "no_caching":
            return ds
        case _:
            raise ValueError(f"Unrecognized cache strategy: {cache_strategy = }")
    

    


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



