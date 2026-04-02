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

from datapipes.datasets.utils.tensor_dataset import TensorDataset

from datapipes.datasets.dataset_mkv import DatasetMkv

# from datapipes.save_datapipe.new_file_format.file_store import load_file_store_as_frames_dataset


from datapipes.sic import sic
from tqdm import tqdm

import fnmatch
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal, Optional

import torch
import numpy as np
from pathlib import Path
from datapipes import sinks
from datapipes.utils import benchmarking

Handler = Callable[[Path], DatasetSource]

def save_tensor_npy(t: torch.Tensor|DatasetSource, path: Path, verbose: bool=True):
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    if not isinstance(t, torch.Tensor):
        t = sinks.accumulate(t, idx=slice(None))
    with path.open("wb") as f:
        if verbose:
            print(f"Saving {benchmarking.human_readable_filesize(benchmarking.get_logical_size(t))}. This might take a while...")
        np.save(f, t.cpu().numpy())
        if verbose:
            print(f"Saved {benchmarking.human_readable_filesize(benchmarking.get_logical_size(t))}")

def load_tensor_npy(path: Path):
    from datapipes.datasets.utils.tensor_dataset import TensorDataset
    with path.open("rb") as f:
        t = torch.from_numpy(np.load(f))
    ds = TensorDataset(t)
    ds._path = path
    return ds

def load_fs_temp(path):
    from datapipes.save_datapipe.new_file_format import file_store, frames, codecs
    return file_store.load_file_store_as_frames_dataset(path)

# TODO: Use glob instead
_dataset_extensions = {
    "*.rls": DatasetRLS,
    # "*.j2k.h5fs": load_file_store_as_frames_dataset,
    "*.j2k.h5": DatasetCompressedImageStreamHdf5,
    "*.hdf5": DatasetHDF5,
    "*.h5": DatasetHDF5,
    "*.zarr": DatasetZarr,
    "*.zarr.zip": DatasetZarr,
    "*.mp4": DatasetVideoFile,
    "*.mkv": DatasetMkv,
    "*.npy": load_tensor_npy,
    "*.fs.h5": load_fs_temp,
}

_extensions_from_dataset_class = {ds: ext[1:] for ext, ds in _dataset_extensions.items()}



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
            del dataset
    cached_datasets.clear()

# def reuse_if_cached(path: Path) -> DatasetSource:
#     if path in cached_datasets.keys():
#         return cached_datasets[path]
#     else:
#         return 

def add_dataset_to_reuse_cache(ds: DatasetSource):
    # To conserve RAM, the limit is currently set at 1 reusable dataset. This is typically enough to prevent reload in a notebook setting. The reuse cache is not intended to manage precached datasets in a batch processing setting.
    if len(cached_datasets) > 0:
        # If limit is later increased from 1, we should probably only evict the oldest dataset
        clear_dataset_reuse_cache()
        cached_datasets[ds.path] = ds

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
                add_dataset_to_reuse_cache(ds)
                return ds
        case "cache_compressed_reuse":
            if path in cached_datasets.keys() and cached_datasets[path]._error is None:
                print(f"Reusing cached dataset: {path.name}")
                return cached_datasets[path]
            else:
                ds = CompressedCachedDataset(underlying_compressed_ds=ds)
                add_dataset_to_reuse_cache(ds)
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
    "DatasetMkv",
    "load_dataset",
    "register_file_type",
    "TensorDataset",
]
# %%



