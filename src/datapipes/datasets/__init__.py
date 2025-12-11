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

from pathlib import Path

_dataset_extensions = {
    ".rls": DatasetRLS,
    ".j2k.h5": DatasetCompressedImageStreamHdf5,
    ".hdf5": DatasetHDF5,
    ".h5": DatasetHDF5,
    ".zarr": DatasetZarr,
    ".zarr.zip": DatasetZarr,
    ".mp4": DatasetVideoFile,
}

_extensions_from_dataset_class = {ds: ext for ext, ds in _dataset_extensions.items()}

def load_dataset(path: str|Path) -> DatasetSource:
    if isinstance(path, str):
        path = Path(path)
    if not isinstance(path, Path):
        raise ValueError(f"Invalid path. Expected str or Path, got {type(path)}: {path}")
    extension = path.suffix
    if not extension in _dataset_extensions:
        raise TypeError(f"Unknown format. Got \"{extension}\". Expected one of [{str.join(", ", _dataset_extensions.keys())}]")
    cls = _dataset_extensions[extension]
    return cls(path)

__all__ = ["DatasetSource", "DatasetHDF5", "DatasetCompressedImageStreamHdf5", "DatasetRLS", "DatasetImageFolder", "DatasetZarr", "load_dataset", "DatasetVideoFile"]
# %%
