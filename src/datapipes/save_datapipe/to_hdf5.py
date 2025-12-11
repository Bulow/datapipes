from datapipes.datapipe import DataPipe
from pathlib import Path
import numpy as np
from datapipes.io_pipeline import Pipeline
from datapipes.ops import Ops

import h5py
import hdf5plugin

import torch

from datapipes.datasets.dataset_hdf5 import LSCI_HDF5_FORMAT

def datapipe_to_hdf5(data: DataPipe, out_path: str|Path, batch_size: int=256, dtype=np.uint8, compression=None, save_mean=False, title_prefix=""):    
    data = data | Ops.numpy
    if isinstance(out_path, str):
        out_path = Path(out_path)
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    # mean = torch.empty(1)
    # if save_mean:
    #     mean = torch.zeros_like(data[0]).to("cuda", torch.float64)
    with h5py.File(out_path, "w") as f:
        frames = f.create_dataset(LSCI_HDF5_FORMAT.frames, shape=(len(data), *data[0].shape), dtype=dtype, compression=compression)
        next_index = 0
        for batch in data.batches_with_progressbar(batch_size=batch_size, title=f"{title_prefix}save_hdf5"):
            length = batch.shape[0]
            frames[next_index:next_index + length] = batch
            next_index += length

        #     if save_mean:
        #         mean += batch.to("cuda").sum(0)
        # if save_mean:
        #     # sic(mean, data, len(data), data.shape)
        #     mean = mean / len(data)
        #     f.create_dataset(LSCI_HDF5_FORMAT.mean_frame, data=mean.cpu().numpy())
        
        # # Important that these come after handling all frames, so datasets that use an array-of-structs memory format can cache metadata while reading frames before they are accessed.
        # timestamps = f.create_dataset(LSCI_HDF5_FORMAT.metadata__timestamps, data=data.timestamps)
        # frame_index_in_recording = f.create_dataset(LSCI_HDF5_FORMAT.metadata__frame_index_in_recording, data=data.frame_index_in_recording)

    print(f"Saved output from DataPipe as {out_path}")
