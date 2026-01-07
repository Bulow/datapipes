
import zarr


# from numcodecs.blosc import Blosc2
# import numcodecs.zarr3
from zarr.codecs import numcodecs

# numcodecs.blosc.set_nthreads(24)
# zarr_compressors = numcodecs # https://numcodecs.readthedocs.io/en/stable/zarr3.html

from datapipes.datapipe import DataPipe
from pathlib import Path
import numpy as np
from datapipes.utils.io_pipeline import Pipeline
from datapipes.ops import Ops

def datapipe_to_zarr(dp: DataPipe, out_path: str|Path, n_writers: int=4, n_fetchers: int=4, batch_size: int=256):

    # Prepare path
    if isinstance(out_path, str):
        out_path = Path(out_path)
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    dp = dp | Ops.numpy
    first_frame = dp[0]
    dtype = first_frame.dtype

    # Prepare Zarr file
    if out_path.suffix == ".zip":
        store = zarr.storage.ZipStore(out_path, mode='w')
        
    elif out_path.suffix == ".zarr":
        store = zarr.storage.LocalStore(out_path)
        
    else:
        raise ValueError("Path must end with .zip or .zarr")
    
    root = zarr.group(store=store, overwrite=False)
    frames = root.create_array(
        name="frames", 
        shape=(len(dp), *first_frame.shape), 
        chunks=(batch_size, *first_frame.shape), 
        dtype=dtype,
        # filters=[numcodecs.Delta(), numcodecs.Shuffle()],
        compressors=None,
        # filters=[numcodecs.zarr3.Delta()],
        # compressors=[numcodecs.zarr3.LZ4(level=5)]
    )

    def fetch(index: slice):
        return dp[index]
    
    def write(index: slice, batch: np.ndarray):
        frames[index] = batch

    pipeline = Pipeline(
        fetch_data=fetch,
        write_data=write,
        max_ready_queue=64,
        num_fetch_workers=n_fetchers,
        num_write_workers=n_writers
    )

    pipeline.run(start=0, stop=len(dp), batch_size=batch_size)

    print(root.tree())
    print(frames.info_complete())
    
    if store is not None:
        store.close()

    print(f"Saved output from DataPipe as {out_path}")

