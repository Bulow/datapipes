
#%%
from datapipes.datasets.dataset_source import DatasetSource
from pathlib import Path
import torch
import zarr
#%%
def get_zarr_paths_in_folder(folder: str):
    folder = Path(folder)
    dataset_paths = {d.name:d for d in folder.glob("*.zarr")}
    return dataset_paths

class LSCI_ZARR_FORMAT:
    frames = "frames"
    mean_frame = "mean_frame"

    metadata = "metadata"
    timestamps = "timestamps"
    frame_index_in_recording = "frame_index_in_recording"

    metadata__timestamps = f"{metadata}/{timestamps}"
    metadata__frame_index_in_recording = f"{metadata}/{frame_index_in_recording}"

# def configure_zarr():
#     zarr.convenience.set_compressor("zlib", level=3)
#     zarr.convenience.set_array_compressor("zlib", level=3)
#     zarr.convenience.set_group_compressor("zlib", level=3)
#     zarr.convenience.set_compressor("blosc", cparams={"clevel": 3, "shuffle": 1})


class DatasetZarr(DatasetSource):
    def __init__(self, path: Path|str, *, max_frames=None):
        if isinstance(path, str):
             path = Path(path)
        self._path = path
        self.store = zarr.storage.ZipStore(path, mode='r')
        self.file = zarr.open_group(store=self.store, mode='r')
        self.frames, self.timestamps, self.frame_index_in_recording = self.load_zarr_dataset(path)
        self.length = self.frames.shape[0]
        if max_frames is not None:
            self.length = min(self.length, max_frames)
    
    def load_zarr_dataset(self, path: Path|str):
        if isinstance(path, str):
            path = Path(path)

        frames = self.file[LSCI_ZARR_FORMAT.frames]


        timestamps = self.file[LSCI_ZARR_FORMAT.timestamps]
        frame_index_in_recording = self.file[LSCI_ZARR_FORMAT.frame_index_in_recording]
        return frames, timestamps, frame_index_in_recording
            
    @property
    def shape(self):
        return self.length, *self.frames.shape[1:]
    
    @property
    def path(self) -> Path:
        return self._path
    
    def close(self):
        self.store.close()

    def __del__(self):
        self.close()
        
    def __len__(self):
        return self.frames.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index) -> torch.Tensor:
        def validate_index(index):
            if isinstance(index, int):
                return index <= self.length
            elif isinstance(index, slice):
                return index.stop <= self.length
            elif isinstance(index, tuple):
                return validate_index(index[0])
            
        if not validate_index(index):
            raise IndexError(f"Index out of bounds: index={index}, length={self.length}")
        
        return torch.from_numpy(self.frames[index])
    
    def get_range_metadata(self, index):
        return self.timestamps[index], self.frame_index_in_recording[index]

# #%%
# ds_path = R"\zarr_test\test13.zarr.zip"
# #%%
# ds = ZarrDataset(ds_path)
# print(ds.shape)

# from data_pipe import DataPipe
# dp = DataPipe(ds)
# print(dp.shape)

# import img
# img.plot(dp[0])
# #%%
# store = zarr.storage.ZipStore(ds_path, mode='r')
# f = zarr.open_group(store=store, mode='r')
# print(f.info_complete())
# # print(f.info_complete())
# print(f.tree())

# # f = f[None]
# frames = f[LSCI_ZARR_FORMAT.frames]
# print(frames.shape)
# print(frames.info_complete())

