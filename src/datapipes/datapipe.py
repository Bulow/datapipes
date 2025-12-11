#%%
import typing
from typing import Tuple, Callable, List, Any, Optional
from datapipes.datasets.dataset_source import DatasetSource
# from tqdm import tqdm
from datapipes.utils.forced_tty_mode_tqdm import tty_tqdm
import math
import torch
import numpy as np
from datapipes.sic import sic
from dataclasses import dataclass
from concurrent.futures import Future, as_completed, ThreadPoolExecutor

# from datapipes.dataset_windows import ValidWindow, AutoUnpaddingWindow
# from datapipes.nd_windows import NdValidWindow, NdAutoUnpaddingWindow
from datapipes.auto_indexing_window import AutoIndexingWindow

# TODO: Make DataPipe composable with abstract datapipes

# TODO: Make windows take dim_pads to work along all dimensions


@dataclass
class FutureSlice:
    index: slice
    data: np.ndarray|torch.Tensor


class DataPipe(DatasetSource):

    def __init__(self, dataset: "DatasetSource|DataPipe", segments: Tuple=()):
        self._dataset = dataset
        self.segments = segments

        self.auto_indexing_window: AutoIndexingWindow = AutoIndexingWindow(dataset=self._dataset, fwd_transforms=self.segments)

        self._shape = self.auto_indexing_window.shape

        # shape_deltas, probed_output = self.probe_self()

        # dim_pads = tuple([d // 2 for d in shape_deltas])

        # temporal_pad_size = dim_pads[0]
        
        # if isinstance(probed_output, torch.Tensor):
        #     self.dtype = probed_output.dtype
        #     self.device = probed_output.device
        # elif isinstance(probed_output, np.numpy):
        #     self.dtype = probed_output.dtype
        #     self.device = "cpu"

        
        # self._dataset = dataset
        

        # # self.valid_window: ValidWindow = ValidWindow.pad(self._dataset, pad=temporal_pad_size)
        # # self.auto_unpadding_window: AutoUnpaddingWindow = self.valid_window.get_unpadding_window(padding=temporal_pad_size)
        # self.valid_window: NdValidWindow = NdValidWindow(data = self._dataset, bounds=tuple([slice(pad, -pad) if pad > 0 else slice(None) for pad in dim_pads]))
        # self.auto_unpadding_window: NdAutoUnpaddingWindow = NdAutoUnpaddingWindow(data=self.valid_window, padding=dim_pads)
        
        # # self._shape: torch.Tensor = [int(s - (pad * 2)) for s, pad in zip(self._dataset.shape, dim_pads)]
        # self._shape = self.valid_window.shape

    # def probe_self(self, td=512, sd=128) -> tuple[list[int], Any]:
    #     # TODO: Support rearrange, different number of dimensions etc.

    #     probe_shape = [td, 1, sd, sd]
    #     test_data = torch.zeros(probe_shape)
    #     probe_output = self._execute_pipe(test_data)

    #     shape_deltas = [d_before - d_after for d_before, d_after in zip(probe_shape, probe_output.shape)]

        
    #     return shape_deltas, probe_output

    def __len__(self):
        return self.shape[0]
      
    def __repr__(self):
        return f"{type(self).__qualname__}(shape={self.shape}, segments={self.segments}, pad_size={self.pad_size})"
    
    @property
    def shape(self) -> torch.Tensor:
        return self._shape
    
    @property
    def ndim(self):
        return 4
    
    @property
    def pad_size(self):
        return self.auto_unpadding_window.padding
    
    def to(self, *args, **kwargs):
        first_frame = self[0]
        if not isinstance(first_frame, torch.Tensor):
            raise TypeError(f"DataPipe must return a torch.Tensor, got {type(first_frame)}")
        return self | (lambda f: f.to(*args, **kwargs))
    
    def _execute_pipe(self, data: DatasetSource) -> DatasetSource:
        with torch.no_grad():
            for segment in self.segments:
                data = segment(data)
            return data
        
    # def compile(self):
    #     compiled_pipe = torch.compile(self._execute_pipe)
    #     return DataPipe(self._dataset, segments=(compiled_pipe, ), pad_size=self.pad_size)

    def __getitem__(self, index):
        data = self.auto_indexing_window[index]
        data = self._execute_pipe(data)

        if isinstance(index, int):
            return data[0]
        else:
            return data

        # if isinstance(index, int) or isinstance(index, slice) or isinstance(index, tuple):
        #     # Expand index to slice covering source frames in dataset
        #     data = self.auto_unpadding_window[index]
        #     data = self._execute_pipe(data)

        #     return data if isinstance(index, slice) else data[0]
        
        # else:
        #     raise TypeError("Index must be int, slice, or tuple. Got", type(index))
        


    def then(self, func): #, time_window: int=1):
        if not callable(func):
            raise ValueError("func must be callable, got", type(func))

        new_segments = (*self.segments, func)
        return DataPipe(self._dataset, new_segments)

    def __or__(self, func):
        return self.then(func)
    



    # TODO: Move out

    def batches(self, batch_size: int, max_frames=None):
        frame_count = len(self) if max_frames is None else min(len(self), max_frames)
        for i in range(0, frame_count, batch_size):
            stop = min(i + batch_size, frame_count)
            yield self[i:stop]

    def batches_with_progressbar(self, *, batch_size: int, max_frames=None, title="", **kwargs):
        frame_count = len(self) if max_frames is None else min(len(self), max_frames)
        for batch in tty_tqdm(self.batches(batch_size=batch_size, max_frames=max_frames), total=math.ceil(frame_count / batch_size), desc=title, **kwargs):
            yield batch

    def get_item_multithreaded(self, index: slice, output_dtype: typing.Literal["numpy", "torch"]="numpy", then: typing.Callable[[FutureSlice], typing.Any]=None) -> FutureSlice:
        data = self[index]
        if output_dtype == "numpy":
            data = data.cpu().numpy()
        future_slice = FutureSlice(index=index, data=data)
        if then is not None:
            return then(future_slice)
        else:
            return future_slice

    def batches_multithreaded(self, batch_size: int, max_frames=None, n_threads: int=8, then: typing.Callable[[FutureSlice], typing.Any]=None, output_dtype: typing.Literal["numpy", "torch"]="numpy") -> typing.Generator[FutureSlice, None, None]:
        frame_count = len(self) if max_frames is None else min(len(self), max_frames)
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = []
            for i in range(0, frame_count, batch_size):
                if i + batch_size >= frame_count:
                    futures.append(executor.submit(self.get_item_multithreaded, slice(i, frame_count), output_dtype=output_dtype, then=then))
                else:
                    futures.append(executor.submit(self.get_item_multithreaded, slice(i, i + batch_size), output_dtype=output_dtype, then=then))
            with tty_tqdm(total=frame_count, desc="Processing frames in concurrent batches", unit="frames") as pbar:
                for future in as_completed(futures):
                    batch = future.result()

                    if then is not None:
                        batch = then(batch)

                    pbar.update(batch.index.stop - batch.index.start)
                    yield batch

# if __name__ == "__main__":
