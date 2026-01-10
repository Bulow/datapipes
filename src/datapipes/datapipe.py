#%%
import typing
from typing import Tuple, Callable, List, Any, Optional
from datapipes.datasets.dataset_source import DatasetSource
from tqdm import tqdm
import math
import torch
import numpy as np
from datapipes.sic import sic
from dataclasses import dataclass
from concurrent.futures import Future, as_completed, ThreadPoolExecutor
from pathlib import Path
# from datapipes.dataset_windows import ValidWindow, AutoUnpaddingWindow
# from datapipes.nd_windows import NdValidWindow, NdAutoUnpaddingWindow
from datapipes.auto_indexing_window import AutoIndexingWindow
from typing import Literal, Callable, Iterable, Iterator, Any, Optional
# from datapipes.utils.logging import progress_bar
# TODO: Make DataPipe composable with abstract datapipes


@dataclass
class FutureSlice:
    index: slice
    data: np.ndarray|torch.Tensor

class DataPipe(DatasetSource):

    def __init__(self, dataset: "DatasetSource|DataPipe", segments: Tuple=(), requires_grad: bool=False):
        self._dataset = dataset
        self.segments = segments

        self.auto_indexing_window: AutoIndexingWindow = AutoIndexingWindow(dataset=self._dataset, fwd_transforms=self.segments)

        self._shape = self.auto_indexing_window.shape

        self.requires_grad = requires_grad


    def __len__(self):
        return self.shape[0]
      
    # def __repr__(self):
    #     return f"{type(self).__qualname__}(shape={self.shape}, segments={self.segments}, pad_size={self.pad_size})"
    
    @property
    def timestamps(self) -> torch.LongTensor:
        return self._dataset.timestamps
    
    @property
    def path(self) -> Path:
        return Path(self._dataset.path)
        return Path(self._dataset.path)
    
    @property
    def shape(self) -> torch.Tensor:
        return self._shape
    
    @property
    def ndim(self):
        return 4
    
    # @property
    # def pad_size(self):
    #     return self.auto_unpadding_window.padding
    
    def to(self, *args, **kwargs):
        first_frame = self[0]
        if not isinstance(first_frame, torch.Tensor):
            raise TypeError(f"DataPipe must return a torch.Tensor, got {type(first_frame)}")
        return self | (lambda f: f.to(*args, **kwargs, non_blocking=True))
    
    def _execute_pipe(self, data: DatasetSource) -> DatasetSource:
        if not self.requires_grad:
            with torch.no_grad():
                for segment in self.segments:
                    data = segment(data)
        else:
            for segment in self.segments:
                    data = segment(data)
        return data
        
    def __getitem__(self, index):
        data = self.auto_indexing_window[index]
        data = self._execute_pipe(data)

        if isinstance(index, int):
            return data[0]
        else:
            return data


    def then(self, func: Callable[[torch.Tensor], torch.Tensor]) -> "DataPipe":
        if not callable(func):
            raise ValueError("func must be callable, got", type(func))

        new_segments = (*self.segments, func)
        return DataPipe(self._dataset, new_segments)

    def __or__(self, func: Callable[[torch.Tensor], torch.Tensor]) -> "DataPipe":
        return self.then(func)
    



    # TODO: Move out

    def batches(self, batch_size: int, max_frames=None):
        frame_count = len(self) if max_frames is None else min(len(self), max_frames)
        for i in range(0, frame_count, batch_size):
            stop = min(i + batch_size, frame_count)
            yield self[i:stop]

    def batches_with_progressbar(self, *, batch_size: int, max_frames=None, title="", progress_bar: Callable[[Iterable, Optional[int], Optional[str]], Iterator] = tqdm):
        frame_count = len(self) if max_frames is None else min(len(self), max_frames)
        for batch in progress_bar(self.batches(batch_size=batch_size, max_frames=max_frames), total=math.ceil(frame_count / batch_size), desc=title):
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
            with tqdm(total=frame_count, desc="Processing frames in concurrent batches", unit="frames") as pbar:
                for future in as_completed(futures):
                    batch = future.result()

                    if then is not None:
                        batch = then(batch)

                    pbar.update(batch.index.stop - batch.index.start)
                    yield batch

# if __name__ == "__main__":
