#%%
import numpy as np
from pathlib import Path

from datapipes.datapipe import DataPipe
from datapipes.utils.deep_hasher import DeepHasher
import h5py

import nvidia.nvimgcodec as nv
from tqdm import tqdm
from typing import Callable, Iterable, Iterator

#%%

from datapipes.save_datapipe.file_format import format_specification, metadata_utils

from datapipes.save_datapipe.file_format.image_compression import torch_encode

def populate_metadata(dp: DataPipe):
    n, c, h, w = dp.shape
    metadata: format_specification.LsciEncodedFramesH5 = format_specification.LsciEncodedFramesH5(
        format_id="au.cfin.lsci.j2k.h5",
        format_version="0.0.0.1",
        frames=format_specification.ImageEncodedFrameStream(
            encoded_frames=format_specification.Placeholder(),
            frame_lengths_bytes=format_specification.Placeholder(),
            frame_start_memory_offsets=format_specification.Placeholder(),
            frame_parameters=format_specification.FrameParameters(
                bit_depth=8,
                channels=c,
                compressed=True,
                compression_parameters=format_specification.CompressionParameters(
                    codec="jpeg2k",
                    quality_type="lossless",
                    quality_value=0,
                    kwargs={"ht": True, "bitstream_type": "j2k"} # TODO: Feels like a leaky abstraction...
                ),
                frames_format_version="0.0.0.1",
                frame_count=n,
                frame_width=w,
                frame_height=h,
                shape=dp.shape,
            ),
        ),
        metadata=format_specification.UserMetadata(
            timestamps=format_specification.Placeholder() # TODO: Make a window for querying a datapipe by its timestamps
        )
    )
    return metadata

def init_hdf5_structure(dp: DataPipe, f: h5py.File|h5py.Group) -> format_specification.LsciEncodedFramesH5:
    metadata = populate_metadata(dp)

    placeholder_dict: format_specification.LsciEncodedFramesH5
    placeholder_dict, manuals = metadata_utils.init_hdf5_structure(metadata, f)

    encoded_frames_placeholder = manuals[placeholder_dict.frames.encoded_frames]
    encoded_frames_placeholder.group.create_dataset(
        name=encoded_frames_placeholder.name,
        shape=(10**9, ),
        maxshape=(None, ),
        dtype=np.uint8,
        chunks=True,
    )

    frame_lengths_bytes_placeholder = manuals[placeholder_dict.frames.frame_lengths_bytes]
    frame_lengths_bytes_placeholder.group.create_dataset(
        name=frame_lengths_bytes_placeholder.name,
        shape=len(dp),
        dtype=np.uint64,
    )

    frame_start_memory_offsets_placeholder = manuals[placeholder_dict.frames.frame_start_memory_offsets]
    frame_start_memory_offsets_placeholder.group.create_dataset(
        name=frame_start_memory_offsets_placeholder.name,
        shape=len(dp),
        dtype=np.uint64,
    )

    timestamps_placeholder = manuals[placeholder_dict.metadata.timestamps]
    timestamps_placeholder.group.create_dataset(
        name=timestamps_placeholder.name,
        shape=len(dp),
        dtype=np.uint64,
    )

    live_view: format_specification.LsciEncodedFramesH5 = metadata_utils.deserialize_hdf5(f, format_specification.LsciEncodedFramesH5)

    return live_view

def datapipe_to_lossless_j2k_h5(dp: DataPipe, out_path: str|Path, batch_size: int=512, progress_bar: Callable[[Iterable, int, str], Iterator] = tqdm) -> DeepHasher:
    # Prepare path
    if isinstance(out_path, str):
        out_path = Path(out_path)
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # frames_buffer_upper_bound = int(np.prod(dp.shape))

    source_hasher = DeepHasher.from_datapipe(dp)

    with h5py.File(out_path, "w") as f:
       
        live_view = init_hdf5_structure(dp, f)

        # rich.inspect(live_view)

        encoded_frames: h5py.Dataset = live_view.frames.encoded_frames
        frame_lengths_bytes: h5py.Dataset = live_view.frames.frame_lengths_bytes
        frame_start_memory_offsets: h5py.Dataset = live_view.frames.frame_start_memory_offsets


        # live_view.frames.encoded_frames[0:10] = np.array(range(10))

        # print(encoded_frames[0:10])

        # inspect_hdf5.visualize_structure(f)
        # return

        jpeg2k_params = nv.Jpeg2kEncodeParams()
        # jpeg2k_params.num_resolutions = 4
        # jpeg2k_params.code_block_size = (block_size, block_size)
        jpeg2k_params.bitstream_type = nv.Jpeg2kBitstreamType.J2K
        # jpeg2k_params.prog_order = nv.Jpeg2kProgOrder.PCRL
        jpeg2k_params.ht = True

        batch_start_frame_index = 0
        batch_start_byte_index = 0
        current_array_size = encoded_frames.shape[0]
        # for batch in dp.batches_with_progressbar(batch_size=batch_size, title=f"Converting file {dp.path.name}", progress_bar=progress_bar):

        def encoded_frames_it() -> Iterator:
            for batch in PrefetchIterator(dp.batches(batch_size=batch_size)):
                encoded = torch_encode(batch, codec="jpeg2k", params=nv.EncodeParams(
                        quality_type = nv.QualityType.LOSSLESS,
                        jpeg2k_encode_params=jpeg2k_params
                    )
                )
                yield batch, encoded
                

        for batch, encoded in progress_bar(PrefetchIterator(encoded_frames_it()), len(dp), f"Converting file {dp.path.name}"):
            print(f"{batch = }, {type(batch) = }")
            source_hasher.ingest_frames(batch)

            # Encode batch
            # encoded = torch_encode(batch, codec="jpeg2k", params=nv.EncodeParams(
            #         quality_type = nv.QualityType.LOSSLESS,
            #         jpeg2k_encode_params=jpeg2k_params
            #     )
            # )

            # encoded = torch_encode(batch, codec="jpeg2k", params=nv.EncodeParams(
            #         quality_type = nv.QualityType.QUANTIZATION_STEP,
            #         quality_value = 2,
            #         jpeg2k_encode_params=jpeg2k_params
            #     )
            # )

            # Compute indices
            lengths = np.fromiter([len(f) for f in encoded], dtype=np.uint64)
            offsets = np.empty_like(lengths)
            offsets[0] = 0
            np.cumsum(lengths[:-1], out=offsets[1:])

            flat_array = np.frombuffer(b"".join(encoded), dtype=np.uint8)

            current_batch_frame_length = len(batch)
            current_batch_byte_length = len(flat_array)

            frame_lengths_bytes[batch_start_frame_index:batch_start_frame_index + current_batch_frame_length] = lengths

            frame_start_memory_offsets[batch_start_frame_index:batch_start_frame_index + current_batch_frame_length] = offsets + batch_start_byte_index

            size_after_write = batch_start_byte_index + current_batch_byte_length
            if (current_array_size < size_after_write):
                current_array_size = size_after_write * 2
                encoded_frames.ds.resize((current_array_size, ))

            encoded_frames[batch_start_byte_index:batch_start_byte_index + current_batch_byte_length] = flat_array

            # Update position
            batch_start_frame_index += current_batch_frame_length
            batch_start_byte_index += current_batch_byte_length

        encoded_frames.ds.resize((batch_start_byte_index, ))

        source_hasher.ingest_metadata(dp.timestamps)

        timestamps: h5py.Dataset = live_view.metadata.timestamps
        timestamps[:] = dp.timestamps

        return source_hasher


def verify_lossless_j2k_h5(path: Path, progress_bar: Callable[[Iterable, int, str], Iterator] = tqdm) -> DeepHasher:
    from datapipes.datasets.dataset_image_encoded_hdf5 import DatasetCompressedImageStreamHdf5
    written_ds = DataPipe(DatasetCompressedImageStreamHdf5(path=path))
    written_hasher = DeepHasher.from_datapipe(written_ds)
    written_hasher.ingest_datapipe(written_ds, progress_bar=progress_bar, pb_description=f"Verifying hash of {path.name}")
    written_hasher.ingest_metadata(written_ds.timestamps)
    return written_hasher


import threading
import queue
from typing import Generic, Iterable, Iterator, TypeVar, Union

T = TypeVar("T")


class PrefetchIterator(Generic[T]):
    """
    Wrap an iterator/iterable and prefetch items into a queue on a background thread.

    - Bounded queue provides backpressure (producer blocks when full).
    - Exceptions in the producer are re-raised in the consumer thread.
    - Supports clean shutdown via close() or context manager.
    """

    _SENTINEL = object()

    def __init__(
        self,
        source: Union[Iterable[T], Iterator[T]],
        *,
        max_prefetch: int = 3,
        daemon: bool = True,
    ) -> None:
        if max_prefetch < 1:
            raise ValueError("max_prefetch must be >= 1")

        self._it: Iterator[T] = iter(source)
        self._q: "queue.Queue[object]" = queue.Queue(maxsize=max_prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=daemon)

        self._started = False
        self._closed = False

    def _ensure_started(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def _put_blocking(self, item: object) -> None:
        """
        Put with periodic stop-checking so close() doesn't hang.
        """
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.1)
                return
            except queue.Full:
                continue
        # If we're stopping, don't block further.

    def _worker(self) -> None:
        try:
            for item in self._it:
                if self._stop.is_set():
                    break
                self._put_blocking(item)
        except BaseException as e:
            # Send exception to consumer
            self._put_blocking(e)
        finally:
            # Signal completion
            self._put_blocking(self._SENTINEL)

    def __iter__(self) -> "PrefetchIterator[T]":
        self._ensure_started()
        return self

    def __next__(self) -> T:
        self._ensure_started()
        if self._closed:
            raise StopIteration

        while True:
            obj = self._q.get()  # blocks until available
            if obj is self._SENTINEL:
                self._closed = True
                raise StopIteration
            if isinstance(obj, BaseException):
                self.close()
                raise obj
            return obj  # type: ignore[return-value]

    def close(self) -> None:
        """
        Stop producer and try to release consumer/producer promptly.
        Safe to call multiple times.
        """
        if self._closed and self._stop.is_set():
            return

        self._stop.set()

        # Try to nudge consumer(s) and allow worker to exit quickly.
        try:
            self._q.put_nowait(self._SENTINEL)
        except queue.Full:
            pass

        self._closed = True

    def __enter__(self) -> "PrefetchIterator[T]":
        self._ensure_started()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# --- Example usage ---
if __name__ == "__main__":
    import time

    def slow_numbers(n: int) -> Iterator[int]:
        for i in range(n):
            time.sleep(0.05)  # simulate slow production
            yield i

    for x in PrefetchIterator(slow_numbers(10), max_prefetch=3):
        # consumer can do work; production overlaps
        time.sleep(0.03)
        print(x)
