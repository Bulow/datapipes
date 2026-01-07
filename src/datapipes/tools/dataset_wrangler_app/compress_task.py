import time
from typing import Iterable, Optional
import gradio as gr
from functools import wraps
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import rich
from pathlib import Path
# from cached_dataset import CachedTensor

from datapipes import DataPipe, datasets
from datapipes.datasets.modifiers.cached_dataset import CachedDataset
from datapipes.datasets.modifiers.compressed_cached_dataset import CompressedCachedDataset
from functools import partial


from datapipes import save_datapipe

from datapipes.gradio_ui.parallel_process_manager import ParallelProcessTask, ProgressEntry, ProgressCallback, TaskStatus

from datapipes.gradio_ui.gradio_progress_bar import progress_tqdm_with_eta
import os
from pathlib import Path
from typing import Callable, List, Optional, Any, Tuple, Dict, Iterable, Iterator
from datapipes.utils.deep_hasher import DeepHasher, compare_hashes
import torch

from datapipes.utils.prefetch_iterator import PrefetchIterator

import gradio as gr

import logging

logger = logging.getLogger(__name__)

result_dict = {}

def get_destination_path(task: ParallelProcessTask) -> Path:
    return Path(task.dst) / f"{Path(task.src).stem}.j2k.h5"

# def convert_one(
#         in_path: Path, 
#         out_dir: Path, 
#         verify_deep_hash: bool, 
#         mark_for_deletion: bool, 
#         save_hashes: bool, 
#         generate_report: bool, 
#         progress: gr.Progress=gr.Progress(), 
#         log_output: Callable[[str], None]=print
#     ) -> None:
#     log_output(f"Converting {in_path.name}")
#     ds = datasets.load_dataset(in_path)

#     out_path = out_dir / f"{in_path.stem}.jk2.h5"
#     # progress = gr.Progress()
#     source_hasher, destination_hasher = save_datapipe.datapipe_to_lossless_j2k_h5(dp=DataPipe(ds), out_path=out_path, verify_deep_hash=verify_deep_hash, progress_bar=progress_tqdm_with_eta(progress), logger=log_output)

#     del(ds) # Explicitly delete dataset reader, closing its file and freeing resources
    
#     result_dict.update({
#         "source_hasher": source_hasher,
#         "destination_hasher": destination_hasher,
#     })

#     if verify_deep_hash:

#         source_hash = source_hasher.digest()
#         destination_hash = destination_hasher.digest()
        
#         if save_hashes:
#             hash_dict = {
#                 "source_hash": source_hash.to_json_dict(),
#                 "destination_hash": destination_hash.to_json_dict(),
#             }
#             import jsonic
#             hash_dict_json: str = jsonic.json.dumps(hash_dict)
#             json_path = out_path.with_name(out_path.name + ".deephashes.json")
#             with open(json_path, "w") as f:            
#                 f.write(hash_dict_json)
            
#         if mark_for_deletion:
#             if source_hash == destination_hash:
#                 origin = Path(source_hash.origin_path)
#                 origin.rename(origin.with_name(origin.name + ".safe_to_delete"))
#                 log_output(f"Hashes match. Marked {origin} as safe to delete")
#             else:
#                 raise ValueError(f"Hashes do not match.")

#     log_output(source_hasher.digest())
#     log_output(destination_hasher.digest())
#     log_output(source_hasher == destination_hasher)



def compress_file(task: ParallelProcessTask, progress_callback: ProgressCallback, _: Any=None) -> Any:
    in_path = Path(task.src)
    out_dir = Path(task.dst)

    ds = datasets.load_dataset(in_path)
    # ds = CachedDataset(_ds)
    bytes_per_frame = ds.shape[-2] * ds.shape[-1]

    out_path = get_destination_path(task=task)

    status_display_label: str = "Compressing"
    progress_callback(task, ProgressEntry(
        processed_bytes=0,
        total_bytes=task.size,
        status=TaskStatus.RUNNING,
        status_display_label=status_display_label,
    ))

    dp = DataPipe(ds)
    
    def batch_iterator_passthrough(it: Iterable[torch.Tensor], *args, **kwargs) -> Iterator[torch.Tensor]:
        processed_bytes = 0
        frames_processed = 0
        total_frames = dp.shape[0]

        for batch in it:
            batch_size = batch[0].shape[0] if isinstance(batch, Tuple) else batch.shape[0]
            
            processed_bytes += batch_size * bytes_per_frame
            frames_processed += batch_size

            progress_callback(task, ProgressEntry(
                processed_bytes=processed_bytes,
                total_bytes=task.size,
                status=TaskStatus.RUNNING,
                status_display_label=status_display_label if frames_processed < total_frames else "Wrapping up...",
            ))
            yield batch

    source_hasher = save_datapipe.datapipe_to_lossless_j2k_h5(dp=dp, out_path=out_path, progress_bar=batch_iterator_passthrough)

    del(ds) # Explicitly delete dataset reader, closing its file and freeing resources.
    # del(_ds)
    return source_hasher


def verify_hashes(task: ParallelProcessTask, progress_callback: ProgressCallback, source_hasher: DeepHasher) -> None:
    logger.debug(f"Starting hash verification for: {task.dst}")
    status_display_label: str = "Verifying deep hash"
    progress = ProgressEntry(
        processed_bytes=0,
        total_bytes=task.size,
        status=TaskStatus.VERIFYING,
        status_display_label=status_display_label,
    )
    progress_callback(task, progress)

    out_path = get_destination_path(task=task)
    destination_ds = datasets.load_dataset(out_path)
    # destination_ds = CompressedCachedDataset(_destination_ds)
    
    bytes_per_frame = destination_ds.shape[-2] * destination_ds.shape[-1]


    destination_datapipe = DataPipe(destination_ds)
    destination_hasher = DeepHasher.from_datapipe(datapipe=destination_datapipe)

    processed_bytes = 0
    for batch in PrefetchIterator(destination_datapipe.batches(batch_size=512)):
        destination_hasher.ingest_frames(batch)
        processed_bytes += batch.shape[0] * bytes_per_frame
        progress_callback(task, ProgressEntry(
                processed_bytes=processed_bytes,
                total_bytes=task.size,
                status=TaskStatus.VERIFYING,
                status_display_label=status_display_label,
            ))

    destination_hasher.ingest_metadata(destination_datapipe.timestamps)

    logger.info(f"{source_hasher.digest(64)=}, {destination_hasher.digest(64)=}")

    success = compare_hashes(source_hasher=source_hasher, destination_hasher=destination_hasher, digest_length=64)

    # del(_destination_ds)
    del(destination_ds)

    if success:
        progress = ProgressEntry(
            processed_bytes=task.size,
            total_bytes=get_raw_file_size(out_path),
            status=TaskStatus.DONE,
            status_display_label="Done and verified",
            custom_display_stats=get_stats(task=task),
        )
        progress_callback(task, progress)
    else:
        raise RuntimeError(f"Hashes did not match")


def get_raw_file_size(path: Path|str):
    path = Path(path)
    if path.exists():
        size_bytes = path.stat().st_size
        return size_bytes
    else:
        raise FileNotFoundError(f"{path} does not exist")

def human_readable_filesize(size_bytes: int, dp=2):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024.0:
            return f"{size_bytes:.{dp}f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.{dp}f} PB"

def get_stats(task: ParallelProcessTask) -> List[str]:
    in_path: Path = Path(task.src)
    out_path: Path = get_destination_path(task=task)
    src_size: int = get_raw_file_size(in_path)
    dst_size: int = get_raw_file_size(out_path)

    ratio: str = f"Compression ratio: {(dst_size / src_size) * 100.0:.1f}%"
    abs_diff: str = f"{human_readable_filesize(src_size - dst_size)} saved"

    return [ratio, abs_diff]

