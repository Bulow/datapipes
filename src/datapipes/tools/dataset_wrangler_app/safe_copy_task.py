import os
import shutil
import logging
from typing import Any
from pathlib import Path
from blake3 import blake3
import json
import base64

from datapipes.gradio_ui.parallel_process_manager import ParallelProcessTask, ProgressEntry, ProgressCallback, TaskStatus

logger = logging.getLogger(__name__)


def copy_file_chunked(task: ParallelProcessTask, progress_callback: ProgressCallback, _: Any=None) -> blake3:
    """
    Docstring for _copy_file_chunked
    
    :param self: Description
    :param task: Description
    :type task: ParallelProcessTask
    :return: blake3 hasher of source file
    :rtype: blake3
    """
    logger.debug(f"Starting file copy: {task.src} -> {task.dst}")
    try:
        CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB
        src = task.src
        dst = task.dst
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        progress = ProgressEntry(
            processed_bytes=0,
            total_bytes=task.size,
            status=TaskStatus.RUNNING,
            status_display_label="Copying",
        )
        progress_callback(task, progress)

        copied = 0
        src_hasher = blake3(max_threads=blake3.AUTO)
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            while True:
                buf = fsrc.read(CHUNK_SIZE)
                if not buf:
                    break
                fdst.write(buf)
                copied += len(buf)
                src_hasher.update(buf)

                progress = ProgressEntry(
                    processed_bytes=copied,
                    total_bytes=task.size,
                    status=TaskStatus.RUNNING,
                    status_display_label="Copying",
                )
                progress_callback(task, progress)

        # Copy metadata last (best-effort)
        try:
            shutil.copystat(src, dst, follow_symlinks=True)
        except Exception as e:
            logger.warning(f"Failed to copy metadata for {dst}: {e}")
        
        logger.debug(f"File copy completed: {task.src} -> {task.dst}, {copied} bytes")
        return src_hasher
    except Exception as e:
        logger.error(f"Error copying file {task.src} -> {task.dst}: {e}", exc_info=True)
        raise

def _write_hashes_json(src: Path, src_hash: bytes, dst: Path, dst_hash: bytes):
        try:
            manifest_dict = {
                "src_path": src.as_uri(),
                "src_hash": base64.b64encode(src_hash).decode("utf-8"),
                "dst_path": dst.as_uri(),
                "dst_hash": base64.b64encode(dst_hash).decode("utf-8"),
                "algorithm": "blake3",
                "digest_length": 64,
                "digest_encoding": "base64"
            }
            
            with open(dst.with_name(dst.name + ".hash.json"), "w") as f:
                json.dump(manifest_dict, f)
            logger.debug(f"Hash manifest written for: {dst}")
        except Exception as e:
            logger.error(f"Error writing hash manifest for {dst}: {e}", exc_info=True)

def verify_hashes(task: ParallelProcessTask, progress_callback: ProgressCallback, source_hasher: blake3) -> None:
    logger.debug(f"Starting hash verification for: {task.dst}")
    try:
        CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB
        progress = ProgressEntry(
            processed_bytes=0,
            total_bytes=task.size,
            status=TaskStatus.VERIFYING,
            status_display_label="Verifying hash",
        )
        progress_callback(task, progress)
        
        # Verify hash
        # print(f"[{dst}]: Verifying hash")
        dst_hasher = blake3(max_threads=blake3.AUTO)
        hashed = 0
        src = Path(task.src)
        dst = Path(task.dst)

        with open(dst, "rb") as fdst:
            print(f"[{dst}]: Verifying hash - {fdst = }")
            fdst.seek(0)
            while True:

                buf = fdst.read(CHUNK_SIZE)
                if not buf:
                    break

                dst_hasher.update(buf)
                hashed += len(buf)

                progress = ProgressEntry(
                    processed_bytes=hashed,
                    total_bytes=task.size,
                    status=TaskStatus.VERIFYING,
                    status_display_label="Verifying hash",
                )
                progress_callback(task, progress)

        n=64
        src_hash = source_hasher.digest(n)
        dst_hash = dst_hasher.digest(n)

        if src_hash == dst_hash:
            # Hashes match - success
            print(f"{src_hash = }, {dst_hash = }")
            
            _write_hashes_json(src=Path(src), src_hash=src_hash, dst=Path(dst), dst_hash=dst_hash)

            progress = ProgressEntry(
                processed_bytes=hashed,
                total_bytes=task.size,
                status=TaskStatus.DONE,
                status_display_label="Done and verified",
            )
            progress_callback(task, progress)
            logger.info(f"Hash verification successful for: {task.dst}")
        else:
            # Error
            # self._progress[src] = (copied, total, "error")
            print(f"[{dst = }] Hashes do not match")
            logger.error(f"Hash verification failed for {task.dst}: source and destination hashes don't match")
            raise RuntimeError("Hashes do not match")
    except Exception as e:
        logger.error(f"Error during hash verification for {task.dst}: {e}", exc_info=True)
        raise
