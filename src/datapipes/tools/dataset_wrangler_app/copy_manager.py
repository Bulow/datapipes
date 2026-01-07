import os
import time
import queue
import shutil
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from blake3 import blake3
import json
import base64


CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB


@dataclass
class CopyTask:
    src: str
    dst: str
    size: int

# @dataclass
# class ProgressEntry:
#     copied_bytes
#     total_bytes
#     status


ProgressEntry = Tuple[int, int, str]  # copied_bytes, total_bytes, status


class CopyManager:
    """
    Manages a queue of copy tasks and runs them in parallel, with live per-task progress.
    Status values: queued, running, done, error, cancelled
    """

    def __init__(
        self, 
        chunk_size: int = CHUNK_SIZE, 
        ema_alpha: float = 0.20,          # smoothing: higher -> more reactive
        min_eta_rate_bps: float = 0.1
    ):
        self._chunk_size = chunk_size
        self._ema_alpha = float(ema_alpha)
        self._min_eta_rate_bps = float(min_eta_rate_bps)

        self._lock = threading.Lock()
        self._q: "queue.Queue[CopyTask]" = queue.Queue()
        self._executor: ThreadPoolExecutor | None = None
        self._futures: List[Future] = []
        self._cancel_event = threading.Event()

        # {src: (copied_bytes, total_bytes, status)}
        self._progress: Dict[str, ProgressEntry] = {}
        self._queued: List[str] = []
        self._running: set[str] = set()

        # Stats for bitrate/ETA
        # {src: {"last_t": float, "last_copied": int, "ema_bps": float}}
        self._stats: Dict[str, Dict[str, float]] = {}

    def reset(self) -> None:
        self.cancel()
        with self._lock:
            self._q = queue.Queue()
            self._progress.clear()
            self._queued = []
            self._running.clear()
            self._futures.clear()
        self._cancel_event.clear()

    def cancel(self) -> None:
        self._cancel_event.set()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = None

        with self._lock:
            # Drain queue
            try:
                while True:
                    self._q.get_nowait()
            except queue.Empty:
                pass
            self._queued = []

            # Mark active as cancelled
            for src, (copied, total, status) in list(self._progress.items()):
                if status in ("queued", "running"):
                    self._progress[src] = (copied, total, "cancelled")

            self._running.clear()

    def enqueue(self, tasks: List[CopyTask]) -> None:
        with self._lock:
            for t in tasks:
                self._q.put(t)
                self._progress[t.src] = (0, t.size, "queued")
                self._queued.append(t.src)

    def start(self, max_workers: int) -> None:
        if self._executor is not None:
            return
        self._cancel_event.clear()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Each worker thread consumes tasks sequentially.
        for _ in range(max_workers):
            self._futures.append(self._executor.submit(self._consumer_loop))

    def is_done(self) -> bool:
        with self._lock:
            return not any(
                status in ("queued", "running", "verifying") for (_, _, status) in self._progress.values()
            )

    def snapshot(self) -> tuple[list[dict], list[str]]:
        """
        Returns:
          rows: list[dict] with keys: file,status,copied,total,percent,copied_mb,total_mb
          queued_list: list[str] of files still waiting (paths)
        """
        with self._lock:
            rows = []
            for src, (copied, total, status) in self._progress.items():
                percent = 0.0 if total == 0 else ((copied / total) * 100.0)
                ema_bps = float(self._stats.get(src, {}).get("ema_bps", 0.0))

                remaining = max(0, total - copied)
                if not status in ["running", "verifying"]:
                    eta_seconds: Optional[float] = 0.0 if status == "done" else None
                else:
                    if ema_bps >= self._min_eta_rate_bps:
                        eta_seconds = remaining / ema_bps
                    else:
                        eta_seconds = None
                rows.append(
                    {
                        "file": src,
                        "status": status,
                        "copied": copied,
                        "total": total,
                        "percent": max(0.0, min(100.0, percent)),
                        "copied_mb": copied / (1024 * 1024),
                        "total_mb": (total / (1024 * 1024)) if total else 0.0,
                        "ema_mbps": ema_bps / (1024 * 1024),
                        "eta_seconds": remaining, # eta_seconds,
                    }
                )
            queued_list = list(self._queued)

        rows.sort(key=lambda r: r["file"])
        return rows, queued_list

    def _consumer_loop(self) -> None:
        while not self._cancel_event.is_set():
            try:
                task = self._q.get(timeout=0.2)
            except queue.Empty:
                if self._cancel_event.is_set():
                    return
                time.sleep(0.05)
                continue
            
            now = time.monotonic()
            with self._lock:
                if task.src in self._queued:
                    self._queued.remove(task.src)
                self._running.add(task.src)
                copied, total, _ = self._progress.get(task.src, (0, task.size, "queued"))
                self._progress[task.src] = (copied, total, "running")

                # Initialize timing baseline for bitrate
                st = self._stats.get(task.src)
                if st is None:
                    self._stats[task.src] = {"last_t": now, "last_copied": float(copied), "ema_bps": 0.0}
                else:
                    st["last_t"] = now
                    st["last_copied"] = float(copied)

            try:
                self._copy_file_chunked(task)
                # with self._lock:
                #     _, total, _ = self._progress.get(task.src, (task.size, task.size, "running"))
                #     self._progress[task.src] = (total, total, "done")
            except Exception:
                with self._lock:
                    copied, total, _ = self._progress.get(task.src, (0, task.size, "running"))
                    if self._cancel_event.is_set():
                        self._progress[task.src] = (copied, total, "cancelled")
                    else:
                        self._progress[task.src] = (copied, total, "error")
            finally:
                with self._lock:
                    self._running.discard(task.src)

    def _write_hashes_json(self, src: Path, src_hash: bytes, dst: Path, dst_hash: bytes):
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

    def _update_ema_rate_locked(self, src: str, copied: int) -> None:
        """
        Update EMA throughput for src. Call only while holding self._lock.
        """
        now = time.monotonic()
        st = self._stats.get(src)
        if st is None:
            self._stats[src] = {"last_t": now, "last_copied": float(copied), "ema_bps": 0.0}
            return

        last_t = float(st.get("last_t", now))
        last_c = float(st.get("last_copied", float(copied)))
        dt = max(1e-6, now - last_t)
        dc = max(0.0, float(copied) - last_c)

        inst_bps = dc / dt
        ema_bps = float(st.get("ema_bps", 0.0))

        # If EMA not seeded yet, seed with first instantaneous estimate
        if ema_bps <= 0.0:
            ema_bps = inst_bps
        else:
            a = self._ema_alpha
            ema_bps = a * inst_bps + (1.0 - a) * ema_bps

        self._stats[src] = {
            "ema_bps": ema_bps,
            "last_t": now,
            "last_copied": float(copied),
        }


    def _copy_file_chunked(self, task: CopyTask) -> None:
        src = task.src
        dst = task.dst
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        copied = 0
        src_hasher = blake3(max_threads=blake3.AUTO)
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            while True:
                if self._cancel_event.is_set():
                    raise RuntimeError("cancelled")

                buf = fsrc.read(self._chunk_size)
                if not buf:
                    break
                fdst.write(buf)
                copied += len(buf)
                src_hasher.update(buf)

                with self._lock:
                    _, total, status = self._progress.get(src, (0, task.size, "running"))
                    if status == "running":
                        self._progress[src] = (copied, total, "running")
                        self._update_ema_rate_locked(src, copied)
                    

        # Copy metadata last (best-effort)
        try:
            shutil.copystat(src, dst, follow_symlinks=True)
        except Exception:
            pass

        # Verify hash
        # print(f"[{dst}]: Verifying hash")
        dst_hasher = blake3(max_threads=blake3.AUTO)
        hashed = 0
        with open(dst, "rb") as fdst:
            print(f"[{dst}]: Verifying hash - {fdst = }")
            fdst.seek(0)
            while True:
                if self._cancel_event.is_set():
                    # print(f"[{dst}]: Verifying hash - {self._cancel_event.is_set() = }")
                    raise RuntimeError("cancelled")

                buf = fdst.read(self._chunk_size)
                # print(f"[{dst}]: Verifying hash - {hashed = }, {len(buf) = }")
                if not buf:
                    break

                dst_hasher.update(buf)
                hashed += len(buf)
                with self._lock:
                    _, total, status = self._progress.get(src, (0, task.size, "running"))
                    if status in ["running", "verifying"]:
                        self._progress[src] = (hashed, total, "verifying")
                        self._update_ema_rate_locked(src, hashed)

        n=64
        src_hash = src_hasher.digest(n)
        dst_hash = dst_hasher.digest(n)

        if src_hash == dst_hash:
            # Hashes match - success
            print(f"{src_hash = }, {dst_hash = }")
            
            self._write_hashes_json(src=Path(src), src_hash=src_hash, dst=Path(dst), dst_hash=dst_hash)

            with self._lock:
                _, total, status = self._progress.get(src, (0, task.size, "running"))
                if status in ["running", "verifying"]:
                    self._progress[src] = (copied, total, "done")
        else:
            # Error
            # self._progress[src] = (copied, total, "error")
            print(f"[{dst = }] Hashes do not match")
            raise RuntimeError("Hashes do not match")
        # TODO: update status and io accordingly
