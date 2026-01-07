import time
import queue
import threading
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Callable, Any
from enum import StrEnum


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParallelProcessTask:
    src: str
    dst: str
    size: int

class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    VERIFYING = "verifying"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"

_active_states: List[TaskStatus] = [TaskStatus.RUNNING, TaskStatus.VERIFYING]
_unfinished_states: List[TaskStatus] = [TaskStatus.QUEUED] + _active_states

_success_states: List[TaskStatus] = [TaskStatus.DONE]
_fail_states: List[TaskStatus] = [TaskStatus.ERROR, TaskStatus.CANCELLED]
_finished_states: List[TaskStatus] = _success_states + _fail_states

@dataclass(kw_only=True)
class ProgressEntry:
    processed_bytes: int
    total_bytes: int
    status: TaskStatus
    status_display_label: Optional[str]=None # TODO: Allow overriding status display
    custom_display_stats: List[str]=None


@dataclass
class TaskStats:
    last_t: float
    last_copied: float
    ema_bps: float

@dataclass
class SnapshotRow:
    src_path: str
    dst_path: str
    status: TaskStatus
    copied: int
    total: int
    percent: float
    bytes_processed: int
    bytes_total: int
    ema_bps: int
    eta_seconds: Optional[float]
    status_display_label: Optional[str]=None
    custom_display_stats: List[str]=None

ProgressCallback = Callable[[ParallelProcessTask, ProgressEntry], None]
TaskStepFunction = Callable[[ParallelProcessTask, ProgressCallback, Any], Any]
# def _copy_file_chunked(self, task: ParallelProcessTask, progress_callback: ProgressCallback) -> blake3:
class ParallelProcessManager:
    """
    Manages a queue of tasks and runs them in parallel, with live per-task progress.
    Status values: queued, running, done, error, cancelled
    """

    def __init__(
        self,
        run_task: TaskStepFunction,
        verify_task: TaskStepFunction,
        ema_alpha: float = 0.20,          # smoothing: higher -> more reactive
        min_eta_rate_bps: float = 0.1
    ):
        logger.info(f"Initializing ParallelProcessManager with {run_task=}, {verify_task=} {ema_alpha=}, {min_eta_rate_bps=}")
        self.run_task = run_task
        self.verify_task = verify_task

        self._ema_alpha = float(ema_alpha)
        self._min_eta_rate_bps = float(min_eta_rate_bps)

        self._lock = threading.Lock()
        self._q: "queue.Queue[ParallelProcessTask]" = queue.Queue()
        self._executor: ThreadPoolExecutor | None = None
        self._futures: List[Future] = []
        self._cancel_event = threading.Event()

        # {src: (copied_bytes, total_bytes, status)}
        self._progress: Dict[ParallelProcessTask, ProgressEntry] = {}
        self._queued: List[str] = []
        self._running: set[str] = set()

        # Stats for bitrate/ETA
        # {src: TaskStats}
        self._stats: Dict[str, TaskStats] = {}

    def reset(self) -> None:
        logger.info("Resetting ParallelProcessManager")
        try:
            self.cancel()
            with self._lock:
                self._q = queue.Queue()
                self._progress.clear()
                self._queued = []
                self._running.clear()
                self._futures.clear()
            self._cancel_event.clear()
            logger.debug("ParallelProcessManager reset successfully")
        except Exception as e:
            logger.error(f"Error during reset: {e}", exc_info=True)

    def cancel(self) -> None:
        logger.info("Cancelling all operations")
        try:
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
                cancelled_count = 0
                for task, entry in list(self._progress.items()):
                    if entry.status in _unfinished_states:
                        self._progress[task] = ProgressEntry(
                            processed_bytes=entry.processed_bytes,
                            total_bytes=entry.total_bytes,
                            status=TaskStatus.CANCELLED
                        )
                        cancelled_count += 1

                self._running.clear()
            logger.info(f"Cancelled {cancelled_count} active operations")
        except Exception as e:
            logger.error(f"Error during cancel: {e}", exc_info=True)

    def enqueue(self, tasks: List[ParallelProcessTask]) -> None:
        logger.info(f"Enqueuing {len(tasks)} tasks")
        try:
            with self._lock:
                for t in tasks:
                    self._q.put(t)
                    self._progress[t] = ProgressEntry(processed_bytes=0, total_bytes=t.size, status=TaskStatus.QUEUED)
                    self._queued.append(t.src)
            logger.debug(f"Successfully enqueued {len(tasks)} tasks")
        except Exception as e:
            logger.error(f"Error enqueuing tasks: {e}", exc_info=True)

    def start(self, max_workers: int) -> None:
        logger.info(f"Starting processing with {max_workers} workers")
        try:
            if self._executor is not None:
                logger.warning("Executor already exists, not starting new one")
                return
            self._cancel_event.clear()
            self._executor = ThreadPoolExecutor(max_workers=max_workers)

            # Each worker thread consumes tasks sequentially.
            for _ in range(max_workers):
                self._futures.append(self._executor.submit(self._consumer_loop))
            logger.info(f"Started {max_workers} worker threads")
        except Exception as e:
            logger.error(f"Error starting workers: {e}", exc_info=True)

    def is_done(self) -> bool:
        try:
            with self._lock:
                done = not any(
                    entry.status in _active_states for entry in self._progress.values()
                )
            if done:
                logger.debug("All tasks completed")
            return done
        except Exception as e:
            logger.error(f"Error checking if done: {e}", exc_info=True)
            return False

    def snapshot(self) -> tuple[list[SnapshotRow], list[str]]:
        """
        Returns:
          rows: list[SnapshotRow] with progress information
          queued_list: list[str] of files still waiting (paths)
        """
        try:
            with self._lock:
                rows = []
                for task, entry in self._progress.items():
                    copied = entry.processed_bytes
                    total = entry.total_bytes
                    status = entry.status

                    percent = 0.0 if total == 0 else ((copied / total) * 100.0)
                    stats = self._stats.get(task.src)
                    ema_bps = stats.ema_bps if stats else 0.0

                    remaining = max(0, total - copied)
                    if status not in _active_states:
                        eta_seconds: Optional[float] = 0.0 if status in _success_states else None
                    else:
                        if ema_bps >= self._min_eta_rate_bps:
                            eta_seconds = remaining / ema_bps
                        else:
                            eta_seconds = None
                    rows.append(
                        SnapshotRow(
                            src_path=task.src,
                            dst_path=task.dst,
                            status=status,
                            copied=copied,
                            total=total,
                            percent=max(0.0, min(100.0, percent)),
                            bytes_processed=copied,
                            bytes_total=(total) if total else 0.0,
                            ema_bps=ema_bps,
                            eta_seconds=eta_seconds,
                            status_display_label=entry.status_display_label,
                            custom_display_stats=entry.custom_display_stats,
                        )
                    )
                queued_list = list(self._queued)

            rows.sort(key=lambda r: r.src_path)
            return rows, queued_list
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}", exc_info=True)
            return [], []

    def _consumer_loop(self) -> None:
        logger.debug("Worker thread started")
        try:
            while not self._cancel_event.is_set():
                try:
                    task = self._q.get(timeout=0.2)
                except queue.Empty:
                    if self._cancel_event.is_set():
                        logger.debug("Worker thread exiting due to cancellation")
                        return
                    time.sleep(0.05)
                    continue
                
                logger.info(f"Processing task: {task.src} -> {task.dst}")
                now = time.monotonic()
                with self._lock:
                    if task.src in self._queued:
                        self._queued.remove(task.src)
                    self._running.add(task.src)
                    entry = self._progress.get(task)
                    if entry is None:
                        entry = ProgressEntry(processed_bytes=0, total_bytes=task.size, status=TaskStatus.QUEUED)
                    self._progress[task] = ProgressEntry(
                        processed_bytes=entry.processed_bytes,
                        total_bytes=entry.total_bytes,
                        status=TaskStatus.RUNNING
                    )

                    # Initialize timing baseline for bitrate
                    st = self._stats.get(task.src)
                    if st is None:
                        self._stats[task.src] = TaskStats(last_t=now, last_copied=float(entry.processed_bytes), ema_bps=0.0)
                    else:
                        st.last_t = now
                        st.last_copied = float(entry.processed_bytes)

                try:
                    self._process_task(task)
                    logger.info(f"Successfully processed task: {task.src}")
                except Exception as e:
                    logger.error(f"Error processing task {task.src}: {e}", exc_info=True)
                    with self._lock:
                        entry = self._progress.get(task)
                        if entry is None:
                            entry = ProgressEntry(processed_bytes=0, total_bytes=task.size, status=TaskStatus.RUNNING)
                        if self._cancel_event.is_set():
                            self._progress[task] = ProgressEntry(
                                processed_bytes=entry.processed_bytes,
                                total_bytes=entry.total_bytes,
                                status=TaskStatus.CANCELLED
                            )
                        else:
                            self._progress[task] = ProgressEntry(
                                processed_bytes=entry.processed_bytes,
                                total_bytes=entry.total_bytes,
                                status=TaskStatus.ERROR
                            )
                finally:
                    with self._lock:
                        self._running.discard(task.src)
        except Exception as e:
            logger.critical(f"Critical error in worker thread: {e}", exc_info=True)

    

    def _update_ema_rate_locked(self, src: str, copied: int) -> None:
        """
        Update EMA throughput for src. Call only while holding self._lock.
        """
        try:
            now = time.monotonic()
            st = self._stats.get(src)
            if st is None:
                self._stats[src] = TaskStats(last_t=now, last_copied=float(copied), ema_bps=0.0)
                return

            last_t = st.last_t
            last_c = st.last_copied
            dt = max(1e-6, now - last_t)
            dc = max(0.0, float(copied) - last_c)

            inst_bps = dc / dt
            ema_bps = st.ema_bps

            # If EMA not seeded yet, seed with first instantaneous estimate
            if ema_bps <= 0.0:
                ema_bps = inst_bps
            else:
                a = self._ema_alpha
                ema_bps = a * inst_bps + (1.0 - a) * ema_bps

            self._stats[src] = TaskStats(
                ema_bps=ema_bps,
                last_t=now,
                last_copied=float(copied),
            )
        except Exception as e:
            logger.error(f"Error updating EMA rate for {src}: {e}", exc_info=True)

    def _process_task(self, task: ParallelProcessTask):
        logger.debug(f"Starting processing of task: {task.src}")
        try:
            def progress_callback(task: ParallelProcessTask, progress: ProgressEntry):
                if self._cancel_event.is_set():
                        raise RuntimeError("cancelled")
                
                with self._lock:
                    entry = self._progress.get(task)
                    if entry and entry.status in _unfinished_states:
                        self._progress[task] = progress
                        self._update_ema_rate_locked(task.src, progress.processed_bytes)

            source_hasher = self.run_task(task, progress_callback=progress_callback)
            self.verify_task(task, progress_callback=progress_callback, source_hasher=source_hasher)
            logger.debug(f"Completed processing of task: {task.src}")
        except Exception as e:
            logger.error(f"Error in _process_task for {task.src}: {e}", exc_info=True)

