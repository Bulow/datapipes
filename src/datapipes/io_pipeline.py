import threading
import queue
from typing import Iterable, List, Optional, Any, Tuple
from tqdm import tqdm

class Pipeline:
    """
    Multithreaded fetch→write pipeline with bounded ready-queue backpressure and graceful shutdown.
    It assumes the user provides two callables:
        - fetch_data(index: slice) -> None
        - write_data(index: slice) -> None
    """

    def __init__(
        self,
        fetch_data,
        write_data,
        *,
        max_ready_queue: int = 64,
        num_fetch_workers: int = 4,
        num_write_workers: int = 2,
        writer_args: List[Any] = None
    ):
        self.fetch_data = fetch_data
        self.write_data = write_data

        if writer_args is not None and len(writer_args) != num_write_workers:
            raise ValueError(f"len(writer_args) != num_write_workers, got len(writer_args): {len(writer_args)}, num_write_workers: {num_write_workers}")
        self.writer_args = writer_args
        

        # Queues
        self.to_fetch: "queue.Queue[Optional[slice]]" = queue.Queue()
        self.ready_to_write: "queue.Queue[Optional[Tuple[slice, Any]]]" = queue.Queue(maxsize=max_ready_queue)

        # Concurrency primitives
        self.stop_event = threading.Event()
        self.errors: "queue.Queue[BaseException]" = queue.Queue()

        # Worker thread handles
        self._fetch_threads: List[threading.Thread] = []
        self._write_threads: List[threading.Thread] = []

        # Config
        self.num_fetch_workers = max(1, num_fetch_workers)
        self.num_write_workers = max(1, num_write_workers)

        self._pbar: Optional[tqdm] = None

    @staticmethod
    def _chunk_range(start: int, stop: int, chunk: int) -> Iterable[slice]:
        if chunk <= 0:
            raise ValueError("chunk must be > 0")
        if stop < start:
            return
        i = start
        while i < stop:
            j = min(i + chunk, stop)
            yield slice(i, j)
            i = j

    def _fetch_worker(self, wid: int):
        try:
            while not self.stop_event.is_set():
                task_slice = self.to_fetch.get()
                if task_slice is None:  # sentinel
                    self.to_fetch.task_done()
                    break
                # with tqdm.get_lock():
                #         print(f"to_fetch: [{self.to_fetch.unfinished_tasks}], consumed task_slice: [{task_slice.start}:{task_slice.stop}]")
                try:
                    data = self.fetch_data(task_slice)
                    # Backpressure: block if ready_to_write is full
                    self.ready_to_write.put((task_slice, data))
                except BaseException as e:
                    self.errors.put(e)
                    self.stop_event.set()
                finally:
                    self.to_fetch.task_done()
        finally:
            # Ensure a sentinel for writers if an early stop is triggered
            if self.stop_event.is_set():
                self.ready_to_write.put(None)

    def _write_worker(self, wid: int, writer_arg: Any=None):
        try:
            while not self.stop_event.is_set():
                item = self.ready_to_write.get()
                if item is None:  # sentinel
                    self.ready_to_write.task_done()
                    break
                try:
                    item_slice, data = item
                    if writer_arg is None:
                        self.write_data(item_slice, data)
                    else:
                        self.write_data(item_slice, data, writer_arg)

                    self._pbar.update(1)
                    # with tqdm.get_lock():
                        # print(f"ready_to_write: [{self.ready_to_write}]")
                except BaseException as e:
                    self.errors.put(e)
                    self.stop_event.set()
                finally:
                    self.ready_to_write.task_done()
        finally:
            pass

    def run(
        self,
        *,
        start: int,
        stop: int,
        batch_size: int,
        slices: Optional[Iterable[slice]] = None,
        pbar_desc: str = "Written",
        pbar_leave: bool = True,
    ):
        """
        Run the pipeline until all chunks are fetched and written, or an error occurs.

        You can specify either:
          - start, stop, chunk  → automatically generate slices [start:stop) in 'chunk' sized pieces
          - slices=...          → provide an explicit iterable of slices to process

        On error, raises the first exception encountered and attempts to stop workers cleanly.
        """
        if slices is None:
            work_iter = list(self._chunk_range(start, stop, batch_size))
        else:
            work_iter = list(slices)

        self._pbar = tqdm(total=len(work_iter), desc=pbar_desc, leave=pbar_leave)

        # Start workers
        self._fetch_threads = [
            threading.Thread(target=self._fetch_worker, args=(i,), daemon=True)
            for i in range(self.num_fetch_workers)
        ]
        
        if self.writer_args is None:
            self._write_threads = [
                threading.Thread(target=self._write_worker, args=(i,), daemon=True)
                for i in range(self.num_write_workers)
            ]
        else:
            self._write_threads = [
                threading.Thread(target=self._write_worker, args=(i,self.writer_args[i]), daemon=True)
                for i in range(self.num_write_workers)
            ]

        for t in self._fetch_threads + self._write_threads:
            t.start()

        # Enqueue fetch tasks
        try:
            for slc in work_iter:
                # Early exit if something failed
                if self.stop_event.is_set():
                    break
                self.to_fetch.put(slc)

            # Send sentinels to fetchers
            for _ in self._fetch_threads:
                self.to_fetch.put(None)

            # Wait for fetchers to finish
            self.to_fetch.join()

            # Send sentinels to writers (one per writer)
            for _ in self._write_threads:
                self.ready_to_write.put(None)

            # Wait for writers to finish
            self.ready_to_write.join()

            # If any errors occurred, raise the first one
            if not self.errors.empty():
                raise self.errors.get()

        finally:
            self.stop_event.set()
            for t in self._fetch_threads + self._write_threads:
                t.join(timeout=5.0)
            if self._pbar is not None:
                self._pbar.close()
                self._pbar = None


