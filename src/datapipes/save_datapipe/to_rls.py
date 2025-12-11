from datapipes.save_datapipe.RLS_file_writer import prepare_rls_file, RLS_Writer
from src.datapipes.datapipe import DataPipe
from pathlib import Path
import numpy as np
from datapipes.io_pipeline import Pipeline

def datapipe_to_rls(dp: DataPipe, out_path: Path, n_writers: int=4, n_fetchers: int=4, batch_size=1024):

    with prepare_rls_file(dp, path=out_path, n_writers=n_writers) as rls_writers:

        def fetch_data(index: slice) -> None:
            batch = dp[index]
            return rls_writers[0].interleave_batch(frames=batch, timestamps=np.arange(start=index.start, stop=index.stop, dtype=np.uint64))

        def write_data(index: slice, data: np.ndarray, writer: RLS_Writer) -> None:
            writer.disk_array_view[index] = data

        # Configure and run
        pipe = Pipeline(
            fetch_data,
            write_data,
            max_ready_queue=64,
            num_fetch_workers=n_fetchers,
            num_write_workers=n_writers,
            writer_args=rls_writers
        )
        pipe.run(start=0, stop=len(dp), batch_size=batch_size)
