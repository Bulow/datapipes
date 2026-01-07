
import time
from typing import Iterable, Optional, Callable
import gradio as gr
from functools import wraps

import time
from typing import Iterable, Optional, Iterator, TypeVar
import gradio as gr

T = TypeVar("T")

def progress_tqdm_with_eta(progress: gr.Progress) -> Callable:
    @wraps(progress.tqdm)
    def tqdm_eta(
        iterable: Iterable[T],
        *,
        total: Optional[int] = None,
        desc: str = "",
        unit: str = "it",
        ema_alpha: float = 0.15,
        min_update_interval: float = 0.0,   # seconds; set e.g. 0.1 to reduce UI spam
    ) -> Iterator[T]:
        """
        Drop-in replacement for progress.tqdm that shows:
        - seconds per iteration (EMA-smoothed)
        - ETA
        in the Gradio progress description.

        Usage:
            for x in tqdm_eta(range(100), progress=progress, desc="Working"):
                ...
        """
        if total is None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                raise ValueError("total must be provided if iterable has no len()")

        start = time.perf_counter()
        last_yield_t = start
        last_ui_t = 0.0
        ema_s_per_it: Optional[float] = None

        # Initialize bar at 0
        progress((0, total), desc=f"{desc}", unit=unit)

        for i, item in enumerate(iterable, start=1):
            # We yield FIRST so the user's work time is measured until the next iteration begins.
            yield item

            now = time.perf_counter()
            dt = now - last_yield_t
            last_yield_t = now

            # EMA smoothing
            if ema_s_per_it is None:
                ema_s_per_it = dt
            else:
                ema_s_per_it = ema_alpha * dt + (1 - ema_alpha) * ema_s_per_it

            remaining = total - i
            eta_s = remaining * ema_s_per_it
            elapsed = now - start

            # Optional throttling to reduce UI updates
            if min_update_interval and (now - last_ui_t) < min_update_interval and i < total:
                continue
            last_ui_t = now

            progress(
                (i, total),
                desc=(
                    f"{desc} - {ema_s_per_it:.3f}s/{unit} - "
                    f"ETA {eta_s:,.1f}s"
                ),
                unit=unit,
            )

        # Ensure final state is shown
        end = time.perf_counter()
        total_elapsed = end - start
        progress((total, total), desc=f"{desc} • done in {total_elapsed:,.1f}s", unit=unit)

    return tqdm_eta