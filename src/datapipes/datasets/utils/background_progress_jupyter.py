# import threading
# import queue
# import time
# import ipywidgets as widgets
# from IPython.display import display

# class ThreadSafeProgress:
#     """
#     IPython/Jupyter-friendly progress widget that keeps updating
#     *after the cell returns*.

#     - Worker threads call .report(n) and .set_status(text) (thread-safe).
#     - A dedicated UI thread consumes queued updates and updates the widget.
#     - Call .close() when done (or use as a context manager).

#     Notes:
#     - In classic Jupyter/Notebook/Lab, widget updates generally work from a background
#       thread via trait updates. This pattern avoids the worker touching widget state.


#     Example use:
#     ```
#     from datapipes.datasets.utils.background_progress_jupyter import ThreadSafeProgress
#     import threading
#     import time

#     def prefetch_worker(p: ThreadSafeProgress, n: int):
#         for i in range(n):
#             time.sleep(0.05)  # simulate IO
#             p.report(1)
#             if i % 20 == 0:
#                 p.set_status(f"Fetched item {i}...")
#         p.set_status("Done")
#         p.close()

#     p = ThreadSafeProgress(total=200, desc="Prefetching").display()
#     threading.Thread(target=prefetch_worker, args=(p, 200), daemon=True).start()


#     ```
#     """
#     _SENTINEL_CLOSE = object()

#     def __init__(self, total: int | None = None, desc: str = "Working", show_eta: bool = True):
#         self.total = total
#         self.desc = desc
#         self.show_eta = show_eta

#         self._q: "queue.Queue[tuple[str, object] | object]" = queue.Queue()
#         self._stop_evt = threading.Event()
#         self._ui_thread: threading.Thread | None = None
#         self._start_time = None
#         self._value = 0
#         self._lock = threading.Lock()

#         # --- widgets ---
#         self._title = widgets.HTML(value=f"<b>{self.desc}</b>")
#         self._bar = widgets.IntProgress(
#             value=0,
#             min=0,
#             max=(total if total is not None else 1),  # updated later for unknown total
#             description="",
#         )
#         self._meta = widgets.HTML(value="")  # e.g. "12 / 100 • 3.1 it/s • ETA 00:28"
#         self._status = widgets.HTML(value="")  # optional arbitrary status text
#         self.widget = widgets.VBox([self._title, self._bar, self._meta, self._status])

#     def display(self):
#         """Display the widget once. Safe to call multiple times."""
#         display(self.widget)
#         # Start UI thread on first display
#         if self._ui_thread is None:
#             self._start_time = time.time()
#             self._ui_thread = threading.Thread(target=self._ui_loop, name="ThreadSafeProgressUI", daemon=True)
#             self._ui_thread.start()
#         return self

#     # ---------- thread-safe reporting API (call from worker threads) ----------
#     def report(self, n: int = 1):
#         """Report that n units of work completed."""
#         if n:
#             self._q.put(("inc", int(n)))

#     def set_total(self, total: int):
#         """Set/adjust the total (useful if you learn it later)."""
#         self._q.put(("total", int(total)))

#     def set_status(self, text: str):
#         """Set a status line under the bar."""
#         self._q.put(("status", str(text)))

#     def close(self):
#         """Stop updating and mark complete."""
#         self._q.put(self._SENTINEL_CLOSE)

#     def stop(self):
#         """Alias for close()."""
#         self.close()

#     # ---------- context manager ----------
#     def __enter__(self):
#         self.display()
#         return self

#     def __exit__(self, exc_type, exc, tb):
#         self.close()
#         return False

#     # ---------- internal UI loop ----------
#     def _ui_loop(self):
#         last_render = 0.0
#         # We batch updates to avoid spamming comms
#         pending_inc = 0
#         pending_status: str | None = None
#         pending_total: int | None = None

#         while not self._stop_evt.is_set():
#             try:
#                 msg = self._q.get(timeout=0.1)
#             except queue.Empty:
#                 msg = None

#             if msg is self._SENTINEL_CLOSE:
#                 # flush any pending increments
#                 if pending_inc:
#                     self._apply_inc(pending_inc)
#                     pending_inc = 0
#                 if pending_total is not None:
#                     self._apply_total(pending_total)
#                     pending_total = None
#                 if pending_status is not None:
#                     self._apply_status(pending_status)
#                     pending_status = None

#                 # mark done visually
#                 self._bar.bar_style = "success"
#                 self._stop_evt.set()
#                 break

#             if isinstance(msg, tuple):
#                 kind, payload = msg
#                 if kind == "inc":
#                     pending_inc += int(payload)
#                 elif kind == "status":
#                     pending_status = str(payload)
#                 elif kind == "total":
#                     pending_total = int(payload)

#             # Render at most ~10 times/sec
#             now = time.time()
#             if now - last_render >= 0.1:
#                 if pending_total is not None:
#                     self._apply_total(pending_total)
#                     pending_total = None
#                 if pending_inc:
#                     self._apply_inc(pending_inc)
#                     pending_inc = 0
#                 if pending_status is not None:
#                     self._apply_status(pending_status)
#                     pending_status = None
#                 self._render_meta()
#                 last_render = now

#         # final render
#         self._render_meta()

#     def _apply_total(self, total: int):
#         with self._lock:
#             self.total = total
#             self._bar.max = max(1, total)  # IntProgress requires max >= 1

#     def _apply_inc(self, n: int):
#         with self._lock:
#             self._value += n
#             # If total unknown, grow max so bar still shows motion
#             if self.total is None:
#                 # Keep a little headroom so it doesn't pin at 100%
#                 self._bar.max = max(self._bar.max, self._value + 1)
#             self._bar.value = min(self._value, self._bar.max)

#     def _apply_status(self, text: str):
#         # escape is handled by HTML widget; if you need raw HTML, pass it intentionally
#         self._status.value = f"<span>{text}</span>"

#     def _render_meta(self):
#         with self._lock:
#             v = self._value
#             total = self.total
#             t0 = self._start_time

#         if not t0:
#             return

#         elapsed = max(1e-9, time.time() - t0)
#         rate = v / elapsed

#         def fmt_eta(seconds: float) -> str:
#             seconds = max(0, int(seconds))
#             m, s = divmod(seconds, 60)
#             h, m = divmod(m, 60)
#             if h:
#                 return f"{h:02d}:{m:02d}:{s:02d}"
#             return f"{m:02d}:{s:02d}"

#         if total is None:
#             meta = f"{v} • {rate:.2f} it/s"
#         else:
#             remaining = max(0, total - v)
#             eta = (remaining / rate) if rate > 0 else float("inf")
#             if self.show_eta and eta != float("inf"):
#                 meta = f"{v} / {total} • {rate:.2f} it/s • ETA {fmt_eta(eta)}"
#             else:
#                 meta = f"{v} / {total} • {rate:.2f} it/s"

#         self._meta.value = f"<span style='font-family: monospace;'>{meta}</span>"



#///

import threading
import queue
import time
import html as _html
import math

import ipywidgets as widgets
from IPython.display import display

from dataclasses import dataclass
from typing import Optional, Callable, Literal
from pathlib import Path

from pathlib import Path
from datapipes.utils.benchmarking import human_readable_filesize, human_readable_time
import time

@dataclass
class ProgressContext:
    # identity
    desc: str

    # progress
    value: int
    total_used_for_stats: Optional[int]
    total_logical_size: Optional[int]
    total_storage_size: Optional[int]
    pct: Optional[float]

    # timing
    rate: float
    elapsed: float
    remaining: Optional[int]
    eta: str

    # rendering helpers
    meta: str
    status: str
    done: bool
    max_for_progress: int

    error: Optional[str] = None



def get_css() -> str:
    return f'''
.section {{ margin: 10px 0 18px 0; }}
.sectitle {{ font-weight: 700; margin: 6px 0 8px 0; font-size: 14px; }}
.filename {{ font-weight: 600; }}
.filepath {{ font-size: 12px; opacity: .7; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; /*max-width: 1px;*/ }}
.filecell {{ max-width: 0; }}
.barwrap progress {{ width: 100%; height: 16px; }}
.bartext {{ display: flex; justify-content: space-between; font-size: 12px; margin-top: 6px; gap: 10px; }}
.muted {{ opacity: .75; white-space: nowrap; }}
.badge {{ display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; line-height: 18px; }}
.b-running {{ background: #2196F3; }}


.pb-container {{
  background-color: #e9ecef; /* Track color */
  border-radius: .25rem; /* Default Bootstrap-like radius */
  overflow: hidden; /* Important for border-radius on fill */
  box-shadow: inset 0 1px 2px rgba(0,0,0,.075);
  height: 20px; /* Default height */
  position: relative; /* For labels if any */
}}

.pb-fill {{
  height: 100%;
  background-color: #007bff; /* Default blue, will be overridden by color classes */
  transition: width .1s ease;
  display: flex; /* For label alignment */
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 0.75rem;
  font-weight: 500;
  white-space: nowrap;
}}


/* Source: https://codeshack.io/pure-css-modern-progress-bars-collection/*/

/* ==== Style Variation 4: Rounded (pb-rounded) ==== */
.pb-rounded {{
  border-radius: 50px; /* Fully rounded container */
  height: 22px; /* Slightly taller for rounded aesthetic */
}}
.pb-rounded .pb-fill {{
  border-radius: 50px; /* Fully rounded fill */
}}
.pb-rounded .pb-label {{
  padding: 0 10px;
}}

/* ==== Style Variation 5: Slim (pb-slim) ==== */
.pb-slim {{
  height: 8px;
  border-radius: 4px;
  box-shadow: none; /* Often no shadow for slim bars */
}}
.pb-slim .pb-fill {{
  /* No text label typical for slim bars */
}}

/* ==== Color Variations for .pb-fill ==== */
/* Blue (Default, but explicit class for clarity) */
.pb-blue .pb-fill {{ background-color: #2196F3; }}

/* Green */
.pb-green .pb-fill {{ background-color: #4CAF50; }}

/* Green */
.pb-green-ratio {{ background-color: #4CAF50BF; }}
.pb-green-ratio .pb-fill {{ background-color: #4CAF50; }}

/* Red */
.pb-red .pb-fill {{ background-color: #f44336; }}

/* Orange */
.pb-orange .pb-fill {{ background-color: #ff9800; }}

/* Grey */
.pb-grey .pb-fill {{ background-color: #78909c; }}


'''

def render_progress_bar(full_path: Path, progress: ProgressContext) -> str:
    
    if progress.total_used_for_stats is not None:

        secondary_total: str = ""
        if progress.total_logical_size is not None and progress.total_logical_size != progress.total_used_for_stats:
            secondary_total = f" (📎{human_readable_filesize(progress.total_logical_size)})"
        elif progress.total_storage_size is not None and progress.total_storage_size != progress.total_used_for_stats:
            secondary_total = f" (💾{human_readable_filesize(progress.total_storage_size)})"


        total_str: str = f"{human_readable_filesize(progress.total_used_for_stats)}{secondary_total}"
    else:
        total_str = "-"
    

    file_size_progress = f'<span class="muted">{human_readable_filesize(progress.value)}/{total_str}</span>'

    load_rate = f'<span class="muted">{human_readable_filesize(int(progress.rate))}/s</span>'

    progress_pct = f'<span>{progress.pct:.1f}%</span>'
    progress_eta = f'<span class="muted">ETA: {progress.eta}</span>'

    pb_color = "pb-green"
    error_display = ""
    if progress.error is not None:
        pb_color = "pb-red"
        error_display = f'''
<h2>Error:</h2>
<span>{progress.error}</span>
'''
    
    pb = f'''
<style>
{get_css()}
</style>
<div class="barcell">
    <span>{progress.desc}</span>
    <br>
    <span><b>{full_path.name}:</b> {progress.status}</span>
    <div class="bartext">
        <span>{progress_pct} ({int(progress.elapsed)}s)</span>
        {file_size_progress}
    </div>
    <div class="pb-container pb-slim {pb_color}"><div class="pb-fill" style="width: {progress.pct:.1f}%;"></div></div>
    <div class="bartext">
        {progress_eta}
        {load_rate}    
    </div>
</div>
{error_display}
'''
    return pb

from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class EmaRate:
    """
    Exponentially-smoothed rate estimator for monotonic counters.

    Minimal interface:
      - update(value, now=None) -> float  (smoothed rate, units/sec)
      - reset(value=0, now=None)

    Design choices:
      - Uses per-interval instantaneous rate (dv/dt).
      - If dv <= 0, leaves EMA unchanged (prevents ETA spikes during pauses).
      - Seeds EMA on first positive dv, else 0.0.
    """
    alpha: float = 0.2

    _ema: Optional[float] = None
    _last_t: Optional[float] = None
    _last_v: Optional[int] = None

    def reset(self, value: int = 0, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        self._ema = None
        self._last_t = now
        self._last_v = int(value)

    def update(self, value: int, now: Optional[float] = None) -> float:
        if now is None:
            now = time.perf_counter()
        value = int(value)

        # First observation
        if self._last_t is None or self._last_v is None:
            self._last_t, self._last_v = now, value
            self._ema = self._ema if self._ema is not None else 0.0
            return float(self._ema)

        dt = max(1e-12, now - self._last_t)
        dv = value - self._last_v

        inst = (dv / dt) if dv > 0 else 0.0

        if self._ema is None:
            self._ema = inst if inst > 0 else 0.0
        else:
            if inst > 0:
                a = float(self.alpha)
                self._ema = a * inst + (1.0 - a) * self._ema
            # else: keep EMA unchanged

        self._last_t, self._last_v = now, value
        return float(self._ema)

EMA_ALPHA: float = 0.2

# TODO: Create matlab cop-out
class ThreadSafeProgress:
    _SENTINEL_CLOSE = object()

    def __init__(
        self,
        *,
        total_logical_size: int | None = None,
        total_storage_size: Optional[int] = None,
        total_type_used_for_stats: Literal["logical", "storage"]="logical",
        desc: str = "Working",
        render_html_func: Callable[[Path, ProgressContext], str] = render_progress_bar,
        path: Path = Path("."),
        escape_status: bool = True,
        show_eta: bool = True,
        update_hz: float = 10.0,
    ):
        """
        total: None for unknown total.
        template: single HTML template string (uses str.format_map on a context dict).
        escape_status: if True, status text is HTML-escaped (safe default). Set False for raw HTML status.
        """
        self.total_logical_size = total_logical_size
        self.total_storage_size = total_storage_size or self.total_logical_size
        self.total_type_used_for_stats: Literal["logical", "storage"] = total_type_used_for_stats
        self.desc = desc
        self.path = path
        self.render_html = render_html_func
        self.escape_status = escape_status
        self.show_eta = show_eta
        self.update_hz = max(1.0, float(update_hz))

        self._rate_ema = EmaRate(alpha=EMA_ALPHA)

        self._error: Optional[str] = None


        self._q: "queue.Queue[tuple[str, object] | object]" = queue.Queue()
        self._stop_evt = threading.Event()
        self._ui_thread: threading.Thread | None = None

        self._lock = threading.Lock()
        self._start_time: float | None = None
        self._value = 0
        self._status = ""

        # single widget output
        self.widget = widgets.HTML(value="")

    def display(self):
        """Display once and start the UI loop thread."""
        display(self.widget)
        if self._ui_thread is None:
            self._start_time = time.time()
            self._rate_ema.reset(value=0, now=self._start_time)

            self._start_time = time.time()
            self._ui_thread = threading.Thread(target=self._ui_loop, name="ThreadSafeProgressUI", daemon=True)
            self._ui_thread.start()
            # initial render
            self._render()
        return self

    # ---------- thread-safe reporting API (call from worker threads) ----------
    def report(self, n: int = 1):
        if n:
            self._q.put(("inc", int(n)))

    def error(self, error_msg: str):
        self._error = error_msg
        self._render()
        self.close()

    def set_total(self, total: int):
        self._q.put(("total", int(total)))

    def set_status(self, text: str):
        self._q.put(("status", str(text)))

    def close(self):
        self._q.put(self._SENTINEL_CLOSE)

    def stop(self):
        self.close()

    # ---------- context manager ----------
    def __enter__(self):
        self.display()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # ---------- internal ----------
    def _ui_loop(self):
        # Batch updates to reduce comm chatter
        pending_inc = 0
        pending_total: int | None = None
        pending_status: str | None = None

        min_interval = 1.0 / self.update_hz
        last_render = 0.0

        while not self._stop_evt.is_set():
            try:
                msg = self._q.get(timeout=min_interval)
            except queue.Empty:
                msg = None

            if msg is self._SENTINEL_CLOSE:
                if pending_total is not None:
                    self._apply_total(pending_total)
                    pending_total = None
                if pending_inc:
                    self._apply_inc(pending_inc)
                    pending_inc = 0
                if pending_status is not None:
                    self._apply_status(pending_status)
                    pending_status = None
                self._stop_evt.set()
                self._render(done=True)
                break

            if isinstance(msg, tuple):
                kind, payload = msg
                if kind == "inc":
                    pending_inc += int(payload)
                elif kind == "total":
                    pending_total = int(payload)
                elif kind == "status":
                    pending_status = str(payload)

            now = time.time()
            if now - last_render >= min_interval:
                if pending_total is not None:
                    self._apply_total(pending_total)
                    pending_total = None
                if pending_inc:
                    self._apply_inc(pending_inc)
                    pending_inc = 0
                if pending_status is not None:
                    self._apply_status(pending_status)
                    pending_status = None
                self._render()
                last_render = now

        # final render (in case of thread exit)
        self._render(done=True)

    def _apply_total(self, total: int):
        with self._lock:
            self.total_logical_size = max(0, total)

    def _apply_inc(self, n: int):
        with self._lock:
            self._value += n
            if self._value < 0:
                self._value = 0

    def _apply_status(self, text: str):
        with self._lock:
            self._status = text

    def _format_eta(self, seconds: float) -> str:
        if not math.isfinite(seconds) or seconds < 0:
            return "--:--"
        seconds = int(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _render(self, done: bool = False):
        with self._lock:
            value = self._value
            total_logical_size = self.total_logical_size
            total_storage_size = self.total_storage_size
            status = self._status
            start_time = self._start_time
            desc = self.desc

        elapsed = max(1e-9, (time.time() - start_time) if start_time else 0.0)

        rate = self._rate_ema.update(value=value, now=time.perf_counter())

        match self.total_type_used_for_stats:
            case "logical":
                total_used_for_stats = total_logical_size
            case "storage":
                total_used_for_stats = total_storage_size
            case _:
                raise ValueError(f"Unrecognized total type: {self.total_type_used_for_stats = }")

        if total_used_for_stats is None:
            pct = None
            remaining = None
            eta = ""
            meta = f"{value} • {rate:.2f} it/s"
            max_for_progress = max(1, value + 1)
        else:
            pct = 0.0 if total_used_for_stats == 0 else min(1.0, max(0.0, value / total_used_for_stats))
            remaining = max(0, total_used_for_stats - value)
            eta_val = (remaining / rate) if (rate > 0 and self.show_eta) else None
            eta = self._format_eta(eta_val) if eta_val is not None else ""
            if eta:
                meta = f"{value} / {total_used_for_stats} • {rate:.2f} it/s • ETA {eta}"
            else:
                meta = f"{value} / {total_used_for_stats} • {rate:.2f} it/s"
            max_for_progress = max(1, total_used_for_stats)

        # escape status unless raw HTML is allowed
        if self.escape_status:
            status_render = _html.escape(status)
        else:
            status_render = status

        ctx = ProgressContext(
            desc=_html.escape(desc),
            value=value,
            total_used_for_stats=total_used_for_stats,
            total_logical_size=total_logical_size,
            total_storage_size=total_storage_size,
            pct=(pct * 100.0) if pct is not None else None,
            rate=rate,
            elapsed=elapsed,
            remaining=remaining,
            eta=eta,
            meta=_html.escape(meta),
            status=status_render,
            done=done,
            max_for_progress=max_for_progress,
            error=self._error,
        )

        try:
            rendered = self.render_html(self.path, ctx)
        except Exception as e:
            rendered = (
                f"<pre style='color:#b00;'>Template error: {_html.escape(str(e))}</pre>"
                f"<pre>{_html.escape(repr(ctx))}</pre>"
            )

        self.widget.value = rendered
        if self._error:
                self.close()
