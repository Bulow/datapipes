# simpletqdm.py
from __future__ import annotations

import sys
import time
from typing import Iterable, Iterator, Optional, TypeVar, Generic, Any

T = TypeVar("T")


class SimpleTqdm(Generic[T]):
    """
    Minimal, log-friendly tqdm-like progress indicator for NON-TTY outputs.

    Behavior:
      - Prints one "scale" line:    <desc> [----------]
      - Then prints a second line:  <desc> [||||      ]   (but as *appends only*)
        by emitting '|' characters as progress increases.
      - No carriage returns, no ANSI, suitable for redirected stdout / CI logs.

    Usage:
      for x in tqdm(items, desc="Work"):
          ...

      p = tqdm(total=100, desc="Upload")
      p.update(5)
      ...
      p.close()
    """

    def __init__(
        self,
        iterable: Optional[Iterable[T]] = None,
        *,
        total: Optional[int] = None,
        desc: str = "",
        file: Any = None,
        width: int = 50,
        mininterval: float = 0.5,
        leave: bool = True,
        unit: str = "it",
        disable: Optional[bool] = None,
    ):
        self.iterable = iterable
        self.total = total if total is not None else (len(iterable) if iterable is not None and hasattr(iterable, "__len__") else None)
        self.desc = desc
        self.file = file if file is not None else sys.stdout
        self.width = max(1, int(width))
        self.mininterval = float(mininterval)
        self.leave = bool(leave)
        self.unit = unit

        # If disable not provided: disable fancy behavior on TTY? (We WANT non-tty),
        # but allow user to force.
        if disable is None:
            self.disable = False
        else:
            self.disable = bool(disable)

        self.n = 0
        self._start = time.time()
        self._last_print = 0.0
        self._printed_units = 0  # number of '|' already emitted on the bar line
        self._header_printed = False
        self._closed = False

        if not self.disable:
            self._print_header_and_start_bar()

    def __iter__(self) -> Iterator[T]:
        if self.iterable is None:
            raise TypeError("tqdm: iterable is None (use tqdm(total=...) and update())")
        for x in self.iterable:
            yield x
            self.update(1)
        self.close()

    def update(self, n: int = 1) -> None:
        if self.disable or self._closed:
            self.n += n
            return

        self.n += n
        now = time.time()
        if (now - self._last_print) < self.mininterval and self.total is not None:
            # If we know total, we can delay printing a bit for noisy logs.
            return

        self._last_print = now
        self._emit_progress()

    def close(self) -> None:
        if self.disable or self._closed:
            self._closed = True
            return
        self._emit_progress(final=True)
        if self.leave:
            # Finish the bar line by printing the closing bracket + stats and newline.
            self._finish_line()
        else:
            # Still end line cleanly.
            self.file.write("\n")
            self.file.flush()
        self._closed = True

    def __enter__(self) -> "SimpleTqdm[T]":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --- internals ---

    def _label(self) -> str:
        return (self.desc + " ") if self.desc else ""

    def _print_header_and_start_bar(self) -> None:
        if self._header_printed:
            return
        label = self._label()
        # Header “scale” line
        self.file.write(f"{label}[{'-' * self.width}]\n")
        # Start bar line: print opening bracket; we will append '|' chars over time.
        self.file.write(f"{label}[")
        self.file.flush()
        self._header_printed = True

    def _emit_progress(self, final: bool = False) -> None:
        if self.total is None or self.total <= 0:
            # Unknown total: just print occasional ticks + count.
            if final:
                self.file.write(f"] {self.n} {self.unit}\n")
                self.file.flush()
            return

        frac = min(1.0, max(0.0, self.n / self.total))
        target_units = int(frac * self.width)

        # Only append the delta '|' characters
        delta = target_units - self._printed_units
        if delta > 0:
            self.file.write("|" * delta)
            self._printed_units = target_units
            self.file.flush()

        if final and self._printed_units < self.width:
            # On final, fill the remainder to reach full width.
            self.file.write("|" * (self.width - self._printed_units))
            self._printed_units = self.width
            self.file.flush()

    def _finish_line(self) -> None:
        elapsed = time.time() - self._start
        rate = (self.n / elapsed) if elapsed > 0 else 0.0

        if self.total is not None and self.total > 0:
            pct = (100.0 * self.n / self.total)
            self.file.write(f"] {self.n}/{self.total} ({pct:5.1f}%) {rate:0.2f} {self.unit}/s elapsed {elapsed:0.1f}s\n")
        else:
            self.file.write(f"] {self.n} {self.unit} {rate:0.2f} {self.unit}/s elapsed {elapsed:0.1f}s\n")
        self.file.flush()
