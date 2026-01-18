from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Iterable, List


# ---- compact opcodes (fast replay, minimal overhead) ----
ADD, SUB, MUL, FLOORDIV, MOD, POW, NEG = 1, 2, 3, 4, 5, 6, 7
RADD, RSUB, RMUL, RFLOORDIV, RMOD, RPOW = 101, 102, 103, 104, 105, 106

Op = Tuple[int, Optional[int]]  # (opcode, const) ; const is None only for NEG


@dataclass(frozen=True)
class TracedInt:
    """
    Int-like value that records a linear "tape" of arithmetic ops and can replay them with a new seed.

    Design constraints:
        - int-only constants (no floats)
        - no combining traces (TraceInt op TraceInt is an error)
        - supports reverse ops (e.g., 3 - t)
        - no true division (/); only floor division (//)
        - relies on Python fallback for += etc. (no __iadd__)
        - skips comparisons/hashing extras
    """
    value: int
    ops: Tuple[Op, ...] = ()

    # ---------- validation ----------
    @staticmethod
    def _require_int_const(x: Any) -> int:
        # Disallow combining traces (keeps recording semantics simple)
        if isinstance(x, TracedInt):
            raise TypeError("Combining traces is not supported (other operand is TraceInt).")
        # Keep it strictly int-only
        if not isinstance(x, int):
            raise TypeError(f"Only int constants are supported (got {type(x).__name__}).")
        return x

    def _with(self, new_value: int, rec: Op) -> "TracedInt":
        return TracedInt(new_value, self.ops + (rec,))

    # ---------- replay ----------
    def replay(self, seed: int) -> int:
        """Replay the recorded tape starting from `seed`."""
        if not isinstance(seed, int):
            raise TypeError("seed must be an int")

        acc = seed
        # Localize ops reference (tiny speed win in hot code)
        for code, c in self.ops:
            if code == ADD:          acc = acc + c  # type: ignore[operator]
            elif code == SUB:        acc = acc - c  # type: ignore[operator]
            elif code == MUL:        acc = acc * c  # type: ignore[operator]
            elif code == FLOORDIV:   acc = acc // c # type: ignore[operator]
            elif code == MOD:        acc = acc % c  # type: ignore[operator]
            elif code == POW:        acc = acc ** c # type: ignore[operator]
            elif code == NEG:        acc = -acc

            elif code == RADD:       acc = c + acc  # type: ignore[operator]
            elif code == RSUB:       acc = c - acc  # type: ignore[operator]
            elif code == RMUL:       acc = c * acc  # type: ignore[operator]
            elif code == RFLOORDIV:  acc = c // acc # type: ignore[operator]
            elif code == RMOD:       acc = c % acc  # type: ignore[operator]
            elif code == RPOW:       acc = c ** acc # type: ignore[operator]
            else:
                raise RuntimeError(f"Unknown opcode: {code}")
        return acc

    def replay_many(self, seeds: Iterable[int]) -> List[int]:
        """
        Efficient-ish helper when you're replaying the same TraceInt many times.
        Avoids per-seed method call overhead if you keep this method call outside the hot loop.
        """
        ops = self.ops
        out: List[int] = []
        append = out.append

        for seed in seeds:
            if not isinstance(seed, int):
                raise TypeError("All seeds must be ints")
            acc = seed
            for code, c in ops:
                if code == ADD:          acc = acc + c  # type: ignore[operator]
                elif code == SUB:        acc = acc - c  # type: ignore[operator]
                elif code == MUL:        acc = acc * c  # type: ignore[operator]
                elif code == FLOORDIV:   acc = acc // c # type: ignore[operator]
                elif code == MOD:        acc = acc % c  # type: ignore[operator]
                elif code == POW:        acc = acc ** c # type: ignore[operator]
                elif code == NEG:        acc = -acc

                elif code == RADD:       acc = c + acc  # type: ignore[operator]
                elif code == RSUB:       acc = c - acc  # type: ignore[operator]
                elif code == RMUL:       acc = c * acc  # type: ignore[operator]
                elif code == RFLOORDIV:  acc = c // acc # type: ignore[operator]
                elif code == RMOD:       acc = c % acc  # type: ignore[operator]
                elif code == RPOW:       acc = c ** acc # type: ignore[operator]
                else:
                    raise RuntimeError(f"Unknown opcode: {code}")
            append(acc)

        return out

    # ---------- int-like ----------
    def __int__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return f"TraceInt(value={self.value}, ops={self.ops})"

    # ---------- forward ops ----------
    def __add__(self, other: Any) -> "TracedInt":
        n = self._require_int_const(other)
        return self._with(self.value + n, (ADD, n))

    def __sub__(self, other: Any) -> "TracedInt":
        n = self._require_int_const(other)
        return self._with(self.value - n, (SUB, n))

    def __mul__(self, other: Any) -> "TracedInt":
        n = self._require_int_const(other)
        return self._with(self.value * n, (MUL, n))

    def __floordiv__(self, other: Any) -> "TracedInt":
        n = self._require_int_const(other)
        return self._with(self.value // n, (FLOORDIV, n))

    def __mod__(self, other: Any) -> "TracedInt":
        n = self._require_int_const(other)
        return self._with(self.value % n, (MOD, n))

    def __pow__(self, other: Any) -> "TracedInt":
        n = self._require_int_const(other)
        return self._with(self.value ** n, (POW, n))

    def __neg__(self) -> "TracedInt":
        return self._with(-self.value, (NEG, None))

    # ---------- reverse ops ----------
    def __radd__(self, other: Any) -> "TracedInt":
        n = self._require_int_const(other)
        return self._with(n + self.value, (RADD, n))

    def __rsub__(self, other: Any) -> "TracedInt":
        n = self._require_int_const(other)
        return self._with(n - self.value, (RSUB, n))

    def __rmul__(self, other: Any) -> "TracedInt":
        n = self._require_int_const(other)
        return self._with(n * self.value, (RMUL, n))

    def __rfloordiv__(self, other: Any) -> "TracedInt":
        n = self._require_int_const(other)
        return self._with(n // self.value, (RFLOORDIV, n))

    def __rmod__(self, other: Any) -> "TracedInt":
        n = self._require_int_const(other)
        return self._with(n % self.value, (RMOD, n))

    def __rpow__(self, other: Any) -> "TracedInt":
        n = self._require_int_const(other)
        return self._with(n ** self.value, (RPOW, n))
