"""

Pretensors: slice-aware lazy-evaluated [pretend-tensors] from pure functions.

```
@pretend(dims="n c h w")
def func(x: torch.Tensor, y: torch.Tensor, R) -> torch.Tensor:
    ...

```

Key properties:

- Output (Pretensor) runtime indexing supports:
    * int | slice | tuple indices
    * fewer-than-ndims indices
    * ellipsis (...)
    * None start/stop values
    * negative index wrap-around
    * step must be None or 1 (v1)



- @pretend(dims="n c h w") named axis specs
- R is label-addressable: R["n"], R[0], and shorthand R.sl("n") / R.sl(0)
- Input indexing inside the traced function supports:
    * fewer-than-ndims indices
    * ellipsis (...)
    * missing dims / ":" interpreted as the requested output slice on that axis: R.sl(axis) (This is applied in BOTH analysis-time proxy indexing and runtime-time replay indexing.)

- Analysis pass records every input __getitem__ slice as an expression of R (requested output slice)
- Strict shape inference derived purely from constraints AND supports negative start offsets by shrinking the output domain via an inferred per-axis base offset (valid-only behavior).
- Runtime evaluation:
    * computes required input slices for the requested output slice
    * fetches each slice separately
    * replays the original function unchanged using wrappers that return the prefetched slices in order
    * enforces runtime access pattern (more/fewer gets) and runtime output-shape correctness
    * PyTorch-friendly: runtime-only compute/type/shape checking (beyond requested output shape).

Notes / constraints:
- Analysis only records how input slices depend on R. It does NOT attempt to run real compute.
- Inputs may only be accessed via __getitem__ inside the function (contract).
- Slice bounds used to index inputs must be of form: R[axis].start/stop (+/- int).
- The function must return exactly one array/tensor and its shape must equal the requested output slice shape.
"""

from __future__ import annotations
from sympy.integrals.risch import NonElementaryIntegral

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal
import inspect
import functools


###
# Axis spec parsing
###

def _parse_axis_spec(ndims: Union[int, str]) -> Tuple[int, Tuple[str, ...], Dict[str, int]]:
    if isinstance(ndims, int):
        if ndims <= 0:
            raise ValueError("ndims must be a positive int.")
        labels = tuple(str(i) for i in range(ndims))
        axis_to_int = {name: i for i, name in enumerate(labels)}
        return ndims, labels, axis_to_int

    if isinstance(ndims, str):
        labels = tuple(s for s in ndims.split() if s)
        if not labels:
            raise ValueError("ndims string must contain at least one label.")
        if len(set(labels)) != len(labels):
            raise ValueError("Axis labels must be unique.")
        axis_to_int = {name: i for i, name in enumerate(labels)}
        return len(labels), labels, axis_to_int

    raise TypeError("ndims must be an int or a space-separated string like 'n c h w'.")


###
# Symbolic slice expression system
###

@dataclass(frozen=True)
class AxisTerm:
    axis: int
    which: Literal["start", "stop"]  # "start" or "stop"

    def __post_init__(self) -> None:
        if self.which not in ("start", "stop"):
            raise ValueError("AxisTerm.which must be 'start' or 'stop'.")

    def __add__(self, k: int) -> "LinExpr":
        if not isinstance(k, int):
            raise TypeError("Can only add int offsets to R bounds.")
        return LinExpr(self, k)

    def __sub__(self, k: int) -> "LinExpr":
        if not isinstance(k, int):
            raise TypeError("Can only subtract int offsets from R bounds.")
        return LinExpr(self, -k)


@dataclass(frozen=True)
class LinExpr:
    term: AxisTerm
    offset: int = 0

    def __add__(self, k: int) -> "LinExpr":
        if not isinstance(k, int):
            raise TypeError("Can only add int offsets.")
        return LinExpr(self.term, self.offset + k)

    def __sub__(self, k: int) -> "LinExpr":
        if not isinstance(k, int):
            raise TypeError("Can only subtract int offsets.")
        return LinExpr(self.term, self.offset - k)

    def eval(self, req: Tuple[slice, ...]) -> int:
        s = req[self.term.axis]
        base = s.start if self.term.which == "start" else s.stop
        return int(base) + int(self.offset)


BoundExpr = Union[AxisTerm, LinExpr]


def _coerce_bound_expr(x: Any) -> LinExpr:
    if isinstance(x, LinExpr):
        return x
    if isinstance(x, AxisTerm):
        return LinExpr(x, 0)
    raise TypeError("Slice bounds must be R[axis].start/stop optionally +/- int.")


@dataclass(frozen=True)
class SliceExpr:
    start: LinExpr
    stop: LinExpr

    def eval(self, req: Tuple[slice, ...]) -> slice:
        a = self.start.eval(req)
        b = self.stop.eval(req)
        if b < a:
            raise ValueError(f"Evaluated slice has stop < start: {a}:{b}")
        return slice(a, b, None)


@dataclass(frozen=True)
class SliceTupleExpr:
    axes: Tuple[SliceExpr, ...]

    def eval(self, req: Tuple[slice, ...]) -> Tuple[slice, ...]:
        return tuple(ax.eval(req) for ax in self.axes)


###
# Requested slice object R (analysis and runtime)
###

class RequestedAxis:
    __slots__ = ("_axis",)

    def __init__(self, axis: int) -> None:
        self._axis = axis

    @property
    def start(self) -> AxisTerm:
        return AxisTerm(self._axis, "start")

    @property
    def stop(self) -> AxisTerm:
        return AxisTerm(self._axis, "stop")


class RequestedSlice:
    """Analysis-time R supporting R['n'] and R.sl('n')."""
    __slots__ = ("ndims", "_labels", "_axis_to_int")

    def __init__(self, ndims: int, labels: Tuple[str, ...], axis_to_int: Dict[str, int]) -> None:
        self.ndims = ndims
        self._labels = labels
        self._axis_to_int = axis_to_int

    def _to_axis(self, key: Union[int, str]) -> int:
        if isinstance(key, str):
            if key not in self._axis_to_int:
                raise KeyError(f"Unknown axis label {key!r}. Known: {self._labels}")
            return self._axis_to_int[key]
        if isinstance(key, int):
            if key < 0 or key >= self.ndims:
                raise IndexError(key)
            return key
        raise TypeError("Axis key must be int or str.")

    def __getitem__(self, key: Union[int, str]) -> RequestedAxis:
        return RequestedAxis(self._to_axis(key))
    
    def __len__(self) -> int:
        return self.ndims
    
    def __iter__(self):
        s = self.sl()
        if isinstance(s, slice):
            yield s
        else:
            yield from s

    def sl(self, key: Optional[Union[int, str]]=None) -> slice|tuple[slice, ...]:
        if key is None:
            axes = [self[dim] for dim in range(self.ndims)]
            return tuple(slice(ax.start, ax.stop, None) or slice(None) for ax in axes)
        ax = self[key]
        return slice(ax.start, ax.stop, None)


class RuntimeR:
    """Runtime R wrapper supporting R['n'] and R.sl('n')."""
    __slots__ = ("_req", "_axis_to_int", "_labels")

    def __init__(self, req: Tuple[slice, ...], labels: Tuple[str, ...], axis_to_int: Dict[str, int]) -> None:
        self._req = req
        self._axis_to_int = axis_to_int
        self._labels = labels

    def _to_axis(self, key: Union[int, str]) -> int:
        if isinstance(key, str):
            if key not in self._axis_to_int:
                raise KeyError(f"Unknown axis label {key!r}. Known: {self._labels}")
            return self._axis_to_int[key]
        if isinstance(key, int):
            return key
        raise TypeError("Axis key must be int or str.")

    def __getitem__(self, key: Union[int, str]) -> slice:
        return self._req[self._to_axis(key)]
    
    def __len__(self) -> int:
        return len(self.req)
    
    def __iter__(self):
        yield from self._req

    def sl(self, key: Optional[Union[int, str]]=None) -> slice:
        if key is None:
            return self._req
        
        s = self[key]
        return slice(int(s.start), int(s.stop), None)


###
# Index expansion used INSIDE the traced function (analysis + runtime replay)
# Missing dims / ":" mean R.sl(axis)
###

FuncIndexAtom = Union[slice, type(Ellipsis)]
FuncIndex = Union[slice, Tuple[FuncIndexAtom, ...]]

def _expand_func_index_relative_to_R(
    idx: FuncIndex,
    ndims: int,
    R: Union[RequestedSlice, RuntimeR],
    labels: Tuple[str, ...],
) -> Tuple[slice, ...]:
    """
    Expand an index used inside the traced function to a full ndims tuple.

    Allowed atoms: slice and ellipsis.
    Missing dims and ellipsis are filled with R.sl(axis) (requested output slice on that axis).
    """
    if isinstance(idx, slice):
        parts: List[FuncIndexAtom] = [idx]
    else:
        parts = list(idx)

    ell_count = sum(1 for p in parts if p is Ellipsis)
    if ell_count > 1:
        raise IndexError("At most one ellipsis (...) is allowed in input indexing.")

    # Validate atom types early
    for p in parts:
        if p is Ellipsis:
            continue
        if not isinstance(p, slice):
            raise TypeError("Inside the traced function, inputs may only be indexed with slices and ellipsis.")

    if ell_count == 1:
        ell_pos = parts.index(Ellipsis)
        n_explicit = len(parts) - 1
        if n_explicit > ndims:
            raise IndexError(f"Too many indices: got {n_explicit} explicit for ndims={ndims}")

        fill = ndims - n_explicit
        # Fill with R.sl for the appropriate axes
        fill_slices = [R.sl(labels[ell_pos + i]) for i in range(fill)]
        parts = parts[:ell_pos] + fill_slices + parts[ell_pos + 1 :]
    else:
        if len(parts) > ndims:
            raise IndexError(f"Too many indices: got {len(parts)} for ndims={ndims}")
        # Pad missing dims with R.sl for remaining axes
        for ax in range(len(parts), ndims):
            parts.append(R.sl(labels[ax]))

    if len(parts) != ndims:
        raise RuntimeError("Internal error: expanded index does not match ndims.")

    # Ensure all are slices
    out: List[slice] = []
    for s in parts:
        assert isinstance(s, slice)
        if s.step is not None:
            raise ValueError("Input indexing inside traced function: slice.step must be None.")
        if s.start is None or s.stop is None:
            # Inside function, start/stop must be explicitly defined relative to R.
            # R.sl(axis) already provides start/stop. Any other None is not allowed.
            raise ValueError("Input indexing inside traced function: slice.start/stop may not be None.")
        out.append(slice(s.start, s.stop, None))
    return tuple(out)


###
# Runtime output indexing normalization for Pretensor.__getitem__
# (Pythonic: int/slice/ellipsis/fewer dims, None bounds, negative wrap-around)
###

RuntimeAtom = Union[int, slice, type(Ellipsis)]
RuntimeIndex = Union[RuntimeAtom, Tuple[RuntimeAtom, ...]]

def _normalize_runtime_output_index(
    idx: RuntimeIndex,
    out_shape: Tuple[int, ...],
) -> Tuple[Tuple[slice, ...], Tuple[Union[int, slice], ...]]:
    """
    Normalize user-provided Pretensor indexing into:
        - req_user: ndims tuple of concrete slices (int start/stop, step=None)
        - post_index: ndims tuple applied to computed result to honor int indexing
        (0 for int axes, slice(None) otherwise)

    Missing dims and ellipsis fill with slice(None) (full output axis).
    """
    nd = len(out_shape)

    if isinstance(idx, tuple):
        parts = list(idx)
    else:
        parts = [idx]

    ell_count = sum(1 for p in parts if p is Ellipsis)
    if ell_count > 1:
        raise IndexError("At most one ellipsis (...) is allowed in output indexing.")

    if ell_count == 1:
        ell_pos = parts.index(Ellipsis)
        n_explicit = len(parts) - 1
        if n_explicit > nd:
            raise IndexError(f"Too many indices: got {n_explicit} explicit for ndims={nd}")
        fill = nd - n_explicit
        parts = parts[:ell_pos] + [slice(None)] * fill + parts[ell_pos + 1 :]
    else:
        if len(parts) > nd:
            raise IndexError(f"Too many indices: got {len(parts)} for ndims={nd}")
        parts = parts + [slice(None)] * (nd - len(parts))

    assert len(parts) == nd

    req: List[slice] = []
    post: List[Union[int, slice]] = []

    for axis, (p, dim) in enumerate(zip(parts, out_shape)):
        if isinstance(p, int):
            i = p
            if i < 0:
                i += dim
            if i < 0 or i >= dim:
                raise IndexError(f"Index {p} out of bounds for axis {axis} with size {dim}")
            req.append(slice(i, i + 1, None))
            post.append(0)

        elif isinstance(p, slice):
            step = p.step
            if step not in (None, 1):
                raise ValueError("Output slicing: step must be None or 1 in v1.")

            start = 0 if p.start is None else int(p.start)
            stop = dim if p.stop is None else int(p.stop)

            # Wrap-around for negatives (Python-like)
            if start < 0:
                start += dim
            if stop < 0:
                stop += dim

            # Clamp to [0, dim]
            if start < 0:
                start = 0
            if start > dim:
                start = dim
            if stop < 0:
                stop = 0
            if stop > dim:
                stop = dim
            if stop < start:
                stop = start

            req.append(slice(start, stop, None))
            post.append(slice(None))

        elif p is Ellipsis:
            raise RuntimeError("Internal error: ellipsis not expanded.")

        else:
            raise TypeError("Output index elements must be int, slice, or ellipsis (...).")

    return tuple(req), tuple(post)


def _requested_shape_from_req(req_user: Tuple[slice, ...]) -> Tuple[int, ...]:
    return tuple(int(s.stop) - int(s.start) for s in req_user)


###
# Analysis recording proxies and plan
###

@dataclass(frozen=True)
class RecordedAccess:
    input_name: str
    slice_expr: SliceTupleExpr


class AnalysisContext:
    def __init__(self) -> None:
        self.accesses: List[RecordedAccess] = []


class AnalysisValue:
    """
    Opaque placeholder for "a sliced tensor" during analysis.
    Any op returns another AnalysisValue so user code can run through analysis.
    """
    __slots__ = ("_tag",)

    def __init__(self, tag: str = "v") -> None:
        self._tag = tag

    # @property
    # def shape(self):
    #     return (-1, )

    def _new(self, op: str, other: Any = None) -> "AnalysisValue":
        if other is None:
            return AnalysisValue(f"({op}{self._tag})")
        o = getattr(other, "_tag", repr(other))
        return AnalysisValue(f"({self._tag}{op}{o})")

    # arithmetic
    def __add__(self, other): return self._new("+", other)
    def __radd__(self, other): return AnalysisValue(f"({getattr(other,'_tag',repr(other))}+{self._tag})")
    def __sub__(self, other): return self._new("-", other)
    def __rsub__(self, other): return AnalysisValue(f"({getattr(other,'_tag',repr(other))}-{self._tag})")
    def __mul__(self, other): return self._new("*", other)
    def __rmul__(self, other): return AnalysisValue(f"({getattr(other,'_tag',repr(other))}*{self._tag})")
    def __truediv__(self, other): return self._new("/", other)
    def __rtruediv__(self, other): return AnalysisValue(f"({getattr(other,'_tag',repr(other))}/{self._tag})")
    def __floordiv__(self, other): return self._new("//", other)
    def __rfloordiv__(self, other): return AnalysisValue(f"({getattr(other,'_tag',repr(other))}//{self._tag})")
    def __pow__(self, other): return self._new("**", other)
    def __rpow__(self, other): return AnalysisValue(f"({getattr(other,'_tag',repr(other))}**{self._tag})")
    def __neg__(self): return AnalysisValue(f"(-{self._tag})")

    # comparisons return placeholders (not bool), preventing accidental data-dependent branching
    def __lt__(self, other): return self._new("<", other)
    def __le__(self, other): return self._new("<=", other)
    def __gt__(self, other): return self._new(">", other)
    def __ge__(self, other): return self._new(">=", other)
    def __eq__(self, other): return self._new("==", other)
    def __ne__(self, other): return self._new("!=", other)

    # method calls become placeholders
    def __getattr__(self, name: str) -> Any:
        def _method(*args: Any, **kwargs: Any) -> "AnalysisValue":
            return AnalysisValue(f"{self._tag}.{name}(...)")
        return _method

    def __getitem__(self, idx: Any) -> "AnalysisValue":
        return AnalysisValue(f"{self._tag}[...]")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return AnalysisValue(f"{getattr(func,'__name__','torchfunc')}(...)")

    def __repr__(self) -> str:
        return f"AnalysisValue({self._tag})"
    


class InputProxyPretensor:
    """
    Analysis-time proxy for an input array.
    Only supports __getitem__ where slices are derived from R bounds (+/- ints),
    and supports fewer-than-ndims / ellipsis expansion with missing dims => R.sl(axis).
    """
    def __init__(
        self,
        name: str,
        ndims: int,
        ctx: AnalysisContext,
        R: RequestedSlice,
        labels: Tuple[str, ...],
    ) -> None:
        self._name = name
        self._ndims = ndims
        self._ctx = ctx
        self._R = R
        self._labels = labels

    def __getitem__(self, idx: FuncIndex) -> AnalysisValue:
        full_idx = _expand_func_index_relative_to_R(idx, self._ndims, self._R, self._labels)

        axes_expr: List[SliceExpr] = []
        for s in full_idx:
            # Already validated step None, start/stop not None in _expand_func_index_relative_to_R
            start = _coerce_bound_expr(s.start)
            stop = _coerce_bound_expr(s.stop)
            axes_expr.append(SliceExpr(start=start, stop=stop))

        st = SliceTupleExpr(tuple(axes_expr))
        self._ctx.accesses.append(RecordedAccess(self._name, st))
        return AnalysisValue(f"{self._name}[{len(self._ctx.accesses)}]")

    def __getattr__(self, attr: str) -> Any:
        raise AttributeError(
            f"Analysis mode: inputs may only be accessed via __getitem__. "
            f"Tried attribute {attr!r} on {self._name}."
        )


@dataclass(frozen=True)
class CompiledPlan:
    ndims: int
    labels: Tuple[str, ...]
    axis_to_int: Dict[str, int]
    input_names: Tuple[str, ...]
    accesses: Tuple[RecordedAccess, ...]


def _run_analysis(
    func: Callable[..., Any],
    input_names: Sequence[str],
    ndims: int,
    labels: Tuple[str, ...],
    axis_to_int: Dict[str, int],
) -> CompiledPlan:
    ctx = AnalysisContext()
    R = RequestedSlice(ndims, labels, axis_to_int)
    proxies = [InputProxyPretensor(name=n, ndims=ndims, ctx=ctx, R=R, labels=labels) for n in input_names]
    func(*proxies, R)  # record
    return CompiledPlan(
        ndims=ndims,
        labels=labels,
        axis_to_int=axis_to_int,
        input_names=tuple(input_names),
        accesses=tuple(ctx.accesses),
    )


###
# Runtime replay wrapper
# It supports fewer-than-ndims / ellipsis expansion for consistency,
# but ignores idx and returns prefetched chunks in order.
###

class _ReplayPretensor:
    def __init__(self, chunks: List[Any], name: str) -> None:
        self._chunks = chunks
        self._i = 0
        self._name = name

    def __getitem__(self, idx: Any) -> Any:
        # We ignore idx (plan determines which slices were fetched)
        if self._i >= len(self._chunks):
            raise IndexError(
                f"{self._name}: function requested more slices at runtime than during analysis."
            )
        chunk = self._chunks[self._i]
        self._i += 1
        return chunk

    def _assert_consumed_all(self) -> None:
        if self._i != len(self._chunks):
            raise RuntimeError(
                f"{self._name}: function consumed fewer slices at runtime than during analysis "
                f"({self._i} consumed, {len(self._chunks)} planned). "
                f"This suggests data-dependent control flow or a mismatch in indexing."
            )


###
# Strict output domain inference (shape + base) from constraints
# Negative start offsets shrink output via base offsets.
###

def _infer_output_domain_strict(plan: CompiledPlan, inputs_by_name: Dict[str, Any]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Infer (out_shape, out_base) strictly from recorded constraints and input shapes.

    User-visible coordinates: [0:O)
    Logical coordinates used in recorded expressions: [base : base+O)

    Strictness:
        - Each axis must have at least one upper bound derived from a stop expression using R[axis].stop
        (i.e., sexpr.stop.term.which == "stop"). Otherwise raise.
    """
    nd = plan.ndims
    base: List[int] = [0] * nd
    upper_O: List[Optional[int]] = [None] * nd
    lower_O: List[int] = [0] * nd

    # Pass 1: base from start expressions using R.start: b + off >= 0 => b >= -off
    for acc in plan.accesses:
        arr = inputs_by_name.get(acc.input_name)
        if arr is None:
            raise KeyError(f"Missing input {acc.input_name!r} for inference.")
        in_shape = getattr(arr, "shape", None)
        # if in_shape is None or len(in_shape) != nd:
        #     raise ValueError(f"Input {acc.input_name!r} must have .shape of length {nd} for inference.")

        for axis, sexpr in enumerate(acc.slice_expr.axes):
            if sexpr.start.term.which == "start":
                off = int(sexpr.start.offset)
                base[axis] = max(base[axis], -off)

    # Pass 2: apply constraints using logical full output slice: R'.start=b, R'.stop=b+O
    for acc in plan.accesses:
        arr = inputs_by_name[acc.input_name]
        in_shape = tuple(int(x) for x in arr.shape)

        for axis, sexpr in enumerate(acc.slice_expr.axes):
            D = in_shape[axis]
            b = base[axis]

            # Start constraint: start >= 0
            if sexpr.start.term.which == "start":
                # start = b + off; b chosen to ensure nonnegative
                pass
            else:
                # start = (b + O) + off >= 0 => O >= -(b + off)
                off = int(sexpr.start.offset)
                lower_O[axis] = max(lower_O[axis], -(b + off))

            # Stop constraint: stop <= D
            if sexpr.stop.term.which == "start":
                # stop = b + off <= D => b <= D - off
                off = int(sexpr.stop.offset)
                if b > D - off:
                    raise ValueError(
                        f"Inference failed: base on axis {axis} is {b}, but constraint requires base <= {D - off} "
                        f"for input {acc.input_name!r}."
                    )
            else:
                # stop = (b + O) + off <= D => O <= D - off - b
                off = int(sexpr.stop.offset)
                ub = D - off - b
                upper_O[axis] = ub if upper_O[axis] is None else min(upper_O[axis], ub)

    out_shape: List[int] = []
    for axis in range(nd):
        if upper_O[axis] is None:
            raise ValueError(
                f"Inference failed on axis {axis} ({plan.labels[axis]!r}): cannot determine output extent "
                f"(no recorded stop bound used R[{axis}].stop)."
            )
        ub = int(upper_O[axis])
        lb = int(lower_O[axis])
        if ub < lb:
            raise ValueError(
                f"Inference failed on axis {axis} ({plan.labels[axis]!r}): no feasible O (lb={lb}, ub={ub})."
            )
        out_shape.append(ub)

    return tuple(out_shape), tuple(base)


# def _infer_output_domain_strict(plan: CompiledPlan, inputs_by_name: Dict[str, Any]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
#     """
#     Infer (out_shape, out_base) strictly from recorded constraints and input shapes.

#     User-visible coordinates: [0:O)
#     Logical coordinates used in recorded expressions: [base : base+O)

#     Strictness:
#         - Each axis must have at least one upper bound derived from a stop expression using R[axis].stop
#         (i.e., sexpr.stop.term.which == "stop"). Otherwise raise.
#     """
#     nd = plan.ndims
#     base: List[int] = [0] * nd
#     upper_O: List[Optional[int]] = [None] * nd
#     lower_O: List[int] = [0] * nd

#     # Pass 1: base from start expressions using R.start: b + off >= 0 => b >= -off
#     for acc in plan.accesses:
#         arr = inputs_by_name.get(acc.input_name)
#         if arr is None:
#             raise KeyError(f"Missing input {acc.input_name!r} for inference.")
#         in_shape = getattr(arr, "shape", None)
#         if in_shape is None or len(in_shape) != nd:
#             raise ValueError(f"Input {acc.input_name!r} must have .shape of length {nd} for inference.")

#         for axis, sexpr in enumerate(acc.slice_expr.axes):
#             if sexpr.start.term.which == "start":
#                 off = int(sexpr.start.offset)
#                 base[axis] = max(base[axis], -off)

#     # Pass 2: apply constraints using logical full output slice: R'.start=b, R'.stop=b+O
#     for acc in plan.accesses:
#         arr = inputs_by_name[acc.input_name]
#         in_shape = tuple(int(x) for x in arr.shape)

#         for axis, sexpr in enumerate(acc.slice_expr.axes):
#             D = in_shape[axis]
#             b = base[axis]

#             # Start constraint: start >= 0
#             if sexpr.start.term.which == "start":
#                 # start = b + off; b chosen to ensure nonnegative
#                 pass
#             else:
#                 # start = (b + O) + off >= 0 => O >= -(b + off)
#                 off = int(sexpr.start.offset)
#                 lower_O[axis] = max(lower_O[axis], -(b + off))

#             # Stop constraint: stop <= D
#             if sexpr.stop.term.which == "start":
#                 # stop = b + off <= D => b <= D - off
#                 off = int(sexpr.stop.offset)
#                 if b > D - off:
#                     raise ValueError(
#                         f"Inference failed: base on axis {axis} is {b}, but constraint requires base <= {D - off} "
#                         f"for input {acc.input_name!r}."
#                     )
#             else:
#                 # stop = (b + O) + off <= D => O <= D - off - b
#                 off = int(sexpr.stop.offset)
#                 ub = D - off - b
#                 upper_O[axis] = ub if upper_O[axis] is None else min(upper_O[axis], ub)

#     out_shape: List[int] = []
#     for axis in range(nd):
#         if upper_O[axis] is None:
#             raise ValueError(
#                 f"Inference failed on axis {axis} ({plan.labels[axis]!r}): cannot determine output extent "
#                 f"(no recorded stop bound used R[{axis}].stop)."
#             )
#         ub = int(upper_O[axis])
#         lb = int(lower_O[axis])
#         if ub < lb:
#             raise ValueError(
#                 f"Inference failed on axis {axis} ({plan.labels[axis]!r}): no feasible O (lb={lb}, ub={ub})."
#             )
#         out_shape.append(ub)

#     return tuple(out_shape), tuple(base)


###
# Pretensor (minimal interface) with runtime-friendly indexing and dtype inference
###

class Pretensor:
    """
    Minimal array-like:
        - .shape
        - .dtype (None until first compute)
        - __getitem__(...) with Pythonic indexing
    """
    def __init__(self, shape: Tuple[int, ...], getter: Callable[[Any], Any], axis_to_int: Dict[str, int]) -> None:
        self._shape = shape
        self._dtype: Optional[Any] = None
        self._device: Optional[Any] = None
        self._getter: Callable[[Any], Any] = getter
        self.axis_to_int: Dict[str, int] = axis_to_int

    @property
    def dtype(self) -> Any:
        if self._dtype is None: # If dtype hasn't been set because we haven't computed yet
            self._dtype = self[0].dtype
        return self._dtype

    def __getitem__(self, idx: Any) -> Any:
        return self._getter(idx)
    
    def __len__(self) -> int:
        return self._shape[0]
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape
    
    @property
    def device(self) -> Tuple[int, ...]:
        if self._device is None: # If dtype hasn't been set because we haven't computed yet
            self._device = self[0].device
        return self._device


###
# Decorator
###

def pretend(*, dims: str) -> Callable[[Callable[..., Any]], Callable[..., Pretensor]]:
    """
    Usage:
        @pretend(dims="n c h w")
        def f(x, R): ...

    - dtype is inferred on first runtime compute: pretend_arr.dtype = result.dtype (if present)
    - Inside the traced function, missing dims and ":" are interpreted as R.sl(axis).
    """
    out_ndims, labels, axis_to_int = _parse_axis_spec(dims)

    
    def decorator(func: Callable[..., Any]) -> Callable[..., Pretensor]:
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        if len(param_names) < 2:
            raise TypeError("Function must accept at least one input and R as the last argument.")
        input_param_names = param_names[:-1]  # last is R by convention

        # Analysis at decoration time
        plan = _run_analysis(func, input_names=input_param_names, ndims=out_ndims, labels=labels, axis_to_int=axis_to_int)

        @functools.wraps(func)
        def factory(*inputs: Any) -> Pretensor:
            if len(inputs) != len(input_param_names):
                raise TypeError(f"Expected {len(input_param_names)} inputs, got {len(inputs)}")

            inputs_by_name: Dict[str, Any] = dict(zip(input_param_names, inputs))

            # Infer output shape and base (valid-only) from constraints
            out_shape, out_base = _infer_output_domain_strict(plan, inputs_by_name)

            # Construct the Pretensor, capturing itself for dtype inference
            pretend_arr: Pretensor

            def getter(user_idx: Any) -> Any:
                nonlocal pretend_arr

                # 1) Normalize user output indexing (Pythonic)
                req_user, post_index = _normalize_runtime_output_index(user_idx, out_shape)

                # Bounds check in user coordinates
                for ax, s in enumerate(req_user):
                    if s.start < 0 or s.stop > out_shape[ax]:
                        raise IndexError(
                            f"Requested output slice {req_user} out of bounds for output shape {out_shape}."
                        )

                # 2) Shift to logical coordinates (valid-only)
                req_logical = tuple(
                    slice(req_user[i].start + out_base[i], req_user[i].stop + out_base[i], None)
                    for i in range(out_ndims)
                )

                # 3) Fetch required input slices
                fetched_chunks: Dict[str, List[Any]] = {n: [] for n in input_param_names}
                for acc in plan.accesses:
                    in_idx = acc.slice_expr.eval(req_logical)  # concrete slices in logical coords
                    # No additional normalization here: recorded expressions already enforce step None, ints.
                    chunk = inputs_by_name[acc.input_name][in_idx]
                    fetched_chunks[acc.input_name].append(chunk)

                # 4) Run user function with replay arrays and runtime R wrapper
                replay_args: List[_ReplayPretensor] = [
                    _ReplayPretensor(fetched_chunks[name], name=name) for name in input_param_names
                ]
                R_runtime = RuntimeR(req_logical, labels=labels, axis_to_int=axis_to_int)
                result = func(*replay_args, R_runtime) # TODO: Move R to the first positional arg

                # 5) Enforce access-pattern match
                for ra in replay_args:
                    ra._assert_consumed_all()

                # 6) Enforce result shape equals requested output slice shape (in user coords)
                expected_shape = _requested_shape_from_req(req_user)
                res_shape = getattr(result, "shape", None)
                if res_shape is None:
                    raise TypeError("Runtime result must have a .shape attribute.")
                if tuple(res_shape) != expected_shape:
                    raise ValueError(
                        f"Runtime result shape {tuple(res_shape)} does not match requested shape {expected_shape}."
                    )

                # 7) Infer dtype and device on first compute
                if pretend_arr._dtype is None:
                    pretend_arr._dtype = getattr(result, "dtype", None)

                if pretend_arr._device is None:
                    pretend_arr._device = getattr(result, "device", "cpu")

                # 8) Apply post-index to honor int indexing (rank reduction)
                if any(isinstance(p, int) for p in post_index):
                    result = result[post_index]

                return result

            pretend_arr = Pretensor(shape=out_shape, getter=getter, axis_to_int=axis_to_int)
            return pretend_arr

        factory.__name__ = func.__name__
        factory.__doc__ = func.__doc__
        factory.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            parameters=[sig.parameters[n] for n in input_param_names],
            return_annotation=Pretensor,
        )
        return factory

    return decorator


###
# Optional: simple logging wrapper for debugging
###

class LoggingPretensor:
    """
    Wrap an array/tensor to log __getitem__ calls. Useful to verify pretend slicing.
    """
    def __init__(self, data: Any, name: str = "x"):
        self._data = data
        self.name = name
        self.shape = tuple(data.shape)
        self.dtype = getattr(data, "dtype", None)
        self.calls: List[Any] = []

    def __getitem__(self, idx: Any) -> Any:
        self.calls.append(idx)
        return self._data[idx]


###
# Minimal sanity demo (backend-agnostic using numpy)
###

if __name__ == "__main__":
    import numpy as np

    # Example: out has axes "n c h w" (4D)
    N, C, H, W = 10, 3, 8, 9
    ds_np = (np.arange(N * C * H * W) % 255).astype(np.uint8).reshape(N, C, H, W)
    ds = LoggingPretensor(ds_np, "ds")

    @pretend(dims="n c h w")
    def temporal_contrast_pretend_array(ds, R):
        k = 5
        pad = k // 2
        # fewer-than-ndims input indexing: missing dims fill with R.sl("c"), R.sl("h"), R.sl("w")
        batch = ds[slice(R["n"].start - pad, R["n"].stop + pad)]
        batch01 = batch / 255.0
        # For demo, just return the center slice (requested output shape) by trimming pad:
        # In a real torch implementation, contrast.temporal_contrast(k)(batch01) would produce requested shape.
        return batch01[pad:-pad, :, :, :]

    out = temporal_contrast_pretend_array(ds)

    # With pad=2, valid-only domain shrinks along "n": output N becomes N-4
    assert out.shape == (N - 4, C, H, W)
    print("out.shape:", out.shape, "dtype before:", out.dtype)

    # Output indexing supports fewer dims / ellipsis / None / negatives:
    y = out[0:1, ...]  # (1, C, H, W)
    print("y.shape:", y.shape, "dtype after first compute:", out.dtype)

    # Verify ds was only sliced as needed
    print("ds __getitem__ calls:")
    for call in ds.calls:
        print(" ", call)
