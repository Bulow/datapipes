import torch
from typing import Callable, Tuple, Any, Optional, Iterable
from datapipes.utils import introspection, Slicer
from numbers import Integral
from dataclasses import dataclass
import rich

# from auto_indexing import nd_bwd_slice, nd_fwd_slice, NdRelativeIndexRange, RelativeIndexRange

# TODO: Eliminate RelativeIndexRange and NdRelativeIndexRange?

@dataclass(frozen=True)
class RelativeIndexRange:
    start: int
    stop: int

    def __add__(self, other: "RelativeIndexRange"|Tuple[int, int]) -> "RelativeIndexRange":
        if isinstance(other, Tuple):
            other = RelativeIndexRange(*other)
            return RelativeIndexRange(self.start + other.start, self.stop + other.stop)
        
    def make_absolute(self) -> "RelativeIndexRange":
        if self.start < 0:
            return RelativeIndexRange(0, self.stop - self.start)
            
    
    def __len__(self) -> int:
        return self.stop - self.start

    def __str__(self):
        return f"[{self.start}:{self.stop}], len={len(self)}"
    
@dataclass(frozen=True)
class NdRelativeIndexRange:
    dims: Tuple[RelativeIndexRange]

    @staticmethod
    def from_abs_slice_tuple(slices: Tuple[slice]) -> "NdRelativeIndexRange":
        return NdRelativeIndexRange(tuple([RelativeIndexRange(start=slc.start, stop=slc.stop) for slc in slices]))
    
    def __add__(self, other: Tuple["RelativeIndexRange"|Tuple[int, int]]|"NdRelativeIndexRange") -> "NdRelativeIndexRange":
        if isinstance(other, Tuple):
            other = NdRelativeIndexRange(other)
            return NdRelativeIndexRange(tuple([s + o for s, o in zip(self, other)]))
        
    def __repr__(self) -> str:
        dim_strs = [f"{d.start}:{d.stop}" for d in self.dims]
        return f"[{", ".join(dim_strs)}]"
    
    def __str__(self) -> str:
        return repr(self)
    
    @property
    def shape(self) -> Iterable[Integral]:
        return [len(d) for d in self.dims]
    
    def __iter__(self):
        yield from self.dims

def fwd_slice(idx_before: RelativeIndexRange, slc: slice) -> RelativeIndexRange:

    rel_start = slc.start or 0
    rel_stop = slc.stop or 0
    step = slc.step or 1

    a_start = idx_before.start + rel_start
    a_stop = a_start + (((idx_before.stop + rel_stop) - (idx_before.start + rel_start)) // step) # using // differs from normal slicing, because we want to ensure we only get full patches. That also makes it important to use this function for computing the shape in the fwd pass of introspection

    idx_after = RelativeIndexRange(a_start, a_stop)
    return idx_after

def bwd_slice(idx_after: RelativeIndexRange, slc: slice) -> RelativeIndexRange:
    rel_start = slc.start or 0
    rel_stop = slc.stop or 0
    step = slc.step or 1

    b_start = idx_after.start - rel_start
    b_stop = ((idx_after.stop - idx_after.start) * step) + b_start - rel_stop + rel_start

    idx_before = RelativeIndexRange(b_start, b_stop)
    return idx_before


def nd_fwd_slice(idx_before: NdRelativeIndexRange, relative_slc: Tuple[slice]) -> NdRelativeIndexRange:
    assert len(idx_before.dims) == len(relative_slc)
    after_indices = tuple([fwd_slice(b, s) for b, s in zip(idx_before.dims, relative_slc)])
    return NdRelativeIndexRange(after_indices)


def nd_bwd_slice(idx_after: NdRelativeIndexRange, relative_slc: slice) -> NdRelativeIndexRange:
    assert len(idx_after.dims) == len(relative_slc)
    before_indices = tuple([bwd_slice(b, s) for b, s in zip(idx_after.dims, relative_slc)])
    return NdRelativeIndexRange(before_indices)

class AutoIndexingWindow:
    def __init__(self, dataset: torch.Tensor|Any, fwd_transforms: Callable[[torch.Tensor], torch.Tensor]):
        self.dataset = dataset

        self.fwd_rel_slices, self.out_shape = introspection.probe_segments_with_input_shape(segments=fwd_transforms, in_shape=self.dataset.shape)

    
        rel_idx = NdRelativeIndexRange(
            dims=tuple([RelativeIndexRange(start=0, stop=s) for s in self.out_shape])
        )

        # print(rel_idx)
        # back prop shape to find relative source input indices
        for s in self.fwd_rel_slices:
            rel_idx = nd_bwd_slice(rel_idx, relative_slc=s)
            # print(rel_idx)

        self.start_pads = [-r.start for r in rel_idx.dims]
        # print(self.start_pads)

    @property
    def shape(self):
        return self.out_shape
    
    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, idx) -> torch.Tensor:
        idx = Slicer.normalize(index=idx, shape=self.shape)
        # print(idx)

        rel_idx = NdRelativeIndexRange.from_abs_slice_tuple(idx)

        # rich.inspect(rel_idx)
        # back prop shape to find relative source input indices
        for s in self.fwd_rel_slices:
            rel_idx = nd_bwd_slice(rel_idx, relative_slc=s)
            # print(rel_idx)
        # rich.inspect(rel_idx)
        ds_slice = tuple([slice(ridx.start + pad, ridx.stop + pad) for ridx, pad in zip(rel_idx, self.start_pads)])

        # print(f"ds_slice: {ds_slice}")
        return self.dataset[ds_slice]

