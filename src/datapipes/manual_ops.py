from dataclasses import dataclass
from typing import Tuple, Callable, List, Any, Optional
import torch
import functools


@dataclass
class ManualUnpadOp:
    op: Callable
    padding: Tuple[int]
    dtype: Optional[Any]=None
    device: Optional[Any]=None

    def __call__(self, *args, **kwargs):
        return self.op(*args, **kwargs)
    

def with_manual_unpad(func: Callable, padding) -> ManualUnpadOp:
    return functools.wraps(func)(ManualUnpadOp(op=func, padding=padding))




@dataclass
class ManualOp:
    op: Callable
    equivalent_slicing_op: Tuple[slice]

    def __call__(self, *args, **kwargs):
        return self.op(*args, **kwargs)
    

def with_manual_op(func: Callable, equivalent_slicing_op=(slice(None), slice(None), slice(None), slice(None))) -> ManualUnpadOp:
    return functools.wraps(func)(ManualOp(op=func, equivalent_slicing_op=equivalent_slicing_op))