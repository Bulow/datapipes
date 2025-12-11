import torch
from typing import Callable, Any, List, Tuple, Iterable
from datapipes.ops import Ops
from datapipes.manual_ops import ManualUnpadOp, with_manual_unpad
# from datapipes.data_pipe import DataPipe

# def probe_op(op: Callable, td=512, sd=128) -> Tuple[List[int], Any]:
#         # TODO: Support rearrange, different number of dimensions etc.

#         probe_shape = [td, 1, sd, sd]
#         test_data = torch.empty(probe_shape, device="meta")
#         probe_output = op(test_data)

#         shape_deltas = [d_before - d_after for d_before, d_after in zip(probe_shape, probe_output.shape)]

        
#         return shape_deltas, probe_output

FramesType = Iterable[Any]
OpType = Callable[[FramesType], FramesType]
class AbstractDataPipe:
    def __init__(self, ops: List[OpType]):
        self.ops = ops

    def then(self, new_op: FramesType) -> "AbstractDataPipe":
        return AbstractDataPipe(ops=[*self.ops, new_op])

    def __or__(self, other: FramesType) -> "AbstractDataPipe":
        return self.then(other)
    
    def __call__(self, frames: FramesType) -> FramesType:
        for op in self.ops:
            frames = op(frames)
        return frames

def _get_relative_slice(in_size1: int, out_size1: int, in_size2: int, out_size2: int) -> slice:
    if in_size1 == out_size1 and in_size2 == out_size2:
        # p = 0
        # s = 1
        return slice(None)
    else:
        if out_size1 == out_size2:
            raise ValueError("Output slices are equal in this dimension. Sizes must be different to solve for slice (prevents division by zero)")
        
        p = (-in_size1*out_size2 + in_size2*out_size1)/(2*(out_size1 - out_size2))
        s = (in_size1 - in_size2)/(out_size1 - out_size2)

        return slice(round(p), round(-p), round(s)) # TODO: find better method than rounding

def get_equivalent_relative_slice_op(func: Callable[[torch.Tensor], torch.Tensor], in_shape1=[1024, 1, 1024, 1024], in_shape2=None) -> Tuple[slice]:
    if in_shape2 is None:
        in_shape2 = [s // 2 if s > 1 else s for s in in_shape1]
    def test_shape(in_shape):
        t_in = torch.empty(size=in_shape, device="meta")
        t_out = func(t_in)
        out_shape = t_out.shape
        return in_shape, out_shape


    X1, Y1 = test_shape(in_shape1)

    X2, Y2 = test_shape(in_shape2)

    slices = tuple([_get_relative_slice(*args) for args in zip(X1, Y1, X2, Y2)])

    return slices

from datapipes.manual_ops import ManualOp
def probe_segments(segments: List[Callable[[torch.Tensor], torch.Tensor]]):
    slice_ops = []
    for op in segments:
        match op:
            case Ops.bytes_to_float01_gpu | Ops.float01_to_bytes_cpu:
                pass
            case ManualOp():
                slice_ops.append(op.equivalent_slicing_op)
            case _:
                slice_ops.append(get_equivalent_relative_slice_op(op))

    return slice_ops


def probe_segments_with_input_shape(segments: List[Callable[[torch.Tensor], torch.Tensor]], in_shape):
    # print(in_shape)
    t = torch.empty(size=in_shape, device="meta")
    slice_ops = []
    for op in segments:
        match op:
            case Ops.bytes_to_float01_gpu | Ops.float01_to_bytes_cpu | Ops.numpy | Ops.pytorch | Ops.matlab_to_py | Ops.py_to_matlab | Ops.to | Ops.cpu | Ops.gpu:
                pass
            case ManualOp():
                new_slc = op.equivalent_slicing_op
                slice_ops.append(new_slc)
                t = t[new_slc]
                # print(f"t.shape: {t.shape}")
            case _:
                new_slc = get_equivalent_relative_slice_op(op, in_shape1=in_shape)
                slice_ops.append(new_slc)
                t = t[new_slc]
                # print(f"t.shape: {t.shape}")

    out_shape = t.shape
    return slice_ops, out_shape

def pretty_slices(t):
    def fmt(item):
        if isinstance(item, slice):
            start = "" if item.start is None else item.start
            stop  = "" if item.stop  is None else item.stop
            step  = "" if item.step  is None or item.step == 1 else item.step

            # Case: full slice ":"
            if item.start is None and item.stop is None and item.step is None:
                return ":"

            # Case: no step → "start:stop"
            if item.step is None:
                return f"{start}:{stop}"

            # Case: full triple slice
            return f"{start}:{stop}:{step}"

        elif item is None:
            return "None"

        else:
            return str(item)

    inner = ", ".join(fmt(x) for x in t)
    return f"[{inner}]"

def inspect_segments(segments):
    slc = probe_segments(segments=segments)
    print("Equivalent slices:")
    for s in slc:
        print(f"\t{pretty_slices(s)}")