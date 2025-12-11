from dataclasses import dataclass
from typing import Tuple, Callable, List, Any, Optional
from datapipes.utils import introspection
from datapipes.nd_windows import NdValidWindow, NdAutoUnpaddingWindow
import torch
from datapipes import subbatching

from datapipes.manual_ops import ManualUnpadOp, with_manual_unpad

# @dataclass
# class ManualUnpadOp:
#     op: Callable
#     padding: Tuple[int]
#     dtype: Optional[Any]=None
#     device: Optional[Any]=None

#     def __call__(self, *args, **kwargs):
#         return self.op(*args, **kwargs)
    

# def with_manual_unpad(func: Callable, padding) -> ManualUnpadOp:
#     return ManualUnpadOp(op=func, padding=padding)

# @staticmethod
# def subbatch_in_blocks(func: Callable[[torch.Tensor], torch.Tensor], batch_size=(None, None, 256, 256)):
#     """
#     Usage example:
#     ```
#         dp = (
#             raw 
#             | Ops.bytes_to_float01_gpu
#             | Ops.subbatch_in_blocks(
#                 contrast.temporal_contrast(window_size=25),
#                 batch_size=(None, None, 256, 256),
#             )
#         )
#     ```
#     """       
#     shape_deltas, probed_output = introspection.probe_func(func)

#     dim_pads = tuple([d // 2 for d in shape_deltas])
#     dtype = probed_output.dtype
#     # device = probed_output.device

#     print(dim_pads)

#     def batched_op(frames: torch.Tensor):
#         # print(f"dim_pads: {dim_pads}")
#         window = NdValidWindow.get_padded(data=frames, padding=dim_pads)
#         unpadding = NdAutoUnpaddingWindow(data=window, padding=dim_pads)

#         results = torch.empty(size=window.shape, dtype=dtype, device=frames.device)

#         for batch_input, idx_dest in subbatching.nd_subbatch_emit_indices(unpadding, window_size=batch_size):
#             # print(idx_dest)
#             t = func(batch_input)
#             # line_val = t.mean()
#             # t[..., 0, :] = line_val
#             # t[..., :, 0] = line_val

#             results[idx_dest] = t
            
        
#         return results


#     return with_manual_unpad(func=batched_op, padding=dim_pads)


