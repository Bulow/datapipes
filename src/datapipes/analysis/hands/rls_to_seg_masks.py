from datapipes import DataPipe, datasets, Ops, contrast, plot
from datapipes.plotting import map01
import torch.nn.functional as F
import kornia.filters as k
import einops
from datapipes.manual_ops import ManualOp
from tqdm import tqdm
from datapipes import sinks
from datapipes.utils.benchmarking import MultiBlockTimer, human_readable_time
import datapipes.analysis.hands
from datapipes.analysis.hands import hand_segmentation
from datapipes.plotting.torch_colormap import TorchColormap
from datapipes.plotting import plots

from datapipes.analysis.hands import segments, named_markers
from datapipes.analysis.hands import visualization
from typing import Callable, Any, Tuple, Iterable, List, Dict, Optional

from datapipes.analysis.hands.fast_anatomical_segmentation import watch

from datapipes.analysis.hands import hand_landmarks
from typing import Dict
from contextlib import contextmanager
from datapipes.analysis.hands import fast_anatomical_segmentation, anatomical_segmentation
from pathlib import Path
from datapipes.save_datapipe import datapipe_to_rls
import torch

def compute_segmentation_masks(input_rls_path: str|Path) -> torch.Tensor:
    path: str = R"C:\Workspace\DataAnalysis\compression_paper\input_datasets\hands.rls"
    dataset = datasets.load_dataset(path, cache_strategy="no_caching", switch_wh_metadata_read_order=True)

    dp = (
        DataPipe(dataset)
        | Ops.bytes_to_float01_gpu 
    )

    seg_dp = dp | anatomical_segmentation.segmentation_mask_op(frames_per_mask=256) #| contrast.get_moving_mean(window_size=32) # | fast_anatomical_segmentation.distinct_colors_op()
    # print(f"{seg_dp.shape = }")

    masks = sinks.accumulate(seg_dp, slice(None), batch_size=8)
    return masks

def create_segmentation_masks(input_rls_path: str|Path, output_path: str|Path):
    masks = compute_segmentation_masks(input_rls_path=input_rls_path)

    datapipe_to_rls(DataPipe(masks), out_path=output_path)

