from __future__ import annotations
import torch
import numpy as np
import json
import base64
from typing import Dict, Tuple, Optional, Any, Iterable, Literal
from pathlib import Path
from dataclasses import dataclass, field, asdict
import datapipes.analysis.hands.segment_query as segment_query
import datapipes.analysis.hands.region_wise as region_wise
import datapipes
import zlib
import skimage
import kornia
import functools
from datapipes.utils import import_resource



@dataclass(frozen=True, kw_only=True)
class BBox:
    min_h: int | torch.Tensor
    max_h: int | torch.Tensor
    min_w: int | torch.Tensor
    max_w: int | torch.Tensor

    @staticmethod
    def _to_index(value: int | torch.Tensor) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.item())
        return int(value)

    def as_slice(self) -> tuple[..., slice, slice]:
        return (
            ...,
            slice(self._to_index(self.min_h), self._to_index(self.max_h)),
            slice(self._to_index(self.min_w), self._to_index(self.max_w)),
        )

@dataclass(frozen=True, kw_only=True)
class SegmentationMap:
    segmentation_map: torch.Tensor = field(repr=False)
    _bboxes: Dict[segment_query.HandSide, BBox]
    _region_values: segment_query.RegionMap = field(default_factory=segment_query.get_region_map)

    def __getitem__(self, query: segment_query.SegmentQuery|int|Iterable[int]|slice):
        match query:
            case segment_query.SegmentQuery():
                return self[query.values] # .get_binary_mask(segmentation_map=self.segmentation_map)
            case slice():
                return self[range(*query.indices(len(self._region_values)))]
            case int():
                return self.segmentation_map == query
            case x if isinstance(x, Iterable):
                return torch.isin(self.segmentation_map, self.segmentation_map.new_tensor(x))
            case None:
                return self[1:]
            case _:
                raise TypeError(f"Unsupported type: {type(query) = }. Supported index types: {", ".join((segment_query.SegmentQuery, slice, int, Iterable[int]))}")
    
    def query(
        self,
        hand: Optional[segment_query.HandSide] = None,
        ray: Optional[segment_query.Ray] = None,
        region: Optional[segment_query.Region] = None,
        webspace: Optional[Tuple[segment_query.Ray, segment_query.Ray]] = None,
        longitudinal_index: Optional[int] = None,
        transverse_plane: Optional[segment_query.TransversePlane] = None,
        transverse_index: Optional[int] = None,
    ) -> torch.Tensor:
        return self[segment_query.SegmentQuery(hand=hand, ray=ray, region=region, webspace=webspace, longitudinal_index=longitudinal_index, transverse_plane=transverse_plane, transverse_index=transverse_index).values]
    
    @property
    def left_roi_slices(self) -> tuple[..., slice, slice]:
        return self._bboxes["left"].as_slice()
    
    @property
    def right_roi_slices(self) -> tuple[..., slice, slice]:
        return self._bboxes["right"].as_slice()
    
    def with_eroded_outer_edges(self, radius: int=5):
        kernel = (torch.from_numpy(skimage.morphology.disk(radius=radius))).to("cuda", dtype=torch.float32)
        mask = (self.segmentation_map != self._region_values.background_value).to(torch.float32)
        while mask.ndim < 4:
            mask = mask.unsqueeze(0)
        eroded = kornia.morphology.erosion(mask, kernel=kernel).to(torch.uint8)

        return SegmentationMap(
            segmentation_map=(self.segmentation_map * eroded).squeeze(0),
            _bboxes = self._bboxes.copy(),
            _region_values = self._region_values
        )
        
    
    def get_region_wise_means(self, frames: torch.Tensor) -> torch.Tensor:
        return region_wise.get_region_wise_means(frames=frames, segmentation_map=self.segmentation_map)
    
    def get_region_wise_reduction(self, frames: torch.Tensor, reduction: Literal["sum", "prod", "mean", "amax", "amin"]) -> torch.Tensor:
        return region_wise.get_region_wise_reduction(frames=frames, segmentation_map=self.segmentation_map, reduction=reduction)
    
    def reconstruct_hands_from_region_values(self, region_wise_values: torch.Tensor) -> torch.Tensor:
        """Renders the values from `region_wise_means` to the corresponding regions of `self`

        Args:
            region_wise_means (torch.Tensor): Values of each region. Shape: (b c m) or (c m)

        Returns:
            torch.Tensor: Reconstructed frames of shape (b c h w) where h, w are the height and with of `self.segmentation_map`
        """
        return region_wise.reconstruct_hands_from_region_values(region_wise_means=region_wise_values, segmentation_map=self.segmentation_map)
    
    def anonymously_reconstruct_hands_from_region_values(self, region_wise_values: torch.Tensor) -> torch.Tensor:
        """Renders the values from `region_wise_means` to the corresponding regions of `self`

        Args:
            region_wise_means (torch.Tensor): Values of each region. Shape: (b c m) or (c m)

        Returns:
            torch.Tensor: Reconstructed frames of shape (b c h w) where h, w are the height and with of `self.segmentation_map`
        """
        return region_wise.reconstruct_hands_from_region_values(region_wise_means=region_wise_values, segmentation_map=_get_default_anonymous_segmentation_map().segmentation_map)
    
    def get_region_sizes(self, output_type: Literal["values", "reconstructed"]="values"):
        region_sizes = self.get_region_wise_reduction(torch.ones_like(self.segmentation_map), reduction="sum")

        match output_type:
            case "values":
                return region_sizes
            case "reconstructed":
                return self.reconstruct_hands_from_region_values(region_sizes)
            case _:
                raise ValueError(f"Unsupported {output_type = }")
    

    @classmethod
    def compute_from_raw_frames(cls, frames: torch.Tensor) -> SegmentationMap:
        if frames.device == "cpu" and torch.cuda.is_available():
            frames = frames.to("cuda")
        if frames.dtype == torch.uint8:
            frames = frames.to(torch.float32) / 255.0

        return datapipes.analysis.hands.anatomical_segmentation.compute_anatomical_mask(frames)
    
    @classmethod
    def compute_from_raw_std_mean(cls, std: torch.Tensor, mean: torch.Tensor) -> SegmentationMap:
        return datapipes.analysis.hands.anatomical_segmentation.compute_segmentation_map_from_std_mean(std=std, mean=mean)

    @classmethod
    def load_from_json(cls, json_path: Path|str) -> SegmentationMap:
        return _load_segmentation_map_from_json_file(json_path)
    
    def save_to_json(self, json_path: Path|str):
        _save_segmentation_map_to_json_file(sm=self, out_path=json_path)

def _segmentation_map_to_json_dict(sm: SegmentationMap) -> dict[str, Any]:
    segmentation_map = sm.segmentation_map.detach().cpu()
    segmentation_map_bytes = base64.b64encode(
        # imagecodecs.zlib_encode(np.ascontiguousarray(segmentation_map.numpy()))
        zlib.compress(np.ascontiguousarray(segmentation_map.numpy()))
    ).decode("ascii")
    region_values = [({k:v for k, v in asdict(k).items() if v is not None}, v) for k, v in sm._region_values.base_map.items()]

    json_dict: dict[str, Any] = {
        "segmentation_map": segmentation_map_bytes,
        "dtype": str(segmentation_map.dtype),
        "shape": list(segmentation_map.shape),
        "bboxes": {
            hand: {
                "min_h": BBox._to_index(bbox.min_h),
                "max_h": BBox._to_index(bbox.max_h),
                "min_w": BBox._to_index(bbox.min_w),
                "max_w": BBox._to_index(bbox.max_w),
            }
            for hand, bbox in sm._bboxes.items()
        },
        "region_values": region_values,
    }
    return json_dict


def _save_segmentation_map_to_json_file(sm: SegmentationMap, out_path: Path|str):
    json_dict: dict[str, Any] = _segmentation_map_to_json_dict(sm=sm)
    out_path = Path(out_path)
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True)
    with out_path.open("w") as f:
        json.dump(
            json_dict,
            fp=f,
            separators=(",", ":"),
            indent=2
        )
def _segmentation_map_to_json_str(sm: SegmentationMap) -> str:
    json_dict: dict[str, Any] = _segmentation_map_to_json_dict(sm=sm)
    return json.dumps(
        json_dict,
        separators=(",", ":"),
        indent=2
    )

def _segmentation_map_from_json_dict(json_dict: dict[str, Any]) -> SegmentationMap:
    dtype_name = json_dict["dtype"].removeprefix("torch.")
    segmentation_map_np = np.frombuffer(
        # imagecodecs.zlib_decode(base64.b64decode(json_dict["segmentation_map"])),
        zlib.decompress(base64.b64decode(json_dict["segmentation_map"])),
        dtype=np.dtype(dtype_name),
    ).reshape(json_dict["shape"])
    segmentation_map = torch.from_numpy(segmentation_map_np.copy())
    bboxes = {
        hand: BBox(
            min_h=bbox["min_h"],
            max_h=bbox["max_h"],
            min_w=bbox["min_w"],
            max_w=bbox["max_w"],
        )
        for hand, bbox in json_dict["bboxes"].items()
    }
    region_values_json: tuple[dict, int] = json_dict["region_values"]
    region_values = {segment_query.SegmentQuery(**k): v for k, v in region_values_json}
    return SegmentationMap(segmentation_map=segmentation_map, _bboxes=bboxes, _region_values=region_values)

def _load_segmentation_map_from_json_file(path: Path|str) -> SegmentationMap:
    path = Path(path)
    with path.open("r") as f:
        payload = json.load(f)
    return _segmentation_map_from_json_dict(json_dict=payload)

def _load_segmentation_map_from_json_str(json_str: str) -> SegmentationMap:
    payload = json.loads(json_str)
    return _segmentation_map_from_json_dict(json_dict=payload)


@functools.cache
def _get_default_anonymous_segmentation_map() -> SegmentationMap:
    with import_resource.as_path(resource_relative_path="default_hands_anonymous.json") as anonymous_path:
        return SegmentationMap.load_from_json(anonymous_path)