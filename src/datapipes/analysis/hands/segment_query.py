#%%

import functools
from contextlib import contextmanager
from dataclasses import dataclass, replace, asdict, field, fields
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple

import einops
import torch

from datapipes.analysis.hands.geometry import normalize, project_marker_along_segment, project_point_onto_line, vec_len
from datapipes.analysis.hands.named_markers import L, add_custom_markers
# import datapipes.analysis.hands.segments as segments
import datapipes

Ray = Literal["thumb", "index", "middle", "ring", "little"]
Region = Literal[
    "palm",
    "finger",
    "carpal",
    "metacarpal",
    "proximal_phalanx",
    "middle_phalanx",
    "distal_phalanx",
    "first_webspace",
    "interdigital_web",
]
TransversePlane = Literal["radial", "central", "ulnar"]
HandSide = Literal["left", "right", "both"]

RAY_ORDER: dict[Ray, int] = {
    "thumb": 0,
    "index": 1,
    "middle": 2,
    "ring": 3,
    "little": 4,
}
REGION_ORDER: dict[Region, int] = {
    "palm": 0,
    "finger": 1,
    "carpal": 2,
    "metacarpal": 3,
    "proximal_phalanx": 4,
    "middle_phalanx": 5,
    "distal_phalanx": 6,
    "first_webspace": 7,
    "interdigital_web": 8,
}
TRANSVERSE_ORDER: dict[TransversePlane, int] = {
    "radial": 0,
    "central": 1,
    "ulnar": 2,
}
DISPLAY_RAY: dict[Ray, str] = {
    "thumb": "Thumb",
    "index": "Index",
    "middle": "Middle",
    "ring": "Ring",
    "little": "Little",
}
DISPLAY_REGION: dict[Region, str] = {
    "palm": "palm",
    "finger": "finger",
    "carpal": "carpal",
    "metacarpal": "metacarpal",
    "proximal_phalanx": "proximal phalanx",
    "middle_phalanx": "middle phalanx",
    "distal_phalanx": "distal phalanx",
    "first_webspace": "first webspace",
    "interdigital_web": "interdigital web",
}
SPECIAL_REGION_MATCHES: dict[Region, set[Region]] = {
    "finger": {"proximal_phalanx", "middle_phalanx", "distal_phalanx"},
    "palm": {"carpal", "metacarpal", "first_webspace", "interdigital_web"},
}

def _expected_actual_suffix(expected, actual) -> str:
    return f"; expected {expected}, actual {actual!r}"

def _is_iterable_query_value(value) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes))


def _normalize_multi_value(value, *, key=None):
    if value is None or not _is_iterable_query_value(value):
        return value
    normalized = tuple(value)
    if len(normalized) == 0:
        raise ValueError(f"iterable query inputs must not be empty{_expected_actual_suffix('a non-empty iterable', normalized)}")
    return tuple(sorted(normalized, key=key)) if key is not None else normalized


def _normalize_single_webspace(webspace):
    if not _is_iterable_query_value(webspace):
        return None
    rays = tuple(webspace)
    if len(rays) != 2 or not all(ray in RAY_ORDER for ray in rays):
        return None
    ordered = tuple(sorted(rays, key=lambda ray: RAY_ORDER[ray]))
    if ordered[0] == ordered[1]:
        raise ValueError(f"webspace rays must be distinct{_expected_actual_suffix('two distinct rays', webspace)}")
    return ordered


def _normalize_webspace(webspace):
    if webspace is None:
        return None
    normalized_single = _normalize_single_webspace(webspace)
    if normalized_single is not None:
        return normalized_single
    if not _is_iterable_query_value(webspace):
        raise ValueError(f"webspace must contain exactly two rays{_expected_actual_suffix('exactly two rays', webspace)}")
    normalized = tuple(_normalize_webspace(candidate) for candidate in webspace)
    if len(normalized) == 0:
        raise ValueError(f"iterable query inputs must not be empty{_expected_actual_suffix('a non-empty iterable', normalized)}")
    return tuple(sorted(normalized, key=lambda pair: tuple(RAY_ORDER[ray] for ray in pair)))


def _is_multi_selector(selector) -> bool:
    return isinstance(selector, tuple)


def _is_multi_webspace_selector(selector) -> bool:
    return isinstance(selector, tuple) and (
        len(selector) == 0 or not all(isinstance(item, str) for item in selector)
    )


def _selector_matches(selector, candidate, *, multi=None) -> bool:
    if selector is None:
        return True
    if multi(selector) if multi is not None else _is_multi_selector(selector):
        return candidate in selector
    return selector == candidate


@dataclass(frozen=True, kw_only=True)
class SegmentQuery:
    hand: Optional[HandSide|Iterable[HandSide]] = None
    ray: Optional[Ray|Iterable[Ray]] = None
    region: Optional[Region|Iterable[Region]] = None
    webspace: Optional[Tuple[Ray, Ray]|Iterable[Tuple[Ray, Ray]]] = None
    longitudinal_index: Optional[int|Iterable[int]] = None
    transverse_plane: Optional[TransversePlane|Iterable[TransversePlane]] = None
    transverse_index: Optional[int|Iterable[int]] = None

    def __post_init__(self):
        object.__setattr__(self, "hand", _normalize_multi_value(self.hand))
        object.__setattr__(self, "ray", _normalize_multi_value(self.ray, key=lambda ray: RAY_ORDER[ray]))
        object.__setattr__(self, "region", _normalize_multi_value(self.region, key=lambda region: REGION_ORDER[region]))
        object.__setattr__(self, "webspace", _normalize_webspace(self.webspace))
        object.__setattr__(self, "longitudinal_index", _normalize_multi_value(self.longitudinal_index))
        object.__setattr__(self, "transverse_plane", _normalize_multi_value(self.transverse_plane, key=lambda plane: TRANSVERSE_ORDER[plane]))
        object.__setattr__(self, "transverse_index", _normalize_multi_value(self.transverse_index))

        if self.longitudinal_index is not None:
            values = self.longitudinal_index if _is_multi_selector(self.longitudinal_index) else (self.longitudinal_index,)
            if any(index < 0 for index in values):
                raise ValueError(f"longitudinal_index must be >= 0{_expected_actual_suffix('>= 0', self.longitudinal_index)}")
        if self.transverse_index is not None:
            values = self.transverse_index if _is_multi_selector(self.transverse_index) else (self.transverse_index,)
            if any(index < 0 for index in values):
                raise ValueError(f"transverse_index must be >= 0{_expected_actual_suffix('>= 0', self.transverse_index)}")

    def is_fully_specified(self) -> bool:
        if (
            self.hand is None
            or self.region is None
            or _is_multi_selector(self.hand)
            or _is_multi_selector(self.region)
            or self.region in SPECIAL_REGION_MATCHES
        ):
            return False
        if self.region == "interdigital_web":
            return (
                self.webspace is not None
                and not _is_multi_webspace_selector(self.webspace)
                and self.ray is None
                and self.longitudinal_index is None
                and self.transverse_plane is None
                and self.transverse_index is None
            )
        if self.region == "first_webspace":
            return (
                self.webspace == ("thumb", "index")
                and self.ray is None
                and self.longitudinal_index is not None
                and self.transverse_plane is not None
                and self.transverse_index is not None
            )
        return (
            self.ray is not None
            and self.webspace is None
            and self.longitudinal_index is not None
            and self.transverse_plane is not None
            and self.transverse_index is not None
        )

    def matches(self, other: "SegmentQuery") -> bool:
        region_matches = self.region is None
        if self.region is not None:
            if _is_multi_selector(self.region):
                region_matches = any(
                    other.region in SPECIAL_REGION_MATCHES[region] if region in SPECIAL_REGION_MATCHES else region == other.region
                    for region in self.region
                )
            elif self.region in SPECIAL_REGION_MATCHES:
                region_matches = other.region in SPECIAL_REGION_MATCHES[self.region]
            else:
                region_matches = self.region == other.region

        hand_matches = self.hand is None
        if self.hand is not None:
            if _is_multi_selector(self.hand):
                hand_matches = "both" in self.hand or other.hand in self.hand
            else:
                hand_matches = self.hand == "both" or self.hand == other.hand
        return all(
            _selector_matches(selector, candidate, multi=multi)
            for selector, candidate, multi in (
                (self.ray, other.ray, _is_multi_selector),
                (self.webspace, other.webspace, _is_multi_webspace_selector),
                (self.longitudinal_index, other.longitudinal_index, _is_multi_selector),
                (self.transverse_plane, other.transverse_plane, _is_multi_selector),
                (self.transverse_index, other.transverse_index, _is_multi_selector),
            )
        ) and region_matches and hand_matches

    @property
    def values(self) -> Iterable[int]:
        rm = get_region_map()
        return tuple(value for candidate, value in rm.both.items() if self.matches(candidate))

    def __str__(self) -> str:
        if any((
            _is_multi_selector(self.hand),
            _is_multi_selector(self.ray),
            _is_multi_selector(self.region),
            _is_multi_webspace_selector(self.webspace),
            _is_multi_selector(self.longitudinal_index),
            _is_multi_selector(self.transverse_plane),
            _is_multi_selector(self.transverse_index),
        )):
            parts = []
            for field in fields(self):
                value = getattr(self, field.name)
                if value is not None:
                    parts.append(f"{field.name}={value}")
            return f"SegmentQuery({', '.join(parts)})"

        side_prefix = ""
        if self.hand is not None:
            side_prefix = "L " if self.hand == "left" else "R "

        if self.region == "interdigital_web" and self.webspace is not None:
            base = f"{side_prefix}{self.webspace[0]}-{self.webspace[1]} web"
            return base if self.is_fully_specified() else f"{base} regions"

        if self.region == "first_webspace":
            parts = [f"{side_prefix}thumb-index web".strip()]
        elif self.ray is not None and self.region is not None:
            parts = [f"{side_prefix}{DISPLAY_RAY[self.ray]} {DISPLAY_REGION[self.region]}".strip()]
        elif self.ray is not None:
            return f"{side_prefix}{self.ray} ray".strip()
        elif self.region is not None:
            region_label = DISPLAY_REGION[self.region]
            if self.transverse_plane is not None:
                prefix = "central" if self.transverse_plane == "central" else self.transverse_plane
                return f"{side_prefix}{prefix} {region_label}".strip()
            return f"{side_prefix}{region_label}".strip()
        else:
            return side_prefix.strip() or "Any segment"

        if self.longitudinal_index is not None:
            parts.append(f"s{self.longitudinal_index}")
        if self.transverse_plane is not None:
            if self.transverse_plane == "central":
                parts.append("c")
            else:
                lane = self.transverse_plane
                if self.transverse_index is not None:
                    lane += str(self.transverse_index)
                parts.append(lane)
        if self.transverse_plane is None and self.transverse_index is not None:
            parts.append(f"t{self.transverse_index}")
        return ", ".join(parts)


def _segment_query_sort_key(query: SegmentQuery) -> tuple:
    webspace = query.webspace or ()
    webspace_key = tuple(RAY_ORDER[ray] for ray in webspace)
    return (
        0 if query.hand == "left" else 1 if query.hand == "right" else 99,
        REGION_ORDER.get(query.region, 99),
        RAY_ORDER.get(query.ray, 99),
        webspace_key,
        query.longitudinal_index if query.longitudinal_index is not None else 99,
        TRANSVERSE_ORDER.get(query.transverse_plane, 99),
        query.transverse_index if query.transverse_index is not None else 99,
    )


def _with_transverse(query: SegmentQuery, plane: TransversePlane, index: int) -> SegmentQuery:
    return replace(query, transverse_plane=plane, transverse_index=index)


def _ray_segment_query(ray: Ray, region: Region, longitudinal_index: int) -> SegmentQuery:
    return SegmentQuery(ray=ray, region=region, longitudinal_index=longitudinal_index)


def _first_webspace_query(longitudinal_index: int) -> SegmentQuery:
    return SegmentQuery(region="first_webspace", webspace=("thumb", "index"), longitudinal_index=longitudinal_index)


def _interdigital_web_query(ray0: Ray, ray1: Ray) -> SegmentQuery:
    return SegmentQuery(region="interdigital_web", webspace=(ray0, ray1))

def _with_hand(query: SegmentQuery, hand: HandSide) -> SegmentQuery:
    return replace(query, hand=hand)

@dataclass(kw_only=True)
class RegionMap:
    background_value: Optional[int] = field(
        default=0,
    )
    base_map: Optional[dict[SegmentQuery, int]] = field(
        default=None,
    )
    def __post_init__(self):
        if self.base_map is None:
            self.base_map = datapipes.analysis.hands.segments.get_query_to_value_dict()
        self.left: dict[SegmentQuery, int] = {_with_hand(query, "left"): value for query, value in self.base_map.items()}
        offset = len(self.base_map) + 1
        self.right: dict[SegmentQuery, int] = {_with_hand(query, "right"): value + offset for query, value in self.base_map.items()}
        

    @property
    def both(self) -> dict[SegmentQuery, int]:
        return self.left | self.right
    
    @property
    def num_segments_per_hand(self) -> int:
        return len(self.base_map)
    
    def __len__(self) -> int:
        return len(self.left) + len(self.right)
    
@functools.cache
def get_region_map() -> RegionMap:
    return RegionMap()



# def query_segmentation_maps(segmentation_maps: torch.Tensor, query: SegmentQuery) -> torch.BoolTensor:
#     # segmentation_maps: (b c h w) or (c h w)
#     if not isinstance(segmentation_maps, torch.Tensor):
#         raise TypeError("segmentation_maps must be a torch.Tensor")

#     matching_ids = sorted(query.resolve_ids())
#     if len(matching_ids) == 0:
#         return torch.zeros_like(segmentation_maps, dtype=torch.bool)

#     if not torch.is_floating_point(segmentation_maps) and not segmentation_maps.dtype.is_complex:
#         maps = segmentation_maps
#     else:
#         maps = segmentation_maps.to(dtype=torch.int64)

#     if len(matching_ids) == 1:
#         return maps == matching_ids[0]

#     ids = torch.tensor(matching_ids, device=maps.device, dtype=maps.dtype)
#     return torch.isin(maps, ids)


