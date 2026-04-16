import functools
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple

import einops
import torch

from datapipes.analysis.hands.geometry import normalize, project_marker_along_segment, project_point_onto_line, vec_len
from datapipes.analysis.hands.named_markers import L, add_custom_markers

import datapipes.analysis.hands.segment_query as segment_query

class HandSegments:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.segs: Dict[segment_query.SegmentQuery, torch.Tensor] = {}
        self.weights: List[float] = []
        self.biases: List[float] = []
        self._seg_ids_ordered: List[segment_query.SegmentQuery] = []
        self._current_weight: float = 1.0
        self._current_bias: float = 0.0
        self._relative_origin: torch.Tensor = torch.zeros(size=(1, 1, 2), dtype=torch.float32, device=self.device)

    def add(self, new_segs: Dict[segment_query.SegmentQuery, torch.Tensor]):
        self.segs.update(new_segs)
        self.weights.extend([self._current_weight] * len(new_segs))
        self.biases.extend([self._current_bias] * len(new_segs))
        self._seg_ids_ordered.extend(new_segs.keys())

    def __len__(self):
        return len(self.segs)

    @contextmanager
    def surface_affinity(self, w: float = 1.0, b: float = 0.0):
        self._current_weight = w
        self._current_bias = b
        yield self
        self._current_weight = 1.0
        self._current_bias = 0.0

    def relative_to(self, origin: torch.Tensor):
        rel_segments = HandSegments()
        rel_segments.__dict__ = self.__dict__.copy()
        rel_segments._relative_origin = einops.rearrange(origin, "c -> 1 1 c")
        return rel_segments

    def get_segments_tensor(self) -> torch.Tensor:
        out = torch.stack(tuple(self.segs.values())).to(device=self.device)
        out = einops.rearrange(out, "s e c -> e s c")
        return out - self._relative_origin

    def get_weights_tensor(self) -> torch.Tensor:
        return einops.rearrange(torch.tensor(self.weights, device=self.device, requires_grad=False), "c -> c 1 1 1")

    def get_biases_tensor(self) -> torch.Tensor:
        return einops.rearrange(torch.tensor(self.biases, device=self.device, requires_grad=False), "c -> c 1 1 1")

    def get_query_to_value_dict(self) -> Dict[segment_query.SegmentQuery, int]:
        return {query: index + 1 for index, query in enumerate(self._seg_ids_ordered)}

    def select(self, query: segment_query.SegmentQuery) -> Dict[segment_query.SegmentQuery, torch.Tensor]:
        return {candidate: segment for candidate, segment in self.segs.items() if query.matches(candidate)}


def smooth_transition(
    segs_to: Dict[segment_query.SegmentQuery, torch.Tensor],
    segs_from: Dict[segment_query.SegmentQuery, torch.Tensor],
) -> Dict[segment_query.SegmentQuery, torch.Tensor]:
    keys_to = tuple(segs_to.keys())
    keys_from = tuple(segs_from.keys())
    last = segs_to[keys_to[-1]]
    first = segs_from[keys_from[0]]
    common_point = last[1]
    assert torch.allclose(common_point, first[0]), "last and first points must be the same"
    to_dir = last[0] - common_point
    from_dir = first[1] - common_point
    avg_dir = 0.5 * (to_dir + from_dir)
    new_common_point = common_point + (0.5 * avg_dir)
    return (
        {k: segs_to[k] for k in keys_to[:-1]}
        | {keys_to[-1]: torch.stack((last[0], new_common_point))}
        | {keys_from[0]: torch.stack((new_common_point, first[1]))}
        | {k: segs_from[k] for k in keys_from[1:]}
    )


def create_subsegments_raw(
    n_subsegments: int,
    start_point: torch.Tensor,
    stop_point: torch.Tensor,
    query_factory: Callable[[int], segment_query.SegmentQuery],
    skip_factor: int = 1,
    skip_offset_frac: float = 0,
) -> Dict[segment_query.SegmentQuery, torch.Tensor]:
    linx = torch.linspace(start_point[0], stop_point[0], steps=skip_factor * n_subsegments + 1, device="cpu")
    liny = torch.linspace(start_point[1], stop_point[1], steps=skip_factor * n_subsegments + 1, device="cpu")
    lin = torch.stack((linx, liny), dim=-1)
    segs = [lin[i : i + 2] for i in range(int(skip_offset_frac * skip_factor), lin.shape[0] - 1, skip_factor)]
    return {query_factory(index): segment for index, segment in enumerate(segs)}


def create_subsegments_idx(
    markers_px: Dict[str, torch.Tensor],
    n_subsegments: int,
    start_idx: int,
    stop_idx: int,
    query_factory: Callable[[int], segment_query.SegmentQuery],
    skip_factor: int = 1,
    skip_offset_frac: float = 0,
) -> Dict[segment_query.SegmentQuery, torch.Tensor]:
    return create_subsegments_raw(
        n_subsegments=n_subsegments,
        start_point=markers_px[start_idx],
        stop_point=markers_px[stop_idx],
        query_factory=query_factory,
        skip_factor=skip_factor,
        skip_offset_frac=skip_offset_frac,
    )


def spread_clones_in_dir(
    thumb_dir_sign: int,
    seg_dict: Dict[segment_query.SegmentQuery, torch.Tensor],
    n_symmetric_clones: int = 1,
    spread_factor: float = 1.0,
    include_original: bool = False,
    join_start: bool = False,
    join_end: bool = False,
) -> Dict[segment_query.SegmentQuery, torch.Tensor]:
    segs: list[torch.Tensor] = []
    queries: list[segment_query.SegmentQuery] = []
    if include_original:
        for query, seg in seg_dict.items():
            segs.append(seg)
            queries.append(segment_query._with_transverse(query, "central", 0))
    n_segs = len(seg_dict)
    for order_idx in range(n_symmetric_clones):
        for segment_idx, (query, seg) in enumerate(seg_dict.items()):
            seg_dir = seg[1] - seg[0]
            unit_seg_dir = seg_dir / (seg_dir**2).sum(-1).sqrt()
            seg_normal = torch.stack((-unit_seg_dir[1], unit_seg_dir[0])) * thumb_dir_sign
            offset = seg_normal * spread_factor * (order_idx + 1)
            if join_start and segment_idx == 0:
                offset = torch.stack((torch.zeros_like(offset), offset))
            elif join_end and segment_idx == (n_segs - 1):
                offset = torch.stack((offset, torch.zeros_like(offset)))
            segs.append(seg - offset)
            queries.append(segment_query._with_transverse(query, "ulnar", order_idx + 1))
            segs.append(seg + offset)
            queries.append(segment_query._with_transverse(query, "radial", order_idx + 1))
    return {query: seg for query, seg in zip(queries, segs)}


def build_segments(markers_px: Dict[str, torch.Tensor], mask: torch.Tensor) -> HandSegments:
    segs = HandSegments()
    principal_direction = normalize(markers_px[L.middle_mcp] - markers_px[L.wrist])
    thumb_direction = normalize(markers_px[L.index_mcp] - markers_px[L.middle_mcp])
    thumb_dir_sign = (torch.stack((-principal_direction[1], principal_direction[0])) * thumb_direction).sum(-1).sign()

    create_subsegments = functools.partial(create_subsegments_idx, markers_px)
    spread_clones = functools.partial(spread_clones_in_dir, thumb_dir_sign)

    thumb_metacarp_sub_seg_n = 4
    palm_sub_seg_n = 5
    palm_skip_factor = 1
    palm_skip_offset_frac = 0.0
    proximal_phalanges_sub_seg_n = 3
    middle_phalanges_sub_seg_n = 2
    distal_phalanges_sub_seg_n = 3
    arm_sub_seg_n = 2

    index_metacarp = create_subsegments(
        palm_sub_seg_n,
        L.index_vantage_wrist,
        L.index_mcp,
        query_factory=lambda idx: segment_query._ray_segment_query("index", "metacarpal", idx),
        skip_factor=palm_skip_factor,
        skip_offset_frac=0,
    )
    middle_metacarp = create_subsegments(
        palm_sub_seg_n,
        L.middle_vantage_wrist,
        L.middle_mcp,
        query_factory=lambda idx: segment_query._ray_segment_query("middle", "metacarpal", idx),
        skip_factor=palm_skip_factor,
        skip_offset_frac=palm_skip_offset_frac,
    )
    ring_metacarp = create_subsegments(
        palm_sub_seg_n,
        L.ring_vantage_wrist,
        L.ring_mcp,
        query_factory=lambda idx: segment_query._ray_segment_query("ring", "metacarpal", idx),
        skip_factor=palm_skip_factor,
        skip_offset_frac=0,
    )
    little_metacarp = create_subsegments(
        palm_sub_seg_n,
        L.pinky_vantage_wrist,
        L.pinky_mcp,
        query_factory=lambda idx: segment_query._ray_segment_query("little", "metacarpal", idx),
        skip_factor=palm_skip_factor,
        skip_offset_frac=palm_skip_offset_frac,
    )
    thumb_metacarp = create_subsegments_raw(
        thumb_metacarp_sub_seg_n,
        markers_px[L.thumb_vantage_wrist],
        markers_px[L.thumb_mcp],
        query_factory=lambda idx: segment_query._ray_segment_query("thumb", "metacarpal", idx),
    )

    wrist_dir = normalize(markers_px[L.arm_spacer_middle_proximal] - markers_px[L.arm_spacer_middle_distal])
    wrist_cut_offset = project_point_onto_line(
        origin=markers_px[L.middle_vantage_wrist],
        vec=(markers_px[L.wrist] - markers_px[L.middle_vantage_wrist]),
        dir=wrist_dir,
    ) - markers_px[L.middle_vantage_wrist]

    thumb_carpal = create_subsegments_raw(
        arm_sub_seg_n,
        markers_px[L.thumb_vantage_wrist] + wrist_cut_offset,
        markers_px[L.thumb_vantage_wrist],
        query_factory=lambda idx: segment_query._ray_segment_query("thumb", "carpal", idx),
    )
    index_carpal = create_subsegments_raw(
        arm_sub_seg_n,
        markers_px[L.index_vantage_wrist] + wrist_cut_offset,
        markers_px[L.index_vantage_wrist],
        query_factory=lambda idx: segment_query._ray_segment_query("index", "carpal", idx),
    )
    middle_carpal = create_subsegments_raw(
        arm_sub_seg_n,
        markers_px[L.middle_vantage_wrist] + wrist_cut_offset,
        markers_px[L.middle_vantage_wrist],
        query_factory=lambda idx: segment_query._ray_segment_query("middle", "carpal", idx),
    )
    ring_carpal = create_subsegments_raw(
        arm_sub_seg_n,
        markers_px[L.ring_vantage_wrist] + wrist_cut_offset,
        markers_px[L.ring_vantage_wrist],
        query_factory=lambda idx: segment_query._ray_segment_query("ring", "carpal", idx),
    )
    little_carpal = create_subsegments_raw(
        arm_sub_seg_n,
        markers_px[L.pinky_vantage_wrist] + wrist_cut_offset,
        markers_px[L.pinky_vantage_wrist],
        query_factory=lambda idx: segment_query._ray_segment_query("little", "carpal", idx),
    )

    palm_spread_factor = 1
    with segs.surface_affinity(w=0.7, b=0.1):
        segs.add(spread_clones(smooth_transition(thumb_carpal, thumb_metacarp), spread_factor=palm_spread_factor))
        segs.add(spread_clones(smooth_transition(index_carpal, index_metacarp), spread_factor=palm_spread_factor))
        segs.add(spread_clones(smooth_transition(middle_carpal, middle_metacarp), spread_factor=palm_spread_factor))
        segs.add(spread_clones(smooth_transition(ring_carpal, ring_metacarp), spread_factor=palm_spread_factor))
        segs.add(spread_clones(smooth_transition(little_carpal, little_metacarp), spread_factor=palm_spread_factor))

    with segs.surface_affinity(w=0.7, b=0.1):
        segs.add(
            spread_clones(
                create_subsegments(
                    2,
                    L.tabertier,
                    L.tabertier_arc,
                    query_factory=segment_query._first_webspace_query,
                ),
                n_symmetric_clones=1,
                spread_factor=10,
            )
        )

    thumb_finger = (
        create_subsegments(
            middle_phalanges_sub_seg_n,
            L.thumb_mcp,
            L.thumb_ip,
            query_factory=lambda idx: segment_query._ray_segment_query("thumb", "proximal_phalanx", idx),
        )
        | create_subsegments(
            distal_phalanges_sub_seg_n,
            L.thumb_ip,
            L.thumb_extended_tip,
            query_factory=lambda idx: segment_query._ray_segment_query("thumb", "distal_phalanx", idx),
        )
    )
    index_finger = (
        create_subsegments(
            proximal_phalanges_sub_seg_n,
            L.index_mcp,
            L.index_pip,
            query_factory=lambda idx: segment_query._ray_segment_query("index", "proximal_phalanx", idx),
        )
        | create_subsegments(
            middle_phalanges_sub_seg_n,
            L.index_pip,
            L.index_dip,
            query_factory=lambda idx: segment_query._ray_segment_query("index", "middle_phalanx", idx),
        )
        | create_subsegments(
            distal_phalanges_sub_seg_n,
            L.index_dip,
            L.index_extended_tip,
            query_factory=lambda idx: segment_query._ray_segment_query("index", "distal_phalanx", idx),
        )
    )
    middle_finger = (
        create_subsegments(
            proximal_phalanges_sub_seg_n,
            L.middle_mcp,
            L.middle_pip,
            query_factory=lambda idx: segment_query._ray_segment_query("middle", "proximal_phalanx", idx),
        )
        | create_subsegments(
            middle_phalanges_sub_seg_n,
            L.middle_pip,
            L.middle_dip,
            query_factory=lambda idx: segment_query._ray_segment_query("middle", "middle_phalanx", idx),
        )
        | create_subsegments(
            distal_phalanges_sub_seg_n,
            L.middle_dip,
            L.middle_extended_tip,
            query_factory=lambda idx: segment_query._ray_segment_query("middle", "distal_phalanx", idx),
        )
    )
    ring_finger = (
        create_subsegments(
            proximal_phalanges_sub_seg_n,
            L.ring_mcp,
            L.ring_pip,
            query_factory=lambda idx: segment_query._ray_segment_query("ring", "proximal_phalanx", idx),
        )
        | create_subsegments(
            middle_phalanges_sub_seg_n,
            L.ring_pip,
            L.ring_dip,
            query_factory=lambda idx: segment_query._ray_segment_query("ring", "middle_phalanx", idx),
        )
        | create_subsegments(
            distal_phalanges_sub_seg_n,
            L.ring_dip,
            L.ring_extended_tip,
            query_factory=lambda idx: segment_query._ray_segment_query("ring", "distal_phalanx", idx),
        )
    )
    little_finger = (
        create_subsegments(
            proximal_phalanges_sub_seg_n,
            L.pinky_mcp,
            L.pinky_pip,
            query_factory=lambda idx: segment_query._ray_segment_query("little", "proximal_phalanx", idx),
        )
        | create_subsegments(
            middle_phalanges_sub_seg_n,
            L.pinky_pip,
            L.pinky_dip,
            query_factory=lambda idx: segment_query._ray_segment_query("little", "middle_phalanx", idx),
        )
        | create_subsegments(
            distal_phalanges_sub_seg_n,
            L.pinky_dip,
            L.pinky_extended_tip,
            query_factory=lambda idx: segment_query._ray_segment_query("little", "distal_phalanx", idx),
        )
    )

    spread_factor = 3

    def split_except_tip(finger: Dict[segment_query.SegmentQuery, torch.Tensor]) -> Dict[segment_query.SegmentQuery, torch.Tensor]:
        finger_tuple = tuple(finger.items())
        to_spread = {query: seg for query, seg in finger_tuple[:-1]}
        tip_query, tip_seg = finger_tuple[-1]
        spread = spread_clones(to_spread, 1, spread_factor=spread_factor, join_start=True, join_end=False)
        return spread | {segment_query._with_transverse(tip_query, "central", 0): tip_seg}

    with segs.surface_affinity(w=0.7, b=-0.05):
        segs.add(split_except_tip(thumb_finger))
        segs.add(split_except_tip(index_finger))
        segs.add(split_except_tip(middle_finger))
        segs.add(split_except_tip(ring_finger))
        segs.add(split_except_tip(little_finger))

    inter_finger_project_frac = 0.2

    def between_fingers(inter_mcp: str, inter_pip: str) -> torch.Tensor:
        return torch.stack(
            (
                markers_px[inter_mcp],
                project_marker_along_segment(start=markers_px[inter_mcp], stop=markers_px[inter_pip], frac=inter_finger_project_frac),
            )
        )

    with segs.surface_affinity(w=1.5, b=-0.1):
        segs.add({segment_query._interdigital_web_query("index", "middle"): between_fingers(inter_mcp=L.inter_mcp_index_middle, inter_pip=L.inter_pip_index_middle)})
        segs.add({segment_query._interdigital_web_query("middle", "ring"): between_fingers(inter_mcp=L.inter_mcp_middle_ring, inter_pip=L.inter_pip_middle_ring)})
        segs.add({segment_query._interdigital_web_query("ring", "little"): between_fingers(inter_mcp=L.inter_mcp_ring_pinky, inter_pip=L.inter_pip_ring_pinky)})

    return segs







@functools.cache
def get_query_to_value_dict() -> Dict[segment_query.SegmentQuery, int]:
    mask = torch.ones((1, 512, 512), dtype=torch.uint8)
    landmarks_px = torch.tensor(
        [
            [256, 420],
            [205, 356],
            [174, 316],
            [144, 278],
            [118, 242],
            [218, 318],
            [208, 248],
            [202, 188],
            [196, 130],
            [256, 304],
            [256, 226],
            [256, 158],
            [256, 96],
            [294, 314],
            [302, 248],
            [308, 192],
            [314, 140],
            [332, 330],
            [344, 278],
            [352, 232],
            [360, 190],
        ],
        dtype=torch.float32,
    )
    markers_px, _ = add_custom_markers(landmarks_px=landmarks_px, mask=mask)
    hand_segments: HandSegments = build_segments(markers_px=markers_px, mask=mask)
    base_map = hand_segments.get_query_to_value_dict()
    return base_map


@functools.cache
def get_value_to_query_dict() -> Dict[int, segment_query.SegmentQuery]:
    return {value: query for query, value in get_query_to_value_dict().items()}

