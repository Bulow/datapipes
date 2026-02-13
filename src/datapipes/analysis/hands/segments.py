import torch
import einops
import itertools
import functools
from dataclasses import dataclass, field

from typing import Dict, Any, Callable, Tuple, List, Optional, Iterable
import rich

from datapipes.analysis.hands.geometry import avg_dir, normalize, project_marker_along_segment, project_point_onto_line, vec_len, get_transition_points_along_line
from datapipes.analysis.hands.named_markers import add_custom_markers, L

from contextlib import contextmanager

import plotly.graph_objects as go
import plotly.express as px

class HandSegments:

    def __init__(self):
        self.segs: Dict[str, torch.Tensor] = {}
        self.weights: List[float] = []
        self.biases: List[float] = []

        self._current_weight: float = 1.0
        self._current_bias: float = 0.0
        self._relative_origin: torch.Tensor = torch.zeros(size=(1, 1, 2), dtype=torch.float32, device="cuda")

    def add(self, new_segs: Dict[str, torch.Tensor]):
        self.segs.update(new_segs)
        self.weights.extend([self._current_weight] * len(new_segs))
        self.biases.extend([self._current_bias] * len(new_segs))

    def __len__(self):
        return len(self.segs)
    

    @contextmanager
    def surface_affinity(self, w: float=1.0, b: float=0.0):
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
        out = torch.stack(tuple(self.segs.values())).to("cuda")
        out = einops.rearrange(out, "s e c -> e s c")
        return out - self._relative_origin
    
    def get_weights_tensor(self) -> torch.Tensor:
        return einops.rearrange(torch.tensor(self.weights, device="cuda", requires_grad=False), "c -> c 1 1 1")
    
    def get_biases_tensor(self) -> torch.Tensor:
        return einops.rearrange(torch.tensor(self.biases, device="cuda", requires_grad=False), "c -> c 1 1 1")
    
    def get_name_to_value_dict(self) -> Dict[str, int]:
        return {"background": 0} | {name:i + 1 for i, name in enumerate(self.segs.keys())}

def smooth_transition(segs_to: Dict[str, torch.Tensor], segs_from: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys_to = tuple(s for s in segs_to.keys())
    keys_from = tuple(s for s in segs_from.keys())

    last = segs_to[keys_to[-1]]
    first = segs_from[keys_from[0]]

    common_point = last[1]
    assert torch.allclose(common_point, first[0]), "last and first points must be the same"

    to_dir = last[0] - common_point
    from_dir = first[1] - common_point

    avg_dir = (0.5 * (to_dir + from_dir))

    new_common_point = common_point + (0.5 * avg_dir)

    out = ({}
        | {k:segs_to[k] for k in keys_to[:-1]}
        | {keys_to[-1]:torch.stack((last[0], new_common_point))}
        | {keys_from[0]:torch.stack((new_common_point, first[1]))}
        | {k:segs_from[k] for k in keys_from[1:]}
    )
    return out


def create_subsegments_raw(n_subsegments: int, start_point: torch.Tensor, stop_point: torch.Tensor, base_name: str="segment", skip_factor: int=1, skip_offset_frac: float=0) -> Dict[str, torch.Tensor]:
    linx = torch.linspace(start_point[0], stop_point[0], steps=skip_factor * n_subsegments + 1, device="cuda")
    liny = torch.linspace(start_point[1], stop_point[1], steps=skip_factor * n_subsegments + 1, device="cuda")
    lin = torch.stack((linx, liny), dim=-1)
    segs = [lin[i:i+2] for i in range(int(skip_offset_frac * skip_factor), lin.shape[0] - 1, skip_factor)]
    # print(f"{segs = }")
    names = [f"{base_name}_d{i}" for i in range(n_subsegments)]
    # print(names)
    segs_dict = {n:s for n, s in zip(names, segs)}
    return segs_dict

def create_subsegments_idx(markers_px, n_subsegments: int, start_idx: int, stop_idx: int, base_name: str="segment", skip_factor: int=1, skip_offset_frac: float=0) -> Dict[str, torch.Tensor]:
        return create_subsegments_raw(n_subsegments=n_subsegments, start_point=markers_px[start_idx], stop_point=markers_px[stop_idx], base_name=base_name, skip_factor=skip_factor, skip_offset_frac=skip_offset_frac)

def spread_clones_in_dir(thumb_dir_sign: int, seg_dict: Dict[str, torch.Tensor], n_symmetric_clones: int=1, spread_factor: float=1.0, include_original: bool=False, join_start: bool = False, join_end: bool = False) -> Dict[str, torch.Tensor]:
    segs = []
    names = []

    if include_original:
        segs.extend(seg_dict)
        names.extend([f"{name}_c" for name in seg_dict.keys()])
    # print(f"{segs = }")
    n_segs = len(seg_dict)
    for order_idx in range(n_symmetric_clones):
        for segment_idx, (name, seg) in enumerate(seg_dict.items()):
            seg_dir = seg[1] - seg[0]
            unit_seg_dir = (seg_dir / (seg_dir ** 2).sum(-1).sqrt())
            seg_normal = torch.stack((-unit_seg_dir[1], unit_seg_dir[0])) * thumb_dir_sign # thumb_dir_sign keeps order consistent between hands
            offset = (seg_normal * spread_factor * (order_idx + 1))

            if join_start and segment_idx == 0:
                offset = torch.stack((torch.zeros_like(offset), offset))
            elif join_end and segment_idx == (n_segs - 1):
                offset = torch.stack((offset, torch.zeros_like(offset)))

            segs.append(seg - offset)
            names.append(f"{name}_u{order_idx + 1}")

            segs.append(seg + offset)
            names.append(f"{name}_r{order_idx + 1}")
    segs_dict = {n:s for n, s in zip(names, segs)}
    return segs_dict




def build_segments(markers_px: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    out: torch.Tensor (segment start_stop coord2D)
    """

    # -------------------------------
    # Setup
    # -------------------------------
    _, h, w = mask.shape
    # landmarks_px = landmarks_px.to(device="cuda", dtype=torch.float32)
    # markers_px, markers_idx = add_custom_markers(landmarks_px, mask=mask)

    segs = HandSegments()
    
    principal_direction = normalize(markers_px[L.middle_mcp] - markers_px[L.wrist])
    thumb_direction = normalize(markers_px[L.index_mcp] - markers_px[L.middle_mcp])
    thumb_dir_sign = (torch.stack((-principal_direction[1], principal_direction[0])) * thumb_direction).sum(-1).sign()
    
    prototypical_hand_scale_factor = 60 # Arbitrarily determined normalization factor
    hand_scale_factor = torch.tensor((
        vec_len(markers_px[L.index_mcp] - markers_px[L.middle_mcp]),
        vec_len(markers_px[L.middle_mcp] - markers_px[L.ring_mcp]),
        vec_len(markers_px[L.ring_mcp] - markers_px[L.pinky_mcp]),
    )).mean() / prototypical_hand_scale_factor

    # Curry segment helpers to current markers
    create_subsegments = functools.partial(create_subsegments_idx, markers_px)
    spread_clones = functools.partial(spread_clones_in_dir, thumb_dir_sign)

    # print(f"{principal_direction = }")
    # print(f"{thumb_direction = }")
    # print(f"{thumb_dir_sign = }")
    # print(f"{hand_scale_factor = }")

    # -------------------------------
    # Config
    # -------------------------------

    thumb_metacarp_sub_seg_n = 4
    palm_sub_seg_n = 5
    palm_skip_factor = 1
    palm_skip_offset_frac = 0.0

    proximal_phalanges_sub_seg_n = 3
    intermediate_phalanges_sub_seg_n = 2
    distal_phalanges_sub_seg_n = 3

    arm_sub_seg_n = 5

    # -------------------------------
    # Metacarps
    # -------------------------------
    index_metacarp = create_subsegments(palm_sub_seg_n, L.index_vantage_wrist, L.index_mcp, base_name="index_metacarp", skip_factor=palm_skip_factor, skip_offset_frac=0)
    middle_metacarp = create_subsegments(palm_sub_seg_n, L.middle_vantage_wrist, L.middle_mcp, base_name="middle_metacarp", skip_factor=palm_skip_factor, skip_offset_frac=palm_skip_offset_frac)
    ring_metacarp = (create_subsegments(palm_sub_seg_n, L.ring_vantage_wrist, L.ring_mcp, base_name="ring_metacarp", skip_factor=palm_skip_factor, skip_offset_frac=0))
    pinky_metacarp = (create_subsegments(palm_sub_seg_n, L.pinky_vantage_wrist, L.pinky_mcp, base_name="pinky_metacarp", skip_factor=palm_skip_factor, skip_offset_frac=palm_skip_offset_frac))
    
    thumb_metacarp = create_subsegments_raw(
        thumb_metacarp_sub_seg_n, 
        markers_px[L.thumb_vantage_wrist], 
        markers_px[L.thumb_mcp], 
        base_name="thumb_metacarp"
    )

    palm_spread_factor = 1

    # -------------------------------
    # Wrist
    # -------------------------------

    extend_units = 200 * ((markers_px[L.arm_spacer_index_distal] - markers_px[L.arm_spacer_index_proximal]) * principal_direction).sum(-1).sign() * -1

    thumb_wrist = create_subsegments_raw(
        arm_sub_seg_n, 
        markers_px[L.thumb_vantage_wrist] + normalize(markers_px[L.arm_spacer_thumb_distal] - markers_px[L.arm_spacer_thumb_proximal]) * extend_units * hand_scale_factor, 
        markers_px[L.thumb_vantage_wrist], 
        base_name="thumb_wrist"
    )

    index_wrist = create_subsegments_raw(
        arm_sub_seg_n, 
        markers_px[L.index_vantage_wrist] + normalize(markers_px[L.arm_spacer_index_distal] - markers_px[L.arm_spacer_index_proximal]) * extend_units * hand_scale_factor, 
        markers_px[L.index_vantage_wrist],
        base_name="index_wrist"
    )

    middle_wrist = create_subsegments_raw(
        arm_sub_seg_n, 
        markers_px[L.middle_vantage_wrist] + normalize(markers_px[L.arm_spacer_middle_distal] - markers_px[L.arm_spacer_middle_proximal]) * extend_units * hand_scale_factor, 
        markers_px[L.middle_vantage_wrist], 
        base_name="middle_wrist"
    )

    ring_wrist = create_subsegments_raw(
        arm_sub_seg_n, 
        markers_px[L.ring_vantage_wrist] + normalize(markers_px[L.arm_spacer_ring_distal] - markers_px[L.arm_spacer_ring_proximal]) * extend_units * hand_scale_factor, 
        markers_px[L.ring_vantage_wrist], 
        base_name="ring_wrist"
    )

    pinky_wrist = create_subsegments_raw(
        arm_sub_seg_n, 
        markers_px[L.pinky_vantage_wrist] + normalize(markers_px[L.arm_spacer_pinky_distal] - markers_px[L.arm_spacer_pinky_proximal]) * extend_units * hand_scale_factor, 
        markers_px[L.pinky_vantage_wrist], 
        base_name="pinky_wrist"
    )


    # -------------------------------
    # Wrist -> metacarps
    # -------------------------------
    with segs.surface_affinity(w=0.7, b=0.1):

        segs.add(spread_clones(smooth_transition(thumb_wrist, thumb_metacarp), spread_factor=palm_spread_factor))

        segs.add(spread_clones(smooth_transition(index_wrist, index_metacarp), spread_factor=palm_spread_factor))
        segs.add(spread_clones(smooth_transition(middle_wrist, middle_metacarp), spread_factor=palm_spread_factor))
        segs.add(spread_clones(smooth_transition(ring_wrist, ring_metacarp), spread_factor=palm_spread_factor))
        segs.add(spread_clones(smooth_transition(pinky_wrist, pinky_metacarp), spread_factor=palm_spread_factor))


    # -------------------------------
    # Tabertier
    # -------------------------------
    with segs.surface_affinity(w=0.7, b=0.1):

        segs.add(spread_clones(create_subsegments(2, L.tabertier, L.tabertier_arc, base_name="tabertier"), n_symmetric_clones=1, spread_factor=10))
    
    thumb_finger = ({}
        | create_subsegments(intermediate_phalanges_sub_seg_n, L.thumb_mcp, L.thumb_ip, base_name="thumb_proximal_phalanx")
        | create_subsegments(distal_phalanges_sub_seg_n, L.thumb_ip, L.thumb_extended_tip, base_name="thumb_distal_phalanx")
    )

    index_finger = ({}
        | create_subsegments(proximal_phalanges_sub_seg_n, L.index_mcp, L.index_pip, base_name="index_proximal_phalanx")
        | create_subsegments(intermediate_phalanges_sub_seg_n, L.index_pip, L.index_dip, base_name="index_intermediate_phalanx")
        | create_subsegments(distal_phalanges_sub_seg_n, L.index_dip, L.index_extended_tip, base_name="index_distal_phalanx")
    )

    middle_finger = ({}
        | create_subsegments(proximal_phalanges_sub_seg_n, L.middle_mcp, L.middle_pip, base_name="middle_proximal_phalanx")
        | create_subsegments(intermediate_phalanges_sub_seg_n, L.middle_pip, L.middle_dip, base_name="middle_intermediate_phalanx")
        | create_subsegments(distal_phalanges_sub_seg_n, L.middle_dip, L.middle_extended_tip, base_name="middle_distal_phalanx")
    )

    ring_finger = ({}
        | create_subsegments(proximal_phalanges_sub_seg_n, L.ring_mcp, L.ring_pip, base_name="ring_proximal_phalanx")
        | create_subsegments(intermediate_phalanges_sub_seg_n, L.ring_pip, L.ring_dip, base_name="ring_intermediate_phalanx")
        | create_subsegments(distal_phalanges_sub_seg_n, L.ring_dip, L.ring_extended_tip, base_name="ring_distal_phalanx")
    )

    pinky_finger = ({}
        | create_subsegments(proximal_phalanges_sub_seg_n, L.pinky_mcp, L.pinky_pip, base_name="pinky_proximal_phalanx")
        | create_subsegments(intermediate_phalanges_sub_seg_n, L.pinky_pip, L.pinky_dip, base_name="pinky_intermediate_phalanx")
        | create_subsegments(distal_phalanges_sub_seg_n, L.pinky_dip, L.pinky_extended_tip, base_name="pinky_distal_phalanx")
    )
    
    spread_factor=3
    # print(f"Before adding fingers: {len(segs) = }")
    def split_except_tip(finger: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        finger_tuple = tuple(finger.items())
        to_spread = {k:v for k, v in finger_tuple[:-1]}
        tip_tuple = finger_tuple[-1]
        tip = {tip_tuple[0]:tip_tuple[1]}
        spread = spread_clones(to_spread, 1, spread_factor=spread_factor, join_start=True, join_end=False)
        return (spread | tip)
    
    # -------------------------------
    # Fingers
    # -------------------------------
    with segs.surface_affinity(w=0.7, b=-0.05):
        segs.add(split_except_tip(thumb_finger))
        segs.add(split_except_tip(index_finger))
        segs.add(split_except_tip(middle_finger))
        segs.add(split_except_tip(ring_finger))
        segs.add(split_except_tip(pinky_finger))

    inter_finger_project_frac = 0.2
    def between_fingers(inter_mcp: str, inter_pip: str) -> Dict[str, torch.Tensor]:
        return torch.stack((markers_px[inter_mcp], project_marker_along_segment(start=markers_px[inter_mcp], stop=markers_px[inter_pip], frac=inter_finger_project_frac)))
        
    # -------------------------------
    # Skin between the 4 ulnar fingers
    # -------------------------------
    with segs.surface_affinity(w=1.5, b=-0.1):
        segs.add({"between_index_middle": between_fingers(inter_mcp=L.inter_mcp_index_middle, inter_pip=L.inter_pip_index_middle)})
        segs.add({"between_middle_ring": between_fingers(inter_mcp=L.inter_mcp_middle_ring, inter_pip=L.inter_pip_middle_ring)})
        segs.add({"between_ring_pinky": between_fingers(inter_mcp=L.inter_mcp_ring_pinky, inter_pip=L.inter_pip_ring_pinky)})


    return segs
