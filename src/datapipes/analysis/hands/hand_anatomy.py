from typing import Dict, Tuple, Optional, List
import torch
import rich
import einops 
from enum import IntEnum, auto

_custom_markers_start_index = 21
class L(IntEnum):
    wrist = 0
    thumb_cmc = 1
    thumb_mcp = 2
    thumb_ip = 3
    thumb_tip = 4
    index_mcp = 5
    index_pip = 6
    index_dip = 7
    index_tip = 8
    middle_mcp = 9
    middle_pip = 10
    middle_dip = 11
    middle_tip = 12
    ring_mcp = 13
    ring_pip = 14
    ring_dip = 15
    ring_tip = 16
    pinky_mcp = 17
    pinky_pip = 18
    pinky_dip = 19
    pinky_tip = 20

    # Custom
    radius = auto()
    ulna = auto()
    index_mc_mid = auto()
    middle_mc_mid = auto()
    ring_mc_mid = auto()
    pinky_mc_mid = auto()
    index_mc_base = auto()
    middle_mc_base = auto()
    ring_mc_base = auto()
    pinky_mc_base = auto()
    thumb_mc_base = auto()
    ulnar_wrist = auto()
    radial_wrist = auto()
    radio_cental_wrist = auto()
    ulnar_central_wrist = auto()
    tabertier = auto()
    tabertier_arc = auto()
    tabertier_thumb = auto()
    tabertier_index = auto()
    deep_tabertier_start = auto()
    mid_forearm_radial = auto()
    mid_forearm_ulnar = auto()
    inter_mcp_index_middle = auto()
    inter_mcp_middle_ring = auto()
    inter_mcp_ring_pinky = auto()
    inter_pip_index_middle = auto()
    inter_pip_middle_ring = auto()
    inter_pip_ring_pinky = auto()

    thumb_extended_tip = auto()
    index_extended_tip = auto()
    middle_extended_tip = auto()
    ring_extended_tip = auto()
    pinky_extended_tip = auto()

    wrist_vantage_point = auto()
    thumb_vantage_wrist = auto()
    index_vantage_wrist = auto()
    middle_vantage_wrist = auto()
    ring_vantage_wrist = auto()
    pinky_vantage_wrist = auto()

    proximal_arm = auto()
    proximal_arm_radial = auto()
    proximal_arm_ulnar = auto()
    






def get_poi_index_to_name_dict() -> Dict[int, str]:
    marker_names = {v: k for k, v in L.__dict__.items() if not k.startswith("_")}
    return marker_names


def get_poi_index_to_marker_name_dict() -> Dict[int, str]:
    L_names = {v:k for k, v in vars(L).items() if not k.startswith("_")}
    fake_landmarks = torch.zeros(size=(L.pinky_tip + 1, 2), dtype=torch.float32)
    markers = add_custom_markers(fake_landmarks)
    marker_names = {v: L_names[k] for k, v in markers.items()}
    return marker_names


def project_marker_along_segment(start: torch.Tensor, stop: torch.Tensor, frac: float) -> torch.Tensor:
    return start + ((stop - start) * frac)

def normalize(vec: torch.Tensor) -> torch.Tensor:
        return vec / (vec * vec).sum(-1).sqrt()
    

def vec_len(vec: torch.Tensor) -> float:
    return (vec * vec).sum(-1).sqrt()

def avg_dir(start: torch.Tensor, stop0: torch.Tensor, stop1: torch.Tensor, normalize_input_dirs=False) -> torch.Tensor:
    dir0 = (stop0 - start)
    dir1 = (stop1 - start)

    if normalize_input_dirs:
        dir0 = normalize(dir0)
        dir1 = normalize(dir1)

    return ((dir0 + dir1) / 2)

def project_point_onto_line(origin: torch.Tensor, vec: torch.Tensor, dir: torch.Tensor):
    dir = normalize(dir)
    return origin + ((dir * vec).sum(-1) * dir)

def add_custom_markers(landmarks: torch.Tensor) -> Dict[int, torch.Tensor]: 
    # Mediapipe markers 
    markers = {i:v for i, v in enumerate(landmarks.clone())}

    

    # markers[L.ulnar_wrist] = markers[L.wrist] + (markers[L.middle_mcp] - markers[L.index_mcp])
    # markers[L.radial_wrist] = markers[L.wrist] - (markers[L.middle_mcp] - markers[L.index_mcp])


    # markers[L.radio_cental_wrist] = project_marker_along_segment(
    #     start=markers[L.wrist],
    #     stop=markers[L.radial_wrist],
    #     frac=0.3
    # )

    # markers[L.ulnar_central_wrist] = project_marker_along_segment(
    #     start=markers[L.wrist],
    #     stop=markers[L.ulnar_wrist],
    #     frac=0.3
    # )

    
    # # # Radius
    # # markers[L.radius] = markers[L.radial_wrist] - (1.0 * (markers[L.index_mcp] - markers[L.radial_wrist]))

    # # mid_forearm
    # markers[L.mid_forearm_radial] = markers[L.radio_cental_wrist] - (1.0 * (markers[L.middle_mcp] - markers[L.wrist]))
    # markers[L.mid_forearm_ulnar] = markers[L.ulnar_central_wrist] - (1.0 * (markers[L.middle_mcp] - markers[L.wrist]))
    # # Ulna
    # markers[L.ulna] = markers[L.ulnar_wrist] - (1.0 * (markers[L.ring_mcp] - markers[L.ulnar_wrist]))

    # ------------------------
    # Crosslinks
    # ------------------------
    markers[L.inter_mcp_index_middle] = 0.5 * (markers[L.index_mcp] + markers[L.middle_mcp])
    markers[L.inter_mcp_middle_ring] = 0.5 * (markers[L.middle_mcp] + markers[L.ring_mcp])
    markers[L.inter_mcp_ring_pinky] = 0.5 * (markers[L.ring_mcp] + markers[L.pinky_mcp])
    markers[L.inter_pip_index_middle] = 0.5 * (markers[L.index_pip] + markers[L.middle_pip])
    markers[L.inter_pip_middle_ring] = 0.5 * (markers[L.middle_pip] + markers[L.ring_pip])
    markers[L.inter_pip_ring_pinky] = 0.5 * (markers[L.ring_pip] + markers[L.pinky_pip])

    # ------------------------
    # Tips
    # ------------------------
    tip_extension_factor = 1.5
    markers[L.thumb_extended_tip] = markers[L.thumb_ip] + tip_extension_factor * (markers[L.thumb_tip] - markers[L.thumb_ip])
    markers[L.index_extended_tip] = markers[L.index_dip] + tip_extension_factor * (markers[L.index_tip] - markers[L.index_dip])
    markers[L.middle_extended_tip] = markers[L.middle_dip] + tip_extension_factor * (markers[L.middle_tip] - markers[L.middle_dip])
    markers[L.ring_extended_tip] = markers[L.ring_dip] + tip_extension_factor * (markers[L.ring_tip] - markers[L.ring_dip])
    markers[L.pinky_extended_tip] = markers[L.pinky_dip] + tip_extension_factor * (markers[L.pinky_tip] - markers[L.pinky_dip])

    # ------------------------
    # Palm
    # ------------------------
    vantage_dist = 3
    wrist_shift_factor = 0.8
    vantage_dir = torch.stack((
        (markers[L.wrist] - markers[L.middle_mcp]),
        (markers[L.wrist] - markers[L.middle_mcp]),
        (markers[L.wrist] - markers[L.index_mcp]),
        (markers[L.wrist] - markers[L.ring_mcp]),
        (markers[L.wrist] - markers[L.pinky_mcp]),
    )).mean(0)
    
    markers[L.wrist_vantage_point] = markers[L.middle_mcp] + (vantage_dist * vantage_dir)

    index_metacarp_length = vec_len(markers[L.index_mcp] - markers[L.wrist_vantage_point])
    
    def create_vantage_wrist_marker(mcp_marker: int) -> torch.Tensor:
        scale = vec_len(markers[mcp_marker] - markers[L.wrist_vantage_point]) / index_metacarp_length
        marker = project_marker_along_segment(markers[mcp_marker], markers[L.wrist_vantage_point], frac=(wrist_shift_factor / vantage_dist) * scale)

        return marker
    

    # markers[L.thumb_vantage_wrist] = (create_vantage_wrist_marker(L.thumb_cmc))
    

    markers[L.index_vantage_wrist] = create_vantage_wrist_marker(L.index_mcp)
    markers[L.middle_vantage_wrist] = create_vantage_wrist_marker(L.middle_mcp)
    markers[L.ring_vantage_wrist] = create_vantage_wrist_marker(L.ring_mcp)
    markers[L.pinky_vantage_wrist] = create_vantage_wrist_marker(L.pinky_mcp)

    # markers[L.thumb_vantage_wrist] = markers[L.thumb_cmc] + (normalize(markers[L.middle_vantage_wrist] - markers[L.middle_mcp]) * (markers[L.index_vantage_wrist] - markers[L.thumb_cmc])).sum(-1) * normalize(markers[L.middle_vantage_wrist] - markers[L.middle_mcp])

    
    
    # markers[L.thumb_vantage_wrist] = project_marker_along_segment(markers[L.thumb_cmc], markers[L.wrist_vantage_point], frac=wrist_shift_factor / vantage_dist)
    # markers[L.index_vantage_wrist] = project_marker_along_segment(markers[L.index_mcp], markers[L.wrist_vantage_point], frac=wrist_shift_factor / vantage_dist)
    # markers[L.middle_vantage_wrist] = project_marker_along_segment(markers[L.middle_mcp], markers[L.wrist_vantage_point], frac=wrist_shift_factor / vantage_dist)
    # markers[L.ring_vantage_wrist] = project_marker_along_segment(markers[L.ring_mcp], markers[L.wrist_vantage_point], frac=wrist_shift_factor / vantage_dist)
    # markers[L.pinky_vantage_wrist] = project_marker_along_segment(markers[L.pinky_mcp], markers[L.wrist_vantage_point], frac=wrist_shift_factor / vantage_dist)
    


    # ------------------------
    # Tabertier
    # ------------------------
    tabertier_pullback = 0.1
    tabertier_vantage_point = 0.5 * (markers[L.index_vantage_wrist] + markers[L.thumb_cmc])
    markers[L.tabertier] = (
        tabertier_vantage_point
        + 0.5 * avg_dir(
            start=tabertier_vantage_point,
            stop0=markers[L.index_mcp],
            stop1=markers[L.thumb_mcp],
        ) 
        # + tabertier_pullback * (markers[L.thumb_cmc] - markers[L.thumb_mcp])
        # + tabertier_pullback * (markers[L.middle_mcp] - markers[L.index_mcp])
    )

    markers[L.tabertier_arc] = 0.5 * (markers[L.thumb_mcp] + markers[L.index_mcp])

    tabertier_edge_standoff = 0.2
    markers[L.tabertier_thumb] = markers[L.thumb_mcp] + (tabertier_edge_standoff * (markers[L.index_mcp] - markers[L.thumb_mcp]))
    markers[L.tabertier_index] = markers[L.index_mcp] + (tabertier_edge_standoff * (markers[L.thumb_mcp] - markers[L.index_mcp]))

    markers[L.deep_tabertier_start] = project_marker_along_segment(
        start=markers[L.tabertier],
        stop=markers[L.tabertier_arc],
        frac=0.5
    )

    markers[L.thumb_vantage_wrist] = project_point_onto_line(
        origin=markers[L.thumb_cmc], 
        vec=markers[L.middle_vantage_wrist] - markers[L.thumb_cmc], 
        dir=markers[L.tabertier] - markers[L.tabertier_arc]
    )

    markers[L.proximal_arm] = project_marker_along_segment(
        start=markers[L.wrist],
        stop=markers[L.middle_vantage_wrist],
        frac=-1.0
    )

    arm_dir = markers[L.wrist] - markers[L.proximal_arm]

    markers[L.proximal_arm_radial] = project_point_onto_line(
        origin=markers[L.proximal_arm],
        vec=(markers[L.thumb_mcp] - markers[L.proximal_arm]) * 1.2,
        dir=torch.stack((-arm_dir[1], arm_dir[0]))
    )
    markers[L.proximal_arm_ulnar] = project_point_onto_line(
        origin=markers[L.proximal_arm],
        vec=markers[L.pinky_mcp] - markers[L.proximal_arm],
        dir=torch.stack((-arm_dir[1], arm_dir[0]))    
    )

    

    # print(f"{line_length = }")
    # print(f"{lix.shape = }")

    return markers

def get_poi_name_to_coords_dict(landmarks: torch.Tensor) -> Dict[str, torch.Tensor]:
    marker_names = {k:landmarks[v] for k, v in L.__dict__.items() if not k.startswith("_")}
    # d = {}
    # rich.print(marker_names)
    return marker_names

segments = {
    "specials": [
        # (L.tabertier, L.tabertier_thumb),
        # (L.tabertier, L.tabertier_index),
        # (L.tabertier, L.tabertier_arc),
        
        # (L.tabertier, L.tabertier_arc),
        # (L.tabertier_arc, L.tabertier_thumb),
        # (L.tabertier_arc, L.tabertier_index),

        # (L.tabertier_arc, L.tabertier_thumb),
        # (L.tabertier, L.tabertier_index),

        # (L.index_mcp, L.middle_mcp),
        # (L.middle_mcp, L.ring_mcp),
        # (L.ring_mcp, L.pinky_mcp),

        # (L.inter_mcp_index_middle, L.inter_pip_index_middle),
        # (L.inter_mcp_middle_ring, L.inter_pip_middle_ring),
        # (L.inter_mcp_ring_pinky, L.inter_pip_ring_pinky),

        # (L.radio_cental_wrist, L.ulnar_central_wrist),
        # (L.inter_mcp_index_middle, L.index_pip),
        # (L.inter_mcp_index_middle, L.middle_pip),
    ]
    

    
}

def get_region_name_to_value_dict() -> Dict[str, int]:
    marker_names = {v:k for k, v in L.__dict__.items() if not k.startswith("_")}
    # rich.print(marker_names)
    
    segs = ["background"]
    for k, v in segments.items():
        segs += [f"{marker_names[start]}->{marker_names[stop]}" for start, stop in v]
    
    segs = {s:i for i, s in enumerate(segs)}

    # rich.print(segs)
    return segs

def get_region_value_to_name_dict() -> Dict[str, int]:
    return {v:n for n, v in get_region_name_to_value_dict().items()}

segment_values: Dict[str, int] = get_region_name_to_value_dict()

def denormalize_landmarks(normalized_landmarks: torch.Tensor, img_width: int, img_height: int) -> torch.Tensor:
    # Convert landmarks to pixel coords
    lm_px = normalized_landmarks.to(device="cuda", dtype=torch.float32).clone()
    lm_px[:, 0] = lm_px[:, 0] * (img_width - 1)
    lm_px[:, 1] = lm_px[:, 1] * (img_height - 1)

    # Clamp to image bounds
    # lm_px[:, 0] = lm_px[:, 0].clamp(0, img_width - 1)
    # lm_px[:, 1] = lm_px[:, 1].clamp(0, img_height - 1)

    return lm_px

import itertools
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Segment:
    seg: torch.Tensor
    weight: float = field(default=1.0, kw_only=True)  
    bias: float = field(default=0.0, kw_only=True)

def build_segments(normalized_landmarks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    out: torch.Tensor (segment start_stop coord2D)
    """

    img_width=mask.shape[-1]
    img_height=mask.shape[-2]

    segs = []
    # def add_segments(*new_segs: Iterable[intermediate_phalanges_sub_seg_n])
    # landmarks_px = denormalize_landmarks(normalized_landmarks=normalized_landmarks, img_width=img_width, img_height=img_height)
    _, h, w = mask.shape
    print(f"{h = }, {w = }")
    landmarks_px = normalized_landmarks.to(device="cuda", dtype=torch.float32).clone() * torch.tensor([[w - 1, h - 1]], device=mask.device)

    
    markers_px = add_custom_markers(landmarks_px)
    
    principal_direction = normalize(landmarks_px[L.middle_mcp] - landmarks_px[L.wrist])
    thumb_direction = normalize(landmarks_px[L.index_mcp] - landmarks_px[L.middle_mcp])
    thumb_dir_sign = (torch.stack((-principal_direction[1], principal_direction[0])) * thumb_direction).sum(-1).sign()
    
    prototypical_hand_scale_factor = 60 # Arbitrarily determined normalization factor
    hand_scale_factor = torch.tensor((
        vec_len(landmarks_px[L.index_mcp] - landmarks_px[L.middle_mcp]),
        vec_len(landmarks_px[L.middle_mcp] - landmarks_px[L.ring_mcp]),
        vec_len(landmarks_px[L.ring_mcp] - landmarks_px[L.pinky_mcp]),
    )).mean() / prototypical_hand_scale_factor
    

    # print(f"{principal_direction = }")
    # print(f"{thumb_direction = }")
    # print(f"{thumb_dir_sign = }")
    # print(f"{hand_scale_factor = }")

    

    # print(f"{markers_px = }")

    
    names = get_poi_index_to_name_dict()
    # for i, (n, p) in enumerate(zip(normalized_landmarks, markers_px.values())):
    #     print(f"build_segments [{names[i]}]: {n = }, {p = }")
    for k in names.keys():
        print(f"build_segments [{names[k]}]: {markers_px[k] = }")

    # rich.print(f"{markers_px[0].shape = }")
    def create_subsegments_raw(n_subsegments: int, start_point: torch.Tensor, stop_point: torch.Tensor, skip_factor: int=1, skip_offset_frac: float=0) -> List[torch.Tensor]:
        linx = torch.linspace(start_point[0], stop_point[0], steps=skip_factor * n_subsegments + 1, device="cuda")
        liny = torch.linspace(start_point[1], stop_point[1], steps=skip_factor * n_subsegments + 1, device="cuda")
        lin = torch.stack((linx, liny), dim=-1)
        segs = [lin[i:i+2] for i in range(int(skip_offset_frac * skip_factor), lin.shape[0] - 1, skip_factor)]
        # print(f"{segs = }")
        return segs


    def create_subsegments(n_subsegments: int, start_idx: int, stop_idx: int, skip_factor: int=1, skip_offset_frac: float=0) -> List[torch.Tensor]:
        return create_subsegments_raw(n_subsegments=n_subsegments, start_point=markers_px[start_idx], stop_point=markers_px[stop_idx], skip_factor=skip_factor, skip_offset_frac=skip_offset_frac)
    
    
    def crosslink(first_seg_list: List[torch.Tensor], second_seg_list: List[torch.Tensor]) -> List[torch.Tensor]:
        # assert len(first_seg_list) == len(second_seg_list)
        # return [torch.stack((f[0], s[0]), dim=-1) for f, s in zip(first_seg_list, second_seg_list)]
        segs = []
        for f, s in zip(first_seg_list, second_seg_list):
            # if f is None or s is None:
            #     break
            # print(f"{f[0] = }, {s[0] = }")
            link = s[0] - f[0]
            margin = 0.15
            segs.append(torch.stack((f[0] + (link * margin), s[0] - (link * margin))))
        return segs
    
    def spread_clones(seg_list: List[torch.Tensor], n_symmetric_clones: int=1, spread_factor: float=1.0, include_original: bool=False) -> List[torch.Tensor]:
        segs = []
        if include_original:
            segs.extend(seg_list)
        # print(f"{segs = }")
        for i in range(n_symmetric_clones):
            for seg in seg_list:
                seg_dir = seg[1] - seg[0]
                unit_seg_dir = (seg_dir / (seg_dir ** 2).sum(-1).sqrt())
                seg_normal = torch.stack((-unit_seg_dir[1], unit_seg_dir[0])) * thumb_dir_sign # thumb_dir_sign keeps order consistent between hands
                offset = (seg_normal * spread_factor * (i + 1))

                segs.append(seg - offset)
                segs.append(seg + offset)
        return segs
    
    
    

    palm_sub_seg_n = 4
    palm_skip_factor = 1
    palm_skip_offset_frac = 0.0

    proximal_phalanges_sub_seg_n = 3 
    intermediate_phalanges_sub_seg_n = 2
    distal_phalanges_sub_seg_n = 3

    arm_sub_seg_n = 6

    index_metacarp = create_subsegments(palm_sub_seg_n, L.index_vantage_wrist, L.index_mcp, skip_factor=palm_skip_factor, skip_offset_frac=0)
    middle_metacarp = create_subsegments(palm_sub_seg_n, L.middle_vantage_wrist, L.middle_mcp, skip_factor=palm_skip_factor, skip_offset_frac=palm_skip_offset_frac)
    ring_metacarp = (create_subsegments(palm_sub_seg_n, L.ring_vantage_wrist, L.ring_mcp, skip_factor=palm_skip_factor, skip_offset_frac=0))
    pinky_metacarp = (create_subsegments(palm_sub_seg_n, L.pinky_vantage_wrist, L.pinky_mcp, skip_factor=palm_skip_factor, skip_offset_frac=palm_skip_offset_frac))
    
    palm_spread_factor = 4
    segs.append(index_metacarp[0])
    segs.extend(spread_clones(index_metacarp[1:], spread_factor=palm_spread_factor))
    segs.append(middle_metacarp[0])
    segs.extend(spread_clones(middle_metacarp[1:], spread_factor=palm_spread_factor))
    segs.append(ring_metacarp[0])
    segs.extend(spread_clones(ring_metacarp[1:], spread_factor=palm_spread_factor))
    segs.append(pinky_metacarp[0])
    segs.extend(spread_clones(pinky_metacarp[1:], spread_factor=palm_spread_factor))
    # segs.extend((thumb_metacarp))

    # print(f"{len(thumb_metacarp) = }, {len(index_metacarp) = }")
    # # segs.extend(crosslink(thumb_metacarp, index_metacarp))
    # segs.extend(crosslink(index_metacarp, middle_metacarp))
    # segs.extend(crosslink(middle_metacarp, ring_metacarp))
    # segs.extend(crosslink(ring_metacarp, pinky_metacarp))
    
    
    arm_dir = - avg_dir(
        start=markers_px[L.wrist],
        stop0=markers_px[L.pinky_mcp],
        stop1=markers_px[L.thumb_mcp],
        normalize_input_dirs=True
    )
    
    extend_units = 300
    # segs.extend(create_subsegments_raw(n_arm_sub_segs, 
    segs.extend(create_subsegments_raw(arm_sub_seg_n, markers_px[L.thumb_vantage_wrist], markers_px[L.thumb_vantage_wrist] + arm_dir * extend_units * hand_scale_factor))
    segs.extend(create_subsegments_raw(arm_sub_seg_n, markers_px[L.index_vantage_wrist], markers_px[L.index_vantage_wrist] + arm_dir * extend_units * hand_scale_factor))
    segs.extend(create_subsegments_raw(arm_sub_seg_n, markers_px[L.middle_vantage_wrist], markers_px[L.middle_vantage_wrist] + arm_dir * extend_units * hand_scale_factor))
    segs.extend(create_subsegments_raw(arm_sub_seg_n, markers_px[L.ring_vantage_wrist], markers_px[L.ring_vantage_wrist] + arm_dir * extend_units * hand_scale_factor))
    segs.extend(create_subsegments_raw(arm_sub_seg_n, markers_px[L.pinky_vantage_wrist], markers_px[L.pinky_vantage_wrist] + arm_dir * extend_units * hand_scale_factor))

    segs.extend(spread_clones(create_subsegments_raw(3, markers_px[L.thumb_vantage_wrist], markers_px[L.thumb_mcp]), n_symmetric_clones=1))
    # segs.extend(create_subsegments(2, L.thumb_cmc, L.tabertier))

    segs.extend(spread_clones(create_subsegments(2, L.tabertier, L.tabertier_arc), n_symmetric_clones=1, spread_factor=5))
    


    thumb_finger = ([]
        + create_subsegments(intermediate_phalanges_sub_seg_n, L.thumb_mcp, L.thumb_ip)
        + create_subsegments(distal_phalanges_sub_seg_n, L.thumb_ip, L.thumb_extended_tip)
    )

    index_finger = ([]
        + create_subsegments(proximal_phalanges_sub_seg_n, L.index_mcp, L.index_pip)
        + create_subsegments(intermediate_phalanges_sub_seg_n, L.index_pip, L.index_dip)
        + create_subsegments(distal_phalanges_sub_seg_n, L.index_dip, L.index_extended_tip)
    )

    middle_finger = ([]
        + create_subsegments(proximal_phalanges_sub_seg_n, L.middle_mcp, L.middle_pip)
        + create_subsegments(intermediate_phalanges_sub_seg_n, L.middle_pip, L.middle_dip)
        + create_subsegments(distal_phalanges_sub_seg_n, L.middle_dip, L.middle_extended_tip)
    )

    ring_finger = ([]
        + create_subsegments(proximal_phalanges_sub_seg_n, L.ring_mcp, L.ring_pip)
        + create_subsegments(intermediate_phalanges_sub_seg_n, L.ring_pip, L.ring_dip)
        + create_subsegments(distal_phalanges_sub_seg_n, L.ring_dip, L.ring_extended_tip)
    )

    pinky_finger = ([]
        + create_subsegments(proximal_phalanges_sub_seg_n, L.pinky_mcp, L.pinky_pip)
        + create_subsegments(intermediate_phalanges_sub_seg_n, L.pinky_pip, L.pinky_dip)
        + create_subsegments(distal_phalanges_sub_seg_n, L.pinky_dip, L.pinky_extended_tip)
    )

    spread_factor=3
    print(f"Before adding fingers: {len(segs) = }")
    segs.extend(spread_clones(index_finger[:-1], 1, spread_factor=spread_factor))
    segs.append(index_finger[-1])

    segs.extend(spread_clones(middle_finger[:-1], 1, spread_factor=spread_factor))
    segs.append(middle_finger[-1])

    segs.extend(spread_clones(ring_finger[:-1], 1, spread_factor=spread_factor))
    segs.append(ring_finger[-1])

    segs.extend(spread_clones(pinky_finger[:-1], 1, spread_factor=spread_factor))
    segs.append(pinky_finger[-1])

    segs.extend(spread_clones(thumb_finger[:-1], 1, spread_factor=spread_factor))
    segs.append(thumb_finger[-1])

    print(f"After adding fingers: {len(segs) = }")

  
    # inter_mcp_sub_seg_n = 1
    # segs.extend(create_subsegments(inter_mcp_sub_seg_n, L.index_mcp, L.middle_mcp))
    # segs.extend(create_subsegments(inter_mcp_sub_seg_n, L.middle_mcp, L.ring_mcp))
    # segs.extend(create_subsegments(inter_mcp_sub_seg_n, L.ring_mcp, L.pinky_mcp))
    for k, v in segments.items():
        segs += [torch.stack((markers_px[start], markers_px[stop])) for start, stop in v]



    # find border points along line from proximal_arm_ulnar to proximal_arm_radial
    
    print(f"{markers_px[L.proximal_arm_radial] = }")
    print(f"{markers_px[L.proximal_arm_ulnar] = }")
    # rich.print(f"{markers_px[L.wrist] = }")
    # rich.print(f"{normalized_landmarks[L.wrist] = }")
    line = markers_px[L.proximal_arm_ulnar] - markers_px[L.proximal_arm_radial]
    line_length = int(vec_len(line))
    coord0 = torch.linspace(start=markers_px[L.proximal_arm_radial][0], end=markers_px[L.proximal_arm_ulnar][0], steps=line_length, device=mask.device) / (img_width - 1)
    # print(f"{coord0 = }")
    coord1 = torch.linspace(start=markers_px[L.proximal_arm_radial][1], end=markers_px[L.proximal_arm_ulnar][1], steps=line_length, device=mask.device) / (img_height - 1)
    # print(f"{coord0.shape = }")

    line_grid = torch.stack((coord0, coord1), dim=-1)
    # print(f"{line_grid = }")
    # print(f"{mask.shape = }")
    line_grid = einops.rearrange(line_grid, "w c -> 1 1 w c")
    line_px = torch.nn.functional.grid_sample(input=mask.unsqueeze(0).to(torch.float32), grid=line_grid, align_corners=False)
    line_px = einops.rearrange(line_px, "1 1 1 w -> w")

    # print(f"{line_px.shape = }")
    
    import plotly.express as px
    fig = px.line(x=range(line_px.shape[0]), y=line_px.cpu().numpy(), height=256)
    fig.show()

    print(f"Number of segments: {len(segs)}")


    out = torch.stack(segs).to("cuda")
    out = einops.rearrange(out, "s e c -> e s c")
    return out


