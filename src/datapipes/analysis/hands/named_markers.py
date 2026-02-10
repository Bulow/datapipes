#%%
import torch
from enum import StrEnum, auto
from typing import Dict, Any, Callable, Tuple, List, Optional, Iterable
import rich

from datapipes.analysis.hands.geometry import avg_dir, normalize, project_marker_along_segment, project_point_onto_line, vec_len, get_transition_points_along_line, get_closest_mask_boundaries_along_line, center_point_in_mask_boundaries, constrain_point_to_line_midpoint_coincident

mediapipe_landmarks = dict(
    wrist = 0,
    thumb_cmc = 1,
    thumb_mcp = 2,
    thumb_ip = 3,
    thumb_tip = 4,
    index_mcp = 5,
    index_pip = 6,
    index_dip = 7,
    index_tip = 8,
    middle_mcp = 9,
    middle_pip = 10,
    middle_dip = 11,
    middle_tip = 12,
    ring_mcp = 13,
    ring_pip = 14,
    ring_dip = 15,
    ring_tip = 16,
    pinky_mcp = 17,
    pinky_pip = 18,
    pinky_dip = 19,
    pinky_tip = 20,
)

class L(StrEnum):
    wrist = auto()
    thumb_cmc = auto()
    thumb_mcp = auto()
    thumb_ip = auto()
    thumb_tip = auto()
    index_mcp = auto()
    index_pip = auto()
    index_dip = auto()
    index_tip = auto()
    middle_mcp = auto()
    middle_pip = auto()
    middle_dip = auto()
    middle_tip = auto()
    ring_mcp = auto()
    ring_pip = auto()
    ring_dip = auto()
    ring_tip = auto()
    pinky_mcp = auto()
    pinky_pip = auto()
    pinky_dip = auto()
    pinky_tip = auto()

    # Custom

    corrected_middle_mcp = auto()
    corrected_ring_mcp = auto()

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

    distal_arm = auto()
    distal_arm_radial = auto()
    distal_arm_ulnar = auto()

    arm_spacer_thumb_proximal = auto()
    arm_spacer_index_proximal = auto()
    arm_spacer_middle_proximal = auto()
    arm_spacer_ring_proximal = auto()
    arm_spacer_pinky_proximal = auto()

    arm_spacer_thumb_distal = auto()
    arm_spacer_index_distal = auto()
    arm_spacer_middle_distal = auto()
    arm_spacer_ring_distal = auto()
    arm_spacer_pinky_distal = auto()
    
    

def add_custom_markers(landmarks_px: torch.Tensor, mask: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]: 
    # Mediapipe markers 
    landmarks_px = landmarks_px.to("cuda")
    markers: Dict[str, torch.Tensor] = {}
    for name, idx in mediapipe_landmarks.items():
        markers[name] = landmarks_px[idx]

    # ------------------------
    # Mask based corrections
    # ------------------------

    tip_strength = 1.0
    dip_strength = 1.0
    pip_strength = 1.0

    # TIP
    markers[L.thumb_tip] = center_point_in_mask_boundaries(
        point_to_center=markers[L.thumb_tip],
        point_in_segment=markers[L.thumb_ip],
        mask=mask,
        strength=tip_strength
    )
        
    markers[L.index_tip] = center_point_in_mask_boundaries(
        point_to_center=markers[L.index_tip],
        point_in_segment=markers[L.index_dip],
        mask=mask,
        strength=tip_strength
    )
        
    markers[L.middle_tip] = center_point_in_mask_boundaries(
        point_to_center=markers[L.middle_tip],
        point_in_segment=markers[L.middle_dip],
        mask=mask,
        strength=tip_strength
    )
        
    markers[L.ring_tip] = center_point_in_mask_boundaries(
        point_to_center=markers[L.ring_tip],
        point_in_segment=markers[L.ring_dip],
        mask=mask,
        strength=tip_strength
    )
        
    markers[L.pinky_tip] = center_point_in_mask_boundaries(
        point_to_center=markers[L.pinky_tip],
        point_in_segment=markers[L.pinky_dip],
        mask=mask,
        strength=tip_strength
    )

    # DIP
    markers[L.thumb_ip] = center_point_in_mask_boundaries(
        point_to_center=markers[L.thumb_ip],
        point_in_segment=markers[L.thumb_mcp],
        mask=mask,
        strength=dip_strength
    )
        
    markers[L.index_dip] = center_point_in_mask_boundaries(
        point_to_center=markers[L.index_dip],
        point_in_segment=markers[L.index_mcp],
        mask=mask,
        strength=dip_strength
    )
        
    markers[L.middle_dip] = center_point_in_mask_boundaries(
        point_to_center=markers[L.middle_dip],
        point_in_segment=markers[L.middle_mcp],
        mask=mask,
        strength=dip_strength
    )
        
    markers[L.ring_dip] = center_point_in_mask_boundaries(
        point_to_center=markers[L.ring_dip],
        point_in_segment=markers[L.ring_mcp],
        mask=mask,
        strength=dip_strength
    )
        
    markers[L.pinky_dip] = center_point_in_mask_boundaries(
        point_to_center=markers[L.pinky_dip],
        point_in_segment=markers[L.pinky_mcp],
        mask=mask,
        strength=dip_strength
    )

    # PIP        
    markers[L.index_pip] = center_point_in_mask_boundaries(
        point_to_center=markers[L.index_pip],
        point_in_segment=markers[L.index_dip],
        mask=mask,
        strength=pip_strength
    )
        
    markers[L.middle_pip] = center_point_in_mask_boundaries(
        point_to_center=markers[L.middle_pip],
        point_in_segment=markers[L.middle_dip],
        mask=mask,
        strength=pip_strength
    )
        
    markers[L.ring_pip] = center_point_in_mask_boundaries(
        point_to_center=markers[L.ring_pip],
        point_in_segment=markers[L.ring_dip],
        mask=mask,
        strength=pip_strength
    )
        
    markers[L.pinky_pip] = center_point_in_mask_boundaries(
        point_to_center=markers[L.pinky_pip],
        point_in_segment=markers[L.pinky_dip],
        mask=mask,
        strength=pip_strength
    )

    # ------------------------
    # Crosslinks
    # ------------------------
    markers[L.inter_mcp_index_middle] = 0.5 * (markers[L.index_mcp] + markers[L.middle_mcp])
    markers[L.inter_mcp_middle_ring] = 0.5 * (markers[L.middle_mcp] + markers[L.ring_mcp])
    markers[L.inter_mcp_ring_pinky] = 0.5 * (markers[L.ring_mcp] + markers[L.pinky_mcp])
    markers[L.inter_pip_index_middle] = 0.5 * (markers[L.index_pip] + markers[L.middle_pip])
    markers[L.inter_pip_middle_ring] = 0.5 * (markers[L.middle_pip] + markers[L.ring_pip])
    markers[L.inter_pip_ring_pinky] = 0.5 * (markers[L.ring_pip] + markers[L.pinky_pip])

    # markers[L.corrected_middle_mcp] = 0.5 * (markers[L.inter_mcp_index_middle] + markers[L.inter_mcp_middle_ring])
    # markers[L.corrected_ring_mcp] = 0.5 * (markers[L.inter_mcp_middle_ring] + markers[L.inter_mcp_ring_pinky])

    # markers[L.middle_mcp] = constrain_point_to_line_midpoint_coincident(point=markers[L.middle_mcp], line_start=markers[L.inter_mcp_index_middle], line_end=markers[L.inter_mcp_middle_ring])
    # markers[L.ring_mcp] = constrain_point_to_line_midpoint_coincident(point=markers[L.ring_mcp], line_start=markers[L.inter_mcp_middle_ring], line_end=markers[L.inter_mcp_ring_pinky])

    # markers[L.inter_mcp_index_middle] = 0.5 * (markers[L.index_mcp] + markers[L.middle_mcp])
    # markers[L.inter_mcp_middle_ring] = 0.5 * (markers[L.middle_mcp] + markers[L.ring_mcp])
    # markers[L.inter_mcp_ring_pinky] = 0.5 * (markers[L.ring_mcp] + markers[L.pinky_mcp])
    # markers[L.inter_pip_index_middle] = 0.5 * (markers[L.index_pip] + markers[L.middle_pip])
    # markers[L.inter_pip_middle_ring] = 0.5 * (markers[L.middle_pip] + markers[L.ring_pip])
    # markers[L.inter_pip_ring_pinky] = 0.5 * (markers[L.ring_pip] + markers[L.pinky_pip])

    # ------------------------
    # Tips
    # -------------------------

    

    tip_extension_factor = 1.5
    markers[L.thumb_extended_tip] = markers[L.thumb_ip] + tip_extension_factor * (markers[L.thumb_tip] - markers[L.thumb_ip])
    markers[L.index_extended_tip] = markers[L.index_dip] + tip_extension_factor * (markers[L.index_tip] - markers[L.index_dip])
    markers[L.middle_extended_tip] = markers[L.middle_dip] + tip_extension_factor * (markers[L.middle_tip] - markers[L.middle_dip])
    markers[L.ring_extended_tip] = markers[L.ring_dip] + tip_extension_factor * (markers[L.ring_tip] - markers[L.ring_dip])
    markers[L.pinky_extended_tip] = markers[L.pinky_dip] + tip_extension_factor * (markers[L.pinky_tip] - markers[L.pinky_dip])

    

    # arm_seed = project_marker_along_segment(
    #     start=markers[L.wrist],
    #     stop=markers[L.middle_vantage_wrist],
    #     frac=-0.6
    # )
    arm_seed = markers[L.wrist] - (20 * normalize(markers[L.middle_mcp] - markers[L.wrist]))

    arm_dir = markers[L.wrist] - arm_seed

    outside_radial_arm = project_point_onto_line(
        origin=arm_seed,
        vec=(markers[L.thumb_mcp] - arm_seed) * 1.2,
        dir=torch.stack((-arm_dir[1], arm_dir[0]))
    )
    outside_ulnar_arm = project_point_onto_line(
        origin=arm_seed,
        vec=markers[L.pinky_mcp] - arm_seed,
        dir=torch.stack((-arm_dir[1], arm_dir[0]))    
    )

    arm_proximal_edges = get_transition_points_along_line(
        img=mask[0],
        start_point=outside_radial_arm,
        end_point=outside_ulnar_arm,
    )
    if len(arm_proximal_edges) == 2:
        markers[L.proximal_arm_radial] = arm_proximal_edges[0]
        markers[L.proximal_arm_ulnar] = arm_proximal_edges[1]
    else:
        markers[L.proximal_arm_radial] = outside_radial_arm
        markers[L.proximal_arm_ulnar] = outside_ulnar_arm

    arm_distal_edges = get_transition_points_along_line(
        img=mask[0],
        start_point=outside_radial_arm + (20 * normalize(arm_dir)),
        end_point=outside_ulnar_arm + (20 * normalize(arm_dir)),
    )

    if len(arm_distal_edges) == 2:
        markers[L.distal_arm_radial] = arm_distal_edges[0]
        markers[L.distal_arm_ulnar] = arm_distal_edges[1]
    else:
        markers[L.distal_arm_radial] = outside_radial_arm + (20 * normalize(arm_dir))
        markers[L.distal_arm_ulnar] = outside_ulnar_arm + (20 * normalize(arm_dir))
    
    markers[L.proximal_arm] = 0.5 * torch.stack(arm_proximal_edges).sum(0)
    markers[L.distal_arm] = 0.5 * torch.stack(arm_distal_edges).sum(0)


    # ------------------------
    # Palm
    # ------------------------
    vantage_dist = 2
    wrist_shift_factor = 0.8
    vantage_dir = torch.stack((
        (markers[L.wrist] - markers[L.middle_mcp]),
        (markers[L.wrist] - markers[L.middle_mcp]),
        (markers[L.wrist] - markers[L.index_mcp]),
        (markers[L.wrist] - markers[L.ring_mcp]),
        (markers[L.wrist] - markers[L.pinky_mcp]),
    )).mean(0)
    # vantage_dir = normalize(markers[L.proximal_arm] - markers[L.distal_arm]) * (30_000 / vec_len(markers[L.thumb_mcp] - markers[L.wrist]))
    
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


    padding = 0.1
    fracs = torch.linspace(start=padding, end=1 - padding, steps=5)
    
    for i, name in enumerate((L.arm_spacer_thumb_proximal, L.arm_spacer_index_proximal, L.arm_spacer_middle_proximal, L.arm_spacer_ring_proximal, L.arm_spacer_pinky_proximal)):
        markers[name] = project_marker_along_segment(
            start=markers[L.proximal_arm_radial],
            stop=markers[L.proximal_arm_ulnar],
            frac=fracs[i]
        )

    for i, name in enumerate((L.arm_spacer_thumb_distal, L.arm_spacer_index_distal, L.arm_spacer_middle_distal, L.arm_spacer_ring_distal, L.arm_spacer_pinky_distal)):
        markers[name] = project_marker_along_segment(
            start=markers[L.distal_arm_radial],
            stop=markers[L.distal_arm_ulnar],
            frac=fracs[i]
        )

    markers[L.thumb_vantage_wrist] = project_point_onto_line(origin=markers[L.arm_spacer_thumb_proximal], vec=markers[L.thumb_vantage_wrist] - markers[L.arm_spacer_thumb_proximal], dir=markers[L.arm_spacer_thumb_distal] - markers[L.arm_spacer_thumb_proximal])

    markers[L.index_vantage_wrist] = project_point_onto_line(origin=markers[L.arm_spacer_index_proximal], vec=markers[L.index_vantage_wrist] - markers[L.arm_spacer_index_proximal], dir=markers[L.arm_spacer_index_distal] - markers[L.arm_spacer_index_proximal])

    markers[L.middle_vantage_wrist] = project_point_onto_line(origin=markers[L.arm_spacer_middle_proximal], vec=markers[L.middle_vantage_wrist] - markers[L.arm_spacer_middle_proximal], dir=markers[L.arm_spacer_middle_distal] - markers[L.arm_spacer_middle_proximal])

    markers[L.ring_vantage_wrist] = project_point_onto_line(origin=markers[L.arm_spacer_ring_proximal], vec=markers[L.ring_vantage_wrist] - markers[L.arm_spacer_ring_proximal], dir=markers[L.arm_spacer_ring_distal] - markers[L.arm_spacer_ring_proximal])

    markers[L.pinky_vantage_wrist] = project_point_onto_line(origin=markers[L.arm_spacer_pinky_proximal], vec=markers[L.pinky_vantage_wrist] - markers[L.arm_spacer_pinky_proximal], dir=markers[L.arm_spacer_pinky_distal] - markers[L.arm_spacer_pinky_proximal])

    # tpoints = get_transition_points_along_line(mask[0], start_point=markers[L.proximal_arm_ulnar], end_point=markers[L.proximal_arm_radial])

    # offset = 60
    
    # wrist_transitions = get_transition_points_along_line(mask[0], start_point=markers[L.proximal_arm_ulnar] + (offset * arm_dir), end_point=markers[L.proximal_arm_radial] + (offset * arm_dir))
    # wrist_arm_mid_point = 0.5 * torch.stack(wrist_transitions).sum(0)

    # assert len(tpoints) == 2, f"Error: {len(tpoints) = }"

    

    # print(f"{line_length = }")
    # print(f"{lix.shape = }")
    markers = {str(k):v for k, v in markers.items()}
    marker_indices: Dict[str, int] = {k:i for i, k in enumerate(markers.keys())}
    return markers, marker_indices

if __name__ == "__main__":
    markers, marker_indices = add_custom_markers(torch.rand(size=(21, 2)))
    rich.print(markers)
    rich.print(marker_indices)


    # markers to tensor_idx


    print(markers[L.index_extended_tip])
    print(marker_indices[L.index_extended_tip])