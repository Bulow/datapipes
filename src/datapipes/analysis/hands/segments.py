import torch
import einops
import itertools
import functools
from dataclasses import dataclass, field

from typing import Dict, Any, Callable, Tuple, List, Optional, Iterable
import rich

from datapipes.analysis.hands.geometry import avg_dir, normalize, project_marker_along_segment, project_point_onto_line, vec_len, get_transition_points_along_line
from datapipes.analysis.hands.named_markers import add_custom_markers, L

import plotly.graph_objects as go
import plotly.express as px

@dataclass(frozen=True)
class Segment:
    seg: torch.Tensor
    weight: float = field(default=1.0, kw_only=True)  
    bias: float = field(default=0.0, kw_only=True)

def create_subsegments_raw(n_subsegments: int, start_point: torch.Tensor, stop_point: torch.Tensor, skip_factor: int=1, skip_offset_frac: float=0) -> List[torch.Tensor]:
    linx = torch.linspace(start_point[0], stop_point[0], steps=skip_factor * n_subsegments + 1, device="cuda")
    liny = torch.linspace(start_point[1], stop_point[1], steps=skip_factor * n_subsegments + 1, device="cuda")
    lin = torch.stack((linx, liny), dim=-1)
    segs = [lin[i:i+2] for i in range(int(skip_offset_frac * skip_factor), lin.shape[0] - 1, skip_factor)]
    # print(f"{segs = }")
    return segs

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

def spread_clones_in_dir(thumb_dir_sign: int, seg_list: List[torch.Tensor], n_symmetric_clones: int=1, spread_factor: float=1.0, include_original: bool=False) -> List[torch.Tensor]:
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

def create_subsegments_idx(markers_px, n_subsegments: int, start_idx: int, stop_idx: int, skip_factor: int=1, skip_offset_frac: float=0) -> List[torch.Tensor]:
        return create_subsegments_raw(n_subsegments=n_subsegments, start_point=markers_px[start_idx], stop_point=markers_px[stop_idx], skip_factor=skip_factor, skip_offset_frac=skip_offset_frac)



def build_segments(landmarks_px: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    out: torch.Tensor (segment start_stop coord2D)
    """

    _, h, w = mask.shape
    # print(f"{h = }, {w = }")
    landmarks_px = landmarks_px.to(device="cuda", dtype=torch.float32)
    markers_px, markers_idx = add_custom_markers(landmarks_px, mask=mask)

    segs = []
    
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
    
    
    # arm_dir = - avg_dir(
    #     start=markers_px[L.wrist],
    #     stop0=markers_px[L.pinky_mcp],
    #     stop1=markers_px[L.thumb_mcp],
    #     normalize_input_dirs=True
    # )

    # tpoints = get_transition_points_along_line(mask[0], start_point=markers_px[L.proximal_arm_ulnar], end_point=markers_px[L.proximal_arm_radial])

    # offset = 60
    # wrist_transitions = get_transition_points_along_line(mask[0], start_point=markers_px[L.proximal_arm_ulnar] + (offset * principal_direction), end_point=markers_px[L.proximal_arm_radial] + (offset * principal_direction))
    # wrist_arm_mid_point = 0.5 * torch.stack(wrist_transitions).sum(0)

    # assert len(tpoints) == 2, f"Error: {len(tpoints) = }"

    # segs.append(torch.stack(wrist_transitions))
    # segs.append(torch.stack(tpoints))

    # arm_mid_point = 0.5 * torch.stack(tpoints).sum(0)
    # arm_dir = normalize(arm_mid_point - wrist_arm_mid_point) #  - markers_px[L.wrist])
    arm_dir = normalize(markers_px[L.proximal_arm] - markers_px[L.distal_arm])

    radial_arm_dir = normalize(markers_px[L.proximal_arm_radial] - markers_px[L.distal_arm_radial])

    ulnar_arm_dir = normalize(markers_px[L.proximal_arm_ulnar] - markers_px[L.distal_arm_ulnar])

    parallel_nudge_factor = 0.6
    parallel_nudge = arm_dir * parallel_nudge_factor
    
    extend_units = 300
    # segs.extend(create_subsegments_raw(n_arm_sub_segs, 
    segs.extend(create_subsegments_raw(arm_sub_seg_n, markers_px[L.thumb_vantage_wrist], markers_px[L.thumb_vantage_wrist] + normalize(radial_arm_dir + parallel_nudge) * extend_units * hand_scale_factor))
    segs.extend(create_subsegments_raw(arm_sub_seg_n, markers_px[L.index_vantage_wrist], markers_px[L.index_vantage_wrist] + normalize(arm_dir + radial_arm_dir + parallel_nudge) * extend_units * hand_scale_factor))
    segs.extend(create_subsegments_raw(arm_sub_seg_n, markers_px[L.middle_vantage_wrist], markers_px[L.middle_vantage_wrist] + normalize(arm_dir + parallel_nudge) * extend_units * hand_scale_factor))
    segs.extend(create_subsegments_raw(arm_sub_seg_n, markers_px[L.ring_vantage_wrist], markers_px[L.ring_vantage_wrist] + normalize(arm_dir + ulnar_arm_dir + parallel_nudge) * extend_units * hand_scale_factor))
    segs.extend(create_subsegments_raw(arm_sub_seg_n, markers_px[L.pinky_vantage_wrist], markers_px[L.pinky_vantage_wrist] + normalize(ulnar_arm_dir + parallel_nudge) * extend_units * hand_scale_factor))

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


    inter_finger_project_frac = 0.2
    segs.append(torch.stack((markers_px[L.inter_mcp_index_middle], project_marker_along_segment(start=markers_px[L.inter_mcp_index_middle], stop=markers_px[L.inter_pip_index_middle], frac=inter_finger_project_frac))))
    segs.append(torch.stack((markers_px[L.inter_mcp_middle_ring], project_marker_along_segment(start=markers_px[L.inter_mcp_middle_ring], stop=markers_px[L.inter_pip_middle_ring], frac=inter_finger_project_frac))))
    segs.append(torch.stack((markers_px[L.inter_mcp_ring_pinky], project_marker_along_segment(start=markers_px[L.inter_mcp_ring_pinky], stop=markers_px[L.inter_pip_ring_pinky], frac=inter_finger_project_frac))))


    # line_px = sample_line(mask[0], start_point=markers_px[L.proximal_arm_ulnar], end_point=markers_px[L.proximal_arm_radial]).to(torch.uint8)

    
    
    # idx = get_transitions(sampled_line=line_px)
    # print(idx)

    

    # segs.append(torch.stack(tpoints))

    # rich.print(tpoints)




    # line = markers_px[L.proximal_arm_ulnar] - markers_px[L.proximal_arm_radial]
    # line_length = int(vec_len(line))
    # coord0 = torch.linspace(start=markers_px[L.proximal_arm_radial][0], end=markers_px[L.proximal_arm_ulnar][0], steps=line_length, device=mask.device) #/ (w - 1)
    # # print(f"{coord0 = }")
    # coord1 = torch.linspace(start=markers_px[L.proximal_arm_radial][1], end=markers_px[L.proximal_arm_ulnar][1], steps=line_length, device=mask.device) #/ (h - 1)
    # # print(f"{coord0.shape = }")

    # line_grid = torch.stack((coord0, coord1), dim=-1) / torch.tensor((w - 1, h - 1), device=mask.device)
    # # print(f"{line_grid = }")
    # # print(f"{mask.shape = }")
    # # line_grid = einops.rearrange(line_grid, "w c -> 1 1 w c")
    # # line_px = torch.nn.functional.grid_sample(input=mask.unsqueeze(0).to(torch.float32), grid=line_grid, align_corners=False)
    # # line_px = einops.rearrange(line_px, "1 1 1 w -> w")

    # print(f"{mask.shape = }")

    # fig = px.imshow(mask.squeeze(0).cpu().numpy(), color_continuous_scale='Viridis', binary_string=None)

    # x_coords = line_grid[0, 0, :, 0] * (w - 1)
    # y_coords = line_grid[0, 0, :, 1] * (h - 1)


    # fig.add_trace(
    #     go.Scatter(
    #         # x=coord0.cpu().numpy(),
    #         # y=coord1.cpu().numpy(),
    #         x=x_coords.cpu().numpy(),
    #         y=y_coords.cpu().numpy(),
    #         mode='markers',
    #         marker=dict(
    #             size=10, 
    #             line=dict(width=2, color='white')
    #         ),

    #         hoverinfo="text+x+y", # Shows the name AND coordinates
    #         name='Points'
    #     )
    # )

    # fig.update_layout(
    #     margin=dict(l=0, r=0, b=0, t=0),
    #     showlegend=False
    # )
    
    # fig.show()
    
    # fig = px.line(x=range(line_px.shape[0]), y=line_px.cpu().numpy(), height=256)
    # fig.show()

    from datapipes.plotting import plot
    # plot(mask)

    # fig = px.line(x=range(h), y=mask[0, :, 18].cpu().numpy(), height=256)
    # fig.show()


    out = torch.stack(segs).to("cuda")
    out = einops.rearrange(out, "s e c -> e s c")
    
    return out