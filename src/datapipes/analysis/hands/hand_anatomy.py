from typing import Dict, Tuple
import torch
import rich
import einops 

class L():
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
    radius = 21
    ulna = 22
    index_mc_mid = 23
    middle_mc_mid = 24
    ring_mc_mid = 25
    pinky_mc_mid = 26


def get_poi_index_to_name_dict() -> Dict[int, str]:
    marker_names = {k:v for k, v in L.__dict__.items() if not k.startswith("_")}
    return marker_names


def project_marker_along_segment(start: torch.Tensor, stop: torch.Tensor, frac: float) -> torch.Tensor:
    return start + ((stop - start) * frac)

def avg_dir(start: torch.Tensor, stop0: torch.Tensor, stop1: torch.Tensor) -> torch.Tensor:
    dir0 = stop0 - start
    dir1 = stop1 - start
    return ((dir0 + dir1) / 2)

def add_custom_markers(landmarks: torch.Tensor) -> Dict[int, torch.Tensor]: 
    # Mediapipe markers 
    markers = {i:v for i, v in enumerate(landmarks)}

    # Radius
    markers[L.radius] = project_marker_along_segment(
        start=markers[L.wrist],
        stop=markers[L.pinky_mcp],
        frac=-0.3
    )

    # Ulna
    markers[L.ulna] = markers[L.wrist] - 0.3 * avg_dir(
        start=markers[L.wrist],
        stop0=markers[L.index_mcp],
        stop1=markers[L.thumb_cmc]
    )

    # index_mc_mid 
    markers[L.index_mc_mid] = project_marker_along_segment(
        start=markers[L.wrist],
        stop=markers[L.index_mcp],
        frac=0.5
    )

    # middle_mc_mid
    markers[L.middle_mc_mid] = project_marker_along_segment(
        start=markers[L.wrist],
        stop=markers[L.middle_mcp],
        frac=0.5
    )

    # ring_mc_mid
    markers[L.ring_mc_mid] = project_marker_along_segment(
        start=markers[L.wrist],
        stop=markers[L.ring_mcp],
        frac=0.5
    )

    # pinky_mc_mid
    markers[L.pinky_mc_mid] = project_marker_along_segment(
        start=markers[L.wrist],
        stop=markers[L.pinky_mcp],
        frac=0.5
    )
    return markers

def get_poi_name_to_coords_dict(landmarks: torch.Tensor) -> Dict[str, torch.Tensor]:
    marker_names = {k:landmarks[v] for k, v in L.__dict__.items() if not k.startswith("_")}
    # d = {}
    # rich.print(marker_names)
    return marker_names

segments = {
    "thumb": [
        (L.thumb_mcp, L.thumb_ip),
        (L.thumb_ip, L.thumb_tip),
    ],
    "index": [
        (L.index_mcp, L.index_pip),
        (L.index_pip, L.index_dip),
        (L.index_dip, L.index_tip),
    ],
    "middle": [
        (L.middle_mcp, L.middle_pip),
        (L.middle_pip, L.middle_dip),
        (L.middle_dip, L.middle_tip),
    ],
    "ring": [
        (L.ring_mcp, L.ring_pip),
        (L.ring_pip, L.ring_dip),
        (L.ring_dip, L.ring_tip),
    ],
    "pinky": [
        (L.pinky_mcp, L.pinky_pip),
        (L.pinky_pip, L.pinky_dip),
        (L.pinky_dip, L.pinky_tip),
    ],
    "palm": [
        (L.index_mcp, L.middle_mcp),
        (L.middle_mcp, L.ring_mcp),
        (L.ring_mcp, L.pinky_mcp),

        (L.thumb_cmc, L.index_mcp),

        (L.wrist, L.thumb_cmc),
        (L.thumb_cmc, L.thumb_mcp),

        (L.wrist, L.index_mc_mid),
        (L.index_mc_mid, L.index_mcp),

        # (L.wrist, L.middle_mc_mid),
        (L.middle_mc_mid, L.middle_mcp),

        # (L.wrist, L.ring_mc_mid),
        (L.ring_mc_mid, L.ring_mcp),

        (L.wrist, L.pinky_mc_mid),
        (L.pinky_mc_mid, L.pinky_mcp),

        (L.wrist, L.radius),
        (L.wrist, L.ulna),
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
    lm_px[:, 0] = lm_px[:, 0].clamp(0, img_width - 1)
    lm_px[:, 1] = lm_px[:, 1].clamp(0, img_height - 1)

    return lm_px


def build_segments(normalized_landmarks: torch.Tensor, img_width: int, img_height: int) -> torch.Tensor:
    """
    out: torch.Tensor (segment start_stop coord2D)
    """
    landmarks_px = denormalize_landmarks(normalized_landmarks=normalized_landmarks, img_width=img_width, img_height=img_height)
    markers_px = add_custom_markers(landmarks_px)
    # rich.print(markers)
    segs = []
    for k, v in segments.items():
        segs += [torch.stack((markers_px[start], markers_px[stop])) for start, stop in v]
    out = torch.stack(segs).to("cuda")
    out = einops.rearrange(out, "s e c -> e s c")
    return out

