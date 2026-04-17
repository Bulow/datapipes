import torch
import torch.nn.functional as F
from datapipes import sic, filters
from datapipes.plotting import plots, map01, plot, crop_to_common_size
import einops
from datapipes.analysis.hands import hand_anatomy, hand_landmarks, hand_segmentation
from typing import Dict, Tuple, Literal, Optional, Callable
import kornia
import inspect
import os
import linecache
import kornia.filters as k
import kornia
from datapipes.analysis.hands.hand_anatomy import get_region_name_to_value_dict
from datapipes.manual_ops import with_manual_op, ManualOp
from datapipes.analysis.hands import named_markers, segments
from dataclasses import dataclass
from datapipes.analysis.hands.visualization import mask_to_distinct_colors
_debug_frame_idx = 6
from datapipes.analysis.hands.named_markers import L
import rich
import datapipes.analysis.hands.segmentation_map as segmentation_map
from datapipes.analysis.hands.geometry import normalize, project_point_onto_line


def _dplot(*vars: str):
    
    assert all([isinstance(s, str) for s in vars]), f"Expected strings, got: {", ".join([str(type(s)) for s in vars])}"
    # print(vars)
    # for var in vars:
    #     print(var.split(sep="=")[0])
    # return
    
    expressions = [var.split("=")[0].strip() for var in vars]
    frame = inspect.currentframe().f_back
    filename = os.path.basename(frame.f_code.co_filename)
    lineno = frame.f_lineno
    func = frame.f_code.co_name
    # source_line = linecache.getline(filename, lineno).strip()

    info = f"{func}:{lineno}"
    print(f"{info}: {", ".join(expressions)}:")
    # plot(eval(expression))
    
    
    plot(*[eval(expression, frame.f_globals, frame.f_locals) for expression in expressions])

def vec_to_map(vecs: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    
    if mask is None:
        mask = _debug_dict["mask"][:, 450:, :]
    print(f"{vecs.shape = }")
    print(f"{mask.shape = }")
    maps = torch.zeros(size=(vecs.shape[-1], 1, *mask.shape[-2:]), device="cuda", dtype=torch.float)
    print(f"{maps.shape = }")
    d = einops.rearrange(vecs, "c d -> d 1 c")
    maps[..., mask > 0] = d
    return maps

def _get_point_to_segment_distance(P, A, B, gradient: list[torch.Tensor], eps=1e-8):
    """
    P: [M,2] points
    A: [N,2] segment start
    B: [N,2] segment end
    Returns: distances [M,N], dot_grad [M,N]
    """
    gx, gy = gradient

    # Expand for broadcasting: P[M,1,2], A[1,N,2], B[1,N,2]
    P = P[:, None, :]              # [M,1,2]
    A = A[None, :, :]              # [1,N,2]
    B = B[None, :, :]              # [1,N,2]

    AB = B - A                     # [1,N,2]
    AP = P - A                     # [M,N,2]

    _debug_dict["AB"] = AB

    # Project AP onto AB, clamp to segment
    ab2 = (AB * AB).sum(dim=-1, keepdim=True) #.clamp_min(eps)  # [1,N,1]
    t = (AP * AB).sum(dim=-1, keepdim=True) #/ ab2             # [M,N,1]

    axial_coord = t

    t = t / ab2

    t = t.clamp(0.0, 1.0)
    closest = A + t * AB           # [M,N,2]

    vec_to_segment = (P - closest)

    _debug_dict["vec_to_segment"] = vec_to_segment
    _debug_dict["t"] = axial_coord

    # coords = (t, signed vec_to_segment)

    grads = torch.stack([gx, gy])
    grads = einops.rearrange(grads, "d n -> n 1 d")

    _debug_dict["grads"] = grads

    # Direction to closest point in segment dotted with gradient at each pixel location
    # dp = (vec_to_segment_dir * grads).sum(-1).to(torch.float32)
    dp = (vec_to_segment * grads).sum(-1).to(torch.float32)

    d = (vec_to_segment ** 2).sum(dim=-1).sqrt()  # [M,N]
    # _debug_dict["d"] = d

    # TODO: Return local basis coordinates of pixels

    return d, dp

def _get_line_segment_distances(
    mask: torch.Tensor,
    gradient: torch.Tensor,
    hand_segments: segments.HandSegments # (segments start_stop=2 coord2D=2)
):
    """
    mask:      [H,W] bool or 0/1 or 0..255 tensor (on CPU or CUDA)
    landmarks: [21,2] tensor. If normalized_landmarks=True -> x,y in [0,1],
               else pixel coords (x,y).
    Returns:
      label_map: [H,W] long, 0=background, 1..N=segment id
      colored:   [H,W,3] uint8 RGB image
      names:     list[str] for ids 1..N (names[id-1])
    """
    device = mask.device
    H, W = mask.shape[-2], mask.shape[-1]

    mask, gradient = plots.crop_to_common_size(mask, gradient)
    _debug_dict["mask"] = mask
    mask_bool = mask.bool()

    # 

    g_mask = mask_bool #einops.repeat(mask_bool, "h w -> 2 h w")
    gx, gy = gradient[0, ...][g_mask], gradient[1, ...][g_mask]

    segs = hand_segments.get_segments_tensor().to("cuda")
    # segs = hand_anatomy.build_segments(lm_px)
    A = segs[0]
    B = segs[1]

    # Coordinates of mask pixels only (faster than all pixels)
    ys, xs = torch.where(mask_bool)         # [M], [M]
    P = torch.stack([xs, ys], dim=-1).float()  # [M,2] in (x,y)

    dists, dot_grad = _get_point_to_segment_distance(P, A, B, gradient=[gx, gy])   # [M,N]
    
    
    return dists, dot_grad

def _compute_normal_map(image: torch.Tensor) -> torch.Tensor:
    """
    Given a grayscale height map image with shape (1, H, W), compute its gradient-based normal map.
    
    Parameters:
        image (torch.Tensor): A tensor of shape (1, H, W) representing the grayscale height map.
        
    Returns:
        torch.Tensor: A tensor of shape (3, H, W) representing the normalized normal vectors.
                      Each normal vector is computed as n = (-grad_x, -grad_y, 1) and then normalized.
    """

    # Ensure the image has a batch dimension: (N, C, H, W)
    if image.dim() == 3:
        image = image.unsqueeze(0)

    # image = torch.log(image + 1e-5)

    grad = kornia.filters.spatial_gradient(image, mode="diff", order=1, normalized=True)[0]
    smooth_grad = grad
    smooth_grad = kornia.filters.guided_blur(guidance=image, input=grad, kernel_size=17, eps=1e-1, subsample=1)
    # plot(smooth_grad)
    return -smooth_grad[0]

def _prepare_gradients(img_data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Log space to be intensity scale invariant
    # print(f"{img_data.shape = }")
    # plot(img_data)
    hand_min = img_data[mask > 0].min()
    # print(f"{hand_min = }")
    img_data[mask == 0] = hand_min
    # img_data = img_data.log1p()
    

    # img_data = map01(img_data)
    # img_data = torch.log(img_data + 1e-3)
    # img_data = kornia.enhance.equalize(img_data)
    # plot(img_data)

    # hand_min = img_data[mask > 0].min()
    
    
    # print(f"{hand_min = }")
    # # img_data[mask == 0] = hand_min
    # img_data = img_data.clamp_min(hand_min)
    img_data = map01(img_data)
    # plot(img_data)

    # Compute normalized gradients using a sobel filter
    gradient = _compute_normal_map(img_data).squeeze(0)

    # gradient = kornia.filters.guided_blur(guidance=img_data.unsqueeze(0), input=gradient.unsqueeze(0), kernel_size=5, eps=1e-2, subsample=1)

    # neg = gradient < 0
    

    # gradient = gradient.abs()
    # gradient = torch.sqrt(gradient.abs() + 1e-5)
    # # gradient = gradient.log1p()
    # gradient /= gradient.max()
    # # gradient = kornia.enhance.equalize(gradient)
    # gradient[neg] *= -1

    # gradient = kornia.filters.guided_blur(guidance=img_data.unsqueeze(0), input=gradient.unsqueeze(0), kernel_size=17, eps=1e-5, subsample=1)


    # plot(gradient)
    # gradient = kornia.enhance.equalize(map01(gradient))
    # plot(gradient.sign())
    # Apply mask
    # _dplot(f"{gradient = }")
    _g, _m = plots.crop_to_common_size(gradient, mask)

    # gg = torch.zeros_like(_m, dtype=torch.float32)
    # gg[_m == 0] = 1
    # # gg = k.bilateral_blur(gg.unsqueeze(0), kernel_size=7, sigma_color=2.0, sigma_space=(2.0, 2.0)).squeeze(0)
    # # gg = k.gaussian_blur2d(gg.unsqueeze(0), kernel_size=5, sigma=(2, 2), separable=True).squeeze(0)
    # gg = k.box_blur(gg.unsqueeze(0), kernel_size=7, separable=True).squeeze(0)
    # gg = k.box_blur(gg.unsqueeze(0), kernel_size=7, separable=True).squeeze(0)
    # gg = k.box_blur(gg.unsqueeze(0), kernel_size=3, separable=True).squeeze(0)
    # # gg = k.box_blur(gg.unsqueeze(0), kernel_size=3, separable=True).squeeze(0)
    # gg = k.box_blur(gg.unsqueeze(0), kernel_size=3, separable=True).squeeze(0)
    # gg = k.box_blur(gg.unsqueeze(0), kernel_size=3, separable=True).squeeze(0)
    # gg = k.box_blur(gg.unsqueeze(0), kernel_size=3, separable=True).squeeze(0)

    # _dplot(f"{_g * _m = }", f"{gg=}")
    # ggrad = k.spatial_gradient(gg.unsqueeze(0), mode="diff").squeeze(0)[0]
    # gradient += gradient * gg * 3
    # _dplot(f"{gradient * _m = }", f"{gg * _m = }", f"{ggrad + gradient = }")

    gradient = (_g) * _m
    return gradient


def _soft_argmin(t: torch.Tensor) -> torch.Tensor:
    p_smoothed = torch.softmax((-t) / t.shape[0], dim=0)
    # p_smoothed = kornia.filters.box_blur(p, kernel_size=3, separable=True)
    p_smoothed = k.gaussian_blur2d(p_smoothed, kernel_size=7, sigma=(2.0, 2.0))
    labels = torch.argmax(p_smoothed, dim=0)
    # print(f"{labels.shape = }")
    return labels

_debug_distances: torch.Tensor
_debug_dict: Dict = {}

@dataclass
class _SegmentationMapBuildingBlocks:
    nearest: torch.Tensor
    raw_distances: torch.Tensor
    adjusted_distances: torch.Tensor
    dot_grad_maps: torch.Tensor
    mask: torch.Tensor

def _closest_segment_mask(mask: torch.Tensor, gradient: torch.Tensor, hand_segments: segments.HandSegments, use_surface_optimization: bool=True) -> tuple[torch.Tensor, _SegmentationMapBuildingBlocks]:

    # gradient = kornia.filters.box_blur(input=gradient.unsqueeze(0), kernel_size=5, separable=True)[0]
    
    _m, _gradient = plots.crop_to_common_size(mask, gradient)
    dists, dot_grad = _get_line_segment_distances(
        mask=_m[0],
        gradient=_gradient,
        hand_segments=hand_segments
    )

    # Reconstruct maps from raw 1D pixels
    
    # Distance to each line segment
    dist_maps = torch.zeros(size=[dists.shape[-1], 1, *mask.shape[-2:]], device="cuda", dtype=torch.float)
    d = einops.rearrange(dists, "c d -> d 1 c")
    dist_maps[..., mask[0] > 0] = d

    # gradient dot direction to each line segment
    dot_grad_maps = torch.zeros(size=[dot_grad.shape[-1], 1, *mask.shape[-2:]], device="cuda", dtype=torch.float)
    d = einops.rearrange(dot_grad, "c d -> d 1 c")
    dot_grad_maps[..., mask[0] > 0] = d

    # dot_grad_maps[dot_grad_maps < 0] *= 1.5
    # plot(dot_grad_maps[64])


    neg = dot_grad_maps < 0
    # dot_grad_maps = torch.log(dot_grad_maps.abs() + 1e-5)
    dot_grad_maps = torch.sqrt(dot_grad_maps.abs() + 1e-5)
    # dot_grad_maps /= dot_grad_maps.max(dim=0).values

    # dot_grad_maps = kornia.enhance.equalize(map01(dot_grad_maps))
    dot_grad_maps[neg] *= -1

    # print(f"{hand_segments.get_weights_tensor().shape = }, {dot_grad_maps.shape = }")
    # Adjust by surface affinity
    dot_grad_maps = (dot_grad_maps * hand_segments.get_weights_tensor()) + hand_segments.get_biases_tensor()

    # dot_grad_maps[97:] += 0.3
    # # dot_grad_maps[67:] *= 0.5

    # dot_grad_maps[:97] += 0.0
    # dot_grad_maps[:97] *= 0.2

    
    _debug_dict["dot_grad_maps"] = dot_grad_maps
    _debug_dict["hand_segments"] = hand_segments

    # Penalize distances where the gradient points away from each line segment
    _d, _g = plots.crop_to_common_size(dist_maps, dot_grad_maps)
    # _dplot(f"{_d[_debug_frame_idx]=}", f"{_g[_debug_frame_idx]=}")

    pos_g = torch.clone(_g)
    pos_g[_g < 0] = 0

    neg_g = torch.clone(_g)
    neg_g[_g > 0] = 0
    

    # print("pos, neg")
    # plot(pos_g[64], neg_g[64])

    # delta_d = map01(_d) - (map01(_g) * 1)
    # delta_d = map01(_d) + (
    #     (
    #         map01(map01(-_d).square()) * (map01(pos_g))
    #     ).square()
    # ) - (
    #     (
    #         map01(map01(-_d).square()) * (map01(neg_g))
    #     ).square()
    # )

    
    # delta_d = map01(_d) - ((map01(map01(-_d).square()) * (map01(pos_g) * (-map01(neg_g)))) ** 2)
    delta_d = map01(_d) - ((map01(map01(-_d).square()) * (map01(pos_g))) ** 2) - ((map01(map01(-_d).square()) * (map01(neg_g))) ** 2)

    # delta_d = _d * (1 - _g)

    # plot(dot_grad_maps[64], delta_d[64], map01(-_d).square()[64], mode="vertical")
    _debug_dict["locals"] = locals()
    
    if use_surface_optimization:
        adjusted_dists = delta_d
    else:
        adjusted_dists = map01(_d)

    adjusted_dists = kornia.filters.box_blur(adjusted_dists, kernel_size=17, separable=True)

    _debug_dict["adjusted_dists"] = adjusted_dists

    # Compute regions as index of closest line segment at each pixel
    # nearest = torch.argmin(adjusted_dists, dim=0)[0] + 1
    nearest = _soft_argmin(adjusted_dists)[0] + 1



    # Apply mask
    smooth_mask = mask.to(torch.float16).unsqueeze(0)
    smooth_mask = kornia.filters.box_blur(mask.to(torch.float32).unsqueeze(0), kernel_size=3)
    smooth_mask = (smooth_mask > 0.5).to(torch.uint8)

    _debug_dict["smooth_mask"] = smooth_mask

    _m, _med_nearest = plots.crop_to_common_size(smooth_mask, nearest)
    out = _med_nearest * _m

    building_blocks = _SegmentationMapBuildingBlocks(
        nearest=_med_nearest,
        raw_distances=_d,
        adjusted_distances=adjusted_dists,
        dot_grad_maps=_g,
        mask=smooth_mask
    )

    return out, building_blocks

# , hand_name: Literal["Right", "Left"]







@torch.no_grad
def compute_anatomical_mask(img_data: torch.Tensor, use_surface_optimization: bool=True, detector: Optional[hand_landmarks.Detector]=None) -> segmentation_map.SegmentationMap:
    """
    Compute segmentation mask

    Returns:
        out_seg_mask: torch.Tensor
        bboxes: Dict[str, segmentation_map.BBox]
        anatomy_maps: Dict[str, torch.Tensor]
    """
    if img_data.ndim != 4:
        raise RuntimeError(f"img_data.ndim must be 4. Got {img_data.ndim = }")
    
    std, mean = torch.std_mean(img_data.to("cuda"), dim=0)

    return compute_segmentation_map_from_std_mean(std=std, mean=mean, use_surface_optimization=use_surface_optimization, detector=detector)

@torch.no_grad
def compute_segmentation_map_from_std_mean(std: torch.Tensor, mean: torch.Tensor, use_surface_optimization: bool=True, detector: Optional[hand_landmarks.Detector]=None) -> tuple[segmentation_map.SegmentationMap, _SegmentationMapBuildingBlocks]:
    """
    Compute segmentation mask

    Returns:
        out_seg_mask: torch.Tensor
        bboxes: Dict[str, segmentation_map.BBox]
        anatomy_maps: Dict[str, torch.Tensor]
    """

    std_gpu = std.to("cuda")
    mean_gpu = mean.to("cuda")
    std_cpu = std.cpu()
    mean_cpu = mean.cpu()

    mask = hand_segmentation.get_hand_mask_from_std_mean(std=std, mean=mean)
    if mask.ndim == 4:
        mask = mask.squeeze(0)
    mask_cpu = mask.to("cpu")
    # plot(mask)
    # print(f"{img_data.shape = }, {mask.shape = }")
    gradient = _prepare_gradients(mean, mask)
    # plot(gradient)
# 
    # print(f"{img_data.shape = }, {mask.shape = }, {gradient.shape = }")
    
    if detector is None:
        detector = hand_landmarks.Detector()
    hands_landmarks_px = detector.detect(mean.cpu(), ema_alpha=None)

    markers_named = {hand_name:named_markers.add_custom_markers(hand_marks, mask=mask_cpu) for hand_name, hand_marks in hands_landmarks_px.items()}

    name_to_idx: Dict[str, int] = {}
    markers = {}
    for hand_name, (hand_marks, hand_mark_indices) in markers_named.items():
        markers[hand_name] = torch.stack(tuple(hand_marks.values()))
        if len(name_to_idx) == 0:
            name_to_idx |= hand_mark_indices
    
    markers_ema = detector.ema(markers=markers, alpha=0.9)
    markers_named_ema = {
        hand_name:{
            marker_name:hand_markers_ema[marker_idx] for marker_name, marker_idx in name_to_idx.items()
        } for hand_name, hand_markers_ema in markers_ema.items()
    }

    hands_segments_px: Dict[str, segments.HandSegments] = {hand_name:segments.build_segments(markers_px=hand_markers, mask=mask_cpu) for hand_name, hand_markers in markers_named_ema.items()}

    # hands_segments_px = {hand_name:segments_hand for hand_name, segments_hand in segments}

    # Crop to bbox of chosen hand
    bboxes: dict[str, segmentation_map.BBox] = {}
    out_seg_map = torch.zeros_like(mask).to("cuda")
    for hand_name, chosen_segments in hands_segments_px.items():
        H, W = mean_gpu.shape[-2:]
        coords = einops.rearrange(chosen_segments.get_segments_tensor(), "e n c -> (e n) c") # coords = [x, y]

        min_coord = coords.min(dim=0).values
        max_coord = coords.max(dim=0).values

        padding = 50 # px
        min_coord -= padding
        max_coord += padding
        
        min_coord = min_coord.clamp(min=0).to(torch.int)
        max_coord = max_coord.clamp(max=torch.tensor([W, H], device=max_coord.device)).to(torch.int)

        min_h = min_coord[-1]
        max_h = max_coord[-1]
        min_w = min_coord[-2]
        max_w = max_coord[-2]

        bbox = segmentation_map.BBox(
            min_h=min_h,
            max_h=max_h,
            min_w=min_w,
            max_w=max_w,
        )
        bboxes[hand_name] = bbox

        # print(f"h={max_h-min_h}, w={max_w-min_w}\nH={H}, W={W}")
        
        cropped_mask = mask[bbox.as_slice()]
        # plot(cropped_mask)
        current_hand_markers = markers_named_ema[hand_name]

        cut_points = torch.stack([
            current_hand_markers[L.thumb_vantage_wrist],
            current_hand_markers[L.index_vantage_wrist],
            current_hand_markers[L.middle_vantage_wrist],
            current_hand_markers[L.ring_vantage_wrist],
            current_hand_markers[L.pinky_vantage_wrist],
        ], dim=0) - min_coord.cpu().unsqueeze(0)
        
        wrist_dir = normalize(current_hand_markers[L.arm_spacer_middle_proximal] - current_hand_markers[L.arm_spacer_middle_distal])# * 100 # px
        wrist_cut_offset = project_point_onto_line(
            origin=current_hand_markers[L.middle_vantage_wrist], 
            vec=(current_hand_markers[L.wrist] - current_hand_markers[L.middle_vantage_wrist]), 
            dir=wrist_dir
        ) - current_hand_markers[L.middle_vantage_wrist]
        cut_points += wrist_cut_offset.unsqueeze(0)
        cropped_mask = hand_segmentation.crop_wrist_along_points(
            cropped_mask,
            wrist_cut_points=cut_points,
            index_metacarp_point=current_hand_markers[L.index_mcp]
        )
        # plot(cropped_mask)

        cropped_gradients = gradient[bbox.as_slice()]

        relative_segments = chosen_segments.relative_to(origin=min_coord)
        seg_map, _ = _closest_segment_mask(mask=cropped_mask, gradient=cropped_gradients, hand_segments=relative_segments, use_surface_optimization=use_surface_optimization)

        out_seg_map[bbox.as_slice()] = seg_map * cropped_mask

    if len(bboxes) == 2 and "right" in bboxes:
        both_hands = out_seg_map.clone().to(torch.int16)
        right_hand = both_hands[bboxes["right"].as_slice()]
        right_hand[right_hand > 0] = right_hand[right_hand > 0] + (out_seg_map.max() + 1)
        out_seg_map = both_hands

    return segmentation_map.SegmentationMap(
        segmentation_map=out_seg_map,
        _bboxes=bboxes,
    )



@torch.no_grad
def wip_refinement__compute_segmentation_map_from_std_mean(std: torch.Tensor, mean: torch.Tensor, use_surface_optimization: bool=True, detector: Optional[hand_landmarks.Detector]=None) -> segmentation_map.SegmentationMap:
    """
    Compute segmentation mask

    Returns:
        out_seg_mask: torch.Tensor
        bboxes: Dict[str, segmentation_map.BBox]
        anatomy_maps: Dict[str, torch.Tensor]
    """

    std_gpu = std.to("cuda")
    mean_gpu = mean.to("cuda")
    std_cpu = std.cpu()
    mean_cpu = mean.cpu()

    mask = hand_segmentation.get_hand_mask_from_std_mean(std=std, mean=mean)
    if mask.ndim == 4:
        mask = mask.squeeze(0)
    mask_cpu = mask.to("cpu")
    # plot(mask)
    # print(f"{img_data.shape = }, {mask.shape = }")
    gradient = _prepare_gradients(mean, mask)
    # plot(gradient)
# 
    # print(f"{img_data.shape = }, {mask.shape = }, {gradient.shape = }")
    
    if detector is None:
        detector = hand_landmarks.Detector()
    hands_landmarks_px = detector.detect(mean.cpu(), ema_alpha=None)

    markers_named = {hand_name:named_markers.add_custom_markers(hand_marks, mask=mask_cpu) for hand_name, hand_marks in hands_landmarks_px.items()}

    name_to_idx: Dict[str, int] = {}
    markers = {}
    for hand_name, (hand_marks, hand_mark_indices) in markers_named.items():
        markers[hand_name] = torch.stack(tuple(hand_marks.values()))
        if len(name_to_idx) == 0:
            name_to_idx |= hand_mark_indices
    
    markers_ema = detector.ema(markers=markers, alpha=0.9)
    markers_named_ema = {
        hand_name:{
            marker_name:hand_markers_ema[marker_idx] for marker_name, marker_idx in name_to_idx.items()
        } for hand_name, hand_markers_ema in markers_ema.items()
    }

    hands_segments_px: Dict[str, segments.HandSegments] = {hand_name:segments.build_segments(markers_px=hand_markers, mask=mask_cpu) for hand_name, hand_markers in markers_named_ema.items()}

    # hands_segments_px = {hand_name:segments_hand for hand_name, segments_hand in segments}

    # Crop to bbox of chosen hand
    out_building_blocks: dict[str, _SegmentationMapBuildingBlocks] = {}
    bboxes: dict[str, segmentation_map.BBox] = {}
    out_seg_map = torch.zeros_like(mask).to("cuda")
    for hand_name, chosen_segments in hands_segments_px.items():
        H, W = mean_gpu.shape[-2:]
        coords = einops.rearrange(chosen_segments.get_segments_tensor(), "e n c -> (e n) c") # coords = [x, y]

        min_coord = coords.min(dim=0).values
        max_coord = coords.max(dim=0).values

        padding = 50 # px
        min_coord -= padding
        max_coord += padding
        
        min_coord = min_coord.clamp(min=0).to(torch.int)
        max_coord = max_coord.clamp(max=torch.tensor([W, H], device=max_coord.device)).to(torch.int)

        min_h = min_coord[-1]
        max_h = max_coord[-1]
        min_w = min_coord[-2]
        max_w = max_coord[-2]

        bbox = segmentation_map.BBox(
            min_h=min_h,
            max_h=max_h,
            min_w=min_w,
            max_w=max_w,
        )
        bboxes[hand_name] = bbox

        # print(f"h={max_h-min_h}, w={max_w-min_w}\nH={H}, W={W}")
        
        cropped_mask = mask[bbox.as_slice()]
        # plot(cropped_mask)
        current_hand_markers = markers_named_ema[hand_name]

        cut_points = torch.stack([
            current_hand_markers[L.thumb_vantage_wrist],
            current_hand_markers[L.index_vantage_wrist],
            current_hand_markers[L.middle_vantage_wrist],
            current_hand_markers[L.ring_vantage_wrist],
            current_hand_markers[L.pinky_vantage_wrist],
        ], dim=0) - min_coord.cpu().unsqueeze(0)
        
        wrist_dir = normalize(current_hand_markers[L.arm_spacer_middle_proximal] - current_hand_markers[L.arm_spacer_middle_distal])# * 100 # px
        wrist_cut_offset = project_point_onto_line(
            origin=current_hand_markers[L.middle_vantage_wrist], 
            vec=(current_hand_markers[L.wrist] - current_hand_markers[L.middle_vantage_wrist]), 
            dir=wrist_dir
        ) - current_hand_markers[L.middle_vantage_wrist]
        cut_points += wrist_cut_offset.unsqueeze(0)
        cropped_mask = hand_segmentation.crop_wrist_along_points(
            cropped_mask,
            wrist_cut_points=cut_points,
            index_metacarp_point=current_hand_markers[L.index_mcp]
        )
        # plot(cropped_mask)

        cropped_gradients = gradient[bbox.as_slice()]

        relative_segments = chosen_segments.relative_to(origin=min_coord)
        seg_map, building_blocks = _closest_segment_mask(mask=cropped_mask, gradient=cropped_gradients, hand_segments=relative_segments, use_surface_optimization=use_surface_optimization)
        out_building_blocks[hand_name] = building_blocks
        out_seg_map[bbox.as_slice()] = seg_map * cropped_mask

    if len(bboxes) == 2 and "right" in bboxes:
        both_hands = out_seg_map.clone().to(torch.int16)
        right_hand = both_hands[bboxes["right"].as_slice()]
        right_hand[right_hand > 0] = right_hand[right_hand > 0] + (out_seg_map.max() + 1)
        out_seg_map = both_hands

    return segmentation_map.SegmentationMap(
        segmentation_map=out_seg_map,
        _bboxes=bboxes,
    ), building_blocks


def segmentation_mask_op(frames_per_mask: int=256) -> Callable[[torch.Tensor], torch.Tensor]:
    detector = hand_landmarks.Detector()
    @torch.no_grad
    def _segmentation_mask_op(frames: torch.Tensor) -> torch.Tensor:
        input_frames = einops.rearrange(frames, "(b t) 1 h w -> b t 1 h w", t=frames_per_mask)
        out_list = []
        for batch in input_frames:
            out_list.append(compute_anatomical_mask(batch, detector=detector).segmentation_map)
        
        return torch.stack(out_list, dim=0)
    return with_manual_op(_segmentation_mask_op, equivalent_slicing_op=(slice(None, None, frames_per_mask), slice(None), slice(None), slice(None)))

def distinct_colors_op() -> Callable[[torch.Tensor], torch.Tensor]:
    @torch.no_grad
    def _distinct_colors_op(frames: torch.Tensor) -> torch.Tensor:
        out_list = []
        for frame in frames:
        #     # logger.info(f"{batch.shape = }")
            out_list.append(mask_to_distinct_colors(mask=frame))
        # # logger.info(f"{len(out_list) = }")
        return torch.stack(out_list, dim=0)

    return with_manual_op(_distinct_colors_op, equivalent_slicing_op=(slice(None), slice(None), slice(None), slice(None)))


def distinct_segmentation_mask_op(frames_per_mask: int=256, stride: Optional[int]=None, use_surface_optimization: bool=True) -> Callable[[torch.Tensor], torch.Tensor]:
    detector = hand_landmarks.Detector()
    if stride is None:
        stride = frames_per_mask
    @torch.no_grad
    def _segmentation_mask_op(frames: torch.Tensor) -> torch.Tensor:
        input_frames_unfolded = frames.unfold(dimension=0, size=frames_per_mask, step=stride)
        input_frames = einops.rearrange(input_frames_unfolded, "b c h w t -> b t c h w")
        out_list = []
        for batch in input_frames:
            mask = compute_anatomical_mask(batch, use_surface_optimization=use_surface_optimization, detector=detector).segmentation_map
            out_list.append(mask_to_distinct_colors(mask=mask))
        return torch.stack(out_list, dim=0)
    return with_manual_op(_segmentation_mask_op, equivalent_slicing_op=(slice(0, - frames_per_mask // 2, stride), slice(None), slice(None), slice(None)))
    


