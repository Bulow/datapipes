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

    # Project AP onto AB, clamp to segment
    ab2 = (AB * AB).sum(dim=-1, keepdim=True) #.clamp_min(eps)  # [1,N,1]
    t = (AP * AB).sum(dim=-1, keepdim=True) #/ ab2             # [M,N,1]

    axial_coord = t

    t = t / ab2

    t = t.clamp(0.0, 1.0)
    closest = A + t * AB           # [M,N,2]

    vec_to_segment = (P - closest)

    grads = torch.stack([gx, gy])
    grads = einops.rearrange(grads, "d n -> n 1 d")

    dp = (vec_to_segment * grads).sum(-1).to(torch.float32)

    d = (vec_to_segment ** 2).sum(dim=-1).sqrt()  # [M,N]

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

    grad = kornia.filters.spatial_gradient(image, mode="diff", order=1, normalized=True)[0]
    smooth_grad = grad
    smooth_grad = kornia.filters.guided_blur(guidance=image, input=grad, kernel_size=17, eps=1e-1, subsample=1)
    # plot(smooth_grad)
    return -smooth_grad[0]

def _prepare_gradients(img_data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    hand_min = img_data[mask > 0].min()
    img_data[mask == 0] = hand_min
    img_data = map01(img_data)
    gradient = _compute_normal_map(img_data).squeeze(0)
    _g, _m = plots.crop_to_common_size(gradient, mask)
    gradient = (_g) * _m
    return gradient


def _soft_argmin(t: torch.Tensor) -> torch.Tensor:
    p_smoothed = torch.softmax((-t) / t.shape[0], dim=0)
    # p_smoothed = kornia.filters.box_blur(p, kernel_size=3, separable=True)
    p_smoothed = k.gaussian_blur2d(p_smoothed, kernel_size=7, sigma=(2.0, 2.0))
    labels = torch.argmax(p_smoothed, dim=0)
    # print(f"{labels.shape = }")
    return labels

def _closest_segment_mask(mask: torch.Tensor, gradient: torch.Tensor, hand_segments: segments.HandSegments, use_surface_optimization: bool=True) -> torch.Tensor:

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

    neg = dot_grad_maps < 0
    dot_grad_maps = torch.sqrt(dot_grad_maps.abs() + 1e-5)
    # dot_grad_maps /= dot_grad_maps.max(dim=0).values
    dot_grad_maps[neg] *= -1

    # Adjust by surface affinity
    dot_grad_maps = (dot_grad_maps * hand_segments.get_weights_tensor()) + hand_segments.get_biases_tensor()

    # Penalize distances where the gradient points away from each line segment
    _d, _g = plots.crop_to_common_size(dist_maps, dot_grad_maps)

    pos_g = torch.clone(_g)
    pos_g[_g < 0] = 0

    neg_g = torch.clone(_g)
    neg_g[_g > 0] = 0

    delta_d = map01(_d) - ((map01(map01(-_d).square()) * (map01(pos_g))) ** 2) - ((map01(map01(-_d).square()) * (map01(neg_g))) ** 2)

    if use_surface_optimization:
        adjusted_dists = delta_d
    else:
        adjusted_dists = map01(_d)

    adjusted_dists = kornia.filters.box_blur(adjusted_dists, kernel_size=17, separable=True)

    # Compute regions as index of closest line segment at each pixel
    # nearest = torch.argmin(adjusted_dists, dim=0)[0] + 1
    nearest = _soft_argmin(adjusted_dists)[0] + 1

    # Apply mask
    smooth_mask = mask.to(torch.float16).unsqueeze(0)
    smooth_mask = kornia.filters.box_blur(mask.to(torch.float32).unsqueeze(0), kernel_size=3)
    smooth_mask = (smooth_mask > 0.5).to(torch.uint8)

    _m, _med_nearest = plots.crop_to_common_size(smooth_mask, nearest)
    out = _med_nearest * _m

    return out

# , hand_name: Literal["Right", "Left"]


@dataclass(frozen=True, kw_only=True)
class SegmentationResult:
    segmentation_mask: torch.Tensor
    bboxes: Dict[str, Dict[str, torch.Tensor]]
    segmentation_masks_by_hand: Dict[str, torch.Tensor]
    hand_segments: Dict[str, segments.HandSegments]

# @torch.no_grad
# def compute_anatomical_mask(img_data: torch.Tensor, use_surface_optimization: bool=True, detector: Optional[hand_landmarks.Detector]=None) -> SegmentationResult:
#     """
#     Compute segmentation mask

#     Returns:
#         out_seg_mask: torch.Tensor
#         bboxes: Dict[str, Dict[str, int]]
#         anatomy_maps: Dict[str, torch.Tensor]
#     """

@torch.no_grad
def compute_seg_mask(mask: torch.Tensor, gradients: torch.Tensor, segs: segments.HandSegments, use_surface_optimization: bool=True) -> torch.Tensor:
    anatomy = _closest_segment_mask(mask=mask, gradient=gradients, hand_segments=segs, use_surface_optimization=use_surface_optimization)

    # out_seg_mask[:, min_h:max_h, min_w:max_w] = anatomy
    # anatomy_maps[hand_name] = anatomy


    # return SegmentationResult(
    #     segmentation_mask=out_seg_mask,
    #     bboxes=bboxes,
    #     segmentation_masks_by_hand=anatomy_maps,
    #     hand_segments=hands_segments_px,
    # )

    return anatomy.to(torch.uint8)


def segmentation_mask_op(frames_per_mask: int=256) -> Callable[[torch.Tensor], torch.Tensor]:
    detector = hand_landmarks.Detector()
    @torch.no_grad
    def _segmentation_mask_op(frames: torch.Tensor) -> torch.Tensor:
        input_frames = einops.rearrange(frames, "(b t) 1 h w -> b t 1 h w", t=frames_per_mask)
        out_list = []
        for batch in input_frames:
            out_list.append(compute_anatomical_mask(batch, detector=detector).segmentation_mask)
        
        return torch.stack(out_list, dim=0)
    return with_manual_op(_segmentation_mask_op, equivalent_slicing_op=(slice(None, None, frames_per_mask), slice(None), slice(None), slice(None)))

def distinct_segmentation_mask_op(frames_per_mask: int=256, stride: Optional[int]=None, use_surface_optimization: bool=True) -> Callable[[torch.Tensor], torch.Tensor]:
    detector = hand_landmarks.Detector()
    if stride is None:
        stride = frames_per_mask
    @torch.no_grad
    def _segmentation_mask_op(frames: torch.Tensor) -> torch.Tensor:
        # input_frames = einops.rearrange(frames, "(b t) 1 h w -> b t 1 h w", t=frames_per_mask)
        input_frames_unfolded = frames.unfold(dimension=0, size=frames_per_mask, step=stride)
        # print(f"{input_frames_unfolded.shape = }")
        input_frames = einops.rearrange(input_frames_unfolded, "b c h w t -> b t c h w")
        # print(f"{frames.shape = }")
        # print(f"{input_frames.shape = }")
        # raise RuntimeError()
        out_list = []
        for batch in input_frames:
            # print(f"{batch.shape = }")
            mask = compute_anatomical_mask(batch, use_surface_optimization=use_surface_optimization, detector=detector).segmentation_mask
            # print(mask)
            out_list.append(mask_to_distinct_colors(mask=mask))
            # out_list.append(mask_to_distinct_colors(mask, overlay_background=batch.mean(0)))
        # print(f"{len(out_list) = }")
        return torch.stack(out_list, dim=0)
    return with_manual_op(_segmentation_mask_op, equivalent_slicing_op=(slice(0, - frames_per_mask // 2, stride), slice(None), slice(None), slice(None)))
    

# def compute_anatomical_markers(image: torch.Tensor, mode: Literal["single_flat_dict", "dict_per_hand"]="dict_per_hand", detector: Optional[hand_landmarks.Detector]=None) -> Dict[str, torch.Tensor]|Dict[str, Dict[str, torch.Tensor]]:
#     # landmarks = hand_landmarks.detect_landmarks(image, detector=detector)
#     # hands = {cat[0].category_name:cat[0].index for cat in landmarks.handedness}
#     # hands = {k:hand_landmarks.landmarks_to_tensor(landmarks, hand_idx=v) for k, v in hands.items()}
#     # hands = {k:hand_anatomy.add_custom_markers(v) for k, v in hands.items()}
#     # hands = {k:hand_anatomy.get_poi_name_to_coords_dict(v) for k, v in hands.items()}

#     raw_landmarks_mediapipe_fmt = hand_landmarks.detect_landmarks(img_data=image, detector=detector)

#     hand_indices = {cat[0].category_name:cat[0].index for cat in raw_landmarks_mediapipe_fmt.handedness}
#     hands_landmarks_px = {hand_name:hand_landmarks.landmarks_to_tensor(raw_landmarks_mediapipe_fmt, img_shape=image.shape, hand_idx=idx, coord_type="px").to("cuda") for hand_name, idx in hand_indices.items()}

#     # hands_landmarks_px = hand_anatomy.add_custom_markers(hands_landmarks_px)

#     return hands_landmarks_px
    # match mode:
    #     case "dict_per_hand":
    #         return hands_landmarks_px
    #     case "single_flat_dict":
    #         # flat_hands = {f"{k}_{hand_name[0]}":v for hand_name, hand in hands.items() for k, v in hand.items()}
    #         return flat_hands
    #     case _:
    #         raise ValueError(f"mode=\"{mode}\" is not supported")




