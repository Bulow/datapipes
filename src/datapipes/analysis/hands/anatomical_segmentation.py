import torch
import torch.nn.functional as F
from datapipes import sic, filters
from datapipes.plotting import plots, map01, plot
import einops
from datapipes.analysis.hands import hand_anatomy, hand_landmarks, hand_segmentation
from typing import Dict, Tuple, Literal
import kornia

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
    ab2 = (AB * AB).sum(dim=-1, keepdim=True).clamp_min(eps)  # [1,N,1]
    t = (AP * AB).sum(dim=-1, keepdim=True) / ab2             # [M,N,1]
    t = t.clamp(0.0, 1.0)
    closest = A + t * AB           # [M,N,2]

    vec_to_segment = (P - closest)

    grads = torch.stack([gx, gy])
    grads = einops.rearrange(grads, "d n -> n 1 d")
    vec_to_segment_dir = vec_to_segment / (vec_to_segment**2).sum(-1).unsqueeze(-1).sqrt()

    # Direction to closest point in segment dotted with gradient at each pixel location
    dp = (vec_to_segment_dir * grads).sum(-1).to(torch.float32)
    # print(dp.shape)
    # Distance from each pixel to the closest point in each segment

    # vec_to_segment = vec_to_segment - dp.unsqueeze(-1) * 16

    d = (vec_to_segment ** 2).sum(dim=-1).sqrt()  # [M,N]

    return d, dp

def _get_line_segment_distances(
    mask: torch.Tensor,
    gradient: torch.Tensor,
    segments: torch.Tensor # (segments start_stop=2 coord2D=2)
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

    segs = segments # TODO: remove
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

    # Define Sobel kernels for gradient computation.
    kernel_x = torch.tensor([[[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]]], dtype=image.dtype, device=image.device)
    kernel_y = torch.tensor([[[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]]], dtype=image.dtype, device=image.device)

    # Reshape kernels to match conv2d weight shape: (out_channels, in_channels, H, W)
    kernel_x = kernel_x.unsqueeze(0)  # Shape: (1, 1, 3, 3)
    kernel_y = kernel_y.unsqueeze(0)  # Shape: (1, 1, 3, 3)
    
    # Compute gradients along x and y directions
    grad_x = F.conv2d(image, kernel_x, padding=1)
    grad_y = F.conv2d(image, kernel_y, padding=1)
    
    # Compute the normal vectors as n = (-grad_x, -grad_y, 1)
    # The result has shape (N, 3, H, W)
    n = torch.cat((-grad_x, -grad_y, torch.ones_like(grad_x)), dim=1)
    
    # Normalize the normal vectors
    # n_norm = torch.sqrt(n[:, 0:1] ** 2 + n[:, 1:2] ** 2 + n[:, 2:3] ** 2)
    n_norm = (n.square().sum(1, keepdim=True).sqrt())
    normal_map = n / (n_norm + 1e-6)
    # normal_map = n
    # Remove the batch dimension and return the normal map with shape (3, H, W)
    return normal_map.squeeze(0)

def _prepare_gradients(img_data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Log space to be intensity scale invariant
    img_data = img_data.log()

    # Filters before gradient extraction. 
    
    # Dilation pushes the mask edge gradient discontinuity out of the preserved area
    # img_data = kornia.morphology.dilation(img_data.unsqueeze(0), torch.ones(size=(2, 2), dtype=torch.float32, device="cuda"), engine="convolution")#.to(torch.uint8)

    # Median creates a sharp divide between fingers that are touching
    # img_data = kornia.filters.median_blur(img_data, kernel_size=(11, 11))


    # Disk blur removes remaining high frequency noise
    # img_data = filters.blurs.uniform_disk_blur(img_data, kernel_size=3)
    # img_data = kornia.filters.box_blur(img_data, kernel_size=(5, 5))
    # Normalize
    img_data = map01(img_data)

    # Compute normalized gradients using a sobel filter
    gradient = _compute_normal_map(img_data).squeeze(0)

    
    # gradient[gradient > 0] = 1
    # gradient[gradient < 0] = -1

    # gradient

    # Apply mask
    _g, _m = plots.crop_to_common_size(gradient, mask)
    gradient = _g * _m
    # plots.plot_raw(gradient[0], gradient[1], quantiles=(0.02, 0.98), cmap="coolwarm")
    return gradient


def _closest_segment_mask(mask: torch.Tensor, gradient: torch.Tensor, segments: torch.Tensor, use_surface_optimization: bool=True) -> torch.Tensor:
    _m, _gradient = plots.crop_to_common_size(mask, gradient)
    dists, dot_grad = _get_line_segment_distances(
        mask=_m[0],
        gradient=_gradient,
        segments=segments
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

    dot_grad_maps = F.sigmoid(dot_grad_maps * 30) - 0.5


    dot_grad_maps = kornia.filters.median_blur(dot_grad_maps, kernel_size=(9, 9))

    dot_grad_maps = kornia.filters.median_blur(dot_grad_maps, kernel_size=(9, 9))
    dot_grad_maps = kornia.filters.box_blur(dot_grad_maps, kernel_size=(9, 9))

    plt_idx = 21


    # Attenuate gradients of non digit segments
    dot_grad_maps[17:] *= 0.1

    # Penalize distances where the gradient points away from each line segment
    _d, _g = plots.crop_to_common_size(dist_maps, dot_grad_maps)
    # plots.plot_raw(_g[8], quantiles=(0.02, 0.98), cmap="coolwarm")

    delta_d = -((_d.sqrt() * (F.sigmoid(_g * 7) - 0.5)) * (15))
    delta_d[delta_d < 0] *= 0

    

    adjusted_dists = _d
    if use_surface_optimization:
        adjusted_dists += delta_d

    # Blur adjusted distances to remove disconnected pockets and increase region compactness
    adjusted_dists = kornia.filters.box_blur(adjusted_dists, kernel_size=[8, 8])
    adjusted_dists = kornia.filters.median_blur(adjusted_dists, kernel_size=[11, 11])

    adjusted_dists = kornia.filters.box_blur(adjusted_dists, kernel_size=[8, 8])


    adjusted_dists = kornia.filters.median_blur(adjusted_dists, kernel_size=[11, 11])

    adjusted_dists = kornia.filters.box_blur(adjusted_dists, kernel_size=[8, 8])
    adjusted_dists = kornia.filters.median_blur(adjusted_dists, kernel_size=[11, 11])

    adjusted_dists = kornia.filters.box_blur(adjusted_dists, kernel_size=[8, 8])

    # plot(med_adjusted_dists[5:9])


    # Compute regions as index of closest line segment at each pixel
    nearest = torch.argmin(adjusted_dists, dim=0)[0] + 1

    # Apply mask
    smooth_mask = kornia.filters.box_blur(mask.to(torch.float32).unsqueeze(0), kernel_size=[3, 3])
    smooth_mask = (smooth_mask > 0.5).to(torch.uint8)
    _m, _med_nearest = plots.crop_to_common_size(smooth_mask, nearest)
    out = _med_nearest * _m

    return out

# , hand_name: Literal["Right", "Left"]
def compute_anatomical_mask(img_data: torch.Tensor, use_surface_optimization: bool=True) -> Tuple[torch.Tensor, Dict[str, int], Dict[Literal["left", "right"], torch.Tensor]]:
    mask = hand_segmentation.get_hand_mask(img_data)
    gradient = _prepare_gradients(img_data.std(0), mask)
    raw_landmarks_mediapipe_fmt = hand_landmarks.detect_landmarks(img_data=img_data.mean(0))

    hand_indices = {cat[0].category_name:cat[0].index for cat in raw_landmarks_mediapipe_fmt.handedness}
    hands_landmarks_normalized = {hand_name:hand_landmarks.landmarks_to_tensor(raw_landmarks_mediapipe_fmt, hand_idx=idx) for hand_name, idx in hand_indices.items()}

    hands_segments_px = {hand_name:hand_anatomy.build_segments(normalized_landmarks=normalized_landmarks_on_hand, img_width=mask.shape[-1], img_height=mask.shape[-2]) for hand_name, normalized_landmarks_on_hand in hands_landmarks_normalized.items()}

    # Crop to bbox of chosen hand
    # chosen_segments = hands_segments_px[hand_name]
    bboxes = {}
    anatomy_maps = {}
    out_seg_mask = torch.zeros_like(mask)
    for hand_name, chosen_segments in hands_segments_px.items():
        H, W = img_data.shape[-2:]
        coords = einops.rearrange(chosen_segments, "e n c -> (e n) c") # coords = [x, y]

        min_coord = coords.min(dim=0).values
        max_coord = coords.max(dim=0).values

        padding = 50 # px
        min_coord -= padding
        max_coord += padding
        
        min_coord = min_coord.clamp(min=0).to(torch.int)
        max_coord = max_coord.clamp(max=torch.tensor([W, H], device=max_coord.device)).to(torch.int)
        # max_coord[-1] = H # Preserve wrist (assumes orientation)

        min_h=min_coord[-1]
        max_h=max_coord[-1]
        min_w=min_coord[-2]
        max_w=max_coord[-2]

        bboxes[hand_name] = dict(
            min_h=min_h,
            max_h=max_h,
            min_w=min_w,
            max_w=max_w,
        )

        # print(f"h={max_h-min_h}, w={max_w-min_w}\nH={H}, W={W}")
        
        # cropped_img_data = img_data[..., min_coord[-1]:max_coord[-1], min_coord[-2]:max_coord[-2]]
        cropped_mask = mask[..., min_coord[-1]:max_coord[-1], min_coord[-2]:max_coord[-2]]
        cropped_gradients = gradient[..., min_coord[-1]:max_coord[-1], min_coord[-2]:max_coord[-2]]

        relative_coords = coords - min_coord
        relative_segments = einops.rearrange(relative_coords, "(e n) c -> e n c", e=2)

        anatomy = _closest_segment_mask(mask=cropped_mask, gradient=cropped_gradients, segments=relative_segments, use_surface_optimization=use_surface_optimization)
        # print(anatomy.shape)
        out_seg_mask[:, min_h:max_h, min_w:max_w] = anatomy
        anatomy_maps[hand_name] = anatomy
    return out_seg_mask, bboxes, anatomy_maps

def compute_anatomical_markers(image: torch.Tensor, mode: Literal["single_flat_dict", "dict_per_hand"]="dict_per_hand") -> Dict[str, torch.Tensor]|Dict[str, Dict[str, torch.Tensor]]:
    landmarks = hand_landmarks.detect_landmarks(image)
    hands = {cat[0].category_name:cat[0].index for cat in landmarks.handedness}
    hands = {k:hand_landmarks.landmarks_to_tensor(landmarks, hand_idx=v) for k, v in hands.items()}
    hands = {k:hand_anatomy.add_custom_markers(v) for k, v in hands.items()}
    hands = {k:hand_anatomy.get_poi_name_to_coords_dict(v) for k, v in hands.items()}

    match mode:
        case "dict_per_hand":
            return hands
        case "single_flat_dict":
            flat_hands = {f"{k}_{hand_name[0]}":v for hand_name, hand in hands.items() for k, v in hand.items()}
            return flat_hands
        case _:
            raise ValueError(f"mode=\"{mode}\" is not supported")




