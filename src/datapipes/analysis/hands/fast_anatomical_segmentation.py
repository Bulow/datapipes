import torch
import torch.nn.functional as F
from datapipes import sic, filters
from datapipes.plotting import plots, map01, plot, crop_to_common_size
import einops
from datapipes.analysis.hands import hand_anatomy, hand_landmarks, hand_segmentation
from typing import Dict, Tuple, Literal, Optional, Callable, Iterable
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
from contextlib import contextmanager
from functools import wraps
from datapipes.utils.benchmarking import MultiBlockTimer
from datapipes.analysis.hands.visualization import mask_to_distinct_colors

@dataclass(frozen=True, kw_only=True)
class SegmentBatch:
    segs: torch.Tensor
    weights: torch.Tensor
    biases: torch.Tensor
    name_to_value_dict: Dict[str, int]

@dataclass(frozen=True, kw_only=True)
class Bbox:
    min_h: torch.Tensor
    max_h: torch.Tensor
    min_w: torch.Tensor
    max_w: torch.Tensor

    @property
    def origin_coord(self) -> torch.Tensor:
        out = torch.stack((self.min_w, self.min_h))
        # print(f"origin_coord: {out}")
        return out

    def as_slice_tuple(self) -> Tuple[slice, slice]:
        return (
            slice(self.min_h, self.max_h),
            slice(self.min_w, self.max_w),
        )
    
    @property
    def height(self) -> int:
        return self.max_h - self.min_h
    
    @property
    def width(self) -> int:
        return self.max_w - self.min_w
    
    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.height, self.width))
    
    # @classmethod
    # def containing_all(cls, *bboxes: Iterable["Bbox"]) -> "Bbox":
    #     min_h = torch.stack(tuple(b.min_h for b in bboxes)).min()
    #     min_w = torch.stack(tuple(b.min_w for b in bboxes)).min()
    #     max_h = torch.stack(tuple(b.max_h for b in bboxes)).max()
    #     max_w = torch.stack(tuple(b.max_w for b in bboxes)).max()

    #     return cls(
    #         min_h=min_h,
    #         min_w=min_w,
    #         max_h=max_h,
    #         max_w=max_w,
    #     )
    

    @classmethod
    def get_max_size(cls, *bboxes: Iterable["Bbox"]) -> torch.Size:
        h_size = torch.stack(tuple(b.max_h - b.min_h for b in bboxes)).max()
        w_size = torch.stack(tuple(b.max_w - b.min_w for b in bboxes)).max()
        return torch.Size((h_size, w_size))
    
    @classmethod
    def to_max_size(cls, *bboxes: Iterable["Bbox"]) -> "Bbox":
        h_size, w_size = Bbox.get_max_size(*bboxes)

        return tuple(cls(
            min_h=b.min_h,
            min_w=b.min_w,
            max_h=b.min_h + h_size,
            max_w=b.min_w + w_size,
        ) for b in bboxes)
    
    def to_size(self, size: torch.Size) -> "Bbox":
        return Bbox(
            min_h=self.min_h,
            min_w=self.min_w,
            max_h=self.min_h + size[0],
            max_w=self.min_w + size[1],
        )
    
    def is_inside(self, other: "Bbox", margin: int = 0) -> bool:
        return all(
            self.min_h >= other.min_h + margin,
            self.min_w >= other.min_w + margin,
            self.max_h <= other.max_h - margin,
            self.max_w <= other.max_w - margin,
        )
    
    def __str__(self) -> str:
        return f"Bbox([{int(self.min_h)}:{int(self.max_h)}, {int(self.min_w)}:{int(self.max_w)}], shape=({int(self.height)}, {int(self.width)}))"

    def __repr__(self) -> str:
        return str(self)

def as_bchw(f: Callable):
    @wraps(f)
    def _inner(t: torch.Tensor, *args, **kwargs):
        assert t.ndim == 5
        b = t.shape[1]
        t = einops.rearrange(t, "s b c h w -> (s b) c h w")
        t = f(t, *args, **kwargs)
        t = einops.rearrange(t, "(s b) c h w -> s b c h w", b=b)
        return t
    return _inner

_watch_indent_level: int = 0
@contextmanager
def watch(name: str=""):
    t = MultiBlockTimer()
    global _watch_indent_level
    _watch_indent_level += 1
    with t:
        yield
    _watch_indent_level -= 1
    print(f"{f" {">" * _watch_indent_level} " if _watch_indent_level > 0 else ""}{name}: {t}")
    

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

    # print(f"{vecs.shape = }")
    # print(f"{mask.shape = }")
    maps = torch.zeros(size=(vecs.shape[-1], 1, *mask.shape[-2:]), device="cuda", dtype=torch.float)
    # print(f"{maps.shape = }")
    d = einops.rearrange(vecs, "c d -> d 1 c")
    maps[..., mask > 0] = d
    return maps

def _get_point_to_segment_distance(P, A, B, gradient: tuple[torch.Tensor], eps=1e-8):
    """
    P: [M,2] points
    A: [N,2] segment start
    B: [N,2] segment end
    Returns: distances [M,N], dot_grad [M,N]
    """
    gx, gy = gradient

    # print(f"{A.shape = }, {B.shape = }, {P.shape = }")

    # Expand for broadcasting: P[M,1,2], A[1,N,2], B[1,N,2]
    P = P[:, None, :, :]              # [M,1,2]
    A = A[:, :, None, :]              # [1,N,2]
    B = B[:, :, None, :]              # [1,N,2]

    # print(f"{A.shape = }, {B.shape = }, {P.shape = }")



    

    AB = B - A                     # [1,N,2]
    AP = P - A                     # [M,N,2]

    # print(f"{AB.shape = }, {AP.shape = }, {P.shape = }")


    # Project AP onto AB, clamp to segment
    ab2 = (AB * AB).sum(dim=-1, keepdim=True) #.clamp_min(eps)  # [1,N,1]
    t = (AP * AB).sum(dim=-1, keepdim=True) #/ ab2             # [M,N,1]

    

    # axial_coord = t

    t = t / ab2

    t = t.clamp(0.0, 1.0)
    closest = A + t * AB           # [M,N,2]

    vec_to_segment = (P - closest)

    # print(f"{ab2.shape = }, {t.shape = }, {vec_to_segment.shape = }")
    

    grads = torch.stack([gx, gy], dim=1)

    
    
    grads = einops.rearrange(grads, "f d n -> f 1 n d")

    # print(f"{gx.shape = }, {gy.shape = }, {grads.shape = }")

    dp = (vec_to_segment * grads).sum(-1).to(torch.float32)

    # print(f"{dp.shape = }")
          
    d = (vec_to_segment ** 2).sum(dim=-1).sqrt()  # [M,N]

    # TODO: Return local basis coordinates of pixels

    return d, dp

def _get_line_segment_distances(
    mask: torch.Tensor,
    gradient: torch.Tensor,
    segs: torch.Tensor, # (segments start_stop=2 coord2D=2)
    px_count_in_computation: int = -1,
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

    ggx = gradient[:, 0:1, :, :]
    ggy = gradient[:, 1:2, :, :]

    # print(f"{ggx.shape = }, {ggy.shape = }")
    # 
    
    segs = segs.to("cuda")

    any_mask = mask_bool.squeeze(1).any(0)

    # px_count = round(len(any_mask.flatten().nonzero()) * 1.2)
    # px_count = round(len(any_mask.flatten()) * 0.6)
    # actual_px_count = len(any_mask.flatten().nonzero())
    # print(f"{round(len(any_mask.flatten()) * 0.7) = }")
    # print(f"{px_count = }, {actual_px_count = }")

    # print(f"{any_mask.shape = }")

    g_mask = any_mask.unsqueeze(0) #einops.repeat(mask_bool, "h w -> 2 h w")
    gx, gy = ggx[:, g_mask], ggy[:, g_mask]
    # print(f"{pgx.shape = }, {pgy.shape = }")
    # gx = torch.zeros(size=(ggx.shape[0], px_count), dtype=torch.float32, device=gradient.device)
    # gx[..., :actual_px_count] = ggx[..., g_mask]

    # gy = torch.zeros(size=(ggy.shape[0], px_count), dtype=torch.float32, device=gradient.device)
    # gy[..., :actual_px_count] = ggy[..., g_mask]

    # gx = einops.rearrange(gx, "(f g) -> f g", f=mask.shape[0])
    # gy = einops.rearrange(gy, "(f g) -> f g", f=mask.shape[0])

    # print(f"{mask_bool.shape = }, {gy.shape = }, {gx.shape = }")
    

    
    segs = segs.to("cuda")
    # segs = hand_anatomy.build_segments(lm_px)
    A = segs[:, 0]
    B = segs[:, 1]

    # print(f"{A.shape = }, {B.shape = }")
    # print(f"{len(torch.where(mask_bool)) = }")

    # w = torch.where(mask_bool)
    # w = torch.vmap(torch.nonzero)(mask_bool.squeeze(1))

    # print(f"{w.shape = }")

    # # for g in w:
    # #     print(f"{g.shape = }")
    # import rich
    # rich.print(w)

    

    # Coordinates of mask pixels only (faster than all pixels)
    ys, xs = torch.where(any_mask)         # [M], [M]
    # print(f"{ys.shape = }, {xs.shape = }")
    # P = torch.zeros(size=(px_count, 2), dtype=torch.float32, device=mask.device)
    # P[:actual_px_count] = torch.stack([xs, ys], dim=-1)
    P = torch.stack([xs, ys], dim=-1).to(torch.float32)  # [M,2] in (x,y)

    

    P = einops.repeat(P.to(torch.float32), "p c -> r p c", r=mask_bool.shape[0], c=2)

    # print(f"{ys.shape = }, {xs.shape = }, {P.shape = }")

    

    dists, dot_grad = _get_point_to_segment_distance(P, A, B, gradient=(gx, gy))   # [M,N]
    
    # print(f"{dists.shape = }, {dot_grad.shape = }")
    
    
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
    # print(f"{image.shape = }")
    # Ensure the image has a batch dimension: (N, C, H, W)
    if image.dim() == 3:
        image = image.unsqueeze(0)

    grad = kornia.filters.spatial_gradient(image, mode="diff", order=1, normalized=True).squeeze(1)
    smooth_grad = grad
    # print(f"{smooth_grad.shape = }")
    smooth_grad = kornia.filters.guided_blur(guidance=image, input=grad, kernel_size=17, eps=1e-1, subsample=1)
    # plot(smooth_grad)
    return -smooth_grad[0]

def _prepare_gradients(img_data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    hand_min = img_data[mask > 0].min(0).values
    img_data[mask == 0] = hand_min
    img_data = map01(img_data)
    gradient = _compute_normal_map(img_data).squeeze(0)
    _g, _m = plots.crop_to_common_size(gradient, mask)
    gradient = (_g) * _m
    return gradient


def _soft_argmin(t: torch.Tensor) -> torch.Tensor:
    assert t.ndim == 5
    b = t.shape[1]
    t = einops.rearrange(t, "s b c h w -> (s b) c h w")


    


    p_smoothed = torch.softmax((-t) / t.shape[0], dim=0)
    # p_smoothed = kornia.filters.box_blur(p, kernel_size=3, separable=True)
    p_smoothed = k.gaussian_blur2d(p_smoothed, kernel_size=7, sigma=(2.0, 2.0))

    p_smoothed = einops.rearrange(p_smoothed, "(s b) c h w -> b s c h w", b=b)

    labels = torch.argmax(p_smoothed, dim=1)
    # print(f"{labels.shape = }")
    return labels

def _closest_segment_mask(mask: torch.Tensor, gradient: torch.Tensor, segs: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor, use_surface_optimization: bool=True) -> torch.Tensor:

    # gradient = kornia.filters.box_blur(input=gradient.unsqueeze(0), kernel_size=5, separable=True)[0]
    
    _m, _gradient = plots.crop_to_common_size(mask, gradient)
    
# with watch("_get_line_segment_distances"):
    dists, dot_grad = _get_line_segment_distances(
        mask=_m,
        gradient=_gradient,
        segs=segs,
    )
    

    # print(f"{_m.shape = }, {_gradient.shape = }")
    # print(f"{dists.shape = }, {dot_grad.shape = }")

    # Reconstruct maps from raw 1D pixels
    
    # Distance to each line segment
# with watch("reform dist_maps"):
    dist_maps = torch.zeros(size=[dists.shape[-2], mask.shape[0], 1, *mask.shape[-2:]], device=mask.device, dtype=torch.float)
    # print(f"{dist_maps.shape = }")
    d = einops.rearrange(dists, "f d c -> d (f c)")
    # print(f"{d.shape = }")
    

    any_mask = einops.repeat((mask > 0).any(dim=0), "1 h w -> f 1 h w", f=mask.shape[0])

    px_count = round(len(any_mask.flatten()) * 0.6)
    actual_px_count = len(any_mask.flatten().nonzero())


    # print(f"{(mask > 0).flatten().shape = }, {any_mask.flatten().shape = }, nonzero in any_mask: {any_mask.flatten().nonzero().shape[0] // 4}")

    # raise RuntimeError()

    # dist_maps[..., (mask > 0).any(dim=0)] = d
    # dist_maps.flatten()[any_mask.flatten()] = d
    dist_maps[..., any_mask] = d
    # dist_maps[..., any_mask] = d[:actual_px_count]

    # print(f"{dist_maps[..., any_mask].shape = }, {d.shape = }, {dist_maps.shape = }")

    # plot(*dist_maps[0])
    
    # raise RuntimeError()

# with watch("reform dot_grad_maps"):

    # gradient dot direction to each line segment
    dot_grad_maps = torch.zeros(size=[dot_grad.shape[-2], mask.shape[0], 1, *mask.shape[-2:]], device=mask.device, dtype=torch.float)
    dg = einops.rearrange(dot_grad, "f d c -> d (f c)")
    dot_grad_maps[..., any_mask] = dg[:actual_px_count]

    # print(f"{dist_maps.shape = }")
    # plot(*dot_grad_maps[8])
    
    neg = dot_grad_maps < 0
    dot_grad_maps = torch.sqrt(dot_grad_maps.abs() + 1e-5)
    # dot_grad_maps /= dot_grad_maps.max(dim=0).values
    dot_grad_maps[neg] *= -1

    # Adjust by surface affinity
    # print(f"{dot_grad_maps.shape = }, {weights.shape = }, {biases.shape = }")
    dot_grad_maps = (dot_grad_maps * weights) + biases

    # plot(*dot_grad_maps[8])


# with watch("compute modified distance metric"):
    # Penalize distances where the gradient points away from each line segment
    _d, _g = plots.crop_to_common_size(dist_maps, dot_grad_maps)

    pos_g = torch.clone(_g)
    pos_g[_g < 0] = 0

    neg_g = torch.clone(_g)
    neg_g[_g > 0] = 0

    delta_d = map01(_d) - ((map01(map01(-_d).square()) * (map01(pos_g))) ** 2) - ((map01(map01(-_d).square()) * (map01(neg_g))) ** 2)

    # print(f"{delta_d.shape = }")
    

    if use_surface_optimization:
        adjusted_dists = delta_d
    else:
        adjusted_dists = map01(_d)

# with watch("blur adjusted_dists"):

    adjusted_dists = einops.rearrange(adjusted_dists, "s b c h w -> (s b) c h w")

    
    adjusted_dists = kornia.filters.box_blur(adjusted_dists, kernel_size=17, separable=True)
    
    adjusted_dists = einops.rearrange(adjusted_dists, "(s b) c h w -> s b c h w", b = mask.shape[0])

    # Compute regions as index of closest line segment at each pixel
    # nearest = torch.argmin(adjusted_dists, dim=0)[0] + 1

# with watch("argmin"):
    nearest = _soft_argmin(adjusted_dists) + 1

    # print(f"{nearest.shape = }")

    
    # Apply mask
    # @as_bchw
    # def get_smoothed_mask(mask: torch.Tensor, kernel_size: int):
    #     smooth_mask = mask.to(torch.float16).unsqueeze(0)
    #     smooth_mask = kornia.filters.box_blur(mask.to(torch.float32).unsqueeze(0), kernel_size=kernel_size)
    #     smooth_mask = (smooth_mask > 0.5).to(torch.uint8)
    #     return smooth_mask
    
    # smooth_mask = get_smoothed_mask(mask, kernel_size=3)

    t = mask
    # print(f"{t.shape = }")
    # raise RuntimeError()
    # b = t.shape[1]
    # t = einops.rearrange(t, "s b c h w -> (s b) c h w")
    # smooth_mask = mask.to(torch.float16).unsqueeze(0)
    # with watch("smooth_mask"):
    #     smooth_mask = kornia.filters.box_blur(t.to(torch.float32), kernel_size=3)
    #     smooth_mask = (smooth_mask > 0.5).to(torch.uint8)
        # smooth_mask = einops.rearrange(smooth_mask, "(s b) c h w -> s b c h w", b=b)
        # print(f"{smooth_mask.shape = }")


    _m, _med_nearest = plots.crop_to_common_size(t, nearest)
    out = _med_nearest * _m

    # plot(*out)


    return out

# , hand_name: Literal["Right", "Left"]


@dataclass(frozen=True, kw_only=True)
class SegmentationResult:
    segmentation_mask: torch.Tensor
    bboxes: Dict[str, Dict[str, torch.Tensor]]
    segmentation_masks_by_hand: Dict[str, torch.Tensor]
    hand_segments: Dict[str, segments.HandSegments]

@torch.no_grad
def build_segments_from_raw_markers_single_hand(raw_markers: torch.Tensor, masks: torch.Tensor):
    assert len(raw_markers) == len(masks)

    @torch.no_grad
    def single_frame_segs(r: torch.Tensor, m: torch.Tensor) -> segments.HandSegments:
        custom_markers: Dict[str, torch.Tensor]
        custom_markers, _ = named_markers.add_custom_markers(r, mask=m)
        segs: segments.HandSegments = segments.build_segments(markers_px=custom_markers, mask=m)
        return segs
    
    segs_list: Iterable[segments.HandSegments] = [single_frame_segs(r=r, m=m) for r, m in zip(raw_markers, masks)]

    segs: torch.Tensor = torch.stack(tuple(s.get_segments_tensor() for s in segs_list), dim=0)
    weights: torch.Tensor = torch.stack(tuple(s.get_weights_tensor() for s in segs_list), dim=1)
    biases: torch.Tensor = torch.stack(tuple(s.get_biases_tensor() for s in segs_list), dim=1)

    name_to_value_dict: Dict[str, int] = segs_list[0].get_name_to_value_dict()

    return SegmentBatch(
        segs=segs.to("cuda"),
        weights=weights.to("cuda"),
        biases=biases.to("cuda"),
        name_to_value_dict=name_to_value_dict,
    )

def get_bbox(hand_seg_batch: SegmentBatch, mask: torch.Tensor, padding: int=20) -> Bbox:
    H, W = mask.shape[-2:]
    coords = einops.rearrange(hand_seg_batch.segs, "b e n c -> (b e n) c")
    # print(f"{coords.shape = }")

    min_coord = coords.min(dim=0).values
    max_coord = coords.max(dim=0).values

    # padding = padding # px
    min_coord -= padding
    max_coord += padding
    
    min_coord = min_coord.clamp(min=0).to(torch.int)
    max_coord = max_coord.clamp(max=torch.tensor([W, H], device=max_coord.device)).to(torch.int)
    # max_coord[-1] = H # Preserve wrist (assumes orientation)

    bbox = Bbox(
        min_h=min_coord[-1],
        max_h=max_coord[-1],
        min_w=min_coord[-2],
        max_w=max_coord[-2],
    )
    return bbox


# def bbox_hand(seg_batch: SegmentBatch, mask: torch.Tensor, gradient: torch.Tensor) -> Tuple[torch.Tensor, Bbox]:
#     with watch("bbox coords"):
#         # H, W = mask.shape[-2:]
#         # coords = einops.rearrange(seg_batch.segs, "b e n c -> (b e n) c")
#         # # print(f"{coords.shape = }")

#         # min_coord = coords.min(dim=0).values
#         # max_coord = coords.max(dim=0).values

#         # padding = 50 # px
#         # min_coord -= padding
#         # max_coord += padding
        
#         # min_coord = min_coord.clamp(min=0).to(torch.int)
#         # max_coord = max_coord.clamp(max=torch.tensor([W, H], device=max_coord.device)).to(torch.int)
#         # # max_coord[-1] = H # Preserve wrist (assumes orientation)

#         # bbox = Bbox(
#         #     min_h=min_coord[-1],
#         #     max_h=max_coord[-1],
#         #     min_w=min_coord[-2],
#         #     max_w=max_coord[-2],
#         # )
#         bbox = get_bbox(seg_batch, mask=mask, padding=50)

#         cropped_mask = mask[..., bbox.min_h:bbox.max_h, bbox.min_w:bbox.max_w]
#         cropped_gradients = gradient[..., bbox.min_h:bbox.max_h, bbox.min_w:bbox.max_w]

#         relative_segments = seg_batch.segs - bbox.origin_coord

#     with watch("compute_seg_mask"):
#         seg_mask = compute_seg_mask(
#             mask=cropped_mask,
#             gradients=cropped_gradients,
#             segs=relative_segments,
#             weights=seg_batch.weights,
#             biases=seg_batch.biases,
#         )

    # print(f"{seg_mask.shape = }")
    # return seg_mask, bbox

def crop(t: torch.Tensor, bbox: Bbox, bbox_size_in_computations: torch.Size) -> torch.Tensor:
    cropped_t = torch.zeros(size=[*t.shape[:-2], *bbox_size_in_computations], dtype=t.dtype, device=t.device)
    cropped_t[..., 0:bbox.height, 0:bbox.width] = t[..., bbox.min_h:bbox.max_h, bbox.min_w:bbox.max_w]
    return cropped_t


def compute_segments_bboxed(seg_batch: SegmentBatch, mask: torch.Tensor, gradient: torch.Tensor, bbox: Bbox, bbox_size_in_computations: torch.Size) -> Tuple[torch.Tensor, Bbox]:
    

    # cropped_mask = torch.zeros(size=[*mask.shape[:-2], *bbox_size_in_computations], dtype=mask.dtype, device=mask.device)
    # cropped_mask[..., 0:bbox.height, 0:bbox.width] = mask[..., bbox.min_h:bbox.max_h, bbox.min_w:bbox.max_w]
    cropped_mask = crop(t=mask, bbox=bbox, bbox_size_in_computations=bbox_size_in_computations)
    cropped_gradients = crop(t=gradient, bbox=bbox, bbox_size_in_computations=bbox_size_in_computations)

    # print(f"{bbox = }")
    
    # plot(*cropped_gradients)
    # cropped_gradients = gradient[..., bbox.min_h:bbox.max_h, bbox.min_w:bbox.max_w]

    relative_segments = seg_batch.segs - bbox.origin_coord

# with watch("compute_seg_mask"):
    seg_mask = compute_seg_mask(
        mask=cropped_mask,
        gradients=cropped_gradients,
        segs=relative_segments,
        weights=seg_batch.weights,
        biases=seg_batch.biases,
    )

    # print(f"{seg_mask.shape = }")
    return seg_mask[..., 0:bbox.height, 0:bbox.width]


@torch.no_grad
def compute_anatomical_masks(img_data: torch.Tensor, use_surface_optimization: bool=True, detector: Optional[hand_landmarks.Detector]=None, bbox_size_in_computations: Optional[torch.Size]=None) -> SegmentationResult:
    # """
    # Compute segmentation mask

    # Returns:
    #     out_seg_mask: torch.Tensor
    #     bboxes: Dict[str, Dict[str, int]]
    #     anatomy_maps: Dict[str, torch.Tensor]
    # """
# with watch("mask"):
    img_data_gpu = img_data.to("cuda", non_blocking=True)
    print(f"{img_data_gpu.shape = }")
    mask = hand_segmentation.get_hand_mask(img_data_gpu)
    print(f"{mask.shape = }")
# with watch("means"):
    means = img_data_gpu.mean(1)
    print(f"{means.shape = }")
# with watch("gradient"):
    gradient = _prepare_gradients(means, mask)
    print(f"{gradient.shape = }")
    if detector is None:
        detector = hand_landmarks.Detector()

# with watch("detect"):
    hands_landmarks_px = [detector.detect(m, ema_alpha=None) for m in means]
    markers_left = torch.stack(tuple(marks["left"] for marks in hands_landmarks_px))
    markers_right = torch.stack(tuple(marks["right"] for marks in hands_landmarks_px))

# with watch("add_custom_markers"):
    left: SegmentBatch = build_segments_from_raw_markers_single_hand(raw_markers=markers_left.to("cpu"), masks=mask.to("cpu"))
    right: SegmentBatch = build_segments_from_raw_markers_single_hand(raw_markers=markers_right.to("cpu"), masks=mask.to("cpu"))


# with watch("bbox"):
    left_bbox = get_bbox(left, mask=mask)
    right_bbox = get_bbox(right, mask=mask)

    if bbox_size_in_computations is None:
        bbox_size_in_computations = Bbox.get_max_size(left_bbox, right_bbox)

    out_mask = torch.zeros_like(mask)
    out_mask[..., *left_bbox.as_slice_tuple()] = compute_segments_bboxed(left, mask=mask, gradient=gradient, bbox=left_bbox, bbox_size_in_computations=bbox_size_in_computations)
    out_mask[..., *right_bbox.as_slice_tuple()] = compute_segments_bboxed(right, mask=mask, gradient=gradient, bbox=right_bbox, bbox_size_in_computations=bbox_size_in_computations)

    return out_mask

def distinct_mask_colors_batched(seg_masks: torch.Tensor) -> torch.Tensor:
    return [mask_to_distinct_colors(ds) for ds in seg_masks]


@torch.no_grad
def compute_seg_mask(mask: torch.Tensor, gradients: torch.Tensor, segs: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor, use_surface_optimization: bool=True) -> torch.Tensor:
    anatomy = _closest_segment_mask(mask=mask, gradient=gradients, segs=segs, weights=weights, biases=biases, use_surface_optimization=use_surface_optimization)

    # out_seg_mask[:, min_h:max_h, min_w:max_w] = anatomy
    # anatomy_maps[hand_name] = anatomy


    # return SegmentationResult(
    #     segmentation_mask=out_seg_mask,
    #     bboxes=bboxes,
    #     segmentation_masks_by_hand=anatomy_maps,
    #     hand_segments=hands_segments_px,
    # )

    return anatomy.to(torch.uint8)

def distinct_colors_op() -> Callable[[torch.Tensor], torch.Tensor]:
    @torch.no_grad
    def _distinct_colors_op(frames: torch.Tensor) -> torch.Tensor:
        out_list = []
        for frame in frames:
        #     # print(f"{batch.shape = }")
            out_list.append(mask_to_distinct_colors(mask=frame))
        # # print(f"{len(out_list) = }")
        return torch.stack(out_list, dim=0)

    return with_manual_op(_distinct_colors_op, equivalent_slicing_op=(slice(None), slice(None), slice(None), slice(None)))


def segmentation_mask_op(frames_per_mask: int=256, stride: int=128) -> Callable[[torch.Tensor], torch.Tensor]:
    detector = hand_landmarks.Detector()
    max_size = None
    @torch.no_grad
    def _segmentation_mask_op(frames: torch.Tensor) -> torch.Tensor:
        # max_size = torch.Size((frames.shape[-2] // 2, frames.shape[-1]))
        # input_frames = einops.rearrange(frames, "(b t) 1 h w -> b t 1 h w", t=frames_per_mask)
        # out_list = []
        # for batch in input_frames:
        #     out_list.append(compute_anatomical_mask(batch, detector=detector).segmentation_mask)
        
        # return torch.stack(out_list, dim=0)
        input_frames_unfolded = frames.unfold(dimension=0, size=frames_per_mask, step=stride)[0:-1]
        # print(f"{input_frames_unfolded.shape = }")
        input_frames = einops.rearrange(input_frames_unfolded, "b c h w t -> b t c h w")
        return compute_anatomical_masks(input_frames, use_surface_optimization=True, detector=detector)
    return with_manual_op(_segmentation_mask_op, equivalent_slicing_op=(slice(frames_per_mask // 2, -frames_per_mask // 2, stride), slice(None), slice(None), slice(None)))
    # return with_manual_op(_segmentation_mask_op, equivalent_slicing_op=(slice(None, None, frames_per_mask), slice(None), slice(None), slice(None)))

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
        # out_list = []
        # for batch in input_frames:
        #     # print(f"{batch.shape = }")
        #     mask = compute_anatomical_mask(batch, use_surface_optimization=use_surface_optimization, detector=detector).segmentation_mask
        #     # print(mask)
        #     out_list.append(mask_to_distinct_colors(mask=mask))
        #     # out_list.append(mask_to_distinct_colors(mask, overlay_background=batch.mean(0)))
        # # print(f"{len(out_list) = }")
        # return torch.stack(out_list, dim=0)

        return compute_anatomical_masks(input_frames, use_surface_optimization=True)

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




