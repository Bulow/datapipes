from __future__ import annotations


import numpy as np

import torch
import plotly.express as px
import plotly.graph_objects as go
from datapipes.analysis.hands import hand_landmarks, hand_anatomy, hand_segmentation, named_markers, segments

from datapipes.plotting import interactive_plots
import einops

from typing import Optional, Any, Dict, Tuple, List, Iterable, Sequence, Union

import torch.nn.functional as F
from datapipes.plotting.torch_colormap import TorchColormap
from pathlib import Path

from datapipes.plotting import map01

def _overlay_points_plotly(image: torch.Tensor, points: torch.Tensor, names: Optional[Iterable[str]]=None):
    """
    Overlays points onto a (1, H, W) tensor with custom tooltips.
    
    Args:
        image (torch.Tensor): Shape (1, H, W)
        points (torch.Tensor): Shape (N, 2) representing (x, y)
        names (list of str, optional): Names for each point tooltip
    """
    # 1. Image Preprocessing
    img_np = image.squeeze(0).cpu().numpy()
    
    # 2. Create base figure (binary_string=True improves performance)
    fig = px.imshow(img_np, color_continuous_scale='Viridis', binary_string=None)
    
    # 3. Extract coordinates
    x_coords = points[:, 0].cpu().numpy()
    y_coords = points[:, 1].cpu().numpy()

    colors = ['red' if i < 21 else 'green' for i in range(len(points))]
    
    # 4. Add the Scatter layer
    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text' if names else 'markers',
            marker=dict(
                color=colors, 
                size=10, 
                line=dict(width=2, color='white')
            ),
            # This handles the tooltips
            hovertext=names,
            hoverinfo="text+x+y", # Shows the name AND coordinates
            name='Points'
        )
    )
    
    # Adjust layout for a cleaner look
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )
    
    fig.show()


def _overlay_points_and_lines_plotly(image: torch.Tensor, points: torch.Tensor, names: Optional[Iterable[str]]=None, line_segments: Optional[torch.Tensor]=None, line_segments_names: Optional[Iterable[str]]=None):
    """
    Overlays points and line segments onto a (1, H, W) tensor with custom tooltips.
    """

    # Image
    img_np = image.squeeze(0).cpu().numpy()
    fig = px.imshow(
        img_np,
        color_continuous_scale='Viridis',
        binary_string=True
    )

    # Points
    x_coords = points[:, 0].cpu().numpy()
    y_coords = points[:, 1].cpu().numpy()

    colors = ['red' if i % (len(points) / 2) < 21 else 'green' for i in range(len(points))]
    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text' if names else 'markers',
            marker=dict(
                color=colors,
                size=10,
                line=dict(width=2, color='white')
            ),
            hovertext=names,
            hoverinfo="text+y+x",
            name='Points'
        )
    )
    
    # Line segments
    if line_segments is not None:
        segs = einops.rearrange(line_segments.cpu().numpy(), "p s c -> s p c") # point, segment, coordinate
        # for seg in segs:
        #     x = seg[:, 0]
        #     y = seg[:, 1]

        x = [seg[:, 0] for seg in segs]
        y = [seg[:, 1] for seg in segs]

        for xx, yy, name in zip(x, y, line_segments_names):

            fig.add_trace(
                go.Scatter(
                    x=xx,
                    y=yy,
                    mode='lines+text' if name else 'lines',
                    line=dict(color='purple', width=2),
                    hovertext=name,
                    hoverinfo="text",
                    showlegend=False,
                    name='Segments'
                )
            )

    # Show figure
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True
    )
    
    # fig.show()
    return fig

def visualize_hand_geometry(input_frames: torch.Tensor, overlay_geometry_background: Optional[torch.Tensor]=None, html_output_path: Optional[Path|str]=None, show_fig: bool=True):
    raw_img = input_frames.mean(0)
    result = hand_landmarks.detect_landmarks(raw_img)
    mask = hand_segmentation.get_hand_mask(input_frames).to("cuda")

    def get_points_and_lines_for_hand(hand_idx: int):
        points = hand_landmarks.landmarks_to_tensor(result, img_shape=mask.shape, hand_idx=hand_idx, coord_type="px")
        # rich.print(f"{points[hand_anatomy.L.wrist] = }")
        lines: segments.HandSegments = segments.build_segments(points, mask)
        
        markers, markers_idx = named_markers.add_custom_markers(points, mask=mask)
        points = torch.stack(tuple(markers.values()))
        _, h, w = raw_img.shape
        
        points_px = points# * torch.tensor([[w - 1, h - 1]], device=points.device)

        markers_px = {k:points_px[markers_idx[k]] for k in markers.keys()}

        return points_px, lines, markers_px
    
    all_points = []
    hand_segments = []
    point_names = []

    for points, lines, markers in (get_points_and_lines_for_hand(i) for i in range(2)):
        all_points.extend(points)
        hand_segments.append(lines)
        point_names.extend([name for name in markers.keys()])
    
    points = torch.stack(all_points, dim=0)
    lines = torch.cat(tuple(segs.get_segments_tensor() for segs in hand_segments), dim=1)
    line_names = list([name for hs in hand_segments for name in hs.segs.keys()])

    fig = _overlay_points_and_lines_plotly(image=overlay_geometry_background or raw_img, points=points, names=point_names, line_segments=lines, line_segments_names=line_names)

    if html_output_path is not None:
        interactive_plots.create_standalone_html_plot(fig, html_output_path)

    if show_fig:
        fig.show()

    return fig


def create_distinct_colormap(img: torch.Tensor) -> torch.Tensor:
    labels = torch.unique(img).to(torch.int64)

    # integer hash -> RGB bytes (deterministic, decorrelates nearby labels)
    x = labels
    x = (x ^ (x >> 16)) * 0x45D9F3B
    x = (x ^ (x >> 16)) * 0x45D9F3B
    x = (x ^ (x >> 16)) & 0xFFFFFFFF

    r = (x & 0xFF).to(torch.uint8)
    g = ((x >> 8) & 0xFF).to(torch.uint8)
    b = ((x >> 16) & 0xFF).to(torch.uint8)

    # keep colors away from extremes for readability: [32, 224]
    def stretch(c):
        c16 = c.to(torch.int16)
        return (32 + (c16 * 192 // 255)).to(torch.uint8)

    r, g, b = stretch(r), stretch(g), stretch(b)

    lut = torch.zeros((256, 3), dtype=torch.uint8, device="cuda")
    lut[labels.to(torch.long), 0] = r
    lut[labels.to(torch.long), 1] = g
    lut[labels.to(torch.long), 2] = b
    return lut

def get_edge_mask(m: torch.Tensor, boundary_thickness: int=1) -> torch.BoolTensor:
    H, W = m.shape[-2:]
    mm = m.to(torch.int16)
    edge = torch.zeros((H, W), dtype=torch.bool, device="cuda")

    edge[:, 1:] |= (mm[:, 1:] != mm[:, :-1])
    edge[:, :-1] |= (mm[:, :-1] != mm[:, 1:])
    edge[1:, :] |= (mm[1:, :] != mm[:-1, :])
    edge[:-1, :] |= (mm[:-1, :] != mm[1:, :])

    if boundary_thickness > 1:
        k = 2 * boundary_thickness + 1
        edge_f = edge.float()[None, None]
        edge = (F.max_pool2d(edge_f, kernel_size=k, stride=1, padding=k // 2)[0, 0] > 0)
    return edge

@torch.no_grad
def mask_to_distinct_colors(
    mask: torch.Tensor,
    add_boundaries: bool = True,
    boundary_color=(0, 0, 0),
    boundary_thickness: int = 1,
    overlay_background: Optional[torch.Tensor]=None, bg_strength: float = 0.1
) -> torch.Tensor:
    """
    Convert a uint8 segmentation mask (1,H,W) or (H,W) to a visually distinct RGB image (3,H,W) uint8.
    Each unique label gets a high-contrast, deterministic color.
    """
    if mask.dim() == 3 and mask.shape[0] == 1:
        m = mask[0]
    elif mask.dim() == 2:
        m = mask
    else:
        raise ValueError("mask must have shape (1,H,W) or (H,W)")

    if m.dtype != torch.uint8:
        raise ValueError("mask must be uint8")

    H, W = m.shape
    device = m.device

    lut = create_distinct_colormap(m)

    rgb = lut[m.to(torch.long)].permute(2, 0, 1).contiguous()  # (3,H,W)

    if add_boundaries:
        edge = get_edge_mask(m, boundary_thickness=boundary_thickness)
        # mm = m.to(torch.int16)
        # edge = torch.zeros((H, W), dtype=torch.bool, device=device)

        # edge[:, 1:] |= (mm[:, 1:] != mm[:, :-1])
        # edge[:, :-1] |= (mm[:, :-1] != mm[:, 1:])
        # edge[1:, :] |= (mm[1:, :] != mm[:-1, :])
        # edge[:-1, :] |= (mm[:-1, :] != mm[1:, :])

        # if boundary_thickness > 1:
        #     k = 2 * boundary_thickness + 1
        #     edge_f = edge.float()[None, None]
        #     edge = (F.max_pool2d(edge_f, kernel_size=k, stride=1, padding=k // 2)[0, 0] > 0)

        # IMPORTANT FIX: rgb[:, edge] has shape (3, N), so boundary color must be (3, 1) or (3,)
        bc = torch.tensor(boundary_color, dtype=torch.uint8, device=device).view(3, 1)
        rgb[:, edge] = bc  # broadcasts across N

    if overlay_background is not None:
        rgb_background = TorchColormap.apply(overlay_background, cmap_name="gray", v_min_max=True)
        rgb = map01(rgb_background) * map01(rgb_background * bg_strength * rgb)
    return rgb



def plot_segmentation_mask_plotly(
    mask: Union[torch.Tensor, np.ndarray],
    id_to_name: Dict[int, str],
    *,
    lut_rgb: Optional[Union[torch.Tensor, np.ndarray]]=None,
    title: str = "Segmentation mask",
    opacity: float = 1.0,
    show_colorbar: bool = False,
    html_output_path: Optional[Path|str]=None
) -> go.Figure:
    # --- Mask to 2D numpy ---
    if isinstance(mask, torch.Tensor):
        
        m = torch.empty_like(mask)
        m.copy_(mask, non_blocking=True)
        edges = get_edge_mask(m[0])
        print(f"{m.shape = }, {edges.shape = }")
        m[..., edges] = 0
        m = m.detach().cpu()
        if m.ndim == 3 and m.shape[0] == 1:
            z = m[0].numpy()
        elif m.ndim == 2:
            z = m.numpy()
        else:
            raise ValueError(f"Expected mask shape (1,H,W) or (H,W), got {tuple(m.shape)}")
    else:
        arr = np.asarray(mask)
        if arr.ndim == 3 and arr.shape[0] == 1:
            z = arr[0]
        elif arr.ndim == 2:
            z = arr
        else:
            raise ValueError(f"Expected mask shape (1,H,W) or (H,W), got {arr.shape}")

    z = z.astype(np.int32, copy=False)

    # --- LUT to numpy ---
    if lut_rgb is None:
        lut_rgb = create_distinct_colormap(mask)
    lut = lut_rgb.detach().cpu().numpy() if isinstance(lut_rgb, torch.Tensor) else np.asarray(lut_rgb)
    if lut.shape != (256, 3):
        raise ValueError(f"Expected lut_rgb shape (256,3), got {lut.shape}")
    lut = lut.astype(np.uint8, copy=False)

    # --- Stepwise colorscale in [0,1] matching z in [0,255] ---
    # Plotly normalizes z via z/255 (since zmin=0,zmax=255). We make each id i occupy
    # [i/255, (i+1)/255) with a tiny epsilon to keep monotonic stops.
    eps = 1e-6
    colorscale = []
    for i in range(256):
        r, g, b = map(int, lut[i])
        color = f"rgb({r},{g},{b})"

        lo = i / 255.0
        if i < 255:
            hi = (i + 1) / 255.0 - eps
        else:
            hi = 1.0

        # Use lists (not tuples) for compatibility across Plotly versions
        colorscale.append([lo, color])
        colorscale.append([hi, color])

    # --- Hover text (name + id) ---
    zmax = int(z.max())
    name_lut = np.full((zmax + 1,), "(unlabeled)", dtype=object)
    for k, v in id_to_name.items():
        kk = int(k)
        if 0 <= kk <= zmax:
            name_lut[kk] = str(v)

    names = name_lut[z] # creates a string per pixel. Stupidly slow. TODO: Improve?
    hovertext = "class: " + names + "<br>id: " + z.astype(str)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            zmin=0,
            zmax=255,
            colorscale=colorscale,
            hovertext=hovertext,
            hoverinfo="text",
            opacity=opacity,
            colorbar=dict(title="class id") if show_colorbar else None,
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed", scaleanchor="x", showgrid=False, zeroline=False),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    if html_output_path is not None:
        interactive_plots.create_standalone_html_plot(fig, html_output_path)
    return fig