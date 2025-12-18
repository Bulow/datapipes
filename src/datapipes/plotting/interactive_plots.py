from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
from icecream import ic
from torchvision.transforms.functional import to_pil_image
import torch
import einops
import numpy as np

from imageio import imwrite

from datapipes.sic import sic

from pathlib import Path

import math

from datapipes.plotting.torch_colormap import TorchColormap

from typing import Optional, Tuple, List, Literal, Callable, Any, Dict
import plotly
import plotly.express as px
import plotly.graph_objects as go

def per_frame_mean(frames: torch.Tensor):
    return frames.reshape((frames.shape[0], -1)).mean(-1)


def plot_1D(frames: torch.Tensor) -> go.Figure:
    return px.line(per_frame_mean(frames).cpu().numpy())

def animate(frames: torch.Tensor, cmap: str = "Viridis") -> go.Figure:
    print(f"animate: {frames.shape}")
    fig = px.imshow(
        frames.squeeze(1).cpu().numpy(), 
        animation_frame=0, 
        # binary_string=True, 
        labels=dict(animation_frame="frame"), contrast_rescaling="minmax", 
        color_continuous_scale=cmap
    )
    fig.show()
    return fig


def imshow_pois(
    image: torch.Tensor,                       # (1,H,W) or (3,H,W) or (H,W) or (H,W,3)
    pois: Dict[str, Tuple[float, float]],       # {name: (x_norm, y_norm)} in [0,1]
    marker_size: int = 10,
    show_labels: bool = True,
    title: Optional[str] = None,
) -> go.Figure:
    arr = image.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):          # (C,H,W) -> (H,W,C)
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]

    H, W = arr.shape[:2]

    names = list(pois.keys())
    xs = [pois[n][0] * (W - 1) for n in names]
    ys = [pois[n][1] * (H - 1) for n in names]

    fig = px.imshow(arr, title=title, color_continuous_scale="Viridis")

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text" if show_labels else "markers",
            text=names if show_labels else None,
            textposition="top center",
            textfont=dict(color='white'),
            marker=dict(
                size=marker_size,
                color=list(range(len(xs))),
                colorscale="Turbo",
                line=dict(width=1, color="white"),
            ),
            hovertemplate="<b>%{text}</b><br>x=%{x:.1f}px<br>y=%{y:.1f}px<extra></extra>",
        )
    )

    fig.update_layout(margin=dict(l=0, r=0, t=30 if title else 0, b=0))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    return fig


def imshow_region_mask(
    mask: torch.Tensor,                  # (H,W) or (1,H,W)
    region_names: Dict[int, str],         # {value: "name"}
    title: Optional[str] = None,
) -> go.Figure:
    m = mask.detach().cpu().numpy()
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    m = m.astype(np.int32)

    labels = np.vectorize(lambda v: region_names.get(int(v), str(int(v))), otypes=[object])(m)

    fig = px.imshow(
        m,
        color_continuous_scale="Turbo",
        title=title,
    )
    fig.update_traces(customdata=labels, hovertemplate="%{customdata}<extra></extra>")

    fig.update_layout(margin=dict(l=0, r=0, t=30 if title else 0, b=0))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    return fig

def create_standalone_html_plot(fig: go.Figure, out_path: Path|str):
    plotly.io.write_html(
        fig=fig.update_layout(dragmode='pan', hovermode='closest', margin=dict(l=0, r=0, t=0, b=0)),
        file=out_path,
        include_plotlyjs=True,
        full_html=True,
        config={"scrollZoom": True}
    )

def imshow_default(frame) -> go.Figure:
    frame = frame.cpu().numpy() if isinstance(frame, torch.Tensor) else frame
    while frame.shape[0] == 1:
        frame = frame[0]
    return px.imshow(frame).update_layout(dragmode='pan', hovermode='closest', margin=dict(l=0, r=0, t=0, b=0))