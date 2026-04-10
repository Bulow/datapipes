"""
Plot tensors
"""

# from datapipes.plotting import plots
from datapipes.plotting.plots import plot, plot_T, map01, crop_to_common_size, qtile
from datapipes.plotting.torch_colormap import TorchColormap as Colormap
__all__ = ["plot", "plot_T", "map01", "crop_to_common_size", "qtile", "Colormap"]