"""
DataPipes
"""
from datapipes import datasets, sinks
from datapipes.datapipe import DataPipe
from datapipes.ops import Ops
from datapipes.analysis import contrast, filters
from datapipes.plotting import plot
from datapipes.sic import sic
# from datapipes import cache_results
from datapipes import sinks
from datapipes.analysis import filters


__all__ = ["DataPipe", "Ops", "filters", "contrast", "Ops", "plot", "datasets", "sic", "cache_results", "sinks"]