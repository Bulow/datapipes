"""
DataPipes
"""
from datapipes import datasets, filters, contrast, sinks
from datapipes.datapipe import DataPipe
from datapipes.ops import Ops
from datapipes import filters
from datapipes import contrast
from datapipes.plotting import plot
from datapipes import sample_data
from datapipes.sic import sic
# from datapipes import cache_results
from datapipes import sinks


__all__ = ["DataPipe", "Ops", "filters", "contrast", "Ops", "plot", "datasets", "sic", "cache_results", "sinks"]