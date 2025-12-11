"""
Save a DataPipe to various formats
"""

from datapipes.save_datapipe.to_rls import datapipe_to_rls  # re-export for convenient import
from datapipes.save_datapipe.to_hdf5 import datapipe_to_hdf5
from datapipes.save_datapipe.to_zarr import datapipe_to_zarr
from datapipes.save_datapipe.to_compressed_image_stream_hdf5 import datapipe_to_compressed_image_stream_hdf5
from datapipes.save_datapipe.to_j2k_h5 import datapipe_to_lossless_j2k_h5
from datapipes.save_datapipe.to_mp4 import datapipe_to_lossless_hevc_mp4, datapipe_to_lossy_av1_mp4


__all__ = ["datapipe_to_rls", "datapipe_to_hdf5", "datapipe_to_zarr", "datapipe_to_lossless_j2k_h5", "datapipe_to_compressed_image_stream_hdf5", "datapipe_to_lossless_hevc_mp4", "datapipe_to_lossy_av1_mp4"]