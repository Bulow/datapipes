#%%
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Dict, Any
import numpy as np
from rich import print


from typing import Generic, TypeVar
import numpy as np

import h5py
T = TypeVar("T")

# @dataclass
class Placeholder(Generic[T]):
    def __init__(self, value: T|h5py.Dataset=None):
        self.value = "Proxy that reads and writes to the underlying hdf5 file" if isinstance(value, h5py.Dataset) else value
        if isinstance(value, h5py.Dataset):
            self.ds: h5py.Dataset = value
    
    def __setitem__(self, idx, val):
        self.ds[idx] = val

    def __getitem__(self, idx):
        return self.ds[idx]
    
    @property
    def shape(self):
        return self.ds.shape




@dataclass(frozen=True, kw_only=True)
class CompressionParameters:
    """
    From [nvImageCodec documentation](https://docs.nvidia.com/cuda/nvimagecodec/py_api.html#nvidia.nvimgcodec.QualityType)

    `DEFAULT` :

        Each plugin decides its default quality setting. quality_value is ignored in this case.
    `LOSSLESS` :

        Image encoding is reversible and keeps original image quality. `quality_value` is ignored, except for the CUDA tiff encoder backend, for which `quality_value=0` means no compression, and `quality_value=1` means LZW compression..
    `QUALITY` :

        `quality_value` is interpreted as JPEG-like quality in range from 1 (worst) to 100 (best).
    `QUANTIZATION_STEP` :

        `quality_value` is interpreted as quantization step (by how much pixel data will be divided). The higher the value, the worse quality image is produced.
    `PSNR` :

        `quality_value` is interpreted as desired Peak Signal-to-Noise Ratio (PSNR) target for the encoded image. The higher the value, the better quality image is produced. Value should be positive.
    `SIZE_RATIO` :

        `quality_value` is interpreted as desired encoded image size ratio compared to original size, should be floating point in range (0.0, 1.0). E.g. value 0.1 means target size of 10% of original image.


    """
    
    codec: Literal["jpeg2k", "jpeg"]
    quality_type: Literal["default", "lossless", "quality", "quantization_step", "psnr", "size_ratio"]
    quality_value: int|float|str
    kwargs: Dict[str, Any]

@dataclass(frozen=True, kw_only=True)
class FrameParameters:
    bit_depth: int
    channels: int
    compressed: bool
    compression_parameters: Optional[CompressionParameters]
    frames_format_version: str
    frame_count: int
    frame_width: int
    frame_height: int
    shape: np.ndarray

@dataclass(frozen=True, kw_only=True)
class ImageEncodedFrameStream:
    encoded_frames: Placeholder[np.ndarray]
    frame_lengths_bytes: Placeholder[np.ndarray]
    frame_start_memory_offsets: Placeholder[np.ndarray]
    frame_parameters: FrameParameters

@dataclass(frozen=True, kw_only=True)
class UserMetadata:
    timestamps: Placeholder[np.ndarray]

@dataclass(frozen=True, kw_only=True)
class LsciEncodedFramesH5:
    format_id: str
    format_version: str
    frames: ImageEncodedFrameStream
    metadata: UserMetadata