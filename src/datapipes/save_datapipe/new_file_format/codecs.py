from __future__ import annotations
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Protocol, Tuple

import einops

# Base on h5py initially
import h5py as storage_backend

# import torch
import imagecodecs
import numpy as np

from datapipes.save_datapipe.new_file_format.parallel_encode import (
    encode_frames_threaded,
)

from datapipes.save_datapipe.new_file_format.lazy_decoding_image_tensor import LazyDecodingImageTensor

from datapipes.datasets import DatasetSource

from datapipes.save_datapipe.new_file_format.wrapper_dataset import WrapperDataset
from typing import Dict, Callable, Any, Optional, Tuple
#%%
import torch
import numpy as np
import einops
import os



from typing import Optional, Literal, Tuple, Iterable

from dataclasses import dataclass
from functools import partial

encoders: Dict[str, Encoder] = {}
decoders: Dict[str, Callable[[Iterable[bytes]], np.ndarray|torch.Tensor]] = {}

@dataclass
class Encoder:
    name: str
    encode_func: Callable[[np.ndarray|torch.Tensor], Iterable[bytes]]
    settings_used: Dict[str, Any]

    def __call__(self, frames: np.ndarray|torch.Tensor): 
        return self.encode_func(frames)
    
def register_encoder(
    name: str,
    encode_func: Callable[[np.ndarray|torch.Tensor, Optional[Dict[str, Any]]], Iterable[bytes]],
    settings_used: Optional[Dict[str, Any]],
) -> None:
    encoder = Encoder(name=name, encode_func=encode_func, settings_used=settings_used)
    encoders[name] = encoder

def register_decoder(
    name: str,
    decode_func: Callable[[Iterable[bytes]], np.ndarray|torch.Tensor],
) -> None:
    decoders[name] = decode_func

def get_encoder(name: str):
    if name not in encoders.keys():
        raise NotImplementedError(f"Unsupported decoder: {name = }. Supported decoders: [{", ".join(encoders.keys())}]")
    return encoders[name]

def get_decoder(name: str):
    if name not in decoders.keys():
        raise NotImplementedError(f"Unsupported decoder: {name = }. Supported decoders: [{", ".join(decoders.keys())}]")
    return decoders[name]


try:
    os.environ['PYNVIMGCODEC_VERBOSITY'] = '5' # uncomment for verbose log output
    from nvidia import nvimgcodec
    from datapipes.save_datapipe.new_file_format.image_compression import torch_encode, torch_decode

    jpeg2k_params = nvimgcodec.Jpeg2kEncodeParams()
    jpeg2k_params.bitstream_type = nvimgcodec.Jpeg2kBitstreamType.J2K
    jpeg2k_params.ht = True

    j2k_nv_encode = partial(
        torch_encode, 
        codec="jpeg2k", 
        params=nvimgcodec.EncodeParams(
            quality_type = nvimgcodec.QualityType.LOSSLESS,
            jpeg2k_encode_params=jpeg2k_params
        )
    )
    settings = dict(lib="nvimgcodec", ht=True, Jpeg2kBitstreamType="J2K", QualityType="LOSSLESS")
    register_encoder(name="j2k", encode_func=j2k_nv_encode, settings_used=settings)

    register_decoder(name="j2k", decode_func=torch_decode)

except ModuleNotFoundError:
    # TODO: Add CPU j2k codec
    register_encoder("j2k", partial(imagecodecs.htj2k_encode, reversible=True), settings_used={"lib": "imagecodecs", "reversible": True})
    register_decoder("j2k", imagecodecs.htj2k_decode)
1

register_encoder("jxl", partial(imagecodecs.jpegxl_encode, lossless=True), settings_used={"lossless": True})
register_decoder("jxl", imagecodecs.jpegxl_decode)

