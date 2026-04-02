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
    decode_frames_threaded,
)
from datapipes.ops import Ops
from datapipes.save_datapipe.new_file_format.lazy_decoding_image_tensor import LazyDecodingImageTensor

from datapipes.datasets import DatasetSource

from datapipes.save_datapipe.new_file_format.wrapper_dataset import WrapperDataset
from typing import Dict, Callable, Any, Optional, Tuple
#%%
import torch
import numpy as np
import einops
import os
import functools

from datapipes.utils import get_logger

logger = get_logger(__name__)


from typing import Optional, Literal, Tuple, Iterable

from dataclasses import dataclass
from functools import partial

encoders: Dict[str, Encoder] = {}
decoders: Dict[str, Callable[[Iterable[bytes]], np.ndarray|torch.Tensor]] = {}

def get_available_encoders() -> Iterable[str]:
    return tuple(encoders.keys())

def make_batched_decoder(decoder: Callable[[bytes], np.ndarray|torch.Tensor]) ->Callable[[bytes], torch.Tensor]:
    def batched_decoder(input_bytes: Iterable[bytes]) -> torch.Tensor:
        # decoded_frames = tuple(decoder(frame_stream) for frame_stream in input_bytes)
        decoded_frames = tuple(decode_frames_threaded(input_bytes, decoder))
        if isinstance(decoded_frames[0], np.ndarray):
            decoded_frames = tuple(torch.from_numpy(frame) for frame in decoded_frames)
        t = torch.stack(decoded_frames, dim=0)
        if t.ndim == 3:
            t = t.unsqueeze(1)
        return t
    return batched_decoder

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
    settings_used: Optional[Dict[str, Any]]=None,
    is_batched_encoder: bool=False
) -> None:
    if not is_batched_encoder:
        encode_func = partial(encode_frames_threaded, encode_func=encode_func, num_workers=8)
    encoder = Encoder(name=name, encode_func=encode_func, settings_used=settings_used)
    encoders[name] = encoder

def register_decoder(
    name: str,
    decode_func: Callable[[Iterable[bytes]], np.ndarray|torch.Tensor],
    is_batched_encoder: bool=False
) -> None:
    if is_batched_encoder:
        decoders[name] = decode_func
    else:
        decoders[name] = make_batched_decoder(decode_func)

def get_encoder(name: str):
    if name not in encoders.keys():
        raise NotImplementedError(f"Unsupported encoder: {name = }. Supported decoders: [{", ".join(encoders.keys())}]")
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
    register_encoder(name="j2k", encode_func=j2k_nv_encode, settings_used=settings, is_batched_encoder=True)

    register_decoder(name="j2k", decode_func=torch_decode)

except ModuleNotFoundError:
    # TODO: Add CPU j2k codec
    register_encoder("j2k", partial(imagecodecs.htj2k_encode, reversible=True), settings_used={"lib": "imagecodecs", "reversible": True})
    register_decoder("j2k", imagecodecs.htj2k_decode)


def move_channels(func: Callable) -> Callable:
    @functools.wraps(func)
    def _move_channels(frames, *args, **kwargs):
        frames = np.moveaxis(frames, 0, -1)
        return func(frames, *args, **kwargs)
    return _move_channels

def encode_jxl(frames):
    frames = np.moveaxis(frames, 0, -1)
    return imagecodecs.jpegxl_encode(frames, lossless=True, numthreads=os.cpu_count())

def encode_jls(frames):
    frames = np.moveaxis(frames, 0, -1)
    return imagecodecs.jpegls_encode(frames)

def encode_jxs(frames, config: str, bitspersample: int):
    frames = np.moveaxis(frames, 0, -1)
    return imagecodecs.jpegxs_encode(frames, config=config, bitspersample=bitspersample)

def encode_j2k_lossy(frames, level: int=80):
    frames = np.moveaxis(frames, 0, -1)
    return imagecodecs.jpeg2k_encode(frames, level=level, reversible=False, numthreads=os.cpu_count())

register_encoder("jxl", encode_jxl, settings_used={"lossless": True}) # , usecontainer=False
# register_encoder("jxl", partial(imagecodecs.jpegxl_encode, lossless=True), settings_used={"lossless": True}) # , usecontainer=False
register_decoder("jxl", imagecodecs.jpegxl_decode)


register_encoder("jls", encode_jls, settings_used={"lossless": True}) # , usecontainer=False
# register_encoder("jxl", partial(imagecodecs.jpegxl_encode, lossless=True), settings_used={"lossless": True}) # , usecontainer=False
register_decoder("jls", imagecodecs.jpegls_decode)

register_encoder("jxs", move_channels(imagecodecs.jpegxs_encode))
register_decoder("jxs", imagecodecs.jpegxs_decode)

register_encoder("jxr", move_channels(imagecodecs.jpegxr_encode))
register_decoder("jxr", imagecodecs.jpegxr_decode)

