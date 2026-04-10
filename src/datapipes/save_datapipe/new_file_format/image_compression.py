#%%
import torch
import numpy as np
import einops
import os
os.environ['PYNVIMGCODEC_VERBOSITY'] = '5' # uncomment for verbose log output

from nvidia import nvimgcodec



from typing import Optional, Literal, Tuple

class QualityType:
    """
    DEFAULT :

        Each plugin decides its default quality setting. quality_value is ignored in this case.
    LOSSLESS :

        Image encoding is reversible and keeps original image quality. quality_value is ignored, except for the CUDA tiff encoder backend, for which quality_value=0 means no compression, and quality_value=1 means LZW compression..
    QUALITY :

        quality_value is interpreted as JPEG-like quality in range from 1 (worst) to 100 (best).
    QUANTIZATION_STEP :

        quality_value is interpreted as quantization step (by how much pixel data will be divided). The higher the value, the worse quality image is produced.
    PSNR :

        quality_value is interpreted as desired Peak Signal-to-Noise Ratio (PSNR) target for the encoded image. The higher the value, the better quality image is produced. Value should be positive.
    SIZE_RATIO :

        quality_value is interpreted as desired encoded image size ratio compared to original size, should be floating point in range (0.0, 1.0). E.g. value 0.1 means target size of 10% of original image.

    """
    default = nvimgcodec.QualityType.DEFAULT 
    lossless = nvimgcodec.QualityType.LOSSLESS
    quality = nvimgcodec.QualityType.QUALITY
    quantization_step = nvimgcodec.QualityType.QUANTIZATION_STEP
    psnr = nvimgcodec.QualityType.PSNR
    size_ratio = nvimgcodec.QualityType.SIZE_RATIO
    
Codec = Literal["jpeg", "jpeg2k"]
    

def _get_gpu_encoder():
    return nvimgcodec.Encoder(backends=[nvimgcodec.Backend(nvimgcodec.GPU_ONLY, load_hint=0.5), nvimgcodec.Backend(nvimgcodec.HYBRID_CPU_GPU)])

def _get_gpu_decoder():
    return nvimgcodec.Decoder(backends=[nvimgcodec.Backend(nvimgcodec.GPU_ONLY, load_hint=0.5), nvimgcodec.Backend(nvimgcodec.HYBRID_CPU_GPU)])

encoder = _get_gpu_encoder()
decoder = _get_gpu_decoder()


def make_lossless_jpeg_encode_params(
    *,
    color_spec: nvimgcodec.ColorSpec = nvimgcodec.ColorSpec.GRAY,
    optimized_huffman: bool = False,
    progressive: bool = False,
) -> nvimgcodec.EncodeParams:
    jpeg_params = nvimgcodec.JpegEncodeParams()
    jpeg_params.optimized_huffman = optimized_huffman
    if hasattr(jpeg_params, "progressive"):
        jpeg_params.progressive = progressive

    return nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.LOSSLESS,
        color_spec=color_spec,
        jpeg_encode_params=jpeg_params,
    )

def torch_encode(frames: torch.Tensor, codec: Optional[Codec]="jpeg2k", params: Optional[nvimgcodec.EncodeParams]=None) -> list[bytes]:
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    nv_images = nvimgcodec.as_images([f for f in einops.rearrange(frames.to("cuda", non_blocking=True), "F 1 H W -> F H W 1").contiguous()])
    encoded = encoder.encode(nv_images, codec, params=params)
    if any(cstream is None for cstream in encoded):
        raise NotImplementedError(
            f"nvimgcodec failed to encode codec={codec!r} with the requested parameters. "
            "For jpeg, verify the installed nvimgcodec backend supports lossless JPEG. "
            "For jpeg2k, high-throughput mode can be picky about parameters."
        )
    encoded = [cstream.__buffer__(0) for cstream in encoded]
    # import rich
    # rich.print(encoded)
    return encoded


def torch_decode(encoded: list[bytes|torch.ByteTensor] | bytes | torch.ByteTensor | np.ndarray, roi: Optional[Tuple[slice]] = None) -> torch.Tensor:
    # TODO: block based ROI decoding
    if roi is not None:
        roi = roi[-3:] # dump temporal dimension from roi, since it is contained in `encoded`

    # nvimgcodec accepts either a single codestream or a batch. Normalize to a
    # batch here so the return type is predictable across nvimgcodec versions.
    if isinstance(encoded, (torch.ByteTensor, np.ndarray, bytes, bytearray, memoryview)):
        encoded = [encoded]
    else:
        encoded = list(encoded)

    if isinstance(encoded[0], torch.ByteTensor):
        encoded = [nvimgcodec.CodeStream(s.numpy()) for s in encoded]
    decoded = decoder.decode(encoded, params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY))
    if not isinstance(decoded, (list, tuple)):
        decoded = [decoded]
    tensor_list = [torch.as_tensor(f) for f in decoded]

    single_tensor = einops.rearrange(torch.stack(tensor_list, dim=0), "F H W 1 -> F 1 H W")
    # single_tensor.names = ["N", "C", "H", "W"]

    if roi is not None:
        return single_tensor[:, *roi]
    else:
        return single_tensor


def get_compression_ratio(frames: torch.Tensor, encoded: list[bytes]) -> float:
    original_size = len(bytes(frames.cpu().numpy()))
    encoded_size = sum(len(bytes(e)) for e in encoded)
    return encoded_size / original_size
