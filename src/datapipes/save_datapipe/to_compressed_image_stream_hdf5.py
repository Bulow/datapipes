#%%
import torch
import numpy as np
from pathlib import Path

from datapipes.datapipe import DataPipe
from datapipes.ops import Ops

import rich
import h5py
import hdf5plugin

from nvidia import nvimgcodec



from typing import Callable, Optional, Any, Tuple, List

#%%

from datapipes.save_datapipe.file_format import format_specification, metadata_utils

from datapipes.save_datapipe.file_format.inspect_hdf5 import visualize_structure
from datapipes.save_datapipe.file_format.image_compression import torch_encode, Codec, QualityType

def populate_metadata(dp: DataPipe, compression_parameters: format_specification.CompressionParameters):
    n, c, h, w = dp.shape
    metadata: format_specification.LsciEncodedFramesH5 = format_specification.LsciEncodedFramesH5(
        format_id="au.cfin.lsci.j2k.h5",
        format_version="0.0.0.1",
        frames=format_specification.ImageEncodedFrameStream(
            encoded_frames=format_specification.Placeholder(),
            frame_lengths_bytes=format_specification.Placeholder(),
            frame_start_memory_offsets=format_specification.Placeholder(),
            frame_parameters=format_specification.FrameParameters(
                bit_depth=8,
                channels=c,
                compressed=True,
                compression_parameters=compression_parameters,
                frames_format_version="0.0.0.1",
                frame_count=n,
                frame_width=w,
                frame_height=h,
                shape=dp.shape,
            ),
        ),
        metadata=format_specification.UserMetadata(
            timestamps=format_specification.Placeholder() # TODO: Make a window for querying a datapipe by its timestamps
        )
    )
    return metadata

def init_hdf5_structure(dp: DataPipe, f: h5py.File|h5py.Group, compression_parameters: format_specification.CompressionParameters) -> format_specification.LsciEncodedFramesH5:
    metadata = populate_metadata(dp, compression_parameters=compression_parameters)

    placeholder_dict: format_specification.LsciEncodedFramesH5
    placeholder_dict, manuals = metadata_utils.init_hdf5_structure(metadata, f)

    encoded_frames_placeholder = manuals[placeholder_dict.frames.encoded_frames]
    encoded_frames_placeholder.group.create_dataset(
        name=encoded_frames_placeholder.name,
        shape=(10**9, ),
        maxshape=(None, ),
        dtype=np.uint8,
        chunks=True,
    )

    frame_lengths_bytes_placeholder = manuals[placeholder_dict.frames.frame_lengths_bytes]
    frame_lengths_bytes_placeholder.group.create_dataset(
        name=frame_lengths_bytes_placeholder.name,
        shape=len(dp),
        dtype=np.uint64,
    )

    frame_start_memory_offsets_placeholder = manuals[placeholder_dict.frames.frame_start_memory_offsets]
    frame_start_memory_offsets_placeholder.group.create_dataset(
        name=frame_start_memory_offsets_placeholder.name,
        shape=len(dp),
        dtype=np.uint64,
    )

    timestamps_placeholder = manuals[placeholder_dict.metadata.timestamps]
    timestamps_placeholder.group.create_dataset(
        name=timestamps_placeholder.name,
        data=np.array(range(len(dp)))
    )

    live_view: format_specification.LsciEncodedFramesH5 = metadata_utils.deserialize_hdf5(f, format_specification.LsciEncodedFramesH5)

    return live_view

def htj2k_params(
    # size_ratio: float = 0.85,  # 15% of original uncompressed size
    # num_resolutions: int = 5,
    # code_block_size: tuple[int, int] = (64, 64),
    # jp2_container: bool = False,
    quality_value,
    ht=True,
):
    jpeg2k_params = nvimgcodec.Jpeg2kEncodeParams(
        # code_block_size=code_block_size,
        # num_resolutions=num_resolutions,
        bitstream_type=nvimgcodec.Jpeg2kBitstreamType.J2K,
        prog_order=nvimgcodec.Jpeg2kProgOrder.LRCP,
        ht=ht,
    )

    enc_params = nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.QUANTIZATION_STEP,
        quality_value=quality_value,
        jpeg2k_encode_params=jpeg2k_params,
        color_spec=nvimgcodec.ColorSpec.UNCHANGED,
        # chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_444,
    )
    return enc_params



class LossyJ2kCompressor:
    def __init__(self, q: int|float):
        # self.codec: Codec = "jpeg2k"
        # quality_value = 50

        # j2k_params = nv.Jpeg2kEncodeParams()
        # # j2k_params.num_resolutions = 4
        # # jpeg2k_params.code_block_size = (block_size, block_size)
        # j2k_params.bitstream_type = nv.Jpeg2kBitstreamType.J2K
        # # jpeg2k_params.prog_order = nv.Jpeg2kProgOrder.PCRL
        # j2k_params.ht = True

        # self.params: nv.EncodeParams = nv.EncodeParams(
        #     quality_type = QualityType.quality,
        #     quality_value = quality_value,
        #     jpeg2k_encode_params=j2k_params,
        # )

        # self.compression_parameters = format_specification.CompressionParameters(
        #     codec=self.codec,
        #     quality_type="lossy",
        #     quality_value=quality_value,
        #     kwargs={"ht": True, "bitstream_type": "j2k"} 
        # )
        self.codec: Codec = "jpeg2k"
        self.params = htj2k_params(ht=True, quality_value=q)
        self.compression_parameters = format_specification.CompressionParameters(
            codec=self.codec,
            quality_type="",
            quality_value=str(q),
            kwargs={"ht": True, "bitstream_type": "j2k"} 
        )

    def __call__(self, frames: torch.Tensor):
        return torch_encode(
            frames=frames, 
            codec=self.codec, 
            params=self.params
        )

def datapipe_to_compressed_image_stream_hdf5(dp: DataPipe, out_path: str|Path, encoder: Callable[[torch.Tensor], List[bytes]], compression_parameters: format_specification.CompressionParameters):
    # Prepare path
    if isinstance(out_path, str):
        out_path = Path(out_path)
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # frames_buffer_upper_bound = int(np.prod(dp.shape))

    with h5py.File(out_path, "w") as f:
       
        live_view = init_hdf5_structure(dp, f, compression_parameters=compression_parameters)

        # rich.inspect(live_view)

        encoded_frames: h5py.Dataset = live_view.frames.encoded_frames
        frame_lengths_bytes: h5py.Dataset = live_view.frames.frame_lengths_bytes
        frame_start_memory_offsets: h5py.Dataset = live_view.frames.frame_start_memory_offsets

        # live_view.frames.encoded_frames[0:10] = np.array(range(10))

        # print(encoded_frames[0:10])

        # inspect_hdf5.visualize_structure(f)
        # return

        

        batch_start_frame_index = 0
        batch_start_byte_index = 0
        current_array_size = encoded_frames.shape[0]
        for batch in dp.batches_with_progressbar(batch_size=1024):

            # Encode batch
            encoded: List[bytes] = encoder(batch)

            # encoded = torch_encode(batch, codec="jpeg2k", params=nv.EncodeParams(
            #         quality_type = nv.QualityType.QUANTIZATION_STEP,
            #         quality_value = 2,
            #         jpeg2k_encode_params=jpeg2k_params
            #     )
            # )

            # Compute indices
            lengths = np.fromiter([len(f) for f in encoded], dtype=np.uint64)
            offsets = np.empty_like(lengths)
            offsets[0] = 0
            np.cumsum(lengths[:-1], out=offsets[1:])

            flat_array = np.frombuffer(b"".join(encoded), dtype=np.uint8)

            current_batch_frame_length = len(batch)
            current_batch_byte_length = len(flat_array)

            frame_lengths_bytes[batch_start_frame_index:batch_start_frame_index + current_batch_frame_length] = lengths

            frame_start_memory_offsets[batch_start_frame_index:batch_start_frame_index + current_batch_frame_length] = offsets + batch_start_byte_index

            size_after_write = batch_start_byte_index + current_batch_byte_length
            if (current_array_size < size_after_write):
                current_array_size = size_after_write * 2
                encoded_frames.ds.resize((current_array_size, ))

            encoded_frames[batch_start_byte_index:batch_start_byte_index + current_batch_byte_length] = flat_array

            # Update position
            batch_start_frame_index += current_batch_frame_length
            batch_start_byte_index += current_batch_byte_length

        encoded_frames.ds.resize((batch_start_byte_index, ))

        # metadata.create_dataset(name="timestamps", data=np.array(range(len(dp)), dtype=np.uint64))

        # print("\n")
        # print(f"Saved datapipe to {str(out_path.parent.absolute())}:")
        # visualize_structure(f, out_path.name)

##%%

# if __name__ == "__main__":
#     raw: DataPipe = (
#         DataPipe(datapipes.datasets.DatasetRLS(R"C:\Workspace\DataAnalysis\ror\data\20230111_PSO01_a1.rls"))
#         # | Ops.to("cuda")

#     )

#     out_path = Path("lossless.j2k.hdf5")
#     datapipe_to_compressed_image_stream_hdf5(raw, out_path=out_path)


#%%
