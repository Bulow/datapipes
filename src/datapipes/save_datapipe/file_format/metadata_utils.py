#%%

import numpy as np
from dataclasses import fields, is_dataclass, asdict, dataclass
from typing import Any, get_origin, get_args, Union, Literal, Callable
from pathlib import Path
import h5py

from datapipes.save_datapipe.file_format import inspect_hdf5, format_specification

from rich import print

from icecream.icecream import ic


def is_optional(t):
    return get_origin(t) is Union and type(None) in get_args(t)


def _iter_dict_like(obj):
    """
    Yield (key, value) pairs from any dict-like object.

    Supports:
      - Normal dicts (has .items())
      - HDF5 groups (h5py.Group has .items() or .keys())
      - Fallback: iterable of (k, v) pairs
    """
    if hasattr(obj, "items"):
        # dict, h5py.Group, etc.
        yield from obj.items()
    elif hasattr(obj, "keys"):
        # something with keys() + __getitem__
        for k in obj.keys():
            yield k, obj[k]
    else:
        # assume already iterable of (k, v)
        for k, v in obj:
            yield k, v

def _is_string_dtype(dt):
    # bytes (S), unicode (U), or object that may contain strings
    return dt.kind in ("S", "U") or dt.kind == "O"

def hdf5_deserialization_preproces(data) -> Any:
    if isinstance(data, h5py.Dataset):
        # If data is a single value
        if data.shape == ():
            val = data[()]
            if hasattr(val, "item"):
                val = val.item()
            # decode if string
            if _is_string_dtype(data.dtype):
                try:
                    val = val.decode('utf-8')
                except Exception:
                    val = val.decode(errors='replace')
            return val
    
    # Passthrough
    return data

def deserialize(data: Any, dataclass_type: Any, preprocessor: Callable=None) -> Any:
    """
    Recursively reconstruct dataclass instances (and numpy arrays)
    from JSON-serializable or dict-like structures.
    """

    if preprocessor is not None and callable(preprocessor):
        # print(dataclass_type)
        if (get_origin(dataclass_type) == format_specification.Placeholder):
            data = format_specification.Placeholder(data)
        else:
            data = preprocessor(data)
            
    # Handle Optional[T]
    if is_optional(dataclass_type):
        inner = [arg for arg in get_args(dataclass_type) if arg is not type(None)][0]
        if data is None:
            return None
        return deserialize(data=data, dataclass_type=inner, preprocessor=preprocessor)

    # Dataclass reconstruction: `data` can be any dict-like (incl. HDF5 group)
    if is_dataclass(dataclass_type):
        kwargs = {}
        for f in fields(dataclass_type):
            field_type = f.type
            # HDF5 Group and dict both support __getitem__
            value = data[f.name]
            kwargs[f.name] = deserialize(data=value, dataclass_type=field_type, preprocessor=preprocessor)
        return dataclass_type(**kwargs)
    
    if dataclass_type == str:
        return str(data)

    # numpy array
    if dataclass_type is np.ndarray:
        return np.array(data)

    # typed list → List[T]
    if get_origin(dataclass_type) is list:
        (item_type,) = get_args(dataclass_type)
        return [deserialize(data=x, dataclass_type=item_type, preprocessor=preprocessor) for x in data]

    # typed dict → Dict[K, V]; data can be any dict-like object
    if get_origin(dataclass_type) is dict:
        key_type, val_type = get_args(dataclass_type)
        return {
            deserialize(data=k, dataclass_type=key_type, preprocessor=preprocessor): deserialize(data=v, dataclass_type=val_type, preprocessor=preprocessor)
            for k, v in _iter_dict_like(data)
        }

    # Literal[...] → just return the value (dataclass validation will enforce correctness)
    if get_origin(dataclass_type) is Literal:
        return data

    # Base case: int, float, str, bool, None, etc.
    return data

def deserialize_hdf5(data: Any, dataclass_type: type):
    return deserialize(data=data, dataclass_type=dataclass_type, preprocessor=hdf5_deserialization_preproces)

def load_hdf5(path: str|Path, dataclass_type: type):
    if isinstance(path, str):
        path = Path(path)
    with h5py.File(path, "r") as f:
        return deserialize_hdf5(data=f, dataclass_type=dataclass_type)

from typing import Optional
class ManualPlaceholder:
    def __init__(self, description: Optional[str]=None):
        self.description = description
        self.group: Optional[h5py.Group] = None
        self._name: str = None
        self.ready_to_populate: bool = False

    def ready(self, group: Optional[h5py.Group], key: str):
        self.group = group
        self._name = key
        self.ready_to_populate = True

    @property
    def name(self):
        if not self.ready_to_populate:
            raise Exception(f"Placeholder is not yet ready to populate. You must first create an HDF5 file structure using `serialize_hdf5`")
        return self._name



def serialize_hdf5(data: dict|object, group: h5py.Group, parents: str="root") -> tuple[dict, dict[str, ManualPlaceholder]]:

    data_dict = asdict(data) if is_dataclass(data) else data#.copy()

    out_dict = data_dict#.copy()
    manuals = {}
    for k, v in data_dict.items():
        # print(v)
        if isinstance(v, dict):
            new_group = group.create_group(k)

            d, m = serialize_hdf5(v, new_group, parents=f"{parents}.{k}")
            out_dict[k] = d
            manuals.update(m)
        elif isinstance(v, format_specification.Placeholder):
            mp = ManualPlaceholder(description=v)
            mp.ready(group=group, key=k)
            path = f"{parents}.{k}"
            manuals[path] = mp
            data_dict[k] = path
            
        else:
            # print({k: v})
            group[k] = v
    return data_dict, manuals

def init_hdf5_structure(data: dict|object, group: h5py.Group) -> tuple[Any, dict[str, ManualPlaceholder]]:
    cls = data.__class__
    data_dict, manuals = serialize_hdf5(data=data, group=group)
    # print(data_dict)
    # ic(data_dict)
    obj = deserialize(data_dict, cls)
    # ic(obj)
    return obj, manuals



##%%


# if __name__ == "__main__":
#     from format_specification import LsciEncodedFramesH5, ImageEncodedFrameStream, FramesMetadata, CompressionParameters, UserMetadata, Placeholder

#     sample_instance: LsciEncodedFramesH5 = LsciEncodedFramesH5(
#         format_id="gg",
#         format_version="0.0.0.1",
#         frames=ImageEncodedFrameStream(
#             encoded_frames=Placeholder(),
#             frame_lengths_bytes=Placeholder(),
#             frame_start_memory_offsets=Placeholder(),
#             metadata=FramesMetadata(
#                 bit_depth=8,
#                 channels=1,
#                 compressed=True,
#                 compression_parameters=CompressionParameters(
#                     codec="jpeg2k",
#                     quality_type="lossless",
#                     quality_value=0,
#                     kwargs={"ht": True, "bitstream_type": "j2k"} # Leaky abstraction...
#                 ),
#                 frames_format_version="0.0.0.1",
#                 frame_count=2048,
#                 frame_width=512,
#                 frame_height=512,
#                 shape=np.array([2048, 1, 512, 512]),
#             ),
#         ),
#         metadata=UserMetadata(
#             timestamps=Placeholder()
#         )
#     )

#     def get_placeholder(placeholder: ManualPlaceholder) -> ManualPlaceholder:
#         if not placeholder.ready_to_populate:
#             raise Exception(f"Placeholder is not yet ready to populate. You must first create an HDF5 file structure using `serialize_hdf5`")
#         return placeholder

#     print(sample_instance.frames.encoded_frames)

#     with h5py.File("gg.hdf5", "w") as f:
#         placeholder_dict: LsciEncodedFramesH5
#         placeholder_dict, manuals = init_hdf5_structure(sample_instance, f)



#         encoded_frames_placeholder = manuals[placeholder_dict.frames.encoded_frames]
#         ic(encoded_frames_placeholder)
#         encoded_frames_placeholder.group.create_dataset(
#             name=encoded_frames_placeholder.name,
#             shape=(10**9, ),
#             maxshape=(None, ),
#             dtype=np.uint8,
#             chunks=True,
#         )

#         frame_lengths_bytes_placeholder = manuals[placeholder_dict.frames.frame_lengths_bytes]
#         frame_lengths_bytes_placeholder.group.create_dataset(
#             name=frame_lengths_bytes_placeholder.name,
#             shape=1024,
#             dtype=np.uint64,
#         )

#         frame_start_memory_offsets_placeholder = manuals[placeholder_dict.frames.frame_start_memory_offsets]
#         frame_start_memory_offsets_placeholder.group.create_dataset(
#             name=frame_start_memory_offsets_placeholder.name,
#             shape=1024,
#             dtype=np.uint64,
#         )

#         timestamps_placeholder = manuals[placeholder_dict.metadata.timestamps]
#         timestamps_placeholder.group.create_dataset(
#             name=timestamps_placeholder.name,
#             data=np.array(range(1024))
#         )

#         print({k: v.__dict__ for k, v in manuals.items()})

#         inspect_hdf5.visualize_structure(f)

#         live_view: LsciEncodedFramesH5 = deserialize_hdf5(f, LsciEncodedFramesH5)
#         print(live_view)
#         live_view.frames.encoded_frames.resize((10**9 + 3, )) #.append(np.array([1, 2, 3]))

#         inspect_hdf5.visualize_structure(f)




#     #%%
