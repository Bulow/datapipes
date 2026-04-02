#%%
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Dict, Any
import numpy as np
from rich import print

import h5py as storage_backend
import numpy as np
import enum
from pathlib import Path
from datapipes.save_datapipe.new_file_format.file_store import FileStore, get_metadata
from datapipes.save_datapipe.new_file_format.frames import Frames

class FieldType(enum.StrEnum):
    metadata_json = "metadata_json"
    frames_group = "frames_group"
    array_dataset = "array_dataset"
    file_folder_group = "file_folder_group"

# Contain field types in field names within container file?
# e.g. user_metadata:metadata_json

# Defines a flat hierarchy. The values in here are used to route to the correct handler
@dataclass
class FileContents:
    file_metadata = FieldType.metadata_json
    raw_frames = FieldType.frames_group
    user_metadata = FieldType.metadata_json
    user_embedded_files = FieldType.file_folder_group
    
    _root: storage_backend.File|storage_backend.Group = None

    @classmethod
    def load(path: Path|str) -> "FileContents":
        path = Path(path)
        f = FileContents()
        f._root = storage_backend.File(path, mode="r")
        
        for var_name, field_type in vars(f):
            match field_type:
                case FieldType.metadata_json:
                    value = get_metadata(group=f._root, name=var_name)
                case FieldType.frames_group:
                    value = Frames(group=f._root)
                case FieldType.file_folder_group:
                    value = f._root.create_group(name=var_name)
                case _:
                    raise NotImplementedError(f"Invalid field type: {field_type}, {var_name = }")
            object.__setattr__(f, name=var_name, value=value)

    def __del__(self):
        if self._root is not None:
            self._root.close()




# @dataclass(frozen=True, kw_only=True)
# class LsciEncodedFramesH5:
#     format_id: str
#     format_version: str
#     frames: ImageEncodedFrameStream
#     metadata: UserMetadata