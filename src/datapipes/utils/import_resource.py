from importlib.resources import files, as_file
import torch
from pathlib import Path

_resources_root: str = "datapipes.utils.resources"

def list_resources() -> list[str]:
    resource = files(_resources_root)
    return [item.name for item in resource.iterdir()]

def load_tensor(resource_relative_path: str) -> torch.Tensor:
    resource = files(_resources_root).joinpath(resource_relative_path)
    with as_file(resource) as resource_path:
        t = torch.load(resource_path)
        
    return t

def get_bytes(resource_relative_path: str) -> bytes:
    resource = files(_resources_root).joinpath(resource_relative_path)
    with as_file(resource) as resource_path:
        content = resource_path.read_bytes()
    
    return content