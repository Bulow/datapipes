from importlib.resources import files, as_file
import torch
from pathlib import Path
import zipfile
from contextlib import contextmanager

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

def extract_file(resource_relative_path: str, destination: Path):
    resource = files(_resources_root).joinpath(resource_relative_path)
    with as_file(resource) as payload_zip:
        if not destination.is_dir():
            raise ValueError(f"Destination {destination} is not a valid folder")
        
        with zipfile.ZipFile(payload_zip, 'r') as zip_ref:
            zip_ref.extractall(destination)

@contextmanager
def as_path(resource_relative_path: str):
    resource = files(_resources_root).joinpath(resource_relative_path)
    with as_file(resource) as resource_path:
        yield resource_path


