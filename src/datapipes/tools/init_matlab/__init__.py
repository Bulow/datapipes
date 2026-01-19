from datapipes.utils import import_resource
from pathlib import Path
import shutil
import sys
from datapipes.tools.init_matlab.install_matlab_library import install_matlab_library

def install_datapipes_in_matlab():
    install_folder = _matlab_default_install_dir()
    print(f"Extracting files to {install_folder}")
    _install_lib_files(install_folder=install_folder)

    print(f"Setting up python environment")
    _write_load_python_function_matlab(install_folder=install_folder)

    # Run install.m in matlab
    # installer = install_folder / "install.m"
    print(f"Running install.m in MATLAB")
    install_matlab_library(library_dir=install_folder)

    print(f"Extracting sample files")
    sample_files = ["brrr.m", "quickstart.m", "markus.m"]

    sample_dest_folder = Path.cwd()
    for s in sample_files:
        with import_resource.as_path(f"matlab/{s}") as q:
            shutil.copyfile(src=q, dst=sample_dest_folder / s)

    print("\n\n_______________________________________________________________________\n")
    print(f"Installed datapipes for matlab into \"{install_folder.as_posix()}\".")
    print(f"Datapipes has been added to matlab path.")
    print(f"For a tutorial, open \"quickstart.m\" in \"{sample_dest_folder.as_posix()}\".")

def _install_lib_files(install_folder: Path):
    
    with import_resource.as_path("matlab/MatDatapipes") as source_path:
        _copy_lib(source_folder=source_path, dest_folder=install_folder)

    # Create loadPython.m function that loads the current python environment
    _write_load_python_function_matlab(install_folder=install_folder)
        
    
def _write_load_python_function_matlab(install_folder: Path):
    python_path = Path(sys.executable).resolve()
    print(python_path)
    matlab_function = f'''
function loadPython()

pe = pyenv;

if pe.Status == "NotLoaded"
    disp("[Datapipes]: Loading python environment - this takes a few seconds and is only done once per session.");
    pyenv(Version="{python_path}", ExecutionMode="InProcess"); %, ExecutionMode="OutOfProcess"); %
end

'''
    with open(install_folder / "+MatDatapipes/loadPython.m", "w") as f:
        f.write(matlab_function)


def _matlab_default_install_dir() -> Path:
    """
    Return ~/Documents/MATLAB/<lib_name>
    """
    lib_name: str = "MatDatapipes"
    return Path.home() / "Documents" / "MATLAB" / lib_name





def _copy_lib(source_folder: Path, dest_folder: Path) -> None:
    source_folder = Path(source_folder)
    dest_folder = Path(dest_folder)

    if not source_folder.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_folder}")

    # Remove existing installation (clean upgrade)
    if dest_folder.exists():
        shutil.rmtree(dest_folder)

    # Ignore common VCS / build junk
    ignore = shutil.ignore_patterns(
        ".git",
        ".github",
        "__pycache__",
        "*.pyc",
        ".DS_Store",
        "*.asv",
        "*.m~",
    )

    shutil.copytree(
        src=source_folder,
        dst=dest_folder,
        ignore=ignore,
        dirs_exist_ok=False,
    )
