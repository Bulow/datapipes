from datapipes.tools.dataset_wrangler_app.dataset_wrangler import launch_dataset_wrangler as dataset_wrangler

import sys
from pathlib import Path

def get_python_exe_cli() -> Path:
    """
    Return the path to the Python executable used for CLI execution.
    """
    return Path(sys.executable).resolve()


def get_pythonw_exe_gui() -> Path:
    """
    Return the path to pythonw.exe on Windows (GUI interpreter).
    On non-Windows platforms, fall back to the normal Python executable.
    """
    exe = Path(sys.executable).resolve()

    if sys.platform.startswith("win"):
        # python.exe -> pythonw.exe
        pythonw = exe.with_name("pythonw.exe")
        if pythonw.exists():
            return pythonw

    # macOS/Linux or fallback
    return exe


def get_python_bin_folder() -> Path:
    """
    Return the directory containing the active Python executable.
    """
    return Path(sys.executable).resolve().parent

def get_src_datapipes_folder() -> Path:
    """
    Return the root .../src/datapipes folder corresponding to the datapipes module
    """
    # return Path(__file__).resolve().parent.parent
    this_file = Path(__file__).resolve()
    folder_name = "datapipes"
    for parent in this_file.parents:
        if parent.name == folder_name:
            return parent
    raise FileNotFoundError(f"No parent folder named {folder_name}. Was the folder renamed? {__file__ = }")

def get_datapipes_cli_command_script() -> Path:
    """
    Return the path to the file that contains the `datapipes` uv tool.
    """
    return get_src_datapipes_folder() / "tools/datapipes_cli.py"


__all__ = ["dataset_wrangler", "get_python_exe_cli", "get_pythonw_exe_gui", "get_python_bin_folder", "get_src_datapipes_folder", "get_datapipes_cli_command_script"]