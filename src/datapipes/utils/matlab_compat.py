from datapipes.utils.output_server import _url_escape, get_server
from datapipes.datasets import DatasetSource

from pathlib import Path
import shutil
import subprocess


def get_progress_url(ds: DatasetSource) -> str:
    base_url = get_server().url
    if hasattr(ds, "path"):
        key = _url_escape(content=ds.path)
    else:
        key = _url_escape(content=ds)

    return f"{base_url}/{key}"



class MatlabInstallError(RuntimeError):
    pass

def install_lib(install_m_path: Path) -> None:
    """
    Run a MATLAB install.m using `matlab -batch`.

    Parameters
    ----------
    install_m_path : Path
        Full path to install.m

    Raises
    ------
    FileNotFoundError
        If install.m or MATLAB cannot be found
    MatlabInstallError
        If MATLAB exits with a nonzero status
    """
    install_m_path = Path(install_m_path)

    if not install_m_path.is_file():
        raise FileNotFoundError(f"install.m not found: {install_m_path}")

    matlab = shutil.which("matlab")
    if matlab is None:
        raise FileNotFoundError(
            "MATLAB executable not found on PATH (expected 'matlab')."
        )

    repo_dir = install_m_path.parent.resolve()
    repo_posix = repo_dir.as_posix()  # safe on Windows + MATLAB

    matlab_cmd = (
        f"try, cd('{repo_posix}'); install; "
        "catch ME, disp(getReport(ME,'extended')); exit(1); "
        "end; exit(0);"
    )

    proc = subprocess.run(
        [matlab, "-batch", matlab_cmd],
        text=True,
        capture_output=True,
    )

    if proc.returncode != 0:
        raise MatlabInstallError(
            "MATLAB install failed\n"
            f"--- STDOUT ---\n{proc.stdout}\n"
            f"--- STDERR ---\n{proc.stderr}"
        )
