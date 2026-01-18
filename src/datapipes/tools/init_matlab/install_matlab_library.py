from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Optional, Sequence


@dataclass
class MatlabRunResult:
    returncode: int
    stdout: str
    stderr: str
    command: Sequence[str]


class MatlabInstallError(RuntimeError):
    """Raised when MATLAB fails to run install.m successfully."""


def install_matlab_library(
    library_dir: str | Path,
    *,
    matlab_executable: Optional[str | Path] = None,
    mode: str = "batch",
    extra_matlab_args: Optional[Sequence[str]] = None,
    timeout_seconds: Optional[int] = None,
    check: bool = True,
) -> MatlabRunResult:
    """
    Install a MATLAB library by launching MATLAB and running install.m in repo_dir.

    Parameters
    ----------
    repo_dir:
        Folder containing install.m (the MATLAB library root).
    matlab_executable:
        Path to MATLAB executable (matlab or matlab.exe). If None, uses 'matlab' on PATH.
    mode:
        'batch' (recommended, R2019a+) or 'legacy' (-nodisplay -nosplash -r).
    extra_matlab_args:
        Extra args appended to the MATLAB command (rarely needed).
    timeout_seconds:
        Kill MATLAB if it runs longer than this.
    check:
        If True, raise MatlabInstallError on nonzero exit.

    Returns
    -------
    MatlabRunResult with returncode/stdout/stderr/command.

    Raises
    ------
    FileNotFoundError:
        If MATLAB executable can't be found or repo_dir/install.m missing.
    ValueError:
        If mode is invalid.
    MatlabInstallError:
        If MATLAB returns nonzero exit code and check=True.
    """
    repo = Path(library_dir)
    if not repo.is_dir():
        raise FileNotFoundError(f"Repo directory not found: {repo}")

    install_m = repo / "install.m"
    if not install_m.is_file():
        raise FileNotFoundError(f"install.m not found in repo root: {install_m}")

    # Find MATLAB executable
    if matlab_executable is None:
        matlab = shutil.which("matlab")
        if not matlab:
            raise FileNotFoundError(
                "Could not find 'matlab' on PATH. Provide matlab_executable with full path to matlab/matlab.exe."
            )
        matlab_path = matlab
    else:
        matlab_path = str(Path(matlab_executable))
        if not Path(matlab_path).is_file():
            raise FileNotFoundError(f"MATLAB executable not found: {matlab_path}")

    # MATLAB string quoting: use forward slashes (MATLAB accepts them on Windows)
    repo_posix = repo.resolve().as_posix()
    cd_cmd = f"cd('{repo_posix}')"

    # Ensure failures propagate to Python with a nonzero exit code.
    # getReport gives a readable stack trace in logs.
    matlab_body = (
        f"try, {cd_cmd}; install; "
        "catch ME, disp(getReport(ME,'extended')); exit(1); "
        "end; exit(0);"
    )

    extra = list(extra_matlab_args) if extra_matlab_args else []

    if mode == "batch":
        cmd = [matlab_path, "-batch", matlab_body, *extra]
    elif mode == "legacy":
        cmd = [matlab_path, "-nodisplay", "-nosplash", "-r", matlab_body, *extra]
    else:
        raise ValueError("mode must be 'batch' or 'legacy'")

    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )

    result = MatlabRunResult(
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        command=cmd,
    )

    if check and result.returncode != 0:
        # Include output in the exception so callers can log/debug easily.
        msg = (
            f"MATLAB install failed (exit code {result.returncode}).\n"
            f"Command: {' '.join(result.command)}\n"
            f"--- MATLAB STDOUT ---\n{result.stdout}\n"
            f"--- MATLAB STDERR ---\n{result.stderr}\n"
        )
        raise MatlabInstallError(msg)

    return result
