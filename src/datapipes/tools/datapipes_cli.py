import argparse
from typing import Optional
from pathlib import Path
from datapipes.tools import dataset_wrangler
from datapipes.tools.dataset_wrangler_cli import main as wrangler
from datapipes.utils import import_resource
import subprocess

def init(argv: Optional[list[str]] = None) -> int:
    if len(argv) > 0:
        raise SyntaxError("init takes no arguments.")
    destination = Path.cwd()
    if (destination / "pyproject.toml").exists():
        raise ValueError(f'{destination.as_posix()} already contains a project. Please run from an empty folder.')
    
    import_resource.extract_file("quickstart.zip", destination=destination)

    install_script = destination / "install.bat"

    subprocess.run(str(install_script), shell=True, check=True)

tools = {
    "wrangler": wrangler,
    "init": init,
}

def main(argv: Optional[list[str]] = None) -> int:
    if len(argv) < 1:
        print("Usage: datapipes tool_name [args...]")
        return 1

    func_name = argv[0]
    args = argv[1:]

    if func_name not in tools:
        print(f"Unknown tool: {func_name}")
        print(f"Available tools: {', '.join(tools)}")
        return 1

    try:
        tools[func_name](*args)
    except TypeError as e:
        print(f"Argument error: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

