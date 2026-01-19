import sys
import argparse
from typing import Optional
from pathlib import Path
from datapipes.tools import dataset_wrangler
from datapipes.tools.dataset_wrangler_cli import main as dataset_wrangler
from datapipes.tools.init_matlab import install_datapipes_in_matlab
from datapipes.utils import import_resource
import subprocess
import shutil

def init(argv: Optional[list[str]] = None) -> int:
    if argv is not None and len(argv) > 0:
        raise SyntaxError("init takes no arguments.")
    destination = Path.cwd()
    if (destination / "pyproject.toml").exists():
        raise ValueError(f'{destination.as_posix()} already contains a project. Please run from an empty folder.')
    
    import_resource.extract_file("quickstart.zip", destination=destination)

    install_script = destination / "install.bat"

    subprocess.run(str(install_script), shell=True, check=True)

    insert_datawrangler_lnk_shortcut_in_cwd()

def insert_datawrangler_lnk_shortcut_in_cwd():
    ps_script = f'''
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{Path.cwd().resolve()}\DatasetWrangler.lnk")
$Shortcut.TargetPath = "{tools.get_python_exe_cli().as_posix()}"
$Shortcut.Arguments = '"{tools.get_datapipes_cli_command_script().as_posix()}" dataset-wrangler'
$Shortcut.WorkingDirectory = "$env:USERPROFILE"
$Shortcut.Save()
    '''

    subprocess.run(["powershell", "-Command", ps_script], check=True)

#     launch_script_bat = f'''
# datapipes dataset-wrangler && pause
# '''
#     with open(Path.cwd() / "launch_DatasetWrangler.bat", "w") as f:
#         f.write(launch_script_bat)

def matlab_installer(argv: Optional[list[str]] = None) -> int:
    insert_datawrangler_lnk_shortcut_in_cwd()
    install_datapipes_in_matlab()

    

def print_help(argv: Optional[list[str]] = None) -> int:
    print(f"Available commands:")
    for t in tools.keys():
        print(f"\t{t}", end="\n")

tools = {
    "help": print_help,
    "init": init,
    "dataset-wrangler": dataset_wrangler,
    "init-matlab": matlab_installer,
}

def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # if argv is None or len(argv) < 1:
    #     print("Usage: datapipes tool_name [args...]")
    #     return 1
    print(f"{argv = }")

    func_name = argv[0]
    args = argv[1:]

    print(f"{func_name = }, {(*args, ) = }")

    if func_name not in tools:
        print(f"Unknown tool: {func_name}")
        print(f"Available tools: {', '.join(tools)}")
        return 1

    try:
        tools[func_name](args)
    except TypeError as e:
        print(f"Argument error: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

