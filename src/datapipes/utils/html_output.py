from datapipes.utils import ExecutionMode, get_execution_mode
from typing import Callable




def _not_implemented(*args):
    raise NotImplementedError()



_set_output_func: Callable[[str, str], None] = _not_implemented 
_show_output_func: Callable[[str], None] = _not_implemented
_outputs = {}

def set_output(key: str, content_html: str):
    init()
    _set_output_func(key, content_html)

def show_output(key: str):
    init()
    _show_output_func(key)


import os
import sys
import webview
import multiprocessing as mp


def _pythonw_exe():
    # sys.executable is usually ...\python.exe
    # pythonw.exe is typically in the same directory
    exe_dir = os.path.dirname(sys.executable)
    candidate = os.path.join(exe_dir, "pythonw.exe")
    return candidate if os.path.exists(candidate) else sys.executable

def launch_webview(key_url: str):
    window = webview.create_window(f"Datapipes", url=key_url)
    webview.start()

def launch_webview_daemon(key_url: str):
    print(f'Loading in the background. The dataset is ready for use but will be slower until fully loaded.\n<a href="matlab:web(\'{key_url}\',\'-browser\')">Click to view progress in browser</a>')
    
    mp.freeze_support()

    if sys.platform.startswith("win"):
        mp.set_executable(_pythonw_exe())

    p = mp.Process( target=launch_webview, args=(key_url, ), daemon=True)
    p.start()
    # t = threading.Thread(target=launch_webview, args=(key_url, ), daemon=True)
    # t.start()

_initialized: bool = False
def init():
    global _initialized
    if _initialized:
        return
    
    global _set_output_func
    global _show_output_func

    mode = get_execution_mode()
    print(f"{mode = }")
    match mode:
        case ExecutionMode.jupyter: 
            import ipywidgets as widgets
            from IPython.display import display

            def _set_output_jupyter(key: str, content_html: str):
                if key not in _outputs.keys():
                    _outputs[key] = widgets.HTML("")
                _outputs[key].value = content_html

            def _show_output_jupyter(key: str):
                display(_outputs[key])

            _set_output_func = _set_output_jupyter
            _show_output_func = _show_output_jupyter
            
        
        case ExecutionMode.matlab:
            from datapipes.utils import output_server

            _set_output_func = output_server.set_output

            def _show_output_matlab(key: str):
                output_server.show_output(key, onconnect=launch_webview_daemon)
                # output_server.show_output(key, lambda key_url: print(f'Loading in the background. The dataset is ready for use but will be slower until fully loaded.\n<a href="matlab:web(\'{key_url}\',\'-browser\')">Click to view progress in browser</a>'))

            _show_output_func = _show_output_matlab


        case ExecutionMode.shell:
            from datapipes.utils import output_server

            _set_output_func = output_server.set_output
            _show_output_func = output_server.show_output
        
        case _:
            raise ValueError(f"Unknown execution mode: {mode = }")
    _initialized = True
