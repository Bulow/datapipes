import os
os.environ["PYTHONIOENCODING"] = "utf-8"

from pathlib import Path
from functools import partial, lru_cache

from datapipes.gradio_ui.ui_utils import html_progress_bar_layered
from datapipes.gradio_ui.ui_utils import normalize_selected

import os
from pathlib import Path
from typing import List, Optional, Tuple, Literal

import gradio as gr

def pick_folder(initial_dir: str) -> str:
    """
    Opens a native folder picker using tkinter.
    Returns selected folder path or "" if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            initialdir=initial_dir or os.getcwd(),
            title="Select folder",
            mustexist=True,
        )
        root.destroy()
        return selected or initial_dir
    except Exception:
        # If tkinter isn't available (common on some servers), just no-op.
        return ""

def list_files_in_folder(folder: str) -> List[str]:
    if not folder:
        return []
    p = Path(folder).expanduser()
    if not p.exists() or not p.is_dir():
        return []
    # Keep it simple: list only files directly under the folder.
    return [str(x) for x in sorted(p.iterdir()) if x.is_file()]

def folder_picker(default_folder: Path=Path(os.getcwd()), label: str="Pick folder", add_drive_stats_bar: bool=True) -> Tuple[gr.Textbox, gr.HTML]:
    with gr.Row():
        with gr.Column():
            # with gr.Column():
            # gr.Markdown(f"**{label}**")
            folder = gr.Textbox(
                value=default_folder,
                label=f"{label}",
                placeholder="/path/to/folder",
                min_width=400,
            )
            pick_input_btn = gr.Button("Browse…", size="sm")

            if add_drive_stats_bar:
                drive_stats = gr.HTML(drive_stats_bar(default_folder))
    
        pick_input_btn.click(
            fn=lambda current: pick_folder(current),
            inputs=folder,
            outputs=folder,
        )

        return folder, drive_stats if add_drive_stats_bar else folder

def connect_drive_stats_bar(folder: gr.Textbox, drive_stats: gr.HTML, file_explorer: Optional[gr.FileExplorer]=None, ignore_selection_if_path_on_same_device_as: Optional[gr.Textbox]=None, selection_size_action: Literal["add_selection_size", "remove_selection_size"]="add_selection_size"):
    if ignore_selection_if_path_on_same_device_as is not None:
        assert file_explorer is not None, "When ignore_selection_if_path_on_same_device_as is set, file_explorer must also be set"

    inputs = [i for i in [folder, file_explorer, ignore_selection_if_path_on_same_device_as] if i is not None]
    
    update_drive_stats = partial(drive_stats_bar, selection_size_action=selection_size_action)

    if file_explorer is not None:
        file_explorer.change(
            fn=update_drive_stats,
            inputs=inputs,
            outputs=drive_stats
        )

    folder.change(
        fn=update_drive_stats,
        inputs=inputs,
        outputs=drive_stats,
    )

    timer = gr.Timer(0.5)
    timer.tick(
        fn=update_drive_stats,
        inputs=inputs,
        outputs=drive_stats
    )
   
def human_readable_filesize(size_bytes: int, dp=2):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024.0:
            return f"{size_bytes:.{dp}f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.{dp}f} PB"

from pathlib import Path
import shutil
def drive_stats_bar(path: Path|str, selected=None, ignore_selection_if_path_on_same_device_as: Optional[Path|str]=None, selection_size_action: Literal["add_selection_size", "remove_selection_size"]="add_selection_size"): # TODO: Support setting a sign for the direction of the selection size change
    path = Path(path).resolve()

    usage = shutil.disk_usage(path)

    free_space = usage.free
    total_space = usage.total
    used_space = total_space - free_space

    if ignore_selection_if_path_on_same_device_as is not None and path.parts[0] == Path(ignore_selection_if_path_on_same_device_as).resolve().parts[0]:
        size_of_selected = 0
    else:
        size_of_selected = compute_size_of_selected(normalized_selection=normalize_selected(selected))

    match selection_size_action:
        case "add_selection_size":
            new_size = used_space + size_of_selected

            lower_size = used_space
            upper_size = new_size
            size_used_for_color_selection = upper_size
        case "remove_selection_size":
            new_size = used_space - size_of_selected

            lower_size = new_size
            upper_size = used_space
            size_used_for_color_selection = lower_size

    lower_pct = (lower_size / total_space) * 100.0
    upper_pct = (upper_size / total_space) * 100.0
    color_pct = (size_used_for_color_selection / total_space) * 100.0

    def bar_color(pct: float) -> str:
        if pct < 80.0:
            return "green"
        elif pct < 90.0:
            return "orange"
        else:
            return "red"

    # html = f'''
    # <div class="barcell" style="margin: 6px;">
    #     <div class="bartext">
    #         <span class="muted">{path.parts[0]}</span>
    #         <span class="muted">Free space: {human_readable_filesize(free_space)}</span>
    #     </div>
    #     {html_progress_bar_slim(progress_pct=used_pct, color=bar_color(used_pct))}
    #     <div class="bartext">
    #         <span class="muted">Used: {used_pct:.1f}%</span>
    #         <span class="muted">Total: {human_readable_filesize(usage.total)}</span>
    #     </div>
    # </div>
    # <br>
    # '''"→"

    # TODO: Handle case where input and output are on the same drive

    html = f'''
    <div class="barcell" style="margin: 6px;">
        <div class="bartext">
            <span class="muted">{path.parts[0]}</span>
            <span class="muted">Free space: {human_readable_filesize(free_space)}{"" if size_of_selected == 0 else f" → {human_readable_filesize(total_space - new_size)}"}</span>
        </div>
        {html_progress_bar_layered(progress_pct_low=lower_pct, color=bar_color(color_pct), progress_pct_high=upper_pct, opacity_high=0.5)}
        <div class="bartext">
            <span class="muted">Used: {(used_space / total_space) * 100.0:.1f}%</span>
            <span class="muted">Total: {human_readable_filesize(usage.total)}</span>
        </div>
    </div>
    <br>
    '''

    return html

@lru_cache(maxsize=128)
def compute_size_of_selected(normalized_selection: Tuple[str]):
    if normalized_selection is None:
        return 0
    selected_paths = [Path(s) for s in normalized_selection]
    sizes = [p.stat().st_size for p in selected_paths]
    return sum(sizes)