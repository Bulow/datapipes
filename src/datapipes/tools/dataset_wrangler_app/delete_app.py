import time
from pathlib import Path
import os
import logging
import traceback

from datapipes.gradio_ui.sys_stats import system_stats_html

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from typing import Dict, Any, Tuple

import gradio as gr

from datapipes.gradio_ui.gradio_folder_picker import folder_picker, connect_drive_stats_bar
from datapipes.gradio_ui.ui_utils import normalize_selected

# from datapipes.tools.dataset_wrangler_app.safe_copy_task import copy_file_chunked, verify_hashes
from datapipes.tools.dataset_wrangler_app import compress_task

from datapipes.gradio_ui.parallel_process_manager import ParallelProcessManager, ParallelProcessTask
from datapipes.gradio_ui.parallel_process_ui import render_progress_html

Task = ParallelProcessTask
manager = ParallelProcessManager(run_task=compress_task.compress_file, verify_task=compress_task.verify_hashes)


def plan_and_start(src_dir: str, selected) -> Tuple[Dict[str, Any], str]:
    try:
        logger.info(f"Deleting safely verified files only.")
        src_dir = (src_dir or "").strip()
        src_dir_path = Path(src_dir)
        selected_paths = normalize_selected(selected)
        dest_dir = (dest_dir or "").strip()

        if not selected_paths:
            logger.warning("No files selected")
            return gr.update(value="Pick at least one file."), render_progress_html([]), []
        if not dest_dir:
            logger.warning("No destination directory specified")
            return gr.update(value="Choose a destination directory."), render_progress_html([]), []

        dest_dir_path = Path(dest_dir).expanduser().resolve()
        if dest_dir_path.exists() and not dest_dir_path.is_dir():
            logger.error(f"Destination path exists but is not a directory: {dest_dir_path}")
            return (
                gr.update(value="Destination path exists but is not a directory."),
                render_progress_html([]),
                [],
            )

        tasks: list[Task] = []
        for src in selected_paths:
            src_path = Path(src)
            if not src_path.exists() or not src_path.is_file():
                logger.warning(f"Skipping invalid file: {src}")
                continue
            size = src_path.stat().st_size
            dst_path = dest_dir_path / src_path.relative_to(src_dir_path)
            dst_path.mkdir(parents=True, exist_ok=True)
            dst = str(dst_path)

            tasks.append(Task(src=str(src_path.resolve()), dst=dst, size=size))

        if not tasks:
            logger.warning("No valid files found after filtering")
            return gr.update(value="No valid files selected (must be existing files)."), render_progress_html([]), []

        logger.info(f"Planning to copy {len(tasks)} files")
        manager.reset()
        manager.enqueue(tasks)
        manager.start(max_workers=1)

        rows, queued_list = manager.snapshot()
        logger.info(f"Started deleting files")
        return (
            gr.update(value=f"Started deleting {len(tasks)} safely verified files."),
            render_progress_html(rows),
        )
    except Exception as e:
        logger.error(f"Error in plan_and_start: {e}", exc_info=True)
        return gr.update(value=f"Error: {str(e)}"), render_progress_html([]), []

def cancel_all():
    try:
        logger.info("Cancelling all copy operations")
        manager.cancel()
        rows, queued_list = manager.snapshot()
        logger.info("All operations cancelled")
        return gr.update(value="Cancelled."), render_progress_html(rows), queued_list
    except Exception as e:
        logger.error(f"Error in cancel_all: {e}", exc_info=True)
        return gr.update(value=f"Error cancelling: {str(e)}"), render_progress_html([]), []


def live_updates():
    try:
        logger.debug("Starting live updates")
        while True:
            rows, queued_list = manager.snapshot()
            yield render_progress_html(rows)
            if manager.is_done():
                logger.info("All copy operations completed")
                break
            time.sleep(0.15)
    except Exception as e:
        logger.error(f"Error in live_updates: {e}", exc_info=True)
        yield render_progress_html([])

def delete_ui(default_input_dir: Path=Path(os.getcwd()), default_output_dir: Path=Path(os.getcwd())):
    # gr.Markdown(
    #     "## Parallel File Copier\n"
    #     "Pick files, choose a destination folder, and copy in parallel with live progress."
    # )

    with gr.Row():
        with gr.Column(scale=2):
            file_explorer = gr.FileExplorer(
                root_dir=default_input_dir,
                label="Select files (multi-select)",
                file_count="multiple",
                glob="**/*.safe_to_delete",
                
            )
        with gr.Column(scale=1):

            input_folder, input_drive_stats = folder_picker(default_folder=default_input_dir, label="From folder")
            connect_drive_stats_bar(input_folder, input_drive_stats, file_explorer=file_explorer, selection_size_action="remove_selection_size")


            # input_folder = folder_picker(default_folder=default_input_dir, label="From folder", file_explorer=file_explorer, selection_size_action="remove_selection_size")
            # output_folder = folder_picker(default_folder=default_output_dir, label="Output folder")
            # output_folder = gr.Textbox(value=R"D:\temp_data_safe_to_delete", label="Destination directory", placeholder="/path/to/output")
            # move_op_type = gr.Dropdown(choices=["copy", "move", "mark"], value="mark", label="Operation")
            # gr.Checkbox(True, label="Verify destination file integrity", interactive=False)
            # workers = gr.Slider(1, 8, value=1, step=1, label="Max parallel compressors")
            start_btn = gr.Button("Delete safely verified files", variant="primary", elem_id="run_btn")
            cancel_btn = gr.Button("Cancel", variant="stop", interactive=False)
            status = gr.Markdown()

    with gr.Row():
        progress_html = gr.HTML(label="Progress")

    
    # move_op_type.change

    def disable_button():
        return gr.Button("Deleting safely verified files...", interactive=False), gr.Button(interactive=True)

    def enable_button():
        return gr.Button("Delete safely verified files", interactive=True), gr.Button(interactive=False)

    start_btn.click(
        disable_button,
        outputs=[start_btn, cancel_btn],
        queue=False,
    ).then(
        fn=plan_and_start,
        inputs=[input_folder, file_explorer],
        outputs=[status, progress_html],
    ).then(
        fn=live_updates,
        inputs=None,
        outputs=[progress_html],
    ).then(
        enable_button,
        outputs=[start_btn, cancel_btn],
        queue=False,
    )

    cancel_btn.click(
        fn=cancel_all,
        inputs=None,
        outputs=[status, progress_html],
    )

    input_folder.change(
        fn=lambda p: gr.FileExplorer(root_dir=p or os.getcwd(), file_count="multiple"),
        inputs=input_folder,
        outputs=file_explorer,
    )

    
    
