import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import os
from typing import Optional
import gradio as gr
from typing import Optional
import gradio as gr

from datapipes.gradio_ui.ui_utils import get_ui_css

from datapipes.tools.dataset_wrangler_app.compress_app import compress_ui
from datapipes.tools.dataset_wrangler_app.copy_app import copy_ui
from datapipes.tools.dataset_wrangler_app.delete_app import delete_ui

from datapipes.gradio_ui import parallel_process_ui

from datapipes.gradio_ui.sys_stats import system_stats_html

def launch_dataset_wrangler(
    default_input_dir: Optional[str] = None,
    default_output_dir: Optional[str] = None,
) -> gr.Blocks:
    title: str = "Dataset Wrangler"
    description: str = "emilbulow@cfin.au.dk"
    in_browser: bool=True

    with gr.Blocks(title=title, fill_height=True, fill_width=True, analytics_enabled=False) as app:
        gr.Markdown(f'<span style="font-size:1.5em">{title}</span> / <span style="font-size:1em">{description}</span>')
        with gr.Tab("Compress datasets losslessly"):
            # compress_rls_ui(convert_one=convert_one, default_input_dir=default_input_dir, default_output_dir=default_output_dir)
            compress_ui(default_input_dir=default_input_dir, default_output_dir=default_output_dir)
        with gr.Tab("Safely move datasets"):
            # move_files_safely_ui(convert_one=convert_one, default_input_dir=default_input_dir, default_output_dir=default_output_dir)
            copy_ui(default_input_dir=default_input_dir, default_output_dir=default_output_dir)

        with gr.Tab("Delete safely verified files"):
            delete_ui(default_input_dir=default_input_dir, default_output_dir=default_output_dir)
        
        
        with gr.Sidebar(position="right", open=False):
            stats_panel = gr.HTML(system_stats_html, every=1)

    css = "\n".join((
        get_ui_css(),
        parallel_process_ui.get_css()
    ))
    app.launch(css=css, inbrowser=in_browser, pwa=True, theme=gr.themes.Default())
    return app


