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
from datapipes.gradio_ui.gradio_folder_picker import folder_picker

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
            with gr.Accordion("Add shortcuts", open=False):
                shortcut_input_folder, _ = folder_picker(default_folder=default_input_dir, label="Default input folder", add_drive_stats_bar=False)
                shortcut_output_folder, _ = folder_picker(default_folder=default_output_dir, label="Default output folder", add_drive_stats_bar=False)#, file_explorer=file_explorer, selection_size_action="add_selection_size")

                add_shortcut_desktop_button = gr.Button("Add shortcut to desktop")
                add_shortcut_desktop_button.click(
                    fn = add_shortcut_desktop,
                    inputs=[shortcut_input_folder, shortcut_output_folder]
                )

                add_shortcut_start_menu_button = gr.Button("Add to start menu")
                add_shortcut_start_menu_button.click(
                    fn = add_shortcut_start_menu,
                    inputs=[shortcut_input_folder, shortcut_output_folder]
                )

            stats_panel = gr.HTML(system_stats_html, every=1)

    css = "\n".join((
        get_ui_css(),
        parallel_process_ui.get_css()
    ))
    app.launch(css=css, inbrowser=in_browser, pwa=True, theme=gr.themes.Default())
    return app

def add_shortcut_desktop(input_folder: Optional[gr.Textbox], output_folder: Optional[gr.Textbox]):
    import subprocess

    input_arg = f' --in "{input_folder}"' if input_folder is not None else ""
    output_arg = f' --out "{output_folder}"' if output_folder is not None else ""

    ps_script = f'''
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\DatasetWrangler.lnk")
$Shortcut.TargetPath = "C:\Windows\System32\cmd.exe"
$Shortcut.Arguments = '/c datapipes dataset-wrangler{input_arg}{output_arg} && pause'
$Shortcut.WorkingDirectory = "$env:USERPROFILE"
$Shortcut.Save()
    '''

    subprocess.run(["powershell", "-Command", ps_script], check=True)

    gr.Info("Added shortcut to desktop", duration=5)


def add_shortcut_start_menu(input_folder: Optional[gr.Textbox], output_folder: Optional[gr.Textbox]):
    import subprocess

    input_arg = f' --in "{input_folder}"' if input_folder is not None else ""
    output_arg = f' --out "{output_folder}"' if output_folder is not None else ""

    ps_script = f'''
$WshShell = New-Object -ComObject WScript.Shell

# Start Menu (current user)
$StartMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"

# Optional subfolder in Start Menu
$Folder = Join-Path $StartMenu "datapipes"
New-Item -ItemType Directory -Path $Folder -Force | Out-Null

$ShortcutPath = Join-Path $Folder "DatasetWrangler.lnk"
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)

# Make the shortcut run a command:
$Shortcut.TargetPath = "$env:WINDIR\System32\cmd.exe"
$Shortcut.Arguments = '/c datapipes dataset-wrangler{input_arg}{output_arg} && pause'
$Shortcut.WorkingDirectory = $env:USERPROFILE

# Optional: icon (EXE, DLL, or ICO). ",0" is icon index inside the file.
# $Shortcut.IconLocation = "$env:WINDIR\System32\cmd.exe,0"

$Shortcut.Save()

Write-Host "Created: $ShortcutPath"
'''

    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script], check=True)

    gr.Info("Added shortcut to start menu", duration=5)

