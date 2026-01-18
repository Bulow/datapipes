from pathlib import Path
from typing import List, Dict, Optional

from datapipes.gradio_ui.ui_utils import escape_html
from datapipes.gradio_ui.ui_utils import html_progress_bar_slim, html_progress_bar_layered
from datapipes.gradio_ui.parallel_process_manager import SnapshotRow, TaskStatus

def _status_badge(status: TaskStatus, status_display_label: Optional[str]=None) -> str:
    cls = {
        TaskStatus.QUEUED: "b-queued",
        TaskStatus.RUNNING: "b-running",
        TaskStatus.VERIFYING: "b-verifying",
        TaskStatus.DONE: "b-done",
        TaskStatus.ERROR: "b-error",
        TaskStatus.CANCELLED: "b-cancelled",
    }.get(status, "b-queued")
    status_label = status_display_label or escape_html(status)
    return f'<span class="badge {cls}">{status_label}</span>'

def _format_eta(seconds: Optional[float]) -> str:
    # return f"{seconds:.2f}"
    if seconds is None:
        return "ETA —"
    if seconds < 0:
        seconds = 0
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"ETA {h:02d}:{m:02d}:{sec:02d}"
    return f"ETA {m:02d}:{sec:02d}"


def _format_rate(bps: int, status: TaskStatus) -> str:
    # Only show rate meaningfully while running; otherwise show dash.
    if status not in [TaskStatus.RUNNING, TaskStatus.VERIFYING]:
        return "— MB/s"
    if bps <= 0:
        return "0.00 MB/s"
    return f"{human_readable_filesize(bps)}/s"


def _render_section(title: str, rows: List[SnapshotRow], empty_text: str) -> str:
    if not rows:
        return f"""
        <div class="section">
          <div class="sectitle">{escape_html(title)}</div>
          <div class="empty">{escape_html(empty_text)}</div>
        </div>
        """

    trs = []

    colors = {
        TaskStatus.QUEUED: "grey",
        TaskStatus.ERROR: "red",
        TaskStatus.CANCELLED: "grey",
        TaskStatus.RUNNING: "blue",
        TaskStatus.VERIFYING: "orange",
        TaskStatus.DONE: "green-ratio",
    }
    
    for r in rows:
        file_name = escape_html(Path(r.src_path).name)
        full_path = escape_html(r.src_path) if r.status != TaskStatus.DONE else escape_html(r.dst_path)
        status = r.status
        status_display_label = r.status_display_label
        percent = r.percent
        # label = f'{r.copied_mb:.2f} / {r.total_mb:.2f} MB'
        label = f"{human_readable_filesize(r.bytes_processed)} {"→" if status == TaskStatus.DONE else "/"} {human_readable_filesize(r.bytes_total)}"
        rate = _format_rate(r.ema_bps, status)
        eta = _format_eta(r.eta_seconds)
        custom_display_stats = r.custom_display_stats or []
        # <span>{eta}</span>
        # <span class="muted">{rate}</span>
        if status in [TaskStatus.RUNNING, TaskStatus.VERIFYING]:
            custom_display_stats += [eta, rate]
        # 
        trs.append(
            f"""
            <tr>
              <td class="filecell">
                <div class="filename" title="{full_path}">{file_name}</div>
                <div class="filepath" title="{full_path}">{full_path}</div>
              </td>
              <td class="statuscell">{_status_badge(status, status_display_label=status_display_label)}</td>
              <td class="barcell">
                <div class="bartext">
                  <span>{percent:.1f}%</span>
                  <span class="muted">{escape_html(label)}</span>
                </div>
                {html_progress_bar_slim(progress_pct=percent, color=f"{colors[status] or "grey"}")}
                <div class="bartext">
                  {"\n".join([f'<span class="muted">{stat}</span>' for stat in custom_display_stats])}
                </div>
              </td>
            </tr>
            """
        )

    table = f"""
    <table class="ptbl">
      <thead>
      <tr>
        <th style="width:auto">File</th>
        <th style="width:16em">Status</th>
        <th style="width:35%">Progress</th>
      </tr>
      </thead>
      <tbody>
      {''.join(trs)}
      </tbody>
    </table>
    """

    return f"""
    <div class="section">
      <div class="sectitle">{escape_html(title)}</div>
      <div class="file-joblist">
        {table}
      </div>
    </div>
    """

def get_css()-> str:
    badge_bg_opacity = "BF"
    style = f"""
    <style>
      .section {{ margin: 10px 0 18px 0; }}
      .sectitle {{ font-weight: 700; margin: 6px 0 8px 0; font-size: 14px; }}
      .filename {{ font-weight: 600; }}
      .filepath {{ font-size: 12px; opacity: .7; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; /*max-width: 1px;*/ }}
      .filecell {{ max-width: 0; }}
      .barwrap progress {{ width: 100%; height: 16px; }}
      .bartext {{ display: flex; justify-content: space-between; font-size: 12px; margin-top: 6px; gap: 10px; }}
      .muted {{ opacity: .75; white-space: nowrap; }}
      .badge {{ display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; line-height: 18px; }}
      .b-queued {{ background: #78909c{badge_bg_opacity}; }}
      .b-running {{ background: #2196F3{badge_bg_opacity}; }}
      .b-verifying {{ background: #ff9800{badge_bg_opacity}; }}
      .b-done {{ background: #4CAF50{badge_bg_opacity}; }}
      .b-error {{ background: #f44336{badge_bg_opacity}; }}
      .b-cancelled {{ background: #78909c{badge_bg_opacity}; }}
      .empty {{ opacity: .8; padding: 10px 0; }}
    </style>
    """
    return style

def render_progress_html(rows: List[SnapshotRow]) -> str:
    running = [r for r in rows if r.status in [TaskStatus.RUNNING, TaskStatus.VERIFYING]]
    queued = [r for r in rows if r.status == TaskStatus.QUEUED]
    finished = [r for r in rows if r.status in (TaskStatus.DONE, TaskStatus.ERROR, TaskStatus.CANCELLED)]

    style = get_css()

    body = (
        _render_section("Running", running, "Nothing is currently copying.")
        + _render_section("Queued", queued, "No files are waiting.")
        + _render_section("Finished", finished, "No finished tasks yet.")
    )
    return style + body



def human_readable_filesize(size_bytes: int, dp=2):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024.0:
            return f"{size_bytes:.{dp}f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.{dp}f} PB"

from pathlib import Path
import shutil
def drive_stats_bar(path: Path):
    path = path.resolve()
    usage = shutil.disk_usage(path)

    free_space = usage.free
    total_space = usage.total
    used_space = total_space - free_space

    html = f'''
    <div class="barcell">
        <div class="bartext">
            <span class="muted">Drive: {path.parts[0]}</span>
            <span class="muted">Free space: {human_readable_filesize(free_space)}</span>
        </div>
        {html_progress_bar_slim(progress_pct_low=float(used_space / total_space), color_low=f"green")}
        <div class="bartext">
            <span class="muted">Used: {used_space / total_space:.1f}</span>
            <span class="muted">Total: {human_readable_filesize(usage.total)}</span>
        </div>
    </div>
    '''

    return html