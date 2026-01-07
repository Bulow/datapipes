from typing import Any, Optional


def normalize_selected(val: Any) -> tuple:
    """
    FileExplorer may return a single str path or list[str] depending on config/version.
    Normalize to tuple.
    """
    if val is None:
        return ()
    if isinstance(val, str):
        return (val,)
    if isinstance(val, (list, tuple)):
        return tuple(str(x) for x in val)
    return (str(val),)


def escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )

def get_ui_css() -> str:
    return R"""
.indented {
    padding-left: 30px; 
}


.gradio-fileexplorer {
    height: 100%;
}

/* Button spinner: shows whenever the underlying <button> is disabled */
#run_btn button[disabled]{
    position: relative;
    padding-right: 2.4em;
}
#run_btn button[disabled]::after{
    content: "";
    position: absolute;
    right: 0.9em;
    top: 50%;
    width: 1em;
    height: 1em;
    margin-top: -0.5em;
    border: 2px solid currentColor;
    border-right-color: transparent;
    border-radius: 999px;
    animation: spin 0.75s linear infinite;
    opacity: 0.9;
}
@keyframes spin { to { transform: rotate(360deg); } }


/* ==== Base Progress Bar Styles ==== */
.pb-container {
  background-color: #e9ecef; /* Track color */
  border-radius: .25rem; /* Default Bootstrap-like radius */
  overflow: hidden; /* Important for border-radius on fill */
  box-shadow: inset 0 1px 2px rgba(0,0,0,.075);
  height: 20px; /* Default height */
  position: relative; /* For labels if any */
}

.pb-fill {
  height: 100%;
  background-color: #007bff; /* Default blue, will be overridden by color classes */
  transition: width .1s ease;
  display: flex; /* For label alignment */
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 0.75rem;
  font-weight: 500;
  white-space: nowrap;
}


/* Source: https://codeshack.io/pure-css-modern-progress-bars-collection/*/

/* ==== Style Variation 4: Rounded (pb-rounded) ==== */
.pb-rounded {
  border-radius: 50px; /* Fully rounded container */
  height: 22px; /* Slightly taller for rounded aesthetic */
}
.pb-rounded .pb-fill {
  border-radius: 50px; /* Fully rounded fill */
}
.pb-rounded .pb-label {
  padding: 0 10px;
}

/* ==== Style Variation 5: Slim (pb-slim) ==== */
.pb-slim {
  height: 8px;
  border-radius: 4px;
  box-shadow: none; /* Often no shadow for slim bars */
}
.pb-slim .pb-fill {
  /* No text label typical for slim bars */
}

/* ==== Color Variations for .pb-fill ==== */
/* Blue (Default, but explicit class for clarity) */
.pb-blue .pb-fill { background-color: #2196F3; }

/* Green */
.pb-green .pb-fill { background-color: #4CAF50; }

/* Green */
.pb-green-ratio { background-color: #4CAF50BF; }
.pb-green-ratio .pb-fill { background-color: #4CAF50; }

/* Red */
.pb-red .pb-fill { background-color: #f44336; }

/* Orange */
.pb-orange .pb-fill { background-color: #ff9800; }

/* Grey */
.pb-grey .pb-fill { background-color: #78909c; }

/* Tables */
:root{
--gr-border: var(--border-color-primary, #e5e7eb);
--gr-bg: var(--background-fill-primary, #fff);
--gr-bg-2: var(--background-fill-secondary, #f6f7f8);
--gr-text: var(--body-text-color, #111827);
--gr-muted: var(--body-text-color-subdued, #6b7280);
--gr-radius: var(--radius-lg, 10px);
--gr-font: var(--font, system-ui);
}

/* Hard override Gradio defaults */
.file-joblist,
.file-joblist table, 
.file-joblist table th, 
.file-joblist table td {
border: none !important;
outline: none !important;
box-shadow: none !important;
}

.file-joblist {
font-family: var(--gr-font);
color: var(--gr-text);
overflow: hidden;
}

.file-joblist table {
width: 100%;
border-collapse: separate;
border-spacing: 0;
font-size: 14px;
}

.file-joblist thead th {
background: #80808010;
font-size: 12px;
font-weight: 600;
padding: 10px 12px;
text-align: left;
border-bottom: 1px solid var(--gr-border);
}

.file-joblist tbody td {
padding: 10px 12px;
border-bottom: 1px solid var(--gr-border);
}

.file-joblist tbody tr:last-child td {
border-bottom: none;
}

.file-joblist tbody tr:hover {
background: #80808010;
}


.progress-bar {
  width: 100%;
  height: 7px;
  background-color: #e0e0e0;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
}

.segment {
  height: 100%;
}

.green {
  background-color: #2ecc71;
}

.light-green {
  background-color: #a9dfbf;
}

.gray {
  background-color: #cccccc;
  flex: 1; /* fills remaining space */
}

.progress-bar {
  width: 100%;
  height: 7px
  --x: 30%;
  --y: 55%;
}

.progress-bar-low { width: var(--x); }
.progress-bar-high { width: calc(var(--y) - var(--x));}
    """

def html_progress_bar(progress_pct: float = 0.0, color: str="green") -> str:
    return f"""<div class="pb-container pb-rounded pb-{color}"><div class="pb-fill" style="width: {progress_pct:.1f}%;"><span class="pb-label">{progress_pct:.1f}%</span></div></div>"""

def html_progress_bar_slim(progress_pct: float = 0.0, color: str="green") -> str:
    # return f"""<div class="pb-container pb-rounded pb-{color}"><div class="pb-fill" style="width: {progress_pct:.1f}%;"><span class="pb-label">{progress_pct:.1f}%</span></div></div>"""

    return f"""<div class="pb-container pb-slim pb-{color}"><div class="pb-fill" style="width: {progress_pct:.1f}%;"></div></div>"""

def html_progress_bar_layered(
    progress_pct_low: float = 0.0, 
    color: str="dark-green",
    progress_pct_high: Optional[float] = None, 
    opacity_high: Optional[float]=None,
) -> str:
    return f"""
<div class="progress-bar" style="--x:{progress_pct_low:.1f}%; --y:{progress_pct_high or progress_pct_low:.1f}%;">
    <div class="segment progress-bar-low" style="background-color:{color};"></div>
    <div class="segment progress-bar-high" style="background-color:{color};opacity:{opacity_high or 0.5};"></div>
    <div class="segment"></div>
</div>
  """