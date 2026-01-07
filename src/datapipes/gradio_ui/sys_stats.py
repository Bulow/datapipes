from __future__ import annotations

import time
import html
import platform
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import psutil


# ----------------------------
# Optional GPU (NVIDIA) support
# ----------------------------
def _import_nvml():
    """
    Prefer nvidia-ml-py (new), fall back to pynvml (deprecated).
    Returns (nvml_module, error_message).
    """
    try:
        import nvidia_ml_py as nvml  # type: ignore
        return nvml, None
    except Exception:
        try:
            import pynvml as nvml  # type: ignore
            return nvml, "Using deprecated pynvml; install nvidia-ml-py to silence warnings."
        except Exception:
            return None, "NVML not available (install nvidia-ml-py and ensure NVIDIA drivers are installed)."


def _nvml_name_to_str(name_val) -> str:
    if isinstance(name_val, bytes):
        return name_val.decode("utf-8", errors="ignore")
    return str(name_val)


def _get_nvidia_gpus_via_nvml():
    """
    Returns (gpu_list, error_message).
    Uses NVML via nvidia-ml-py (preferred) or pynvml (fallback).
    """
    nvml, warn = _import_nvml()
    if nvml is None:
        return None, warn

    try:
        nvml.nvmlInit()
        count = nvml.nvmlDeviceGetCount()
        gpus = []

        for i in range(count):
            h = nvml.nvmlDeviceGetHandleByIndex(i)

            name = _nvml_name_to_str(nvml.nvmlDeviceGetName(h))
            mem = nvml.nvmlDeviceGetMemoryInfo(h)

            util = None
            try:
                u = nvml.nvmlDeviceGetUtilizationRates(h)
                util = {"gpu": int(u.gpu), "mem": int(u.memory)}
            except Exception:
                pass

            temp_c = None
            try:
                temp_c = int(nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass

            power_w = None
            power_limit_w = None
            try:
                power_w = nvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                pass

            try:
                power_limit_w = nvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
            except Exception:
                pass

            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "mem_used": mem.used,
                    "mem_total": mem.total,
                    "util": util,
                    "temp_c": temp_c,
                    "power_w": power_w,
                    "power_limit_w": power_limit_w,
                }
            )

        try:
            nvml.nvmlShutdown()
        except Exception:
            pass

        # Surface warning only if we fell back to pynvml
        return gpus, warn

    except Exception as e:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass
        return None, f"NVIDIA GPU stats error: {e}"

# ----------------------------
# Helpers
# ----------------------------
def _bytes_to_gb(x: float) -> float:
    return x / (1024 ** 3)

def _bytes_to_mb(x: float) -> float:
    return x / (1024 ** 2)

def _fmt_gb(x_bytes: float) -> str:
    return f"{_bytes_to_gb(x_bytes):.2f} GB"

def _fmt_mb_per_s(x_bytes_per_s: float) -> str:
    return f"{_bytes_to_mb(x_bytes_per_s):.2f} MB/s"

def _clamp_pct(x: float) -> float:
    return max(0.0, min(100.0, x))

def _bar(pct: float, label: str, sublabel: str = "") -> str:
    pct = _clamp_pct(pct)
    label_esc = html.escape(label)
    sub_esc = html.escape(sublabel)

    return f"""
    <div class="pm-row">
      <div class="pm-row-top">
        <div class="pm-row-label">{label_esc}</div>
        <div class="pm-row-val pm-mono">{pct:.1f}%</div>
      </div>
      <div class="pm-bar" role="progressbar" aria-valuenow="{pct:.1f}" aria-valuemin="0" aria-valuemax="100">
        <div class="pm-fill" style="width:{pct:.1f}%;"></div>
      </div>
      {"<div class='pm-row-sub'>" + sub_esc + "</div>" if sublabel else ""}
    </div>
    """

def _kv(label: str, value: str) -> str:
    return f"""
    <div class="pm-kv">
      <div class="pm-k">{html.escape(label)}</div>
      <div class="pm-v pm-mono">{html.escape(value)}</div>
    </div>
    """

def human_readable_filesize(size_bytes: int, dp=2):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024.0:
            return f"{size_bytes:.{dp}f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.{dp}f} PB"


@dataclass
class PerfMonitor:
    """
    Call snapshot_html() periodically to get an HTML string themed by Gradio.
    First call yields 0 speeds (no previous baseline).
    """
    prev_ts: Optional[float] = None
    prev_disk: Dict[str, psutil._common.sdiskio] = field(default_factory=dict)
    prev_net_total: Optional[psutil._common.snetio] = None

    def snapshot_html(self) -> str:
        now = time.time()

        # CPU
        cpu_total = psutil.cpu_percent(interval=0.0)
        cpu_per_core = psutil.cpu_percent(interval=0.0, percpu=True)
        cpu_freq = psutil.cpu_freq()
        load_avg = None
        try:
            load_avg = psutil.getloadavg()
        except Exception:
            load_avg = None

        # Memory
        vm = psutil.virtual_memory()

        # Disk space (root)
        try:
            du = psutil.disk_usage("/")
        except Exception:
            du = None

        # Bandwidth: Disk IO
        disk_io = psutil.disk_io_counters(perdisk=True) or {}
        disk_speeds = {}  # name -> (read_Bps, write_Bps)

        dt = None
        if self.prev_ts is not None:
            dt = max(1e-6, now - self.prev_ts)
            for name, cur in disk_io.items():
                prev = self.prev_disk.get(name)
                if prev is None:
                    continue
                read_bps = (cur.read_bytes - prev.read_bytes) / dt
                write_bps = (cur.write_bytes - prev.write_bytes) / dt
                disk_speeds[name] = (max(0.0, read_bps), max(0.0, write_bps))

        # Bandwidth: Network
        net_total = psutil.net_io_counters(pernic=False)
        net_speeds = {}  # "Total" -> (recv_Bps, sent_Bps)
        if self.prev_ts is not None and dt is not None and net_total is not None and self.prev_net_total is not None:
            recv_bps = (net_total.bytes_recv - self.prev_net_total.bytes_recv) / dt
            sent_bps = (net_total.bytes_sent - self.prev_net_total.bytes_sent) / dt
            net_speeds["Total"] = (max(0.0, recv_bps), max(0.0, sent_bps))

        # Update baselines
        self.prev_ts = now
        self.prev_disk = disk_io
        self.prev_net_total = net_total

        # GPU
        gpus, gpu_err = _get_nvidia_gpus_via_nvml()

        # Host/OS
        host = platform.node() or "unknown-host"
        os_name = f"{platform.system()} {platform.release()}"

        # ----------------------------
        # Gradio-theme-friendly CSS
        # ----------------------------
        # Gradio themes expose many CSS variables; names can differ slightly by version.
        # We use a small set with fallbacks so it still looks okay.
        css = """
        <style>
          /* Scope styles to this component only */
          .pm-wrap {
            font-family: var(--font, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif);
            font-size: 13px;
            line-height: 1.35;
            color: var(--body-text-color, var(--neutral-800, #111827));
          }

          .pm-card {
            background: var(--panel-background-fill, var(--background-fill-secondary, rgba(255,255,255,0.7)));
            border: 1px solid var(--border-color-primary, var(--neutral-200, rgba(0,0,0,0.12)));
            border-radius: var(--radius-lg, 12px);
            padding: 12px;
            margin: 10px 0;
            box-shadow: var(--shadow-drop, none);
          }

          .pm-title {
            font-size: 14px;
            font-weight: 700;
            margin: 0 0 8px 0;
            color: var(--body-text-color, var(--neutral-900, #0b1220));
          }

          .pm-meta {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px 14px;
            margin-bottom: 6px;
          }

          .pm-kv { display: flex; justify-content: space-between; gap: 10px; }
          .pm-k  { color: var(--body-text-color-subdued, var(--neutral-600, #4b5563)); }
          .pm-v  { color: var(--body-text-color, var(--neutral-900, #111827)); }

          .pm-grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
          .pm-row { margin: 8px 0; }
          .pm-row-top { display:flex; justify-content: space-between; align-items: baseline; gap: 10px; }
          .pm-row-label { font-weight: 600; }
          .pm-row-val { color: var(--body-text-color-subdued, var(--neutral-600, #4b5563)); }
          .pm-row-sub { color: var(--body-text-color-subdued, var(--neutral-600, #4b5563)); margin-top: 3px; font-size: 12px; }

          .pm-bar {
            height: 9px;
            background: var(--input-background-fill, var(--background-fill-secondary, rgba(0,0,0,0.04)));
            border: 1px solid var(--border-color-primary, var(--neutral-200, rgba(0,0,0,0.12)));
            border-radius: 999px;
            overflow: hidden;
          }

          .pm-fill {
            height: 100%;
            background: var(--primary-500, var(--color-accent, #2563eb));
            width: 0%;
            transition: width .1s ease;
          }

          .pm-small { font-size: 12px; color: var(--body-text-color-subdued, var(--neutral-600, #4b5563)); }
          .pm-mono { font-variant-numeric: tabular-nums; font-feature-settings: "tnum"; }

          .pm-table { width: 100%; border-collapse: collapse; }
          .pm-table th, .pm-table td {
            text-align: left;
            padding: 6px 8px;
            border-bottom: 1px solid var(--border-color-primary, var(--neutral-200, rgba(0,0,0,0.12)));
          }
          .pm-table th { color: var(--body-text-color-subdued, var(--neutral-600, #4b5563)); font-weight: 600; }

          .pm-badge {
            display:inline-block;
            padding:2px 8px;
            border-radius: 999px;
            background: var(--input-background-fill, var(--background-fill-secondary, rgba(0,0,0,0.04)));
            border: 1px solid var(--border-color-primary, var(--neutral-200, rgba(0,0,0,0.12)));
            color: var(--body-text-color-subdued, var(--neutral-700, #374151));
          }

          /* Make core grid responsive */
          @media (max-width: 680px) {
            .pm-meta { grid-template-columns: 1fr; }
            .pm-grid2 { grid-template-columns: 1fr; }
          }

          /* === Bandwidth: sidebar-optimized === */

          .pm-bw-stack {
            display: flex;
            flex-direction: column;
            gap: 8px;
          }

          /* Sub-panels match Memory/GPU feel */
          .pm-subcard {
            background: var(--input-background-fill, var(--background-fill-secondary, rgba(0,0,0,0.04)));
            border: 1px solid var(--border-color-primary, var(--neutral-200, rgba(0,0,0,0.12)));
            border-radius: var(--radius-lg, 12px);
            padding: 8px 10px;
          }

          /* Compact subtitle */
          .pm-subtitle {
            font-size: 12.5px;
            font-weight: 700;
            margin: 0 0 6px 0;
            color: var(--body-text-color, var(--neutral-900, #0b1220));
          }

          /* --- TABLE OVERRIDES (Gradio-safe) --- */
          .pm-table-compact {
            width: 100%;
            border-collapse: collapse !important;
            border-spacing: 0 !important;
            margin: 0 !important;
          }

          /* Kill Gradio borders & spacing */
          .pm-table-compact,
          .pm-table-compact th,
          .pm-table-compact td {
            border: none !important;
            background: transparent !important;
          }

          /* Tight rows */
          .pm-table-compact th,
          .pm-table-compact td {
            padding: 3px 4px !important;
            line-height: 1.25;
            font-size: 12px;
          }

          /* Header */
          .pm-table-compact th {
            font-weight: 600;
            color: var(--body-text-color-subdued, var(--neutral-600, #4b5563));
          }

          /* Alignment helpers */
          .pm-right { text-align: right; }
          .pm-left { text-align: left; }

          /* Prevent interface names from wrapping awkwardly */
          .pm-nowrap {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }

          /* Reduce bottom noise */
          .pm-bw-note {
            font-size: 11.5px;
            margin-top: 6px;
            color: var(--body-text-color-subdued, var(--neutral-600, #4b5563));
          }


        </style>
        """

        # ----------------------------
        # Sections
        # ----------------------------
        header_html = f"""
        <div class="pm-card">
          <div class="pm-title">System</div>
          <div class="pm-meta">
            {_kv("Host", host)}
            {_kv("OS", os_name)}
            {_kv("Time", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)))}
            {_kv("Poll delta", f"{(dt if dt is not None else 0.0):.2f}s")}
          </div>
        </div>
        """

        cpu_lines = []
        cpu_lines.append(_bar(cpu_total, "CPU Total", f"{cpu_freq.current:.0f} MHz" if cpu_freq else ""))
        if load_avg:
            cpu_lines.append(f"<div class='pm-small'>Load avg: <span class='pm-mono'>{load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}</span></div>")

        core_bars = []
        for i, pct in enumerate(cpu_per_core):
            core_bars.append(_bar(pct, f"Core {i}"))

        cpu_html = f"""
        <div class="pm-card">
          <div class="pm-title">CPU</div>
          {''.join(cpu_lines)}
          <div class="pm-grid2">
            {''.join(core_bars)}
          </div>
        </div>
        """

        mem_html = f"""
        <div class="pm-card">
          <div class="pm-title">Memory</div>
          {_bar(vm.percent, "RAM", f"Used {_fmt_gb(vm.used)} / Total {_fmt_gb(vm.total)} • Available {_fmt_gb(vm.available)}")}
        </div>
        """

        if du:
            disk_space_html = f"""
            <div class="pm-card">
              <div class="pm-title">Disk Space</div>
              {_bar(du.percent, "Filesystem (/)", f"Used {_fmt_gb(du.used)} / Total {_fmt_gb(du.total)} • Free {_fmt_gb(du.free)}")}
            </div>
            """
        else:
            disk_space_html = """
            <div class="pm-card">
              <div class="pm-title">Disk Space</div>
              <div class="pm-small">Disk usage unavailable on this platform/context.</div>
            </div>
            """

        # disk_rows = []
        # for name, (r, w) in sorted(disk_speeds.items(), key=lambda kv: (kv[1][0] + kv[1][1]), reverse=True):
        #     disk_rows.append(f"""
        #       <tr>
        #         <td>{html.escape(name)}</td>
        #         <td class="pm-mono">{html.escape(_fmt_mb_per_s(r))}</td>
        #         <td class="pm-mono">{html.escape(_fmt_mb_per_s(w))}</td>
        #       </tr>
        #     """)
        # if not disk_rows:
        #     disk_rows.append("<tr><td colspan='3' class='pm-small'>No baseline yet (call once more), or no per-disk counters available.</td></tr>")

        # net_rows = []
        # for nic, (rx, tx) in sorted(net_speeds.items(), key=lambda kv: (kv[1][0] + kv[1][1]), reverse=True):
        #     net_rows.append(f"""
        #       <tr>
        #         <td>{html.escape(nic)}</td>
        #         <td class="pm-mono">{html.escape(_fmt_mb_per_s(rx))}</td>
        #         <td class="pm-mono">{html.escape(_fmt_mb_per_s(tx))}</td>
        #       </tr>
        #     """)
        # if not net_rows:
        #     net_rows.append("<tr><td colspan='3' class='pm-small'>No baseline yet (call once more), or no per-NIC counters available.</td></tr>")
        # Disk rows (unchanged sorting/logic)

        # Disk rows (logic unchanged)
        disk_rows = []
        for name, (r, w) in disk_speeds.items():
            disk_rows.append(f"""
              <tr>
                <td class="pm-left pm-nowrap">{html.escape(name)}</td>
                <td class="pm-mono pm-right">{html.escape(human_readable_filesize(r))}/s</td>
                <td class="pm-mono pm-right">{html.escape(human_readable_filesize(w))}/s</td>
              </tr>
            """)
        if not disk_rows:
            disk_rows.append(
                "<tr><td colspan='3' class='pm-bw-note'>No baseline yet</td></tr>"
            )

        # Network rows (logic unchanged)
        net_rows = []
        for nic, (rx, tx) in sorted(net_speeds.items(), key=lambda kv: (kv[1][0] + kv[1][1]), reverse=True):
            net_rows.append(f"""
              <tr>
                <td class="pm-left pm-nowrap">{html.escape(nic)}</td>
                <td class="pm-mono pm-right">{html.escape(human_readable_filesize(rx))}/s</td>
                <td class="pm-mono pm-right">{html.escape(human_readable_filesize(tx))}/s</td>
              </tr>
            """)
        if not net_rows:
            net_rows.append(
                "<tr><td colspan='3' class='pm-bw-note'>No baseline yet</td></tr>"
            )



        bandwidth_html = f"""
<div class="pm-card">
  <div class="pm-title">Bandwidth</div>

  <div class="pm-bw-stack">

    <div class="pm-subtitle">Disk</div>
      <table class="pm-table-compact pm-mono">
        <thead>
          <tr>
            <th class="pm-left">Disk</th>
            <th class="pm-right">Read</th>
            <th class="pm-right">Write</th>
          </tr>
        </thead>
        <tbody>
          {''.join(disk_rows)}
        </tbody>
      </table>

    <div class="pm-subtitle">Network</div>
      <table class="pm-table-compact pm-mono">
        <thead>
          <tr>
            <th class="pm-left">Interface</th>
            <th class="pm-right">Recv</th>
            <th class="pm-right">Send</th>
          </tr>
        </thead>
        <tbody>
          {''.join(net_rows)}
        </tbody>
      </table>

  </div>

  <div class="pm-bw-note">
    Δ bytes / poll interval
  </div>
</div>
"""


        if gpus is None:
            gpu_html = f"""
            <div class="pm-card">
              <div class="pm-title">GPU</div>
              <div class="pm-small">{html.escape(gpu_err or "GPU stats unavailable.")}</div>
            </div>
            """
        elif len(gpus) == 0:
            gpu_html = """
            <div class="pm-card">
              <div class="pm-title">GPU</div>
              <div class="pm-small">No NVIDIA GPUs detected by NVML.</div>
            </div>
            """
        else:
            gpu_cards = []
            for g in gpus:
                mem_used = g["mem_used"]
                mem_total = g["mem_total"]
                mem_pct = (mem_used / mem_total * 100.0) if mem_total else 0.0

                util_gpu = g["util"]["gpu"] if g["util"] else None
                util_mem = g["util"]["mem"] if g["util"] else None

                details = []
                details.append(_bar(mem_pct, f"GPU {g['index']}: {g['name']}", f"VRAM {_fmt_gb(mem_used)} / {_fmt_gb(mem_total)}"))
                if util_gpu is not None:
                    details.append(_bar(util_gpu, "GPU Utilization"))
                if util_mem is not None:
                    details.append(_bar(util_mem, "VRAM Controller Utilization"))

                extra_bits = []
                if g["temp_c"] is not None:
                    extra_bits.append(_kv("Temp", f"{g['temp_c']} °C"))
                if g["power_w"] is not None:
                    if g["power_limit_w"] is not None:
                        extra_bits.append(_kv("Power", f"{g['power_w']:.1f} W / {g['power_limit_w']:.1f} W"))
                    else:
                        extra_bits.append(_kv("Power", f"{g['power_w']:.1f} W"))

                gpu_cards.append(f"""
                <div class="pm-card">
                  <div class="pm-title">GPU</div>
                  {''.join(details)}
                  <div class="pm-meta">
                    {''.join(extra_bits) if extra_bits else "<div class='pm-small'>Additional GPU sensors unavailable.</div>"}
                  </div>
                </div>
                """)

            gpu_html = "".join(gpu_cards)

        return f"""
        {css}
        <div class="pm-wrap">
          {gpu_html}
          {mem_html}
          {bandwidth_html}
          {cpu_html}
          {header_html}
        </div>
        """


# ----------------------------
# Example Gradio wiring
# ----------------------------
# import gradio as gr
#
# monitor = PerfMonitor()
#
# def render():
#     return monitor.snapshot_html()
#
# with gr.Blocks() as demo:
#     out = gr.HTML(render())
#     demo.load(render, None, out, every=1)  # refresh every 1 second
#
# demo.launch()


perf_monitor = PerfMonitor()

def system_stats_html() -> str:
    return perf_monitor.snapshot_html()

