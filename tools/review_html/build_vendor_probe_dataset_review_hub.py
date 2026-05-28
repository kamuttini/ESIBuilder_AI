#!/usr/bin/env python3
"""Build a single structured HTML hub for vendor/probe dataset rect tools."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple


def _count_folders(tool_dir: Path) -> int:
    payload = tool_dir / "folders_payload.json"
    if not payload.exists():
        return 0
    try:
        data = json.loads(payload.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return 0
    if isinstance(data, list):
        return len(data)
    return 0


def _ensure_link(link_path: Path, target_dir: Path) -> None:
    if link_path.is_symlink():
        if link_path.resolve() == target_dir.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        if link_path.is_dir():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target_dir.resolve(), target_is_directory=True)


def _build_html(vendor_count: int, probe_count: int) -> str:
    return f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Dataset Review Hub - Vendor & Probe</title>
  <style>
    :root {{
      --bg: #0d1118;
      --panel: #121a27;
      --panel-2: #0f1724;
      --border: #2f3f5a;
      --text: #e8eef8;
      --muted: #9fb0ca;
      --active: #4f8cff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: radial-gradient(1300px 600px at 80% -20%, #1a2741 0%, var(--bg) 62%);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1800px;
      margin: 0 auto;
      padding: 14px;
    }}
    .top {{
      border: 1px solid var(--border);
      border-radius: 10px;
      background: linear-gradient(180deg, #172337, var(--panel));
      padding: 12px;
      display: grid;
      gap: 10px;
    }}
    h1 {{ margin: 0; font-size: 22px; }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    .stats {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      font-size: 12px;
    }}
    .chip {{
      background: #1a2a42;
      border: 1px solid #3c5275;
      border-radius: 999px;
      padding: 5px 10px;
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }}
    button.tab {{
      border: 1px solid #445d84;
      border-radius: 7px;
      background: #1b2c47;
      color: #eef4ff;
      padding: 8px 12px;
      cursor: pointer;
      font-weight: 700;
    }}
    button.tab.active {{
      background: var(--active);
      border-color: #8ab4ff;
      color: #fff;
    }}
    .actions {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .actions a {{
      color: #d7e8ff;
      text-decoration: none;
      border: 1px solid #445d84;
      border-radius: 7px;
      padding: 6px 10px;
      background: #1b2c47;
      font-size: 12px;
    }}
    .frames {{
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      background: var(--panel-2);
      min-height: calc(100vh - 220px);
      position: relative;
    }}
    iframe {{
      width: 100%;
      height: calc(100vh - 220px);
      border: 0;
      display: none;
      background: #0b1018;
    }}
    iframe.active {{
      display: block;
    }}
    @media (max-width: 900px) {{
      iframe {{
        height: calc(100vh - 250px);
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="top">
      <h1>Dataset Review Hub - Vendor + Probe</h1>
      <div class="muted">
        Un unico punto di accesso alle due interfacce. Tab: <b>Vendor</b> (linea 13) e <b>Probe</b> (linea 14).
        Shortcut hub: <code>1</code>=Vendor, <code>2</code>=Probe.
        Shortcut interni (frecce/X/R) funzionano dentro ogni tab.
      </div>
      <div class="stats">
        <span class="chip">Vendor folders: {vendor_count}</span>
        <span class="chip">Probe folders: {probe_count}</span>
      </div>
      <div class="tabs">
        <button id="tab-vendor" class="tab active" type="button">1 · Vendor</button>
        <button id="tab-probe" class="tab" type="button">2 · Probe</button>
      </div>
      <div class="actions">
        <a href="vendor_tool/index.html" target="_blank" rel="noopener">Apri Vendor in nuova scheda</a>
        <a href="probe_tool/index.html" target="_blank" rel="noopener">Apri Probe in nuova scheda</a>
      </div>
    </section>

    <section class="frames">
      <iframe id="frame-vendor" class="active" src="vendor_tool/index.html" title="Vendor tool"></iframe>
      <iframe id="frame-probe" src="probe_tool/index.html" title="Probe tool"></iframe>
    </section>
  </div>

  <script>
    const tabVendor = document.getElementById("tab-vendor");
    const tabProbe = document.getElementById("tab-probe");
    const frameVendor = document.getElementById("frame-vendor");
    const frameProbe = document.getElementById("frame-probe");

    function setActive(kind) {{
      const isVendor = kind === "vendor";
      tabVendor.classList.toggle("active", isVendor);
      tabProbe.classList.toggle("active", !isVendor);
      frameVendor.classList.toggle("active", isVendor);
      frameProbe.classList.toggle("active", !isVendor);
    }}

    tabVendor.addEventListener("click", () => setActive("vendor"));
    tabProbe.addEventListener("click", () => setActive("probe"));

    document.addEventListener("keydown", (ev) => {{
      if (ev.key === "1") {{
        ev.preventDefault();
        setActive("vendor");
      }} else if (ev.key === "2") {{
        ev.preventDefault();
        setActive("probe");
      }}
    }});
  </script>
</body>
</html>
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a single structured HTML hub for vendor/probe dataset review tools."
    )
    p.add_argument(
        "--vendor-tool-dir",
        type=Path,
        required=True,
        help="Directory containing vendor rect_correction_tool/index.html",
    )
    p.add_argument(
        "--probe-tool-dir",
        type=Path,
        required=True,
        help="Directory containing probe rect_correction_tool/index.html",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/40_outputs_eval/vendor_probe_dataset_review_hub"),
        help="Output directory for combined hub HTML.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    vendor_tool_dir = args.vendor_tool_dir.expanduser().resolve()
    probe_tool_dir = args.probe_tool_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vendor_index = vendor_tool_dir / "index.html"
    probe_index = probe_tool_dir / "index.html"
    if not vendor_index.exists():
        raise FileNotFoundError(f"Vendor tool index not found: {vendor_index}")
    if not probe_index.exists():
        raise FileNotFoundError(f"Probe tool index not found: {probe_index}")

    _ensure_link(output_dir / "vendor_tool", vendor_tool_dir)
    _ensure_link(output_dir / "probe_tool", probe_tool_dir)
    # Keep absolute image paths reachable from the hub server root.
    for root_name in ("Volumes", "Users"):
        src = Path("/") / root_name
        dst = output_dir / root_name
        try:
            if src.exists() and src.is_dir() and not dst.exists() and not dst.is_symlink():
                dst.symlink_to(src)
        except OSError:
            pass

    vendor_count = _count_folders(vendor_tool_dir)
    probe_count = _count_folders(probe_tool_dir)

    html_path = output_dir / "index.html"
    html_path.write_text(_build_html(vendor_count=vendor_count, probe_count=probe_count), encoding="utf-8")

    summary = {
        "output_dir": output_dir.as_posix(),
        "hub_index_html": html_path.as_posix(),
        "vendor_tool_dir": vendor_tool_dir.as_posix(),
        "probe_tool_dir": probe_tool_dir.as_posix(),
        "vendor_folders": vendor_count,
        "probe_folders": probe_count,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Hub index: {html_path}", flush=True)
    print(f"Vendor folders: {vendor_count}", flush=True)
    print(f"Probe folders: {probe_count}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
