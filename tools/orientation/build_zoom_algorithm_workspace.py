#!/usr/bin/env python3
"""Build an interactive HTML workspace to design a "max zoom" selector per group+label."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError


FEATURE_NAMES = ("nonblack_ratio", "bright_ratio", "bbox_ratio", "low_high_ratio")
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _safe_rel(from_dir: Path, to_path: Path) -> str:
    try:
        return to_path.relative_to(from_dir).as_posix()
    except Exception:  # noqa: BLE001
        return os.path.relpath(to_path.as_posix(), start=from_dir.as_posix())


def _weak_hint_from_name(name: str) -> int:
    n = name.lower()
    if re.search(r"(?:^|[_\-\s])(?:maxzoom|max[_\-]?zoom|zoommax)(?:[_\-\s]|$)", n):
        return 1
    if re.search(r"(?:^|[_\-\s])(?:minzoom|min[_\-]?zoom|zoommin)(?:[_\-\s]|$)", n):
        return -1
    return 0


def _image_features(image_path: Path, threshold: int = 8, bright_thr: int = 70) -> Dict[str, float] | None:
    try:
        arr = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    mask = arr > float(threshold)
    if int(mask.sum()) < 200:
        return None

    nz = arr[mask]
    nonblack_ratio = float(mask.mean())
    bright_ratio = float((nz > float(bright_thr)).mean())

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bbox_ratio = float(((y1 - y0) * (x1 - x0)) / (arr.shape[0] * arr.shape[1]))

    # spectral texture proxy on a downsampled view
    ds = np.array(
        Image.fromarray(arr.astype(np.uint8)).resize((128, 128), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    dmask = ds > float(threshold)
    ds[~dmask] = 0.0
    ds = ds - float(ds.mean())
    f = np.fft.rfft2(ds)
    pw = (f.real * f.real + f.imag * f.imag).astype(np.float64)
    fy = np.fft.fftfreq(128)[:, None]
    fx = np.fft.rfftfreq(128)[None, :]
    fr = np.sqrt(fx * fx + fy * fy)
    low = float(pw[fr < 0.08].sum())
    high = float(pw[(fr >= 0.18) & (fr < 0.45)].sum()) + 1e-9
    low_high_ratio = float(low / high)

    return {
        "nonblack_ratio": nonblack_ratio,
        "bright_ratio": bright_ratio,
        "bbox_ratio": bbox_ratio,
        "low_high_ratio": low_high_ratio,
    }


def _robust_stats(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, (1.4826 * mad if mad > 1e-9 else 1.0)


def _load_manifest_meta(manifest_csv: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with manifest_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sid = str(row.get("sample_id", "")).strip()
            if not sid:
                continue
            setup = Path(str(row.get("setup_folder", "")).strip())
            group_name = setup.name.strip() if setup.name.strip() else "UNKNOWN_GROUP"
            out_name = Path(str(row.get("output_image_path", "")).strip()).name
            out[sid] = {
                "group_name": group_name,
                "output_name": out_name,
                "source_image_path": str(row.get("source_image_path", "")).strip(),
            }
    return out


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _infer_label_from_path(image_path: Path, root_dir: Path) -> str:
    rel = image_path.relative_to(root_dir).as_posix().lower()
    name = image_path.name.lower()
    text = f"{rel} {name}"
    if re.search(r"(?:^|[^a-z0-9])(?:giu|ud|lrud|upside[_\- ]?down|down)(?:[^a-z0-9]|$)", text):
        return "giu"
    if re.search(r"(?:^|[^a-z0-9])(?:su|nf|noflip|no[_\- ]?flip|lr|upright|up)(?:[^a-z0-9]|$)", text):
        return "su"
    return "unknown"


def _payload_from_groups(
    grouped: Dict[str, List[Dict[str, object]]],
    *,
    source_kind: str,
    source_ref: str,
    output_dir: Path,
    min_group_size: int,
    max_groups: int,
    skipped_missing: int,
    skipped_features: int,
) -> Dict[str, object]:
    groups_payload: List[Dict[str, object]] = []
    for key, items in sorted(grouped.items(), key=lambda kv: kv[0].lower()):
        if len(items) < int(min_group_size):
            continue
        label = str(items[0]["label"])
        group_name = str(items[0]["group_name"])
        stats: Dict[str, Dict[str, float]] = {}
        for fname in FEATURE_NAMES:
            values = [float(x["features"][fname]) for x in items]
            med, scale = _robust_stats(values)
            stats[fname] = {"median": med, "scale": scale}
        groups_payload.append(
            {
                "group_key": key,
                "group_name": group_name,
                "label": label,
                "num_images": len(items),
                "stats": stats,
                "images": items,
            }
        )

    if int(max_groups) > 0:
        groups_payload = groups_payload[: int(max_groups)]

    return {
        "meta": {
            "source_kind": source_kind,
            "source_ref": source_ref,
            "output_dir": output_dir.as_posix(),
            "num_groups": len(groups_payload),
            "num_images": int(sum(int(g["num_images"]) for g in groups_payload)),
            "skipped_missing": skipped_missing,
            "skipped_features": skipped_features,
            "feature_names": list(FEATURE_NAMES),
            "defaults": {
                "w_nonblack_ratio": 0.55,
                "w_bright_ratio": 0.30,
                "w_bbox_ratio": 0.10,
                "w_low_high_ratio": 0.05,
                "weak_hint_bonus": 0.35,
            },
        },
        "groups": groups_payload,
    }


def build_workspace(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    skipped_missing = 0
    skipped_features = 0

    if args.input_dir:
        input_dir = args.input_dir.expanduser().resolve()
        if not input_dir.exists() or not input_dir.is_dir():
            raise FileNotFoundError(f"Input dir not found (or not dir): {input_dir}")

        files_iter = input_dir.rglob("*") if bool(args.recursive) else input_dir.glob("*")
        max_images = int(args.max_images) if int(args.max_images) > 0 else None
        seen = 0
        for image_path in files_iter:
            if not _is_image_file(image_path):
                continue
            seen += 1
            if max_images is not None and seen > max_images:
                break

            feats = _image_features(image_path, threshold=int(args.black_threshold), bright_thr=int(args.bright_threshold))
            if feats is None:
                skipped_features += 1
                continue

            if args.group_by == "single":
                group_name = input_dir.name
            elif args.group_by == "leaf":
                group_name = image_path.parent.name or input_dir.name
            else:
                try:
                    rel_parent = image_path.parent.relative_to(input_dir).as_posix()
                except Exception:  # noqa: BLE001
                    rel_parent = image_path.parent.as_posix()
                group_name = rel_parent if rel_parent not in {"", "."} else input_dir.name

            label = _infer_label_from_path(image_path, input_dir) if args.input_label_mode == "auto" else "unknown"
            key = f"{group_name}__{label}"
            sample_id = image_path.relative_to(input_dir).as_posix()
            grouped[key].append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "group_name": group_name,
                    "split": "input",
                    "image_path": image_path.as_posix(),
                    "image_uri": image_path.as_uri(),
                    "image_src_rel": _safe_rel(output_dir, image_path),
                    "output_name": image_path.name,
                    "source_image_path": image_path.as_posix(),
                    "weak_hint": _weak_hint_from_name(image_path.name),
                    "features": feats,
                }
            )

        payload = _payload_from_groups(
            grouped,
            source_kind="input_dir",
            source_ref=input_dir.as_posix(),
            output_dir=output_dir,
            min_group_size=int(args.min_group_size),
            max_groups=int(args.max_groups),
            skipped_missing=skipped_missing,
            skipped_features=skipped_features,
        )
    else:
        manifest_csv = args.manifest.expanduser().resolve()
        crop_manifest_csv = args.crop_manifest.expanduser().resolve()
        if not manifest_csv.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
        if not crop_manifest_csv.exists():
            raise FileNotFoundError(f"Crop manifest not found: {crop_manifest_csv}")

        meta = _load_manifest_meta(manifest_csv)
        with crop_manifest_csv.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                sid = str(row.get("sample_id", "")).strip()
                label = str(row.get("label", "")).strip().lower()
                if not sid or label not in {"su", "giu"}:
                    continue
                if sid not in meta:
                    skipped_missing += 1
                    continue
                image_path = Path(str(row.get("image_path", "")).strip()).expanduser().resolve()
                if not image_path.exists():
                    skipped_missing += 1
                    continue
                feats = _image_features(
                    image_path,
                    threshold=int(args.black_threshold),
                    bright_thr=int(args.bright_threshold),
                )
                if feats is None:
                    skipped_features += 1
                    continue

                m = meta[sid]
                group_name = m["group_name"]
                key = f"{group_name}__{label}"
                grouped[key].append(
                    {
                        "sample_id": sid,
                        "label": label,
                        "group_name": group_name,
                        "split": str(row.get("split", "")).strip().lower(),
                        "image_path": image_path.as_posix(),
                        "image_uri": image_path.as_uri(),
                        "image_src_rel": _safe_rel(output_dir, image_path),
                        "output_name": m["output_name"],
                        "source_image_path": m["source_image_path"],
                        "weak_hint": _weak_hint_from_name(m["output_name"]),
                        "features": feats,
                    }
                )

        payload = _payload_from_groups(
            grouped,
            source_kind="manifest_pair",
            source_ref=f"manifest={manifest_csv.as_posix()} | crop_manifest={crop_manifest_csv.as_posix()}",
            output_dir=output_dir,
            min_group_size=int(args.min_group_size),
            max_groups=int(args.max_groups),
            skipped_missing=skipped_missing,
            skipped_features=skipped_features,
        )

    html_out = output_dir / "index.html"
    json_out = output_dir / "workspace_payload.json"
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_out.write_text(_html_with_payload(payload), encoding="utf-8")

    print(f"Workspace HTML: {html_out}", flush=True)
    print(f"Workspace payload: {json_out}", flush=True)
    print(f"Source: {payload['meta'].get('source_kind', '-')}", flush=True)
    print(f"Groups: {payload['meta']['num_groups']} | Images: {payload['meta']['num_images']}", flush=True)
    return 0


def _html_with_payload(payload: Dict[str, object]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Zoom Algorithm Workspace</title>
  <style>
    :root {{
      --bg: #0f1722;
      --panel: #172333;
      --panel2: #121c2a;
      --border: #2b3a4f;
      --text: #e7edf5;
      --muted: #9fb0c3;
      --accent: #2563eb;
      --ok: #22c55e;
      --warn: #f59e0b;
      --bad: #ef4444;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; background: var(--bg); color: var(--text); }}
    .wrap {{ display: grid; grid-template-columns: 360px 1fr; min-height: 100vh; }}
    .left {{ border-right: 1px solid var(--border); background: var(--panel2); padding: 12px; overflow: auto; }}
    .right {{ padding: 14px; overflow: auto; }}
    h1 {{ margin: 0 0 6px; font-size: 20px; }}
    h2 {{ margin: 0 0 8px; font-size: 16px; }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    .search {{
      width: 100%;
      margin-top: 10px;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #0d1520;
      color: var(--text);
    }}
    .group-list {{ margin-top: 10px; display: grid; gap: 8px; }}
    .group-btn {{
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      text-align: left;
      padding: 8px;
      border-radius: 8px;
      cursor: pointer;
    }}
    .group-btn.active {{ border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }}
    .chips {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }}
    .chip {{ border: 1px solid var(--border); background: var(--panel2); padding: 4px 8px; border-radius: 999px; font-size: 12px; color: #c7d4e4; }}
    .panel {{
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--panel);
      padding: 10px;
      margin-bottom: 10px;
    }}
    .weights {{ display: grid; grid-template-columns: repeat(2, minmax(240px, 1fr)); gap: 8px; }}
    .weight-row {{ font-size: 12px; color: #c7d4e4; }}
    .weight-row input[type="range"] {{ width: 100%; }}
    .flex {{ display: flex; gap: 10px; align-items: flex-start; flex-wrap: wrap; }}
    .preview {{ width: min(560px, 100%); border: 1px solid var(--border); border-radius: 8px; background: #000; }}
    .preview img {{ width: 100%; display: block; }}
    .evidence {{ flex: 1; min-width: 280px; }}
    .evidence pre {{
      margin: 0;
      white-space: pre-wrap;
      background: #0f1722;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px;
      color: #d6e3f0;
      font-size: 12px;
    }}
    .controls {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
    .btn {{
      border: 1px solid var(--border);
      background: var(--panel2);
      color: var(--text);
      border-radius: 8px;
      padding: 7px 10px;
      cursor: pointer;
    }}
    .btn.primary {{ background: var(--accent); border-color: var(--accent); }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--border); border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #223043; padding: 6px 8px; text-align: left; }}
    th {{ position: sticky; top: 0; background: #152334; z-index: 1; }}
    tr.sel {{ background: #1f2f45; }}
    tr.err {{ background: #3a1e24; }}
    .tiny {{ width: 74px; height: 52px; object-fit: cover; border-radius: 4px; border: 1px solid #2b3a4f; background: #000; }}
    textarea, select {{
      width: 100%;
      background: #0d1520;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px;
      font-size: 12px;
    }}
    .ok {{ color: #86efac; }}
    .warn {{ color: #facc15; }}
    .bad {{ color: #fca5a5; }}
    @media (max-width: 1100px) {{
      .wrap {{ grid-template-columns: 1fr; }}
      .left {{ border-right: 0; border-bottom: 1px solid var(--border); max-height: 40vh; }}
      .weights {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <aside class="left">
      <h1>Workspace Zoom</h1>
      <div class="muted">Ragionamento guidato con evidenze + annotazioni per gruppo/classe.</div>
      <div class="chips" id="global-chips"></div>
      <input id="group-search" class="search" placeholder="Cerca gruppo o label...">
      <div class="group-list" id="group-list"></div>
    </aside>

    <main class="right">
      <section class="panel">
        <h2 id="group-title">Nessun gruppo selezionato</h2>
        <div class="chips" id="group-chips"></div>
        <div class="weights">
          <div class="weight-row">w nonblack <span id="v-w-nonblack_ratio"></span><input type="range" id="w-nonblack_ratio" min="-1.5" max="2.0" step="0.01"></div>
          <div class="weight-row">w bright <span id="v-w-bright_ratio"></span><input type="range" id="w-bright_ratio" min="-1.5" max="2.0" step="0.01"></div>
          <div class="weight-row">w bbox <span id="v-w-bbox_ratio"></span><input type="range" id="w-bbox_ratio" min="-1.5" max="2.0" step="0.01"></div>
          <div class="weight-row">w low/high <span id="v-w-low_high_ratio"></span><input type="range" id="w-low_high_ratio" min="-1.5" max="2.0" step="0.01"></div>
          <div class="weight-row">bonus weak hint <span id="v-weak_hint_bonus"></span><input type="range" id="weak_hint_bonus" min="0" max="1.5" step="0.01"></div>
        </div>
        <div class="controls">
          <button class="btn" id="btn-default-weights">Default pesi</button>
          <button class="btn primary" id="btn-select-top1">Seleziona top-1</button>
          <button class="btn" id="btn-export-json">Export annotazioni JSON</button>
          <button class="btn" id="btn-export-csv">Export annotazioni CSV</button>
          <span class="muted" id="save-state"></span>
        </div>
      </section>

      <section class="panel">
        <div class="flex">
          <div class="preview"><img id="big-preview" alt="preview"></div>
          <div class="evidence">
            <h2>Evidenze</h2>
            <pre id="evidence-text"></pre>
            <div class="muted" id="path-text" style="margin-top:8px;"></div>
          </div>
        </div>
      </section>

      <section class="panel">
        <h2>Annotazioni gruppo</h2>
        <div style="display:grid;grid-template-columns:220px 1fr;gap:8px;align-items:start;">
          <div>
            <label class="muted">Stato review</label>
            <select id="group-status">
              <option value="pending">pending</option>
              <option value="confirmed">confirmed</option>
              <option value="needs_review">needs_review</option>
            </select>
          </div>
          <div>
            <label class="muted">Commento gruppo</label>
            <textarea id="group-comment" rows="3" placeholder="Note qualitative sul comportamento dello score..."></textarea>
          </div>
        </div>
      </section>

      <section class="panel">
        <h2>Ranking immagini</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>pick</th>
                <th>img</th>
                <th>sample_id</th>
                <th>score</th>
                <th>weak</th>
                <th>nonblack</th>
                <th>bright</th>
                <th>bbox</th>
                <th>low/high</th>
                <th>note</th>
              </tr>
            </thead>
            <tbody id="rows-body"></tbody>
          </table>
        </div>
      </section>
    </main>
  </div>

  <script>
    const DATA = {data_json};
    const STORE_KEY = 'zoom-workspace-annotations-v1';
    const DEFAULTS = DATA.meta.defaults;

    let state = loadState();
    let currentGroupKey = null;
    let currentRows = [];

    function loadState() {{
      try {{
        const raw = localStorage.getItem(STORE_KEY);
        if (!raw) return {{ groups: {{}}, weights: {{...DEFAULTS}} }};
        const parsed = JSON.parse(raw);
        if (!parsed.groups) parsed.groups = {{}};
        if (!parsed.weights) parsed.weights = {{...DEFAULTS}};
        return parsed;
      }} catch (_e) {{
        return {{ groups: {{}}, weights: {{...DEFAULTS}} }};
      }}
    }}

    function saveState(msg='saved') {{
      localStorage.setItem(STORE_KEY, JSON.stringify(state));
      document.getElementById('save-state').textContent = `autosave: ${{msg}}`;
    }}

    function esc(v) {{
      return String(v)
        .replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')
        .replaceAll('"','&quot;').replaceAll(\"'\",'&#39;');
    }}

    function groupRecord(groupKey) {{
      if (!state.groups[groupKey]) {{
        state.groups[groupKey] = {{
          selected_sample_id: '',
          status: 'pending',
          comment: '',
          image_notes: {{}},
        }};
      }}
      return state.groups[groupKey];
    }}

    function zFor(image, gstats, featureName) {{
      const med = Number(gstats[featureName].median || 0);
      const scale = Number(gstats[featureName].scale || 1);
      const x = Number(image.features[featureName] || 0);
      return (x - med) / (scale + 1e-9);
    }}

    function scoreImage(image, gstats) {{
      const w = state.weights;
      const z_non = zFor(image, gstats, 'nonblack_ratio');
      const z_bri = zFor(image, gstats, 'bright_ratio');
      const z_bbox = zFor(image, gstats, 'bbox_ratio');
      const z_lh = zFor(image, gstats, 'low_high_ratio');
      const hint = Number(image.weak_hint || 0) * Number(w.weak_hint_bonus || 0);
      const score = (
        Number(w.w_nonblack_ratio) * z_non +
        Number(w.w_bright_ratio) * z_bri +
        Number(w.w_bbox_ratio) * z_bbox +
        Number(w.w_low_high_ratio) * z_lh +
        hint
      );
      return {{ score, z_non, z_bri, z_bbox, z_lh, hint }};
    }}

    function attachImgFallback(root) {{
      root.querySelectorAll('img[data-fallback-src]').forEach((img) => {{
        img.addEventListener('error', () => {{
          if (img.dataset.fallbackUsed === '1') return;
          const fb = img.getAttribute('data-fallback-src') || '';
          if (!fb) return;
          img.dataset.fallbackUsed = '1';
          img.setAttribute('src', fb);
        }});
      }});
    }}

    function renderGlobalChips() {{
      const chips = [
        `gruppi: ${{DATA.meta.num_groups}}`,
        `immagini: ${{DATA.meta.num_images}}`,
        `skip missing: ${{DATA.meta.skipped_missing}}`,
        `skip features: ${{DATA.meta.skipped_features}}`,
      ];
      document.getElementById('global-chips').innerHTML = chips.map((c) => `<span class="chip">${{esc(c)}}</span>`).join('');
    }}

    function renderGroupList() {{
      const q = String(document.getElementById('group-search').value || '').trim().toLowerCase();
      const list = document.getElementById('group-list');
      const html = [];
      for (const g of DATA.groups) {{
        const rec = groupRecord(g.group_key);
        const blob = `${{g.group_name}} ${{g.label}}`.toLowerCase();
        if (q && !blob.includes(q)) continue;
        const done = rec.status === 'confirmed';
        html.push(`
          <button class="group-btn ${{currentGroupKey===g.group_key?'active':''}}" data-k="${{esc(g.group_key)}}">
            <div><b>${{esc(g.group_name)}}</b> <span class="muted">[${{esc(g.label)}}]</span></div>
            <div class="muted">n=${{g.num_images}} | status: <span class="${{done?'ok':'warn'}}">${{esc(rec.status)}}</span></div>
          </button>
        `);
      }}
      list.innerHTML = html.join('');
      list.querySelectorAll('.group-btn').forEach((el) => {{
        el.addEventListener('click', () => {{
          currentGroupKey = el.getAttribute('data-k');
          renderCurrentGroup();
          renderGroupList();
        }});
      }});
    }}

    function currentGroup() {{
      return DATA.groups.find((g) => g.group_key === currentGroupKey) || null;
    }}

    function setWeightInputs() {{
      const ids = ['w_nonblack_ratio','w_bright_ratio','w_bbox_ratio','w_low_high_ratio','weak_hint_bonus'];
      for (const id of ids) {{
        const el = document.getElementById(id);
        el.value = String(state.weights[id]);
        const v = document.getElementById(`v-${{id}}`);
        if (v) v.textContent = Number(state.weights[id]).toFixed(2);
      }}
    }}

    function renderEvidence(group, row) {{
      if (!row) {{
        document.getElementById('evidence-text').textContent = 'Seleziona una riga.';
        document.getElementById('path-text').textContent = '';
        document.getElementById('big-preview').removeAttribute('src');
        return;
      }}
      const rec = row.__score;
      const lines = [];
      lines.push(`sample_id: ${{row.sample_id}}`);
      lines.push(`score totale: ${{rec.score.toFixed(4)}}`);
      lines.push('');
      lines.push(`contributi:`)
      lines.push(`- nonblack: z=${{rec.z_non.toFixed(3)}} * w=${{Number(state.weights.w_nonblack_ratio).toFixed(2)}}`);
      lines.push(`- bright:   z=${{rec.z_bri.toFixed(3)}} * w=${{Number(state.weights.w_bright_ratio).toFixed(2)}}`);
      lines.push(`- bbox:     z=${{rec.z_bbox.toFixed(3)}} * w=${{Number(state.weights.w_bbox_ratio).toFixed(2)}}`);
      lines.push(`- low/high: z=${{rec.z_lh.toFixed(3)}} * w=${{Number(state.weights.w_low_high_ratio).toFixed(2)}}`);
      lines.push(`- weak hint: ${{Number(row.weak_hint)}} * bonus=${{Number(state.weights.weak_hint_bonus).toFixed(2)}} => ${{rec.hint.toFixed(3)}}`);
      lines.push('');
      lines.push('feature raw:');
      lines.push(`- nonblack_ratio = ${{Number(row.features.nonblack_ratio).toFixed(4)}}`);
      lines.push(`- bright_ratio   = ${{Number(row.features.bright_ratio).toFixed(4)}}`);
      lines.push(`- bbox_ratio     = ${{Number(row.features.bbox_ratio).toFixed(4)}}`);
      lines.push(`- low_high_ratio = ${{Number(row.features.low_high_ratio).toFixed(4)}}`);
      document.getElementById('evidence-text').textContent = lines.join('\\n');
      const img = document.getElementById('big-preview');
      img.setAttribute('src', row.image_src_rel);
      img.setAttribute('data-fallback-src', row.image_uri);
      attachImgFallback(document.body);
      document.getElementById('path-text').innerHTML = `file: <code>${{esc(row.output_name)}}</code><br>source: <code>${{esc(row.source_image_path || '')}}</code>`;
    }}

    function renderCurrentGroup() {{
      const g = currentGroup();
      if (!g) return;
      const ann = groupRecord(g.group_key);
      document.getElementById('group-title').textContent = `${{g.group_name}} [${{g.label}}]`;
      document.getElementById('group-status').value = ann.status || 'pending';
      document.getElementById('group-comment').value = ann.comment || '';

      const weakMax = g.images.filter((x) => Number(x.weak_hint) > 0).length;
      const weakMin = g.images.filter((x) => Number(x.weak_hint) < 0).length;
      document.getElementById('group-chips').innerHTML = [
        `immagini: ${{g.num_images}}`,
        `weak max hint: ${{weakMax}}`,
        `weak min hint: ${{weakMin}}`,
        `selected: ${{ann.selected_sample_id || '-'}}`,
      ].map((c) => `<span class="chip">${{esc(c)}}</span>`).join('');

      const scored = g.images.map((im) => {{
        const s = scoreImage(im, g.stats);
        return {{ ...im, __score: s }};
      }}).sort((a,b) => Number(b.__score.score) - Number(a.__score.score));
      currentRows = scored;

      if (!ann.selected_sample_id && scored.length) {{
        ann.selected_sample_id = scored[0].sample_id;
      }}

      const tb = document.getElementById('rows-body');
      tb.innerHTML = scored.map((r, idx) => {{
        const sel = ann.selected_sample_id === r.sample_id;
        const note = ann.image_notes?.[r.sample_id] || '';
        const weak = Number(r.weak_hint) > 0 ? 'max' : (Number(r.weak_hint) < 0 ? 'min' : '-');
        const cls = sel ? 'sel' : '';
        return `
          <tr class="${{cls}}" data-sid="${{esc(r.sample_id)}}">
            <td>${{idx+1}}</td>
            <td><input type="radio" name="pick" data-sid="${{esc(r.sample_id)}}" ${{sel?'checked':''}}></td>
            <td><img class="tiny" src="${{esc(r.image_src_rel)}}" data-fallback-src="${{esc(r.image_uri)}}" alt="img"></td>
            <td><code>${{esc(r.sample_id)}}</code></td>
            <td>${{Number(r.__score.score).toFixed(4)}}</td>
            <td>${{esc(weak)}}</td>
            <td>${{Number(r.features.nonblack_ratio).toFixed(4)}}</td>
            <td>${{Number(r.features.bright_ratio).toFixed(4)}}</td>
            <td>${{Number(r.features.bbox_ratio).toFixed(4)}}</td>
            <td>${{Number(r.features.low_high_ratio).toFixed(4)}}</td>
            <td><input type="text" data-note-sid="${{esc(r.sample_id)}}" value="${{esc(note)}}" placeholder="nota" style="width:220px;background:#0d1520;color:#e7edf5;border:1px solid #2b3a4f;border-radius:6px;padding:4px 6px;"></td>
          </tr>
        `;
      }}).join('');
      attachImgFallback(tb);

      tb.querySelectorAll('input[type="radio"][name="pick"]').forEach((el) => {{
        el.addEventListener('change', () => {{
          ann.selected_sample_id = el.getAttribute('data-sid') || '';
          saveState('pick updated');
          renderCurrentGroup();
          renderGroupList();
        }});
      }});

      tb.querySelectorAll('input[data-note-sid]').forEach((el) => {{
        el.addEventListener('input', () => {{
          const sid = el.getAttribute('data-note-sid');
          if (!ann.image_notes) ann.image_notes = {{}};
          ann.image_notes[sid] = el.value || '';
          saveState('note updated');
        }});
      }});

      tb.querySelectorAll('tr').forEach((tr) => {{
        tr.addEventListener('click', (evt) => {{
          if (evt.target && (evt.target.tagName === 'INPUT')) return;
          const sid = tr.getAttribute('data-sid');
          const row = currentRows.find((x) => x.sample_id === sid);
          renderEvidence(g, row || null);
          tb.querySelectorAll('tr').forEach((t) => t.classList.remove('sel'));
          tr.classList.add('sel');
        }});
      }});

      renderEvidence(g, scored[0] || null);
      saveState('group rendered');
    }}

    function exportJson() {{
      const blob = new Blob([JSON.stringify(state, null, 2)], {{type:'application/json;charset=utf-8'}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'zoom_workspace_annotations.json';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}

    function csvEsc(v) {{
      return `"${{String(v ?? '').replaceAll('"', '""')}}"`;
    }}

    function exportCsv() {{
      const rows = [[
        'group_key','group_name','label','selected_sample_id','status','group_comment','sample_id','image_note'
      ]];
      for (const g of DATA.groups) {{
        const ann = groupRecord(g.group_key);
        const notes = ann.image_notes || {{}};
        const keys = Object.keys(notes);
        if (keys.length === 0) {{
          rows.push([
            g.group_key, g.group_name, g.label, ann.selected_sample_id || '', ann.status || '', ann.comment || '', '', ''
          ]);
          continue;
        }}
        for (const sid of keys) {{
          rows.push([
            g.group_key, g.group_name, g.label, ann.selected_sample_id || '', ann.status || '', ann.comment || '', sid, notes[sid] || ''
          ]);
        }}
      }}
      const csv = rows.map((r) => r.map(csvEsc).join(',')).join('\\n');
      const blob = new Blob([csv], {{type:'text/csv;charset=utf-8'}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'zoom_workspace_annotations.csv';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}

    function wireWeights() {{
      const ids = ['w_nonblack_ratio','w_bright_ratio','w_bbox_ratio','w_low_high_ratio','weak_hint_bonus'];
      for (const id of ids) {{
        const el = document.getElementById(id);
        el.addEventListener('input', () => {{
          state.weights[id] = Number(el.value);
          const v = document.getElementById(`v-${{id}}`);
          if (v) v.textContent = Number(state.weights[id]).toFixed(2);
          saveState('weights updated');
          if (currentGroupKey) renderCurrentGroup();
        }});
      }}
      document.getElementById('btn-default-weights').addEventListener('click', () => {{
        state.weights = {{...DEFAULTS}};
        setWeightInputs();
        saveState('weights default');
        if (currentGroupKey) renderCurrentGroup();
      }});
    }}

    function boot() {{
      renderGlobalChips();
      setWeightInputs();
      wireWeights();

      document.getElementById('group-search').addEventListener('input', renderGroupList);
      document.getElementById('group-status').addEventListener('change', (e) => {{
        const g = currentGroup(); if (!g) return;
        const ann = groupRecord(g.group_key);
        ann.status = e.target.value || 'pending';
        saveState('status updated');
        renderGroupList();
      }});
      document.getElementById('group-comment').addEventListener('input', (e) => {{
        const g = currentGroup(); if (!g) return;
        const ann = groupRecord(g.group_key);
        ann.comment = e.target.value || '';
        saveState('group comment updated');
      }});
      document.getElementById('btn-select-top1').addEventListener('click', () => {{
        const g = currentGroup(); if (!g || !currentRows.length) return;
        const ann = groupRecord(g.group_key);
        ann.selected_sample_id = currentRows[0].sample_id;
        saveState('top1 selected');
        renderCurrentGroup();
        renderGroupList();
      }});
      document.getElementById('btn-export-json').addEventListener('click', exportJson);
      document.getElementById('btn-export-csv').addEventListener('click', exportCsv);

      if (DATA.groups.length > 0) {{
        currentGroupKey = DATA.groups[0].group_key;
      }}
      renderGroupList();
      if (currentGroupKey) renderCurrentGroup();
    }}

    boot();
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build interactive workspace for max-zoom algorithm design.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional: scan this folder directly (recursive by default) instead of using manifest + crop-manifest.",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=("relative_parent", "leaf", "single"),
        default="relative_parent",
        help="When using --input-dir: group by relative parent path, leaf folder name, or a single global group.",
    )
    parser.add_argument(
        "--input-label-mode",
        type=str,
        choices=("auto", "unknown"),
        default="auto",
        help="When using --input-dir: infer su/giu from names/paths, or force unknown.",
    )
    parser.add_argument("--recursive", action="store_true", default=True, help="When using --input-dir: scan recursively.")
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="When using --input-dir: scan only top-level files in the input folder.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="When using --input-dir: optional cap, 0 = all.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/manifest_exclusions_applied.csv"),
    )
    parser.add_argument(
        "--crop-manifest",
        type=Path,
        default=Path("artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/cropped_dataset/crop_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/zoom_workspace"),
    )
    parser.add_argument("--min-group-size", type=int, default=3)
    parser.add_argument("--max-groups", type=int, default=0, help="0 = all groups")
    parser.add_argument("--black-threshold", type=int, default=8)
    parser.add_argument("--bright-threshold", type=int, default=70)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return build_workspace(args)


if __name__ == "__main__":
    raise SystemExit(main())
