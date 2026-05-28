#!/usr/bin/env python3
"""Build interactive HTML to confirm real exclusions from transrectal quality suspects."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Dict, List


def _safe_text(value: str, max_len: int = 260) -> str:
    out = (value or "").strip()
    if len(out) <= max_len:
        return out
    return out[: max_len - 1] + "..."


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _load_rows(suspects_csv: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with suspects_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            image_path = (row.get("image_path") or "").strip()
            if not image_path:
                continue
            score = _safe_int(row.get("suspect_score"), 0)
            rows.append(
                {
                    "image_path": image_path,
                    "split": (row.get("split") or "").strip().lower(),
                    "label_lt": (row.get("label_lt") or "").strip().upper(),
                    "manufacturer": _safe_text(row.get("manufacturer") or "UNKNOWN", 120),
                    "model_name": _safe_text(row.get("model_name") or "", 180),
                    "fss_id_probe": (row.get("fss_id_probe") or "").strip(),
                    "suspect_score": str(score),
                    "suspect_reason": _safe_text(row.get("suspect_reason") or "", 220),
                    "crop_source": (row.get("crop_source") or "").strip(),
                    "pix_std": (row.get("pix_std") or "").strip(),
                    "pix_entropy": (row.get("pix_entropy") or "").strip(),
                    "pix_black_ratio": (row.get("pix_black_ratio") or "").strip(),
                }
            )
    rows.sort(key=lambda r: int(r.get("suspect_score", "0")), reverse=True)
    return rows


def _build_html(rows: List[Dict[str, str]], recommend_score: int) -> str:
    payload = json.dumps(rows, ensure_ascii=False)
    style = """
:root {
  --bg: #f4f6fb;
  --panel: #ffffff;
  --line: #d4dced;
  --txt: #16253d;
  --muted: #5d6c87;
  --warn: #d97706;
  --danger: #b91c1c;
  --ok: #166534;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", Tahoma, sans-serif;
  color: var(--txt);
  background: radial-gradient(circle at 8% 0%, #e7edf9, #f4f6fb 40%);
}
.wrap { max-width: 1900px; margin: 0 auto; padding: 14px; display: grid; gap: 10px; }
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px;
}
.row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.grow { flex: 1 1 auto; }
.small { font-size: 12px; color: var(--muted); }
.input, .select, .textarea {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  color: var(--txt);
  padding: 6px 8px;
  font-size: 13px;
}
.textarea {
  width: 100%;
  min-height: 56px;
  resize: vertical;
}
.button {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  color: var(--txt);
  padding: 6px 9px;
  font-size: 12px;
  cursor: pointer;
}
.button:hover { background: #eef3ff; }
.button-danger { border-color: #f2b5b5; color: var(--danger); }
.button-ok { border-color: #b8e6c6; color: var(--ok); }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 10px;
}
.card {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 8px;
}
.card-marked {
  border-color: #d9534f;
  box-shadow: 0 0 0 1px #d9534f inset;
}
.thumb {
  width: 100%;
  height: 210px;
  object-fit: contain;
  background: #000;
  border-radius: 8px;
}
.pill {
  display: inline-block;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 700;
}
.pill-l { background: #ddf6e8; color: #0f7a42; }
.pill-t { background: #ffe8ce; color: #9b5600; }
.pill-rec { background: #ffefda; color: var(--warn); }
.pill-exc { background: #ffe3e3; color: var(--danger); }
.meta { margin-top: 6px; font-size: 12px; line-height: 1.35; }
.path {
  margin-top: 6px;
  font-size: 10px;
  color: #495a78;
  word-break: break-all;
}
"""

    return f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Confirm Exclusions - Transrectal L/T</title>
  <style>{style}</style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <div class="row">
        <div class="grow">
          <div><b>Conferma esclusioni reali (immagini sospette)</b></div>
          <div class="small">Preselezione suggerita: score >= {recommend_score}</div>
        </div>
        <div class="small">rows loaded: <b id="rowsLoaded">{len(rows)}</b></div>
        <div class="small">rows visible: <b id="rowsVisible">0</b></div>
        <div class="small">exclude real: <b id="rowsExcluded">0</b></div>
      </div>
      <div class="row" style="margin-top:8px;">
        <label class="small">Azione</label>
        <select id="actionFilter" class="select">
          <option value="ALL">All</option>
          <option value="EXCLUDE">Solo marcate exclude</option>
          <option value="KEEP">Solo non escluse</option>
          <option value="RECOMMENDED">Solo recommended</option>
        </select>
        <label class="small">Label</label>
        <select id="labelFilter" class="select">
          <option value="ALL">All</option>
          <option value="L">L</option>
          <option value="T">T</option>
        </select>
        <label class="small">Vendor</label>
        <select id="vendorFilter" class="select"><option value="ALL">All</option></select>
        <label class="small">Score >=</label>
        <input id="scoreMinFilter" class="input" value="0" style="width:90px;" />
        <label class="small">Cerca</label>
        <input id="searchFilter" class="input" placeholder="reason/path/model/probe..." style="min-width:220px;" />
      </div>
      <div class="row" style="margin-top:8px;">
        <button id="markVisibleExcludeBtn" class="button button-danger" type="button">Segna visibili EXCLUDE</button>
        <button id="markVisibleKeepBtn" class="button button-ok" type="button">Segna visibili KEEP</button>
        <button id="exportExcludeBtn" class="button button-danger" type="button">Export solo EXCLUDE</button>
        <button id="exportAllBtn" class="button" type="button">Export tutte decisioni</button>
        <button id="resetBtn" class="button" type="button">Reset</button>
      </div>
    </div>
    <div id="grid" class="grid"></div>
  </div>
  <script>
    const DATA = {payload};
    const RECOMMEND_SCORE = {recommend_score};
    const STORAGE_KEY = "transrectal_quality_real_exclusions_v1";

    const grid = document.getElementById("grid");
    const rowsVisible = document.getElementById("rowsVisible");
    const rowsExcluded = document.getElementById("rowsExcluded");
    const actionFilter = document.getElementById("actionFilter");
    const labelFilter = document.getElementById("labelFilter");
    const vendorFilter = document.getElementById("vendorFilter");
    const scoreMinFilter = document.getElementById("scoreMinFilter");
    const searchFilter = document.getElementById("searchFilter");
    const markVisibleExcludeBtn = document.getElementById("markVisibleExcludeBtn");
    const markVisibleKeepBtn = document.getElementById("markVisibleKeepBtn");
    const exportExcludeBtn = document.getElementById("exportExcludeBtn");
    const exportAllBtn = document.getElementById("exportAllBtn");
    const resetBtn = document.getElementById("resetBtn");

    let decisions = {{}};
    try {{
      decisions = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");
    }} catch (_) {{
      decisions = {{}};
    }}

    function esc(text) {{
      return String(text || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    function rowKey(row) {{
      return `${{row.image_path}}|${{row.label_lt}}|${{row.split}}`;
    }}

    function normalizeDecision(raw, row) {{
      const recommended = Number(row.suspect_score || 0) >= RECOMMEND_SCORE ? 1 : 0;
      if (!raw) {{
        return {{
          exclude_real: recommended,
          note: "",
          source: "default_recommended",
        }};
      }}
      if (typeof raw === "string") {{
        return {{
          exclude_real: raw === "1" ? 1 : 0,
          note: "",
          source: "legacy_string",
        }};
      }}
      return {{
        exclude_real: Number(raw.exclude_real || 0) === 1 ? 1 : 0,
        note: String(raw.note || ""),
        source: "user_decision",
      }};
    }}

    function getDecision(row) {{
      return normalizeDecision(decisions[rowKey(row)], row);
    }}

    function isRecommended(row) {{
      return Number(row.suspect_score || 0) >= RECOMMEND_SCORE;
    }}

    function persistDecision(row, next) {{
      const key = rowKey(row);
      decisions[key] = {{
        exclude_real: Number(next.exclude_real || 0) === 1 ? 1 : 0,
        note: String(next.note || ""),
      }};
      localStorage.setItem(STORAGE_KEY, JSON.stringify(decisions));
    }}

    function excludedCount() {{
      let n = 0;
      for (const row of DATA) {{
        if (getDecision(row).exclude_real === 1) n += 1;
      }}
      rowsExcluded.textContent = String(n);
    }}

    function refreshVendorOptions() {{
      const vendors = Array.from(new Set(DATA.map(r => String(r.manufacturer || "UNKNOWN")))).sort();
      const current = vendorFilter.value || "ALL";
      vendorFilter.innerHTML = "";
      const first = document.createElement("option");
      first.value = "ALL";
      first.textContent = "All";
      vendorFilter.appendChild(first);
      for (const v of vendors) {{
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        vendorFilter.appendChild(opt);
      }}
      if ([...vendorFilter.options].some(o => o.value === current)) {{
        vendorFilter.value = current;
      }} else {{
        vendorFilter.value = "ALL";
      }}
    }}

    function rowMatches(row) {{
      const dec = getDecision(row);
      const af = actionFilter.value || "ALL";
      if (af === "EXCLUDE" && dec.exclude_real !== 1) return false;
      if (af === "KEEP" && dec.exclude_real === 1) return false;
      if (af === "RECOMMENDED" && !isRecommended(row)) return false;

      if (labelFilter.value !== "ALL" && row.label_lt !== labelFilter.value) return false;
      if (vendorFilter.value !== "ALL" && row.manufacturer !== vendorFilter.value) return false;

      const scoreMin = Number(scoreMinFilter.value || "0");
      if (!Number.isNaN(scoreMin) && Number(row.suspect_score || 0) < scoreMin) return false;

      const q = String(searchFilter.value || "").trim().toLowerCase();
      if (!q) return true;
      const hay = `${{row.suspect_reason}} ${{row.image_path}} ${{row.model_name}} ${{row.fss_id_probe}} ${{row.manufacturer}}`.toLowerCase();
      return hay.includes(q);
    }}

    function render() {{
      const visible = DATA.filter(rowMatches);
      rowsVisible.textContent = String(visible.length);
      const cards = visible.map(row => {{
        const key = rowKey(row);
        const dec = getDecision(row);
        const excluded = dec.exclude_real === 1;
        const recommended = isRecommended(row);
        const img = row.image_path.startsWith("file://") ? row.image_path : `file://${{row.image_path}}`;
        const pillClass = row.label_lt === "L" ? "pill pill-l" : "pill pill-t";
        return `
          <div class="card ${{excluded ? "card-marked" : ""}}">
            <img class="thumb" src="${{esc(img)}}" loading="lazy" alt="suspect" />
            <div class="meta"><span class="${{pillClass}}">${{esc(row.label_lt)}}</span> ${{recommended ? '<span class="pill pill-rec">RECOMMENDED</span>' : ''}} ${{excluded ? '<span class="pill pill-exc">EXCLUDE</span>' : ''}}</div>
            <div class="meta">score=${{esc(row.suspect_score)}} | reason=${{esc(row.suspect_reason)}}</div>
            <div class="meta">vendor=${{esc(row.manufacturer)}} | probe=${{esc(row.fss_id_probe)}} | split=${{esc(row.split)}}</div>
            <div class="meta">std=${{esc(row.pix_std || "-")}} | entropy=${{esc(row.pix_entropy || "-")}} | black_ratio=${{esc(row.pix_black_ratio || "-")}}</div>
            <div class="path">${{esc(row.image_path)}}</div>
            <div class="row" style="margin-top:6px;">
              <label class="small"><input type="checkbox" class="exclude-check" data-key="${{esc(key)}}" ${{excluded ? "checked" : ""}} /> Escludi davvero</label>
            </div>
            <div style="margin-top:6px;">
              <textarea class="textarea note-input" data-key="${{esc(key)}}" placeholder="Note...">${{esc(dec.note || "")}}</textarea>
            </div>
          </div>
        `;
      }}).join("");
      grid.innerHTML = cards;

      for (const ch of grid.querySelectorAll(".exclude-check")) {{
        const key = ch.getAttribute("data-key") || "";
        const row = DATA.find(r => rowKey(r) === key);
        if (!row) continue;
        ch.addEventListener("change", () => {{
          const prev = getDecision(row);
          persistDecision(row, {{
            exclude_real: ch.checked ? 1 : 0,
            note: prev.note || "",
          }});
          excludedCount();
          render();
        }});
      }}

      for (const ta of grid.querySelectorAll(".note-input")) {{
        const key = ta.getAttribute("data-key") || "";
        const row = DATA.find(r => rowKey(r) === key);
        if (!row) continue;
        ta.addEventListener("change", () => {{
          const prev = getDecision(row);
          persistDecision(row, {{
            exclude_real: prev.exclude_real,
            note: ta.value || "",
          }});
          excludedCount();
        }});
      }}
      excludedCount();
    }}

    function toCsvCell(value) {{
      const s = String(value ?? "");
      if (s.includes(",") || s.includes('"') || s.includes("\\n")) {{
        return `"${{s.replaceAll('"', '""')}}"`;
      }}
      return s;
    }}

    function exportCsv(onlyExcluded) {{
      const header = [
        "row_key",
        "image_path",
        "split",
        "label_lt",
        "manufacturer",
        "fss_id_probe",
        "suspect_score",
        "suspect_reason",
        "recommended_exclude",
        "exclude_real",
        "note",
        "exported_at",
      ];
      const lines = [header];
      for (const row of DATA) {{
        const key = rowKey(row);
        const dec = getDecision(row);
        if (onlyExcluded && dec.exclude_real !== 1) continue;
        lines.push([
          key,
          row.image_path || "",
          row.split || "",
          row.label_lt || "",
          row.manufacturer || "",
          row.fss_id_probe || "",
          row.suspect_score || "",
          row.suspect_reason || "",
          isRecommended(row) ? "1" : "0",
          dec.exclude_real === 1 ? "1" : "0",
          dec.note || "",
          new Date().toISOString(),
        ]);
      }}
      const csv = lines.map(r => r.map(toCsvCell).join(",")).join("\\n");
      const blob = new Blob([csv], {{ type: "text/csv;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = onlyExcluded
        ? "transrectal_real_exclusions.csv"
        : "transrectal_exclusion_decisions_all.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}

    function markVisible(excludeValue) {{
      const visible = DATA.filter(rowMatches);
      for (const row of visible) {{
        const prev = getDecision(row);
        persistDecision(row, {{
          exclude_real: excludeValue ? 1 : 0,
          note: prev.note || "",
        }});
      }}
      excludedCount();
      render();
    }}

    for (const el of [actionFilter, labelFilter, vendorFilter]) {{
      el.addEventListener("change", render);
    }}
    for (const el of [scoreMinFilter, searchFilter]) {{
      el.addEventListener("input", render);
    }}
    markVisibleExcludeBtn.addEventListener("click", () => markVisible(true));
    markVisibleKeepBtn.addEventListener("click", () => markVisible(false));
    exportExcludeBtn.addEventListener("click", () => exportCsv(true));
    exportAllBtn.addEventListener("click", () => exportCsv(false));
    resetBtn.addEventListener("click", () => {{
      decisions = {{}};
      localStorage.removeItem(STORAGE_KEY);
      excludedCount();
      render();
    }});

    refreshVendorOptions();
    excludedCount();
    render();
  </script>
</body>
</html>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build HTML to confirm final image exclusions from quality_suspects.csv."
    )
    parser.add_argument(
        "--suspects-csv",
        type=Path,
        default=Path("artifacts/20_datasets/transrectal_lt_rect_dataset/quality_scan/quality_suspects.csv"),
        help="Input suspects CSV.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("artifacts/20_datasets/transrectal_lt_rect_dataset/quality_scan/quality_exclusion_review.html"),
        help="Output HTML path.",
    )
    parser.add_argument(
        "--recommend-score",
        type=int,
        default=120,
        help="Default recommended exclude threshold (score >= threshold).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    suspects_csv = args.suspects_csv.expanduser().resolve()
    output_html = args.output_html.expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)

    if not suspects_csv.is_file():
        raise FileNotFoundError(f"Suspects CSV non trovato: {suspects_csv}")

    rows = _load_rows(suspects_csv)
    if not rows:
        raise RuntimeError("Suspects CSV vuoto.")

    doc = _build_html(rows=rows, recommend_score=int(args.recommend_score))
    output_html.write_text(doc, encoding="utf-8")
    print(f"Rows loaded: {len(rows)}", flush=True)
    print(f"Output HTML: {output_html}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
