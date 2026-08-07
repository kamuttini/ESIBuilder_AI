#!/usr/bin/env python3
"""One page listing every review produced by a sweep, so they can be judged side by side.

Reads the sweep's results JSON and links each study, showing at a glance where the scale zone
was found, how the frames were classified and how many numbers came out suspect.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "artifacts" / "50_scale_study"

CSS = """
:root{--bg:#0b0d11;--panel:#14171e;--ink:#e9eaee;--mut:#98a0ad;--line:#262c37;
      --good:#37d67a;--warn:#ffb020;--bad:#ff5470}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:13px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{margin:0 0 4px;font-size:19px} .muted{color:var(--mut)}
.wrap{padding:18px 24px 60px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600;position:sticky;top:0;background:var(--bg)}
tr:hover td{background:#12141a}
a{color:#6aa6ff;text-decoration:none} a:hover{text-decoration:underline}
.fam{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.pill{display:inline-block;padding:1px 7px;border-radius:99px;border:1px solid var(--line);
      font-size:11px;color:var(--mut);white-space:nowrap}
.ok{border-color:var(--good);color:var(--good)}
.wr{border-color:var(--warn);color:var(--warn)}
.bd{border-color:var(--bad);color:var(--bad)}
.sum{display:flex;gap:22px;flex-wrap:wrap;margin:12px 0 20px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:11px 15px}
.card b{display:block;font-size:21px;font-weight:700}
.nm{max-width:430px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
"""


def pill(txt: str, cls: str = "") -> str:
    return f'<span class="pill {cls}">{html.escape(txt)}</span>'


def build(results: list, title: str) -> str:
    rows = []
    n_ok = sum(1 for r in results if r.get("ok"))
    zf = sum(r.get("zone_found", 0) for r in results)
    zt = sum(r.get("zone_total", 0) for r in results)
    acc = sum((r.get("stati") or {}).get("accepted", 0) for r in results)
    tot = sum(sum((r.get("stati") or {}).values()) for r in results)
    sus = sum(r.get("suspect", 0) for r in results)

    for r in sorted(results, key=lambda x: (x.get("family", ""), x.get("acquisition", ""))):
        st = r.get("stati") or {}
        n = sum(st.values())
        parts = []
        for k, cls in (("accepted", "ok"), ("review", "wr"), ("reject", "bd")):
            if st.get(k):
                parts.append(pill(f"{st[k]} {k}", cls))
        zone = ""
        if r.get("zone_total"):
            f, t = r["zone_found"], r["zone_total"]
            zone = pill(f"{f}/{t}", "ok" if f == t else ("wr" if f >= t * 0.6 else "bd"))
        ori = ", ".join(f"{k}:{v}" for k, v in (r.get("orient") or {}).items()) or "—"
        vend = html.escape(r.get("vendor") or "default")
        vsrc = html.escape(r.get("vendor_src") or "")
        name = html.escape(r.get("acquisition", ""))
        link = (f'<a href="{html.escape(r["out"])}" target="_blank">{name}</a>'
                if r.get("ok") else f'{name} <span class="pill bd">non riuscito</span>')
        s = r.get("suspect")
        rows.append(
            f'<tr><td class="fam">{html.escape(r.get("family",""))}</td>'
            f'<td class="nm">{link}<div class="muted">{r.get("n_images","?")} frame '
            f'nella cartella</div></td>'
            f'<td>{vend}<div class="muted">{vsrc}</div></td>'
            f'<td>{zone}</td><td>{" ".join(parts) or "—"}</td>'
            f'<td>{pill(f"{s}/{n}", "wr" if s else "ok") if s is not None else "—"}</td>'
            f'<td class="muted">{html.escape(ori)}</td>'
            f'<td class="muted">{r.get("seconds","")}s</td></tr>')

    return f"""<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<header><h1>{html.escape(title)}</h1>
<div class="muted">Ogni riga e' una cartella diversa. Clicca il nome per aprire la revisione
 con i cinque stadi, dove puoi correggere e commentare.</div></header>
<div class="wrap">
 <div class="sum">
  <div class="card"><b>{n_ok}/{len(results)}</b><span class="muted">studi riusciti</span></div>
  <div class="card"><b>{len({r.get('family') for r in results})}</b>
    <span class="muted">famiglie di ecografi</span></div>
  <div class="card"><b>{zf}/{zt}</b><span class="muted">frame con zona scala trovata</span></div>
  <div class="card"><b>{acc}/{tot}</b><span class="muted">frame accepted</span></div>
  <div class="card"><b>{sus}</b><span class="muted">frame con numeri sospetti</span></div>
 </div>
 <table><tr><th>famiglia</th><th>cartella</th><th>vendor</th><th>zona</th><th>stati</th>
   <th>sospetti</th><th>orientamento</th><th>tempo</th></tr>
 {"".join(rows)}</table>
</div></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(OUT_DIR / "v3_sweep.json"))
    ap.add_argument("--out", default=str(OUT_DIR / "v3_indice.html"))
    ap.add_argument("--title", default="Studio della scala — volume 3")
    args = ap.parse_args()
    data = json.loads(Path(args.results).read_text("utf-8"))
    Path(args.out).write_text(build(data, args.title), "utf-8")
    print(f"[ok] {len(data)} studi -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
