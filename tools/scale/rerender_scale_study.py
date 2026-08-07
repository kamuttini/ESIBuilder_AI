#!/usr/bin/env python3
"""Rebuild already-produced study pages against the current page template.

The measurements are embedded in each page as JSON, so improving the review interface does not
require re-running the analysis: pull the payload out and inject it into today's template. An
hour of detection work is not spent again to gain a text box.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "artifacts" / "50_scale_study"
PAGE = Path(__file__).with_name("study_scale_folder_page.html")
MARK = re.compile(r"^const D = (\{.*\});\s*$", re.MULTILINE)


def payload(html: str) -> dict | None:
    m = MARK.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="v3_*.html")
    ap.add_argument("--dir", default=str(OUT_DIR))
    args = ap.parse_args()
    tpl = PAGE.read_text("utf-8")
    done = skipped = 0
    for p in sorted(Path(args.dir).glob(args.glob)):
        if p.name.endswith("indice.html"):
            continue
        data = payload(p.read_text("utf-8"))
        if data is None:
            print(f"  salto {p.name}: non trovo i dati dentro la pagina")
            skipped += 1
            continue
        p.write_text(tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False)), "utf-8")
        done += 1
    print(f"[ok] {done} pagine riscritte col template attuale"
          + (f", {skipped} saltate" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
