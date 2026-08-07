#!/usr/bin/env python3
"""Run the 5-stage scale study over a whole corpus volume and index the reviews.

The volume holds thousands of leaf folders, far more than anyone can look at, so this picks a
sample that spans the machines and studies one leaf folder per acquisition.

Two rules keep the sample honest:
  - the leaf folder inside an acquisition is chosen by frame count alone, never by name, so no
    folder is favoured for looking like a depth sweep;
  - the vendor comes from the classifier inside the study tool, not from the path, which on this
    volume would leak the answer (the acquisition folders are named after the machine).

The acquisition names are used for one thing only: spreading the sample across vendors, which is
an operator's sampling decision and never reaches the detector.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "artifacts" / "50_scale_study"

# For sampling only: the families we want represented.
FAMILIES = ("esaote", "bk", "ge", "hitachi", "arietta", "fujifilm", "mindray", "canon", "toshiba",
            "aplio", "philips", "siemens", "acuson", "alpinion", "terason", "supersonic",
            "aixplorer", "exact", "edap", "koelis", "vinno", "sonostar", "carbon", "echolaser",
            "biopsee", "uronav", "samsung")
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def family_of(name: str) -> str:
    low = name.lower()
    for f in FAMILIES:
        if f in low:
            return {"arietta": "hitachi", "fujifilm": "hitachi", "aplio": "canon",
                    "toshiba": "canon", "acuson": "siemens", "aixplorer": "supersonic",
                    "edap": "exact"}.get(f, f)
    return "altri"


def count_images(d: str) -> int:
    try:
        return sum(1 for e in os.scandir(d)
                   if e.is_file() and e.name.lower().endswith(IMG_EXT)
                   and not e.name.startswith("."))
    except OSError:
        return 0


def pick_leaf(acq: str, target: int, lo: int, hi: int) -> Optional[Tuple[str, int]]:
    """Leaf folder whose frame count sits closest to the target — by count, never by name."""
    best: Optional[Tuple[float, str, int]] = None
    for dp, dn, _fn in os.walk(acq):
        dn[:] = [d for d in dn if not d.startswith(".")]
        n = count_images(dp)
        if n < lo:
            continue
        score = abs(n - target) + (1000 if n > hi else 0)
        if best is None or score < best[0]:
            best = (score, dp, n)
    return (best[1], best[2]) if best else None


def slug(s: str, n: int = 58) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", s)).strip("_")[:n]


def select(root: str, per_family: int, target: int, lo: int, hi: int) -> List[dict]:
    acqs = sorted([e.path for e in os.scandir(root)
                   if e.is_dir() and not e.name.startswith(("$", "."))
                   and e.name != "System Volume Information"])
    if len(acqs) == 1:  # a single wrapper directory: descend into it
        inner = sorted([e.path for e in os.scandir(acqs[0])
                        if e.is_dir() and not e.name.startswith(("$", "."))])
        if inner:
            acqs = inner
    by_fam: Dict[str, List[str]] = {}
    for a in acqs:
        by_fam.setdefault(family_of(os.path.basename(a)), []).append(a)

    chosen: List[dict] = []
    for fam in sorted(by_fam):
        # newest last in this corpus, and the tail is the interesting part: recent machines are
        # the ones the tool will actually meet in the field
        for acq in list(reversed(by_fam[fam]))[:per_family]:
            got = pick_leaf(acq, target, lo, hi)
            if not got:
                continue
            leaf, n = got
            chosen.append({"family": fam, "acquisition": os.path.basename(acq),
                           "folder": leaf, "n_images": n,
                           "out": f"v3_{slug(os.path.basename(acq))}.html"})
    return chosen


def run_one(item: dict, python_bin: str, max_images: int, with_depth: bool,
            pattern: str) -> dict:
    cmd = [python_bin, str(REPO / "tools/scale/study_scale_folder.py"),
           "--folder", item["folder"], "--pattern", pattern,
           "--max-images", str(max_images), "--out", str(OUT_DIR / item["out"])]
    if not with_depth:
        cmd.append("--no-depth")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=1800)
        out = (p.stdout or "") + (p.stderr or "")
        ok = p.returncode == 0 and (OUT_DIR / item["out"]).exists()
    except subprocess.TimeoutExpired:
        out, ok = "timeout", False
    res = dict(item)
    res.update({"ok": ok, "seconds": round(time.time() - t0, 1), "log": out.strip().splitlines()})
    for line in res["log"]:
        if line.startswith("[info]") and "vendor=" in line:
            m = re.search(r"vendor='([^']*)'", line)
            res["vendor"] = m.group(1) if m else ""
            res["vendor_src"] = line.split("(", 1)[1].rstrip(")") if "(" in line else ""
        elif line.startswith("[A]"):
            m = re.search(r"trovata su (\d+)/(\d+)", line)
            if m:
                res["zone_found"], res["zone_total"] = int(m.group(1)), int(m.group(2))
        elif line.startswith("[C/D] stati:"):
            try:
                res["stati"] = json.loads(line.split(":", 1)[1].strip().replace("'", '"'))
            except Exception:  # noqa: BLE001
                pass
        elif line.startswith("[D] frame con numeri sospetti:"):
            m = re.search(r"(\d+)/(\d+)", line)
            if m:
                res["suspect"] = int(m.group(1))
        elif line.startswith("[B] orientamento:"):
            try:
                res["orient"] = json.loads(line.split(":", 1)[1].strip().replace("'", '"'))
            except Exception:  # noqa: BLE001
                pass
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Volumes/SSD_esi1_n3")
    ap.add_argument("--per-family", type=int, default=3)
    ap.add_argument("--max-images", type=int, default=12)
    ap.add_argument("--target", type=int, default=14, help="ideal frame count of the leaf folder")
    ap.add_argument("--min-images", type=int, default=8)
    ap.add_argument("--max-leaf", type=int, default=60)
    ap.add_argument("--pattern", default="*.png")
    ap.add_argument("--with-depth", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--results", default=str(OUT_DIR / "v3_sweep.json"))
    ap.add_argument("--only-select", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[sel] esamino {args.root}")
    items = select(args.root, args.per_family, args.target, args.min_images, args.max_leaf)
    print(f"[sel] {len(items)} cartelle scelte su "
          f"{len({i['family'] for i in items})} famiglie di ecografi")
    for i in items:
        print(f"  {i['family']:<11} {i['n_images']:4d} img  {i['acquisition'][:70]}")
    if args.only_select:
        return 0

    done: List[dict] = []
    for k, item in enumerate(items, 1):
        print(f"\n[{k}/{len(items)}] {item['acquisition'][:70]}")
        if not os.path.isdir(item["folder"]):
            print("  cartella non raggiungibile (volume smontato?), la salto")
            continue
        r = run_one(item, args.python, args.max_images, args.with_depth, args.pattern)
        done.append(r)
        head = [l for l in r["log"] if l.startswith(("[info]", "[A]", "[C/D]", "[D]", "[E]"))]
        for line in head:
            print("  " + line)
        if not r["ok"]:
            print("  NON riuscito")
        Path(args.results).write_text(json.dumps(done, indent=1, ensure_ascii=False), "utf-8")
    print(f"\n[ok] {sum(1 for d in done if d['ok'])}/{len(done)} studi riusciti "
          f"-> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
