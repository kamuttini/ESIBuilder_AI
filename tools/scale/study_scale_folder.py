"""Study the depth scale of one folder in five stages, and open it up for correction.

The stages follow the operator's reasoning rather than the code's convenience:

  A  scale zone      where the ruler lives in *this* folder, from all its frames at once,
                     so a frame that fails on its own can still be told where to look
  B  orientation     split the frames by up/down with the orientation net, then place the
                     zero at the end that orientation implies (up -> top, down -> bottom)
  C  ticks           every tick of the ruler, with the last one marked apart
  D  numbers         OCR the labels, then use the fact that the tick pitch is constant
                     inside one frame to check the numbers against each other and refit
  E  depth check     if the depth is printed in the interface (the depth module reads it),
                     compare it with the depth the scale implies at its last tick — the two
                     should land in the same ten

Everything is a proposal: the page shows each stage's outcome per frame and lets the ruler
position, the zero, the ticks and the numbers be corrected by hand, with a comment. The
export feeds ``ingest_scale_corrections.py``.

Usage:
  OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/study_scale_folder.py \
    --folder "/Volumes/SSD_esi1_n1/BK 3000 6.7.21842.36 E14CL4b - L" \
    --out artifacts/50_scale_study/bk3000.html

  # raw material, and skip the depth cross-check:
  ... --pattern "*.png" --no-depth
"""
from __future__ import annotations

import argparse
import base64
import csv
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "scale"))
from scale.detect_scale_ladder import detect_scale, load_image, profile_for  # noqa: E402

# Deliberately no filename parsing anywhere in this tool. In production the input is raw
# material whose names carry no setup index, no depth and no vendor, so anything inferred from
# a name would work on the curated corpus and quietly break on real acquisitions. Frames are
# ordered by their position in the listing, and the depth module is joined by image path.


# --------------------------------------------------------------------------- #
# stage B helpers: orientation net (optional)
# --------------------------------------------------------------------------- #
def load_orientation_models(device_name: str = "auto"):
    """Rect regressor + up/down classifier, or None when unavailable.

    The up/down net was trained on rect crops, so it needs the rect model in front of it.
    Both are optional: without them the direction falls back to what the labels say.
    """
    try:
        import importlib.util
        import torch
    except Exception:
        return None
    rect_ck = REPO / "artifacts/10_active_pipeline/pipeline_fss_head/models/rect_training_e40_run2/best_model.pt"
    su_ck = REPO / "artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/model_su_giu_rect/best_model.pt"
    if not (rect_ck.exists() and su_ck.exists()):
        return None
    try:
        # the pipeline module imports its siblings by bare name, so its own directory and the
        # repo root have to be importable before it is executed
        for extra in (REPO, REPO / "tools" / "ultrasound", REPO / "tools" / "orientation"):
            if str(extra) not in sys.path:
                sys.path.insert(0, str(extra))
        spec = importlib.util.spec_from_file_location(
            "off_mod", REPO / "tools/ultrasound/predict_fss_head_from_acquisitions.py")
        off = importlib.util.module_from_spec(spec)
        sys.modules["off_mod"] = off
        spec.loader.exec_module(off)
        dev = torch.device("mps" if (device_name in ("auto", "mps") and torch.backends.mps.is_available())
                           else "cpu")
        rc = torch.load(rect_ck, map_location="cpu", weights_only=False)
        rect = off.RectRegressor(pretrained=False).to(dev)
        rect.load_state_dict(rc["model_state_dict"])
        rect.eval()
        sc = torch.load(su_ck, map_location="cpu", weights_only=False)
        su = off.SuGiuRectClassifier().to(dev)
        su.load_state_dict(sc["model_state_dict"])
        su.eval()
        return {
            "off": off, "dev": dev, "rect": rect, "su": su,
            "rect_size": int(rc.get("args", {}).get("image_size", 320)),
            "su_size": int(sc["args"]["image_size"]),
            "classes": [str(c).strip().lower() for c in sc["class_names"]],
        }
    except Exception as exc:  # noqa: BLE001 - orientation is a bonus, never a blocker
        print(f"[orient] non disponibile: {exc}")
        return None


def predict_orientation(models, paths: List[Path]) -> Dict[str, dict]:
    """{image_path: {label, conf, rect}} using rect model + up/down net."""
    if not models or not paths:
        return {}
    off = models["off"]
    try:
        boxes, _sizes = off._predict_rect_boxes_abs(
            models["rect"], paths, models["rect_size"], 16, models["dev"])
        rows = off._predict_su_giu_on_rect_crops(
            models["su"], models["classes"], paths, boxes, None,
            models["su_size"], 16, models["dev"])
    except Exception as exc:  # noqa: BLE001
        print(f"[orient] inferenza fallita: {exc}")
        return {}
    out: Dict[str, dict] = {}
    for r, box in zip(rows, boxes):
        out[str(r["image_path"])] = {
            "label": r["pred_label"], "conf": round(float(r["confidence"]), 3),
            "rect": [round(float(v), 1) for v in box],
        }
    return out


# --------------------------------------------------------------------------- #
# stage D: the constant-pitch coherence check
# --------------------------------------------------------------------------- #
def check_numbers(labels_cm: List[Tuple[float, float]], pitch: float,
                  mm_per_px: Optional[float]) -> dict:
    """Are the read numbers consistent with each other, given a constant tick pitch?

    Inside one frame the ticks are equally spaced, so between two labels the number of tick
    steps is ``round(dy / pitch)`` and the value difference must be that many times the step.
    Pairs that agree vote for the same step; the ones that disagree are the misreads. This is
    what lets a wrong digit be spotted without any ground truth.
    """
    res = {"pairs": [], "step_votes": {}, "coherent": [], "suspect": [], "step_mm": None}
    if len(labels_cm) < 2 or pitch <= 0:
        return res
    lab = sorted(labels_cm, key=lambda t: t[0])
    votes: Dict[float, int] = {}
    for i in range(len(lab)):
        for j in range(i + 1, len(lab)):
            (y1, v1), (y2, v2) = lab[i], lab[j]
            k = round(abs(y2 - y1) / pitch)
            if k <= 0:
                continue
            step = abs(v2 - v1) * 10.0 / k  # mm per tick implied by this pair
            res["pairs"].append({"y1": round(y1, 1), "y2": round(y2, 1),
                                 "v1": v1, "v2": v2, "ticks": k, "step_mm": round(step, 2)})
            for cand in (1.0, 2.0, 2.5, 5.0, 10.0, 20.0):
                if abs(step - cand) / cand <= 0.12:
                    votes[cand] = votes.get(cand, 0) + 1
                    break
    res["step_votes"] = {str(k): v for k, v in sorted(votes.items())}
    if not votes:
        res["suspect"] = [list(x) for x in lab]
        return res
    step = max(votes, key=lambda k: votes[k])
    res["step_mm"] = step
    # a label is coherent when at least one pair through it agrees with the winning step
    ok = set()
    for p in res["pairs"]:
        if abs(p["step_mm"] - step) / step <= 0.12:
            ok.add(p["y1"])
            ok.add(p["y2"])
    for y, v in lab:
        (res["coherent"] if round(y, 1) in ok else res["suspect"]).append([round(y, 1), v])
    return res


# --------------------------------------------------------------------------- #
# stage E: depth printed in the interface
# --------------------------------------------------------------------------- #
def run_depth_module(folder: str, python_bin: str,
                     only_paths: Optional[List[str]] = None) -> Dict[str, dict]:
    """{realpath: {depth_mm, from_interface}} from the depth module.

    ``only_paths`` stages exactly the frames under study into a temporary tree of symlinks and
    points the module there, so its answers belong to *these* images. Left to itself the module
    picks its own frames and there is nothing to pair with: measured on an Esaote folder it read
    the depth on 40 images, none of which were the 12 being studied.
    """
    script = REPO / "tools/depth/predict_rect_depth_autonomous.py"
    if not script.exists():
        return {}
    with tempfile.TemporaryDirectory() as td:
        target = folder
        if only_paths:
            staged = Path(td) / "stage" / "image_samples"
            staged.mkdir(parents=True, exist_ok=True)
            for p in only_paths:
                link = staged / os.path.basename(p)
                if not link.exists():
                    try:
                        link.symlink_to(os.path.realpath(p))
                    except OSError:
                        pass
            target = str(Path(td) / "stage")
        cmd = [python_bin, str(script), "--folder", target, "--output-dir", td,
               "--max-images", str(max(40, len(only_paths or [])))]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except Exception as exc:  # noqa: BLE001
            print(f"[depth] non eseguito: {exc}")
            return {}
        if p.returncode != 0:
            print(f"[depth] uscito con {p.returncode}: {(p.stderr or '')[-300:]}")
        # Joined by image path, i.e. by identity of the file. The depth module may well look at
        # other frames of the same folder than the ones under study; when it does, stage E
        # simply reports "non verificabile" instead of pairing two different acquisitions
        # because their names happened to share a number.
        out: Dict[str, dict] = {}
        for csv_path in sorted(Path(td).glob("*.csv")):
            try:
                rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
            except Exception:
                continue
            for r in rows:
                name = os.path.basename(r.get("image_path") or r.get("image") or "")
                if not name:
                    continue
                key = os.path.realpath(r.get("image_path") or r.get("image") or "")
                d = None
                for k in ("depth_mm", "ocr_snapped_depth_mm", "depth"):
                    try:
                        v = float(r.get(k, "") or "")
                        if v > 0:
                            d = v
                            break
                    except (TypeError, ValueError):
                        continue
                if d is None:
                    continue
                has_text = str(r.get("ocr_text_has_depth", "")).strip().lower() in ("1", "true", "yes")
                prev = out.get(key)
                if prev is None or (has_text and not prev["from_interface"]):
                    out[key] = {"depth_mm": round(d, 1), "from_interface": has_text}
        if out:
            print(f"[depth] letta la depth su {len(out)} immagini")
        return out


# --------------------------------------------------------------------------- #
# main study
# --------------------------------------------------------------------------- #
def _b64(img: np.ndarray, q: int = 72) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def study(folder: str, pattern: str, max_images: int, cap_w: int, vendor: str,
          with_depth: bool, python_bin: str, device: str) -> Optional[dict]:
    base = os.path.join(folder, "image_samples")
    if not os.path.isdir(base):
        base = folder
    paths = sorted(glob.glob(os.path.join(base, pattern)))[:max_images]
    if not paths:
        print(f"[error] nessuna immagine con pattern {pattern} in {base}")
        return None
    vendor = vendor or _guess_vendor(folder)
    prof = profile_for(vendor)
    print(f"[info] {len(paths)} immagini, vendor='{vendor or 'default'}'")

    # ---- stage B first: orientation, so the zero end is known before we look at ticks ----
    models = load_orientation_models(device)
    orient = predict_orientation(models, [Path(p) for p in paths]) if models else {}
    if orient:
        groups: Dict[str, int] = {}
        for v in orient.values():
            groups[v["label"]] = groups.get(v["label"], 0) + 1
        print(f"[B] orientamento: {groups}")
    else:
        print("[B] orientamento non disponibile: il verso verra' dalle etichette")

    depth_info = run_depth_module(folder, python_bin, paths) if with_depth else {}

    # ---- first pass: detect on every frame, to learn the folder's scale zone ----
    first: List[dict] = []
    for p in paths:
        img = load_image(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        o = orient.get(p) or orient.get(str(p)) or {}
        rect = None
        if o.get("rect"):
            t, l, b, rr = o["rect"]  # the pipeline returns (top,left,bottom,right)
            rect = (int(l), int(t), int(rr - l), int(b - t))
        pred = detect_scale(gray, rect=rect, vendor=vendor)
        first.append({"path": p, "gray": gray, "pred": pred, "orient": o, "rect": rect})

    xs = [f["pred"].x for f in first if f["pred"].x is not None]
    zone = None
    if xs:
        cx = float(statistics.median(xs))
        spread = float(statistics.median([abs(x - cx) for x in xs])) if len(xs) > 1 else 0.0
        half = max(40.0, 3.0 * spread + 25.0)
        zone = {"x": round(cx, 1), "half": round(half, 1),
                "found": len(xs), "total": len(first)}
        print(f"[A] zona scala: x={cx:.0f} ±{half:.0f} px (trovata su {len(xs)}/{len(first)})")

    # ---- second pass: frames without a ruler get told where the folder's zone is ----
    frames: List[dict] = []
    for f in first:
        pred = f["pred"]
        used_zone = False
        if zone and (not pred.ok or pred.mm_per_px is None):
            retry = detect_scale(f["gray"], rect=f["rect"], vendor=vendor,
                                 prior_x=zone["x"], prior_band_px=int(zone["half"]))
            if retry.n_ticks and (retry.mm_per_px or not pred.n_ticks):
                pred, used_zone = retry, True

        name = os.path.basename(f["path"])
        idx = len(frames)  # position in the listing: no meaning is claimed beyond ordering

        # stage B: the zero end implied by orientation, and whether detection agrees
        ori = (f["orient"] or {}).get("label", "")
        zero_from_orient = "top" if ori == "su" else ("bottom" if ori == "giu" else "")
        zero_detected = "top" if pred.direction >= 0 else "bottom"
        y_zero = pred.y_zero
        zero_moved = False
        if zero_from_orient and pred.ticks_y and zero_from_orient != zero_detected:
            # trust the orientation for *which end*, keep the ticks for *where*
            y_zero = min(pred.ticks_y) if zero_from_orient == "top" else max(pred.ticks_y)
            zero_moved = True

        # stage C: ticks, last one apart
        dirn = 1 if (zero_from_orient or zero_detected) == "top" else -1
        far = max(pred.ticks_y) if (dirn >= 0 and pred.ticks_y) else (
            min(pred.ticks_y) if pred.ticks_y else None)

        # stage D: numbers vs constant pitch
        labels_cm = [(y, round(v / 10.0, 2)) for y, v in (pred.labels_all or [])]
        coh = check_numbers(labels_cm, pred.tick_pitch_px or 0.0, pred.mm_per_px)

        # stage E: does the scale agree with the depth printed in the interface?
        dep = depth_info.get(os.path.realpath(f["path"]))
        scale_depth = None
        depth_verdict = ""
        if pred.mm_per_px and y_zero is not None and far is not None:
            scale_depth = round(abs(far - y_zero) * pred.mm_per_px, 1)
        if dep and scale_depth:
            same_ten = round(dep["depth_mm"] / 10.0) == round(scale_depth / 10.0)
            depth_verdict = "coerente" if same_ten else "NON coerente"
            if not dep["from_interface"]:
                depth_verdict += " (depth non letta dall'interfaccia)"

        s = min(1.0, cap_w / float(f["gray"].shape[1]))
        view = cv2.cvtColor(f["gray"], cv2.COLOR_GRAY2BGR)
        if s < 1.0:
            view = cv2.resize(view, (int(f["gray"].shape[1] * s), int(f["gray"].shape[0] * s)),
                              interpolation=cv2.INTER_AREA)
        frames.append({
            "name": name, "idx": idx, "img": _b64(view),
            "w": f["gray"].shape[1], "h": f["gray"].shape[0], "scale": round(s, 6),
            "status": pred.status, "reason": pred.reason,
            "x": pred.x, "y_zero": y_zero, "y_far": far,
            "ticks": [round(t, 1) for t in (pred.ticks_y or [])],
            "labels": [[round(y, 1), v] for y, v in labels_cm],
            "mm_per_px": pred.mm_per_px, "pitch": pred.tick_pitch_px,
            "n_ticks": pred.n_ticks,
            "A_used_zone": used_zone,
            "B_orient": ori, "B_conf": (f["orient"] or {}).get("conf"),
            "B_zero_end": zero_from_orient or zero_detected,
            "B_zero_moved": zero_moved,
            "D_step_mm": coh["step_mm"], "D_votes": coh["step_votes"],
            "D_coherent": coh["coherent"], "D_suspect": coh["suspect"],
            "E_depth_interface": dep["depth_mm"] if dep else None,
            "E_depth_from_interface": bool(dep and dep["from_interface"]),
            "E_depth_scale": scale_depth, "E_verdict": depth_verdict,
        })

    st: Dict[str, int] = {}
    for fr in frames:
        st[fr["status"]] = st.get(fr["status"], 0) + 1
    print(f"[C/D] stati: {st}")
    n_sus = sum(1 for fr in frames if fr["D_suspect"])
    print(f"[D] frame con numeri sospetti: {n_sus}/{len(frames)}")
    if depth_info:
        agree = sum(1 for fr in frames if fr["E_verdict"].startswith("coerente"))
        checked = sum(1 for fr in frames if fr["E_verdict"])
        print(f"[E] depth coerente su {agree}/{checked} frame verificabili")
    return {"folder": os.path.basename(folder.rstrip("/\\")), "vendor": vendor or "default",
            "zone": zone, "orient_groups": {k: sum(1 for v in orient.values() if v["label"] == k)
                                            for k in {v["label"] for v in orient.values()}} if orient else {},
            "status": st, "frames": frames}


def _guess_vendor(folder: str) -> str:
    from scale.detect_scale_ladder import PROFILES
    n = os.path.basename(folder.rstrip("/\\")).lower()
    for v in PROFILES:
        if v != "default" and v.lower() in n:
            return v
    for alias, v in (("toshiba", "Canon"), ("bk", "BK")):
        if alias in n:
            return v
    return ""


PAGE = Path(__file__).with_name("study_scale_folder_page.html")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--pattern", default="image_depth_value_setup_*.png")
    ap.add_argument("--vendor", default="")
    ap.add_argument("--max-images", type=int, default=14)
    ap.add_argument("--cap-width", type=int, default=1400)
    ap.add_argument("--no-depth", action="store_true", help="Skip the stage-E depth cross-check.")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--high-recall", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.high_recall:
        os.environ["SCALE_HIGH_RECALL"] = "1"

    data = study(args.folder, args.pattern, args.max_images, args.cap_width, args.vendor,
                 not args.no_depth, args.python_bin, args.device)
    if not data:
        return 2
    tpl = PAGE.read_text(encoding="utf-8")
    html = tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"[ok] {len(data['frames'])} frame -> {args.out} ({len(html)/1024/1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
