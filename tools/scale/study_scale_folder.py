"""Study the depth scale of one folder in six stages, and open it up for correction.

The stages follow the operator's reasoning rather than the code's convenience:

  1  orientation     from the orientation marker (the net is only its prior): up means the zero
                     is at the top, down means the bottom. Cross-checked against the OCR: the
                     smallest number must sit at the distance from the zero its value implies
  2  depth           the depth printed in the interface, cross-checked against the largest
                     number on the ruler and against the last tick - they should land close
  3  ruler           where the ruler is, one position for the whole folder, so a frame that
                     fails on its own can still be told where to look
  4  ticks           every tick, with the zero end and the far end kept distinct
  5  pitch           the step between one tick and the next
  6  numbers         OCR, verified against each other through that constant pitch

Detection necessarily runs before all of this: the cross-checks in 1 and 2 need numbers and
ticks to exist. The order above is the order of the decisions, not of the computation.

Everything is a proposal: the page shows each stage's outcome per frame and lets the ruler
position, the zero, the ticks, the numbers, the orientation and the depth be corrected by
hand, with a comment. Corrections about the marker and the depth are exported in the shapes
those two modules already read.

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
from typing import Dict, List, Optional, Sequence, Tuple

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


def load_marker_detector():
    """The orientation-marker detector bundle, or None.

    The marker is the evidence the operator trusts for orientation: it can *correct* the up/down
    the net predicted, and up/down is exactly what decides which end of the ruler carries the
    zero. The net stays in front of it as the prior it expects.
    """
    bundle = REPO / "artifacts/41_orientation_marker_detector_bundle"
    if not (bundle / "orientation_marker_detector" / "detector.py").is_file():
        return None
    try:
        if str(bundle) not in sys.path:
            sys.path.insert(0, str(bundle))
        from orientation_marker_detector import detector as omd  # noqa: PLC0415
        return omd
    except Exception as exc:  # noqa: BLE001
        print(f"[marker] bundle non caricabile: {exc}")
        return None


def marker_orientation(omd, paths: List[str], vendor: str,
                       orient: Dict[str, dict]) -> Dict[str, dict]:
    """{path: {vertical, group, corrected, score, status}} from the marker templates.

    The template bank is chosen by vendor, and the vendor here comes from the classifier, not
    from the folder name — on raw material the name says nothing.
    """
    if omd is None:
        return {}
    templates = []
    for name in (vendor, vendor.capitalize(), vendor.upper()):
        if not name:
            break
        try:
            templates = omd.load_vendor_templates(None, name)
        except Exception as exc:  # noqa: BLE001
            print(f"[marker] libreria non leggibile: {exc}")
            return {}
        if templates:
            break
    if not templates:
        print(f"[marker] nessun template per vendor '{vendor or 'sconosciuto'}'")
        return {}
    inputs = []
    for p in paths:
        o = orient.get(p) or {}
        rect = o.get("rect")
        inputs.append(omd.ImageInput(
            image_path=Path(p), image_id=os.path.basename(p),
            crop_rect=tuple(int(round(v)) for v in rect) if rect else None,
            crop_source="rect_net" if rect else "",
            sugiu_pred=str(o.get("label", "")),
            sugiu_conf=float(o.get("conf") or 0.0)))
    params = omd.DetectionParams()
    try:
        template, _rows, _rank = omd.select_best_template(inputs[:4], templates, params)
    except Exception as exc:  # noqa: BLE001
        print(f"[marker] scelta del template fallita: {exc}")
        return {}
    out: Dict[str, dict] = {}
    for inp in inputs:
        try:
            r = omd.detect_marker(inp, template, params)
        except Exception as exc:  # noqa: BLE001
            print(f"[marker] {inp.image_id}: {exc}")
            continue
        out[str(inp.image_path)] = {
            "vertical": str(r.vertical_final or ""), "group": str(r.orientation_group or ""),
            "corrected": str(r.vertical_correction or ""), "source": str(r.vertical_source or ""),
            "score": round(float(r.match_score), 3) if r.match_score is not None else None,
            "status": str(r.status or ""), "template": str(r.template_name or ""),
            "box": list(r.marker_box_abs) if r.marker_box_abs else None,
        }
    if out:
        ncorr = sum(1 for v in out.values() if v["corrected"] == "corrected")
        print(f"[1] marker: template '{template.name}', trovato su "
              f"{sum(1 for v in out.values() if v['score'])}/{len(out)} frame"
              + (f", ha corretto il su/giu' su {ncorr}" if ncorr else ""))
    return out


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
# cross-checks: the zero against the OCR, the scale against the interface depth
# --------------------------------------------------------------------------- #
def check_zero_with_ocr(y_zero: Optional[float], labels_mm: List[Tuple[float, float]],
                        mm_per_px: Optional[float], pitch: Optional[float],
                        y_far: Optional[float] = None) -> dict:
    """Do the read numbers agree that the zero is where orientation says it is?

    The smallest label is the one nearest the zero, so its distance from the zero must equal its
    own value converted to pixels. If it does not, either the zero is at the wrong end or the
    number was misread — and saying which is not this function's job, only that they disagree.
    """
    res = {"verdict": "non verificabile", "detail": "", "expected_y": None, "label": None}
    if y_zero is None:
        res["detail"] = "nessuno zero da verificare"
        return res
    if not labels_mm:
        res["detail"] = "nessun numero letto sulla scala"
        return res
    y_lo, v_lo = min(labels_mm, key=lambda t: t[1])
    res["label"] = [round(y_lo, 1), round(v_lo / 10.0, 2)]
    if not mm_per_px or mm_per_px <= 0:
        res["detail"] = "senza mm per pixel non posso convertire"
        return res
    dist_px = abs(y_lo - y_zero)
    expected_px = v_lo / mm_per_px
    res["expected_y"] = round(y_zero + (1 if y_lo > y_zero else -1) * expected_px, 1)
    tol = max(6.0, 0.35 * (pitch or 8.0))
    # If the smallest label is as far from one end of the ladder as from the other, the check
    # passes whichever end is called zero: it confirms nothing. Measured on BK 18L5, where the
    # ruler runs 0..2 cm and the 1 cm label sits 162 px from one end and 161 from the other.
    if y_far is not None and abs(abs(y_lo - y_zero) - abs(y_lo - y_far)) <= tol:
        res["verdict"] = "ambiguo"
        res["detail"] = (f"il numero piu' piccolo ({v_lo / 10.0:g} cm) sta a mezza scala: dista "
                         f"{abs(y_lo - y_zero):.0f} px da un capo e {abs(y_lo - y_far):.0f} px "
                         f"dall'altro, quindi l'OCR non puo' dire quale dei due sia lo zero")
        return res
    if abs(dist_px - expected_px) <= tol:
        res["verdict"] = "confermato"
        res["detail"] = (f"il numero piu' piccolo ({v_lo / 10.0:g} cm) sta a {dist_px:.0f} px "
                         f"dallo zero, e {expected_px:.0f} px e' quello che dovrebbe essere")
    else:
        res["verdict"] = "in disaccordo"
        res["detail"] = (f"il numero piu' piccolo ({v_lo / 10.0:g} cm) sta a {dist_px:.0f} px "
                         f"dallo zero, ma dovrebbe stare a {expected_px:.0f} px: "
                         f"o lo zero e' all'altro capo, o quel numero e' letto male")
    return res


def check_depth_against_scale(depth_mm: Optional[float], from_interface: bool,
                              labels_mm: List[Tuple[float, float]],
                              scale_depth: Optional[float]) -> dict:
    """Interface depth against the largest number on the ruler, and against the last tick.

    The biggest label is the deepest mark the machine drew, so with the depth printed in the
    interface the two should land close. Only a depth actually printed there is evidence: one
    derived from the scale would be compared with itself.
    """
    res = {"verdict": "non verificabile", "detail": "", "max_label_mm": None}
    if labels_mm:
        res["max_label_mm"] = round(max(v for _y, v in labels_mm), 1)
    if depth_mm is None:
        res["detail"] = "il modulo depth non ha dato un valore"
        return res
    if not from_interface:
        res["detail"] = "la depth non e' scritta nell'interfaccia, confrontarla non prova nulla"
        return res
    parts = []
    ok = None
    if res["max_label_mm"] is not None:
        d = abs(depth_mm - res["max_label_mm"])
        near = d <= max(10.0, 0.15 * depth_mm)
        ok = near
        parts.append(f"numero piu' grande {res['max_label_mm']:g} mm contro depth "
                     f"{depth_mm:g} mm: {'vicini' if near else f'lontani di {d:.0f} mm'}")
    if scale_depth is not None:
        d2 = abs(depth_mm - scale_depth)
        near2 = round(depth_mm / 10.0) == round(scale_depth / 10.0) or d2 <= max(10.0, 0.15 * depth_mm)
        ok = near2 if ok is None else (ok and near2)
        parts.append(f"ultima tacca {scale_depth:g} mm: {'vicina' if near2 else f'lontana di {d2:.0f} mm'}")
    if ok is None:
        res["detail"] = "niente da confrontare sulla scala"
        return res
    res["verdict"] = "conferma" if ok else "NON conferma"
    res["detail"] = "; ".join(parts)
    return res


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
                     only_paths: Optional[List[str]] = None,
                     vendor: str = "",
                     rect_echo: Optional[Tuple[float, float, float, float]] = None,
                     ) -> Dict[str, dict]:
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
        # The module's own checkpoint asks for vendor and the ultrasound rectangle when there is
        # no .fss: both drive its vendor profile and its search region, and we already know them
        # from the classifier and the rect regressor. Calling it bare threw that away.
        if vendor:
            cmd += ["--vendor", vendor]
        if rect_echo:
            cmd += ["--rect-echo", ",".join(str(int(round(v))) for v in rect_echo)]
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
                # 'mode' is the module's own verdict on where the value came from:
                # direct_label means it read it in the interface text, introduced by
                # Depth/D/P/R or by stable unit evidence. Using ocr_text_has_depth instead
                # demanded the literal word "depth" and threw away every machine that writes
                # it differently — the module had found it, this join discarded it.
                mode = str(r.get("mode", "") or "").strip()
                status = str(r.get("status", "") or "").strip()
                from_iface = mode == "direct_label" or str(
                    r.get("ocr_text_has_depth", "")).strip().lower() in ("1", "true", "yes")
                rank = ({"accepted": 2, "review": 1}.get(status, 0), 1 if from_iface else 0)
                prev = out.get(key)
                if prev is None or rank > prev["rank"]:
                    # Keep the evidence, not just the number: where the module looked, what it
                    # read there and why it chose it. A depth with no visible evidence cannot be
                    # judged, only believed.
                    box = None
                    try:
                        box = [float(r[k]) for k in ("left", "top", "right", "bottom")]
                    except (KeyError, TypeError, ValueError):
                        box = None
                    out[key] = {"depth_mm": round(d, 1), "from_interface": from_iface,
                                "mode": mode, "status": status, "rank": rank,
                                "ocr_text": str(r.get("ocr_text", "") or "")[:80],
                                "box": box,
                                "score": _f(r.get("score")), "ranker": _f(r.get("ranker_score")),
                                "reason": str(r.get("reason", "") or "")[:160]}
        if out:
            n_iface = sum(1 for v in out.values() if v["from_interface"])
            print(f"[depth] letta su {len(out)} immagini, {n_iface} dall'interfaccia "
                  f"(modi: {_tally(v['mode'] for v in out.values())})")
        return out


def load_folder_corrections(path: str, folder: str) -> dict:
    """The operator's corrections for this folder: {frame_name: {...}}, plus the folder note.

    Keyed by the folder's own name, the same key the review page uses. An empty result is normal
    and must never be treated as an error: most folders have never been reviewed.
    """
    if not path:
        return {}
    try:
        allc = json.loads(Path(path).read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[corr] non riesco a leggere {path}: {exc}")
        return {}
    entry = allc.get(os.path.basename(folder.rstrip("/\\"))) or {}
    frames = entry.get("frames") or {}
    if frames or entry.get("note"):
        print(f"[corr] correzioni per questa cartella: {len(frames)} frame"
              + (", piu' il commento" if entry.get("note") else ""))
    return entry


def _corr_anchor_x(frames: dict) -> Optional[float]:
    """The column the operator settled on, if they agreed with themselves across frames.

    A single corrected column is worth more than the median of the detections, but only when the
    corrections do not disagree among themselves — otherwise it is not a folder-wide fact.
    """
    xs = [float(c["x"]) for c in frames.values() if c.get("x") not in (None, "")]
    if not xs:
        return None
    if max(xs) - min(xs) > 12.0:
        print(f"[corr] colonne corrette discordi ({min(xs):.0f}..{max(xs):.0f}), non le uso "
              f"come ancora della cartella")
        return None
    return float(statistics.median(xs))


def _f(v) -> Optional[float]:
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return None


def _tally(items) -> str:
    c: Dict[str, int] = {}
    for i in items:
        c[i or "?"] = c.get(i or "?", 0) + 1
    return ", ".join(f"{k}:{v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------- #
# main study
# --------------------------------------------------------------------------- #
def _b64(img: np.ndarray, q: int = 72) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def evidence_from_pipeline(context: dict) -> Tuple[List[str], str, Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
    """``(paths, vendor, orient, marker, depth_info)`` from a pipeline stage context JSON.

    The pipeline already computed all of this on these very frames, so reviewing its answers
    means *reading* its evidence rather than producing a second, slightly different opinion
    with the same nets. What comes back is shaped exactly like what the local loaders return,
    so the six stages below do not know the difference.
    """
    paths: List[str] = []
    orient: Dict[str, dict] = {}
    marker: Dict[str, dict] = {}
    depth_info: Dict[str, dict] = {}
    for fr in context.get("frames") or []:
        path = str(fr.get("image_path") or "")
        if not path or not os.path.isfile(path):
            continue
        paths.append(path)
        rect_ltrb = fr.get("rect_ltrb") or []
        rect_tlbr = None
        if len(rect_ltrb) >= 4:  # the study speaks (top,left,bottom,right)
            left, top, right, bottom = (float(v) for v in rect_ltrb[:4])
            rect_tlbr = [top, left, bottom, right]
        orient[path] = {
            "label": str(fr.get("sugiu") or ""),
            "conf": fr.get("sugiu_conf"),
            "rect": rect_tlbr,
            "source": str(fr.get("sugiu_source") or "pipeline"),
            "net_label": "",
        }
        marker[path] = {
            "vertical": str(fr.get("sugiu") or ""),
            "group": str(fr.get("orientation_group") or ""),
            "corrected": "corrected" if "corretto" in str(fr.get("sugiu_source") or "") else "",
            "source": str(fr.get("sugiu_source") or ""),
            "score": fr.get("marker_score"),
            "status": "", "template": "", "box": None,
        }
        depth_mm = fr.get("depth_mm")
        if depth_mm is not None:
            depth_info[os.path.realpath(path)] = {
                "depth_mm": float(depth_mm),
                "from_interface": bool(fr.get("depth_from_interface")),
                "mode": str(fr.get("depth_mode") or ""),
                "status": str(fr.get("depth_status") or ""),
                "ocr_text": str(fr.get("depth_ocr_text") or ""),
                "box": fr.get("depth_box"),
                "score": None, "ranker": None, "reason": "",
            }
    return paths, str(context.get("vendor_predicted") or ""), orient, marker, depth_info


def study(folder: str, pattern: str, max_images: int, cap_w: int, vendor: str,
          with_depth: bool, python_bin: str, device: str,
          corrections: str = "", pipeline_context: Optional[dict] = None,
          only: Optional[Sequence[str]] = None) -> Optional[dict]:
    from_pipeline = bool(pipeline_context)
    if from_pipeline:
        paths, ctx_vendor, orient, marker, depth_info = evidence_from_pipeline(pipeline_context or {})
        # ``only`` names the frames to study, and it is the caller that knows which ones are
        # worth it: the ruler is built once per depth, on a single orientation, so studying
        # all four is three quarters of the work for no extra evidence.
        if only:
            voluti = {os.path.basename(str(n)) for n in only}
            paths = [p for p in paths if os.path.basename(p) in voluti]
        paths = paths[:max_images]
        vendor = vendor or ctx_vendor
        vendor_src = "dalla pipeline"
    else:
        base = os.path.join(folder, "image_samples")
        if not os.path.isdir(base):
            base = folder
        paths = sorted(glob.glob(os.path.join(base, pattern)))[:max_images]
    if not paths:
        if from_pipeline:
            print("[error] il contesto della pipeline non contiene frame leggibili")
        else:
            print(f"[error] nessuna immagine con pattern {pattern} in {base}")
        return None
    corr_entry = load_folder_corrections(corrections, folder)
    corr = corr_entry.get("frames") or {}
    corr_x = _corr_anchor_x(corr)
    if not from_pipeline:
        vendor_src = "indicato a mano"
        if not vendor:
            vendor, vendor_src = predict_vendor(paths, device)
    prof = profile_for(vendor)
    print(f"[info] {len(paths)} immagini, vendor='{vendor or 'default'}' ({vendor_src})")

    # ---- 1. orientation, from the marker, to decide which end carries the zero ----
    # The net runs first only because the marker detector takes it as its prior and its rect crop;
    # the marker has the last word, and can overturn it. When the evidence comes from the
    # pipeline this has already happened there, on these same frames.
    if not from_pipeline:
        models = load_orientation_models(device)
        orient = predict_orientation(models, [Path(p) for p in paths]) if models else {}
        marker = marker_orientation(load_marker_detector(), paths, vendor, orient)
    for p, m in marker.items():
        if m.get("vertical") in ("su", "giu"):
            o = orient.setdefault(p, {})
            o["net_label"] = o.get("net_label", "") or o.get("label", "")
            o["label"] = m["vertical"]
            o["source"] = o.get("source") if from_pipeline else (
                "marker" + (" (ha corretto la rete)" if m.get("corrected") == "corrected" else "")
            )
    if orient:
        groups: Dict[str, int] = {}
        for v in orient.values():
            groups[v.get("label", "?")] = groups.get(v.get("label", "?"), 0) + 1
        n_mark = sum(1 for v in orient.values() if str(v.get("source", "")).startswith("marker"))
        print(f"[1] orientamento: {groups} ({n_mark}/{len(orient)} dal marker, il resto dalla rete)")
    else:
        print("[1] orientamento non disponibile: il verso verra' dalle etichette")

    # ---- 2. depth from the interface, to be cross-checked against the ruler later ----
    if not from_pipeline:
        # median rect over the folder, in the left,top,right,bottom order the depth module wants
        rects = [o["rect"] for o in orient.values() if o.get("rect")]
        rect_echo = None
        if rects:
            med = [statistics.median([r[k] for r in rects]) for k in range(4)]
            rect_echo = (med[1], med[0], med[3], med[2])  # (top,left,bottom,right) -> l,t,r,b
        depth_info = (run_depth_module(folder, python_bin, paths, vendor=vendor, rect_echo=rect_echo)
                      if with_depth else {})
    elif not with_depth:
        depth_info = {}

    # ---- 3. the ruler's column: detected per frame, then agreed folder-wide ----
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
        print(f"[3] righello: x={cx:.0f} ±{half:.0f} px, uguale per tutta la cartella "
              f"(trovato su {len(xs)}/{len(first)} frame)")

    # A column the operator corrected consistently replaces the detections' median: it anchors the
    # zone for every frame, including the ones where no ruler was found at all.
    if corr_x is not None:
        if zone:
            print(f"[corr] zona della cartella: x={corr_x:.0f} dalle tue correzioni "
                  f"(era {zone['x']:.0f})")
            zone = {**zone, "x": round(corr_x, 1), "from_corrections": True}
        else:
            zone = {"x": round(corr_x, 1), "half": 60.0, "found": 0, "total": len(first),
                    "from_corrections": True}
            print(f"[corr] zona della cartella: x={corr_x:.0f} dalle tue correzioni "
                  f"(il detector non l'aveva trovata)")

    # ---- second pass: frames without a ruler get told where the folder's zone is ----
    frames: List[dict] = []
    n_applied = 0
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

        # 1. the zero, from the orientation the marker settled: up means zero at the top
        ori = (f["orient"] or {}).get("label", "")
        zero_from_orient = "top" if ori == "su" else ("bottom" if ori == "giu" else "")
        zero_detected = "top" if pred.direction >= 0 else "bottom"
        y_zero = pred.y_zero
        zero_moved = False
        if zero_from_orient and pred.ticks_y and zero_from_orient != zero_detected:
            # trust the orientation for *which end*, keep the ticks for *where*
            y_zero = min(pred.ticks_y) if zero_from_orient == "top" else max(pred.ticks_y)
            zero_moved = True

        # 4. ticks: the zero end and the far end are the two that must not be confused
        dirn = 1 if (zero_from_orient or zero_detected) == "top" else -1
        far = max(pred.ticks_y) if (dirn >= 0 and pred.ticks_y) else (
            min(pred.ticks_y) if pred.ticks_y else None)

        # The operator's corrections for this very frame win over the detection: they are the one
        # piece of ground truth in the loop. Applied here, after detection, so the run still shows
        # what the detector would have said on its own.
        c = corr.get(name) or {}
        applied: List[str] = []
        # The end the zero sits at, when the operator overturned the marker. Applied before the
        # zero itself, because it decides which extreme of the ladder the zero is taken from.
        if c.get("zero_end") in ("top", "bottom"):
            zero_from_orient = str(c["zero_end"])
            if pred.ticks_y:
                y_zero = (min(pred.ticks_y) if zero_from_orient == "top" else max(pred.ticks_y))
            zero_moved = zero_from_orient != zero_detected
            # direction and far end were derived from the old verso a few lines above
            dirn = 1 if zero_from_orient == "top" else -1
            if pred.ticks_y:
                far = max(pred.ticks_y) if dirn >= 0 else min(pred.ticks_y)
            applied.append("verso")
        if c.get("x") not in (None, ""):
            pred.x = float(c["x"]); applied.append("colonna")
        if c.get("y_zero") not in (None, ""):
            y_zero = float(c["y_zero"]); zero_moved = False; applied.append("zero")
        # An explicit ladder replaces the detection outright. It is what the app writes once
        # the operator has fixed one tick: inside a frame the pitch does not change, so the
        # whole ladder follows from one, and sending it as add/delete pairs would only make
        # the same fact harder to read.
        if c.get("ticks"):
            pred.ticks_y = sorted(float(v) for v in c["ticks"])
            pred.n_ticks = len(pred.ticks_y)
            if len(pred.ticks_y) > 1:
                d = sorted(pred.ticks_y[k + 1] - pred.ticks_y[k]
                           for k in range(len(pred.ticks_y) - 1))
                pred.tick_pitch_px = d[len(d) // 2]
            applied.append("tacche")
        elif c.get("ticks_del") or c.get("ticks_add"):
            dele = {round(float(v)) for v in (c.get("ticks_del") or [])}
            keep = [t for t in (pred.ticks_y or []) if round(t) not in dele]
            pred.ticks_y = sorted(keep + [float(v) for v in (c.get("ticks_add") or [])])
            pred.n_ticks = len(pred.ticks_y)
            if len(pred.ticks_y) > 1:
                d = sorted(pred.ticks_y[k + 1] - pred.ticks_y[k]
                           for k in range(len(pred.ticks_y) - 1))
                pred.tick_pitch_px = d[len(d) // 2]
            applied.append("tacche")
        if c.get("y_far") not in (None, ""):
            far = float(c["y_far"]); applied.append("ultima tacca")
        elif "tacche" in applied and pred.ticks_y:
            far = max(pred.ticks_y) if dirn >= 0 else min(pred.ticks_y)
        if applied:
            n_applied += 1

        # 6. the numbers, checked against each other through the constant pitch
        labels_cm = [(y, round(v / 10.0, 2)) for y, v in (pred.labels_all or [])]
        for pair in (c.get("nums") or []):
            try:
                labels_cm.append((float(pair[0]), float(str(pair[1]).replace(",", "."))))
            except (TypeError, ValueError, IndexError):
                continue
        if c.get("nums"):
            labels_cm.sort()
            applied.append("numeri")
            # Numbers the operator read are the calibration: refit mm/px on them through the zero
            if y_zero is not None and len(labels_cm) >= 1:
                num = den = 0.0
                for y, v in labels_cm:
                    if v == 0:
                        continue
                    d = dirn * (y - y_zero)
                    num += v * 10.0 * d
                    den += d * d
                if den > 0.5:
                    pred.mm_per_px = num / den
        coh = check_numbers(labels_cm, pred.tick_pitch_px or 0.0, pred.mm_per_px)

        # 1b. cross-check the zero with the OCR: the smallest number must sit at the distance
        # from the zero that its own value implies
        labels_mm = [(y, v * 10.0) for y, v in labels_cm]
        zchk = check_zero_with_ocr(y_zero, labels_mm, pred.mm_per_px, pred.tick_pitch_px, far)

        # 2b. cross-check the interface depth against the ruler's largest number
        dep = depth_info.get(os.path.realpath(f["path"]))
        if c.get("depth_mm") not in (None, ""):
            try:
                # read by a person off the interface: the strongest kind of evidence there is
                dep = {"depth_mm": float(str(c["depth_mm"]).replace(",", ".")),
                       "from_interface": True, "mode": "letta dall'operatore",
                       "status": "accepted", "ocr_text": ""}
                applied.append("depth")
            except ValueError:
                pass
        scale_depth = None
        depth_verdict = ""
        if pred.mm_per_px and y_zero is not None and far is not None:
            scale_depth = round(abs(far - y_zero) * pred.mm_per_px, 1)
        if dep and scale_depth:
            # Only a depth actually printed in the interface is independent evidence. A derived
            # one is guessed from the scale itself, so comparing the two proves nothing — it was
            # reported as "NON coerente" and counted as verified, which made stage E look broken.
            if not dep["from_interface"]:
                depth_verdict = "non verificabile (depth non scritta nell'interfaccia)"
            else:
                same_ten = round(dep["depth_mm"] / 10.0) == round(scale_depth / 10.0)
                depth_verdict = "coerente" if same_ten else "NON coerente"
        dchk = check_depth_against_scale(
            (dep or {}).get("depth_mm"), bool(dep and dep.get("from_interface")),
            labels_mm, scale_depth)

        s = min(1.0, cap_w / float(f["gray"].shape[1]))
        view = cv2.cvtColor(f["gray"], cv2.COLOR_GRAY2BGR)
        if s < 1.0:
            view = cv2.resize(view, (int(f["gray"].shape[1] * s), int(f["gray"].shape[0] * s)),
                              interpolation=cv2.INTER_AREA)
        frames.append({
            # `path` serves consumers that render the frame themselves (the ESIBuilder app
            # serves it from its own deduplicated mirror) and so do not want the base64 copy.
            "name": name, "idx": idx, "img": _b64(view), "path": f["path"],
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
            "E_depth_mode": (dep or {}).get("mode", ""),
            "E_depth_status": (dep or {}).get("status", ""),
            "E_depth_ocr": (dep or {}).get("ocr_text", ""),
            "E_depth_box": (dep or {}).get("box"),
            "E_depth_score": (dep or {}).get("score"),
            "E_depth_ranker": (dep or {}).get("ranker"),
            "E_depth_reason": (dep or {}).get("reason", ""),
            "rect": list(f["orient"].get("rect") or []) if f.get("orient") else [],
            "E_depth_scale": scale_depth, "E_verdict": depth_verdict,
            "corr_applied": applied,
            "M_marker": marker.get(f["path"]) or {},
            "M_orient_source": (f["orient"] or {}).get("source", "rete"),
            "M_net_label": (f["orient"] or {}).get("net_label", ""),
            "Z_check": zchk, "DE_check": dchk,
        })

    st: Dict[str, int] = {}
    for fr in frames:
        st[fr["status"]] = st.get(fr["status"], 0) + 1
    if n_applied:
        print(f"[corr] applicate le tue correzioni su {n_applied}/{len(frames)} frame")
    zc = _tally(fr["Z_check"]["verdict"] for fr in frames)
    print(f"[1b] zero controverificato con l'OCR: {zc}")
    if depth_info:
        dc = _tally(fr["DE_check"]["verdict"] for fr in frames)
        print(f"[2b] depth d'interfaccia contro il numero piu' grande della scala: {dc}")
    n_t = [len(fr["ticks"]) for fr in frames]
    if n_t:
        print(f"[4] tacche per frame: da {min(n_t)} a {max(n_t)}, "
              f"mediana {int(statistics.median(n_t))}")
    pit = [fr["pitch"] for fr in frames if fr["pitch"]]
    if pit:
        print(f"[5] passo fra le tacche: mediana {statistics.median(pit):.1f} px "
              f"(da {min(pit):.1f} a {max(pit):.1f})")
    print(f"[6] stati: {st}")
    n_sus = sum(1 for fr in frames if fr["D_suspect"])
    print(f"[6] frame con numeri sospetti: {n_sus}/{len(frames)}")
    if depth_info:
        agree = sum(1 for fr in frames if fr["E_verdict"].startswith("coerente"))
        checked = sum(1 for fr in frames if fr["E_verdict"] in ("coerente", "NON coerente"))
        if checked:
            print(f"[E] depth coerente su {agree}/{checked} frame verificabili")
        else:
            print(f"[E] nessun frame verificabile: la depth non e' scritta "
                  f"nell'interfaccia di questi {len(frames)} frame")
    return {"folder": os.path.basename(folder.rstrip("/\\")), "vendor": vendor or "default",
            "zone": zone, "orient_groups": {k: sum(1 for v in orient.values() if v["label"] == k)
                                            for k in {v["label"] for v in orient.values()}} if orient else {},
            "status": st, "frames": frames}


def predict_vendor(paths: Sequence[str], device_name: str = "auto") -> Tuple[str, str]:
    """Vendor from the pixels, by majority over the folder's frames.

    Reading it from the folder name works on the prepared corpus and breaks on raw material,
    where the name carries nothing — exactly the case this tool exists to study. The production
    vendor classifier (test acc 0.9824) is the honest source; the name stays as a last resort
    and says so out loud.
    """
    from scale.detect_scale_ladder import PROFILES
    known = {v.lower(): v for v in PROFILES if v != "default"}
    # The classifier's vocabulary is wider than the scale profiles: names that mean the same
    # machine family must map onto the profile, or a confident answer gets thrown away. Measured
    # on the corpus: 'Toshiba' at 0.97 was being discarded and the Canon profile lost.
    known.update({"toshiba": "Canon", "aloka": "Hitachi", "fujifilm": "Hitachi"})
    # Below this the answer is noise: 'Mindray 0.20' on an Alpinion and 'Hitachi 0.19' on an
    # Alpio were both accepted as fact. The default profile is the honest answer instead.
    MIN_CONF = 0.35
    try:
        import importlib.util
        import numpy as np
        import torch
        from PIL import Image

        ck_path = (REPO / "artifacts/10_active_pipeline/pipeline_fss_head/models"
                        / "vendor_training_no_negative_v2_power/best_model.pt")
        if not ck_path.exists():
            raise FileNotFoundError(ck_path)
        for extra in (REPO, REPO / "tools" / "ultrasound"):
            if str(extra) not in sys.path:
                sys.path.insert(0, str(extra))
        spec = importlib.util.spec_from_file_location(
            "vend_mod", REPO / "tools/ultrasound/train_ultrasound_vendor_classifier.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["vend_mod"] = mod
        spec.loader.exec_module(mod)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        classes = [str(c) for c in ck["class_names"]]
        size = int(ck.get("args", {}).get("image_size", 320))
        dev = torch.device("mps" if (device_name in ("auto", "mps")
                                     and torch.backends.mps.is_available()) else "cpu")
        net = mod.VendorClassifier(num_classes=len(classes), pretrained=False).to(dev)
        net.load_state_dict(ck["model_state_dict"])
        net.eval()
        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std = np.array([0.229, 0.224, 0.225], dtype="float32")
        votes: Dict[str, float] = {}
        with torch.no_grad():
            for p in paths[:10]:
                im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
                a = (np.asarray(im, dtype="float32") / 255.0 - mean) / std
                t = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).to(dev)
                pr = torch.softmax(net(t)[0], 0).cpu().numpy()
                i = int(pr.argmax())
                votes[classes[i]] = votes.get(classes[i], 0.0) + float(pr[i])
        if votes:
            best = max(votes, key=lambda k: votes[k])
            n = len(paths[:10])
            conf = votes[best] / n
            src = f"dalla rete: {best}, {conf:.2f} su {n} frame"
            if conf < MIN_CONF:
                return "", src + " — troppo incerto, uso il profilo generico"
            mapped = known.get(best.lower(), "")
            if not mapped:
                return "", src + " — nessun profilo scala per questo vendor"
            return mapped, src
    except Exception as exc:  # noqa: BLE001 - never block the study on the classifier
        print(f"[vendor] rete non disponibile ({exc}); ricado sul nome della cartella")
    guessed = _guess_vendor(folder_of(paths[0]))
    return guessed, "dal nome della cartella (non affidabile su immagini grezze)"


def folder_of(path: str) -> str:
    return os.path.dirname(path)


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


def pick_folder(start: str = "/Volumes") -> str:
    """Ask the operator for a folder with the system dialog, so no path has to be typed."""
    if sys.platform == "darwin":
        script = (f'POSIX path of (choose folder with prompt '
                  f'"Scegli la cartella da studiare" default location POSIX file "{start}")')
        try:
            p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=300)
            if p.returncode == 0 and p.stdout.strip():
                return p.stdout.strip().rstrip("/")
            return ""  # cancelled
        except Exception as exc:  # noqa: BLE001
            print(f"[pick] finestra non disponibile: {exc}")
    # anywhere else: type or paste the path
    try:
        return input("Percorso della cartella: ").strip().rstrip("/")
    except EOFError:
        return ""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", default="", help="Omit it (or pass --pick) to choose from a dialog.")
    ap.add_argument("--pick", action="store_true", help="Always ask, even if --folder is given.")
    ap.add_argument("--open", dest="open_after", action="store_true",
                    help="Open the page in the browser when it is ready.")
    ap.add_argument("--pattern", default="image_depth_value_setup_*.png")
    ap.add_argument("--vendor", default="")
    ap.add_argument("--corrections", default="",
                    help="JSON with the operator's corrections, fed back into this run")
    ap.add_argument("--max-images", type=int, default=14)
    ap.add_argument("--only-file", type=Path, default=None,
                    help="JSON list of frame file names to study, instead of the first "
                         "--max-images of the folder.")
    ap.add_argument("--cap-width", type=int, default=1400)
    ap.add_argument("--no-depth", action="store_true", help="Skip the stage-E depth cross-check.")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--high-recall", action="store_true")
    ap.add_argument("--from-pipeline", type=Path, default=None,
                    help="Cartella dello stadio scala della pipeline (quella con "
                         "pipeline_context.json): la pagina mostra le predizioni della pipeline "
                         "invece di ricalcolarle, e le tue correzioni valgono su quelle.")
    ap.add_argument("--data-json", type=Path, default=None,
                    help="Also write the study data as JSON, without the embedded frame "
                         "images. For callers that render the frames themselves.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Default: artifacts/50_scale_study/<cartella>.html")
    args = ap.parse_args(argv)

    if args.high_recall:
        os.environ["SCALE_HIGH_RECALL"] = "1"

    pipeline_context = None
    if args.from_pipeline is not None:
        ctx_path = args.from_pipeline
        if ctx_path.is_dir():
            ctx_path = ctx_path / "pipeline_context.json"
        if not ctx_path.is_file():
            print(f"[error] contesto della pipeline non trovato: {ctx_path}")
            return 2
        pipeline_context = json.loads(ctx_path.read_text(encoding="utf-8"))
        if not args.folder:
            args.folder = str(pipeline_context.get("folder_path") or "")
        print(f"[info] evidenze dalla pipeline: {ctx_path}")

    folder = args.folder
    if args.pick or (not folder and pipeline_context is None):
        folder = pick_folder()
        if not folder:
            print("[info] nessuna cartella scelta")
            return 1
        print(f"[info] cartella scelta: {folder}")
    args.folder = folder
    if args.out is None:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(folder.rstrip("/\\")))[:60]
        args.out = REPO / "artifacts" / "50_scale_study" / f"{safe or 'studio'}.html"

    solo = None
    if args.only_file is not None:
        if not args.only_file.is_file():
            print(f"[error] elenco dei fotogrammi non trovato: {args.only_file}")
            return 2
        solo = json.loads(args.only_file.read_text(encoding="utf-8"))
        print(f"[info] studio limitato a {len(solo)} fotogrammi")

    data = study(args.folder, args.pattern, args.max_images, args.cap_width, args.vendor,
                 not args.no_depth, args.python_bin, args.device, args.corrections,
                 pipeline_context=pipeline_context, only=solo)
    if not data:
        return 2
    if args.data_json is not None:
        # Without the base64 frames: they are what makes the HTML weigh megabytes, and a
        # caller that has the images already does not need a second copy of them.
        magro = {**data, "frames": [{k: v for k, v in f.items() if k != "img"}
                                    for f in data.get("frames") or []]}
        args.data_json.parent.mkdir(parents=True, exist_ok=True)
        args.data_json.write_text(json.dumps(magro, ensure_ascii=False), encoding="utf-8")
        print(f"[ok] dati dello studio -> {args.data_json}")
    tpl = PAGE.read_text(encoding="utf-8")
    html = tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"[ok] {len(data['frames'])} frame -> {args.out} ({len(html)/1024/1024:.1f} MB)")
    if args.open_after and sys.platform == "darwin":
        subprocess.run(["open", str(args.out)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
