"""The scale study as a pipeline stage: same reasoning, evidence supplied instead of recomputed.

``study_scale_folder.py`` is the reference: it studies one folder in six stages and lets an
operator correct every one of them. To reach those stages it has to reconstruct the evidence
by itself — it loads the vendor classifier, the rect regressor, the up/down net, the
orientation-marker bundle and it runs the depth module as a subprocess.

Inside the official pipeline all of that already exists, computed once, on the same frames:

  ============================  ==========================================================
  stage of the study            where the pipeline already has it
  ============================  ==========================================================
  1  orientation (zero end)     ``su_giu_pred`` of the LR-marker rows, which is *already*
                                the marker's verdict over the net's prior, with
                                ``bundle_vertical_correction`` saying when it overturned it
  2  depth from the interface   the RECT_DEPTH autonomous stage, per image, with its own
                                ``mode``/``status``/box, so its evidence travels with it
  -  vendor                     the vendor classifier (mean softmax over the folder)
  -  the ultrasound rectangle   line #11, and the per-image rect of the marker rows
  3  ruler column               nothing upstream: this is the scale's own job
  4  ticks / 5 pitch / 6 numbers  ditto
  ============================  ==========================================================

So this module keeps the *reasoning* of the study and drops its *recomputation*: it receives
the evidence in a context JSON and does only what is genuinely the scale's work. Nothing here
re-runs a net or the depth module, which is what makes it cheap enough to run on every folder
of a batch, and what guarantees the scale and the rest of the pipeline never disagree about
the same frame.

What this stage adds on top of the study
----------------------------------------
The study answers per *frame*. Line #21 is per *depth*: ``#17`` defines the depth groups and
``#18``-``#22`` all carry one entry each, in that order. The bridge is the depth the RECT_DEPTH
stage read on each frame:

    frames -> group by depth value -> order the groups by ascending depth -> depth_index

which is exactly the ordering the legacy files use (checked on the shipped templates: ``#18``
``20|25|...|100`` ascending, with ``#19``/``#20`` ascending alongside). That ordering is not a
convenience: ``consolidate_scale_setup`` fits ``mm_per_px`` with an isotonic (non-decreasing)
regression along the depth index, so a wrong order would fight the one invariant the consensus
is built on.

Per-depth aggregation is deliberately conservative. Several frames of one depth are the same
machine state, so their rulers should agree; when they do not, that disagreement is used only
to *demote* the depth to weak evidence, never to invent a new value by averaging. Inventing an
acceptance rule without measuring it is precisely what regressed this block three times
(see the scale docs): the frame with the strongest evidence carries the depth, the rest are
recorded as corroboration.

Rotation is refused, not guessed: the detector looks for a vertical ladder of horizontal
dashes, so on a folder the pipeline normalised by 90/180/270 degrees the ruler is not where
(or how) it looks. Such a folder comes back ``rotation_not_supported`` instead of a number.

Output (in ``--output-dir``)
---------------------------
* ``scale_per_image.csv``  one row per studied frame: column, zero, far end, mm/px, pitch,
  ticks, numbers, the three cross-checks, and which evidence drove the zero end
* ``scale_per_depth.csv``  one row per depth index: the consolidated answer and its source
  (``detected`` / ``interpolated`` / ``none``), plus how many frames voted for it
* ``scale_lines.json``     the ``.fss`` rows #18/#19/#20/#21 implied by the above
* ``summary.json``         counters, folder-wide ruler zone, and the review reasons

Usage (normally called by the pipeline, but runnable by hand on a context JSON):
  OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/predict_scale_from_pipeline.py \
    --context-json <run>/scale/0001_folder/pipeline_context.json \
    --output-dir   <run>/scale/0001_folder
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "scale"))

from scale.consolidate_scale_setup import ScaleCandidate, consolidate_setup  # noqa: E402
from scale.detect_scale_ladder import PROFILES, detect_scale, load_image  # noqa: E402
from scale.study_scale_folder import (  # noqa: E402  - the cross-checks, verbatim
    _corr_anchor_x,
    check_depth_against_scale,
    check_numbers,
    check_zero_with_ocr,
    load_folder_corrections,
)

# Below this the vendor answer is noise and the generic profile is the honest choice. Same
# threshold as the study (``predict_vendor``), which measured 'Mindray 0.20' on an Alpinion.
VENDOR_MIN_CONF = 0.35
# The classifier's vocabulary is wider than the scale profiles: these names mean the same
# machine family, and dropping them would throw away a confident answer.
VENDOR_PROFILE_ALIASES = {"toshiba": "Canon", "aloka": "Hitachi", "fujifilm": "Hitachi"}
# Two frames of the same depth whose calibrations differ by more than this do not agree.
# Wider than the frame-to-frame OCR jitter, far narrower than the failures worth catching.
GROUP_AGREEMENT_TOL = 0.10
# The ``tick`` field of line #21, in cm. Constant in the legacy corpus (128/128 groups of the
# ten shipped .fss) and hardcoded the same way in ``ScalePrediction.fss_segment``.
TICK_CM_LEGACY = 0.5

PER_IMAGE_FIELDS = [
    "image_id", "image_path", "depth_mm", "depth_index", "status", "reason",
    "x", "y_zero", "y_far", "mm_per_px", "tick_pitch_px", "tick_step_mm", "n_ticks",
    "direction", "zero_end", "zero_end_source", "zero_moved", "label_side",
    "n_labels", "labels_json", "used_folder_zone", "calib_source", "out_of_band",
    "numbers_step_mm", "numbers_suspect_json", "zero_check", "zero_check_detail",
    "depth_check", "depth_check_detail", "depth_from_interface", "depth_mode",
    "sugiu", "sugiu_source", "marker_score", "confidence",
]
PER_DEPTH_FIELDS = [
    "depth_index", "depth_mm", "status", "source", "x", "y_zero", "mm_per_px", "direction",
    "confidence", "length_mm", "tick_cm", "tick_step_mm_detected", "side", "y_far",
    "frames_total", "frames_usable", "frames_agree", "group_spread", "weak_anchor",
    "best_image_id", "notes",
]


# --------------------------------------------------------------------------- #
# context
# --------------------------------------------------------------------------- #
def _f(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # drop NaN


def profile_name_for_vendor(vendor: str, confidence: Optional[float]) -> Tuple[str, str]:
    """``(profile name, why)``. Empty name = the generic profile, said out loud."""
    if not vendor:
        return "", "nessun vendor dalla pipeline: profilo generico"
    if confidence is not None and confidence < VENDOR_MIN_CONF:
        return "", f"vendor '{vendor}' a {confidence:.2f}: troppo incerto, profilo generico"
    known = {v.lower(): v for v in PROFILES if v != "default"}
    known.update(VENDOR_PROFILE_ALIASES)
    mapped = known.get(vendor.strip().lower(), "")
    if not mapped:
        return "", f"nessun profilo scala per '{vendor}': profilo generico"
    return mapped, f"vendor '{vendor}' dalla pipeline" + (
        f" ({confidence:.2f})" if confidence is not None else ""
    )


def rect_for_detector(
    rect_ltrb: Optional[Sequence[float]],
) -> Optional[Tuple[int, int, int, int]]:
    """The rect in the shape ``detect_scale`` is called with by the reference study.

    ``study_scale_folder`` passes ``(left, top, width, height)`` — that is the behaviour every
    number Camilla has looked at was produced with, so it is what this stage reproduces, and
    the conversion lives in one named place instead of being open-coded.

    Be warned that the convention is *not* settled inside the detector: ``_bands_for`` reads
    index 2 as the right edge (i.e. it wants ``x1,y1,x2,y2``) while ``_out_of_band`` reads
    ``rect[0] + rect[2]`` as the right edge (i.e. it wants ``x,y,w,h``), and the eval harness
    feeds it a third thing again. Resolving that needs the chain eval on the SSD corpus to
    prove which one wins; until then, changing it here would silently move predictions.
    """
    if not rect_ltrb or len(rect_ltrb) < 4:
        return None
    left, top, right, bottom = (float(v) for v in rect_ltrb[:4])
    if right <= left or bottom <= top:
        return None
    return (int(left), int(top), int(right - left), int(bottom - top))


def parse_line11_ltrb(text: str) -> Optional[Tuple[float, float, float, float]]:
    """``#11`` is ``TOP|LEFT|BOTTOM|RIGHT|`` -> ``(left, top, right, bottom)``.

    Verified on the ten .fss shipped with the legacy app: the fourth token exceeds
    ``VIDEO_Y_SIZE`` on 8 of them, so it can only be an X. (``scale_common.parse_rect_echo``
    reads the same row as ``x1|y1|x2|y2``; that is wrong, and it is why the audit reports the
    ruler as sitting outside the rectangle. It is used only by the GT audit, never here.)
    """
    toks = [t for t in str(text or "").strip().split("|") if t.strip()]
    if len(toks) < 4:
        return None
    try:
        top, left, bottom, right = (float(t) for t in toks[:4])
    except ValueError:
        return None
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


# --------------------------------------------------------------------------- #
# stage 1-2: the evidence the pipeline already has
# --------------------------------------------------------------------------- #
def zero_end_from_orientation(sugiu: str) -> str:
    """``su`` = the zero is at the top, ``giu`` = at the bottom. Anything else: no opinion."""
    label = str(sugiu or "").strip().lower()
    if label in ("su", "up", "top"):
        return "top"
    if label in ("giu", "giù", "down", "bottom"):
        return "bottom"
    return ""


def depth_groups(frames: Sequence[dict], snap_mm: float = 1.0) -> Tuple[Dict[str, int], List[float]]:
    """``({image_path: depth_index}, [depth_mm per index])``, ordered by ascending depth.

    Ascending is not cosmetic: the consensus fits ``mm_per_px`` with an isotonic regression
    along the index, so this order is the invariant it leans on. Values are snapped to the
    millimetre because they come from OCR, where 39.9 and 40.0 are the same depth setting.
    """
    by_value: Dict[float, List[str]] = {}
    for fr in frames:
        depth = _f(fr.get("depth_mm"))
        if depth is None or depth <= 0:
            continue
        key = round(depth / snap_mm) * snap_mm
        by_value.setdefault(float(key), []).append(str(fr.get("image_path", "")))
    ordered = sorted(by_value)
    index_of: Dict[str, int] = {}
    for idx, value in enumerate(ordered):
        for path in by_value[value]:
            index_of[path] = idx
    return index_of, ordered


# --------------------------------------------------------------------------- #
# stage 3-6: the scale's own work
# --------------------------------------------------------------------------- #
def order_frames_for_study(frames: Sequence[dict], index_of: Dict[str, int]) -> List[dict]:
    """Frames with a depth first, taken round-robin so every depth is reached early.

    ``--max-frames`` cuts this list, and the cut must not decide *which depths* get an answer.
    A folder can hold hundreds of frames while the depth stage read only some of them, so
    plain listing order would happily spend the whole budget on frames that carry no depth at
    all — and on the depths that happen to come first. Depth-less frames still travel at the
    end: they cannot fill a #21 entry, but they can help agree the ruler's column.
    """
    by_index: Dict[int, List[dict]] = {}
    without: List[dict] = []
    for frame in frames:
        idx = index_of.get(str(frame.get("image_path", "")))
        if idx is None:
            without.append(frame)
        else:
            by_index.setdefault(idx, []).append(frame)
    ordered: List[dict] = []
    round_no = 0
    while any(len(group) > round_no for group in by_index.values()):
        for idx in sorted(by_index):
            group = by_index[idx]
            if len(group) > round_no:
                ordered.append(group[round_no])
        round_no += 1
    return ordered + without


def study_frames(
    frames: Sequence[dict],
    profile_name: str,
    max_frames: int,
    corrected_x: Optional[float] = None,
) -> Tuple[List[dict], Optional[dict]]:
    """Detect per frame, agree the column folder-wide, then retry the frames that failed.

    The two passes are the study's stage 3: a frame that cannot find the ruler on its own is
    told where the rest of the folder found it, which is the difference between a hole and a
    reviewable proposal.
    """
    first: List[dict] = []
    for fr in list(frames)[:max_frames]:
        path = str(fr.get("image_path", ""))
        image = load_image(path)
        if image is None:
            continue
        import cv2  # local: the module must import even where cv2 is missing

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        rect = rect_for_detector(fr.get("rect_ltrb"))
        pred = detect_scale(gray, rect=rect, vendor=profile_name)
        first.append({"frame": fr, "gray": gray, "rect": rect, "pred": pred})

    xs = [f["pred"].x for f in first if f["pred"].x is not None]
    zone: Optional[dict] = None
    if xs:
        centre = float(statistics.median(xs))
        spread = float(statistics.median([abs(x - centre) for x in xs])) if len(xs) > 1 else 0.0
        zone = {
            "x": round(centre, 1),
            "half": round(max(40.0, 3.0 * spread + 25.0), 1),
            "found": len(xs),
            "total": len(first),
        }
    # A column the operator corrected consistently across frames replaces the detections'
    # median: it anchors the zone for *every* frame, including the ones where no ruler was
    # found at all. Same rule as the reference study, and the only correction this stage
    # applies by itself — the per-frame ones live in the review page, where they are visible.
    if corrected_x is not None:
        zone = ({**zone, "x": round(corrected_x, 1), "from_corrections": True} if zone else
                {"x": round(corrected_x, 1), "half": 60.0, "found": 0, "total": len(first),
                 "from_corrections": True})

    out: List[dict] = []
    for item in first:
        pred = item["pred"]
        used_zone = False
        if zone and (not pred.ok or pred.mm_per_px is None):
            retry = detect_scale(
                item["gray"], rect=item["rect"], vendor=profile_name,
                prior_x=zone["x"], prior_band_px=int(zone["half"]),
            )
            if retry.n_ticks and (retry.mm_per_px or not pred.n_ticks):
                pred, used_zone = retry, True
        item["pred"] = pred
        item["used_zone"] = used_zone
        item.pop("gray", None)  # the frames are not kept in memory past detection
        out.append(item)
    return out, zone


def describe_frame(item: dict, depth_index: Optional[int]) -> dict:
    """One per-image row: the prediction, the zero the orientation settled, the cross-checks."""
    fr, pred = item["frame"], item["pred"]
    path = str(fr.get("image_path", ""))

    # stage 1: the zero end comes from the orientation the marker already settled upstream;
    # the ticks say *where* that end is. Same rule as the study, minus the recomputation.
    zero_from_orient = zero_end_from_orientation(fr.get("sugiu"))
    zero_detected = "top" if pred.direction >= 0 else "bottom"
    y_zero = pred.y_zero
    zero_moved = False
    if zero_from_orient and pred.ticks_y and zero_from_orient != zero_detected:
        y_zero = min(pred.ticks_y) if zero_from_orient == "top" else max(pred.ticks_y)
        zero_moved = True
    zero_end = zero_from_orient or zero_detected
    direction = 1 if zero_end == "top" else -1
    far = None
    if pred.ticks_y:
        far = max(pred.ticks_y) if direction >= 0 else min(pred.ticks_y)

    # stage 6: the numbers checked against each other through the constant pitch
    labels_cm = [(y, round(v / 10.0, 2)) for y, v in (pred.labels_all or [])]
    coherence = check_numbers(labels_cm, pred.tick_pitch_px or 0.0, pred.mm_per_px)
    labels_mm = [(y, v * 10.0) for y, v in labels_cm]
    zero_check = check_zero_with_ocr(y_zero, labels_mm, pred.mm_per_px, pred.tick_pitch_px, far)

    # stage 2: the interface depth against the deepest number and the last tick
    scale_depth = None
    if pred.mm_per_px and y_zero is not None and far is not None:
        scale_depth = round(abs(far - y_zero) * pred.mm_per_px, 1)
    depth_check = check_depth_against_scale(
        _f(fr.get("depth_mm")), bool(fr.get("depth_from_interface")), labels_mm, scale_depth,
    )

    return {
        "image_id": os.path.basename(path),
        "image_path": path,
        "depth_mm": _f(fr.get("depth_mm")),
        "depth_index": "" if depth_index is None else depth_index,
        "status": pred.status,
        "reason": pred.reason,
        "x": None if pred.x is None else round(pred.x, 1),
        "y_zero": None if y_zero is None else round(y_zero, 1),
        "y_far": None if far is None else round(far, 1),
        "mm_per_px": None if pred.mm_per_px is None else round(pred.mm_per_px, 6),
        "tick_pitch_px": None if pred.tick_pitch_px is None else round(pred.tick_pitch_px, 2),
        "tick_step_mm": pred.tick_step_mm,
        "n_ticks": pred.n_ticks,
        "direction": direction,
        "zero_end": zero_end,
        "zero_end_source": "marker/rete" if zero_from_orient else "etichette della scala",
        "zero_moved": int(zero_moved),
        "label_side": pred.label_side or "",
        "n_labels": len(pred.labels or []),
        "labels_json": json.dumps([[round(y, 1), v] for y, v in labels_cm], separators=(",", ":")),
        "used_folder_zone": int(bool(item.get("used_zone"))),
        "calib_source": str(pred.debug.get("calib_source", "labels")),
        "out_of_band": int(bool(pred.debug.get("out_of_band"))),
        "numbers_step_mm": coherence["step_mm"],
        "numbers_suspect_json": json.dumps(coherence["suspect"], separators=(",", ":")),
        "zero_check": zero_check["verdict"],
        "zero_check_detail": zero_check["detail"],
        "depth_check": depth_check["verdict"],
        "depth_check_detail": depth_check["detail"],
        "depth_from_interface": int(bool(fr.get("depth_from_interface"))),
        "depth_mode": str(fr.get("depth_mode", "") or ""),
        "sugiu": str(fr.get("sugiu", "") or ""),
        "sugiu_source": str(fr.get("sugiu_source", "") or ""),
        "marker_score": _f(fr.get("marker_score")),
        "confidence": round(float(pred.confidence), 3),
    }


# --------------------------------------------------------------------------- #
# per-depth aggregation and consensus
# --------------------------------------------------------------------------- #
def build_candidates(
    items: Sequence[dict],
    rows_by_path: Dict[str, dict],
    index_of: Dict[str, int],
) -> Tuple[List[ScaleCandidate], Dict[int, dict]]:
    """One candidate per depth index: the frame with the strongest evidence carries it.

    The other frames of the same depth are corroboration, not votes to be averaged: they can
    only *demote* the depth to a weak anchor when they disagree with the chosen frame. Weak
    anchors still fill coverage and can be corrected, but they never define the folder trend
    and are never auto-accepted — the policy that fixed this block every time it regressed.
    """
    per_index: Dict[int, List[dict]] = {}
    for item in items:
        path = str(item["frame"].get("image_path", ""))
        idx = index_of.get(path)
        if idx is None:
            continue
        per_index.setdefault(idx, []).append(item)

    rank = {"accepted": 2, "review": 1}
    candidates: List[ScaleCandidate] = []
    evidence: Dict[int, dict] = {}
    for idx, group in sorted(per_index.items()):
        usable = [it for it in group if it["pred"].ok and it["pred"].mm_per_px]
        best = max(
            group,
            key=lambda it: (
                rank.get(it["pred"].status, 0),
                1 if (it["pred"].ok and it["pred"].mm_per_px) else 0,
                float(it["pred"].confidence),
            ),
        )
        pred = best["pred"]
        row = rows_by_path.get(str(best["frame"].get("image_path", "")), {})

        spread = 0.0
        agree = len(usable)
        if pred.mm_per_px and len(usable) > 1:
            ratios = [
                abs(float(it["pred"].mm_per_px) / float(pred.mm_per_px) - 1.0) for it in usable
            ]
            spread = round(max(ratios), 4)
            agree = sum(1 for r in ratios if r <= GROUP_AGREEMENT_TOL)
        group_disagrees = bool(len(usable) > 1 and spread > GROUP_AGREEMENT_TOL and agree < 2)

        weak = bool(
            pred.debug.get("calib_source") == "geometry"
            or pred.debug.get("out_of_band")
            or group_disagrees
        )
        candidates.append(
            ScaleCandidate(
                depth_index=idx,
                x=pred.x,
                y_zero=_f(row.get("y_zero")),
                mm_per_px=pred.mm_per_px,
                # the direction the orientation settled, not the detector's raw guess
                direction=int(row.get("direction", pred.direction) or pred.direction),
                confidence=float(pred.confidence),
                status=pred.status,
                n_labels=len(pred.labels or []),
                weak_anchor=weak,
            )
        )
        evidence[idx] = {
            "frames_total": len(group),
            "frames_usable": len(usable),
            "frames_agree": agree,
            "group_spread": spread,
            "weak_anchor": int(weak),
            "best_image_id": row.get("image_id", ""),
            "tick_step_mm": pred.tick_step_mm,
            "label_side": pred.label_side or "",
            "y_far": row.get("y_far"),
            "span_ticks": list(pred.ticks_y or []),
        }
    return candidates, evidence


def fss_rows(
    consensus: Sequence[object],
    depths_mm: Sequence[float],
    evidence: Dict[int, dict],
) -> Tuple[Dict[str, str], List[dict]]:
    """The ``.fss`` rows #18/#19/#20/#21, and the per-depth table behind them.

    #21 is rebuilt from the *consolidated* geometry, so the far end is recomputed after the
    consensus may have corrected ``mm_per_px``. Its far end is snapped to the tick grid for
    the same reason the detector does it: the segment must land on a tick the scanner drew,
    not half a tick short of it.

    ``tick`` is written in centimetres and ``length_mm`` in millimetres — the legacy
    inconsistency, kept because the reader depends on it. ``side`` is the label side
    (-1 = left, +1 = right) taken from where the numbers were actually read.

    ``tick`` is the legacy constant 0.5, as ``ScalePrediction.fss_segment`` writes it: all 128
    scale groups of the ten .fss shipped with the legacy app use 0.5, and the detector cannot
    tell 0.5 cm ticks from every-other-tick anyway (the binary multiple ambiguity documented
    in ``consolidate_scale_setup``). The step it did measure travels beside it as
    ``tick_step_mm_detected`` so the ambiguity stays visible instead of being written into the
    row. The step still drives the far-end snapping, where it is the right quantity.
    """
    per_depth: List[dict] = []
    segments: List[str] = []
    ratios: List[str] = []
    depth_values: List[str] = []
    for cons in consensus:
        idx = int(getattr(cons, "depth_index"))
        ev = evidence.get(idx, {})
        depth_mm = depths_mm[idx] if 0 <= idx < len(depths_mm) else None
        mm_per_px = _f(getattr(cons, "mm_per_px"))
        x = _f(getattr(cons, "x"))
        y_zero = _f(getattr(cons, "y_zero"))
        direction = int(getattr(cons, "direction") or 1)
        step_mm = _f(ev.get("tick_step_mm")) or 5.0
        side = -1 if str(ev.get("label_side", "")) != "right" else 1

        length_mm: Optional[float] = None
        y_far: Optional[float] = None
        if mm_per_px and y_zero is not None and x is not None:
            ticks = [t for t in (ev.get("span_ticks") or [])]
            span_px = max((abs(t - y_zero) for t in ticks), default=0.0)
            span_mm = span_px * mm_per_px
            if span_mm > 0:
                length_mm = round(max(1, round(span_mm / step_mm)) * step_mm, 3)
                y_far = y_zero + direction * length_mm / mm_per_px

        row = {
            "depth_index": idx,
            "depth_mm": depth_mm,
            "status": str(getattr(cons, "status")),
            "source": str(getattr(cons, "source")),
            "x": None if x is None else round(x, 1),
            "y_zero": None if y_zero is None else round(y_zero, 1),
            "mm_per_px": None if mm_per_px is None else round(mm_per_px, 6),
            "direction": direction,
            "confidence": round(float(getattr(cons, "confidence") or 0.0), 3),
            "length_mm": length_mm,
            "tick_cm": TICK_CM_LEGACY,
            "tick_step_mm_detected": ev.get("tick_step_mm", ""),
            "side": side,
            "y_far": None if y_far is None else round(y_far, 1),
            "notes": ";".join(getattr(cons, "notes") or []),
        }
        row.update({k: ev.get(k, "") for k in
                    ("frames_total", "frames_usable", "frames_agree", "group_spread",
                     "weak_anchor", "best_image_id")})
        per_depth.append(row)

        if depth_mm is not None:
            depth_values.append(f"{depth_mm:g}")
        ratios.append("" if mm_per_px is None else f"{mm_per_px:g}")
        if length_mm is not None and x is not None and y_zero is not None and y_far is not None:
            segments.append(
                f"{int(round(x))}|{int(round(x))}|{int(round(y_zero))}|{int(round(y_far))}"
                f"|{length_mm:g}|{TICK_CM_LEGACY:g}|{side}|"
            )
        else:
            segments.append("")

    complete = bool(segments) and all(segments) and all(ratios)
    lines = {
        # A row is emitted only when every depth has an answer: a #21 with a hole in the
        # middle would silently shift the pairing with #17/#18 in the legacy reader.
        "line_18_vect_depth": ("|".join(depth_values) + "|") if len(depth_values) == len(per_depth) and depth_values else "",
        "line_19_pixel_ratio_x": ("|".join(ratios) + "|") if complete else "",
        "line_20_pixel_ratio_y": ("|".join(ratios) + "|") if complete else "",
        "line_21_scale_line": (";".join(segments) + ";") if complete else "",
        "line_21_complete": "1" if complete else "0",
    }
    return lines, per_depth


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def run(context: dict, output_dir: Path, max_frames: int, corrections: str = "") -> dict:
    folder = str(context.get("folder_path", ""))
    frames = [f for f in (context.get("frames") or []) if f.get("image_path")]
    rotation = int(_f(context.get("rotation_deg_clockwise")) or 0)
    vendor = str(context.get("vendor_predicted", "") or "")
    vendor_conf = _f(context.get("vendor_confidence"))
    out: Dict[str, object] = {
        "folder_path": folder,
        "frames_in_context": len(frames),
        "frames_studied": 0,
        "status": "review",
        "source": "not_run",
        "review_reasons": [],
        "line_18_vect_depth": "",
        "line_19_pixel_ratio_x": "",
        "line_20_pixel_ratio_y": "",
        "line_21_scale_line": "",
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    if rotation % 360 != 0:
        out["source"] = "rotation_not_supported"
        out["review_reasons"] = ["scale_rotation_not_supported"]
        return out
    if not frames:
        out["source"] = "no_frames"
        out["review_reasons"] = ["scale_no_frames"]
        return out

    profile_name, profile_why = profile_name_for_vendor(vendor, vendor_conf)
    out["profile"] = profile_name or "default"
    out["profile_reason"] = profile_why
    print(f"[scala] {len(frames)} frame, profilo '{profile_name or 'default'}' ({profile_why})",
          flush=True)

    # The rect travels per frame (the marker rows carry it); line #11 is the folder-wide
    # fallback for frames the marker never saw.
    line11 = parse_line11_ltrb(str(context.get("line_11_rect_echo", "")))
    for fr in frames:
        if not fr.get("rect_ltrb") and line11:
            fr["rect_ltrb"] = list(line11)

    index_of, depths_mm = depth_groups(frames)
    if not depths_mm:
        out["review_reasons"] = ["scale_no_depth_groups"]
    print(f"[scala] depth riconosciute: {len(depths_mm)} "
          f"({', '.join(f'{d:g}' for d in depths_mm) or 'nessuna'})", flush=True)

    corrected_x = None
    if corrections:
        corr_frames = (load_folder_corrections(corrections, folder) or {}).get("frames") or {}
        corrected_x = _corr_anchor_x(corr_frames)
        out["corrections_frames"] = len(corr_frames)
        out["corrections_applied"] = ["colonna"] if corrected_x is not None else []

    items, zone = study_frames(
        order_frames_for_study(frames, index_of), profile_name, max_frames,
        corrected_x=corrected_x,
    )
    out["frames_studied"] = len(items)
    out["zone"] = zone
    if zone:
        print(f"[scala] righello: x={zone['x']:.0f} ±{zone['half']:.0f} px "
              f"(trovato su {zone['found']}/{zone['total']} frame)", flush=True)

    rows = []
    rows_by_path: Dict[str, dict] = {}
    for item in items:
        path = str(item["frame"].get("image_path", ""))
        row = describe_frame(item, index_of.get(path))
        rows.append(row)
        rows_by_path[path] = row
    _write_csv(output_dir / "scale_per_image.csv", rows, PER_IMAGE_FIELDS)

    candidates, evidence = build_candidates(items, rows_by_path, index_of)
    consensus = consolidate_setup(
        candidates, depth_indices=list(range(len(depths_mm))),
    ) if candidates else []
    lines, per_depth = fss_rows(consensus, depths_mm, evidence)
    _write_csv(output_dir / "scale_per_depth.csv", per_depth, PER_DEPTH_FIELDS)
    (output_dir / "scale_lines.json").write_text(
        json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8")
    out.update(lines)

    accepted = sum(1 for r in per_depth if r["status"] == "accepted")
    reviewed = sum(1 for r in per_depth if r["status"] == "review")
    rejected = sum(1 for r in per_depth if r["status"] == "reject")
    interpolated = sum(1 for r in per_depth if r["source"] == "interpolated")
    out.update({
        "depths_total": len(depths_mm),
        "depths_accepted": accepted,
        "depths_review": reviewed,
        "depths_reject": rejected,
        "depths_interpolated": interpolated,
        "acceptance_ratio": float(accepted / max(1, len(per_depth))),
        "source": "ladder_ocr_consensus",
    })
    reasons: List[str] = list(out["review_reasons"])  # type: ignore[arg-type]
    if not per_depth:
        reasons.append("scale_no_candidates")
    if per_depth and not lines["line_21_scale_line"]:
        reasons.append("scale_line21_incomplete")
    if rejected:
        reasons.append("scale_depth_without_answer")
    out["review_reasons"] = reasons
    out["status"] = "ok" if (per_depth and not reasons) else "review"
    print(f"[scala] depth: {accepted} accettate, {reviewed} da rivedere, {rejected} senza "
          f"risposta ({interpolated} interpolate) -> {out['status']}", flush=True)

    (output_dir / "summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context-json", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=48,
                    help="Frame studiati al massimo (OCR e' la parte costosa).")
    ap.add_argument("--corrections", default="",
                    help="JSON delle correzioni esportate dalla pagina di studio. Da qui viene "
                         "usata solo la colonna del righello (fatto di cartella); le correzioni "
                         "per singolo frame restano nella pagina, dove si vedono.")
    args = ap.parse_args(argv)

    context = json.loads(args.context_json.read_text(encoding="utf-8"))
    run(context, args.output_dir, max(1, int(args.max_frames)), corrections=args.corrections)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
