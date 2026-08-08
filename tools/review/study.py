#!/usr/bin/env python3
"""Read one folder's pipeline output into a compact per-image study.

The official pipeline already writes everything the review needs, one CSV per module. What it
does not do is put them side by side: to judge a depth you want the rect it was read inside,
the up/down the ruler assumed, and the calibration the scala derived - on the *same* image.
This module is that join, plus the cross-area checks that only exist once the join is done.

Nothing here recomputes a prediction. If a number is not in the pipeline's output, it is
absent from the study too, and the UI says so.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- helpers
def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _f(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _i(value: object, default: Optional[int] = None) -> Optional[int]:
    got = _f(value, None)
    return default if got is None else int(round(got))


def _key(image_path: object) -> str:
    """Join key across modules: the file name.

    The pipeline sees the folder through a symlink (``input_ref/...``) while the depth and
    scala stages record absolute paths on the volume. The base name is the one thing every
    module agrees on, and within one acquisition folder it is unique.
    """
    return Path(str(image_path or "")).name.strip().lower()


def _parse_line11(text: str) -> Optional[Dict[str, int]]:
    parts = [p for p in str(text or "").split("|") if p.strip()]
    if len(parts) < 4:
        return None
    try:
        top, left, bottom, right = (int(round(float(p))) for p in parts[:4])
    except ValueError:
        return None
    return {"top": top, "left": left, "bottom": bottom, "right": right}


def _box_well_inside(outer: Dict[str, int], inner: Dict[str, float], inset: float = 0.10) -> bool:
    """True when ``inner`` sits in the middle of ``outer``, not hugging one of its borders.

    A plain containment test is useless here: on many machines the depth label sits exactly on
    the top edge of the ultrasound rectangle, so "inside" fires on every frame. What is really
    suspicious is a value read from the *middle* of the sector.
    """
    width = float(outer["right"] - outer["left"])
    height = float(outer["bottom"] - outer["top"])
    if width <= 0 or height <= 0:
        return False
    left = outer["left"] + width * inset
    right = outer["right"] - width * inset
    top = outer["top"] + height * inset
    bottom = outer["bottom"] - height * inset
    return (float(inner["left"]) >= left and float(inner["right"]) <= right
            and float(inner["top"]) >= top and float(inner["bottom"]) <= bottom)


# --------------------------------------------------------------------------- study
class FolderStudy:
    """Everything the review page shows for one folder, read from one run dir."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.output_dir = self.run_dir / "pipeline_output"
        self.step_checks: Dict[str, object] = {}
        checks_path = self.run_dir / "step_checks.json"
        if checks_path.is_file():
            try:
                loaded = json.loads(checks_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.step_checks = loaded
            except json.JSONDecodeError:
                self.step_checks = {}
        rows = _read_csv(self.output_dir / "folder_fss_head_predictions.csv")
        self.folder_row: Dict[str, str] = rows[0] if rows else {}

    # ------------------------------------------------------------------ ready?
    @property
    def available(self) -> bool:
        return bool(self.folder_row)

    # ------------------------------------------------------------------ folder
    def folder_summary(self) -> Dict[str, object]:
        row = self.folder_row
        raw = _i(row.get("images_total_raw"), 0) or 0
        unique = _i(row.get("images_total"), 0) or 0
        removed = _i(row.get("images_duplicates_removed"), 0) or 0
        pct = (removed / raw * 100.0) if raw else 0.0
        rect = _parse_line11(row.get("line_11_rect_echo", ""))
        return {
            "folder_name": row.get("folder_name", ""),
            "folder_path": row.get("folder_path", ""),
            "status": row.get("status", ""),
            "review_reasons": [r for r in str(row.get("review_reasons", "")).split(",") if r],
            "dedup": {
                # Descriptive only, as asked: the process output in one sentence.
                "images_raw": raw,
                "images_unique": unique,
                "removed": removed,
                "removed_pct": round(pct, 1),
                "text": (
                    f"{raw} immagini nella cartella, {removed} duplicati esatti rimossi "
                    f"({pct:.1f}%) → {unique} immagini studiate"
                    if raw
                    else "nessuna immagine trovata"
                ),
            },
            "rotazione": {
                "deg": _i(row.get("rotation_deg_clockwise"), 0),
                "source": row.get("rotation_source", ""),
                "vote_ratio": _f(row.get("rotation_vote_ratio"), 0.0),
                "reason": row.get("rotation_decision_reason", ""),
            },
            "vendor": {
                "value": row.get("vendor_predicted", ""),
                "confidence": _f(row.get("vendor_confidence"), 0.0),
                "margin": _f(row.get("vendor_margin_top1_top2"), 0.0),
                "threshold": _f(row.get("vendor_threshold"), 0.0),
                "source": row.get("vendor_source", ""),
                "reason": row.get("vendor_decision_reason", ""),
                "top3": _topk(row.get("vendor_top3_json", "")),
                "ocr_used": row.get("vendor_ocr_used", ""),
                "ocr_text": row.get("vendor_ocr_text", ""),
            },
            "probe": {
                "value": row.get("line_03_id_probe", ""),
                "name": self._probe_name(row.get("line_03_id_probe", "")),
                "confidence": _f(row.get("line_03_probe_confidence"), 0.0),
                "threshold": _f(row.get("probe_threshold"), 0.0),
                "source": row.get("probe_source", ""),
                "top3": _topk(row.get("probe_top3_json", "")),
                "probe_type": row.get("line_04_probe_type", ""),
                "probe_type_source": row.get("line_04_probe_type_source", ""),
            },
            "rect": {
                "line_11": row.get("line_11_rect_echo", ""),
                "box": rect,
                "method": row.get("line_11_method", ""),
                "source": row.get("line_11_source", ""),
                "checkpoint": row.get("line_11_model_checkpoint", ""),
                "red_winner_group": row.get("line_11_rect_red_winner_group", ""),
            },
            "line13": {
                "value": row.get("line_13_rect_name_echo", ""),
                "source": row.get("line_13_source", ""),
                "support": _f(row.get("line_13_support"), 0.0),
                "route": row.get("line_13_model_route", ""),
            },
            "su_giu": {
                "majority": row.get("su_giu_majority_label", ""),
                "vote_ratio": _f(row.get("su_giu_majority_vote_ratio"), 0.0),
                "mean_confidence": _f(row.get("su_giu_mean_confidence"), 0.0),
                "images": _i(row.get("su_giu_images_predicted"), 0),
                "source": row.get("su_giu_source", ""),
            },
            "lr_marker": {
                "majority": row.get("lr_marker_majority_label", ""),
                "vote_ratio": _f(row.get("lr_marker_majority_vote_ratio"), 0.0),
                "best_score": _f(row.get("lr_marker_best_score"), 0.0),
                "best_template": row.get("lr_marker_best_template_path", ""),
                "images": _i(row.get("lr_marker_images_predicted"), 0),
                "source": row.get("lr_marker_source", ""),
                "line_16": row.get("line_16_rect_orientation", ""),
                "line_16_source": row.get("line_16_source", ""),
            },
            "lt": {
                "majority": row.get("lt_majority_label", ""),
                "vote_ratio": _f(row.get("lt_majority_vote_ratio"), 0.0),
                "mean_confidence": _f(row.get("lt_mean_confidence"), 0.0),
                "images": _i(row.get("lt_images_predicted"), 0),
                "source": row.get("lt_source", ""),
            },
            "depth": {
                "status": row.get("rect_depth_status", ""),
                "images": _i(row.get("rect_depth_images_predicted"), 0),
                "accepted": _i(row.get("rect_depth_accepted_count"), 0),
                "review": _i(row.get("rect_depth_review_count"), 0),
                "reject": _i(row.get("rect_depth_reject_count"), 0),
                "missing": _i(row.get("rect_depth_missing_count"), 0),
                "acceptance_ratio": _f(row.get("rect_depth_acceptance_ratio"), 0.0),
                "majority_mode": row.get("rect_depth_majority_mode", ""),
                "unique_depths": _json_list(row.get("rect_depth_unique_depths_json", "[]")),
                "source": row.get("rect_depth_source", ""),
            },
            "scala": {
                "status": row.get("scale_status", ""),
                "source": row.get("scale_source", ""),
                "profile": row.get("scale_profile", ""),
                "frames_studied": _i(row.get("scale_frames_studied"), 0),
                "depths_total": _i(row.get("scale_depths_total"), 0),
                "depths_accepted": _i(row.get("scale_depths_accepted"), 0),
                "depths_review": _i(row.get("scale_depths_review"), 0),
                "depths_reject": _i(row.get("scale_depths_reject"), 0),
                "depths_interpolated": _i(row.get("scale_depths_interpolated"), 0),
                "acceptance_ratio": _f(row.get("scale_acceptance_ratio"), 0.0),
                "ruler_x": row.get("scale_ruler_x", ""),
                "line_18": row.get("line_18_vect_depth", ""),
                "line_19": row.get("line_19_pixel_ratio_x", ""),
                "line_20": row.get("line_20_pixel_ratio_y", ""),
                "line_21": row.get("line_21_scale_line", ""),
                "per_depth": self.scale_per_depth(),
            },
            "fss": {"lines": self.fss_lines()},
        }

    def _probe_name(self, probe_id: object) -> str:
        checks = self.step_checks.get("checks")
        if isinstance(checks, list):
            for item in checks:
                if isinstance(item, dict) and item.get("step") == "probe":
                    return str(item.get("line_03_probe_name", "") or "")
        return ""

    def fss_lines(self) -> List[Dict[str, str]]:
        """The head as the pipeline would write it, read back from the preview."""
        preview = self.output_dir / "folder_fss_head_preview.txt"
        if not preview.is_file():
            return []
        out: List[Dict[str, str]] = []
        for line in preview.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text.startswith("#"):
                continue
            parts = text.split(" ", 1)
            out.append({"line": parts[0], "value": parts[1] if len(parts) > 1 else ""})
        return out

    def scale_per_depth(self) -> List[Dict[str, object]]:
        rows = _read_csv(self.output_dir / "scale_per_depth.csv")
        if not rows:
            for candidate in sorted(self.output_dir.glob("scale/*/scale_per_depth.csv")):
                rows = _read_csv(candidate)
                if rows:
                    break
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ images
    def images(self) -> List[Dict[str, object]]:
        """One row per studied image, with every module's verdict on it."""
        merged: Dict[str, Dict[str, object]] = {}
        order: List[str] = []

        def slot(image_path: str) -> Dict[str, object]:
            key = _key(image_path)
            if key not in merged:
                merged[key] = {
                    "key": key,
                    "image_id": Path(str(image_path)).name,
                    "image_path": str(image_path),
                }
                order.append(key)
            return merged[key]

        for row in _read_csv(self.output_dir / "su_giu_per_image_predictions.csv"):
            item = slot(row.get("image_path", ""))
            item["index"] = _i(row.get("image_index"), None)
            item["su_giu"] = {
                "label": row.get("pred_label", ""),
                "confidence": _f(row.get("confidence"), None),
                "prob_su": _f(row.get("prob_su"), None),
                "prob_giu": _f(row.get("prob_giu"), None),
                "crop_source": row.get("crop_source", ""),
            }
            item["rect_per_image"] = {
                "top": _i(row.get("crop_top"), None),
                "left": _i(row.get("crop_left"), None),
                "bottom": _i(row.get("crop_bottom"), None),
                "right": _i(row.get("crop_right"), None),
                "source": row.get("crop_source", ""),
            }

        for row in _read_csv(self.output_dir / "lt_per_image_predictions.csv"):
            item = slot(row.get("image_path", ""))
            item["lt"] = {
                "label": row.get("pred_label", ""),
                "confidence": _f(row.get("confidence"), None),
                "prob_l": _f(row.get("prob_l"), None),
                "prob_t": _f(row.get("prob_t"), None),
            }

        for row in _read_csv(self.output_dir / "lr_marker_per_image_predictions.csv"):
            item = slot(row.get("image_path", ""))
            item["lr_marker"] = {
                "label": row.get("lr_label", ""),
                "label_it": row.get("lr_label_it", ""),
                "status": row.get("status", ""),
                "review_reason": row.get("review_reason", ""),
                "orientation_group": row.get("orientation_group", ""),
                "score": _f(row.get("match_score"), None),
                "detected_side": row.get("detected_marker_side", ""),
                "search_strategy": row.get("search_strategy", ""),
                "template": row.get("template_path", "") or row.get("template_rel_path", ""),
                "box": _marker_box(row),
            }
            if not item.get("image_width"):
                item["image_width"] = _i(row.get("image_width"), None)
                item["image_height"] = _i(row.get("image_height"), None)

        for row in _read_csv(self.output_dir / "rect_depth_autonomous_predictions.csv"):
            item = slot(row.get("image_path", ""))
            item["depth"] = {
                "status": row.get("status", ""),
                "depth_mm": _f(row.get("depth_mm"), None),
                "score": _f(row.get("score"), None),
                "ranker_score": _f(row.get("ranker_score"), None),
                "mode": row.get("mode", ""),
                "ocr_text": row.get("ocr_text", ""),
                "reason": row.get("reason", ""),
                "candidates": _i(row.get("candidates"), None),
                "box": {
                    "left": _f(row.get("left"), None),
                    "top": _f(row.get("top"), None),
                    "right": _f(row.get("right"), None),
                    "bottom": _f(row.get("bottom"), None),
                },
            }

        for row in _read_csv(self.output_dir / "scale_per_image_predictions.csv"):
            item = slot(row.get("image_path", ""))
            item["scala"] = {
                "status": row.get("status", ""),
                "reason": row.get("reason", ""),
                "depth_mm": _f(row.get("depth_mm"), None),
                "depth_index": _i(row.get("depth_index"), None),
                "x": _f(row.get("x"), None),
                "y_zero": _f(row.get("y_zero"), None),
                "y_far": _f(row.get("y_far"), None),
                "mm_per_px": _f(row.get("mm_per_px"), None),
                "tick_pitch_px": _f(row.get("tick_pitch_px"), None),
                "tick_step_mm": _f(row.get("tick_step_mm"), None),
                "n_ticks": _i(row.get("n_ticks"), None),
                "n_labels": _i(row.get("n_labels"), None),
                "direction": row.get("direction", ""),
                "zero_end": row.get("zero_end", ""),
                "label_side": row.get("label_side", ""),
                "labels_json": row.get("labels_json", ""),
                "calib_source": row.get("calib_source", ""),
                "out_of_band": row.get("out_of_band", ""),
                "used_folder_zone": row.get("used_folder_zone", ""),
                "zero_check": row.get("zero_check", ""),
                "depth_check": row.get("depth_check", ""),
                "depth_mode": row.get("depth_mode", ""),
                "sugiu": row.get("sugiu", ""),
                "sugiu_source": row.get("sugiu_source", ""),
                "confidence": _f(row.get("confidence"), None),
            }

        self._attach_rect_discrepancy(merged)
        self._attach_sizes(merged)

        folder = self.folder_summary()
        items = [merged[k] for k in order]
        for item in items:
            item["checks"] = coherence_checks(item, folder)
        items.sort(key=lambda it: (it.get("index") is None, it.get("index") or 0, it["image_id"]))
        return items

    def _attach_rect_discrepancy(self, merged: Dict[str, Dict[str, object]]) -> None:
        """IoU of the per-image rect against the folder's #11, computed by the safe runner."""
        evidence = (self.step_checks.get("recognition_evidence") or {})
        rect_per_image = evidence.get("rect_per_image") if isinstance(evidence, dict) else None
        if not isinstance(rect_per_image, dict):
            return
        items = rect_per_image.get("items")
        if not isinstance(items, list):
            return
        for entry in items:
            if not isinstance(entry, dict):
                continue
            key = _key(entry.get("image_rel", ""))
            if key not in merged:
                continue
            merged[key]["rect_vs_folder"] = {
                "iou": _f(entry.get("iou"), None),
                "discrepancy": _f(entry.get("discrepancy"), None),
                "center_distance_norm": _f(entry.get("center_distance_norm"), None),
                "single_rect": entry.get("single_rect"),
                "folder_rect_scaled": entry.get("global_rect_scaled"),
            }

    def _attach_sizes(self, merged: Dict[str, Dict[str, object]]) -> None:
        evidence = (self.step_checks.get("recognition_evidence") or {})
        rect_per_image = evidence.get("rect_per_image") if isinstance(evidence, dict) else None
        if not isinstance(rect_per_image, dict):
            return
        for entry in rect_per_image.get("items") or []:
            if not isinstance(entry, dict):
                continue
            key = _key(entry.get("image_rel", ""))
            if key in merged and not merged[key].get("image_width"):
                merged[key]["image_width"] = _i(entry.get("image_width"), None)
                merged[key]["image_height"] = _i(entry.get("image_height"), None)


def _topk(text: object) -> List[Dict[str, object]]:
    try:
        payload = json.loads(str(text or "[]"))
    except json.JSONDecodeError:
        return []
    out: List[Dict[str, object]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                label = item.get("label", item.get("class", ""))
                prob = item.get("prob", item.get("confidence", item.get("support")))
                out.append({"label": str(label), "prob": _f(prob, 0.0)})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append({"label": str(item[0]), "prob": _f(item[1], 0.0)})
    return out


def _json_list(text: object) -> List[object]:
    try:
        payload = json.loads(str(text or "[]"))
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _marker_box(row: Dict[str, str]) -> Optional[Dict[str, int]]:
    """The marker box in original image pixels.

    Only the ``_abs`` columns: the ``_crop`` ones are relative to the rect crop and drawing
    them on the full frame would put the box in the wrong place.
    """
    left = _i(row.get("marker_left_abs"), None)
    top = _i(row.get("marker_top_abs"), None)
    right = _i(row.get("marker_right_abs"), None)
    bottom = _i(row.get("marker_bottom_abs"), None)
    if None in (left, top, right, bottom):
        return None
    if right <= left or bottom <= top:  # type: ignore[operator]
        return None
    return {"left": left, "top": top, "right": right, "bottom": bottom}  # type: ignore[dict-item]


# --------------------------------------------------------------------------- coherence
# Verdicts of the scala's own counter-checks that mean "nothing to check here", not "wrong".
SCALE_CHECK_NEUTRAL = {"ok", "unknown", "not_checked", "unverifiable", "non verificabile",
                       "non verificabile.", "-", "none"}


def txt_or(value: object, fallback: str = "?") -> str:
    text = str(value if value is not None else "").strip()
    return text or fallback


def coherence_checks(image: Dict[str, object], folder: Dict[str, object]) -> List[Dict[str, str]]:
    """Cross-area checks on one image: the part no single module can see.

    Each module is right on its own terms and can still be wrong in company - a depth read
    outside the ultrasound rectangle, a ruler that runs against the up/down the marker
    settled. These are the checks that turn "every module says ok" into a real verdict, and
    they are also the seed of the orchestrator: each one names the two areas that disagree.
    """
    out: List[Dict[str, str]] = []

    def add(check_id: str, level: str, message: str, areas: str) -> None:
        out.append({"id": check_id, "level": level, "message": message, "areas": areas})

    depth = image.get("depth") or {}
    scala = image.get("scala") or {}
    su_giu = image.get("su_giu") or {}
    rect_img = image.get("rect_per_image") or {}
    folder_rect = (folder.get("rect") or {}).get("box")

    # 1. the depth value must be read inside the ultrasound rectangle's image, not outside it
    box = depth.get("box") or {}
    if all(box.get(k) is not None for k in ("left", "top", "right", "bottom")):
        reference = None
        if all(rect_img.get(k) is not None for k in ("left", "top", "right", "bottom")):
            reference = {k: int(rect_img[k]) for k in ("left", "top", "right", "bottom")}  # type: ignore[index]
        elif isinstance(folder_rect, dict):
            reference = folder_rect
        width = image.get("image_width")
        height = image.get("image_height")
        if width and height:
            if not (0 <= box["left"] < box["right"] <= float(width) + 2
                    and 0 <= box["top"] < box["bottom"] <= float(height) + 2):
                add("depth_box_out_of_image", "error",
                    "Il box della depth cade fuori dall'immagine", "depth")
        if reference and _box_well_inside(reference, box):
            add("depth_box_inside_rect", "warn",
                "Il box della depth è in mezzo al settore ecografico, lontano dai bordi: "
                "l'etichetta sta nell'interfaccia, non nell'immagine", "depth+rect")

    # 2. depth and scala must agree on the span they describe
    mm_per_px = scala.get("mm_per_px")
    y_zero, y_far = scala.get("y_zero"), scala.get("y_far")
    depth_mm = depth.get("depth_mm") or scala.get("depth_mm")
    if mm_per_px and y_zero is not None and y_far is not None and depth_mm:
        span_mm = abs(float(y_far) - float(y_zero)) * float(mm_per_px)
        if span_mm > 0:
            delta = abs(span_mm - float(depth_mm)) / float(depth_mm)
            if delta > 0.10:
                add("depth_scala_mismatch", "error",
                    f"Scala e depth non tornano: il righello copre {span_mm:.0f} mm, "
                    f"la depth dice {float(depth_mm):.0f} mm ({delta * 100:.0f}% di scarto)",
                    "depth+scala")
            else:
                add("depth_scala_agree", "ok",
                    f"Scala e depth si accordano ({span_mm:.0f} mm vs {float(depth_mm):.0f} mm)",
                    "depth+scala")

    # 3. up/down: the scala stage is *given* the verso, so a mismatch with the classifier's own
    #    verdict on this frame means the two are working from different assumptions.
    #    (``direction`` is not the verso: it is ±1, the sign of increasing depth.)
    scala_sugiu = str(scala.get("sugiu", "") or "").strip().lower()
    su_label = str(su_giu.get("label", "") or "").strip().lower()
    if scala_sugiu and su_label and scala_sugiu != su_label:
        add("sugiu_scala_disagree", "warn",
            f"Verso discordante: il classificatore dice '{su_label}', "
            f"la scala ha lavorato con '{scala_sugiu}' (da {txt_or(scala.get('sugiu_source'))})",
            "su_giu+scala")
    if str(scala.get("zero_moved", "")).strip().lower() in {"1", "true", "yes"}:
        add("scala_zero_moved", "warn",
            f"Lo stadio scala ha spostato lo zero rispetto al verso ricevuto "
            f"(zero a {txt_or(scala.get('zero_end'))}, da {txt_or(scala.get('zero_end_source'))})",
            "su_giu+scala")

    # 4. per-image rect far from the folder's #11
    rect_vs = image.get("rect_vs_folder") or {}
    iou = rect_vs.get("iou")
    if iou is not None and float(iou) < 0.70:
        add("rect_per_image_far", "warn",
            f"Il rect di questa immagine si discosta dal #11 di cartella (IoU {float(iou):.2f})",
            "rect")

    # 5. a depth value that no other image in the folder shows
    unique_depths = [(_f(d, None)) for d in (folder.get("depth") or {}).get("unique_depths") or []]
    unique_depths = [d for d in unique_depths if d is not None]
    if depth.get("depth_mm") and unique_depths:
        value = float(depth["depth_mm"])
        if all(abs(value - d) > 0.5 for d in unique_depths):
            add("depth_outlier", "warn",
                f"Depth {value:.0f} mm fuori dai valori della cartella "
                f"({', '.join(f'{d:.0f}' for d in unique_depths)})", "depth")

    # 6. scala accepted on weak evidence: the failure mode that keeps coming back
    if str(scala.get("status", "")).strip() == "accepted":
        weak = []
        if str(scala.get("out_of_band", "")).strip().lower() in {"1", "true", "yes"}:
            weak.append("righello fuori dalla banda del vendor")
        if str(scala.get("used_folder_zone", "")).strip().lower() in {"1", "true", "yes"}:
            weak.append("colonna presa dalla zona di cartella")
        if str(scala.get("calib_source", "")).strip() == "geometry":
            weak.append("calibrazione geometrica")
        if weak:
            add("scala_weak_evidence", "warn",
                "Scala accepted con evidenza debole: " + ", ".join(weak), "scala")

    # 7. the modules' own consistency fields, surfaced instead of buried in a CSV.
    #    "non verificabile" is not a failure: it means the check had nothing to work with, and
    #    warning on it would bury the real disagreements under noise on every frame.
    for field, label in (("zero_check", "zero"), ("depth_check", "depth")):
        verdict = str(scala.get(field, "") or "").strip().lower()
        if verdict in SCALE_CHECK_NEUTRAL or not verdict:
            continue
        add(f"scala_{field}", "warn", f"Controprova {label} della scala: {verdict}", "scala")

    return out


def folder_coherence(folder: Dict[str, object], images: List[Dict[str, object]]) -> List[Dict[str, str]]:
    """Folder-level checks, including the ones that are about *distributions*."""
    out: List[Dict[str, str]] = []

    def add(check_id: str, level: str, message: str, areas: str) -> None:
        out.append({"id": check_id, "level": level, "message": message, "areas": areas})

    vendor = folder.get("vendor") or {}
    if vendor.get("confidence") is not None and vendor.get("threshold"):
        if float(vendor["confidence"]) < float(vendor["threshold"]):
            add("vendor_below_threshold", "warn",
                f"Vendor sotto soglia ({float(vendor['confidence']):.2f} < "
                f"{float(vendor['threshold']):.2f}): richiede conferma", "vendor")

    probe = folder.get("probe") or {}
    if probe.get("confidence") is not None and probe.get("threshold"):
        if float(probe["confidence"]) < float(probe["threshold"]):
            add("probe_below_threshold", "warn",
                f"Sonda sotto soglia ({float(probe['confidence']):.2f} < "
                f"{float(probe['threshold']):.2f}): richiede conferma", "probe")

    depth = folder.get("depth") or {}
    if depth.get("images") and depth.get("acceptance_ratio") is not None:
        if float(depth["acceptance_ratio"]) < 0.80:
            add("depth_low_acceptance", "warn",
                f"Depth accettate solo al {float(depth['acceptance_ratio']) * 100:.0f}%", "depth")

    scala = folder.get("scala") or {}
    depth_ok = bool((folder.get("depth") or {}).get("images"))
    if depth_ok and not scala.get("depths_total") and str(scala.get("source", "")) != "disabled":
        # The blocker that matters: the depth found its values but the ruler produced no #21,
        # so the head cannot be written even though every other stage says ok.
        add("scala_no_answer", "error",
            f"La scala non ha prodotto nessuna riga #21 su {scala.get('frames_studied') or 0} "
            f"frame studiati (righello a x={txt_or(scala.get('ruler_x'), '?')})", "scala+depth")
    if scala.get("depths_total") and scala.get("acceptance_ratio") is not None:
        if float(scala["acceptance_ratio"]) < 0.80:
            add("scala_low_acceptance", "warn",
                f"Scala: solo {scala.get('depths_accepted')} depth su "
                f"{scala.get('depths_total')} con risposta accettata", "scala")
    if scala.get("depths_reject"):
        add("scala_depth_without_answer", "warn",
            f"{scala.get('depths_reject')} depth senza righello risolto", "scala+depth")

    # the two orientation sources, on the same folder
    su_majority = str((folder.get("su_giu") or {}).get("majority", ""))
    counted: Dict[str, int] = {}
    for image in images:
        label = str((image.get("scala") or {}).get("sugiu", "") or "").strip().lower()
        if label:
            counted[label] = counted.get(label, 0) + 1
    if su_majority and counted:
        top = max(counted.items(), key=lambda kv: kv[1])[0]
        if top and top != su_majority:
            add("sugiu_scala_folder_disagree", "warn",
                f"Sulla cartella il verso prevalente da su/giù è '{su_majority}', "
                f"ma la scala ha lavorato con '{top}'", "su_giu+scala")

    disagreeing = sum(1 for i in images
                      if any(c["id"] == "depth_scala_mismatch" for c in i.get("checks", [])))
    if disagreeing:
        add("depth_scala_folder_mismatch", "error",
            f"{disagreeing} immagini con depth e scala in disaccordo", "depth+scala")

    return out
