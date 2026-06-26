#!/usr/bin/env python3
"""Apply the portable marker-bundle rule on official pipeline rect/SU-GIU rows.

Input is the `su_giu_per_image_predictions.csv` emitted by
`tools/ultrasound/predict_fss_head_from_acquisitions.py`. That CSV already
contains:

- the official ultrasound rectangle used for each image;
- the official SU/GIU classification on that rectangle crop.

This adapter keeps those two official decisions and swaps only the marker
orientation logic with the portable bundle logic:

- search accepted historical marker templates for the predicted vendor;
- split left/right and top/bottom using the median axes of the official rect;
- assign NF/LR/UD/LRUD from marker quadrant;
- build one envelope box per orientation and save its coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import types
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_ZIP = Path("/Users/Shared/41_orientation_marker_detector_bundle.zip")
DEFAULT_BUNDLE_DIR = REPO_ROOT / "artifacts/41_orientation_marker_detector_bundle"
DEFAULT_LIBRARY_ROOT = DEFAULT_BUNDLE_DIR / "orientation_marker_detector" / "templates"
ORIENTATION_ORDER = ("NF", "LR", "UD", "LRUD")
GROUP_TO_VERTICAL = {"NF": "su", "LR": "su", "UD": "giu", "LRUD": "giu"}

Rect = Tuple[int, int, int, int]  # top, left, bottom, right inclusive


@dataclass
class Envelope:
    folder_name: str
    vendor: str
    orientation_group: str
    top: int
    left: int
    bottom: int
    right: int
    boxes: int
    median_marker_width: float
    median_marker_height: float
    median_score: float


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default


def _ensure_bundle_import(bundle_zip: Path, bundle_dir: Optional[Path]) -> Path:
    if bundle_dir is not None:
        root = bundle_dir.expanduser().resolve()
    else:
        if not bundle_zip.expanduser().is_file():
            raise FileNotFoundError(f"Bundle zip not found: {bundle_zip}")
        root = REPO_ROOT / "artifacts/41_orientation_marker_detector_bundle_unpacked"
        marker = root / "orientation_marker_detector" / "detector.py"
        if not marker.is_file():
            root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(bundle_zip.expanduser().resolve()) as zf:
                zf.extractall(root.parent)
            extracted = root.parent / "41_orientation_marker_detector_bundle"
            if extracted.is_dir() and extracted != root:
                if root.exists():
                    # Existing incomplete extraction: leave files alone and use extracted path.
                    root = extracted
                else:
                    extracted.rename(root)
    if not (root / "orientation_marker_detector" / "detector.py").is_file():
        raise FileNotFoundError(f"Bundle detector.py not found under: {root}")
    sys.path.insert(0, root.as_posix())
    return root


def _install_cv2_fallback_if_needed() -> None:
    try:
        import cv2  # noqa: F401

        return
    except Exception:
        pass

    from PIL import Image as PILImage
    from scipy.signal import correlate2d

    cv2_stub = types.ModuleType("cv2")
    cv2_stub.TM_CCOEFF_NORMED = 5
    cv2_stub.INTER_AREA = 3

    def _resize(image: np.ndarray, size: Tuple[int, int], interpolation: int = 3) -> np.ndarray:
        width, height = int(size[0]), int(size[1])
        pil = PILImage.fromarray(np.asarray(image).astype(np.uint8))
        return np.asarray(pil.resize((width, height), PILImage.Resampling.BOX), dtype=np.uint8)

    def _match_template(search: np.ndarray, template: np.ndarray, method: int) -> np.ndarray:
        if method != cv2_stub.TM_CCOEFF_NORMED:
            raise ValueError("cv2 fallback supports only TM_CCOEFF_NORMED")
        src = np.asarray(search, dtype=np.float32)
        tpl = np.asarray(template, dtype=np.float32)
        th, tw = tpl.shape[:2]
        if src.ndim != 2 or tpl.ndim != 2 or src.shape[0] < th or src.shape[1] < tw:
            return np.zeros((0, 0), dtype=np.float32)
        tpl_z = tpl - float(tpl.mean())
        tpl_norm = float(np.sqrt(np.sum(tpl_z * tpl_z)))
        if tpl_norm <= 1e-8:
            return np.zeros((src.shape[0] - th + 1, src.shape[1] - tw + 1), dtype=np.float32)
        ones = np.ones((th, tw), dtype=np.float32)
        patch_sum = correlate2d(src, ones, mode="valid")
        patch_sum_sq = correlate2d(src * src, ones, mode="valid")
        area = float(th * tw)
        patch_mean = patch_sum / area
        patch_var_sum = np.maximum(patch_sum_sq - area * patch_mean * patch_mean, 1e-8)
        numerator = correlate2d(src, tpl_z[::-1, ::-1], mode="valid")
        denom = np.sqrt(patch_var_sum) * tpl_norm
        return (numerator / np.maximum(denom, 1e-8)).astype(np.float32)

    def _min_max_loc(result: np.ndarray):
        arr = np.asarray(result)
        if arr.size == 0:
            return 0.0, 0.0, (0, 0), (0, 0)
        min_flat = int(np.argmin(arr))
        max_flat = int(np.argmax(arr))
        min_y, min_x = np.unravel_index(min_flat, arr.shape)
        max_y, max_x = np.unravel_index(max_flat, arr.shape)
        return float(arr[min_y, min_x]), float(arr[max_y, max_x]), (int(min_x), int(min_y)), (int(max_x), int(max_y))

    cv2_stub.resize = _resize
    cv2_stub.matchTemplate = _match_template
    cv2_stub.minMaxLoc = _min_max_loc
    sys.modules["cv2"] = cv2_stub


def _median(values: Sequence[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.median(vals)) if vals else default


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.expanduser().resolve().open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _group_rows(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        key = str(row.get("folder_name") or row.get("group_name") or Path(row.get("image_path", "")).parent.name)
        grouped.setdefault(key, []).append(row)
    return grouped


def _crop_rect_from_row(row: Dict[str, str]) -> Optional[Rect]:
    top = _to_int(row.get("crop_top"), -1)
    left = _to_int(row.get("crop_left"), -1)
    # Pipeline crop CSV stores bottom/right as PIL-exclusive limits.
    bottom = _to_int(row.get("crop_bottom"), -1) - 1
    right = _to_int(row.get("crop_right"), -1) - 1
    if top < 0 or left < 0 or bottom <= top or right <= left:
        return None
    return top, left, bottom, right


def _side_vertical_to_group(side: str, vertical: str) -> str:
    if side == "left" and vertical == "su":
        return "NF"
    if side == "right" and vertical == "su":
        return "LR"
    if side == "left" and vertical == "giu":
        return "UD"
    if side == "right" and vertical == "giu":
        return "LRUD"
    return ""


def _box_center(rect: Rect) -> Tuple[float, float]:
    t, l, b, r = rect
    return (l + r) / 2.0, (t + b) / 2.0


def _row_to_image_input(row: Dict[str, str], image_input_cls) -> Optional[object]:
    image_path_s = str(row.get("image_path", "")).strip()
    if not image_path_s:
        return None
    image_path = Path(image_path_s).expanduser().resolve()
    if not image_path.is_file():
        return None
    crop_rect = _crop_rect_from_row(row)
    if crop_rect is None:
        return None
    sugiu_pred = str(row.get("pred_label", "")).strip().lower()
    if sugiu_pred not in {"su", "giu"}:
        sugiu_pred = ""
    conf = _to_float(row.get("confidence"), 0.0)
    return image_input_cls(
        image_path=image_path,
        image_id=str(row.get("image_index", "") or image_path.name),
        crop_rect=crop_rect,
        crop_source=str(row.get("crop_source", "")),
        sugiu_pred=sugiu_pred,
        sugiu_conf=conf,
        expected_group="",
    )


def _result_to_row(result: object, source_row: Dict[str, str]) -> Dict[str, object]:
    d = result.to_dict() if hasattr(result, "to_dict") else dict(asdict(result))
    marker_box = d.get("marker_box_abs")
    crop_rect = d.get("crop_rect_abs")
    orientation_group = str(d.get("orientation_group") or "")
    if marker_box and crop_rect:
        mt, ml, mb, mr = [int(v) for v in marker_box]
        ct, cl, cb, cr = [int(v) for v in crop_rect]
        marker_cx, marker_cy = _box_center((mt, ml, mb, mr))
        rect_mid_x = (cl + cr) / 2.0
        rect_mid_y = (ct + cb) / 2.0
        detected_side_by_official_rect_axis = "left" if marker_cx < rect_mid_x else "right"
        vertical_by_official_rect_axis = "su" if marker_cy < rect_mid_y else "giu"
        quadrant_group = _side_vertical_to_group(detected_side_by_official_rect_axis, vertical_by_official_rect_axis)
    else:
        detected_side_by_official_rect_axis = ""
        vertical_by_official_rect_axis = ""
        quadrant_group = ""
    out: Dict[str, object] = {
        "folder_name": source_row.get("folder_name", ""),
        "folder_path": source_row.get("folder_path", ""),
        "image_index": source_row.get("image_index", ""),
        "image_path": d.get("image_path", source_row.get("image_path", "")),
        "status": d.get("status", ""),
        "review_reason": d.get("review_reason", ""),
        "orientation_group": orientation_group or quadrant_group,
        "orientation_group_from_official_rect_axes": quadrant_group,
        "detected_marker_side": d.get("detected_marker_side", ""),
        "detected_marker_side_by_official_rect_axis": detected_side_by_official_rect_axis,
        "vertical_final": d.get("vertical_final", ""),
        "vertical_by_official_rect_axis": vertical_by_official_rect_axis,
        "vertical_source": d.get("vertical_source", ""),
        "vertical_correction": d.get("vertical_correction", ""),
        "match_score": d.get("match_score", ""),
        "search_scope": d.get("search_scope", ""),
        "template_name": d.get("template_name", ""),
        "template_path": d.get("template_path", ""),
        "top_half_score": d.get("top_half_score", ""),
        "bottom_half_score": d.get("bottom_half_score", ""),
        "full_crop_score": d.get("full_crop_score", ""),
        "expanded_score": d.get("expanded_score", ""),
        "su_giu_pred": d.get("sugiu_pred", source_row.get("pred_label", "")),
        "su_giu_conf": d.get("sugiu_conf", source_row.get("confidence", "")),
    }
    if crop_rect:
        ct, cl, cb, cr = [int(v) for v in crop_rect]
        out.update(
            {
                "echo_rect_top_abs": ct,
                "echo_rect_left_abs": cl,
                "echo_rect_bottom_abs": cb,
                "echo_rect_right_abs": cr,
                "echo_rect_mid_x_abs": (cl + cr) / 2.0,
                "echo_rect_mid_y_abs": (ct + cb) / 2.0,
            }
        )
    if marker_box:
        mt, ml, mb, mr = [int(v) for v in marker_box]
        out.update(
            {
                "marker_top_abs": mt,
                "marker_left_abs": ml,
                "marker_bottom_abs": mb,
                "marker_right_abs": mr,
                "marker_width": mr - ml + 1,
                "marker_height": mb - mt + 1,
            }
        )
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_envelopes(
    rows: Sequence[Dict[str, object]],
    min_match_score: float,
    folder_name: str,
    vendor: str,
) -> List[Envelope]:
    out: List[Envelope] = []
    for group in ORIENTATION_ORDER:
        group_rows: List[Dict[str, object]] = []
        for row in rows:
            if str(row.get("status", "") or "") != "ok":
                continue
            if str(row.get("orientation_group_from_official_rect_axes") or row.get("orientation_group") or "") != group:
                continue
            score = _to_float(row.get("match_score"), -1.0)
            if score < min_match_score:
                continue
            if str(row.get("marker_top_abs", "") or "") == "":
                continue
            group_rows.append(row)
        if not group_rows:
            continue
        tops = [_to_int(r.get("marker_top_abs")) for r in group_rows]
        lefts = [_to_int(r.get("marker_left_abs")) for r in group_rows]
        bottoms = [_to_int(r.get("marker_bottom_abs")) for r in group_rows]
        rights = [_to_int(r.get("marker_right_abs")) for r in group_rows]
        out.append(
            Envelope(
                folder_name=folder_name,
                vendor=vendor,
                orientation_group=group,
                top=min(tops),
                left=min(lefts),
                bottom=max(bottoms),
                right=max(rights),
                boxes=len(group_rows),
                median_marker_width=_median([_to_float(r.get("marker_width")) for r in group_rows]),
                median_marker_height=_median([_to_float(r.get("marker_height")) for r in group_rows]),
                median_score=_median([_to_float(r.get("match_score")) for r in group_rows]),
            )
        )
    return out


def _draw_overlay(row: Dict[str, object], envelopes: Sequence[Envelope], output_path: Path) -> None:
    image_path = Path(str(row.get("image_path", ""))).expanduser().resolve()
    if not image_path.is_file():
        return
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        draw = ImageDraw.Draw(rgb)
        try:
            et = _to_int(row.get("echo_rect_top_abs"))
            el = _to_int(row.get("echo_rect_left_abs"))
            eb = _to_int(row.get("echo_rect_bottom_abs"))
            er = _to_int(row.get("echo_rect_right_abs"))
            draw.rectangle((el, et, er, eb), outline=(245, 158, 11), width=3)
            draw.line(((el + er) // 2, et, (el + er) // 2, eb), fill=(245, 158, 11), width=2)
            draw.line((el, (et + eb) // 2, er, (et + eb) // 2), fill=(245, 158, 11), width=2)
        except Exception:
            pass
        try:
            mt = _to_int(row.get("marker_top_abs"))
            ml = _to_int(row.get("marker_left_abs"))
            mb = _to_int(row.get("marker_bottom_abs"))
            mr = _to_int(row.get("marker_right_abs"))
            draw.rectangle((ml, mt, mr, mb), outline=(8, 145, 178), width=4)
        except Exception:
            pass
        colors = {"NF": (34, 197, 94), "LR": (239, 68, 68), "UD": (59, 130, 246), "LRUD": (168, 85, 247)}
        for env in envelopes:
            color = colors.get(env.orientation_group, (255, 255, 255))
            draw.rectangle((env.left, env.top, env.right, env.bottom), outline=color, width=3)
        rgb.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(output_path, quality=90)


def _pick_overlay_rows(rows: Sequence[Dict[str, object]], limit: int) -> List[Dict[str, object]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    selected: List[Dict[str, object]] = []
    seen: set[int] = set()
    for i in range(limit):
        idx = round(i * (len(rows) - 1) / max(1, limit - 1))
        if idx not in seen:
            selected.append(rows[idx])
            seen.add(idx)
    return selected


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bundle marker logic on official pipeline rect/SU-GIU predictions.")
    p.add_argument("--su-giu-csv", type=Path, required=True, help="su_giu_per_image_predictions.csv from official pipeline.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--vendor", type=str, default="", help="Force vendor. If empty, one is inferred from folder name.")
    p.add_argument("--library-root", type=Path, default=DEFAULT_LIBRARY_ROOT, help="Accepted marker template library.")
    p.add_argument("--bundle-zip", type=Path, default=DEFAULT_BUNDLE_ZIP)
    p.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    p.add_argument("--min-match-score", type=float, default=0.55)
    p.add_argument("--vertical-delta", type=float, default=0.05)
    p.add_argument("--fallback-threshold", type=float, default=0.58)
    p.add_argument("--expanded-threshold", type=float, default=0.62)
    p.add_argument("--match-max-side", type=int, default=720)
    p.add_argument("--selection-images", type=int, default=18)
    p.add_argument("--overlay-limit-per-folder", type=int, default=24)
    return p


def main() -> int:
    args = build_parser().parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    _install_cv2_fallback_if_needed()
    bundle_root = _ensure_bundle_import(args.bundle_zip, args.bundle_dir)

    from orientation_marker_detector.detector import (  # type: ignore
        DetectionParams,
        ImageInput,
        analyze_images,
        infer_vendor_from_text,
    )

    params = DetectionParams(
        min_match_score=float(args.min_match_score),
        vertical_delta=float(args.vertical_delta),
        fallback_threshold=float(args.fallback_threshold),
        expanded_threshold=float(args.expanded_threshold),
        match_max_side=int(args.match_max_side),
    )

    source_rows = _read_rows(args.su_giu_csv)
    by_folder = _group_rows(source_rows)
    all_prediction_rows: List[Dict[str, object]] = []
    all_envelopes: List[Envelope] = []
    folder_summaries: List[Dict[str, object]] = []

    for folder_name, folder_rows in sorted(by_folder.items(), key=lambda kv: kv[0].lower()):
        vendor = str(args.vendor).strip() or infer_vendor_from_text(folder_name)
        images: List[object] = []
        image_to_source: List[Dict[str, str]] = []
        for row in folder_rows:
            image_input = _row_to_image_input(row, ImageInput)
            if image_input is None:
                continue
            images.append(image_input)
            image_to_source.append(row)
        if not images:
            folder_summaries.append({"folder_name": folder_name, "vendor": vendor, "status": "no_valid_images"})
            continue
        try:
            analysis = analyze_images(
                images,
                vendor=vendor,
                library_root=args.library_root.expanduser().resolve(),
                params=params,
                selection_images=int(args.selection_images),
            )
        except Exception as exc:
            folder_summaries.append({"folder_name": folder_name, "vendor": vendor, "status": "error", "error": str(exc)})
            continue
        pred_rows = [_result_to_row(result, source) for result, source in zip(analysis.rows, image_to_source)]
        envelopes = _build_envelopes(
            pred_rows,
            min_match_score=float(args.min_match_score),
            folder_name=folder_name,
            vendor=vendor,
        )
        all_prediction_rows.extend(pred_rows)
        all_envelopes.extend(envelopes)
        folder_summaries.append(
            {
                "folder_name": folder_name,
                "vendor": vendor,
                "status": "ok",
                "images": len(images),
                "selected_template_name": analysis.template.name,
                "selected_template_path": str(analysis.template.path),
                "predictions": len(pred_rows),
                "envelopes": len(envelopes),
                "missing_envelopes": [g for g in ORIENTATION_ORDER if g not in {e.orientation_group for e in envelopes}],
            }
        )
        overlay_rows = _pick_overlay_rows(pred_rows, int(args.overlay_limit_per_folder))
        env_by_folder = [env for env in envelopes if env.folder_name == folder_name]
        for idx, row in enumerate(overlay_rows):
            safe_folder = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in folder_name)[:90] or "folder"
            _draw_overlay(row, env_by_folder, out / "overlays" / safe_folder / f"{idx:04d}.jpg")

    _write_csv(out / "bundle_marker_per_image_predictions.csv", all_prediction_rows)
    _write_csv(out / "bundle_orientation_boxes.csv", [asdict(env) for env in all_envelopes])
    _write_csv(out / "folder_summary.csv", folder_summaries)
    (out / "bundle_orientation_boxes.json").write_text(
        json.dumps(
            {
                "source_su_giu_csv": args.su_giu_csv.expanduser().resolve().as_posix(),
                "bundle_root": bundle_root.as_posix(),
                "library_root": args.library_root.expanduser().resolve().as_posix(),
                "params": vars(args),
                "orientation_boxes": [asdict(env) for env in all_envelopes],
                "folder_summary": folder_summaries,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": out.as_posix(),
                "folders": len(by_folder),
                "prediction_rows": len(all_prediction_rows),
                "orientation_boxes": len(all_envelopes),
                "summary_csv": (out / "folder_summary.csv").as_posix(),
                "boxes_csv": (out / "bundle_orientation_boxes.csv").as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
