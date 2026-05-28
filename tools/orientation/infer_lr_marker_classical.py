#!/usr/bin/env python3
"""Classical LR marker inference using DB_echo orientation templates.

This is the production-oriented version of the reviewed dataset builder logic:
detect the orientation marker with normalized cross-correlation and infer
normal vs right-mirrored from the marker side.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile

from prepare_lr_marker_dataset_sugiu_v2 import (
    LocatedMatch,
    TemplateRow,
    _choose_device,
    _clip_rect,
    _collect_templates,
    _expand_rect,
    _extract_rect_line,
    _load_sugiu_model,
    _load_template,
    _locate_match_in_abs_rect,
    _lr_from_detected_side,
    _parse_float_steps,
    _parse_rect_line,
    _predict_sugiu,
    _roi_from_sugiu,
    _score_of,
    infer_manufacturer,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class ClassicalConfig:
    config_path: Path
    manufacturer: str
    fss_path: Path
    setup_token: str
    echo_rect: Tuple[int, int, int, int]
    templates: Tuple[TemplateRow, ...]
    template_paths_total: int


def _load_config(
    config_path: Path,
    device: torch.device,
    blank_template_max_value: int,
    exclude_vendors: Sequence[str],
) -> ClassicalConfig:
    config_path = config_path.expanduser().resolve()
    manufacturer = infer_manufacturer(config_path.name)
    excluded = {vendor.strip().lower() for vendor in exclude_vendors if vendor.strip()}
    if manufacturer.lower() in excluded:
        raise RuntimeError(f"Vendor excluded from classical LR inference: {manufacturer}")

    fss_files = sorted(p for p in (config_path / "DB_setup").glob("*.fss") if p.is_file())
    if len(fss_files) != 1:
        raise RuntimeError(f"Expected exactly one .fss in {config_path / 'DB_setup'}, found {len(fss_files)}")
    fss_path = fss_files[0].resolve()
    lines = fss_path.read_text(encoding="utf-8", errors="replace").splitlines()
    rect_line = _extract_rect_line(lines)
    echo_rect = _parse_rect_line(lines, rect_line)
    setup_token = fss_path.stem.split("_")[-1]

    template_paths = _collect_templates(config_path, setup_token)
    templates: List[TemplateRow] = []
    for path in template_paths:
        template = _load_template(path, device=device, blank_template_max_value=blank_template_max_value)
        if template is not None:
            templates.append(template)
    if not templates:
        raise RuntimeError(
            "No usable orientation templates after black/near-black filtering "
            f"(found {len(template_paths)} paths)."
        )

    return ClassicalConfig(
        config_path=config_path,
        manufacturer=manufacturer,
        fss_path=fss_path,
        setup_token=setup_token,
        echo_rect=echo_rect,
        templates=tuple(templates),
        template_paths_total=len(template_paths),
    )


def _fmt_score(value: Optional[LocatedMatch]) -> str:
    return "" if value is None else f"{value.score:.6f}"


def infer_image(
    image_path: Path,
    config: ClassicalConfig,
    sugiu_model,
    sugiu_image_size: int,
    device: torch.device,
    min_match_score: float,
    full_crop_fallback_threshold: float,
    expanded_search_threshold: float,
    expanded_search_steps: Sequence[float],
    min_sugiu_confidence: float,
) -> Dict[str, object]:
    image_path = image_path.expanduser().resolve()
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        rect = _clip_rect(config.echo_rect, width=width, height=height)
        if rect is None:
            raise RuntimeError(f"Bad echo rect for image size {width}x{height}: {config.echo_rect}")

        top, left, bottom, right = rect
        crop = rgb.crop((left, top, right + 1, bottom + 1))
        crop_width, crop_height = crop.size
        sugiu_pred, sugiu_conf, prob_su, prob_giu = _predict_sugiu(
            model=sugiu_model,
            crop=crop,
            image_size=sugiu_image_size,
            device=device,
        )
        roi_top, roi_left, roi_bottom, roi_right = _roi_from_sugiu(crop_width, crop_height, sugiu_pred)
        gray_full = torch.from_numpy(np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0).to(
            device=device,
            dtype=torch.float32,
        )

        initial_search_rect_abs = (top + roi_top, left + roi_left, top + roi_bottom, left + roi_right)
        initial_loc = _locate_match_in_abs_rect(
            gray_full=gray_full,
            search_rect_abs=initial_search_rect_abs,
            echo_rect_abs=rect,
            templates=config.templates,
            search_strategy="sugiu_roi",
            search_scope="predicted_half",
            search_margin_px=0,
        )
        best_loc = initial_loc
        full_crop_loc: Optional[LocatedMatch] = None

        if _score_of(best_loc) < float(full_crop_fallback_threshold):
            full_crop_loc = _locate_match_in_abs_rect(
                gray_full=gray_full,
                search_rect_abs=rect,
                echo_rect_abs=rect,
                templates=config.templates,
                search_strategy="full_crop_low_score_fallback",
                search_scope="full_echo_crop",
                search_margin_px=0,
            )
            if _score_of(full_crop_loc) > _score_of(best_loc):
                best_loc = full_crop_loc

        if _score_of(best_loc) < float(expanded_search_threshold):
            for step in expanded_search_steps:
                margin_px = max(1, int(round(max(crop_width, crop_height) * float(step))))
                expanded_rect = _expand_rect(initial_search_rect_abs, width=width, height=height, margin_px=margin_px)
                if expanded_rect is None:
                    continue
                expanded_loc = _locate_match_in_abs_rect(
                    gray_full=gray_full,
                    search_rect_abs=expanded_rect,
                    echo_rect_abs=rect,
                    templates=config.templates,
                    search_strategy="expanded_rect_fallback",
                    search_scope="expanded_predicted_half",
                    search_margin_px=margin_px,
                )
                if _score_of(expanded_loc) > _score_of(best_loc):
                    best_loc = expanded_loc
                if _score_of(best_loc) >= float(expanded_search_threshold):
                    break

        result: Dict[str, object] = {
            "image_path": image_path.as_posix(),
            "config_path": config.config_path.as_posix(),
            "manufacturer": config.manufacturer,
            "fss_path": config.fss_path.as_posix(),
            "setup_token": config.setup_token,
            "template_paths_total": config.template_paths_total,
            "usable_templates": len(config.templates),
            "su_giu_pred": sugiu_pred,
            "su_giu_conf": f"{sugiu_conf:.6f}",
            "prob_su": f"{prob_su:.6f}",
            "prob_giu": f"{prob_giu:.6f}",
            "initial_match_score": _fmt_score(initial_loc),
            "full_crop_fallback_score": _fmt_score(full_crop_loc),
            "echo_rect_top_abs": top,
            "echo_rect_left_abs": left,
            "echo_rect_bottom_abs": bottom,
            "echo_rect_right_abs": right,
            "echo_rect_width": crop_width,
            "echo_rect_height": crop_height,
            "roi_top_crop": roi_top,
            "roi_left_crop": roi_left,
            "roi_bottom_crop": roi_bottom,
            "roi_right_crop": roi_right,
        }
        if best_loc is None:
            result.update(
                {
                    "status": "review",
                    "review_reason": "no_template_match",
                    "lr_label": "",
                    "lr_label_it": "",
                    "lr_binary": -1,
                    "detected_marker_side": "",
                    "match_score": "",
                }
            )
            return result

        marker_cx = (best_loc.marker_left_crop + best_loc.marker_right_crop) / 2.0
        marker_cy = (best_loc.marker_top_crop + best_loc.marker_bottom_crop) / 2.0
        detected_side = "left" if marker_cx < (crop_width / 2.0) else "right"
        lr_label, lr_label_it, lr_binary = _lr_from_detected_side(detected_side)
        reasons: List[str] = []
        if sugiu_conf < float(min_sugiu_confidence):
            reasons.append("low_sugiu_conf")
        if best_loc.score < float(min_match_score):
            reasons.append("low_template_score")
        status = "ok" if not reasons else "review"
        result.update(
            {
                "status": status,
                "review_reason": ";".join(reasons),
                "lr_label": lr_label,
                "lr_label_it": lr_label_it,
                "lr_binary": lr_binary,
                "detected_marker_side": detected_side,
                "match_score": f"{best_loc.score:.6f}",
                "search_strategy": best_loc.search_strategy,
                "search_scope": best_loc.search_scope,
                "search_margin_px": best_loc.search_margin_px,
                "search_rect_top_abs": best_loc.search_rect_abs[0],
                "search_rect_left_abs": best_loc.search_rect_abs[1],
                "search_rect_bottom_abs": best_loc.search_rect_abs[2],
                "search_rect_right_abs": best_loc.search_rect_abs[3],
                "template_path": best_loc.template.path.as_posix(),
                "template_width": best_loc.template.width,
                "template_height": best_loc.template.height,
                "marker_top_crop": best_loc.marker_top_crop,
                "marker_left_crop": best_loc.marker_left_crop,
                "marker_bottom_crop": best_loc.marker_bottom_crop,
                "marker_right_crop": best_loc.marker_right_crop,
                "marker_top_abs": best_loc.marker_top_abs,
                "marker_left_abs": best_loc.marker_left_abs,
                "marker_bottom_abs": best_loc.marker_bottom_abs,
                "marker_right_abs": best_loc.marker_right_abs,
                "marker_cx_crop_norm": f"{marker_cx / max(1, crop_width):.6f}",
                "marker_cy_crop_norm": f"{marker_cy / max(1, crop_height):.6f}",
            }
        )
        return result


def _manifest_rows(path: Path, limit: int) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if limit > 0:
        rows = rows[:limit]
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_manifest(args: argparse.Namespace) -> Dict[str, object]:
    device = _choose_device(args.device)
    sugiu_model, sugiu_image_size = _load_sugiu_model(args.sugiu_checkpoint.expanduser().resolve(), device=device)
    steps = _parse_float_steps(str(args.expanded_search_steps))
    rows = _manifest_rows(args.manifest.expanduser().resolve(), limit=int(args.limit))

    config_cache: Dict[str, ClassicalConfig] = {}
    predictions: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []
    for idx, row in enumerate(rows, start=1):
        config_path = row.get("config_path", "")
        image_path = row.get("image_path", "")
        if not config_path or not image_path:
            errors.append({"row": idx, "error": "missing config_path or image_path"})
            continue
        try:
            if config_path not in config_cache:
                config_cache[config_path] = _load_config(
                    Path(config_path),
                    device=device,
                    blank_template_max_value=int(args.blank_template_max_value),
                    exclude_vendors=tuple(args.exclude_vendors),
                )
            pred = infer_image(
                image_path=Path(image_path),
                config=config_cache[config_path],
                sugiu_model=sugiu_model,
                sugiu_image_size=sugiu_image_size,
                device=device,
                min_match_score=float(args.min_match_score),
                full_crop_fallback_threshold=float(args.full_crop_fallback_threshold),
                expanded_search_threshold=float(args.expanded_search_threshold),
                expanded_search_steps=steps,
                min_sugiu_confidence=float(args.min_sugiu_confidence),
            )
            expected = row.get("lr_label", "")
            pred["manifest_sample_id"] = row.get("sample_id", "")
            pred["expected_lr_label"] = expected
            pred["expected_match"] = int(bool(expected) and expected == pred.get("lr_label"))
            predictions.append(pred)
        except Exception as exc:  # pylint: disable=broad-except
            errors.append({"row": idx, "image_path": image_path, "config_path": config_path, "error": str(exc)})

    if args.output_csv:
        _write_csv(args.output_csv.expanduser().resolve(), predictions)

    total_with_expected = sum(1 for p in predictions if p.get("expected_lr_label"))
    matching = sum(1 for p in predictions if p.get("expected_match") == 1)
    summary = {
        "mode": "manifest",
        "manifest": args.manifest.expanduser().resolve().as_posix(),
        "rows_input": len(rows),
        "predictions": len(predictions),
        "errors": len(errors),
        "expected_rows": total_with_expected,
        "expected_matches": matching,
        "expected_accuracy": (matching / total_with_expected) if total_with_expected else None,
        "status_counts": _counts(predictions, "status"),
        "lr_label_counts": _counts(predictions, "lr_label"),
        "search_strategy_counts": _counts(predictions, "search_strategy"),
        "errors_preview": errors[:20],
        "output_csv": "" if not args.output_csv else args.output_csv.expanduser().resolve().as_posix(),
    }
    if args.output_json:
        args.output_json.expanduser().resolve().write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _counts(rows: Sequence[Dict[str, object]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "") or "(empty)")
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def _run_single(args: argparse.Namespace) -> Dict[str, object]:
    if args.config_dir is None or args.image is None:
        raise RuntimeError("Single-image mode requires --image and --config-dir.")
    device = _choose_device(args.device)
    sugiu_model, sugiu_image_size = _load_sugiu_model(args.sugiu_checkpoint.expanduser().resolve(), device=device)
    config = _load_config(
        args.config_dir.expanduser().resolve(),
        device=device,
        blank_template_max_value=int(args.blank_template_max_value),
        exclude_vendors=tuple(args.exclude_vendors),
    )
    result = infer_image(
        image_path=args.image.expanduser().resolve(),
        config=config,
        sugiu_model=sugiu_model,
        sugiu_image_size=sugiu_image_size,
        device=device,
        min_match_score=float(args.min_match_score),
        full_crop_fallback_threshold=float(args.full_crop_fallback_threshold),
        expanded_search_threshold=float(args.expanded_search_threshold),
        expanded_search_steps=_parse_float_steps(str(args.expanded_search_steps)),
        min_sugiu_confidence=float(args.min_sugiu_confidence),
    )
    if args.output_json:
        args.output_json.expanduser().resolve().write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classical LR marker inference.")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--sugiu-checkpoint",
        type=Path,
        default=Path("artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/model_su_giu_rect/best_model.pt"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--exclude-vendors", nargs="*", default=["Biopsee"])
    parser.add_argument("--min-match-score", type=float, default=0.45)
    parser.add_argument("--full-crop-fallback-threshold", type=float, default=0.55)
    parser.add_argument("--expanded-search-threshold", type=float, default=0.55)
    parser.add_argument("--expanded-search-steps", type=str, default="0.03,0.06,0.10,0.15,0.20")
    parser.add_argument("--blank-template-max-value", type=int, default=3)
    parser.add_argument("--min-sugiu-confidence", type=float, default=0.80)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.manifest is not None:
        result = _run_manifest(args)
    else:
        result = _run_single(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
