#!/usr/bin/env python3
"""OCR refine for probe folder predictions on review-only rows."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def _normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _iter_images(image_dir: Path) -> List[Path]:
    images: List[Path] = []
    if not image_dir.is_dir():
        return images
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if "negative" in path.name.lower():
            continue
        images.append(path)
    return images


def _select_subset(paths: Sequence[Path], limit: int) -> List[Path]:
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    if limit == 1:
        return [paths[len(paths) // 2]]
    total = len(paths)
    selected_idx: List[int] = []
    seen = set()
    for i in range(limit):
        idx = round(i * (total - 1) / (limit - 1))
        if idx not in seen:
            selected_idx.append(idx)
            seen.add(idx)
    if len(selected_idx) < limit:
        for idx in range(total):
            if idx in seen:
                continue
            selected_idx.append(idx)
            seen.add(idx)
            if len(selected_idx) >= limit:
                break
    selected_idx.sort()
    return [paths[i] for i in selected_idx]


def _run_ocr_text(image_path: Path, lang: str, psm: int, timeout_sec: float) -> str:
    try:
        proc = subprocess.run(
            [
                "tesseract",
                str(image_path),
                "stdout",
                "--oem",
                "1",
                "--psm",
                str(psm),
                "-l",
                lang,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except Exception:
        return ""
    return proc.stdout or ""


def _extract_probe_candidates(model_name: str) -> Set[str]:
    up = model_name.upper()
    raw_tokens: Set[str] = set()
    for tok in re.split(r"[^A-Z0-9]+", up):
        if tok:
            raw_tokens.add(tok)
    for tok in re.findall(r"[A-Z]{1,4}\d[A-Z0-9\-]{0,8}|\d[A-Z]{1,4}[A-Z0-9\-]{0,8}", up):
        tok = tok.strip("-")
        if tok:
            raw_tokens.add(tok)
    return {_normalize(tok) for tok in raw_tokens if tok}


def _build_probe_keyword_map(
    manifest_path: Path,
    min_len: int,
    max_len: int,
    max_keywords_per_probe: int,
) -> Dict[str, List[str]]:
    stop_prefixes = (
        "SW",
        "REV",
        "VER",
        "BUILD",
        "MYLAB",
        "LOGIQ",
        "VOLUSON",
        "ARIETTA",
        "APLIO",
        "AFFINITI",
        "EPIQ",
        "CONSONA",
        "RESONA",
        "FLEXFOCUS",
        "PROFOCUS",
    )

    unique_pairs: Set[Tuple[str, str]] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            probe_id = (row.get("fss_id_probe") or "").strip()
            model_name = (row.get("model_name") or "").strip()
            if not probe_id or not model_name:
                continue
            unique_pairs.add((probe_id, model_name))

    probe_token_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for probe_id, model_name in unique_pairs:
        candidates = _extract_probe_candidates(model_name)
        for tok in candidates:
            if len(tok) < min_len or len(tok) > max_len:
                continue
            if tok.startswith(stop_prefixes):
                continue
            if not any(ch.isalpha() for ch in tok) or not any(ch.isdigit() for ch in tok):
                continue
            if re.match(r"^F\d{5,}$", tok):
                continue
            if re.match(r"^\d{4,}$", tok):
                continue
            probe_token_counts[probe_id][tok] += 1

    token_to_probes: Dict[str, Set[str]] = defaultdict(set)
    for probe_id, counts in probe_token_counts.items():
        for tok in counts:
            token_to_probes[tok].add(probe_id)

    keyword_map: Dict[str, List[str]] = {}
    for probe_id, counts in probe_token_counts.items():
        unique_tokens = [
            (tok, count)
            for tok, count in counts.items()
            if len(token_to_probes[tok]) == 1
        ]
        unique_tokens.sort(key=lambda x: (-x[1], -len(x[0]), x[0]))
        selected = [tok for tok, _ in unique_tokens[:max_keywords_per_probe]]
        if selected:
            keyword_map[probe_id] = selected
    return keyword_map


def _ocr_probe_hits(
    image_paths: Sequence[Path],
    keyword_map: Dict[str, Sequence[str]],
    lang: str,
    psm: int,
    timeout_sec: float,
) -> Counter[str]:
    hits: Counter[str] = Counter()
    for image_path in image_paths:
        raw = _run_ocr_text(image_path, lang=lang, psm=psm, timeout_sec=timeout_sec)
        if not raw:
            continue
        text_norm = _normalize(raw)
        if not text_norm:
            continue
        for probe_id, keywords in keyword_map.items():
            if any(keyword in text_norm for keyword in keywords):
                hits[probe_id] += 1
    return hits


def _parse_topk_probe_ids(topk_field: str) -> Set[str]:
    out: Set[str] = set()
    for part in topk_field.split("|"):
        bit = part.strip()
        if not bit:
            continue
        if ":" in bit:
            out.add(bit.split(":", 1)[0].strip())
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refine probe folder predictions with OCR on review rows."
    )
    parser.add_argument(
        "--folder-csv",
        type=Path,
        default=Path("artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/folder_probe_predictions.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/10_active_pipeline/pipeline_fss_head/references/rect_dataset_no_negative_v2/manifest_rect_echo.csv"),
    )
    parser.add_argument("--ocr-lang", type=str, default="eng")
    parser.add_argument("--ocr-psm", type=int, default=6)
    parser.add_argument("--ocr-timeout-sec", type=float, default=5.0)
    parser.add_argument("--ocr-max-images-per-folder", type=int, default=24)
    parser.add_argument("--ocr-review-min-hit-count", type=int, default=2)
    parser.add_argument("--ocr-review-min-hit-ratio", type=float, default=0.20)
    parser.add_argument("--ocr-min-token-len", type=int, default=4)
    parser.add_argument("--ocr-max-token-len", type=int, default=10)
    parser.add_argument("--ocr-max-keywords-per-probe", type=int, default=8)
    parser.add_argument("--disable-topk-guard", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1"),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.ocr_max_images_per_folder <= 0:
        raise ValueError("--ocr-max-images-per-folder must be > 0")
    if args.ocr_review_min_hit_count < 1:
        raise ValueError("--ocr-review-min-hit-count must be >= 1")
    if not (0.0 <= args.ocr_review_min_hit_ratio <= 1.0):
        raise ValueError("--ocr-review-min-hit-ratio must be in [0,1]")
    if args.ocr_min_token_len < 2 or args.ocr_max_token_len < args.ocr_min_token_len:
        raise ValueError("Invalid OCR token length range.")

    folder_csv = args.folder_csv.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    keyword_map = _build_probe_keyword_map(
        manifest_path=manifest_path,
        min_len=args.ocr_min_token_len,
        max_len=args.ocr_max_token_len,
        max_keywords_per_probe=args.ocr_max_keywords_per_probe,
    )

    rows: List[Dict[str, str]] = []
    with folder_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row["ocr_top_probe_id"] = ""
            row["ocr_hit_count"] = "0"
            row["ocr_hit_ratio"] = "0.000000"
            row["ocr_images_checked"] = "0"
            row["ocr_strong"] = "0"
            row["recommended_probe_id"] = row.get("predicted_probe_id", "")
            row["recommended_source_policy"] = "predicted_probe_id"
            rows.append(row)

    review_processed = 0
    review_with_any_hit = 0
    review_strong = 0
    overrides = 0
    warnings: List[str] = []

    for row in rows:
        if row.get("status") != "review":
            continue

        folder_path = Path(row["folder_path"])
        images = _iter_images(folder_path / "image_samples")
        if not images:
            warnings.append(f"{folder_path.as_posix()}: no valid images")
            continue
        sample = _select_subset(images, limit=args.ocr_max_images_per_folder)

        hits = _ocr_probe_hits(
            image_paths=sample,
            keyword_map=keyword_map,
            lang=args.ocr_lang,
            psm=args.ocr_psm,
            timeout_sec=args.ocr_timeout_sec,
        )
        review_processed += 1
        row["ocr_images_checked"] = str(len(sample))
        if not hits:
            continue

        review_with_any_hit += 1
        top_probe, top_count = hits.most_common(1)[0]
        hit_ratio = float(top_count / max(1, len(sample)))
        row["ocr_top_probe_id"] = top_probe
        row["ocr_hit_count"] = str(int(top_count))
        row["ocr_hit_ratio"] = f"{hit_ratio:.6f}"

        strong = (
            top_count >= args.ocr_review_min_hit_count
            and hit_ratio >= args.ocr_review_min_hit_ratio
        )
        if strong:
            row["ocr_strong"] = "1"
            review_strong += 1

        if not strong:
            continue

        topk_guard_ok = True
        if not args.disable_topk_guard:
            topk_ids = _parse_topk_probe_ids(row.get("topk", ""))
            topk_guard_ok = top_probe in topk_ids

        if topk_guard_ok and top_probe and top_probe != row.get("predicted_probe_id", ""):
            row["recommended_probe_id"] = top_probe
            row["recommended_source_policy"] = "ocr_override_on_review"
            overrides += 1

    all_refined_csv = output_dir / "folder_probe_predictions_ocr_refined.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with all_refined_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    review_refined_csv = output_dir / "folder_probe_predictions_review_ocr_refined.csv"
    with review_refined_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row.get("status") == "review":
                writer.writerow(row)

    recommended_csv = output_dir / "folder_probe_for_fss_recommended_ocr.csv"
    with recommended_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["folder_path", "recommended_probe_id", "source_policy"])
        for row in rows:
            writer.writerow(
                [
                    row["folder_path"],
                    row["recommended_probe_id"],
                    row["recommended_source_policy"],
                ]
            )

    keyword_map_json = output_dir / "probe_ocr_keyword_map.json"
    keyword_map_json.write_text(json.dumps(keyword_map, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "folder_csv": folder_csv.as_posix(),
        "manifest": manifest_path.as_posix(),
        "review_rows_total": sum(1 for row in rows if row.get("status") == "review"),
        "review_rows_processed": review_processed,
        "review_rows_with_any_ocr_hit": review_with_any_hit,
        "review_rows_with_strong_ocr_hit": review_strong,
        "recommended_overrides": overrides,
        "topk_guard_enabled": not bool(args.disable_topk_guard),
        "ocr": {
            "lang": args.ocr_lang,
            "psm": args.ocr_psm,
            "timeout_sec": args.ocr_timeout_sec,
            "max_images_per_folder": args.ocr_max_images_per_folder,
            "review_min_hit_count": args.ocr_review_min_hit_count,
            "review_min_hit_ratio": args.ocr_review_min_hit_ratio,
            "min_token_len": args.ocr_min_token_len,
            "max_token_len": args.ocr_max_token_len,
            "max_keywords_per_probe": args.ocr_max_keywords_per_probe,
        },
        "keyword_map_probe_count": len(keyword_map),
        "output_files": {
            "all_refined_csv": all_refined_csv.as_posix(),
            "review_refined_csv": review_refined_csv.as_posix(),
            "recommended_csv": recommended_csv.as_posix(),
            "keyword_map_json": keyword_map_json.as_posix(),
        },
        "warnings_count": len(warnings),
        "warnings": warnings[:200],
    }
    summary_json = output_dir / "summary_probe_ocr_refine.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Review rows total: {summary['review_rows_total']}", flush=True)
    print(f"OCR processed review rows: {review_processed}", flush=True)
    print(f"OCR any-hit rows: {review_with_any_hit}", flush=True)
    print(f"OCR strong-hit rows: {review_strong}", flush=True)
    print(f"Recommended OCR overrides: {overrides}", flush=True)
    print(f"Refined all rows: {all_refined_csv}", flush=True)
    print(f"Refined review rows: {review_refined_csv}", flush=True)
    print(f"Recommended for .fss: {recommended_csv}", flush=True)
    print(f"Keyword map: {keyword_map_json}", flush=True)
    print(f"Summary: {summary_json}", flush=True)
    if warnings:
        print(f"Warnings: {len(warnings)} (see summary)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
