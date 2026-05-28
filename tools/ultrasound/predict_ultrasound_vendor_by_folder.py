#!/usr/bin/env python3
"""Predict one manufacturer per dataset folder for downstream .fss generation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from prepare_ultrasound_rect_dataset import infer_manufacturer
from train_ultrasound_vendor_classifier import VendorClassifier, choose_device


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def _build_ocr_keyword_map(class_names: Sequence[str]) -> Dict[str, List[str]]:
    base_map: Dict[str, List[str]] = {
        "BK": ["bk", "profocus", "flexfocus"],
        "Esaote": ["esaote", "mylab"],
        "Hitachi": ["hitachi", "arietta"],
        "GE": ["logiq", "voluson"],
        "Mindray": ["mindray", "resona", "te7"],
        "Canon": ["canon", "aplio"],
        "Philips": ["philips", "affiniti", "epiq"],
        "Toshiba": ["toshiba", "aplio"],
        "Siemens": ["siemens", "acuson"],
        "Koelis": ["koelis"],
        "Biopsee": ["biopsee", "biojet"],
        "ExactVu": ["exactvu"],
        "Alpinion": ["alpinion", "ecube"],
        "Terason": ["terason"],
        "Sonostar": ["sonostar"],
        "Vinno": ["vinno"],
    }
    return {name: base_map.get(name, [name.lower()]) for name in class_names}


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
    return (proc.stdout or "").lower()


def _ocr_vendor_hits(
    image_paths: Sequence[Path],
    keyword_map: Dict[str, Sequence[str]],
    lang: str,
    psm: int,
    timeout_sec: float,
) -> Counter[str]:
    hits: Counter[str] = Counter()
    for image_path in image_paths:
        text = _run_ocr_text(image_path, lang=lang, psm=psm, timeout_sec=timeout_sec)
        if not text:
            continue
        for vendor, keywords in keyword_map.items():
            if any(keyword in text for keyword in keywords):
                hits[vendor] += 1
    return hits


def _iter_images(image_dir: Path, exclude_name_re: Optional[re.Pattern[str]]) -> Tuple[List[Path], int]:
    images: List[Path] = []
    excluded = 0
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if exclude_name_re and exclude_name_re.search(path.name):
            excluded += 1
            continue
        images.append(path)
    return images, excluded


def _select_subset(paths: Sequence[Path], limit: int) -> List[Path]:
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    if limit == 1:
        return [paths[len(paths) // 2]]

    # Uniformly sample across the whole folder timeline.
    total = len(paths)
    selected_idx = []
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


def _load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as img:
        image = img.convert("RGB")
        image = TF.resize(
            image,
            size=[image_size, image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(
            tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    return tensor


def _predict_probabilities(
    model: torch.nn.Module,
    image_paths: Sequence[Path],
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if not image_paths:
        return np.zeros((0, 0), dtype=np.float32)

    all_probs: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch_tensors = [_load_image_tensor(path, image_size=image_size) for path in batch_paths]
            images = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            all_probs.append(probs.astype(np.float32))
    return np.concatenate(all_probs, axis=0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict one vendor/manufacturer per dataset folder."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset"),
        help="Root cartella Dataset (default: Dataset).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/10_active_pipeline/pipeline_fss_head/models/vendor_training_no_negative_v2_power/best_model.pt"),
        help="Checkpoint best_model.pt del classificatore vendor.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/40_outputs_eval/vendor_folder_predictions"),
        help="Cartella output report folder-level.",
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=0, help="Compat arg (non usato).")
    parser.add_argument("--image-size", type=int, default=0, help="Override image size. 0=dal checkpoint.")
    parser.add_argument(
        "--sample-per-folder",
        type=int,
        default=0,
        help="Numero massimo immagini per cartella (0=tutte).",
    )
    parser.add_argument("--topk", type=int, default=3, help="Top-k classi da esportare.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (es. cpu, cuda, mps).",
    )
    parser.add_argument(
        "--exclude-folder-regex",
        type=str,
        default=None,
        help="Regex opzionale per escludere cartelle.",
    )
    parser.add_argument(
        "--exclude-image-regex",
        type=str,
        default="(?i)negative",
        help="Regex immagini da ignorare per nome file.",
    )
    parser.add_argument(
        "--min-folder-confidence",
        type=float,
        default=0.55,
        help="Soglia minima sulla probabilita media top-1.",
    )
    parser.add_argument(
        "--min-folder-margin",
        type=float,
        default=0.10,
        help="Soglia minima su (top1 - top2) della probabilita media folder.",
    )
    parser.add_argument(
        "--min-vote-ratio",
        type=float,
        default=0.50,
        help="Soglia minima su percentuale immagini che votano la classe finale.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        help="Log progresso ogni N cartelle.",
    )
    parser.add_argument(
        "--enable-ocr-review-refine",
        action="store_true",
        help="Applica OCR solo sui folder in review per rifinire il vendor raccomandato.",
    )
    parser.add_argument("--ocr-lang", type=str, default="eng")
    parser.add_argument("--ocr-psm", type=int, default=6)
    parser.add_argument("--ocr-timeout-sec", type=float, default=5.0)
    parser.add_argument("--ocr-max-images-per-folder", type=int, default=24)
    parser.add_argument("--ocr-review-min-hit-count", type=int, default=2)
    parser.add_argument("--ocr-review-min-hit-ratio", type=float, default=0.20)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_size <= 0:
        raise ValueError("--batch-size deve essere > 0.")
    if args.sample_per_folder < 0:
        raise ValueError("--sample-per-folder deve essere >= 0.")
    if not (0.0 <= args.min_folder_confidence <= 1.0):
        raise ValueError("--min-folder-confidence deve essere in [0, 1].")
    if not (0.0 <= args.min_vote_ratio <= 1.0):
        raise ValueError("--min-vote-ratio deve essere in [0, 1].")
    if args.min_folder_margin < 0.0:
        raise ValueError("--min-folder-margin deve essere >= 0.")
    if args.ocr_max_images_per_folder <= 0:
        raise ValueError("--ocr-max-images-per-folder deve essere > 0.")
    if args.ocr_review_min_hit_count < 1:
        raise ValueError("--ocr-review-min-hit-count deve essere >= 1.")
    if not (0.0 <= args.ocr_review_min_hit_ratio <= 1.0):
        raise ValueError("--ocr-review-min-hit-ratio deve essere in [0, 1].")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise RuntimeError("Checkpoint senza class_names.")
    class_names = list(class_names)

    image_size = int(args.image_size)
    if image_size <= 0:
        image_size = int(checkpoint.get("args", {}).get("image_size", 320))

    device = choose_device(args.device)
    model = VendorClassifier(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    exclude_folder_re = re.compile(args.exclude_folder_regex) if args.exclude_folder_regex else None
    exclude_image_re = re.compile(args.exclude_image_regex) if args.exclude_image_regex else None

    folders = [
        p for p in sorted(dataset_root.iterdir())
        if p.is_dir() and (p / "image_samples").is_dir()
        and (exclude_folder_re is None or not exclude_folder_re.search(p.name))
    ]
    if not folders:
        raise RuntimeError("Nessuna cartella valida trovata in dataset-root.")

    folder_rows: List[Dict[str, object]] = []
    image_rows: List[Dict[str, object]] = []
    warnings: List[str] = []

    total_images_available = 0
    total_images_excluded = 0
    total_images_used = 0
    review_count = 0
    status_counter: Counter[str] = Counter()
    pred_counter: Counter[str] = Counter()
    folder_selected_images: Dict[str, List[Path]] = {}
    ocr_keyword_map = _build_ocr_keyword_map(class_names)
    ocr_review_processed = 0
    ocr_review_with_hits = 0
    ocr_review_strong = 0

    start_time = time.time()
    for idx, folder in enumerate(folders, start=1):
        image_dir = folder / "image_samples"
        all_images, excluded_here = _iter_images(image_dir, exclude_name_re=exclude_image_re)
        total_images_excluded += excluded_here
        total_images_available += len(all_images) + excluded_here

        if not all_images:
            warnings.append(f"{folder.name}: nessuna immagine valida.")
            continue

        selected_images = _select_subset(all_images, limit=args.sample_per_folder)
        folder_selected_images[folder.as_posix()] = selected_images
        probs = _predict_probabilities(
            model=model,
            image_paths=selected_images,
            image_size=image_size,
            batch_size=args.batch_size,
            device=device,
        )
        if probs.size == 0:
            warnings.append(f"{folder.name}: impossibile calcolare probabilita.")
            continue

        total_images_used += len(selected_images)
        mean_probs = probs.mean(axis=0)
        sorted_idx = np.argsort(mean_probs)[::-1]
        pred_idx = int(sorted_idx[0])
        pred_label = class_names[pred_idx]
        pred_counter[pred_label] += 1

        top2_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else pred_idx
        top1_prob = float(mean_probs[pred_idx])
        top2_prob = float(mean_probs[top2_idx]) if top2_idx != pred_idx else 0.0
        margin = float(top1_prob - top2_prob)

        image_vote_idx = np.argmax(probs, axis=1)
        vote_ratio = float((image_vote_idx == pred_idx).mean())

        reasons: List[str] = []
        if top1_prob < args.min_folder_confidence:
            reasons.append(f"top1_prob<{args.min_folder_confidence:.2f}")
        if margin < args.min_folder_margin:
            reasons.append(f"margin<{args.min_folder_margin:.2f}")
        if vote_ratio < args.min_vote_ratio:
            reasons.append(f"vote_ratio<{args.min_vote_ratio:.2f}")

        status = "ok" if not reasons else "review"
        status_counter[status] += 1
        if status == "review":
            review_count += 1

        folder_name_hint = infer_manufacturer(folder.name)
        hint_note = ""
        if folder_name_hint in class_names and folder_name_hint != pred_label:
            hint_note = f"name_hint={folder_name_hint}"
            if status == "review":
                reasons.append(hint_note)

        k = max(1, min(args.topk, len(class_names)))
        topk_items = []
        for rank in range(k):
            cls_idx = int(sorted_idx[rank])
            topk_items.append((class_names[cls_idx], float(mean_probs[cls_idx])))
        topk_str = " | ".join(f"{label}:{prob:.4f}" for label, prob in topk_items)

        folder_rows.append(
            {
                "folder_path": folder.as_posix(),
                "folder_name": folder.name,
                "images_total_valid": len(all_images),
                "images_used": len(selected_images),
                "predicted_vendor": pred_label,
                "folder_top1_prob": f"{top1_prob:.6f}",
                "folder_margin_top1_top2": f"{margin:.6f}",
                "folder_vote_ratio": f"{vote_ratio:.6f}",
                "status": status,
                "review_reasons": ",".join(reasons),
                "name_hint_vendor": folder_name_hint if folder_name_hint in class_names else "",
                "topk": topk_str,
                "ocr_top_vendor": "",
                "ocr_hit_count": "0",
                "ocr_hit_ratio": "0.000000",
                "ocr_images_checked": "0",
            }
        )

        for image_path, image_prob in zip(selected_images, probs):
            img_pred_idx = int(np.argmax(image_prob))
            img_pred = class_names[img_pred_idx]
            img_conf = float(image_prob[img_pred_idx])
            image_rows.append(
                {
                    "folder_path": folder.as_posix(),
                    "image_path": image_path.as_posix(),
                    "image_pred_vendor": img_pred,
                    "image_pred_confidence": f"{img_conf:.6f}",
                    "matches_folder_vendor": int(img_pred == pred_label),
                }
            )

        if args.log_interval > 0 and (idx % args.log_interval == 0 or idx == len(folders)):
            elapsed = time.time() - start_time
            print(
                f"[folder {idx}/{len(folders)}] elapsed {elapsed:.1f}s | "
                f"review {review_count} | avg images/folder "
                f"{(total_images_used / max(1, idx)):.1f}",
                flush=True,
            )

    if args.enable_ocr_review_refine:
        for row in folder_rows:
            if row["status"] != "review":
                continue
            folder_path = str(row["folder_path"])
            selected_images = folder_selected_images.get(folder_path, [])
            if not selected_images:
                continue
            ocr_images = _select_subset(selected_images, limit=args.ocr_max_images_per_folder)
            ocr_hits = _ocr_vendor_hits(
                image_paths=ocr_images,
                keyword_map=ocr_keyword_map,
                lang=args.ocr_lang,
                psm=args.ocr_psm,
                timeout_sec=args.ocr_timeout_sec,
            )
            ocr_review_processed += 1

            row["ocr_images_checked"] = str(len(ocr_images))
            if not ocr_hits:
                continue

            ocr_review_with_hits += 1
            top_vendor, top_count = ocr_hits.most_common(1)[0]
            hit_ratio = float(top_count / max(1, len(ocr_images)))
            row["ocr_top_vendor"] = top_vendor
            row["ocr_hit_count"] = str(int(top_count))
            row["ocr_hit_ratio"] = f"{hit_ratio:.6f}"
            if top_count >= args.ocr_review_min_hit_count and hit_ratio >= args.ocr_review_min_hit_ratio:
                ocr_review_strong += 1

    folder_csv = output_dir / "folder_vendor_predictions.csv"
    with folder_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "folder_path",
                "folder_name",
                "images_total_valid",
                "images_used",
                "predicted_vendor",
                "folder_top1_prob",
                "folder_margin_top1_top2",
                "folder_vote_ratio",
                "status",
                "review_reasons",
                "name_hint_vendor",
                "topk",
                "ocr_top_vendor",
                "ocr_hit_count",
                "ocr_hit_ratio",
                "ocr_images_checked",
            ],
        )
        writer.writeheader()
        writer.writerows(folder_rows)

    review_csv = output_dir / "folder_vendor_predictions_review.csv"
    with review_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "folder_path",
                "folder_name",
                "images_total_valid",
                "images_used",
                "predicted_vendor",
                "folder_top1_prob",
                "folder_margin_top1_top2",
                "folder_vote_ratio",
                "status",
                "review_reasons",
                "name_hint_vendor",
                "topk",
                "ocr_top_vendor",
                "ocr_hit_count",
                "ocr_hit_ratio",
                "ocr_images_checked",
            ],
        )
        writer.writeheader()
        for row in folder_rows:
            if row["status"] == "review":
                writer.writerow(row)

    for_fss_csv = output_dir / "folder_vendor_for_fss.csv"
    with for_fss_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["folder_path", "predicted_vendor"])
        for row in folder_rows:
            writer.writerow([row["folder_path"], row["predicted_vendor"]])

    # Recommended mapping for .fss generation:
    # Conservative policy:
    # 1) if review and OCR strongly agrees with folder-name hint (and differs from CNN), override with that vendor
    # 2) else if review and folder-name hint differs from CNN, override with folder-name hint
    # 3) else keep CNN prediction
    for_fss_recommended_csv = output_dir / "folder_vendor_for_fss_recommended.csv"
    recommended_overrides = 0
    recommended_ocr_hint_overrides = 0
    recommended_name_hint_overrides = 0
    with for_fss_recommended_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["folder_path", "recommended_vendor", "source_policy"])
        for row in folder_rows:
            vendor = str(row["predicted_vendor"])
            policy = "predicted_vendor"
            hint = str(row["name_hint_vendor"])
            ocr_vendor = str(row["ocr_top_vendor"])
            ocr_hit_count = int(str(row["ocr_hit_count"]))
            ocr_hit_ratio = float(str(row["ocr_hit_ratio"]))
            ocr_strong = (
                ocr_vendor
                and ocr_hit_count >= args.ocr_review_min_hit_count
                and ocr_hit_ratio >= args.ocr_review_min_hit_ratio
            )
            if (
                row["status"] == "review"
                and ocr_strong
                and hint
                and ocr_vendor == hint
                and hint != vendor
            ):
                vendor = ocr_vendor
                policy = "ocr_and_name_hint_override_on_review"
                recommended_overrides += 1
                recommended_ocr_hint_overrides += 1
            elif (
                row["status"] == "review"
                and hint
                and hint != vendor
            ):
                vendor = hint
                policy = "name_hint_override_on_review"
                recommended_overrides += 1
                recommended_name_hint_overrides += 1
            writer.writerow([row["folder_path"], vendor, policy])

    image_csv = output_dir / "per_image_predictions.csv"
    with image_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "folder_path",
                "image_path",
                "image_pred_vendor",
                "image_pred_confidence",
                "matches_folder_vendor",
            ],
        )
        writer.writeheader()
        writer.writerows(image_rows)

    summary = {
        "dataset_root": dataset_root.as_posix(),
        "checkpoint": checkpoint_path.as_posix(),
        "device": str(device),
        "image_size": image_size,
        "batch_size": args.batch_size,
        "sample_per_folder": args.sample_per_folder,
        "thresholds": {
            "min_folder_confidence": args.min_folder_confidence,
            "min_folder_margin": args.min_folder_margin,
            "min_vote_ratio": args.min_vote_ratio,
        },
        "folders_total": len(folder_rows),
        "folders_status": dict(status_counter),
        "recommended_overrides_from_name_hint": recommended_overrides,
        "recommended_overrides_breakdown": {
            "ocr_and_name_hint_override_on_review": recommended_ocr_hint_overrides,
            "name_hint_override_on_review": recommended_name_hint_overrides,
        },
        "ocr_review_refine": {
            "enabled": bool(args.enable_ocr_review_refine),
            "review_rows_processed": ocr_review_processed,
            "review_rows_with_any_ocr_hit": ocr_review_with_hits,
            "review_rows_with_strong_ocr_hit": ocr_review_strong,
            "ocr_lang": args.ocr_lang,
            "ocr_psm": args.ocr_psm,
            "ocr_timeout_sec": args.ocr_timeout_sec,
            "ocr_max_images_per_folder": args.ocr_max_images_per_folder,
            "ocr_review_min_hit_count": args.ocr_review_min_hit_count,
            "ocr_review_min_hit_ratio": args.ocr_review_min_hit_ratio,
        },
        "predicted_vendor_distribution": dict(sorted(pred_counter.items())),
        "images": {
            "available_including_excluded": total_images_available,
            "excluded_by_name_filter": total_images_excluded,
            "used_for_prediction": total_images_used,
        },
        "warnings_count": len(warnings),
        "warnings": warnings[:200],
    }
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Device: {device}", flush=True)
    print(f"Folders processed: {len(folder_rows)}", flush=True)
    print(f"Folder predictions: {folder_csv}", flush=True)
    print(f"Folder predictions (review only): {review_csv}", flush=True)
    print(f"Folder->vendor for .fss: {for_fss_csv}", flush=True)
    print(f"Folder->recommended vendor for .fss: {for_fss_recommended_csv}", flush=True)
    if args.enable_ocr_review_refine:
        print(
            f"OCR review refine: processed={ocr_review_processed}, "
            f"any_hit={ocr_review_with_hits}, strong_hit={ocr_review_strong}",
            flush=True,
        )
    print(f"Per-image predictions: {image_csv}", flush=True)
    print(f"Summary: {summary_json}", flush=True)
    if warnings:
        print(f"Warnings: {len(warnings)} (vedi summary.json)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
