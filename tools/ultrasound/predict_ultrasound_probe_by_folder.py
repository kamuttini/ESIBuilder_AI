#!/usr/bin/env python3
"""Predict one probe_id (fss_id_probe) per dataset folder."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from train_ultrasound_probe_classifier import ProbeClassifier, choose_device


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict one probe_id per dataset folder."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("Dataset"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/10_active_pipeline/pipeline_fss_head/models/probe_training_no_negative_v1/best_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1"),
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=0, help="0=use checkpoint arg")
    parser.add_argument("--sample-per-folder", type=int, default=80)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--exclude-folder-regex", type=str, default=None)
    parser.add_argument("--exclude-image-regex", type=str, default="(?i)negative")
    parser.add_argument("--min-folder-confidence", type=float, default=0.50)
    parser.add_argument("--min-folder-margin", type=float, default=0.08)
    parser.add_argument("--min-vote-ratio", type=float, default=0.50)
    parser.add_argument("--log-interval", type=int, default=25)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size deve essere > 0.")
    if args.sample_per_folder < 0:
        raise ValueError("--sample-per-folder deve essere >= 0.")
    if not (0.0 <= args.min_folder_confidence <= 1.0):
        raise ValueError("--min-folder-confidence deve essere in [0,1].")
    if args.min_folder_margin < 0:
        raise ValueError("--min-folder-margin deve essere >= 0.")
    if not (0.0 <= args.min_vote_ratio <= 1.0):
        raise ValueError("--min-vote-ratio deve essere in [0,1].")

    dataset_root = args.dataset_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise RuntimeError("Checkpoint senza class_names.")
    class_names = list(class_names)

    image_size = int(args.image_size)
    if image_size <= 0:
        image_size = int(checkpoint.get("args", {}).get("image_size", 320))

    device = choose_device(args.device)
    model = ProbeClassifier(num_classes=len(class_names), pretrained=False).to(device)
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
                "predicted_probe_id": pred_label,
                "folder_top1_prob": f"{top1_prob:.6f}",
                "folder_margin_top1_top2": f"{margin:.6f}",
                "folder_vote_ratio": f"{vote_ratio:.6f}",
                "status": status,
                "review_reasons": ",".join(reasons),
                "topk": topk_str,
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
                    "image_pred_probe_id": img_pred,
                    "image_pred_confidence": f"{img_conf:.6f}",
                    "matches_folder_probe": int(img_pred == pred_label),
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

    folder_csv = output_dir / "folder_probe_predictions.csv"
    with folder_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "folder_path",
                "folder_name",
                "images_total_valid",
                "images_used",
                "predicted_probe_id",
                "folder_top1_prob",
                "folder_margin_top1_top2",
                "folder_vote_ratio",
                "status",
                "review_reasons",
                "topk",
            ],
        )
        writer.writeheader()
        writer.writerows(folder_rows)

    review_csv = output_dir / "folder_probe_predictions_review.csv"
    with review_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "folder_path",
                "folder_name",
                "images_total_valid",
                "images_used",
                "predicted_probe_id",
                "folder_top1_prob",
                "folder_margin_top1_top2",
                "folder_vote_ratio",
                "status",
                "review_reasons",
                "topk",
            ],
        )
        writer.writeheader()
        for row in folder_rows:
            if row["status"] == "review":
                writer.writerow(row)

    for_fss_csv = output_dir / "folder_probe_for_fss.csv"
    with for_fss_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["folder_path", "predicted_probe_id"])
        for row in folder_rows:
            writer.writerow([row["folder_path"], row["predicted_probe_id"]])

    image_csv = output_dir / "per_image_probe_predictions.csv"
    with image_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "folder_path",
                "image_path",
                "image_pred_probe_id",
                "image_pred_confidence",
                "matches_folder_probe",
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
        "predicted_probe_distribution": dict(sorted(pred_counter.items())),
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
    print(f"Folder->probe for .fss: {for_fss_csv}", flush=True)
    print(f"Per-image predictions: {image_csv}", flush=True)
    print(f"Summary: {summary_json}", flush=True)
    if warnings:
        print(f"Warnings: {len(warnings)} (vedi summary.json)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
