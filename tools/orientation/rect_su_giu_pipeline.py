#!/usr/bin/env python3
"""End-to-end pipeline: rect inference review + su/giu training on rect crops.

Workflow:
1) prepare-review
   - Runs rectangle inference with vendor routing (global + specialized, e.g. BK).
   - Writes predictions.csv.
   - Builds interactive HTML review package for bbox confirmation/correction.

2) finalize-train
   - Reads exported corrections_*.csv and exclusions_*.csv from review package.
   - Builds cropped dataset (only ultrasound rectangle content).
   - Trains binary classifier (su vs giu) on cropped images, with resume support via checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

TOOLS_ROOT = Path(__file__).resolve().parents[1]
REVIEW_HTML_TOOLS = TOOLS_ROOT / "review_html"
if str(REVIEW_HTML_TOOLS) not in sys.path:
    sys.path.insert(0, str(REVIEW_HTML_TOOLS))

import infer_ultrasound_rect_folder_html as rect_infer  # noqa: E402


LABELS = ("su", "giu")


@dataclass(frozen=True)
class OrientationManifestRow:
    sample_id: str
    split: str
    label: str
    output_image_path: Path
    output_image_rel: str
    setup_folder: str
    source_image_path: str


@dataclass(frozen=True)
class CropRow:
    sample_id: str
    split: str
    label: str
    image_path: Path
    source_image_path: str
    crop_top: int
    crop_left: int
    crop_bottom: int
    crop_right: int
    box_source: str
    width: int
    height: int


@dataclass(frozen=True)
class TrainRow:
    image_path: Path
    split: str
    label_idx: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:  # noqa: BLE001
        return default


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _is_truthy(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "on", "si", "sì"}


def _atomic_torch_save(payload: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out_path)


def _norm_group_name(setup_folder: str, fallback: str) -> str:
    if setup_folder:
        return Path(setup_folder).name.strip() or fallback
    return fallback


def _load_orientation_manifest(manifest_csv: Path) -> Tuple[List[OrientationManifestRow], List[rect_infer.ImageRecord]]:
    required = {
        "sample_id",
        "split",
        "label",
        "output_image_path",
        "output_image_rel",
        "setup_folder",
        "source_image_path",
    }
    rows: List[OrientationManifestRow] = []
    image_records: List[rect_infer.ImageRecord] = []
    seen_ids: set[str] = set()

    with manifest_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Manifest missing columns: {sorted(missing)}")

        for raw in reader:
            sample_id = str(raw["sample_id"]).strip()
            split = str(raw["split"]).strip().lower()
            label = str(raw["label"]).strip().lower()
            output_image_path = Path(str(raw["output_image_path"])).expanduser().resolve()
            output_image_rel = str(raw["output_image_rel"]).strip()
            setup_folder = str(raw.get("setup_folder", "")).strip()
            source_image = str(raw.get("source_image_path", "")).strip()

            if not sample_id or sample_id in seen_ids:
                continue
            seen_ids.add(sample_id)
            if split not in {"train", "val", "test"}:
                continue
            if label not in LABELS:
                continue
            if not output_image_path.exists():
                continue

            row = OrientationManifestRow(
                sample_id=sample_id,
                split=split,
                label=label,
                output_image_path=output_image_path,
                output_image_rel=output_image_rel,
                setup_folder=setup_folder,
                source_image_path=source_image,
            )
            rows.append(row)
            image_records.append(
                rect_infer.ImageRecord(
                    sample_id=sample_id,
                    image_path=output_image_path,
                    rel_path=output_image_rel,
                    group_name=_norm_group_name(setup_folder=setup_folder, fallback=split),
                    orientation_name=label.upper(),
                )
            )

    if not rows:
        raise RuntimeError(f"No valid rows found in manifest: {manifest_csv}")
    return rows, image_records


def _stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
    ordered = sorted(values)
    n = len(ordered)
    p50 = ordered[n // 2]
    p90 = ordered[min(n - 1, int(round(0.90 * (n - 1))))]
    return {"mean": float(sum(ordered) / n), "p50": float(p50), "p90": float(p90)}


def _run_prepare_review(args: argparse.Namespace) -> int:
    set_seed(int(args.seed))

    manifest_csv = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = output_dir / "predictions.csv"
    summary_json = output_dir / "summary_prepare_review.json"
    review_package_dir = output_dir / "review_package"

    checkpoint_path = args.checkpoint.expanduser().resolve()
    rect_vendor_map_path = args.rect_vendor_map.expanduser().resolve()
    vendor_checkpoint_path = args.vendor_checkpoint.expanduser().resolve()

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Rect checkpoint not found: {checkpoint_path}")
    if not vendor_checkpoint_path.exists():
        raise FileNotFoundError(f"Vendor checkpoint not found: {vendor_checkpoint_path}")

    manifest_rows, records = _load_orientation_manifest(manifest_csv=manifest_csv)
    device = rect_infer.choose_device(args.device)

    ckpt_cpu = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_img_size = int(ckpt_cpu.get("args", {}).get("image_size", 320))
    image_size = int(args.image_size) if int(args.image_size) > 0 else ckpt_img_size

    print(f"Manifest rows: {len(manifest_rows)}", flush=True)
    print(f"Records for rect inference: {len(records)}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Global rect checkpoint: {checkpoint_path}", flush=True)
    print(f"Vendor checkpoint: {vendor_checkpoint_path}", flush=True)
    print(f"Rect vendor map: {rect_vendor_map_path}", flush=True)

    vendor_by_group, vendor_summary = rect_infer._infer_vendor_by_group(
        records=records,
        checkpoint_path=vendor_checkpoint_path,
        image_size_override=int(args.vendor_image_size),
        batch_size=int(args.vendor_batch_size),
        sample_per_group=int(args.vendor_sample_per_group),
        device=device,
        log_interval=int(args.vendor_log_interval),
    )

    rect_vendor_map = rect_infer._load_rect_vendor_map(
        map_path=rect_vendor_map_path,
        disable_routing=bool(args.disable_rect_vendor_routing),
    )
    routes_by_group, route_counts_groups = rect_infer._pick_rect_route_by_group(
        group_names=[r.group_name for r in records],
        vendor_by_group=vendor_by_group,
        global_checkpoint=checkpoint_path,
        global_image_size=image_size,
        vendor_rect_map=rect_vendor_map,
        image_size_override=int(args.image_size),
        vendor_min_confidence=float(args.rect_vendor_min_confidence),
    )

    predicted_raw, skipped_rows, route_counts_images = rect_infer._infer_predictions_with_routing(
        records=records,
        rect_routes_by_group=routes_by_group,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
    )
    if not predicted_raw:
        raise RuntimeError("No predictions produced.")

    group_rect_norm, group_inliers = rect_infer._compute_group_global_rects(
        predicted_rows=predicted_raw,
        inlier_iou_threshold=float(args.inlier_iou_threshold),
    )
    predicted_rows = rect_infer._enrich_with_group_global(
        predicted_rows=predicted_raw,
        group_global_rect_norm=group_rect_norm,
        vendor_by_group=vendor_by_group,
    )
    predicted_rows.sort(key=lambda r: r.sample_id)
    rect_infer._write_predictions_csv(predicted_rows, predictions_csv)

    ious = [row.global_iou for row in predicted_rows]
    by_label: Dict[str, List[float]] = {k: [] for k in LABELS}
    label_by_sid = {r.sample_id: r.label for r in manifest_rows}
    for row in predicted_rows:
        lb = label_by_sid.get(row.sample_id, "")
        if lb in by_label:
            by_label[lb].append(float(row.global_iou))

    summary = {
        "manifest": manifest_csv.as_posix(),
        "num_manifest_rows": len(manifest_rows),
        "predictions_csv": predictions_csv.as_posix(),
        "review_package_dir": review_package_dir.as_posix(),
        "device": str(device),
        "global_checkpoint": checkpoint_path.as_posix(),
        "vendor_checkpoint": vendor_checkpoint_path.as_posix(),
        "rect_vendor_map": rect_vendor_map_path.as_posix(),
        "rect_vendor_map_loaded": {k: v.as_posix() for k, v in sorted(rect_vendor_map.items())},
        "images_predicted": len(predicted_rows),
        "images_skipped": len(skipped_rows),
        "skipped_examples": skipped_rows[:20],
        "groups": len(set(r.group_name for r in records)),
        "vendor_recognition": vendor_summary,
        "rect_routing": {
            "disabled": bool(args.disable_rect_vendor_routing),
            "min_vendor_confidence": float(args.rect_vendor_min_confidence),
            "counts_by_group": dict(sorted(route_counts_groups.items())),
            "counts_by_image": dict(sorted(route_counts_images.items())),
        },
        "group_global_rects": {
            "groups_total": len(group_rect_norm),
            "inlier_iou_threshold": float(args.inlier_iou_threshold),
            "inliers_total": int(sum(group_inliers.values())),
        },
        "iou_pred_vs_group_global": _stats(ious),
        "iou_by_label": {k: _stats(v) for k, v in by_label.items()},
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    package_builder = REVIEW_HTML_TOOLS / "build_ultrasound_rect_review_package.py"
    subprocess.run(
        [
            sys.executable,
            str(package_builder),
            "--predictions-csv",
            str(predictions_csv),
            "--output-dir",
            str(review_package_dir),
            "--package-name",
            "US Rect Review for SU/GIU Training",
        ],
        check=True,
    )

    print(f"Predictions CSV: {predictions_csv}", flush=True)
    print(f"Prepare summary: {summary_json}", flush=True)
    print(f"Review package index: {review_package_dir / 'index.html'}", flush=True)
    print("Export corrections_*.csv and exclusions_*.csv from folder pages before finalize-train.", flush=True)
    return 0


def _load_predictions_by_sample(predictions_csv: Path) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    with predictions_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {
            "sample_id",
            "pred_top",
            "pred_left",
            "pred_bottom",
            "pred_right",
            "global_top",
            "global_left",
            "global_bottom",
            "global_right",
            "width",
            "height",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Predictions CSV missing columns: {sorted(missing)}")
        for row in reader:
            sid = str(row.get("sample_id", "")).strip()
            if not sid:
                continue
            out[sid] = dict(row)
    if not out:
        raise RuntimeError(f"No prediction rows found: {predictions_csv}")
    return out


def _load_corrections_by_sample(corrections_dir: Path, pattern: str) -> Tuple[Dict[str, Dict[str, int]], List[Path]]:
    files = sorted(corrections_dir.rglob(pattern), key=lambda p: (p.stat().st_mtime, p.name))
    corr: Dict[str, Dict[str, int]] = {}
    for path in files:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            required = {"sample_id", "corr_top", "corr_left", "corr_bottom", "corr_right"}
            if required.difference(reader.fieldnames or []):
                continue
            for row in reader:
                sid = str(row.get("sample_id", "")).strip()
                if not sid:
                    continue
                corr[sid] = {
                    "top": _to_int(row.get("corr_top", 0)),
                    "left": _to_int(row.get("corr_left", 0)),
                    "bottom": _to_int(row.get("corr_bottom", 0)),
                    "right": _to_int(row.get("corr_right", 0)),
                }
    return corr, files


def _load_exclusions(
    corrections_dir: Path,
    pattern: str,
) -> Tuple[set[str], set[str], List[Path]]:
    files = sorted(corrections_dir.rglob(pattern), key=lambda p: (p.stat().st_mtime, p.name))
    excluded_samples: set[str] = set()
    excluded_groups: set[str] = set()
    for path in files:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                continue
            has_exclude_col = "exclude_training" in reader.fieldnames
            for row in reader:
                include_row = True
                if has_exclude_col:
                    include_row = _is_truthy(row.get("exclude_training", "0"))
                if not include_row:
                    continue

                row_type = str(row.get("type", "image")).strip().lower()
                sample_id = str(row.get("sample_id", "")).strip()
                group_name = str(row.get("group_name", "")).strip()
                if row_type == "folder":
                    if group_name:
                        excluded_groups.add(group_name.casefold())
                else:
                    if sample_id:
                        excluded_samples.add(sample_id)
    return excluded_samples, excluded_groups, files


def _clamp_box_px(top: int, left: int, bottom: int, right: int, width: int, height: int) -> Tuple[int, int, int, int]:
    top = max(0, min(int(top), max(0, height - 1)))
    left = max(0, min(int(left), max(0, width - 1)))
    bottom = max(1, min(int(bottom), int(height)))
    right = max(1, min(int(right), int(width)))
    if bottom <= top:
        bottom = min(height, top + 1)
    if right <= left:
        right = min(width, left + 1)
    return top, left, bottom, right


def _build_rect_crops(
    manifest_rows: Sequence[OrientationManifestRow],
    preds_by_sample: Dict[str, Dict[str, object]],
    corr_by_sample: Dict[str, Dict[str, int]],
    crop_root: Path,
    default_box_source: str,
) -> Tuple[List[CropRow], Dict[str, object]]:
    crop_rows: List[CropRow] = []
    missing_pred: List[str] = []
    source_counter = Counter()

    for row in manifest_rows:
        pred = preds_by_sample.get(row.sample_id)
        if pred is None:
            missing_pred.append(row.sample_id)
            continue

        width = _to_int(pred.get("width", 0))
        height = _to_int(pred.get("height", 0))
        if width <= 1 or height <= 1:
            continue

        if row.sample_id in corr_by_sample:
            box = corr_by_sample[row.sample_id]
            top, left, bottom, right = box["top"], box["left"], box["bottom"], box["right"]
            box_source = "correction"
        else:
            if default_box_source == "pred":
                top = _to_int(pred.get("pred_top", 0))
                left = _to_int(pred.get("pred_left", 0))
                bottom = _to_int(pred.get("pred_bottom", height))
                right = _to_int(pred.get("pred_right", width))
                box_source = "pred_default"
            else:
                top = _to_int(pred.get("global_top", 0))
                left = _to_int(pred.get("global_left", 0))
                bottom = _to_int(pred.get("global_bottom", height))
                right = _to_int(pred.get("global_right", width))
                box_source = "global_default"

        source_counter[box_source] += 1
        top, left, bottom, right = _clamp_box_px(top, left, bottom, right, width, height)

        out_dir = crop_root / "images" / row.split / row.label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{row.sample_id}.png"

        with Image.open(row.output_image_path) as img:
            rgb = img.convert("RGB")
            crop = rgb.crop((left, top, right, bottom))
            crop.save(out_path, format="PNG", optimize=True)

        crop_rows.append(
            CropRow(
                sample_id=row.sample_id,
                split=row.split,
                label=row.label,
                image_path=out_path,
                source_image_path=row.source_image_path,
                crop_top=top,
                crop_left=left,
                crop_bottom=bottom,
                crop_right=right,
                box_source=box_source,
                width=int(crop.size[0]),
                height=int(crop.size[1]),
            )
        )

    summary = {
        "rows_in_manifest": len(manifest_rows),
        "rows_with_predictions": len(manifest_rows) - len(missing_pred),
        "rows_cropped": len(crop_rows),
        "missing_prediction_count": len(missing_pred),
        "missing_prediction_preview": missing_pred[:50],
        "box_source_counts": dict(sorted(source_counter.items())),
    }
    return crop_rows, summary


def _write_crop_manifest(crop_rows: Sequence[CropRow], out_csv: Path) -> None:
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "sample_id",
                "split",
                "label",
                "class_target",
                "image_path",
                "source_image_path",
                "crop_top",
                "crop_left",
                "crop_bottom",
                "crop_right",
                "crop_width",
                "crop_height",
                "box_source",
            ]
        )
        for r in sorted(crop_rows, key=lambda x: (x.split, x.label, x.sample_id)):
            writer.writerow(
                [
                    r.sample_id,
                    r.split,
                    r.label,
                    0 if r.label == "su" else 1,
                    r.image_path.as_posix(),
                    r.source_image_path,
                    r.crop_top,
                    r.crop_left,
                    r.crop_bottom,
                    r.crop_right,
                    r.width,
                    r.height,
                    r.box_source,
                ]
            )


class BinaryOrientationDataset(Dataset):
    def __init__(self, rows: Sequence[TrainRow], image_size: int, augment: bool) -> None:
        self.rows = list(rows)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):  # type: ignore[override]
        row = self.rows[idx]
        try:
            with Image.open(row.image_path) as img:
                image = img.convert("RGB")
                if self.augment:
                    if random.random() < 0.45:
                        image = TF.adjust_brightness(image, 0.9 + 0.2 * random.random())
                    if random.random() < 0.45:
                        image = TF.adjust_contrast(image, 0.9 + 0.2 * random.random())
                    if random.random() < 0.35:
                        image = TF.hflip(image)

                image = TF.resize(
                    image,
                    size=[self.image_size, self.image_size],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                x = TF.to_tensor(image)
                x = (x - self.mean) / self.std
        except Exception:
            return None

        y = torch.tensor(row.label_idx, dtype=torch.long)
        return x, y


def _collate_drop_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    return x, y


class BinaryOrientationClassifier(nn.Module):
    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


def _make_loader(rows: List[TrainRow], image_size: int, batch_size: int, num_workers: int, augment: bool) -> DataLoader:
    ds = BinaryOrientationDataset(rows=rows, image_size=image_size, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=_collate_drop_none,
        drop_last=False,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    class_weights: torch.Tensor,
) -> Dict[str, object]:
    train = optimizer is not None
    model.train(mode=train)

    total = 0
    total_loss = 0.0
    correct = 0
    conf = np.zeros((2, 2), dtype=np.int64)

    for packed in loader:
        if packed is None:
            continue
        x, y = packed
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=class_weights)
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        pred = torch.argmax(logits, dim=1)
        bs = x.size(0)
        total += bs
        total_loss += float(loss.detach().cpu()) * bs
        correct += int((pred == y).sum().item())

        y_cpu = y.detach().cpu().numpy()
        p_cpu = pred.detach().cpu().numpy()
        for yi, pi in zip(y_cpu, p_cpu):
            conf[int(yi), int(pi)] += 1

    acc = float(correct / max(1, total))
    recalls = []
    for c in range(2):
        denom = conf[c].sum()
        if denom > 0:
            recalls.append(float(conf[c, c] / denom))
    bal_acc = float(sum(recalls) / len(recalls)) if recalls else 0.0
    return {
        "loss": float(total_loss / max(1, total)),
        "acc": acc,
        "balanced_acc": bal_acc,
        "samples": int(total),
        "confusion": conf.tolist(),
    }


def _load_train_rows_from_crop_manifest(crop_manifest_csv: Path) -> Dict[str, List[TrainRow]]:
    split_rows: Dict[str, List[TrainRow]] = {"train": [], "val": [], "test": []}
    with crop_manifest_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"split", "label", "image_path"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Crop manifest missing columns: {sorted(missing)}")
        for row in reader:
            split = str(row.get("split", "")).strip().lower()
            label = str(row.get("label", "")).strip().lower()
            if split not in split_rows or label not in LABELS:
                continue
            image_path = Path(str(row.get("image_path", ""))).expanduser().resolve()
            if not image_path.exists():
                continue
            label_idx = 0 if label == "su" else 1
            split_rows[split].append(
                TrainRow(
                    image_path=image_path,
                    split=split,
                    label_idx=label_idx,
                )
            )
    return split_rows


def _train_binary_classifier(
    crop_manifest_csv: Path,
    model_out_dir: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    image_size: int,
    lr: float,
    weight_decay: float,
    pretrained: bool,
    early_stopping_patience: int,
    seed: int,
    device: Optional[str],
    resume: bool,
    resume_checkpoint: Optional[Path],
) -> Dict[str, object]:
    set_seed(seed)
    model_out_dir.mkdir(parents=True, exist_ok=True)
    split_rows = _load_train_rows_from_crop_manifest(crop_manifest_csv)
    if not split_rows["train"] or not split_rows["val"] or not split_rows["test"]:
        raise RuntimeError(
            "Empty split in cropped dataset. "
            f"Counts: train={len(split_rows['train'])}, val={len(split_rows['val'])}, test={len(split_rows['test'])}"
        )

    dev = rect_infer.choose_device(device)
    train_loader = _make_loader(split_rows["train"], image_size, batch_size, num_workers, augment=True)
    val_loader = _make_loader(split_rows["val"], image_size, batch_size, num_workers, augment=False)
    test_loader = _make_loader(split_rows["test"], image_size, batch_size, num_workers, augment=False)

    model = BinaryOrientationClassifier(pretrained=pretrained).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    train_counts = Counter([r.label_idx for r in split_rows["train"]])
    counts = np.array([train_counts.get(0, 0), train_counts.get(1, 0)], dtype=np.float32)
    weights = np.zeros((2,), dtype=np.float32)
    mask = counts > 0
    weights[mask] = counts[mask].sum() / (counts[mask] * mask.sum())
    if mask.any():
        weights[mask] /= weights[mask].mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=dev)

    best_path = model_out_dir / "best_model.pt"
    last_checkpoint_path = (
        resume_checkpoint.expanduser().resolve()
        if resume_checkpoint is not None
        else (model_out_dir / "last_checkpoint.pt")
    )
    history_path = model_out_dir / "history.json"

    train_args = {
        "image_size": image_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "pretrained": pretrained,
        "split_counts": {k: len(v) for k, v in split_rows.items()},
    }

    best_val = -1.0
    best_epoch = -1
    history: List[Dict[str, object]] = []
    patience = 0
    start_epoch = 1
    last_completed_epoch = 0
    resumed_from: Optional[str] = None
    interrupted = False

    if resume:
        if last_checkpoint_path.exists():
            resume_state = torch.load(last_checkpoint_path, map_location=dev, weights_only=False)
            if "model_state_dict" in resume_state:
                model.load_state_dict(resume_state["model_state_dict"])
            if "optimizer_state_dict" in resume_state:
                optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            if "scheduler_state_dict" in resume_state:
                scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            best_val = float(resume_state.get("best_val_balanced_acc", resume_state.get("best_val", -1.0)))
            best_epoch = int(resume_state.get("best_epoch", -1))
            history = list(resume_state.get("history", []))
            patience = int(resume_state.get("patience", 0))
            last_completed_epoch = int(resume_state.get("epoch", 0))
            start_epoch = last_completed_epoch + 1
            resumed_from = last_checkpoint_path.as_posix()
            print(
                f"Resuming training from checkpoint: {resumed_from} (last epoch={last_completed_epoch})",
                flush=True,
            )
        else:
            print(
                f"Resume requested but checkpoint not found: {last_checkpoint_path}. Starting from scratch.",
                flush=True,
            )

    try:
        for epoch in range(start_epoch, epochs + 1):
            train_m = _run_epoch(model, train_loader, dev, optimizer, class_weights)
            val_m = _run_epoch(model, val_loader, dev, None, class_weights)
            scheduler.step()

            history.append({"epoch": epoch, "train": train_m, "val": val_m})
            print(
                f"epoch {epoch:03d} | train acc {train_m['acc']:.4f} bal {train_m['balanced_acc']:.4f} "
                f"| val acc {val_m['acc']:.4f} bal {val_m['balanced_acc']:.4f}",
                flush=True,
            )

            score = float(val_m["balanced_acc"])
            if score > best_val:
                best_val = score
                best_epoch = epoch
                patience = 0
                _atomic_torch_save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_names": list(LABELS),
                        "args": train_args,
                        "best_epoch": best_epoch,
                        "best_val_balanced_acc": best_val,
                    },
                    best_path,
                )
            else:
                patience += 1

            last_completed_epoch = epoch
            _atomic_torch_save(
                {
                    "epoch": last_completed_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "class_names": list(LABELS),
                    "args": train_args,
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_val_balanced_acc": best_val,
                    "patience": patience,
                },
                last_checkpoint_path,
            )

            if patience >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break
    except KeyboardInterrupt:
        interrupted = True
        print(
            "Training interrupted. Last checkpoint is available for resume: "
            f"{last_checkpoint_path}",
            flush=True,
        )
        if not last_checkpoint_path.exists():
            _atomic_torch_save(
                {
                    "epoch": last_completed_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "class_names": list(LABELS),
                    "args": train_args,
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_val_balanced_acc": best_val,
                    "patience": patience,
                },
                last_checkpoint_path,
            )

    if not best_path.exists():
        # Fallback: persist current model so resume/interrupted runs always keep an usable checkpoint.
        if best_epoch < 0:
            best_epoch = max(0, last_completed_epoch)
        _atomic_torch_save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": list(LABELS),
                "args": train_args,
                "best_epoch": best_epoch,
                "best_val_balanced_acc": best_val,
            },
            best_path,
        )

    test_m: Dict[str, object] = {}
    if not interrupted:
        ckpt = torch.load(best_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_m = _run_epoch(model, test_loader, dev, None, class_weights)

    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "crop_manifest": crop_manifest_csv.as_posix(),
        "device": str(dev),
        "class_names": list(LABELS),
        "split_counts": {k: len(v) for k, v in split_rows.items()},
        "train_class_counts": {"su": int(train_counts.get(0, 0)), "giu": int(train_counts.get(1, 0))},
        "best_epoch": best_epoch,
        "best_val_balanced_acc": best_val,
        "test_metrics": test_m,
        "best_model": best_path.as_posix(),
        "last_checkpoint": last_checkpoint_path.as_posix(),
        "history_path": history_path.as_posix(),
        "resume_enabled": bool(resume),
        "resumed_from": resumed_from,
        "start_epoch": int(start_epoch),
        "last_completed_epoch": int(last_completed_epoch),
        "interrupted": bool(interrupted),
    }
    (model_out_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _run_finalize_train(args: argparse.Namespace) -> int:
    set_seed(int(args.seed))

    manifest_csv = args.manifest.expanduser().resolve()
    predictions_csv = args.predictions_csv.expanduser().resolve()
    corrections_dir = args.corrections_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_root = output_dir / "cropped_dataset"
    crop_manifest_csv = crop_root / "crop_manifest.csv"
    finalize_summary_json = output_dir / "summary_finalize_train.json"

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")
    if not corrections_dir.exists():
        raise FileNotFoundError(f"Corrections directory not found: {corrections_dir}")
    if args.default_box_source not in {"global", "pred"}:
        raise ValueError("--default-box-source must be one of: global, pred")

    manifest_rows, _ = _load_orientation_manifest(manifest_csv=manifest_csv)
    preds_by_sample = _load_predictions_by_sample(predictions_csv=predictions_csv)
    corr_by_sample, corr_files = _load_corrections_by_sample(
        corrections_dir=corrections_dir,
        pattern=str(args.corrections_pattern),
    )
    excluded_samples, excluded_groups, exclusion_files = _load_exclusions(
        corrections_dir=corrections_dir,
        pattern=str(args.exclusions_pattern),
    )

    kept_manifest_rows: List[OrientationManifestRow] = []
    excluded_manifest_rows: List[OrientationManifestRow] = []
    for row in manifest_rows:
        group_name = _norm_group_name(setup_folder=row.setup_folder, fallback=row.split)
        if row.sample_id in excluded_samples or group_name.casefold() in excluded_groups:
            excluded_manifest_rows.append(row)
        else:
            kept_manifest_rows.append(row)

    print(f"Manifest rows: {len(manifest_rows)}", flush=True)
    print(f"Predictions rows: {len(preds_by_sample)}", flush=True)
    print(f"Corrections files found: {len(corr_files)}", flush=True)
    print(f"Corrections unique sample_id: {len(corr_by_sample)}", flush=True)
    print(f"Exclusions files found: {len(exclusion_files)}", flush=True)
    print(f"Excluded sample_id (explicit): {len(excluded_samples)}", flush=True)
    print(f"Excluded groups (folder-level): {len(excluded_groups)}", flush=True)
    print(f"Manifest rows excluded: {len(excluded_manifest_rows)}", flush=True)

    crop_rows, crop_summary = _build_rect_crops(
        manifest_rows=kept_manifest_rows,
        preds_by_sample=preds_by_sample,
        corr_by_sample=corr_by_sample,
        crop_root=crop_root,
        default_box_source=str(args.default_box_source),
    )
    if not crop_rows:
        raise RuntimeError("No cropped rows produced.")

    _write_crop_manifest(crop_rows=crop_rows, out_csv=crop_manifest_csv)

    model_out_dir = output_dir / "model_su_giu_rect"
    train_summary = _train_binary_classifier(
        crop_manifest_csv=crop_manifest_csv,
        model_out_dir=model_out_dir,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        image_size=int(args.train_image_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        pretrained=bool(args.pretrained),
        early_stopping_patience=int(args.early_stopping_patience),
        seed=int(args.seed),
        device=args.device,
        resume=bool(args.resume),
        resume_checkpoint=(args.resume_checkpoint.expanduser().resolve() if args.resume_checkpoint else None),
    )

    payload = {
        "manifest": manifest_csv.as_posix(),
        "predictions_csv": predictions_csv.as_posix(),
        "corrections_dir": corrections_dir.as_posix(),
        "corrections_pattern": str(args.corrections_pattern),
        "corrections_files": [p.as_posix() for p in corr_files],
        "corrections_unique_sample_ids": len(corr_by_sample),
        "exclusions_pattern": str(args.exclusions_pattern),
        "exclusions_files": [p.as_posix() for p in exclusion_files],
        "exclusions_samples_explicit": len(excluded_samples),
        "exclusions_groups": len(excluded_groups),
        "exclusions_groups_preview": sorted(excluded_groups)[:50],
        "manifest_rows_total": len(manifest_rows),
        "manifest_rows_excluded": len(excluded_manifest_rows),
        "manifest_rows_kept": len(kept_manifest_rows),
        "excluded_sample_ids_preview": [r.sample_id for r in excluded_manifest_rows[:50]],
        "default_box_source": str(args.default_box_source),
        "resume_enabled": bool(args.resume),
        "resume_checkpoint": (
            args.resume_checkpoint.expanduser().resolve().as_posix()
            if args.resume_checkpoint is not None
            else None
        ),
        "crop_manifest": crop_manifest_csv.as_posix(),
        "crop_summary": crop_summary,
        "train_summary": train_summary,
    }
    finalize_summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Crop manifest: {crop_manifest_csv}", flush=True)
    print(f"Best model: {train_summary['best_model']}", flush=True)
    if train_summary.get("last_checkpoint"):
        print(f"Last checkpoint: {train_summary['last_checkpoint']}", flush=True)
    print(f"Train summary: {model_out_dir / 'train_summary.json'}", flush=True)
    print(f"Finalize summary: {finalize_summary_json}", flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline su/giu con rettangolo ecografico: "
            "inference+review bbox e training sul solo crop interno."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser(
        "prepare-review",
        help="Inferenza rettangolo (routing globale+BK) e generazione review HTML interattiva.",
    )
    p_prepare.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/20_datasets/us_orientation_binary_20260320/manifest.csv"),
    )
    p_prepare.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline"),
    )
    p_prepare.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/10_active_pipeline/pipeline_fss_head/models/rect_training_e40_run2/best_model.pt"),
        help="Checkpoint detector rettangolo globale.",
    )
    p_prepare.add_argument(
        "--rect-vendor-map",
        type=Path,
        default=Path("artifacts/10_active_pipeline/pipeline_fss_head/maps/vendor_rect_map_bk_only.json"),
        help="JSON vendor->checkpoint rettangolo specializzato.",
    )
    p_prepare.add_argument(
        "--vendor-checkpoint",
        type=Path,
        default=Path("artifacts/10_active_pipeline/pipeline_fss_head/models/vendor_training_no_negative_v2_power/best_model.pt"),
        help="Checkpoint classificatore vendor.",
    )
    p_prepare.add_argument("--disable-rect-vendor-routing", action="store_true")
    p_prepare.add_argument("--rect-vendor-min-confidence", type=float, default=0.0)
    p_prepare.add_argument("--image-size", type=int, default=0)
    p_prepare.add_argument("--vendor-image-size", type=int, default=0)
    p_prepare.add_argument("--vendor-batch-size", type=int, default=48)
    p_prepare.add_argument("--vendor-sample-per-group", type=int, default=0)
    p_prepare.add_argument("--vendor-log-interval", type=int, default=25)
    p_prepare.add_argument("--batch-size", type=int, default=64)
    p_prepare.add_argument("--num-workers", type=int, default=2)
    p_prepare.add_argument("--inlier-iou-threshold", type=float, default=0.20)
    p_prepare.add_argument("--seed", type=int, default=42)
    p_prepare.add_argument("--device", type=str, default=None)

    p_finalize = sub.add_parser(
        "finalize-train",
        help="Applica correzioni/esclusioni esportate, crea crop e allena la rete su/giu.",
    )
    p_finalize.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/20_datasets/us_orientation_binary_20260320/manifest.csv"),
    )
    p_finalize.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/predictions.csv"),
    )
    p_finalize.add_argument(
        "--corrections-dir",
        type=Path,
        default=Path("/Users/camilla/Downloads"),
        help="Directory contenente i corrections_*.csv esportati dalla review HTML.",
    )
    p_finalize.add_argument("--corrections-pattern", type=str, default="corrections_*.csv")
    p_finalize.add_argument(
        "--exclusions-pattern",
        type=str,
        default="exclusions_*.csv",
        help="Pattern CSV esclusioni training esportate dalla review HTML.",
    )
    p_finalize.add_argument(
        "--default-box-source",
        type=str,
        default="global",
        help="Box di default se non corretto: global oppure pred.",
    )
    p_finalize.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline"),
    )
    p_finalize.add_argument("--train-image-size", type=int, default=256)
    p_finalize.add_argument("--epochs", type=int, default=22)
    p_finalize.add_argument("--batch-size", type=int, default=64)
    p_finalize.add_argument("--num-workers", type=int, default=2)
    p_finalize.add_argument("--lr", type=float, default=1e-3)
    p_finalize.add_argument("--weight-decay", type=float, default=1e-4)
    p_finalize.add_argument("--early-stopping-patience", type=int, default=6)
    p_finalize.add_argument("--seed", type=int, default=42)
    p_finalize.add_argument("--pretrained", action="store_true")
    p_finalize.add_argument(
        "--resume",
        action="store_true",
        help="Riprende training da last_checkpoint.pt se disponibile.",
    )
    p_finalize.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint esplicito per resume (default: <output-dir>/model_su_giu_rect/last_checkpoint.pt).",
    )
    p_finalize.add_argument("--device", type=str, default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "prepare-review":
        return _run_prepare_review(args)
    if args.command == "finalize-train":
        return _run_finalize_train(args)
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
