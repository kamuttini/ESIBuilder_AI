#!/usr/bin/env python3
"""Train a binary LR-marker classifier from the reviewed marker manifest.

The input is the reviewed SU/GIU-driven LR marker manifest. The model sees the
echo rectangle, optionally padded, and predicts whether the image is normal or
mirrored to the right according to the detected marker side.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASS_NAMES = ("not_lr_flipped", "lr_flipped")
CLASS_LABEL_IT = ("normale", "specchiata_a_destra")


@dataclass(frozen=True)
class LRRow:
    row_idx: int
    sample_uid: str
    manifest_sample_id: str
    split: str
    image_path: Path
    label: int
    manufacturer: str
    config_folder: str
    image_name: str
    su_giu_pred: str
    su_giu_conf: str
    match_score: str
    search_strategy: str
    echo_rect: Tuple[int, int, int, int]
    original: Dict[str, str]


@dataclass(frozen=True)
class PreparedRow:
    row: LRRow
    split: str
    label: int
    crop_rect: Tuple[int, int, int, int]
    transform: str = "original"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(prefer: Optional[str]) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def atomic_torch_save(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def parse_int(value: str, field: str, row_idx: int) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception as exc:
        raise ValueError(f"Invalid integer field {field!r} at row {row_idx}: {value!r}") from exc


def parse_echo_rect(row: Dict[str, str], row_idx: int) -> Tuple[int, int, int, int]:
    return (
        parse_int(row.get("echo_rect_top_abs", ""), "echo_rect_top_abs", row_idx),
        parse_int(row.get("echo_rect_left_abs", ""), "echo_rect_left_abs", row_idx),
        parse_int(row.get("echo_rect_bottom_abs", ""), "echo_rect_bottom_abs", row_idx),
        parse_int(row.get("echo_rect_right_abs", ""), "echo_rect_right_abs", row_idx),
    )


def clip_rect(rect: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    top, left, bottom, right = rect
    top = max(0, min(height - 1, top))
    left = max(0, min(width - 1, left))
    bottom = max(0, min(height - 1, bottom))
    right = max(0, min(width - 1, right))
    if bottom < top:
        top, bottom = bottom, top
    if right < left:
        left, right = right, left
    return top, left, bottom, right


def padded_rect(
    rect: Tuple[int, int, int, int],
    width: int,
    height: int,
    padding_frac: float,
) -> Tuple[int, int, int, int]:
    top, left, bottom, right = rect
    rect_w = max(1, right - left + 1)
    rect_h = max(1, bottom - top + 1)
    pad = int(round(max(rect_w, rect_h) * max(0.0, padding_frac)))
    return clip_rect((top - pad, left - pad, bottom + pad, right + pad), width=width, height=height)


def label_to_idx(value: str) -> Optional[int]:
    normalized = str(value).strip().lower()
    if normalized in {"0", "not_lr_flipped", "normal", "normale", "left"}:
        return 0
    if normalized in {"1", "lr_flipped", "flipped", "specchiata_a_destra", "right"}:
        return 1
    return None


def load_rows(manifest_csv: Path) -> List[LRRow]:
    rows: List[LRRow] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {
            "image_path",
            "lr_label",
            "manufacturer",
            "config_folder",
            "echo_rect_top_abs",
            "echo_rect_left_abs",
            "echo_rect_bottom_abs",
            "echo_rect_right_abs",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Manifest missing columns: {sorted(missing)}")

        for row_idx, row in enumerate(reader, start=1):
            label = label_to_idx(row.get("lr_binary", "") or row.get("lr_label", ""))
            if label is None:
                continue
            image_path = Path(row.get("image_path", "")).expanduser().resolve()
            if not image_path.exists():
                continue
            manifest_sample_id = row.get("sample_id", "").strip()
            review_batch = row.get("review_batch", "").strip()
            sample_uid = f"{review_batch}:{manifest_sample_id}" if review_batch else f"row:{row_idx}"
            rows.append(
                LRRow(
                    row_idx=row_idx,
                    sample_uid=sample_uid,
                    manifest_sample_id=manifest_sample_id,
                    split=row.get("split", "train").strip().lower() or "train",
                    image_path=image_path,
                    label=label,
                    manufacturer=row.get("manufacturer", "UNKNOWN").strip() or "UNKNOWN",
                    config_folder=row.get("config_folder", "UNKNOWN").strip() or "UNKNOWN",
                    image_name=row.get("image_name", image_path.name).strip() or image_path.name,
                    su_giu_pred=row.get("su_giu_pred", "").strip(),
                    su_giu_conf=row.get("su_giu_conf", "").strip(),
                    match_score=row.get("match_score", "").strip(),
                    search_strategy=row.get("search_strategy", "").strip(),
                    echo_rect=parse_echo_rect(row, row_idx=row_idx),
                    original=dict(row),
                )
            )
    return rows


def split_counts(rows: Sequence[PreparedRow]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        c = Counter(r.label for r in rows if r.split == split)
        out[split] = {CLASS_NAMES[i]: int(c.get(i, 0)) for i in range(2)}
    return out


def assign_manifest_split(rows: Sequence[LRRow]) -> Dict[int, str]:
    assigned: Dict[int, str] = {}
    for row in rows:
        split = row.split if row.split in {"train", "val", "test"} else "train"
        assigned[row.row_idx] = split
    return assigned


def _counts_for_group(n: int, val_frac: float, test_frac: float) -> Tuple[int, int, int]:
    if n <= 1:
        return n, 0, 0
    if n == 2:
        return 1, 0, 1
    n_val = max(1, int(round(n * val_frac)))
    n_test = max(1, int(round(n * test_frac)))
    while n_val + n_test > n - 1:
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            break
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def assign_stratified_split(
    rows: Sequence[LRRow],
    seed: int,
    val_frac: float,
    test_frac: float,
    stratify_by_vendor: bool,
) -> Dict[int, str]:
    groups: DefaultDict[Tuple[str, int], List[LRRow]] = defaultdict(list)
    for row in rows:
        key = (row.manufacturer if stratify_by_vendor else "__all__", row.label)
        groups[key].append(row)

    rng = random.Random(seed)
    assigned: Dict[int, str] = {}
    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        n_train, n_val, n_test = _counts_for_group(len(group), val_frac=val_frac, test_frac=test_frac)
        for row in group[:n_train]:
            assigned[row.row_idx] = "train"
        for row in group[n_train : n_train + n_val]:
            assigned[row.row_idx] = "val"
        for row in group[n_train + n_val : n_train + n_val + n_test]:
            assigned[row.row_idx] = "test"
    return assigned


def make_crop_rect(
    image_path: Path,
    echo_rect: Tuple[int, int, int, int],
    crop_mode: str,
    padding_frac: float,
) -> Tuple[int, int, int, int]:
    with Image.open(image_path) as img:
        width, height = img.size
    rect = clip_rect(echo_rect, width=width, height=height)
    if crop_mode == "echo":
        return rect
    if crop_mode == "padded_echo":
        return padded_rect(rect, width=width, height=height, padding_frac=padding_frac)
    raise ValueError(f"Unsupported crop mode: {crop_mode}")


def prepare_rows(
    rows: Sequence[LRRow],
    assigned_split: Dict[int, str],
    crop_mode: str,
    padding_frac: float,
) -> List[PreparedRow]:
    prepared: List[PreparedRow] = []
    for row in rows:
        split = assigned_split.get(row.row_idx, "train")
        crop_rect = make_crop_rect(
            image_path=row.image_path,
            echo_rect=row.echo_rect,
            crop_mode=crop_mode,
            padding_frac=padding_frac,
        )
        prepared.append(PreparedRow(row=row, split=split, label=row.label, crop_rect=crop_rect))
    return prepared


def add_hflip_label_swap_rows(rows: Sequence[PreparedRow]) -> List[PreparedRow]:
    augmented = list(rows)
    for packed in rows:
        if packed.split != "train":
            continue
        augmented.append(
            PreparedRow(
                row=packed.row,
                split=packed.split,
                label=1 - packed.label,
                crop_rect=packed.crop_rect,
                transform="hflip_label_swap",
            )
        )
    return augmented


class LRMarkerDataset(Dataset):
    def __init__(self, rows: Sequence[PreparedRow], image_size: int, augment: bool) -> None:
        self.rows = list(rows)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):  # type: ignore[override]
        packed = self.rows[idx]
        row = packed.row
        try:
            with Image.open(row.image_path) as img:
                image = img.convert("RGB")
                top, left, bottom, right = packed.crop_rect
                image = image.crop((left, top, right + 1, bottom + 1))
                if packed.transform == "hflip_label_swap":
                    image = TF.hflip(image)
                if self.augment:
                    if random.random() < 0.35:
                        image = TF.adjust_brightness(image, 0.9 + 0.2 * random.random())
                    if random.random() < 0.35:
                        image = TF.adjust_contrast(image, 0.9 + 0.2 * random.random())

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

        y = torch.tensor(packed.label, dtype=torch.long)
        sample_uid = row.sample_uid if packed.transform == "original" else f"{row.sample_uid}:{packed.transform}"
        meta = {
            "row_idx": row.row_idx,
            "sample_uid": sample_uid,
            "manifest_sample_id": row.manifest_sample_id,
            "split": packed.split,
            "transform": packed.transform,
            "image_path": row.image_path.as_posix(),
            "image_name": row.image_name,
            "manufacturer": row.manufacturer,
            "config_folder": row.config_folder,
            "su_giu_pred": row.su_giu_pred,
            "su_giu_conf": row.su_giu_conf,
            "match_score": row.match_score,
            "search_strategy": row.search_strategy,
            "crop_rect": packed.crop_rect,
        }
        return x, y, meta


def collate_drop_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    meta = [b[2] for b in batch]
    return x, y, meta


def make_loader(
    rows: Sequence[PreparedRow],
    image_size: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
) -> DataLoader:
    return DataLoader(
        LRMarkerDataset(rows, image_size=image_size, augment=augment),
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_drop_none,
        drop_last=False,
    )


class LRMarkerClassifier(nn.Module):
    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


def metrics_from_confusion(conf: np.ndarray) -> Dict[str, object]:
    total = int(conf.sum())
    correct = int(np.trace(conf))
    recalls: List[float] = []
    per_class: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(CLASS_NAMES):
        support = int(conf[i].sum())
        pred_count = int(conf[:, i].sum())
        tp = int(conf[i, i])
        recall = float(tp / support) if support else 0.0
        precision = float(tp / pred_count) if pred_count else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        if support:
            recalls.append(recall)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }
    return {
        "samples": total,
        "loss": 0.0,
        "acc": float(correct / total) if total else 0.0,
        "balanced_acc": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "confusion": conf.astype(int).tolist(),
        "per_class": per_class,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    class_weights: torch.Tensor,
    collect_predictions: bool = False,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    train = optimizer is not None
    model.train(mode=train)

    total = 0
    total_loss = 0.0
    conf = np.zeros((2, 2), dtype=np.int64)
    predictions: List[Dict[str, object]] = []

    for packed in loader:
        if packed is None:
            continue
        x, y, meta = packed
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

        probs = F.softmax(logits.detach(), dim=1)
        pred = torch.argmax(probs, dim=1)
        bs = x.size(0)
        total += bs
        total_loss += float(loss.detach().cpu()) * bs

        y_cpu = y.detach().cpu().numpy()
        p_cpu = pred.detach().cpu().numpy()
        prob_cpu = probs.detach().cpu().numpy()
        for idx, (yi, pi) in enumerate(zip(y_cpu, p_cpu)):
            yi_i = int(yi)
            pi_i = int(pi)
            conf[yi_i, pi_i] += 1
            if collect_predictions:
                crop_top, crop_left, crop_bottom, crop_right = meta[idx]["crop_rect"]
                predictions.append(
                    {
                        **meta[idx],
                        "true_idx": yi_i,
                        "true_label": CLASS_NAMES[yi_i],
                        "pred_idx": pi_i,
                        "pred_label": CLASS_NAMES[pi_i],
                        "prob_not_lr_flipped": float(prob_cpu[idx, 0]),
                        "prob_lr_flipped": float(prob_cpu[idx, 1]),
                        "correct": int(yi_i == pi_i),
                        "crop_top_abs": int(crop_top),
                        "crop_left_abs": int(crop_left),
                        "crop_bottom_abs": int(crop_bottom),
                        "crop_right_abs": int(crop_right),
                    }
                )

    metrics = metrics_from_confusion(conf)
    metrics["loss"] = float(total_loss / max(1, total))
    return metrics, predictions


def per_group_metrics(predictions: Sequence[Dict[str, object]], group_key: str) -> Dict[str, object]:
    grouped: DefaultDict[str, np.ndarray] = defaultdict(lambda: np.zeros((2, 2), dtype=np.int64))
    for pred in predictions:
        group = str(pred.get(group_key, "UNKNOWN") or "UNKNOWN")
        true_idx = int(pred["true_idx"])
        pred_idx = int(pred["pred_idx"])
        grouped[group][true_idx, pred_idx] += 1
    return {group: metrics_from_confusion(conf) for group, conf in sorted(grouped.items())}


def write_effective_manifest(path: Path, rows: Sequence[PreparedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_idx",
        "sample_uid",
        "manifest_sample_id",
        "effective_split",
        "transform",
        "label",
        "label_it",
        "source_label",
        "manufacturer",
        "config_folder",
        "image_path",
        "image_name",
        "su_giu_pred",
        "su_giu_conf",
        "match_score",
        "search_strategy",
        "echo_rect_top_abs",
        "echo_rect_left_abs",
        "echo_rect_bottom_abs",
        "echo_rect_right_abs",
        "crop_top_abs",
        "crop_left_abs",
        "crop_bottom_abs",
        "crop_right_abs",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for packed in sorted(rows, key=lambda r: (r.split, r.row.manufacturer, r.label, r.transform, r.row.row_idx)):
            row = packed.row
            top, left, bottom, right = row.echo_rect
            crop_top, crop_left, crop_bottom, crop_right = packed.crop_rect
            sample_uid = row.sample_uid if packed.transform == "original" else f"{row.sample_uid}:{packed.transform}"
            writer.writerow(
                {
                    "row_idx": row.row_idx,
                    "sample_uid": sample_uid,
                    "manifest_sample_id": row.manifest_sample_id,
                    "effective_split": packed.split,
                    "transform": packed.transform,
                    "label": CLASS_NAMES[packed.label],
                    "label_it": CLASS_LABEL_IT[packed.label],
                    "source_label": CLASS_NAMES[row.label],
                    "manufacturer": row.manufacturer,
                    "config_folder": row.config_folder,
                    "image_path": row.image_path.as_posix(),
                    "image_name": row.image_name,
                    "su_giu_pred": row.su_giu_pred,
                    "su_giu_conf": row.su_giu_conf,
                    "match_score": row.match_score,
                    "search_strategy": row.search_strategy,
                    "echo_rect_top_abs": top,
                    "echo_rect_left_abs": left,
                    "echo_rect_bottom_abs": bottom,
                    "echo_rect_right_abs": right,
                    "crop_top_abs": crop_top,
                    "crop_left_abs": crop_left,
                    "crop_bottom_abs": crop_bottom,
                    "crop_right_abs": crop_right,
                }
            )


def write_predictions(path: Path, predictions: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_idx",
        "sample_uid",
        "manifest_sample_id",
        "split",
        "transform",
        "manufacturer",
        "config_folder",
        "image_path",
        "image_name",
        "true_label",
        "pred_label",
        "prob_not_lr_flipped",
        "prob_lr_flipped",
        "correct",
        "su_giu_pred",
        "su_giu_conf",
        "match_score",
        "search_strategy",
        "crop_top_abs",
        "crop_left_abs",
        "crop_bottom_abs",
        "crop_right_abs",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for pred in predictions:
            writer.writerow(pred)


def write_confusion_csv(path: Path, conf: Sequence[Sequence[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for idx, name in enumerate(CLASS_NAMES):
            writer.writerow([name, *list(conf[idx])])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train binary LR marker classifier.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/20_datasets/lr_marker_sugiu_v2_reviewed_v2/manifest_train_ready.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/30_models/lr_marker_classifier_generic_v2"),
    )
    parser.add_argument("--split-mode", choices=("manifest", "stratified"), default="stratified")
    parser.add_argument("--no-vendor-stratification", action="store_true")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--crop-mode", choices=("echo", "padded_echo"), default="padded_echo")
    parser.add_argument("--input-padding-frac", type=float, default=0.12)
    parser.add_argument(
        "--add-hflip-label-swap-train",
        action="store_true",
        help="Duplicate train rows with horizontal flip and inverted LR label. Val/test remain real rows.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    set_seed(int(args.seed))

    manifest = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(manifest)
    if not rows:
        raise RuntimeError(f"No usable rows found in manifest: {manifest}")

    if args.split_mode == "manifest":
        assigned = assign_manifest_split(rows)
    else:
        assigned = assign_stratified_split(
            rows,
            seed=int(args.seed),
            val_frac=float(args.val_frac),
            test_frac=float(args.test_frac),
            stratify_by_vendor=not bool(args.no_vendor_stratification),
        )

    prepared = prepare_rows(
        rows,
        assigned_split=assigned,
        crop_mode=args.crop_mode,
        padding_frac=float(args.input_padding_frac),
    )
    if bool(args.add_hflip_label_swap_train):
        prepared = add_hflip_label_swap_rows(prepared)
    split_rows = {
        "train": [r for r in prepared if r.split == "train"],
        "val": [r for r in prepared if r.split == "val"],
        "test": [r for r in prepared if r.split == "test"],
    }
    counts = split_counts(prepared)
    if not split_rows["train"] or not split_rows["val"] or not split_rows["test"]:
        raise RuntimeError(f"Empty split after assignment: {counts}")
    if any(counts[split][CLASS_NAMES[1]] == 0 for split in ("train", "val", "test")):
        raise RuntimeError(f"At least one split has no lr_flipped rows: {counts}")
    if any(counts[split][CLASS_NAMES[0]] == 0 for split in ("train", "val", "test")):
        raise RuntimeError(f"At least one split has no not_lr_flipped rows: {counts}")

    effective_manifest = output_dir / "effective_manifest.csv"
    write_effective_manifest(effective_manifest, prepared)

    device = choose_device(args.device)
    print("rows", len(prepared), flush=True)
    print("split_counts", json.dumps(counts, ensure_ascii=False), flush=True)
    print("device", device, flush=True)
    print("effective_manifest", effective_manifest, flush=True)

    train_loader = make_loader(split_rows["train"], args.image_size, args.batch_size, args.num_workers, augment=True)
    val_loader = make_loader(split_rows["val"], args.image_size, args.batch_size, args.num_workers, augment=False)
    test_loader = make_loader(split_rows["test"], args.image_size, args.batch_size, args.num_workers, augment=False)

    model = LRMarkerClassifier(pretrained=bool(args.pretrained)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))

    train_label_counts = Counter(r.label for r in split_rows["train"])
    train_counts = np.array([train_label_counts.get(0, 0), train_label_counts.get(1, 0)], dtype=np.float32)
    weights = np.zeros((2,), dtype=np.float32)
    mask = train_counts > 0
    weights[mask] = train_counts[mask].sum() / (train_counts[mask] * mask.sum())
    if mask.any():
        weights[mask] /= weights[mask].mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    best_path = output_dir / "best_model.pt"
    last_checkpoint = output_dir / "last_checkpoint.pt"
    history_path = output_dir / "history.json"
    history: List[Dict[str, object]] = []
    best_val = -1.0
    best_epoch = -1
    patience = 0
    train_args = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "manifest": manifest.as_posix(),
        "class_names": list(CLASS_NAMES),
        "split_counts": counts,
    }

    for epoch in range(1, int(args.epochs) + 1):
        train_m, _ = run_epoch(model, train_loader, device, optimizer, class_weights)
        val_m, _ = run_epoch(model, val_loader, device, None, class_weights)
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
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_balanced_acc": best_val,
                    "class_names": list(CLASS_NAMES),
                    "label_it": list(CLASS_LABEL_IT),
                    "args": train_args,
                },
                best_path,
            )
        else:
            patience += 1

        atomic_torch_save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_epoch": best_epoch,
                "best_val_balanced_acc": best_val,
                "class_names": list(CLASS_NAMES),
                "args": train_args,
                "history": history,
                "patience": patience,
            },
            last_checkpoint,
        )

        if patience >= int(args.early_stopping_patience):
            print("early stopping", flush=True)
            break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m, test_predictions = run_epoch(
        model,
        test_loader,
        device,
        None,
        class_weights,
        collect_predictions=True,
    )
    val_m, val_predictions = run_epoch(
        model,
        val_loader,
        device,
        None,
        class_weights,
        collect_predictions=True,
    )

    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    write_confusion_csv(output_dir / "test_confusion_matrix.csv", test_m["confusion"])
    write_predictions(output_dir / "test_predictions.csv", test_predictions)
    write_predictions(output_dir / "val_predictions.csv", val_predictions)

    summary = {
        "manifest": manifest.as_posix(),
        "effective_manifest": effective_manifest.as_posix(),
        "device": str(device),
        "class_names": list(CLASS_NAMES),
        "label_it": list(CLASS_LABEL_IT),
        "split_counts": counts,
        "class_weights": [float(x) for x in weights.tolist()],
        "best_epoch": best_epoch,
        "best_val_balanced_acc": best_val,
        "final_val_metrics": val_m,
        "test_metrics": test_m,
        "test_metrics_by_vendor": per_group_metrics(test_predictions, "manufacturer"),
        "test_metrics_by_search_strategy": per_group_metrics(test_predictions, "search_strategy"),
        "best_model": best_path.as_posix(),
        "last_checkpoint": last_checkpoint.as_posix(),
        "history_path": history_path.as_posix(),
        "args": train_args,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("best_model", best_path, flush=True)
    print("train_summary", output_dir / "train_summary.json", flush=True)
    print("test_acc", test_m["acc"], flush=True)
    print("test_balanced_acc", test_m["balanced_acc"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
