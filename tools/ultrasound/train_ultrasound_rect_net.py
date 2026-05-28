#!/usr/bin/env python3
"""Train a single-network bbox regressor for ultrasound rectangle detection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class SampleRow:
    image_path: Path
    split: str
    manufacturer: str
    model_name: str
    x1: float
    y1: float
    x2: float
    y2: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[float, float, float, float]:
    x1 = float(max(0.0, min(x1, width - 1)))
    y1 = float(max(0.0, min(y1, height - 1)))
    x2 = float(max(1.0, min(x2, width)))
    y2 = float(max(1.0, min(y2, height)))
    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return x1, y1, x2, y2


class RectDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[SampleRow],
        image_size: int,
        augment: bool,
    ) -> None:
        self.rows = list(rows)
        self.image_size = image_size
        self.augment = augment
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):  # type: ignore[override]
        row = self.rows[idx]
        with Image.open(row.image_path) as img:
            image = img.convert("RGB")
            width, height = image.size

            x1, y1, x2, y2 = _clamp_box(row.x1, row.y1, row.x2, row.y2, width, height)

            if self.augment:
                # Explicit LR/UD augmentation keeps geometric symmetry constraints.
                if random.random() < 0.5:
                    image = TF.hflip(image)
                    x1, x2 = width - x2, width - x1
                if random.random() < 0.5:
                    image = TF.vflip(image)
                    y1, y2 = height - y2, height - y1

            scale_x = self.image_size / width
            scale_y = self.image_size / height
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y

            image = TF.resize(
                image,
                size=[self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.to_tensor(image)
            tensor = self.normalize(tensor)

            target = torch.tensor(
                [
                    x1 / self.image_size,
                    y1 / self.image_size,
                    x2 / self.image_size,
                    y2 / self.image_size,
                ],
                dtype=torch.float32,
            )

            metadata = {
                "manufacturer": row.manufacturer,
                "model_name": row.model_name,
                "image_path": row.image_path.as_posix(),
            }
            return tensor, target, metadata


def collate_batch(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    metadata = [item[2] for item in batch]
    return images, targets, metadata


class RectRegressor(nn.Module):
    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feats = self.backbone(images)
        return self.head(feats)


def normalize_box_order(boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
    out = torch.stack([x1, y1, x2, y2], dim=1)
    return torch.clamp(out, 0.0, 1.0)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter_x1 = torch.maximum(a[:, 0], b[:, 0])
    inter_y1 = torch.maximum(a[:, 1], b[:, 1])
    inter_x2 = torch.minimum(a[:, 2], b[:, 2])
    inter_y2 = torch.minimum(a[:, 3], b[:, 3])

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h

    area_a = torch.clamp(a[:, 2] - a[:, 0], min=0.0) * torch.clamp(a[:, 3] - a[:, 1], min=0.0)
    area_b = torch.clamp(b[:, 2] - b[:, 0], min=0.0) * torch.clamp(b[:, 3] - b[:, 1], min=0.0)
    union = area_a + area_b - inter_area + 1e-8
    return inter_area / union


def compute_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_boxes = normalize_box_order(pred_boxes)
    target_boxes = normalize_box_order(target_boxes)
    iou = box_iou(pred_boxes, target_boxes)
    iou = torch.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)
    l1 = F.smooth_l1_loss(pred_boxes, target_boxes)
    l1 = torch.nan_to_num(l1, nan=1.0, posinf=1.0, neginf=1.0)
    loss = l1 + (1.0 - iou).mean()
    loss = torch.nan_to_num(loss, nan=10.0, posinf=10.0, neginf=10.0)
    return loss, iou


def choose_device(prefer: Optional[str]) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_manifest_rows(
    manifest_path: Path,
    strict_manifest: bool = False,
) -> Tuple[List[SampleRow], int]:
    rows: List[SampleRow] = []
    missing_count = 0
    filtered_exclude_count = 0
    filtered_keyword_count = 0
    filtered_invalid_bbox_count = 0
    missing_examples: List[str] = []
    filtered_examples: List[str] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_image_path = str(row.get("image_path", "")).strip()
            if not raw_image_path:
                filtered_invalid_bbox_count += 1
                continue

            action = str(row.get("manual_folder_action", "")).strip().lower()
            if action == "exclude":
                filtered_exclude_count += 1
                if len(filtered_examples) < 5:
                    filtered_examples.append(raw_image_path)
                continue

            image_path_lower = raw_image_path.lower()
            if ("negative" in image_path_lower) or ("proibite" in image_path_lower):
                filtered_keyword_count += 1
                if len(filtered_examples) < 5:
                    filtered_examples.append(raw_image_path)
                continue

            image_path = Path(raw_image_path).expanduser().resolve()
            if not image_path.exists():
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(image_path.as_posix())
                continue

            try:
                x1 = float(row["bbox_xmin"])
                y1 = float(row["bbox_ymin"])
                x2 = float(row["bbox_xmax"])
                y2 = float(row["bbox_ymax"])
            except Exception:
                filtered_invalid_bbox_count += 1
                if len(filtered_examples) < 5:
                    filtered_examples.append(image_path.as_posix())
                continue

            if not (x2 > x1 and y2 > y1):
                filtered_invalid_bbox_count += 1
                if len(filtered_examples) < 5:
                    filtered_examples.append(image_path.as_posix())
                continue

            rows.append(
                SampleRow(
                    image_path=image_path,
                    split=row["split"].strip().lower(),
                    manufacturer=row["manufacturer"].strip(),
                    model_name=row["model_name"].strip(),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
    if missing_count > 0:
        msg = (
            f"Manifest contiene {missing_count} immagini mancanti. "
            "Le righe sono state ignorate."
        )
        if strict_manifest:
            raise RuntimeError(msg + f" Esempi: {missing_examples}")
        print(msg, flush=True)
        for sample in missing_examples:
            print(f"- missing: {sample}", flush=True)
    filtered_total = filtered_exclude_count + filtered_keyword_count + filtered_invalid_bbox_count
    if filtered_total > 0:
        print(
            "Manifest filtering applied: "
            f"exclude={filtered_exclude_count}, "
            f"keyword_negative_proibite={filtered_keyword_count}, "
            f"invalid_bbox={filtered_invalid_bbox_count}",
            flush=True,
        )
        for sample in filtered_examples:
            print(f"- filtered: {sample}", flush=True)
    return rows, missing_count


def subset_rows(rows: List[SampleRow], limit: int, seed: int) -> List[SampleRow]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    chosen = sorted(idx[:limit])
    return [rows[i] for i in chosen]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    phase: str,
    log_interval: int = 0,
) -> Dict[str, object]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_iou = 0.0
    total_samples = 0
    total_steps = len(loader)
    epoch_start = time.time()

    per_manufacturer: Dict[str, List[float]] = defaultdict(list)
    per_model: Dict[str, List[float]] = defaultdict(list)

    for step_idx, (images, targets, metadata) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            preds = model(images)
            loss, iou = compute_loss(preds, targets)
            if is_train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = images.size(0)
        total_samples += batch_size
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            continue
        total_loss += loss_value * batch_size
        iou_cpu = iou.detach().cpu()
        total_iou += float(iou_cpu.mean()) * batch_size

        for i in range(batch_size):
            manufacturer = metadata[i]["manufacturer"]
            model_name = metadata[i]["model_name"]
            iou_value = float(iou_cpu[i])
            per_manufacturer[manufacturer].append(iou_value)
            per_model[model_name].append(iou_value)

        if log_interval > 0 and (step_idx % log_interval == 0 or step_idx == total_steps):
            elapsed = time.time() - epoch_start
            progress = step_idx / max(1, total_steps)
            eta = (elapsed / progress - elapsed) if progress > 0 else 0.0
            running_loss = total_loss / max(1, total_samples)
            running_iou = total_iou / max(1, total_samples)
            print(
                f"[Epoch {epoch:03d}][{phase}] step {step_idx}/{total_steps} "
                f"({progress * 100:.1f}%) loss {running_loss:.4f} iou {running_iou:.4f} "
                f"elapsed {elapsed:.1f}s eta {max(0.0, eta):.1f}s",
                flush=True,
            )

    mean_loss = total_loss / max(1, total_samples)
    mean_iou = total_iou / max(1, total_samples)

    return {
        "loss": mean_loss,
        "mean_iou": mean_iou,
        "samples": total_samples,
        "per_manufacturer_iou": {
            key: float(sum(values) / len(values)) for key, values in sorted(per_manufacturer.items())
        },
        "per_model_iou": {
            key: float(sum(values) / len(values)) for key, values in sorted(per_model.items())
        },
    }


def make_loader(
    rows: List[SampleRow],
    image_size: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
) -> DataLoader:
    dataset = RectDataset(rows=rows, image_size=image_size, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_batch,
        drop_last=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train bbox regressor for ultrasound RECT_ECHO."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/20_datasets/vendor_template_dataset_v1/manifest_vendor_template_no_negative.csv"),
        help="Manifest CSV (bbox+template) senza negativi.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/30_models/rect_training"),
        help="Cartella output checkpoint e metriche.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (es. cpu, cuda, mps).",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Usa pesi ImageNet su ResNet18.",
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument(
        "--strict-manifest",
        action="store_true",
        help="Fallisce se il manifest contiene immagini mancanti.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="Log progresso ogni N batch (0 = solo fine epoca).",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Riprende training da un checkpoint esistente (best_model.pt).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)

    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    resume_checkpoint_path: Optional[Path] = None
    if args.resume_checkpoint is not None:
        resume_checkpoint_path = args.resume_checkpoint.expanduser().resolve()
        if not resume_checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, missing_rows = load_manifest_rows(
        manifest_path,
        strict_manifest=args.strict_manifest,
    )
    split_rows = {
        "train": [row for row in rows if row.split == "train"],
        "val": [row for row in rows if row.split == "val"],
        "test": [row for row in rows if row.split == "test"],
    }

    for split, limit in (
        ("train", args.max_train_samples),
        ("val", args.max_val_samples),
        ("test", args.max_test_samples),
    ):
        split_seed_offset = {"train": 11, "val": 23, "test": 37}[split]
        split_rows[split] = subset_rows(split_rows[split], limit=limit, seed=args.seed + split_seed_offset)

    if not split_rows["train"] or not split_rows["val"] or not split_rows["test"]:
        raise RuntimeError(
            "Train/val/test devono contenere almeno un campione. "
            f"Conteggi: train={len(split_rows['train'])}, val={len(split_rows['val'])}, test={len(split_rows['test'])}"
        )

    device = choose_device(args.device)
    print(f"Device: {device}", flush=True)
    print(
        f"Samples train/val/test: "
        f"{len(split_rows['train'])}/{len(split_rows['val'])}/{len(split_rows['test'])}"
    , flush=True)
    if missing_rows > 0:
        print(f"Righe ignorate dal manifest (file mancanti): {missing_rows}", flush=True)

    train_loader = make_loader(
        split_rows["train"],
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
    )
    val_loader = make_loader(
        split_rows["val"],
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
    )
    test_loader = make_loader(
        split_rows["test"],
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
    )

    model = RectRegressor(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )

    history: List[Dict[str, object]] = []
    best_val_iou = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    start_epoch = 1
    best_model_path = output_dir / "best_model.pt"

    if resume_checkpoint_path is not None:
        resume_ckpt = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])

        optimizer_state = resume_ckpt.get("optimizer_state_dict")
        if isinstance(optimizer_state, dict):
            try:
                optimizer.load_state_dict(optimizer_state)
            except ValueError as exc:
                print(
                    f"Warning: optimizer state non compatibile, continuo con optimizer nuovo ({exc})",
                    flush=True,
                )

        start_epoch = max(1, int(resume_ckpt.get("epoch", 0)) + 1)
        best_val_iou = float(resume_ckpt.get("best_val_iou", -math.inf))
        if not math.isfinite(best_val_iou):
            best_val_iou = -math.inf
        best_epoch = int(resume_ckpt.get("epoch", -1))
        epochs_without_improvement = 0

        if resume_checkpoint_path != best_model_path:
            torch.save(resume_ckpt, best_model_path)

        print(
            f"Resume checkpoint: {resume_checkpoint_path} | "
            f"start_epoch={start_epoch} | best_epoch={best_epoch} | "
            f"best_val_iou={best_val_iou:.4f}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            epoch=epoch,
            phase="train",
            log_interval=args.log_interval,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            epoch=epoch,
            phase="val",
            log_interval=args.log_interval,
        )
        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f}, iou {train_metrics['mean_iou']:.4f} | "
            f"val loss {val_metrics['loss']:.4f}, iou {val_metrics['mean_iou']:.4f}"
        , flush=True)

        if float(val_metrics["mean_iou"]) > best_val_iou:
            best_val_iou = float(val_metrics["mean_iou"])
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_iou": best_val_iou,
                    "args": {
                        key: (str(value) if isinstance(value, Path) else value)
                        for key, value in vars(args).items()
                    },
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(
                    "Early stopping: nessun miglioramento su validation IoU "
                    f"per {args.early_stopping_patience} epoche."
                , flush=True)
                break

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(
        model,
        test_loader,
        device,
        optimizer=None,
        epoch=best_epoch if best_epoch > 0 else args.epochs,
        phase="test",
        log_interval=args.log_interval,
    )
    print(
        f"Best epoch: {best_epoch} | best val IoU: {best_val_iou:.4f} | "
        f"test IoU: {test_metrics['mean_iou']:.4f}"
    , flush=True)

    metrics_path = output_dir / "metrics.json"
    metrics = {
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "test": test_metrics,
        "history": history,
        "train_samples": len(split_rows["train"]),
        "val_samples": len(split_rows["val"]),
        "test_samples": len(split_rows["test"]),
        "device": str(device),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    per_manufacturer_path = output_dir / "test_per_manufacturer_iou.csv"
    with per_manufacturer_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["manufacturer", "mean_iou"])
        for name, value in sorted(test_metrics["per_manufacturer_iou"].items()):
            writer.writerow([name, f"{value:.6f}"])

    per_model_path = output_dir / "test_per_model_iou.csv"
    with per_model_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model_name", "mean_iou"])
        for name, value in sorted(test_metrics["per_model_iou"].items()):
            writer.writerow([name, f"{value:.6f}"])

    print(f"Checkpoint best model: {best_model_path}", flush=True)
    print(f"Metriche complete: {metrics_path}", flush=True)
    print(f"IoU per produttore: {per_manufacturer_path}", flush=True)
    print(f"IoU per modello: {per_model_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
