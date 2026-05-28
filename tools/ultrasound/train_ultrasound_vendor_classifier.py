#!/usr/bin/env python3
"""Train a vendor classifier (manufacturer) from HDMI ultrasound frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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


def load_manifest_rows(manifest_path: Path) -> List[SampleRow]:
    rows: List[SampleRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                SampleRow(
                    image_path=Path(row["image_path"]).expanduser().resolve(),
                    split=row["split"].strip().lower(),
                    manufacturer=row["manufacturer"].strip(),
                    model_name=row["model_name"].strip(),
                )
            )
    return rows


def subset_rows(rows: List[SampleRow], limit: int, seed: int) -> List[SampleRow]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    chosen = sorted(idx[:limit])
    return [rows[i] for i in chosen]


class VendorDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[SampleRow],
        label_to_idx: Dict[str, int],
        image_size: int,
        augment: bool,
    ) -> None:
        self.rows = list(rows)
        self.label_to_idx = label_to_idx
        self.image_size = image_size
        self.augment = augment
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        self.train_jitter = transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.08,
            hue=0.02,
        )
        self.train_erasing = transforms.RandomErasing(
            p=0.12,
            scale=(0.01, 0.08),
            ratio=(0.4, 2.5),
            value="random",
            inplace=False,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):  # type: ignore[override]
        row = self.rows[idx]
        with Image.open(row.image_path) as img:
            image = img.convert("RGB")
            image = TF.resize(
                image,
                size=[self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            if self.augment:
                if random.random() < 0.45:
                    image = self.train_jitter(image)
                if random.random() < 0.10:
                    image = TF.gaussian_blur(image, kernel_size=3)
            tensor = TF.to_tensor(image)
            if self.augment:
                tensor = self.train_erasing(tensor)
            tensor = self.normalize(tensor)

        target = self.label_to_idx[row.manufacturer]
        metadata = {
            "manufacturer": row.manufacturer,
            "model_name": row.model_name,
            "image_path": row.image_path.as_posix(),
        }
        return tensor, target, metadata


def collate_batch(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    metadata = [item[2] for item in batch]
    return images, targets, metadata


class VendorClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feats = self.backbone(images)
        return self.head(feats)


def _compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
    idx_to_label: Sequence[str],
) -> Dict[str, object]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    support = confusion.sum(axis=1)
    pred_count = confusion.sum(axis=0)
    tp = np.diag(confusion)

    per_class: List[Dict[str, object]] = []
    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0
    classes_with_support = 0

    for i in range(num_classes):
        precision = float(tp[i] / pred_count[i]) if pred_count[i] > 0 else 0.0
        recall = float(tp[i] / support[i]) if support[i] > 0 else 0.0
        denom = precision + recall
        f1 = float(2.0 * precision * recall / denom) if denom > 0 else 0.0

        if support[i] > 0:
            macro_p += precision
            macro_r += recall
            macro_f1 += f1
            classes_with_support += 1

        per_class.append(
            {
                "manufacturer": idx_to_label[i],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support[i]),
            }
        )

    macro_div = max(1, classes_with_support)
    accuracy = float(tp.sum() / max(1, confusion.sum()))
    return {
        "accuracy": accuracy,
        "macro_precision": macro_p / macro_div,
        "macro_recall": macro_r / macro_div,
        "macro_f1": macro_f1 / macro_div,
        "classes_with_support": classes_with_support,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    phase: str,
    idx_to_label: Sequence[str],
    log_interval: int = 0,
) -> Dict[str, object]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_samples = 0
    total_steps = len(loader)
    epoch_start = time.time()

    y_true: List[int] = []
    y_pred: List[int] = []

    for step_idx, (images, targets, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)
            if is_train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = images.size(0)
        total_samples += batch_size
        loss_value = float(loss.detach().cpu())
        if math.isfinite(loss_value):
            total_loss += loss_value * batch_size

        preds = torch.argmax(logits.detach(), dim=1)
        y_true.extend(targets.detach().cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        if log_interval > 0 and (step_idx % log_interval == 0 or step_idx == total_steps):
            elapsed = time.time() - epoch_start
            progress = step_idx / max(1, total_steps)
            eta = (elapsed / progress - elapsed) if progress > 0 else 0.0
            running_loss = total_loss / max(1, total_samples)
            running_acc = float(
                (np.array(y_true, dtype=np.int64) == np.array(y_pred, dtype=np.int64)).mean()
            )
            print(
                f"[Epoch {epoch:03d}][{phase}] step {step_idx}/{total_steps} "
                f"({progress * 100:.1f}%) loss {running_loss:.4f} acc {running_acc:.4f} "
                f"elapsed {elapsed:.1f}s eta {max(0.0, eta):.1f}s",
                flush=True,
            )

    mean_loss = total_loss / max(1, total_samples)
    metrics = _compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=len(idx_to_label),
        idx_to_label=idx_to_label,
    )
    metrics["loss"] = mean_loss
    metrics["samples"] = total_samples
    return metrics


def make_loader(
    rows: List[SampleRow],
    label_to_idx: Dict[str, int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
    use_balanced_sampler: bool,
) -> DataLoader:
    dataset = VendorDataset(rows=rows, label_to_idx=label_to_idx, image_size=image_size, augment=augment)

    sampler = None
    shuffle = False
    if augment:
        if use_balanced_sampler:
            counts = Counter(row.manufacturer for row in rows)
            sample_weights = [1.0 / counts[row.manufacturer] for row in rows]
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
        else:
            shuffle = True

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_batch,
        drop_last=False,
    )


def _build_class_weights(
    rows: Sequence[SampleRow],
    label_to_idx: Dict[str, int],
    power: float,
) -> torch.Tensor:
    counts = Counter(row.manufacturer for row in rows)
    weights = torch.ones(len(label_to_idx), dtype=torch.float32)
    total = float(sum(counts.values()))
    num_classes = float(len(label_to_idx))
    safe_power = max(0.0, float(power))
    for label, idx in label_to_idx.items():
        cls_count = float(counts[label])
        base_weight = total / (num_classes * max(1.0, cls_count))
        weights[idx] = float(base_weight ** safe_power)
    weights = weights / weights.mean().clamp(min=1e-12)
    return weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train manufacturer classifier from ultrasound HDMI frames."
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
        default=Path("artifacts/30_models/vendor_training"),
        help="Cartella output checkpoint e metriche.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=3e-4)
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
        "--log-interval",
        type=int,
        default=0,
        help="Log progresso ogni N batch (0 = solo fine epoca).",
    )
    parser.add_argument(
        "--disable-balanced-sampler",
        action="store_true",
        help="Disabilita sampler bilanciato in train.",
    )
    parser.add_argument(
        "--disable-class-weights",
        action="store_true",
        help="Disabilita pesi di classe nella loss.",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.5,
        help="Esponente dei pesi di classe inversi (0=uniformi, 1=inversi pieni).",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.03,
        help="Label smoothing CrossEntropy (range [0, 1)).",
    )
    return parser


def _write_per_class_csv(path: Path, per_class: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["manufacturer", "precision", "recall", "f1", "support"])
        for row in per_class:
            writer.writerow(
                [
                    row["manufacturer"],
                    f"{float(row['precision']):.6f}",
                    f"{float(row['recall']):.6f}",
                    f"{float(row['f1']):.6f}",
                    int(row["support"]),
                ]
            )


def _write_confusion_csv(path: Path, confusion: Sequence[Sequence[int]], labels: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, confusion):
            writer.writerow([label, *row])


def main() -> int:
    args = build_parser().parse_args()
    if not (0.0 <= args.label_smoothing < 1.0):
        raise ValueError("--label-smoothing deve essere in [0, 1).")
    if args.class_weight_power < 0.0:
        raise ValueError("--class-weight-power deve essere >= 0.")
    set_seed(args.seed)

    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest_rows(manifest_path)
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

    manufacturers_train = sorted({row.manufacturer for row in split_rows["train"]})
    label_to_idx = {label: idx for idx, label in enumerate(manufacturers_train)}
    idx_to_label = manufacturers_train

    unseen_val = sorted({row.manufacturer for row in split_rows["val"] if row.manufacturer not in label_to_idx})
    unseen_test = sorted({row.manufacturer for row in split_rows["test"] if row.manufacturer not in label_to_idx})
    if unseen_val or unseen_test:
        raise RuntimeError(
            "Classi presenti in val/test ma assenti in train. "
            f"val_only={unseen_val}, test_only={unseen_test}"
        )

    device = choose_device(args.device)
    print(f"Device: {device}", flush=True)
    print(
        "Samples train/val/test: "
        f"{len(split_rows['train'])}/{len(split_rows['val'])}/{len(split_rows['test'])}"
    , flush=True)
    print(f"Manufacturers (train classes): {len(idx_to_label)}", flush=True)

    train_loader = make_loader(
        split_rows["train"],
        label_to_idx=label_to_idx,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        use_balanced_sampler=not args.disable_balanced_sampler,
    )
    val_loader = make_loader(
        split_rows["val"],
        label_to_idx=label_to_idx,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
        use_balanced_sampler=False,
    )
    test_loader = make_loader(
        split_rows["test"],
        label_to_idx=label_to_idx,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
        use_balanced_sampler=False,
    )

    model = VendorClassifier(num_classes=len(idx_to_label), pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )
    class_weights: Optional[torch.Tensor] = None
    if not args.disable_class_weights:
        class_weights = _build_class_weights(
            split_rows["train"],
            label_to_idx=label_to_idx,
            power=args.class_weight_power,
        ).to(device)
        print(
            f"Class weights: enabled (power={args.class_weight_power:.3f})",
            flush=True,
        )
    else:
        print("Class weights: disabled", flush=True)
    print(f"Label smoothing: {args.label_smoothing:.3f}", flush=True)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(args.label_smoothing),
    )

    history: List[Dict[str, object]] = []
    best_val_macro_f1 = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            phase="train",
            idx_to_label=idx_to_label,
            log_interval=args.log_interval,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            criterion=criterion,
            optimizer=None,
            epoch=epoch,
            phase="val",
            idx_to_label=idx_to_label,
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
            f"train loss {train_metrics['loss']:.4f}, acc {train_metrics['accuracy']:.4f}, macroF1 {train_metrics['macro_f1']:.4f} | "
            f"val loss {val_metrics['loss']:.4f}, acc {val_metrics['accuracy']:.4f}, macroF1 {val_metrics['macro_f1']:.4f}"
        , flush=True)

        curr_val_f1 = float(val_metrics["macro_f1"])
        if curr_val_f1 > best_val_macro_f1:
            best_val_macro_f1 = curr_val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1": best_val_macro_f1,
                    "class_names": idx_to_label,
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
                    "Early stopping: nessun miglioramento su validation macro-F1 "
                    f"per {args.early_stopping_patience} epoche."
                , flush=True)
                break

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(
        model,
        test_loader,
        device,
        criterion=criterion,
        optimizer=None,
        epoch=best_epoch if best_epoch > 0 else args.epochs,
        phase="test",
        idx_to_label=idx_to_label,
        log_interval=args.log_interval,
    )
    print(
        f"Best epoch: {best_epoch} | best val macro-F1: {best_val_macro_f1:.4f} | "
        f"test acc: {test_metrics['accuracy']:.4f} | test macro-F1: {test_metrics['macro_f1']:.4f}"
    , flush=True)

    metrics_path = output_dir / "metrics.json"
    metrics = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "class_names": idx_to_label,
        "test": test_metrics,
        "history": history,
        "train_samples": len(split_rows["train"]),
        "val_samples": len(split_rows["val"]),
        "test_samples": len(split_rows["test"]),
        "device": str(device),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    per_class_path = output_dir / "test_per_manufacturer_metrics.csv"
    _write_per_class_csv(per_class_path, test_metrics["per_class"])

    confusion_path = output_dir / "test_confusion_matrix.csv"
    _write_confusion_csv(confusion_path, test_metrics["confusion_matrix"], idx_to_label)

    print(f"Checkpoint best model: {best_model_path}", flush=True)
    print(f"Metriche complete: {metrics_path}", flush=True)
    print(f"Metriche per produttore: {per_class_path}", flush=True)
    print(f"Confusion matrix: {confusion_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
