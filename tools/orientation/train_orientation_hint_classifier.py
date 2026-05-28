#!/usr/bin/env python3
"""Train orientation hint classifier (NF/LR/UD/LRUD) from frame manifest hints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

@dataclass(frozen=True)
class Row:
    image_path: Path
    split: str
    manufacturer: str
    model_name: str
    label: int


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


class HintDataset(Dataset):
    def __init__(self, rows: Sequence[Row], image_size: int, augment: bool) -> None:
        self.rows = list(rows)
        self.image_size = image_size
        self.augment = augment
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
                    if random.random() < 0.2:
                        image = TF.adjust_brightness(image, 0.9 + 0.2 * random.random())
                    if random.random() < 0.2:
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

        y = torch.tensor(row.label, dtype=torch.long)
        meta = {
            "image_path": row.image_path.as_posix(),
            "manufacturer": row.manufacturer,
            "model_name": row.model_name,
        }
        return x, y, meta


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    meta = [b[2] for b in batch]
    return x, y, meta


class OrientationHintClassifier(nn.Module):
    def __init__(self, pretrained: bool, num_classes: int = 4) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet18(weights=weights)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        self.model = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


def load_rows(frame_manifest: Path) -> List[Row]:
    rows: List[Row] = []
    with frame_manifest.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label_s = row.get("orientation_hint", "")
            if label_s == "":
                continue
            label = int(float(label_s))
            if label < 0 or label > 3:
                continue
            image_path = Path(row["image_path"]).expanduser().resolve()
            if not image_path.exists():
                continue
            rows.append(
                Row(
                    image_path=image_path,
                    split=row.get("split", "train").strip().lower() or "train",
                    manufacturer=row.get("manufacturer", "UNKNOWN"),
                    model_name=row.get("model_name", "UNKNOWN"),
                    label=label,
                )
            )
    return rows


def make_loader(rows: List[Row], image_size: int, batch_size: int, num_workers: int, augment: bool) -> DataLoader:
    ds = HintDataset(rows, image_size=image_size, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate,
        drop_last=False,
    )


def run_epoch(
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
    conf = np.zeros((4, 4), dtype=np.int64)

    for packed in loader:
        if packed is None:
            continue
        x, y, _ = packed
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

    acc = correct / max(1, total)
    bal = []
    for c in range(4):
        denom = conf[c].sum()
        if denom > 0:
            bal.append(conf[c, c] / denom)
    bal_acc = float(sum(bal) / len(bal)) if bal else 0.0

    return {
        "loss": total_loss / max(1, total),
        "acc": acc,
        "balanced_acc": bal_acc,
        "samples": total,
        "confusion": conf.tolist(),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train orientation hint classifier.")
    p.add_argument("--frame-manifest", type=Path, default=Path("artifacts/20_datasets/orientation_symbolic_dataset/frame_manifest.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/30_models/orientation_hint_classifier"))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--early-stopping-patience", type=int, default=6)
    return p


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.frame_manifest.expanduser().resolve())
    split_rows = {
        "train": [r for r in rows if r.split == "train"],
        "val": [r for r in rows if r.split == "val"],
        "test": [r for r in rows if r.split == "test"],
    }
    if not split_rows["train"] or not split_rows["val"] or not split_rows["test"]:
        raise RuntimeError("train/val/test empty")

    def lbl_counts(rs: Sequence[Row]) -> Dict[int, int]:
        c = Counter([r.label for r in rs])
        return {i: int(c.get(i, 0)) for i in range(4)}

    print("train", len(split_rows["train"]), lbl_counts(split_rows["train"]), flush=True)
    print("val", len(split_rows["val"]), lbl_counts(split_rows["val"]), flush=True)
    print("test", len(split_rows["test"]), lbl_counts(split_rows["test"]), flush=True)

    device = choose_device(args.device)
    print("device", device, flush=True)

    train_loader = make_loader(split_rows["train"], args.image_size, args.batch_size, args.num_workers, augment=True)
    val_loader = make_loader(split_rows["val"], args.image_size, args.batch_size, args.num_workers, augment=False)
    test_loader = make_loader(split_rows["test"], args.image_size, args.batch_size, args.num_workers, augment=False)

    model = OrientationHintClassifier(pretrained=args.pretrained, num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    train_counts = np.array([lbl_counts(split_rows["train"]).get(i, 0) for i in range(4)], dtype=np.float32)
    weights = np.zeros((4,), dtype=np.float32)
    mask = train_counts > 0
    weights[mask] = train_counts[mask].sum() / (train_counts[mask] * mask.sum())
    if mask.any():
        weights[mask] /= weights[mask].mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    best_val = -1.0
    best_epoch = -1
    best_path = out / "best_model.pt"
    history: List[Dict[str, object]] = []
    patience = 0

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, device, optimizer, class_weights)
        val_m = run_epoch(model, val_loader, device, None, class_weights)
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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_balanced_acc": best_val,
                    "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                },
                best_path,
            )
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                print("early stopping", flush=True)
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = run_epoch(model, test_loader, device, None, class_weights)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_balanced_acc": best_val,
        "test": test_m,
        "history": history,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # confusion CSV
    conf = np.array(test_m["confusion"], dtype=np.int64)
    with (out / "test_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred", "NF", "LR", "UD", "LRUD"])
        for i, name in enumerate(["NF", "LR", "UD", "LRUD"]):
            w.writerow([name] + conf[i].tolist())

    print("best", best_path, flush=True)
    print("metrics", out / "metrics.json", flush=True)
    print("test bal acc", test_m["balanced_acc"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
