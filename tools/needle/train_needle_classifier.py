#!/usr/bin/env python3
"""Train the needle-acquisition classifier on cached echo-rectangle crops.

Input is `manifest_crops.csv` from `extract_needle_crops.py`. The split is
already leak-free at acquisition level, so this script never re-splits.

Design choices that this dataset forces:

- Sampling. Positives are concentrated in a handful of acquisitions (one alone
  holds a third of the training positives) and consecutive frames are near
  duplicates. A WeightedRandomSampler balances the two classes and, inside each
  class, damps the weight of the large acquisitions, so an epoch is not four
  sessions repeated over and over.
- Letterbox resize. Crops have very different aspect ratios; squashing them to a
  square would change needle angles, which are part of the signal. Crops are
  padded instead.
- Mild photometric jitter only. "Brighter than a normal scan" is genuine signal
  here (the acquisitions are done in water), so aggressive brightness/contrast
  augmentation would erase the very feature the model needs.

Metrics are reported per image, per leaf directory (the operational unit: one
acquisition sub-folder is needle or not) and per vendor, plus a two-threshold
accepted/review/reject policy calibrated on validation.

Example:
  python3 tools/needle/train_needle_classifier.py \
    --crops-dir artifacts/90_needle_dataset/v1/crops \
    --output-dir artifacts/91_needle_models/convnext_v1 \
    --arch convnext_tiny --image-size 352 --epochs 18 --device mps
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class CropRow:
    crop_path: Path
    label: int
    split: str
    group: str
    leaf_dir: str
    vendor: str
    rel_path: str


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


def load_rows(crops_dir: Path) -> List[CropRow]:
    manifest = crops_dir / "manifest_crops.csv"
    rows: List[CropRow] = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split = (row.get("split") or "").strip().lower()
            if split not in {"train", "val", "test"}:
                continue
            rows.append(
                CropRow(
                    crop_path=crops_dir / row["crop_path"],
                    label=int(row["label"]),
                    split=split,
                    group=row["group"],
                    leaf_dir=row["leaf_dir"],
                    vendor=(row.get("vendor_pred") or row.get("vendor_hint") or "UNKNOWN"),
                    rel_path=row["rel_path"],
                )
            )
    return rows


def letterbox(image: Image.Image, size: int) -> Image.Image:
    """Resize preserving aspect ratio and pad to a square with black."""
    w, h = image.size
    scale = size / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


class CropDataset(Dataset):
    def __init__(self, rows: Sequence[CropRow], image_size: int, train: bool, jitter: float) -> None:
        self.rows = list(rows)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.jitter = float(jitter)
        self.color = (
            transforms.ColorJitter(brightness=jitter, contrast=jitter)
            if train and jitter > 0
            else None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        try:
            with Image.open(row.crop_path) as img:
                image = img.convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
            if random.random() < 0.5:
                # mild zoom/shift: never below 70% of the area, so a thin needle
                # is unlikely to be cropped out entirely
                scale = random.uniform(0.80, 1.0)
                w, h = image.size
                cw, ch = int(w * scale), int(h * scale)
                x0 = random.randint(0, max(0, w - cw))
                y0 = random.randint(0, max(0, h - ch))
                image = image.crop((x0, y0, x0 + cw, y0 + ch))
            if random.random() < 0.3:
                angle = random.uniform(-6.0, 6.0)
                image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
            if self.color is not None:
                image = self.color(image)

        image = letterbox(image, self.image_size)
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return tensor, torch.tensor(float(row.label)), index


def build_model(arch: str, dropout: float) -> nn.Module:
    from torchvision import models

    arch = arch.lower()
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, 1))
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, 1))
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(model.classifier[1].in_features, 1)
        )
    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, 1)
    else:
        raise SystemExit(f"arch non supportata: {arch}")
    return model


def sample_weights(rows: Sequence[CropRow], group_damping: float) -> List[float]:
    """Balance the classes, and damp acquisitions that dominate their class."""
    class_count = Counter(r.label for r in rows)
    group_count = Counter((r.group, r.label) for r in rows)
    weights = []
    for row in rows:
        w = 1.0 / max(1, class_count[row.label])
        w /= max(1.0, group_count[(row.group, row.label)]) ** group_damping
        weights.append(w)
    total = sum(weights) or 1.0
    return [w / total for w in weights]


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cum_tp = np.cumsum(sorted_labels)
    precision = cum_tp / np.arange(1, labels.size + 1)
    return float((precision * sorted_labels).sum() / labels.sum())


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    fn = labels.sum() - tp
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0)
    best = int(np.argmax(f1))
    return float(sorted_scores[best]), float(f1[best])


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(1, labels.size),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "auc": roc_auc(labels, scores),
        "ap": average_precision(labels, scores),
        "n": int(labels.size),
        "n_pos": int(labels.sum()),
    }


def review_policy(labels: np.ndarray, scores: np.ndarray, target_precision: float,
                  target_recall: float) -> Dict[str, float]:
    """Two thresholds: accept above hi, reject below lo, review in between.

    hi is the lowest score whose precision on the accepted side still meets
    target_precision; lo is the highest score that still keeps target_recall.
    """
    order = np.argsort(-scores, kind="mergesort")
    sl = labels[order]
    ss = scores[order]
    tp = np.cumsum(sl)
    fp = np.cumsum(1 - sl)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / max(1, labels.sum())

    hi_idx = np.flatnonzero(precision >= target_precision)
    hi = float(ss[hi_idx[-1]]) if hi_idx.size else float(ss[0])
    lo_idx = np.flatnonzero(recall >= target_recall)
    lo = float(ss[lo_idx[0]]) if lo_idx.size else float(ss[-1])
    if lo > hi:
        lo, hi = hi, lo
    review = float(((scores > lo) & (scores < hi)).mean())
    return {"accept_above": hi, "reject_below": lo, "review_rate": review}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, tta: bool) -> np.ndarray:
    model.eval()
    out = np.zeros(len(loader.dataset), dtype=np.float64)
    for images, _, indices in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images).squeeze(1).float()
        probs = torch.sigmoid(logits)
        if tta:
            probs = 0.5 * (probs + torch.sigmoid(model(torch.flip(images, dims=[3])).squeeze(1).float()))
        out[indices.numpy()] = probs.detach().cpu().numpy()
    return out


def grouped_metrics(rows: Sequence[CropRow], scores: np.ndarray, threshold: float,
                    key) -> Dict[str, float]:
    """Aggregate per key (mean score over the images of the unit)."""
    bucket_scores: Dict[str, List[float]] = defaultdict(list)
    bucket_label: Dict[str, int] = {}
    for row, score in zip(rows, scores):
        k = key(row)
        bucket_scores[k].append(float(score))
        bucket_label[k] = max(bucket_label.get(k, 0), row.label)
    keys = sorted(bucket_scores)
    labels = np.array([bucket_label[k] for k in keys])
    means = np.array([float(np.mean(bucket_scores[k])) for k in keys])
    return binary_metrics(labels, means, threshold)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--crops-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arch", type=str, default="convnext_tiny")
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--jitter", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--group-damping", type=float, default=0.5,
                        help="0 = plain class balance, 1 = every acquisition equally likely.")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dataloader-timeout", type=float, default=300.0,
                        help="Seconds to wait for a worker batch before failing loudly.")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-per-epoch", type=int, default=0,
                        help="0 = one pass over the training rows.")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--val-subsample", type=int, default=0,
                        help="Images used for the per-epoch validation pass (0 = all). "
                             "The final evaluation always uses the full split.")
    parser.add_argument("--target-precision", type=float, default=0.98)
    parser.add_argument("--target-recall", type=float, default=0.98)
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device or None)
    crops_dir = args.crops_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(crops_dir)
    by_split: Dict[str, List[CropRow]] = defaultdict(list)
    for row in rows:
        by_split[row.split].append(row)
    print(
        "dataset: "
        + "  ".join(
            f"{s}={len(by_split[s])} (pos {sum(r.label for r in by_split[s])})"
            for s in ("train", "val", "test")
        )
    )

    train_rows = by_split["train"]
    weights = sample_weights(train_rows, args.group_damping)
    steps = args.steps_per_epoch or len(train_rows)
    sampler = WeightedRandomSampler(weights, num_samples=steps, replacement=True)

    # persistent_workers is deliberately off and a timeout is set: on macOS a dead
    # worker otherwise leaves the main process blocked on the queue forever
    # (seen here as an epoch that never ends while the process burns no CPU).
    loader_kwargs: Dict[str, object] = {"num_workers": args.num_workers}
    if args.num_workers > 0:
        loader_kwargs["timeout"] = args.dataloader_timeout
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        CropDataset(train_rows, args.image_size, train=True, jitter=args.jitter),
        batch_size=args.batch_size, sampler=sampler, drop_last=True, **loader_kwargs,
    )
    eval_loaders = {
        split: DataLoader(
            CropDataset(by_split[split], args.image_size, train=False, jitter=0.0),
            batch_size=args.batch_size, shuffle=False, **loader_kwargs,
        )
        for split in ("val", "test")
        if by_split[split]
    }

    # Epoch selection can run on a stratified slice of validation; only the final
    # report needs the whole split. On a busy machine this is the difference
    # between a run that finishes and one that does not.
    epoch_val_rows = by_split["val"]
    if args.val_subsample and args.val_subsample < len(epoch_val_rows):
        rng = random.Random(args.seed)
        pos = [r for r in epoch_val_rows if r.label == 1]
        neg = [r for r in epoch_val_rows if r.label == 0]
        share = args.val_subsample / len(epoch_val_rows)
        epoch_val_rows = (
            rng.sample(pos, max(1, int(len(pos) * share)))
            + rng.sample(neg, max(1, int(len(neg) * share)))
        )
        print(f"validazione per epoca su {len(epoch_val_rows)} immagini "
              f"({sum(r.label for r in epoch_val_rows)} positive)")
    epoch_val_loader = DataLoader(
        CropDataset(epoch_val_rows, args.image_size, train=False, jitter=0.0),
        batch_size=args.batch_size, shuffle=False, **loader_kwargs,
    )
    epoch_val_labels = np.array([r.label for r in epoch_val_rows])

    model = build_model(args.arch, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, steps // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * max(0, args.warmup_epochs)

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    history: List[Dict[str, object]] = []
    best_ap = -1.0
    best_epoch = -1
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        running = 0.0
        seen = 0
        nonfinite_batches = 0
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if args.label_smoothing > 0:
                targets = targets * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
            logits = model(images).squeeze(1).float()
            loss = F.binary_cross_entropy_with_logits(logits, targets)

            # The MPS backend occasionally returns a non-finite loss or gradient
            # under memory pressure. Applying such a step turns every weight into
            # NaN and silently wastes the whole run, so the batch is dropped
            # instead. Frequency is reported per epoch.
            if not torch.isfinite(loss):
                nonfinite_batches += 1
                scheduler.step()
                global_step += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(grad_norm):
                nonfinite_batches += 1
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                continue

            optimizer.step()
            scheduler.step()
            global_step += 1
            running += float(loss.item()) * images.size(0)
            seen += images.size(0)

        # no TTA while selecting the epoch: it doubles the cost of every
        # validation pass and only shifts the scores slightly. The final
        # evaluation below re-runs validation with TTA.
        val_scores = predict(model, epoch_val_loader, device, tta=False)
        val_labels = epoch_val_labels
        val_ap = average_precision(val_labels, val_scores)
        val_auc = roc_auc(val_labels, val_scores)
        thr, f1 = best_f1_threshold(val_labels, val_scores)

        entry = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "val_ap": val_ap,
            "val_auc": val_auc,
            "val_best_f1": f1,
            "val_best_threshold": thr,
            "lr": optimizer.param_groups[0]["lr"],
            "nonfinite_batches": nonfinite_batches,
            "seconds": time.time() - started,
        }
        history.append(entry)
        print(
            f"epoch {epoch:2d}/{args.epochs}  loss {entry['train_loss']:.4f}  "
            f"val AP {val_ap:.4f}  AUC {val_auc:.4f}  bestF1 {f1:.4f} @ {thr:.3f}  "
            f"({entry['seconds']:.0f}s"
            + (f", {nonfinite_batches} batch NaN scartati)" if nonfinite_batches else ")"),
            flush=True,
        )

        if math.isfinite(val_ap) and val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "arch": args.arch,
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "val_ap": val_ap,
                    "args": vars(args) | {"crops_dir": str(crops_dir), "output_dir": str(out_dir)},
                },
                out_dir / "best_model.pt",
            )

    # ---- final evaluation with the best checkpoint ---------------------------
    checkpoint = torch.load(out_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    val_rows = by_split["val"]
    val_scores = predict(model, eval_loaders["val"], device, tta=not args.no_tta)
    val_labels = np.array([r.label for r in val_rows])
    threshold, _ = best_f1_threshold(val_labels, val_scores)
    policy = review_policy(val_labels, val_scores, args.target_precision, args.target_recall)

    report: Dict[str, object] = {
        "arch": args.arch,
        "image_size": args.image_size,
        "best_epoch": best_epoch,
        "best_val_ap": best_ap,
        "threshold_from_val": threshold,
        "review_policy_from_val": policy,
        "history": history,
        "splits": {},
    }

    for split, loader in eval_loaders.items():
        split_rows = by_split[split]
        scores = val_scores if split == "val" else predict(model, loader, device, tta=not args.no_tta)
        labels = np.array([r.label for r in split_rows])
        per_image = binary_metrics(labels, scores, threshold)
        per_leaf = grouped_metrics(split_rows, scores, threshold, key=lambda r: f"{r.group}||{r.leaf_dir}")
        per_vendor: Dict[str, Dict[str, float]] = {}
        vendors = sorted({r.vendor for r in split_rows})
        for vendor in vendors:
            idx = np.array([i for i, r in enumerate(split_rows) if r.vendor == vendor])
            if idx.size < 10:
                continue
            per_vendor[vendor] = binary_metrics(labels[idx], scores[idx], threshold)
        report["splits"][split] = {
            "per_image": per_image,
            "per_leaf_dir": per_leaf,
            "per_vendor": per_vendor,
        }

        with (out_dir / f"predictions_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rel_path", "group", "leaf_dir", "vendor", "label", "score"])
            for row, score in zip(split_rows, scores):
                writer.writerow([row.rel_path, row.group, row.leaf_dir, row.vendor, row.label, f"{score:.6f}"])

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nbest epoch {best_epoch} (val AP {best_ap:.4f}), soglia da val {threshold:.3f}")
    for split in ("val", "test"):
        if split not in report["splits"]:
            continue
        pi = report["splits"][split]["per_image"]
        pl = report["splits"][split]["per_leaf_dir"]
        print(
            f"{split:5s} immagine: AP {pi['ap']:.4f} AUC {pi['auc']:.4f} "
            f"P {pi['precision']:.3f} R {pi['recall']:.3f} F1 {pi['f1']:.3f} acc {pi['accuracy']:.4f}"
        )
        print(
            f"      cartella: AP {pl['ap']:.4f} AUC {pl['auc']:.4f} "
            f"P {pl['precision']:.3f} R {pl['recall']:.3f} F1 {pl['f1']:.3f} (n={pl['n']})"
        )
    print(f"report: {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
