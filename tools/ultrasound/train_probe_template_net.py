#!/usr/bin/env python3
"""Train the probe-template detector (line #14 RECT_NAME_PROBE).

The target box is tiny (median 74x25 px on 1920x1080, ~0.09% of the frame), so a
pooled bbox regressor cannot place it precisely. This trainer uses a CenterNet-style
heatmap head at stride 4: center heatmap + sub-cell offset + size, which keeps
spatial resolution and yields a usable confidence score (heatmap peak).

Frames without the probe label are kept as negatives (empty heatmap) so the model
learns to abstain instead of always firing.

No left-right / up-down augmentation here: the target is rendered text, mirrored
text never occurs in a real UI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

STRIDE = 4


@dataclass(frozen=True)
class SampleRow:
    image_path: Path
    split: str
    folder: str
    manufacturer: str
    group_id: str
    width: int
    height: int
    has_template: int
    x1: float
    y1: float
    x2: float
    y2: float
    fss_x1: float
    fss_y1: float
    fss_x2: float
    fss_y2: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(prefer: Optional[str]) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_manifest(
    manifest: Path,
    splits: Sequence[str],
    gt_status: Sequence[str],
    vendors: Optional[Sequence[str]],
    max_per_folder: int,
    keep_negatives: bool,
    seed: int,
) -> List[SampleRow]:
    by_folder: Dict[str, List[SampleRow]] = defaultdict(list)
    with manifest.open(encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            if raw["split"] not in splits:
                continue
            if gt_status and raw["gt_status"] not in gt_status:
                continue
            if vendors and raw["manufacturer"] not in vendors:
                continue
            positive = raw["has_template"] == "1"
            if not positive and not keep_negatives:
                continue
            row = SampleRow(
                image_path=Path(raw["image_path"]),
                split=raw["split"],
                folder=raw["dataset_folder"],
                manufacturer=raw["manufacturer"],
                group_id=raw["group_id"],
                width=int(raw["image_width"]),
                height=int(raw["image_height"]),
                has_template=1 if positive else 0,
                x1=float(raw["bbox_xmin"]) if positive else 0.0,
                y1=float(raw["bbox_ymin"]) if positive else 0.0,
                x2=float(raw["bbox_xmax"]) if positive else 0.0,
                y2=float(raw["bbox_ymax"]) if positive else 0.0,
                fss_x1=float(raw["fss_left"] or 0),
                fss_y1=float(raw["fss_top"] or 0),
                fss_x2=float(raw["fss_right"] or 0),
                fss_y2=float(raw["fss_bottom"] or 0),
            )
            by_folder[row.folder].append(row)

    rng = random.Random(seed)
    rows: List[SampleRow] = []
    for folder in sorted(by_folder):
        group = by_folder[folder]
        if 0 < max_per_folder < len(group):
            idx = np.linspace(0, len(group) - 1, max_per_folder).round().astype(int)
            group = [group[i] for i in sorted(set(idx.tolist()))]
        rows.extend(group)
    rng.shuffle(rows)
    return rows


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    a1, b1, c1 = 1.0, height + width, width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0.0))) / (2 * a1)
    a2, b2, c2 = 4.0, 2 * (height + width), (1 - min_overlap) * width * height
    r2 = (b2 - math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0.0))) / (2 * a2)
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (-b3 + math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0.0))) / (2 * a3)
    return max(0.0, min(r1, r2, r3))


def draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, radius: int) -> None:
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (2 * sigma * sigma))
    h, w = heatmap.shape
    left, right = min(cx, radius), min(w - cx, radius + 1)
    top, bottom = min(cy, radius), min(h - cy, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return
    masked = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    np.maximum(masked, kernel[radius - top:radius + bottom, radius - left:radius + right], out=masked)


class ProbeTemplateDataset(Dataset):
    def __init__(self, rows: Sequence[SampleRow], image_size: int, augment: bool) -> None:
        self.rows = list(rows)
        self.image_size = image_size
        self.augment = augment
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.out_size = image_size // STRIDE
        self.read_failures = 0

    def __len__(self) -> int:
        return len(self.rows)

    def _read(self, row: SampleRow):
        """The dataset lives on a USB volume: a read can fail once and succeed right after.
        Losing a whole run to one hiccup is not acceptable, so retry, then move on."""
        for attempt in range(3):
            try:
                with Image.open(row.image_path) as img:
                    image = img.convert("RGB")
                return image
            except (OSError, ValueError) as exc:
                if attempt == 2:
                    self.read_failures += 1
                    if self.read_failures <= 5:
                        print(f"  [lettura fallita] {row.image_path}: {exc}", flush=True)
                    return None
                time.sleep(0.2)
        return None

    def __getitem__(self, idx: int):  # type: ignore[override]
        row = self.rows[idx]
        image = self._read(row)
        if image is None:
            # The neighbour keeps the batch full with real data; only if that fails too
            # does the sample become an empty frame, which is a harmless negative.
            row = self.rows[(idx + 1) % len(self.rows)]
            image = self._read(row)
        if image is None:
            row = replace(row, has_template=0)
            image = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

        width, height = image.size
        scale = self.image_size / max(width, height)
        new_w, new_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        canvas.paste(image, (0, 0))

        tensor = transforms.functional.to_tensor(canvas)
        if self.augment:
            # Photometric only: geometry is the signal, mirrored text never occurs.
            if random.random() < 0.5:
                tensor = torch.clamp(tensor * random.uniform(0.75, 1.25), 0.0, 1.0)
            if random.random() < 0.3:
                tensor = torch.clamp(tensor + torch.randn_like(tensor) * 0.02, 0.0, 1.0)
        tensor = self.normalize(tensor)

        heatmap = np.zeros((self.out_size, self.out_size), dtype=np.float32)
        offset = np.zeros((2, self.out_size, self.out_size), dtype=np.float32)
        size = np.zeros((2, self.out_size, self.out_size), dtype=np.float32)
        mask = np.zeros((self.out_size, self.out_size), dtype=np.float32)

        if row.has_template:
            bx1, by1 = row.x1 * scale, row.y1 * scale
            bx2, by2 = row.x2 * scale, row.y2 * scale
            bw, bh = max(bx2 - bx1, 1e-3), max(by2 - by1, 1e-3)
            cx, cy = (bx1 + bx2) / 2.0 / STRIDE, (by1 + by2) / 2.0 / STRIDE
            ix, iy = int(cx), int(cy)
            if 0 <= ix < self.out_size and 0 <= iy < self.out_size:
                radius = max(1, int(gaussian_radius(bh / STRIDE, bw / STRIDE)))
                draw_gaussian(heatmap, ix, iy, radius)
                offset[0, iy, ix] = cx - ix
                offset[1, iy, ix] = cy - iy
                size[0, iy, ix] = bw / STRIDE
                size[1, iy, ix] = bh / STRIDE
                mask[iy, ix] = 1.0

        meta = {
            "image_path": str(row.image_path),
            "folder": row.folder,
            "manufacturer": row.manufacturer,
            "width": row.width,
            "height": row.height,
            "scale": scale,
            "has_template": row.has_template,
            "gt": [row.x1, row.y1, row.x2, row.y2],
            "fss": [row.fss_x1, row.fss_y1, row.fss_x2, row.fss_y2],
        }
        return (
            tensor,
            torch.from_numpy(heatmap),
            torch.from_numpy(offset),
            torch.from_numpy(size),
            torch.from_numpy(mask),
            meta,
        )


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    heat = torch.stack([b[1] for b in batch])
    off = torch.stack([b[2] for b in batch])
    size = torch.stack([b[3] for b in batch])
    mask = torch.stack([b[4] for b in batch])
    meta = [b[5] for b in batch]
    return images, heat, off, size, mask, meta


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(skip_ch, out_ch, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.conv(torch.cat([x, self.reduce(skip)], dim=1))


class ProbeTemplateNet(nn.Module):
    """ResNet18 + light FPN decoder to stride 4, with CenterNet heads."""

    def __init__(self, pretrained: bool = True, head_channels: int = 64) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)  # stride 4
        self.layer1, self.layer2 = net.layer1, net.layer2                      # 4, 8
        self.layer3, self.layer4 = net.layer3, net.layer4                      # 16, 32
        self.up1 = UpBlock(512, 256, 128)
        self.up2 = UpBlock(128, 128, 96)
        self.up3 = UpBlock(96, 64, head_channels)
        self.heat = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 1, 1),
        )
        self.offset = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 2, 1),
        )
        self.size = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 2, 1),
        )
        self.heat[-1].bias.data.fill_(-4.0)  # rare positives

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stem(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p = self.up1(c5, c4)
        p = self.up2(p, c3)
        p = self.up3(p, c2)
        return self.heat(p).squeeze(1), self.offset(p), self.size(p)


def focal_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.clamp(torch.sigmoid(pred_logits), 1e-4, 1 - 1e-4)
    pos = target.ge(0.999).float()
    neg = 1.0 - pos
    neg_weights = torch.pow(1.0 - target, 4)
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, 2) * pos
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg
    n_pos = pos.sum()
    if n_pos == 0:
        return neg_loss.sum()
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def decode_batch(
    heat_logits: torch.Tensor,
    offset: torch.Tensor,
    size: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scores, boxes_in_canvas_px) taking the heatmap peak of each sample."""
    heat = torch.sigmoid(heat_logits)
    b, h, w = heat.shape
    flat = heat.view(b, -1)
    scores, idx = flat.max(dim=1)
    ys = (idx // w).float()
    xs = (idx % w).float()
    off = offset.view(b, 2, -1).gather(2, idx.view(b, 1, 1).expand(b, 2, 1)).squeeze(2)
    siz = size.view(b, 2, -1).gather(2, idx.view(b, 1, 1).expand(b, 2, 1)).squeeze(2)
    cx = (xs + off[:, 0]) * STRIDE
    cy = (ys + off[:, 1]) * STRIDE
    bw = siz[:, 0].clamp(min=0) * STRIDE
    bh = siz[:, 1].clamp(min=0) * STRIDE
    boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1)
    return scores.detach().cpu().numpy(), boxes.detach().cpu().numpy()


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, device, optimizer=None, size_weight=0.1, offset_weight=1.0):
    train = optimizer is not None
    model.train(train)
    totals = Counter()
    n_batches = 0
    ious: List[float] = []
    with torch.set_grad_enabled(train):
        for images, heat, off, size, mask, meta in loader:
            images = images.to(device, non_blocking=True)
            heat = heat.to(device)
            off = off.to(device)
            size = size.to(device)
            mask = mask.to(device)

            heat_pred, off_pred, size_pred = model(images)
            loss_heat = focal_loss(heat_pred, heat)
            denom = mask.sum().clamp(min=1.0)
            m = mask.unsqueeze(1)
            loss_off = (F.l1_loss(off_pred * m, off * m, reduction="sum") / denom)
            loss_size = (F.l1_loss(size_pred * m, size * m, reduction="sum") / denom)
            loss = loss_heat + offset_weight * loss_off + size_weight * loss_size

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            totals["loss"] += float(loss.item())
            totals["heat"] += float(loss_heat.item())
            totals["off"] += float(loss_off.item())
            totals["size"] += float(loss_size.item())
            n_batches += 1

            if not train:
                scores, boxes = decode_batch(heat_pred, off_pred, size_pred)
                for i, info in enumerate(meta):
                    if not info["has_template"]:
                        continue
                    s = info["scale"]
                    pred = [boxes[i][0] / s, boxes[i][1] / s, boxes[i][2] / s, boxes[i][3] / s]
                    ious.append(iou_xyxy(pred, info["gt"]))
    out = {k: v / max(n_batches, 1) for k, v in totals.items()}
    out["iou"] = float(statistics.mean(ious)) if ious else 0.0
    out["iou_median"] = float(statistics.median(ious)) if ious else 0.0
    return out


@torch.no_grad()
def evaluate(model, loader, device, score_threshold: float) -> Dict[str, object]:
    model.eval()
    rows: List[Dict[str, object]] = []
    for images, heat, off, size, mask, meta in loader:
        images = images.to(device, non_blocking=True)
        heat_pred, off_pred, size_pred = model(images)
        scores, boxes = decode_batch(heat_pred, off_pred, size_pred)
        for i, info in enumerate(meta):
            s = info["scale"]
            pred = [float(boxes[i][j] / s) for j in range(4)]
            rows.append({
                "image_path": info["image_path"],
                "folder": info["folder"],
                "manufacturer": info["manufacturer"],
                "has_template": int(info["has_template"]),
                "score": float(scores[i]),
                "pred_x1": pred[0], "pred_y1": pred[1], "pred_x2": pred[2], "pred_y2": pred[3],
                "gt_x1": info["gt"][0], "gt_y1": info["gt"][1], "gt_x2": info["gt"][2], "gt_y2": info["gt"][3],
                "fss_x1": info["fss"][0], "fss_y1": info["fss"][1], "fss_x2": info["fss"][2], "fss_y2": info["fss"][3],
                "iou": iou_xyxy(pred, info["gt"]) if info["has_template"] else "",
            })

    positives = [r for r in rows if r["has_template"] == 1]
    negatives = [r for r in rows if r["has_template"] == 0]
    ious = [float(r["iou"]) for r in positives]

    # Folder level: median predicted box over confident frames (what the pipeline writes).
    per_folder: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in positives:
        per_folder[str(r["folder"])].append(r)
    folder_rows: List[Dict[str, object]] = []
    for folder, group in sorted(per_folder.items()):
        confident = [r for r in group if float(r["score"]) >= score_threshold] or group
        med = [statistics.median([float(r[k]) for r in confident]) for k in ("pred_x1", "pred_y1", "pred_x2", "pred_y2")]
        fss = [float(group[0][k]) for k in ("fss_x1", "fss_y1", "fss_x2", "fss_y2")]
        gt = [statistics.median([float(r[k]) for r in group]) for k in ("gt_x1", "gt_y1", "gt_x2", "gt_y2")]
        folder_rows.append({
            "folder": folder,
            "manufacturer": group[0]["manufacturer"],
            "n_frames": len(group),
            "n_confident": sum(1 for r in group if float(r["score"]) >= score_threshold),
            "median_score": statistics.median([float(r["score"]) for r in group]),
            "pred_x1": med[0], "pred_y1": med[1], "pred_x2": med[2], "pred_y2": med[3],
            "iou_vs_gt": iou_xyxy(med, gt),
            "iou_vs_fss": iou_xyxy(med, fss),
            "max_coord_err_px": max(abs(med[i] - fss[i]) for i in range(4)) if any(fss) else "",
        })

    def frac(values: List[float], thr: float) -> float:
        return sum(1 for v in values if v >= thr) / len(values) if values else 0.0

    folder_ious = [float(r["iou_vs_fss"]) for r in folder_rows]
    coord_errs = [float(r["max_coord_err_px"]) for r in folder_rows if r["max_coord_err_px"] != ""]
    per_vendor: Dict[str, Dict[str, float]] = {}
    for vendor in sorted({str(r["manufacturer"]) for r in folder_rows}):
        sub = [r for r in folder_rows if r["manufacturer"] == vendor]
        sub_iou = [float(r["iou_vs_fss"]) for r in sub]
        per_vendor[vendor] = {
            "folders": len(sub),
            "iou_median": float(statistics.median(sub_iou)) if sub_iou else 0.0,
            "iou_ge_0.5": frac(sub_iou, 0.5),
            "iou_ge_0.75": frac(sub_iou, 0.75),
        }

    summary = {
        "images": len(rows),
        "positives": len(positives),
        "negatives": len(negatives),
        "image_iou_mean": float(statistics.mean(ious)) if ious else 0.0,
        "image_iou_median": float(statistics.median(ious)) if ious else 0.0,
        "image_iou_ge_0.5": frac(ious, 0.5),
        "image_iou_ge_0.75": frac(ious, 0.75),
        "folders": len(folder_rows),
        "folder_iou_median": float(statistics.median(folder_ious)) if folder_ious else 0.0,
        "folder_iou_ge_0.5": frac(folder_ious, 0.5),
        "folder_iou_ge_0.75": frac(folder_ious, 0.75),
        "folder_max_coord_err_le_2px": sum(1 for e in coord_errs if e <= 2) / len(coord_errs) if coord_errs else 0.0,
        "folder_max_coord_err_le_5px": sum(1 for e in coord_errs if e <= 5) / len(coord_errs) if coord_errs else 0.0,
        "folder_max_coord_err_le_10px": sum(1 for e in coord_errs if e <= 10) / len(coord_errs) if coord_errs else 0.0,
        "folder_max_coord_err_median": float(statistics.median(coord_errs)) if coord_errs else 0.0,
        "negative_score_median": float(statistics.median([float(r["score"]) for r in negatives])) if negatives else "",
        "positive_score_median": float(statistics.median([float(r["score"]) for r in positives])) if positives else "",
        "score_threshold": score_threshold,
        "per_vendor": per_vendor,
    }
    return {"summary": summary, "image_rows": rows, "folder_rows": folder_rows}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--vendors", type=str, default="", help="comma separated: train only on these vendors")
    p.add_argument("--gt-status", type=str, default="ok", help="comma separated manifest gt_status values to keep")
    p.add_argument("--max-images-per-folder", type=int, default=60, help="0 = all frames in the manifest")
    p.add_argument("--max-eval-images-per-folder", type=int, default=30)
    p.add_argument("--drop-negatives", action="store_true", help="train only on frames showing the label")
    p.add_argument("--score-threshold", type=float, default=0.30)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--eval-split", type=str, default="test")
    return p


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = choose_device(args.device or None)
    vendors = [v.strip() for v in args.vendors.split(",") if v.strip()] or None
    gt_status = [s.strip() for s in args.gt_status.split(",") if s.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_manifest(args.manifest, ["train"], gt_status, vendors,
                               args.max_images_per_folder, not args.drop_negatives, args.seed)
    val_rows = load_manifest(args.manifest, ["val"], gt_status, vendors,
                             args.max_eval_images_per_folder, not args.drop_negatives, args.seed)
    test_rows = load_manifest(args.manifest, [args.eval_split], gt_status, vendors,
                              args.max_eval_images_per_folder, not args.drop_negatives, args.seed)
    print(f"Device: {device}  | train {len(train_rows)}  val {len(val_rows)}  {args.eval_split} {len(test_rows)}")
    print(f"Vendor filter: {vendors or 'ALL'}  | folders train={len({r.folder for r in train_rows})}"
          f" val={len({r.folder for r in val_rows})} test={len({r.folder for r in test_rows})}")
    if not train_rows or not val_rows:
        print("ERROR: empty train/val set", flush=True)
        return 2

    def loader(rows, shuffle, augment):
        ds = ProbeTemplateDataset(rows, args.image_size, augment)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, collate_fn=collate,
                          pin_memory=False, persistent_workers=args.num_workers > 0)

    train_loader = loader(train_rows, True, True)
    val_loader = loader(val_rows, False, False)
    test_loader = loader(test_rows, False, False)

    model = ProbeTemplateNet(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_iou, best_epoch = -1.0, -1
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, optimizer)
        va = run_epoch(model, val_loader, device, None)
        scheduler.step()
        dt = time.time() - t0
        history.append({"epoch": epoch, "train": tr, "val": va, "seconds": round(dt, 1)})
        print(f"epoch {epoch:2d}/{args.epochs}  train_loss {tr['loss']:.4f}  val_loss {va['loss']:.4f}  "
              f"val_IoU {va['iou']:.4f} (med {va['iou_median']:.4f})  {dt:.0f}s", flush=True)
        if va["iou"] > best_iou:
            best_iou, best_epoch = va["iou"], epoch
            torch.save({
                "model_state": model.state_dict(),
                "image_size": args.image_size,
                "stride": STRIDE,
                "vendors": vendors or "ALL",
                "epoch": epoch,
                "val_iou": va["iou"],
                "arch": "resnet18_fpn_centernet",
            }, args.output_dir / "best_model.pt")

    ckpt = torch.load(args.output_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    result = evaluate(model, test_loader, device, args.score_threshold)

    with (args.output_dir / f"predictions_{args.eval_split}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result["image_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(result["image_rows"])
    with (args.output_dir / f"folder_predictions_{args.eval_split}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result["folder_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(result["folder_rows"])

    metrics = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_iou": best_iou,
        "train_images": len(train_rows),
        "val_images": len(val_rows),
        "test_images": len(test_rows),
        "train_folders": len({r.folder for r in train_rows}),
        "test_folders": len({r.folder for r in test_rows}),
        "history": history,
        f"{args.eval_split}_summary": result["summary"],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    s = result["summary"]
    print(f"\n== {args.eval_split} ==")
    print(f"immagini: IoU medio {s['image_iou_mean']:.4f}  mediano {s['image_iou_median']:.4f}  "
          f">0.5 {s['image_iou_ge_0.5']*100:.1f}%  >0.75 {s['image_iou_ge_0.75']*100:.1f}%")
    print(f"cartelle: IoU mediano {s['folder_iou_median']:.4f}  >0.5 {s['folder_iou_ge_0.5']*100:.1f}%  "
          f">0.75 {s['folder_iou_ge_0.75']*100:.1f}%")
    print(f"errore max coord: mediana {s['folder_max_coord_err_median']:.1f}px  "
          f"<=5px {s['folder_max_coord_err_le_5px']*100:.1f}%  <=10px {s['folder_max_coord_err_le_10px']*100:.1f}%")
    print(f"score mediano positivi {s['positive_score_median']}  negativi {s['negative_score_median']}")
    for vendor, stats in s["per_vendor"].items():
        print(f"  {vendor:10s} cartelle {stats['folders']:3d}  IoU med {stats['iou_median']:.3f}  "
              f">0.5 {stats['iou_ge_0.5']*100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
