#!/usr/bin/env python3
"""Train a small full-frame model to predict the RECT_DEPTH bbox.

Input is the CSV produced by:

  python3 tools/depth/rect_depth_hybrid.py manifest ...

This trainer is intentionally modest: it filters malformed legacy rows, splits
by configuration/fss path to avoid leakage, and trains a small coordinate-aware
CNN that regresses the depth rectangle in normalized image coordinates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


VALID_MATCH_METHODS = {"0", "5", "6", "7"}
VALID_B = {"0", "1"}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    split: str
    config_folder: str
    fss_path: str
    setup_id: str
    depth_index0: int
    depth_mm: float
    flip_state: str
    image_path: str
    image_w: int
    image_h: int
    x1_norm: float
    y1_norm: float
    x2_norm: float
    y2_norm: float


def _f(text: str) -> float:
    return float(str(text).strip())


def _i(text: str) -> int:
    return int(float(str(text).strip()))


def _safe_slug(text: str, max_len: int = 120) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    return out[:max_len].strip("_") or "item"


def _hash_to_unit(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _assign_split(group_key: str, train_ratio: float, val_ratio: float) -> str:
    x = _hash_to_unit(group_key)
    if x < train_ratio:
        return "train"
    if x < train_ratio + val_ratio:
        return "val"
    return "test"


def _clip_box(left: float, top: float, right: float, bottom: float, w: int, h: int) -> Tuple[float, float, float, float]:
    x1 = max(0.0, min(float(w - 1), min(left, right)))
    y1 = max(0.0, min(float(h - 1), min(top, bottom)))
    x2 = max(x1 + 1.0, min(float(w), max(left, right) + 1.0))
    y2 = max(y1 + 1.0, min(float(h), max(top, bottom) + 1.0))
    return x1, y1, x2, y2


def _row_is_clean(row: Dict[str, str], allow_fallback: bool) -> bool:
    if row.get("b", "") not in VALID_B:
        return False
    if row.get("match_method", "") not in VALID_MATCH_METHODS:
        return False
    if not row.get("source_image"):
        return False
    if not allow_fallback and str(row.get("source_strategy", "")).startswith("fallback:"):
        return False
    try:
        w = _i(row.get("width", "0"))
        h = _i(row.get("height", "0"))
    except Exception:
        return False
    return 4 <= w <= 260 and 4 <= h <= 160


def load_samples(
    manifest: Path,
    limit: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    allow_fallback: bool,
    include_flips: Sequence[str],
    max_per_config: int,
) -> Tuple[List[Sample], Dict[str, object]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    skipped = Counter()
    include_flip_set = {f.strip().lower() for f in include_flips if f.strip()}

    with manifest.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if include_flip_set and row.get("flip_state", "").lower() not in include_flip_set:
                skipped["flip_filter"] += 1
                continue
            if not _row_is_clean(row, allow_fallback=allow_fallback):
                skipped["not_clean"] += 1
                continue
            rows.append(row)

    by_config: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_config[row["config_folder"]].append(row)

    sampled_rows: List[Dict[str, str]] = []
    for cfg, cfg_rows in sorted(by_config.items()):
        cfg_rows = list(cfg_rows)
        rng.shuffle(cfg_rows)
        if max_per_config > 0:
            cfg_rows = cfg_rows[:max_per_config]
        sampled_rows.extend(cfg_rows)

    rng.shuffle(sampled_rows)
    if limit > 0:
        sampled_rows = sampled_rows[:limit]

    samples: List[Sample] = []
    split_by_fss = {
        fss: _assign_split(fss, train_ratio=train_ratio, val_ratio=val_ratio)
        for fss in sorted({r["fss_path"] for r in sampled_rows})
    }

    for idx, row in enumerate(sampled_rows):
        image_path = Path(row["source_image"])
        try:
            with Image.open(image_path) as im:
                iw, ih = im.size
        except Exception:
            skipped["image_open_failed"] += 1
            continue
        if iw <= 2 or ih <= 2:
            skipped["bad_image_size"] += 1
            continue

        try:
            left = _f(row["left"])
            top = _f(row["top"])
            right = _f(row["right"])
            bottom = _f(row["bottom"])
            depth_index0 = _i(row["depth_index0"])
            depth_mm = _f(row["depth_mm"])
        except Exception:
            skipped["bad_numeric"] += 1
            continue

        x1, y1, x2, y2 = _clip_box(left, top, right, bottom, iw, ih)
        sid = f"{row['setup_id']}_{depth_index0}_{row['flip_state']}_{idx}"
        samples.append(
            Sample(
                sample_id=sid,
                split=split_by_fss[row["fss_path"]],
                config_folder=row["config_folder"],
                fss_path=row["fss_path"],
                setup_id=row["setup_id"],
                depth_index0=depth_index0,
                depth_mm=depth_mm,
                flip_state=row["flip_state"],
                image_path=image_path.as_posix(),
                image_w=iw,
                image_h=ih,
                x1_norm=x1 / float(iw),
                y1_norm=y1 / float(ih),
                x2_norm=x2 / float(iw),
                y2_norm=y2 / float(ih),
            )
        )

    meta = {
        "rows_input_clean": len(rows),
        "rows_sampled_before_image_check": len(sampled_rows),
        "samples_loaded": len(samples),
        "skipped": dict(skipped),
        "configs": len({s.config_folder for s in samples}),
        "fss_files": len({s.fss_path for s in samples}),
        "splits": Counter(s.split for s in samples),
        "flip_states": Counter(s.flip_state for s in samples),
    }
    return samples, meta


class RectDepthBBoxDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        image_size: Tuple[int, int],
        train: bool,
    ) -> None:
        self.samples = list(samples)
        self.train = bool(train)
        self.out_w, self.out_h = int(image_size[0]), int(image_size[1])
        self.resize = transforms.Resize((self.out_h, self.out_w), interpolation=transforms.InterpolationMode.BILINEAR)
        self.jitter = transforms.ColorJitter(brightness=0.18, contrast=0.22, saturation=0.05, hue=0.01)
        self.blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.7))
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        try:
            img = Image.open(s.image_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (max(8, s.image_w), max(8, s.image_h)), color=(0, 0, 0))
        y = np.array([s.x1_norm, s.y1_norm, s.x2_norm, s.y2_norm], dtype=np.float32)
        img = self.resize(img)
        if self.train:
            if random.random() < 0.85:
                img = self.jitter(img)
            if random.random() < 0.10:
                img = self.blur(img)
        x = self.norm(TF.to_tensor(img))
        return x, torch.tensor(y, dtype=torch.float32), idx


class SmallBBoxRegressor(nn.Module):
    def __init__(self, use_coord_channels: bool = True) -> None:
        super().__init__()
        self.use_coord_channels = bool(use_coord_channels)
        in_ch = 3 + (2 if self.use_coord_channels else 0)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(320),
            nn.SiLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(320, 96),
            nn.SiLU(inplace=True),
            nn.Dropout(0.12),
            nn.Linear(96, 4),
        )

    def _coord_maps(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_coord_channels:
            return x
        b, _, h, w = x.shape
        xs = torch.linspace(-1.0, 1.0, steps=w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        ys = torch.linspace(-1.0, 1.0, steps=h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        return torch.cat([x, xs, ys], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.backbone(self._coord_maps(x))))


def pick_device(requested: str) -> torch.device:
    req = (requested or "auto").strip().lower()
    if req == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if req == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if req == "cpu":
        return torch.device("cpu")
    # Keep auto conservative: this small model is often used for smoke tests
    # from external volumes, and MPS showed occasional non-finite loss here.
    return torch.device("cpu")


def bbox_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    per = F.smooth_l1_loss(pred, target, beta=0.02, reduction="none")
    weights = pred.new_tensor([1.0, 1.4, 1.0, 1.4]).view(1, 4)
    base = (per * weights).mean()
    order_penalty = (torch.relu(pred[:, 0] - pred[:, 2]) + torch.relu(pred[:, 1] - pred[:, 3])).mean()
    return base + 0.15 * order_penalty


def box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def compute_metrics(samples: Sequence[Sample], preds_norm: np.ndarray) -> Dict[str, float]:
    x_err, y_err, ious = [], [], []
    for s, p in zip(samples, preds_norm):
        px1, py1, px2, py2 = float(min(p[0], p[2])), float(min(p[1], p[3])), float(max(p[0], p[2])), float(max(p[1], p[3]))
        gx1, gy1, gx2, gy2 = s.x1_norm, s.y1_norm, s.x2_norm, s.y2_norm
        x_err.append(0.5 * (abs(px1 - gx1) + abs(px2 - gx2)) * s.image_w)
        y_err.append(0.5 * (abs(py1 - gy1) + abs(py2 - gy2)) * s.image_h)
        ious.append(
            box_iou(
                (px1 * s.image_w, py1 * s.image_h, px2 * s.image_w, py2 * s.image_h),
                (gx1 * s.image_w, gy1 * s.image_h, gx2 * s.image_w, gy2 * s.image_h),
            )
        )
    return {
        "rows": float(len(samples)),
        "mae_x_px": float(np.mean(x_err)) if x_err else float("nan"),
        "mae_y_px": float(np.mean(y_err)) if y_err else float("nan"),
        "iou_mean": float(np.mean(ious)) if ious else float("nan"),
        "score_xy": float(np.mean(x_err) + np.mean(y_err)) if x_err and y_err else float("inf"),
    }


@torch.no_grad()
def infer_dataset(model: nn.Module, loader: DataLoader, samples: Sequence[Sample], device: torch.device) -> Tuple[np.ndarray, Dict[str, float]]:
    model.eval()
    preds = np.zeros((len(samples), 4), dtype=np.float32)
    for xb, _yb, idxs in loader:
        xb = xb.to(device, non_blocking=True)
        out = model(xb).detach().cpu().numpy().astype(np.float32)
        for j, sample_idx in enumerate(idxs.numpy().tolist()):
            preds[sample_idx] = out[j]
    return preds, compute_metrics(samples, preds)


def save_predictions(path: Path, samples: Sequence[Sample], preds: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "config_folder",
        "setup_id",
        "depth_index0",
        "depth_mm",
        "flip_state",
        "image_path",
        "x1_gt",
        "y1_gt",
        "x2_gt",
        "y2_gt",
        "x1_pred",
        "y1_pred",
        "x2_pred",
        "y2_pred",
        "x_err_px",
        "y_err_px",
        "iou",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames)
        wr.writeheader()
        for s, p in zip(samples, preds):
            x1p, y1p, x2p, y2p = float(min(p[0], p[2])), float(min(p[1], p[3])), float(max(p[0], p[2])), float(max(p[1], p[3]))
            gt = (s.x1_norm * s.image_w, s.y1_norm * s.image_h, s.x2_norm * s.image_w, s.y2_norm * s.image_h)
            pr = (x1p * s.image_w, y1p * s.image_h, x2p * s.image_w, y2p * s.image_h)
            wr.writerow(
                {
                    "sample_id": s.sample_id,
                    "split": s.split,
                    "config_folder": s.config_folder,
                    "setup_id": s.setup_id,
                    "depth_index0": s.depth_index0,
                    "depth_mm": f"{s.depth_mm:.3f}",
                    "flip_state": s.flip_state,
                    "image_path": s.image_path,
                    "x1_gt": f"{gt[0]:.2f}",
                    "y1_gt": f"{gt[1]:.2f}",
                    "x2_gt": f"{gt[2]:.2f}",
                    "y2_gt": f"{gt[3]:.2f}",
                    "x1_pred": f"{pr[0]:.2f}",
                    "y1_pred": f"{pr[1]:.2f}",
                    "x2_pred": f"{pr[2]:.2f}",
                    "y2_pred": f"{pr[3]:.2f}",
                    "x_err_px": f"{0.5 * (abs(pr[0] - gt[0]) + abs(pr[2] - gt[2])):.2f}",
                    "y_err_px": f"{0.5 * (abs(pr[1] - gt[1]) + abs(pr[3] - gt[3])):.2f}",
                    "iou": f"{box_iou(pr, gt):.5f}",
                }
            )


def build_review_html(path: Path, title: str, samples: Sequence[Sample], preds: np.ndarray, max_rows: int) -> str:
    rows = []
    for idx, (s, p) in enumerate(zip(samples, preds)):
        gt = (s.x1_norm * s.image_w, s.y1_norm * s.image_h, s.x2_norm * s.image_w, s.y2_norm * s.image_h)
        pr = (
            float(min(p[0], p[2])) * s.image_w,
            float(min(p[1], p[3])) * s.image_h,
            float(max(p[0], p[2])) * s.image_w,
            float(max(p[1], p[3])) * s.image_h,
        )
        err = 0.5 * (abs(pr[0] - gt[0]) + abs(pr[2] - gt[2])) + 0.5 * (abs(pr[1] - gt[1]) + abs(pr[3] - gt[3]))
        rows.append((err, idx, s, gt, pr))
    rows.sort(reverse=True, key=lambda x: x[0])
    out_dir = path.parent / f"{path.stem}_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for rank, (err, idx, s, gt, pr) in enumerate(rows[:max_rows], start=1):
        try:
            with Image.open(s.image_path).convert("RGB") as im:
                draw = ImageDraw.Draw(im)
                draw.rectangle(gt, outline=(0, 255, 0), width=3)
                draw.rectangle(pr, outline=(255, 64, 64), width=3)
                im.thumbnail((520, 320))
                img_name = f"{rank:03d}_{_safe_slug(s.sample_id)}.jpg"
                im.save(out_dir / img_name, quality=88)
        except Exception:
            continue
        cards.append(
            f"<div class='card'><img src='{out_dir.name}/{img_name}'><p>rank={rank} err={err:.1f}px "
            f"iou={box_iou(pr, gt):.3f}<br>{s.config_folder}<br>setup={s.setup_id} depth={s.depth_mm:g} flip={s.flip_state}</p></div>"
        )
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #f7f7f4; color: #191919; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 14px; }}
.card {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px; }}
.card img {{ max-width: 100%; display: block; cursor: zoom-in; }}
p {{ font-size: 12px; line-height: 1.35; }}
.lightbox {{ position: fixed; inset: 0; z-index: 9999; display: none; align-items: center; justify-content: center; background: rgba(0, 0, 0, 0.94); padding: 18px; }}
.lightbox.open {{ display: flex; }}
.lightbox img {{ max-width: 98vw; max-height: 94vh; object-fit: contain; cursor: zoom-out; }}
.lightbox button {{ position: fixed; top: 14px; right: 18px; width: 42px; height: 42px; border: 0; border-radius: 21px; background: rgba(255, 255, 255, 0.16); color: white; font-size: 28px; line-height: 1; cursor: pointer; }}
</style>
<h1>{title}</h1>
<p>Green = GT, red = prediction.</p>
<div class="grid">{''.join(cards)}</div>
<div id="lightbox" class="lightbox" aria-hidden="true"><button type="button" aria-label="Close">×</button><img alt=""></div>
<script>
const lightbox = document.getElementById("lightbox");
const lightboxImg = lightbox.querySelector("img");
function closeLightbox() {{
  lightbox.classList.remove("open");
  lightbox.setAttribute("aria-hidden", "true");
  lightboxImg.removeAttribute("src");
}}
document.querySelectorAll(".card img").forEach((img) => {{
  img.addEventListener("click", () => {{
    lightboxImg.src = img.src;
    lightboxImg.alt = img.alt || "";
    lightbox.classList.add("open");
    lightbox.setAttribute("aria-hidden", "false");
  }});
}});
lightbox.addEventListener("click", (event) => {{
  if (event.target === lightbox || event.target.tagName === "BUTTON" || event.target === lightboxImg) closeLightbox();
}});
document.addEventListener("keydown", (event) => {{
  if (event.key === "Escape") closeLightbox();
}});
</script>
"""
    path.write_text(html, encoding="utf-8")
    return path.as_posix()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train small RECT_DEPTH bbox regressor.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-per-config", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--image-width", type=int, default=384)
    ap.add_argument("--image-height", type=int, default=216)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--train-ratio", type=float, default=0.72)
    ap.add_argument("--val-ratio", type=float, default=0.14)
    ap.add_argument("--allow-fallback", action="store_true")
    ap.add_argument("--include-flips", default="", help="Comma-separated subset, e.g. nf,ud. Empty keeps all.")
    ap.add_argument("--no-coord-channels", action="store_true")
    ap.add_argument("--max-review-rows", type=int, default=80)
    args = ap.parse_args()

    set_seed(args.seed)
    manifest = Path(args.manifest).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    include_flips = [x.strip() for x in args.include_flips.split(",") if x.strip()]
    samples, load_meta = load_samples(
        manifest=manifest,
        limit=int(args.limit),
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        allow_fallback=bool(args.allow_fallback),
        include_flips=include_flips,
        max_per_config=int(args.max_per_config),
    )
    print(json.dumps({"load_meta": load_meta}, default=str), flush=True)
    splits: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}
    for s in samples:
        splits[s.split].append(s)
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise RuntimeError(f"Insufficient split sizes: { {k: len(v) for k, v in splits.items()} }")

    image_size = (int(args.image_width), int(args.image_height))
    loaders = {
        "train": DataLoader(
            RectDepthBBoxDataset(splits["train"], image_size=image_size, train=True),
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=False,
        ),
        "val": DataLoader(
            RectDepthBBoxDataset(splits["val"], image_size=image_size, train=False),
            batch_size=max(1, int(args.batch_size) * 2),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=False,
        ),
        "test": DataLoader(
            RectDepthBBoxDataset(splits["test"], image_size=image_size, train=False),
            batch_size=max(1, int(args.batch_size) * 2),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=False,
        ),
    }

    device = pick_device(args.device)
    model = SmallBBoxRegressor(use_coord_channels=not args.no_coord_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(2, int(args.epochs)))

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_score = float("inf")
    history: List[Dict[str, float]] = []
    log_path = out_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_fh:
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            losses: List[float] = []
            for xb, yb, _idx in loaders["train"]:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = bbox_loss(pred, yb)
                if not torch.isfinite(loss).all():
                    raise RuntimeError(f"Non-finite loss at epoch={epoch}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                losses.append(float(loss.item()))
            sched.step()
            val_pred, val_metrics = infer_dataset(model, loaders["val"], splits["val"], device)
            rec = {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "lr": float(opt.param_groups[0]["lr"]),
                "val_mae_x_px": val_metrics["mae_x_px"],
                "val_mae_y_px": val_metrics["mae_y_px"],
                "val_iou_mean": val_metrics["iou_mean"],
                "val_score_xy": val_metrics["score_xy"],
            }
            history.append(rec)
            log_fh.write(json.dumps(rec) + "\n")
            log_fh.flush()
            print(
                f"[epoch {epoch:03d}] loss={rec['train_loss']:.5f} "
                f"val(x,y,iou)=({rec['val_mae_x_px']:.2f}, {rec['val_mae_y_px']:.2f}, {rec['val_iou_mean']:.3f})",
                flush=True,
            )
            if val_metrics["score_xy"] < best_score:
                best_score = val_metrics["score_xy"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_state,
                        "image_size": image_size,
                        "val_score_xy": best_score,
                        "args": vars(args),
                    },
                    out_dir / "best_model.pt",
                )

    if best_state is None:
        raise RuntimeError("No best checkpoint produced.")
    model.load_state_dict(best_state, strict=True)
    train_pred, train_metrics = infer_dataset(model, loaders["train"], splits["train"], device)
    val_pred, val_metrics = infer_dataset(model, loaders["val"], splits["val"], device)
    test_pred, test_metrics = infer_dataset(model, loaders["test"], splits["test"], device)

    save_predictions(out_dir / "train_predictions_best.csv", splits["train"], train_pred)
    save_predictions(out_dir / "val_predictions_best.csv", splits["val"], val_pred)
    save_predictions(out_dir / "test_predictions_best.csv", splits["test"], test_pred)
    val_html = build_review_html(out_dir / "val_review.html", "RECT_DEPTH val worst predictions", splits["val"], val_pred, int(args.max_review_rows))
    test_html = build_review_html(out_dir / "test_review.html", "RECT_DEPTH test worst predictions", splits["test"], test_pred, int(args.max_review_rows))

    summary = {
        "manifest": manifest.as_posix(),
        "output_dir": out_dir.as_posix(),
        "device": str(device),
        "image_size": image_size,
        "load_meta": load_meta,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "best_epoch": best_epoch,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "best_model": (out_dir / "best_model.pt").as_posix(),
        "val_review_html": val_html,
        "test_review_html": test_html,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
