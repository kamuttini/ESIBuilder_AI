#!/usr/bin/env python3
"""Train an image model to predict scale-line geometry on full-frame ultrasound images.

Targets:
- x center of the vertical scale line (normalized)
- y_top and y_bottom of the scale segment (normalized)

Outputs:
- best checkpoint (torch .pt)
- metrics summary JSON
- per-sample predictions CSV
- quick HTML review for worst validation/test errors
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class Sample:
    sample_id: str
    split: str
    fss_path: str
    setup_id: str
    depth_index: int
    depth_mm: float
    video_w: int
    video_h: int
    x_norm: float
    y_top_norm: float
    y_bottom_norm: float
    length_mm: float
    tick_mm: float
    image_path: str


def _f(text: str) -> float:
    return float(text.strip())


def _i(text: str) -> int:
    return int(float(text.strip()))


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:120]


def _row_filter_text(sample_id: str, fss_path: str, setup_id: str) -> str:
    return f"{sample_id} {fss_path} {setup_id}".lower()


def row_matches_exclusion(sample_id: str, fss_path: str, setup_id: str, label: str) -> bool:
    text = _row_filter_text(sample_id=sample_id, fss_path=fss_path, setup_id=setup_id)
    if label == "fusion":
        return "fusion" in text
    if label == "negative":
        return "negative" in text
    if label == "proibite":
        return any(
            tok in text
            for tok in (
                "proibite",
                "proibita",
                "proibiti",
                "proibito",
                "prohibited",
                "sbagliat",
                "non usare",
                "non_usare",
            )
        )
    return False


def find_full_frame_image(fss_path: str, setup_id: str, depth_index: int) -> Optional[Path]:
    fss = Path(fss_path)
    root = fss.parent.parent
    sid = setup_id.strip() or fss.stem.replace("setup_", "")
    image_samples = root / "image_samples"
    if not image_samples.exists():
        return None

    idx0 = max(0, depth_index - 1)
    stems = [
        f"image_depth_value_setup_{idx0}",
        f"image_depth_find_flip_ud_setup_{idx0}",
        f"image_depth_value_setup_{depth_index}",
        f"image_depth_find_flip_ud_setup_{depth_index}",
        f"image_depth_value_setup_{sid}_{idx0}",
        f"image_depth_value_setup_{sid}_{depth_index}",
        "image_orientation_setup_0",
        "image_th_echo_negative_0",
        "image_th_probe_negative_0",
    ]
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    for stem in stems:
        for ext in exts:
            p = image_samples / f"{stem}{ext}"
            if p.exists():
                return p

    for ext in ("image_*.png", "image_*.jpg", "image_*.jpeg", "image_*.bmp", "image_*.tif", "image_*.tiff"):
        found = [p for p in sorted(image_samples.glob(ext)) if not p.name.startswith("._")]
        if found:
            return found[0]

    return None


def load_samples(
    manifest: Path,
    limit: int = 0,
    progress_every: int = 500,
    exclude_labels: Optional[Sequence[str]] = None,
) -> Tuple[List[Sample], Dict[str, int], int]:
    out: List[Sample] = []
    excluded_by_filter: Counter[str] = Counter()
    excluded_total = 0
    active_labels = list(exclude_labels or [])
    with manifest.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {
            "sample_id",
            "split",
            "fss_path",
            "setup_id",
            "depth_index",
            "depth_mm",
            "x1",
            "y1",
            "y2",
            "video_x_size",
            "video_y_size",
            "length_mm",
            "tick_mm",
        }
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise RuntimeError(f"Manifest missing columns: {missing}")

        for i, row in enumerate(reader, start=1):
            split = (row.get("split") or "").strip().lower()
            if split not in {"train", "val", "test"}:
                continue
            sample_id = row["sample_id"]
            fss_path = row["fss_path"]
            setup_id = row.get("setup_id", "")
            matched = [label for label in active_labels if row_matches_exclusion(sample_id, fss_path, setup_id, label)]
            if matched:
                excluded_total += 1
                for label in matched:
                    excluded_by_filter[label] += 1
                continue
            depth_index = _i(row["depth_index"])
            img_path = find_full_frame_image(fss_path, setup_id, depth_index)
            if img_path is None:
                continue

            y1 = _f(row["y1"])
            y2 = _f(row["y2"])
            y_top = min(y1, y2)
            y_bottom = max(y1, y2)
            vw = _i(row["video_x_size"])
            vh = _i(row["video_y_size"])
            x_norm = _f(row["x1"]) / max(1.0, float(vw))
            y_top_norm = y_top / max(1.0, float(vh))
            y_bottom_norm = y_bottom / max(1.0, float(vh))

            out.append(
                Sample(
                    sample_id=sample_id,
                    split=split,
                    fss_path=fss_path,
                    setup_id=setup_id,
                    depth_index=depth_index,
                    depth_mm=_f(row["depth_mm"]),
                    video_w=vw,
                    video_h=vh,
                    x_norm=x_norm,
                    y_top_norm=y_top_norm,
                    y_bottom_norm=y_bottom_norm,
                    length_mm=_f(row["length_mm"]),
                    tick_mm=_f(row["tick_mm"]),
                    image_path=img_path.as_posix(),
                )
            )
            if progress_every > 0 and len(out) % progress_every == 0:
                print(f"[load] resolved images: {len(out)} samples", flush=True)
            if limit > 0 and len(out) >= limit:
                break
            if progress_every > 0 and i % max(1000, progress_every * 2) == 0 and len(out) == 0:
                print(f"[load] scanned rows: {i}, still 0 samples with images", flush=True)
    return out, dict(excluded_by_filter), excluded_total


class ScaleDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        image_size: Tuple[int, int],
        train: bool,
        train_shift_frac: float = 0.0,
        train_shift_prob: float = 0.0,
        train_augment: bool = True,
    ) -> None:
        self.samples = list(samples)
        self.train = train
        self.train_shift_frac = max(0.0, float(train_shift_frac))
        self.train_shift_prob = float(np.clip(train_shift_prob, 0.0, 1.0))
        self.train_augment = bool(train_augment)
        w, h = image_size
        self.out_w = int(w)
        self.out_h = int(h)
        # Keep PIL-stage transforms explicit so we can update target x when shifting.
        self.resize = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_jitter = transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.08, hue=0.015)
        self.train_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        y = np.array([s.x_norm, s.y_top_norm, s.y_bottom_norm], dtype=np.float32)
        img = self.resize(img)

        if self.train and self.train_augment:
            if random.random() < 0.90:
                img = self.train_jitter(img)
            if random.random() < 0.15:
                img = self.train_blur(img)

        # Geometric x-shift augmentation with target correction:
        # helps reduce setup-specific x memorization and improves side generalization.
        if self.train and self.train_shift_frac > 0.0 and self.train_shift_prob > 0.0:
            if random.random() < self.train_shift_prob:
                max_shift_px = int(round(self.train_shift_frac * self.out_w))
                if max_shift_px > 0:
                    dx = random.randint(-max_shift_px, max_shift_px)
                    if dx != 0:
                        img = TF.affine(
                            img,
                            angle=0.0,
                            translate=[dx, 0],
                            scale=1.0,
                            shear=[0.0, 0.0],
                            interpolation=transforms.InterpolationMode.BILINEAR,
                            fill=0,
                        )
                        y[0] = float(np.clip(y[0] + (float(dx) / float(self.out_w)), 0.0, 1.0))

        x = self.norm(TF.to_tensor(img))
        y = torch.tensor(y, dtype=torch.float32)
        return x, y, idx


class SmallScaleRegressor(nn.Module):
    def __init__(self, head_type: str = "spatial", use_coord_channels: bool = True) -> None:
        super().__init__()
        self.head_type = head_type
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
        if self.head_type == "pooled":
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 96),
                nn.SiLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(96, 3),
            )
        elif self.head_type == "spatial":
            # Keep spatial layout information for better line localization.
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(320),
                nn.SiLU(inplace=True),
                nn.Dropout(0.20),
                nn.Linear(320, 96),
                nn.SiLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(96, 3),
            )
        else:
            raise ValueError(f"Unsupported head_type={head_type!r}; expected 'pooled' or 'spatial'.")

    def _coord_maps(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_coord_channels:
            return x
        b, _, h, w = x.shape
        xs = torch.linspace(-1.0, 1.0, steps=w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        ys = torch.linspace(-1.0, 1.0, steps=h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        return torch.cat([x, xs, ys], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._coord_maps(x)
        z = self.backbone(x)
        out = self.head(z)
        # Keep outputs in [0,1]
        return torch.sigmoid(out)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def line_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 0.02,
    wx: float = 1.0,
    wy_top: float = 1.0,
    wy_bottom: float = 1.0,
    order_penalty_weight: float = 0.25,
) -> torch.Tensor:
    per = F.smooth_l1_loss(pred, target, beta=float(beta), reduction="none")
    weights = pred.new_tensor([float(wx), float(wy_top), float(wy_bottom)]).view(1, 3)
    base = (per * weights).sum(dim=1).mean() / max(1e-6, float(wx + wy_top + wy_bottom))
    # Encourage y_bottom >= y_top.
    order_penalty = torch.relu(pred[:, 1] - pred[:, 2]).mean()
    return base + float(order_penalty_weight) * order_penalty


def pick_device(requested: str) -> torch.device:
    req = (requested or "auto").strip().lower()
    # On this workstation MPS produced unstable NaNs in training; prefer CPU for reliability.
    if req == "auto":
        return torch.device("cpu")
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if req == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")


def compute_errors(samples: Sequence[Sample], preds_norm: np.ndarray) -> Dict[str, float]:
    x_err = []
    y_err = []
    for s, p in zip(samples, preds_norm):
        x_pred = float(p[0]) * s.video_w
        y_top_pred = float(min(p[1], p[2])) * s.video_h
        y_bottom_pred = float(max(p[1], p[2])) * s.video_h
        x_gt = s.x_norm * s.video_w
        y_top_gt = s.y_top_norm * s.video_h
        y_bottom_gt = s.y_bottom_norm * s.video_h
        x_err.append(abs(x_pred - x_gt))
        y_err.append(0.5 * (abs(y_top_pred - y_top_gt) + abs(y_bottom_pred - y_bottom_gt)))
    return {
        "rows": float(len(samples)),
        "mae_x_px": float(np.mean(x_err)) if x_err else float("nan"),
        "mae_y_px": float(np.mean(y_err)) if y_err else float("nan"),
        "score_xy": float(np.mean(x_err) + np.mean(y_err)) if x_err else float("nan"),
    }


@torch.no_grad()
def infer_dataset(
    model: nn.Module,
    loader: DataLoader,
    samples: Sequence[Sample],
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, float]]:
    model.eval()
    out = np.zeros((len(samples), 3), dtype=np.float32)
    for xb, _, idxb in loader:
        xb = xb.to(device)
        pb = model(xb).detach().cpu().numpy()
        out[idxb.numpy()] = pb
    return out, compute_errors(samples, out)


def save_predictions_csv(path: Path, samples: Sequence[Sample], preds_norm: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "sample_id",
                "split",
                "image_path",
                "x_gt",
                "y_top_gt",
                "y_bottom_gt",
                "x_pred",
                "y_top_pred",
                "y_bottom_pred",
                "x_err_px",
                "y_err_px",
                "depth_mm",
                "length_mm_gt",
                "tick_mm_gt",
            ]
        )
        for s, p in zip(samples, preds_norm):
            x_gt = s.x_norm * s.video_w
            y_top_gt = s.y_top_norm * s.video_h
            y_bottom_gt = s.y_bottom_norm * s.video_h
            x_pred = float(p[0]) * s.video_w
            y_top_pred = float(min(p[1], p[2])) * s.video_h
            y_bottom_pred = float(max(p[1], p[2])) * s.video_h
            x_err = abs(x_pred - x_gt)
            y_err = 0.5 * (abs(y_top_pred - y_top_gt) + abs(y_bottom_pred - y_bottom_gt))
            w.writerow(
                [
                    s.sample_id,
                    s.split,
                    s.image_path,
                    f"{x_gt:.4f}",
                    f"{y_top_gt:.4f}",
                    f"{y_bottom_gt:.4f}",
                    f"{x_pred:.4f}",
                    f"{y_top_pred:.4f}",
                    f"{y_bottom_pred:.4f}",
                    f"{x_err:.4f}",
                    f"{y_err:.4f}",
                    f"{s.depth_mm:.3f}",
                    f"{s.length_mm:.3f}",
                    f"{s.tick_mm:.3f}",
                ]
            )


def draw_preview(sample: Sample, pred_norm: np.ndarray, out_path: Path) -> None:
    img = Image.open(sample.image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    x_gt = sample.x_norm * sample.video_w
    y1_gt = sample.y_top_norm * sample.video_h
    y2_gt = sample.y_bottom_norm * sample.video_h

    x_pr = float(pred_norm[0]) * sample.video_w
    y1_pr = float(min(pred_norm[1], pred_norm[2])) * sample.video_h
    y2_pr = float(max(pred_norm[1], pred_norm[2])) * sample.video_h

    draw.line((x_gt, y1_gt, x_gt, y2_gt), fill=(0, 255, 120, 255), width=6)
    draw.line((x_pr, y1_pr, x_pr, y2_pr), fill=(255, 0, 220, 255), width=5)

    r = 5
    for x, y, c in [
        (x_gt, y1_gt, (0, 255, 120, 255)),
        (x_gt, y2_gt, (0, 255, 120, 255)),
        (x_pr, y1_pr, (255, 0, 220, 255)),
        (x_pr, y2_pr, (255, 0, 220, 255)),
    ]:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=c, outline=(255, 255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=92)


def build_review_html(
    out_dir: Path,
    split_name: str,
    samples: Sequence[Sample],
    preds_norm: np.ndarray,
    max_rows: int,
) -> Path:
    rows = []
    for s, p in zip(samples, preds_norm):
        x_gt = s.x_norm * s.video_w
        y1_gt = s.y_top_norm * s.video_h
        y2_gt = s.y_bottom_norm * s.video_h
        x_pr = float(p[0]) * s.video_w
        y1_pr = float(min(p[1], p[2])) * s.video_h
        y2_pr = float(max(p[1], p[2])) * s.video_h
        x_err = abs(x_pr - x_gt)
        y_err = 0.5 * (abs(y1_pr - y1_gt) + abs(y2_pr - y2_gt))
        score = x_err + y_err
        rows.append((score, x_err, y_err, s, p))
    rows.sort(key=lambda t: t[0], reverse=True)
    worst = rows[: max(0, max_rows)]

    preview_dir = out_dir / f"review_{split_name}_previews"
    html_path = out_dir / f"review_{split_name}.html"
    trs: List[str] = []
    for i, (score, x_err, y_err, s, p) in enumerate(worst, start=1):
        out_img = preview_dir / f"{i:04d}_{_safe_slug(s.sample_id)}.jpg"
        try:
            draw_preview(s, p, out_img)
            img_tag = f'<a href="{preview_dir.name}/{out_img.name}" target="_blank"><img src="{preview_dir.name}/{out_img.name}" loading="lazy" /></a>'
        except Exception:
            img_tag = "preview_error"
        trs.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{img_tag}</td>"
            f"<td>{s.sample_id}</td>"
            f"<td>{score:.2f}</td>"
            f"<td>{x_err:.2f}</td>"
            f"<td>{y_err:.2f}</td>"
            f"<td>{Path(s.image_path).name}</td>"
            "</tr>"
        )

    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Scale Image Model Review - {split_name}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 16px; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    img {{ width: 520px; max-width: 100%; border: 1px solid #cbd5e1; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Scale Image Model - Worst {split_name} Errors</h1>
  <p>Overlay: <b>green=GT</b>, <b>magenta=pred</b></p>
  <table>
    <thead>
      <tr><th>#</th><th>Preview</th><th>sample_id</th><th>score</th><th>x_err_px</th><th>y_err_px</th><th>source</th></tr>
    </thead>
    <tbody>
      {''.join(trs)}
    </tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )
    return html_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train scale-line image regressor.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/23_scale_manifest_ssd_esi1_n1_leakfree/manifest_scale_train_val_test_leakfree.csv"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--image-height", type=int, default=360)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--head-type",
        type=str,
        default="spatial",
        choices=["spatial", "pooled"],
        help="Regression head type. 'spatial' preserves localization cues better.",
    )
    p.add_argument(
        "--no-coord-channels",
        action="store_true",
        help="Disable extra x/y coordinate channels (enabled by default).",
    )
    p.add_argument("--loss-beta", type=float, default=0.02, help="SmoothL1 beta.")
    p.add_argument("--loss-wx", type=float, default=2.2, help="Loss weight for x coordinate.")
    p.add_argument("--loss-wy-top", type=float, default=0.7, help="Loss weight for y_top coordinate.")
    p.add_argument("--loss-wy-bottom", type=float, default=0.7, help="Loss weight for y_bottom coordinate.")
    p.add_argument(
        "--order-penalty-weight",
        type=float,
        default=0.20,
        help="Penalty weight to discourage y_top > y_bottom.",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--max-review-rows", type=int, default=140)
    p.add_argument(
        "--eval-train-every",
        type=int,
        default=0,
        help="If >0, compute train pixel metrics every N epochs (0 disables per-epoch train metrics).",
    )
    p.add_argument("--limit", type=int, default=0, help="Optional max manifest rows for quick experiments.")
    p.add_argument(
        "--train-shift-frac",
        type=float,
        default=0.08,
        help="Max horizontal train shift as fraction of image width (target x corrected accordingly).",
    )
    p.add_argument(
        "--train-shift-prob",
        type=float,
        default=0.75,
        help="Probability of applying horizontal train shift augmentation.",
    )
    p.add_argument(
        "--disable-train-augment",
        action="store_true",
        help="Disable color jitter/blur augmentation to fit exact corrected images.",
    )
    p.add_argument(
        "--exclude-fusion",
        action="store_true",
        help="Exclude rows with 'fusion' in setup/fss/sample identifiers.",
    )
    p.add_argument(
        "--exclude-negative",
        action="store_true",
        help="Exclude rows with 'negative' in setup/fss/sample identifiers.",
    )
    p.add_argument(
        "--exclude-proibite",
        action="store_true",
        help="Exclude rows with forbidden/proibite markers in setup/fss/sample identifiers.",
    )
    p.add_argument(
        "--exclude-prohibited",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    manifest = args.manifest.expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_labels: List[str] = []
    if args.exclude_fusion:
        exclude_labels.append("fusion")
    if args.exclude_negative:
        exclude_labels.append("negative")
    if args.exclude_proibite or args.exclude_prohibited:
        exclude_labels.append("proibite")

    samples, excluded_by_filter, excluded_total = load_samples(
        manifest,
        limit=args.limit,
        exclude_labels=exclude_labels,
    )
    splits: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}
    for s in samples:
        splits[s.split].append(s)

    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise RuntimeError(
            f"Not enough samples by split after image resolution. "
            f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
        )

    image_size = (args.image_width, args.image_height)
    ds_train = ScaleDataset(
        splits["train"],
        image_size=image_size,
        train=True,
        train_shift_frac=args.train_shift_frac,
        train_shift_prob=args.train_shift_prob,
        train_augment=not args.disable_train_augment,
    )
    ds_val = ScaleDataset(splits["val"], image_size=image_size, train=False)
    ds_test = ScaleDataset(splits["test"], image_size=image_size, train=False)

    loader_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    loader_test = DataLoader(
        ds_test,
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    device = pick_device(args.device)
    model = SmallScaleRegressor(
        head_type=args.head_type,
        use_coord_channels=not args.no_coord_channels,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(2, args.epochs))

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_val_score = float("inf")
    history: List[Dict[str, float]] = []

    log_path = out_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_fh:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses: List[float] = []
            for xb, yb, _ in loader_train:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = line_loss(
                    pred,
                    yb,
                    beta=args.loss_beta,
                    wx=args.loss_wx,
                    wy_top=args.loss_wy_top,
                    wy_bottom=args.loss_wy_bottom,
                    order_penalty_weight=args.order_penalty_weight,
                )
                if not torch.isfinite(loss).all():
                    raise RuntimeError(
                        f"Non-finite loss detected at epoch={epoch}. "
                        f"Try --device cpu and verify input images."
                    )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                opt.step()
                train_losses.append(float(loss.item()))

            sched.step()

            val_pred, val_metrics = infer_dataset(model, loader_val, splits["val"], device)
            train_mae_x = float("nan")
            train_mae_y = float("nan")
            if args.eval_train_every > 0 and (epoch % args.eval_train_every == 0):
                _, train_metrics_epoch = infer_dataset(model, loader_train, splits["train"], device)
                train_mae_x = train_metrics_epoch["mae_x_px"]
                train_mae_y = train_metrics_epoch["mae_y_px"]

            rec = {
                "epoch": float(epoch),
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "lr": float(opt.param_groups[0]["lr"]),
                "train_mae_x_px": train_mae_x,
                "train_mae_y_px": train_mae_y,
                "val_mae_x_px": val_metrics["mae_x_px"],
                "val_mae_y_px": val_metrics["mae_y_px"],
                "val_score_xy": val_metrics["score_xy"],
            }
            history.append(rec)
            log_fh.write(json.dumps(rec) + "\n")
            log_fh.flush()

            print(
                f"[epoch {epoch:03d}] "
                f"loss={rec['train_loss']:.5f} "
                f"val(x,y)=({rec['val_mae_x_px']:.2f}, {rec['val_mae_y_px']:.2f}) "
                f"val_score={rec['val_score_xy']:.2f}",
                flush=True,
            )

            if val_metrics["score_xy"] < best_val_score:
                best_val_score = val_metrics["score_xy"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_state,
                        "val_score_xy": best_val_score,
                        "image_size": image_size,
                    },
                    out_dir / "best_model.pt",
                )
                save_predictions_csv(out_dir / "val_predictions_best.csv", splits["val"], val_pred)

    if best_state is None:
        raise RuntimeError("Training did not produce any best checkpoint.")

    model.load_state_dict(best_state, strict=True)
    val_pred, val_metrics = infer_dataset(model, loader_val, splits["val"], device)
    test_pred, test_metrics = infer_dataset(model, loader_test, splits["test"], device)
    train_pred, train_metrics = infer_dataset(model, loader_train, splits["train"], device)

    save_predictions_csv(out_dir / "train_predictions_best.csv", splits["train"], train_pred)
    save_predictions_csv(out_dir / "val_predictions_best.csv", splits["val"], val_pred)
    save_predictions_csv(out_dir / "test_predictions_best.csv", splits["test"], test_pred)

    val_html = build_review_html(out_dir, "val", splits["val"], val_pred, max_rows=args.max_review_rows)
    test_html = build_review_html(out_dir, "test", splits["test"], test_pred, max_rows=args.max_review_rows)

    summary = {
        "manifest": manifest.as_posix(),
        "output_dir": out_dir.as_posix(),
        "exclusion_filters_enabled": exclude_labels,
        "rows_excluded_total": int(excluded_total),
        "rows_excluded_by_filter": excluded_by_filter,
        "rows_excluded_fusion": int(excluded_by_filter.get("fusion", 0)),
        "rows_excluded_negative": int(excluded_by_filter.get("negative", 0)),
        "rows_excluded_proibite": int(excluded_by_filter.get("proibite", 0)),
        "rows_total_with_images": len(samples),
        "rows_by_split": {k: len(v) for k, v in splits.items()},
        "image_size": {"width": args.image_width, "height": args.image_height},
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "head_type": args.head_type,
        "use_coord_channels": bool(not args.no_coord_channels),
        "loss_beta": args.loss_beta,
        "loss_wx": args.loss_wx,
        "loss_wy_top": args.loss_wy_top,
        "loss_wy_bottom": args.loss_wy_bottom,
        "order_penalty_weight": args.order_penalty_weight,
        "train_shift_frac": args.train_shift_frac,
        "train_shift_prob": args.train_shift_prob,
        "eval_train_every": args.eval_train_every,
        "best_epoch": best_epoch,
        "best_val_score_xy": best_val_score,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "artifacts": {
            "best_model": (out_dir / "best_model.pt").as_posix(),
            "train_log_jsonl": (out_dir / "train_log.jsonl").as_posix(),
            "train_predictions_csv": (out_dir / "train_predictions_best.csv").as_posix(),
            "val_predictions_csv": (out_dir / "val_predictions_best.csv").as_posix(),
            "test_predictions_csv": (out_dir / "test_predictions_best.csv").as_posix(),
            "val_review_html": val_html.as_posix(),
            "test_review_html": test_html.as_posix(),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
