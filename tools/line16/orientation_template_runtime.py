#!/usr/bin/env python3
"""Runtime utilities for line16 orientation-template models.

This module contains only reusable inference/eval components.
The previous training script has been intentionally removed and will be redesigned.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(frozen=True)
class SampleRow:
    image_path: Path
    split: str
    manufacturer: str
    model_name: str
    candidates: Tuple[Tuple[float, float, float, float], ...]  # (x1,y1,x2,y2) * 4


def _clamp_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    x1 = float(max(0.0, min(x1, width - 1.0)))
    y1 = float(max(0.0, min(y1, height - 1.0)))
    x2 = float(max(1.0, min(x2, float(width))))
    y2 = float(max(1.0, min(y2, float(height))))
    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return x1, y1, x2, y2


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
    x1 = torch.minimum(boxes[..., 0], boxes[..., 2])
    y1 = torch.minimum(boxes[..., 1], boxes[..., 3])
    x2 = torch.maximum(boxes[..., 0], boxes[..., 2])
    y2 = torch.maximum(boxes[..., 1], boxes[..., 3])
    out = torch.stack([x1, y1, x2, y2], dim=-1)
    return torch.clamp(out, 0.0, 1.0)


def choose_device(prefer: Optional[str]) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_manifest_rows(manifest_path: Path, strict_manifest: bool) -> Tuple[List[SampleRow], int]:
    rows: List[SampleRow] = []
    missing_count = 0
    missing_examples: List[str] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            image_path = Path(row["image_path"]).expanduser().resolve()
            if not image_path.exists():
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(image_path.as_posix())
                continue

            def _read_rect(prefix: str) -> Tuple[float, float, float, float]:
                left = float(row[f"{prefix}_left"])
                top = float(row[f"{prefix}_top"])
                right = float(row[f"{prefix}_right"])
                bottom = float(row[f"{prefix}_bottom"])
                return left, top, right, bottom

            candidates = (
                _read_rect("rect0"),
                _read_rect("rect1"),
                _read_rect("rect2"),
                _read_rect("rect3"),
            )
            rows.append(
                SampleRow(
                    image_path=image_path,
                    split=row["split"].strip().lower(),
                    manufacturer=row["manufacturer"].strip(),
                    model_name=row["model_name"].strip(),
                    candidates=candidates,
                )
            )

    if missing_count > 0:
        msg = f"Manifest contains {missing_count} missing images; rows were skipped."
        if strict_manifest:
            raise RuntimeError(msg + f" Examples: {missing_examples}")
        print(msg, flush=True)
        for sample in missing_examples:
            print(f"- missing: {sample}", flush=True)

    return rows, missing_count
