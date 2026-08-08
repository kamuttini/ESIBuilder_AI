#!/usr/bin/env python3
"""Inference for the per-vendor scale heatmap localiser.

Loads the checkpoint produced by ``train_scale_heatmap.py`` and returns a
``HeatmapPrior``: where the ruler column is, where the zero is, which way the numbers
count, and a fallback calibration.

The prior is meant to be *composed* with the classical detector, not to replace it:

* the network says **which column** to look in, which is the decision the threshold-based
  stage gets wrong (whole setups came back ``no_ladder``);
* the classical stage then finds the individual ticks inside that narrow band, where it is
  precise (median 1 px on the zero when it locks on);
* the OCR still supplies the absolute scale, because the printed numbers are the only
  direct evidence of it — the network's ``mm_per_px`` is a fallback and a cross-check.

Torch is imported lazily so that importing this module costs nothing on a machine without
the training environment, and ``load_registry`` simply returns an empty registry there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from heatmap_codec import decode_prediction


@dataclass
class HeatmapPrior:
    """What the network believes about one frame."""

    x: float
    y_zero: float
    mm_per_px: float
    direction: int
    x_confidence: float
    y_confidence: float
    direction_confidence: float
    x_sharpness: float
    y_sharpness: float
    vendor_model: str

    @property
    def ambiguous_column(self) -> bool:
        """Two equally plausible rulers — typical of fusion / biplane layouts.

        Worth surfacing rather than hiding: on those frames the answer is a coin flip and
        the setup consensus should be the one to break the tie.
        """
        return self.x_sharpness < 1.5


class ScaleHeatmapModel:
    """One vendor's checkpoint, ready to run."""

    def __init__(self, checkpoint_path: Path, device: str = "auto") -> None:
        import torch

        from train_scale_heatmap import build_model, pick_device

        self.path = Path(checkpoint_path)
        ckpt = torch.load(self.path, map_location="cpu")
        self.device = pick_device(device)
        self.in_w = int(ckpt.get("image_width", 512))
        self.in_h = int(ckpt.get("image_height", 512))
        self.model = build_model(int(ckpt.get("width", 48)))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()
        self.val_metrics = ckpt.get("val", {})
        self._torch = torch

    def predict(self, image: np.ndarray, vendor_model: str = "") -> HeatmapPrior:
        torch = self._torch
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        small = cv2.resize(gray, (self.in_w, self.in_h), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy((small.astype(np.float32) / 255.0)[None, None]).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
            xh = torch.softmax(out["x_logits"], dim=1)[0].cpu().numpy()
            yh = torch.softmax(out["y_logits"], dim=1)[0].cpu().numpy()
            ls = float(out["log_span_mm"][0].cpu())
            dl = out["direction_logits"][0].cpu().numpy()
        d = decode_prediction(xh, yh, ls, dl, w, h)
        return HeatmapPrior(
            x=d["x"],
            y_zero=d["y_zero"],
            mm_per_px=d["mm_per_px"],
            direction=d["direction"],
            x_confidence=d["x_confidence"],
            y_confidence=d["y_confidence"],
            direction_confidence=d["direction_confidence"],
            x_sharpness=d["x_sharpness"],
            y_sharpness=d["y_sharpness"],
            vendor_model=vendor_model,
        )


class HeatmapRegistry:
    """Vendor -> model, with a general model as the fallback for the long tail.

    Eight of the fifteen vendors in the corpus have 67 rows or fewer, so they cannot carry
    their own model; they fall back to whichever model is registered as ``default``.
    """

    def __init__(self, models: Dict[str, ScaleHeatmapModel]) -> None:
        self.models = models

    def __bool__(self) -> bool:
        return bool(self.models)

    def for_vendor(self, vendor: str) -> Optional[ScaleHeatmapModel]:
        key = (vendor or "").lower().replace(" ", "_")
        return self.models.get(key) or self.models.get("default")

    def predict(self, image: np.ndarray, vendor: str) -> Optional[HeatmapPrior]:
        model = self.for_vendor(vendor)
        if model is None:
            return None
        return model.predict(image, vendor_model=model.path.parent.name)


def load_registry(root: Optional[Path], device: str = "auto") -> HeatmapRegistry:
    """Load every ``<root>/<vendor_slug>/best_model.pt``.

    Returns an empty registry — rather than raising — when the directory is absent or
    torch is not installed, so the classical pipeline keeps working untouched.
    """
    models: Dict[str, ScaleHeatmapModel] = {}
    if root is None:
        return HeatmapRegistry(models)
    root = Path(root)
    if not root.is_dir():
        print(f"[heatmap] no model directory at {root}, running classical only")
        return HeatmapRegistry(models)
    for ckpt in sorted(root.glob("*/best_model.pt")):
        slug = ckpt.parent.name
        try:
            models[slug] = ScaleHeatmapModel(ckpt, device=device)
            print(f"[heatmap] loaded {slug} from {ckpt}")
        except Exception as exc:  # noqa: BLE001 - a missing torch must not break the run
            print(f"[heatmap] cannot load {ckpt}: {exc}")
    return HeatmapRegistry(models)


def describe_registry(registry: HeatmapRegistry) -> str:
    if not registry:
        return "nessun modello heatmap caricato"
    parts = []
    for slug, model in sorted(registry.models.items()):
        strict = model.val_metrics.get("strict_ok_pct")
        parts.append(f"{slug}(val strict={strict}%)")
    return ", ".join(parts)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Run the scale heatmap model on one image.")
    p.add_argument("--models", type=Path, required=True, help="Model root directory.")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--vendor", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    registry = load_registry(args.models, args.device)
    if not registry:
        print("[error] no models loaded")
        return 2
    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[error] cannot read {args.image}")
        return 2
    prior = registry.predict(img, args.vendor)
    if prior is None:
        print("[error] no model for this vendor and no default")
        return 2
    print(json.dumps(prior.__dict__, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
