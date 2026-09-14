#!/usr/bin/env python3
"""Run the needle classifier on raw acquisition folders.

Mirrors the training-time preprocessing exactly: official vendor CNN to route the
rect model, official rect regressor for the echo rectangle, crop, then the needle
classifier on the crop. A folder gets a per-image score and a folder verdict from
the mean score, classified accepted / review / reject with the two thresholds
calibrated on validation (stored in the model's metrics.json).

Example:
  python3 tools/needle/predict_needle_on_folder.py \
    --model artifacts/91_needle_models/resnet18_288_cpu/best_model.pt \
    --folder "/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION/60.GE_LogiqP9_MSK/AGHI" \
    --output-csv /tmp/needle_pred.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools/needle"))

from extract_needle_crops import _artifacts_root, _import_official, trim_dark_border  # noqa: E402
from train_needle_classifier import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_model,
    choose_device,
    letterbox,
)
from torchvision.transforms import functional as TF  # noqa: E402

PIPE_DIR = _artifacts_root() / "10_active_pipeline/pipeline_fss_head"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def iter_images(folder: Path, recursive: bool) -> List[Path]:
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(
        p for p in it if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._")
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--folder", type=Path, action="append", required=True,
                        help="Acquisition folder. Repeatable.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=0, help="0 = all")
    parser.add_argument("--margin-pct", type=float, default=2.0)
    parser.add_argument("--trim-dark-border", action="store_true")
    parser.add_argument("--accept-above", type=float, default=None,
                        help="Overrides the threshold stored next to the model.")
    parser.add_argument("--reject-below", type=float, default=None)
    parser.add_argument("--vendor-sample", type=int, default=24)
    parser.add_argument("--rect-vendor-min-confidence", type=float, default=0.70)
    args = parser.parse_args()

    device = choose_device(args.device or None)
    official = _import_official()

    checkpoint = torch.load(args.model.expanduser().resolve(), map_location="cpu", weights_only=False)
    model = build_model(checkpoint["arch"], dropout=0.0)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    image_size = int(checkpoint["image_size"])

    accept_above = args.accept_above
    reject_below = args.reject_below
    metrics_path = args.model.parent / "metrics.json"
    if (accept_above is None or reject_below is None) and metrics_path.is_file():
        policy = json.loads(metrics_path.read_text(encoding="utf-8")).get("review_policy_from_val", {})
        accept_above = accept_above if accept_above is not None else policy.get("accept_above")
        reject_below = reject_below if reject_below is not None else policy.get("reject_below")
    if accept_above is None or reject_below is None:
        raise SystemExit("soglie non disponibili: passa --accept-above / --reject-below")

    # official vendor + rect models
    vendor_ckpt = torch.load(PIPE_DIR / "models/vendor_training_no_negative_v2_power/best_model.pt",
                             map_location="cpu", weights_only=False)
    vendor_classes = list(vendor_ckpt.get("class_names") or [])
    vendor_image_size = int(vendor_ckpt.get("args", {}).get("image_size", 384))
    vendor_model = official.VendorClassifier(num_classes=len(vendor_classes), pretrained=False).to(device)
    vendor_model.load_state_dict(vendor_ckpt["model_state_dict"])
    vendor_model.eval()

    def load_rect(path: Path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        size = int(ckpt.get("args", {}).get("image_size", 320))
        net = official.RectRegressor(pretrained=False).to(device)
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()
        return net, size

    rect_models = {"generic": load_rect(PIPE_DIR / "models/rect_training_e40_run2/best_model.pt")}
    vendor_map_path = PIPE_DIR / "maps/vendor_rect_map_bk_only.json"
    if vendor_map_path.is_file():
        for vendor, rel in json.loads(vendor_map_path.read_text(encoding="utf-8")).items():
            ckpt_path = (vendor_map_path.parent / rel).resolve()
            if ckpt_path.is_file():
                rect_models[vendor] = load_rect(ckpt_path)

    out_rows: List[Dict[str, object]] = []
    for folder in args.folder:
        folder = folder.expanduser().resolve()
        images = iter_images(folder, args.recursive)
        if args.max_images:
            images = official._select_uniform_subset(images, args.max_images)
        if not images:
            print(f"{folder}: nessuna immagine")
            continue

        sample = official._select_uniform_subset(images, args.vendor_sample)
        probs = official._predict_mean_probs(
            vendor_model, sample, vendor_image_size, args.batch_size, device
        )
        best = int(probs.argmax())
        vendor, vendor_conf = vendor_classes[best], float(probs[best])
        source = vendor if (vendor in rect_models and vendor_conf >= args.rect_vendor_min_confidence) else "generic"
        rect_model, rect_size = rect_models[source]

        boxes, sizes = official._predict_rect_boxes_abs(
            rect_model, images, rect_size, args.batch_size, device
        )

        scores: List[float] = []
        with torch.inference_mode():
            for start in range(0, len(images), args.batch_size):
                chunk = images[start : start + args.batch_size]
                tensors = []
                for offset, path in enumerate(chunk):
                    idx = start + offset
                    if idx >= len(boxes):
                        continue
                    x1, y1, x2, y2 = boxes[idx]
                    ow, oh = sizes[idx]
                    mx = (x2 - x1) * args.margin_pct / 100.0
                    my = (y2 - y1) * args.margin_pct / 100.0
                    box = (
                        int(max(0, round(x1 - mx))), int(max(0, round(y1 - my))),
                        int(min(ow, round(x2 + mx))), int(min(oh, round(y2 + my))),
                    )
                    with Image.open(path) as img:
                        crop = img.convert("RGB").crop(box)
                    if args.trim_dark_border:
                        crop, _ = trim_dark_border(crop)
                    tensor = TF.normalize(
                        TF.to_tensor(letterbox(crop, image_size)),
                        mean=IMAGENET_MEAN, std=IMAGENET_STD,
                    )
                    tensors.append(tensor)
                if not tensors:
                    continue
                batch = torch.stack(tensors).to(device)
                logits = model(batch).squeeze(1).float()
                probs_b = torch.sigmoid(logits)
                probs_b = 0.5 * (probs_b + torch.sigmoid(model(torch.flip(batch, dims=[3])).squeeze(1).float()))
                scores.extend(probs_b.detach().cpu().numpy().tolist())

        if not scores:
            print(f"{folder}: nessuna predizione")
            continue

        mean_score = float(np.mean(scores))
        verdict = (
            "accepted_needle" if mean_score >= accept_above
            else "rejected_needle" if mean_score <= reject_below
            else "review"
        )
        frac_above = float(np.mean(np.asarray(scores) >= accept_above))
        print(
            f"{folder.name}: score medio {mean_score:.3f} -> {verdict}  "
            f"(vendor {vendor} {vendor_conf:.2f}, rect {source}, "
            f"{len(scores)} img, {frac_above:.0%} sopra soglia)"
        )
        for path, score in zip(images, scores):
            out_rows.append(
                {
                    "folder": str(folder), "image": path.name, "score": f"{score:.6f}",
                    "folder_mean_score": f"{mean_score:.6f}", "folder_verdict": verdict,
                    "vendor_pred": vendor, "vendor_conf": f"{vendor_conf:.4f}",
                    "rect_source": source,
                }
            )

    if args.output_csv and out_rows:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"csv: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
