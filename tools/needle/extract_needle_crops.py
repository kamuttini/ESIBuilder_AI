#!/usr/bin/env python3
"""Cache the echo-rectangle crop of every image in the needle manifest.

Why crop at all: the full frame is dominated by the vendor UI (logos, menus,
patient banner, depth ruler) and by the rectangle geometry itself. A classifier
trained on full frames learns "which machine / which preset" instead of "is
there a needle in this image", and the label here correlates with the machine.
Cropping to the echo rectangle removes that shortcut.

Why cache: the raw frames are 1920x1080 PNGs on an external USB volume. Decoding
them every epoch dominates training time; the crops are small and local.

Models reused as-is from the official pipeline (`predict_fss_head_from_acquisitions`):
vendor CNN for routing, generic rect regressor, plus the vendor-specialised rect
checkpoints from the official map (BK today). Rotation normalisation (OSD) is NOT
applied, consistently with the marker runner on this volume.

Example:
  python3 tools/needle/extract_needle_crops.py \
    --manifest artifacts/90_needle_dataset/v1/manifest_needle.csv \
    --output-dir artifacts/90_needle_dataset/v1/crops \
    --device mps --resume
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]


def _artifacts_root() -> Path:
    """artifacts/ is not committed, so inside a git worktree it lives in the main checkout."""
    local = REPO_ROOT / "artifacts"
    if (local / "10_active_pipeline").is_dir():
        return local
    git_file = REPO_ROOT / ".git"
    if git_file.is_file():
        gitdir = Path(git_file.read_text(encoding="utf-8").split("gitdir:", 1)[1].strip())
        main_root = gitdir.parents[2]  # .git/worktrees/<name> -> repo root
        if (main_root / "artifacts/10_active_pipeline").is_dir():
            return main_root / "artifacts"
    return local


ARTIFACTS_ROOT = _artifacts_root()
PIPE_DIR = ARTIFACTS_ROOT / "10_active_pipeline/pipeline_fss_head"

CROP_FIELDS = [
    "image_path", "rel_path", "group", "leaf_dir", "label", "split", "vendor_hint",
    "vendor_pred", "vendor_conf", "rect_source", "crop_path",
    "rect_x1", "rect_y1", "rect_x2", "rect_y2", "img_w", "img_h",
    "crop_w", "crop_h",
]


def _import_official():
    sys.path.insert(0, str(REPO_ROOT / "tools/ultrasound"))
    spec = importlib.util.spec_from_file_location(
        "predict_fss_head_from_acquisitions",
        REPO_ROOT / "tools/ultrasound/predict_fss_head_from_acquisitions.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class RectInputDataset(Dataset):
    """Loads frames with the official rect preprocessing, in worker processes."""

    def __init__(self, paths: Sequence[Path], image_size: int, loader) -> None:
        self.paths = list(paths)
        self.image_size = int(image_size)
        self.loader = loader

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        try:
            tensor, width, height, ow, oh = self.loader(path, image_size=self.image_size)
            return tensor, index, width, height, ow, oh, 1
        except Exception:
            dummy = torch.zeros(3, self.image_size, self.image_size)
            return dummy, index, 0, 0, 0, 0, 0


def trim_dark_border(
    crop: Image.Image, percentile: float = 98.0, dark_max: float = 10.0, keep_px: int = 2
) -> Tuple[Image.Image, bool]:
    """Drop the uniformly black margins of a RECT_ECHO crop.

    RECT_ECHO is the configured display area, not the active sector, so a zoomed
    or narrow acquisition leaves most of the crop pure black. Keeping it would
    waste most of the resolution budget on padding. Rows/columns are dropped only
    while they are black up to `percentile`, so genuine anechoic tissue (which
    still carries speckle above 0) survives.
    """
    gray = np.asarray(crop.convert("L"), dtype=np.uint8)
    if gray.size == 0:
        return crop, False
    row_level = np.percentile(gray, percentile, axis=1)
    col_level = np.percentile(gray, percentile, axis=0)
    rows = np.flatnonzero(row_level > dark_max)
    cols = np.flatnonzero(col_level > dark_max)
    if rows.size == 0 or cols.size == 0:
        return crop, False

    top = max(0, int(rows[0]) - keep_px)
    bottom = min(gray.shape[0], int(rows[-1]) + 1 + keep_px)
    left = max(0, int(cols[0]) - keep_px)
    right = min(gray.shape[1], int(cols[-1]) + 1 + keep_px)
    if right - left < 16 or bottom - top < 16:
        return crop, False
    return crop.crop((left, top, right, bottom)), True


def _identity_collate(batch):
    """Keep the per-item tuples as-is; the batch is stacked after dropping failed loads."""
    return batch


def crop_filename(rel_path: str) -> str:
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]
    stem = Path(rel_path).stem[:48].replace(" ", "_")
    return f"{digest}_{stem}.png"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--crop-max-side", type=int, default=512,
                        help="Longest side of the cached crop, in pixels.")
    parser.add_argument("--margin-pct", type=float, default=2.0,
                        help="Margin added around the predicted rectangle, in % of its size.")
    parser.add_argument("--trim-dark-border", action="store_true",
                        help="Trim the uniformly black padding of RECT_ECHO. Off by default: the "
                             "grey-ramp strip and the vendor watermark often sit inside the "
                             "rectangle and defeat the row/column test.")
    parser.add_argument("--vendor-sample", type=int, default=24,
                        help="Frames per acquisition used to vote the vendor for rect routing.")
    parser.add_argument("--vendor-checkpoint", type=Path,
                        default=PIPE_DIR / "models/vendor_training_no_negative_v2_power/best_model.pt")
    parser.add_argument("--rect-checkpoint", type=Path,
                        default=PIPE_DIR / "models/rect_training_e40_run2/best_model.pt")
    parser.add_argument("--rect-vendor-map", type=Path,
                        default=PIPE_DIR / "maps/vendor_rect_map_bk_only.json")
    parser.add_argument("--rect-vendor-min-confidence", type=float, default=0.70)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    official = _import_official()
    device = torch.device(args.device)

    rows = list(csv.DictReader(args.manifest.expanduser().resolve().open(encoding="utf-8")))
    if not rows:
        raise SystemExit("manifest vuoto")

    out_dir = args.output_dir.expanduser().resolve()
    crops_dir = out_dir / "images"
    crops_dir.mkdir(parents=True, exist_ok=True)
    crops_manifest = out_dir / "manifest_crops.csv"

    done: Dict[str, Dict[str, str]] = {}
    if args.resume and crops_manifest.is_file() and crops_manifest.stat().st_size:
        for row in csv.DictReader(crops_manifest.open(encoding="utf-8")):
            if (crops_dir / Path(row["crop_path"]).name).is_file():
                done[row["rel_path"]] = row
        print(f"resume: {len(done)} crop gia' presenti")

    # --- vendor vote per acquisition (only for rect routing) -------------------
    vendor_ckpt = torch.load(args.vendor_checkpoint, map_location="cpu", weights_only=False)
    vendor_classes = list(vendor_ckpt.get("class_names") or [])
    vendor_image_size = int(vendor_ckpt.get("args", {}).get("image_size", 384))
    vendor_model = official.VendorClassifier(num_classes=len(vendor_classes), pretrained=False).to(device)
    vendor_model.load_state_dict(vendor_ckpt["model_state_dict"])
    vendor_model.eval()

    by_group: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)

    vendor_of_group: Dict[str, Tuple[str, float]] = {}
    for group, group_rows in sorted(by_group.items()):
        sample = official._select_uniform_subset(
            [Path(r["image_path"]) for r in group_rows], args.vendor_sample
        )
        probs = official._predict_mean_probs(
            vendor_model, sample, vendor_image_size, args.batch_size, device
        )
        best = int(probs.argmax())
        vendor_of_group[group] = (vendor_classes[best], float(probs[best]))
    del vendor_model
    print(f"vendor votato su {len(vendor_of_group)} acquisizioni")

    # --- rect models -----------------------------------------------------------
    def load_rect(path: Path) -> Tuple[torch.nn.Module, int]:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        size = int(ckpt.get("args", {}).get("image_size", 320))
        model = official.RectRegressor(pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, size

    rect_models: Dict[str, Tuple[torch.nn.Module, int]] = {
        "generic": load_rect(args.rect_checkpoint.expanduser().resolve())
    }
    vendor_map: Dict[str, str] = {}
    if args.rect_vendor_map.is_file():
        raw_map = json.loads(args.rect_vendor_map.read_text(encoding="utf-8"))
        for vendor, ckpt_rel in raw_map.items():
            ckpt_path = (args.rect_vendor_map.parent / ckpt_rel).resolve()
            if ckpt_path.is_file():
                rect_models[vendor] = load_rect(ckpt_path)
                vendor_map[vendor] = vendor
    print(f"rect models: {sorted(rect_models)}")

    # route every pending image to the rect model it must use
    pending: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["rel_path"] in done:
            continue
        vendor, conf = vendor_of_group[row["group"]]
        source = "generic"
        if vendor in vendor_map and conf >= args.rect_vendor_min_confidence:
            source = vendor
        row["vendor_pred"] = vendor
        row["vendor_conf"] = f"{conf:.4f}"
        row["rect_source"] = source
        pending[source].append(row)

    total_pending = sum(len(v) for v in pending.values())
    print(f"da elaborare: {total_pending} immagini")

    new_file = not crops_manifest.is_file() or not crops_manifest.stat().st_size
    handle = crops_manifest.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=CROP_FIELDS, extrasaction="ignore")
    if new_file:
        writer.writeheader()

    stats = Counter()
    started = time.time()
    processed = 0

    for source, source_rows in pending.items():
        model, image_size = rect_models[source]
        dataset = RectInputDataset(
            [Path(r["image_path"]) for r in source_rows],
            image_size,
            official._load_rect_image_tensor,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_identity_collate,
        )
        with torch.inference_mode():
            for batch in loader:
                ok = [item for item in batch if item[6] == 1]
                if not ok:
                    stats["load_failed"] += len(batch)
                    continue
                images = torch.stack([item[0] for item in ok]).to(device)
                preds = official.normalize_box_order(model(images)).detach().cpu()

                out_rows: List[Dict[str, object]] = []
                for pred, item in zip(preds, ok):
                    _, index, width, height, ow, oh, _ = item
                    row = source_rows[index]
                    x1 = max(0.0, min(float(pred[0]) * width, width - 1.0))
                    y1 = max(0.0, min(float(pred[1]) * height, height - 1.0))
                    x2 = max(x1 + 1.0, min(float(pred[2]) * width, float(width)))
                    y2 = max(y1 + 1.0, min(float(pred[3]) * height, float(height)))

                    mx = (x2 - x1) * args.margin_pct / 100.0
                    my = (y2 - y1) * args.margin_pct / 100.0
                    cx1 = int(max(0, round(x1 - mx)))
                    cy1 = int(max(0, round(y1 - my)))
                    cx2 = int(min(ow, round(x2 + mx)))
                    cy2 = int(min(oh, round(y2 + my)))
                    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                        stats["degenerate_box"] += 1
                        continue

                    crop_path = crops_dir / crop_filename(row["rel_path"])
                    try:
                        with Image.open(row["image_path"]) as img:
                            crop = img.convert("RGB").crop((cx1, cy1, cx2, cy2))
                        if args.trim_dark_border:
                            crop, trimmed = trim_dark_border(crop)
                            stats["trimmed" if trimmed else "trim_skipped"] += 1
                        crop.thumbnail(
                            (args.crop_max_side, args.crop_max_side), Image.LANCZOS
                        )
                        crop.save(crop_path, format="PNG", optimize=True)
                    except Exception:
                        stats["crop_failed"] += 1
                        continue

                    out_rows.append(
                        {
                            **row,
                            "crop_path": str(crop_path.relative_to(out_dir)),
                            "rect_x1": f"{x1:.1f}", "rect_y1": f"{y1:.1f}",
                            "rect_x2": f"{x2:.1f}", "rect_y2": f"{y2:.1f}",
                            "img_w": ow, "img_h": oh,
                            "crop_w": crop.width, "crop_h": crop.height,
                        }
                    )
                    stats["ok"] += 1

                writer.writerows(out_rows)
                handle.flush()
                processed += len(batch)
                if processed % (args.batch_size * 20) < args.batch_size:
                    rate = processed / max(1e-6, time.time() - started)
                    eta = (total_pending - processed) / max(1e-6, rate)
                    print(
                        f"  {processed}/{total_pending}  {rate:.1f} img/s  ETA {eta/60:.1f} min",
                        flush=True,
                    )

    handle.close()
    print(f"fatto: {dict(stats)}  in {(time.time()-started)/60:.1f} min")
    print(f"manifest crop: {crops_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
