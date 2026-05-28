#!/usr/bin/env python3
"""Infer ultrasound rectangle on a folder and build HTML review package.

Uses the existing ultrasound-rectangle network (train_ultrasound_rect_net.py).
No orientation classification is performed.
The "global" rectangle is computed per acquisition folder (group_name),
never as a single rectangle for the whole dataset.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

# Keep compatibility after moving this script under tools/review_html.
TOOLS_ROOT = Path(__file__).resolve().parents[1]
ULTRASOUND_TOOLS = TOOLS_ROOT / "ultrasound"
if str(ULTRASOUND_TOOLS) not in sys.path:
    sys.path.insert(0, str(ULTRASOUND_TOOLS))
from train_ultrasound_rect_net import RectRegressor, choose_device, normalize_box_order
from train_ultrasound_vendor_classifier import VendorClassifier

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ORIENTATION_NAMES = {"NF", "LR", "UD", "LRUD"}


@dataclass(frozen=True)
class ImageRecord:
    sample_id: str
    image_path: Path
    rel_path: str
    group_name: str
    orientation_name: str


@dataclass(frozen=True)
class PredictionRow:
    sample_id: str
    image_path: str
    rel_path: str
    group_name: str
    orientation_name: str
    vendor_predicted: str
    vendor_top1_prob: float
    vendor_margin_top1_top2: float
    vendor_vote_ratio: float
    rect_route_type: str
    rect_route_vendor: str
    rect_model_checkpoint: str
    rect_model_image_size: int
    width: int
    height: int
    pred_top: float
    pred_left: float
    pred_bottom: float
    pred_right: float
    pred_top_norm: float
    pred_left_norm: float
    pred_bottom_norm: float
    pred_right_norm: float
    pred_area_norm: float
    global_iou: float
    global_top: int
    global_left: int
    global_bottom: int
    global_right: int


class FolderDataset(Dataset):
    def __init__(self, records: Sequence[ImageRecord], image_size: int) -> None:
        self.records = list(records)
        self.image_size = int(image_size)
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):  # type: ignore[override]
        record = self.records[idx]
        try:
            with Image.open(record.image_path) as img:
                image = img.convert("RGB")
                width, height = image.size
                resized = TF.resize(
                    image,
                    size=[self.image_size, self.image_size],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                tensor = TF.to_tensor(resized)
                tensor = (tensor - self.mean) / self.std
            return {
                "ok": True,
                "record": record,
                "width": width,
                "height": height,
                "tensor": tensor,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "record": record,
                "error": str(exc),
            }


def collate_folder(batch):
    ok_items = [item for item in batch if item["ok"]]
    bad_items = [item for item in batch if not item["ok"]]
    if not ok_items:
        return None, bad_items
    images = torch.stack([item["tensor"] for item in ok_items], dim=0)
    meta = [
        (
            item["record"],
            int(item["width"]),
            int(item["height"]),
        )
        for item in ok_items
    ]
    return (images, meta), bad_items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inferenza rettangolo ecografico su cartella immagini con rete esistente "
            "e generazione review HTML. "
            "Il rettangolo globale viene calcolato per cartella acquisizione."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Dataset_orientation/dataset_oriented_strict_20260310_153419"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/10_active_pipeline/pipeline_fss_head/models/rect_training_e40_run2/best_model.pt"),
    )
    parser.add_argument(
        "--rect-vendor-map",
        type=Path,
        default=Path("artifacts/10_active_pipeline/pipeline_fss_head/maps/vendor_rect_map_bk_only.json"),
        help="JSON vendor->checkpoint rect specializzato (es. BK).",
    )
    parser.add_argument(
        "--disable-rect-vendor-routing",
        action="store_true",
        help="Disattiva routing rect vendor-specific e usa sempre --checkpoint.",
    )
    parser.add_argument(
        "--rect-vendor-min-confidence",
        type=float,
        default=0.0,
        help="Soglia confidenza vendor per usare il checkpoint specializzato (default 0.0 = sempre).",
    )
    parser.add_argument(
        "--vendor-checkpoint",
        type=Path,
        default=Path("artifacts/10_active_pipeline/pipeline_fss_head/models/vendor_training_no_negative_v2_power/best_model.pt"),
        help="Checkpoint best_model.pt del classificatore vendor.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/40_outputs_eval/ultrasound_rect_inference_dataset_orientation"),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        help="Override image-size del modello. 0 = letto dal checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--vendor-image-size", type=int, default=0, help="Override image-size vendor. 0=dal checkpoint.")
    parser.add_argument("--vendor-batch-size", type=int, default=48, help="Batch size inferenza vendor.")
    parser.add_argument(
        "--vendor-sample-per-group",
        type=int,
        default=0,
        help="Max immagini per cartella per stimare il vendor (0=tutte).",
    )
    parser.add_argument("--vendor-log-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--inlier-iou-threshold",
        type=float,
        default=0.20,
        help="Soglia IoU per filtrare outlier prima del rettangolo globale finale.",
    )
    parser.add_argument(
        "--emit-quick-review-html",
        action="store_true",
        help=(
            "Genera il report rapido review.html + overlays. "
            "Default: disattivo (usa solo il pacchetto review completo vendor/cartella)."
        ),
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=0,
        help="Debug: limita il numero immagini lette (0=tutte).",
    )
    return parser


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _norm_box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    at, al, ab, ar = a
    bt, bl, bb, br = b
    inter_t = max(at, bt)
    inter_l = max(al, bl)
    inter_b = min(ab, bb)
    inter_r = min(ar, br)
    inter_h = max(0.0, inter_b - inter_t)
    inter_w = max(0.0, inter_r - inter_l)
    inter = inter_h * inter_w
    area_a = max(0.0, ab - at) * max(0.0, ar - al)
    area_b = max(0.0, bb - bt) * max(0.0, br - bl)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def _clamp_norm_box(t: float, l: float, b: float, r: float) -> Tuple[float, float, float, float]:
    t = max(0.0, min(1.0, float(t)))
    l = max(0.0, min(1.0, float(l)))
    b = max(0.0, min(1.0, float(b)))
    r = max(0.0, min(1.0, float(r)))
    if b <= t:
        b = min(1.0, t + 1.0 / 1080.0)
    if r <= l:
        r = min(1.0, l + 1.0 / 1920.0)
    return t, l, b, r


def _to_int_box(norm_box: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    t, l, b, r = norm_box
    top = int(round(max(0.0, min(t * height, height - 1))))
    left = int(round(max(0.0, min(l * width, width - 1))))
    bottom = int(round(max(1.0, min(b * height, float(height)))))
    right = int(round(max(1.0, min(r * width, float(width)))))
    if bottom <= top:
        bottom = min(height, top + 1)
    if right <= left:
        right = min(width, left + 1)
    return top, left, bottom, right


def _find_images(input_dir: Path, limit_images: int) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    idx = 0
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in VALID_EXT:
            continue
        rel = path.relative_to(input_dir).as_posix()
        parts = Path(rel).parts
        group_name = parts[0] if len(parts) >= 1 else ""
        orientation_name = parts[1] if len(parts) >= 2 and parts[1] in ORIENTATION_NAMES else ""
        sample_id = f"S{idx:07d}"
        records.append(
            ImageRecord(
                sample_id=sample_id,
                image_path=path,
                rel_path=rel,
                group_name=group_name,
                orientation_name=orientation_name,
            )
        )
        idx += 1
        if limit_images > 0 and len(records) >= limit_images:
            break
    return records


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _select_subset(paths: Sequence[Path], limit: int) -> List[Path]:
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    if limit == 1:
        return [paths[len(paths) // 2]]
    total = len(paths)
    selected_idx: List[int] = []
    seen: set[int] = set()
    for i in range(limit):
        idx = round(i * (total - 1) / (limit - 1))
        if idx not in seen:
            selected_idx.append(idx)
            seen.add(idx)
    if len(selected_idx) < limit:
        for idx in range(total):
            if idx in seen:
                continue
            selected_idx.append(idx)
            seen.add(idx)
            if len(selected_idx) >= limit:
                break
    selected_idx.sort()
    return [paths[i] for i in selected_idx]


def _load_vendor_image_tensor(path: Path, image_size: int) -> Optional[torch.Tensor]:
    try:
        with Image.open(path) as img:
            image = img.convert("RGB")
            image = TF.resize(
                image,
                size=[image_size, image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.to_tensor(image)
            tensor = TF.normalize(
                tensor,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        return tensor
    except Exception:  # noqa: BLE001
        return None


def _predict_vendor_probabilities(
    model: torch.nn.Module,
    image_paths: Sequence[Path],
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    probs_chunks: List[torch.Tensor] = []
    used_images = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch_tensors = [
                tensor
                for path in batch_paths
                for tensor in [_load_vendor_image_tensor(path, image_size=image_size)]
                if tensor is not None
            ]
            if not batch_tensors:
                continue
            images = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).detach().cpu()
            probs_chunks.append(probs)
            used_images += int(images.shape[0])
    if not probs_chunks:
        return torch.zeros((0, 0), dtype=torch.float32), 0
    return torch.cat(probs_chunks, dim=0), used_images


def _infer_vendor_by_group(
    records: Sequence[ImageRecord],
    checkpoint_path: Path,
    image_size_override: int,
    batch_size: int,
    sample_per_group: int,
    device: torch.device,
    log_interval: int,
) -> Tuple[Dict[str, Dict[str, float | str | int]], Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise RuntimeError("Vendor checkpoint senza class_names.")
    class_names = list(class_names)

    image_size = int(image_size_override) if int(image_size_override) > 0 else int(
        checkpoint.get("args", {}).get("image_size", 320)
    )

    model = VendorClassifier(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    by_group_paths: Dict[str, List[Path]] = {}
    for rec in records:
        by_group_paths.setdefault(rec.group_name, []).append(rec.image_path)

    vendor_map: Dict[str, Dict[str, float | str | int]] = {}
    vendor_dist: Dict[str, int] = {}
    groups_sorted = sorted(by_group_paths.items(), key=lambda kv: kv[0].lower())
    for idx, (group_name, paths) in enumerate(groups_sorted, start=1):
        unique_paths = sorted({p for p in paths})
        selected_paths = _select_subset(unique_paths, limit=sample_per_group)
        probs, used_images = _predict_vendor_probabilities(
            model=model,
            image_paths=selected_paths,
            image_size=image_size,
            batch_size=batch_size,
            device=device,
        )
        if probs.numel() == 0:
            vendor_map[group_name] = {
                "vendor_predicted": "UNKNOWN",
                "vendor_top1_prob": 0.0,
                "vendor_margin_top1_top2": 0.0,
                "vendor_vote_ratio": 0.0,
                "vendor_images_used": int(used_images),
            }
            continue

        mean_probs = probs.mean(dim=0)
        sorted_idx = torch.argsort(mean_probs, descending=True)
        pred_idx = int(sorted_idx[0].item())
        pred_label = str(class_names[pred_idx])
        top1_prob = float(mean_probs[pred_idx].item())
        if len(sorted_idx) > 1:
            top2_prob = float(mean_probs[int(sorted_idx[1].item())].item())
        else:
            top2_prob = 0.0
        margin = float(top1_prob - top2_prob)
        img_vote_idx = torch.argmax(probs, dim=1)
        vote_ratio = float((img_vote_idx == pred_idx).float().mean().item())

        vendor_map[group_name] = {
            "vendor_predicted": pred_label,
            "vendor_top1_prob": top1_prob,
            "vendor_margin_top1_top2": margin,
            "vendor_vote_ratio": vote_ratio,
            "vendor_images_used": int(used_images),
        }
        vendor_dist[pred_label] = vendor_dist.get(pred_label, 0) + 1

        if log_interval > 0 and (idx % log_interval == 0 or idx == len(groups_sorted)):
            print(
                f"[vendor {idx}/{len(groups_sorted)}] group='{group_name}' -> {pred_label} "
                f"(p={top1_prob:.3f}, margin={margin:.3f}, vote={vote_ratio:.3f})",
                flush=True,
            )

    vendor_summary: Dict[str, object] = {
        "checkpoint": checkpoint_path.as_posix(),
        "image_size_model": image_size,
        "groups_total": len(groups_sorted),
        "class_names": class_names,
        "predicted_vendor_distribution": dict(sorted(vendor_dist.items())),
    }
    return vendor_map, vendor_summary


def _resolve_checkpoint_path(raw_path: str, base_dir: Path) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    from_map = (base_dir / candidate).resolve()
    if from_map.exists():
        return from_map
    return (Path.cwd() / candidate).resolve()


def _read_model_image_size(checkpoint_path: Path, override: int, fallback: int = 320) -> int:
    if override > 0:
        return int(override)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return int(ckpt.get("args", {}).get("image_size", fallback))


def _load_rect_vendor_map(
    map_path: Path,
    disable_routing: bool,
) -> Dict[str, Path]:
    if disable_routing:
        return {}
    if not map_path.exists():
        print(
            f"[rect-routing] map non trovata: {map_path} -> uso solo checkpoint globale",
            flush=True,
        )
        return {}
    raw = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Rect vendor map non valida (atteso oggetto vendor->checkpoint): {map_path}")
    out: Dict[str, Path] = {}
    for vendor_raw, ckpt_raw in raw.items():
        vendor = str(vendor_raw).strip()
        if not vendor:
            continue
        ckpt_path = _resolve_checkpoint_path(str(ckpt_raw), base_dir=map_path.parent)
        if not ckpt_path.exists():
            print(
                f"[rect-routing] checkpoint non trovato per vendor '{vendor}': {ckpt_path}",
                flush=True,
            )
            continue
        out[vendor] = ckpt_path
    return out


def _pick_rect_route_by_group(
    group_names: Sequence[str],
    vendor_by_group: Dict[str, Dict[str, float | str | int]],
    global_checkpoint: Path,
    global_image_size: int,
    vendor_rect_map: Dict[str, Path],
    image_size_override: int,
    vendor_min_confidence: float,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, int]]:
    routes: Dict[str, Dict[str, object]] = {}
    route_counts: Dict[str, int] = {}
    vendor_map_ci = {k.lower(): v for k, v in vendor_rect_map.items()}
    ckpt_size_cache: Dict[str, int] = {global_checkpoint.as_posix(): int(global_image_size)}
    for group_name in sorted(set(group_names)):
        vendor_info = vendor_by_group.get(group_name, {})
        vendor_pred = str(vendor_info.get("vendor_predicted", "UNKNOWN")).strip() or "UNKNOWN"
        vendor_conf = float(vendor_info.get("vendor_top1_prob", 0.0))
        specialized_ckpt = vendor_map_ci.get(vendor_pred.lower())

        route_type = "global_no_specialized_vendor"
        route_vendor = ""
        checkpoint = global_checkpoint
        if specialized_ckpt is not None:
            if vendor_conf >= vendor_min_confidence:
                route_type = "vendor_specialized"
                route_vendor = vendor_pred
                checkpoint = specialized_ckpt
            else:
                route_type = "global_low_vendor_conf"
                route_vendor = vendor_pred

        checkpoint_key = checkpoint.as_posix()
        if checkpoint_key not in ckpt_size_cache:
            ckpt_size_cache[checkpoint_key] = _read_model_image_size(
                checkpoint_path=checkpoint,
                override=image_size_override,
                fallback=global_image_size,
            )
        model_image_size = int(ckpt_size_cache[checkpoint_key])
        routes[group_name] = {
            "rect_route_type": route_type,
            "rect_route_vendor": route_vendor,
            "rect_model_checkpoint": checkpoint_key,
            "rect_model_image_size": model_image_size,
        }
        route_counts[route_type] = route_counts.get(route_type, 0) + 1
    return routes, route_counts


def _infer_predictions(
    records: Sequence[ImageRecord],
    checkpoint_path: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    progress_prefix: str = "",
    progress_log_interval_batches: int = 120,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]], int]:
    dataset = FolderDataset(records, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_folder,
        drop_last=False,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = RectRegressor(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    predicted_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, str]] = []
    total_batches = max(1, math.ceil(len(records) / max(1, int(batch_size))))

    with torch.no_grad():
        for batch_idx, (payload, bad_items) in enumerate(loader, start=1):
            for bad in bad_items:
                record = bad["record"]
                skipped_rows.append(
                    {
                        "sample_id": record.sample_id,
                        "image_path": record.image_path.as_posix(),
                        "error": str(bad.get("error", "")),
                    }
                )

            if payload is None:
                continue

            images, meta = payload
            images = images.to(device, non_blocking=True)
            preds = normalize_box_order(model(images)).detach().cpu()

            for i, pred in enumerate(preds):
                record, width, height = meta[i]
                t, l, b, r = _clamp_norm_box(
                    float(pred[1]),  # y1
                    float(pred[0]),  # x1
                    float(pred[3]),  # y2
                    float(pred[2]),  # x2
                )
                top, left, bottom, right = _to_int_box((t, l, b, r), width=width, height=height)
                predicted_rows.append(
                    {
                        "sample_id": record.sample_id,
                        "image_path": record.image_path.as_posix(),
                        "rel_path": record.rel_path,
                        "group_name": record.group_name,
                        "orientation_name": record.orientation_name,
                        "width": width,
                        "height": height,
                        "pred_top": float(top),
                        "pred_left": float(left),
                        "pred_bottom": float(bottom),
                        "pred_right": float(right),
                        "pred_top_norm": t,
                        "pred_left_norm": l,
                        "pred_bottom_norm": b,
                        "pred_right_norm": r,
                    }
                )

            if (
                progress_log_interval_batches > 0
                and (batch_idx % progress_log_interval_batches == 0 or batch_idx == total_batches)
            ):
                prefix = progress_prefix.strip()
                prefix = f"{prefix} " if prefix else ""
                print(
                    f"{prefix}batch {batch_idx}/{total_batches} | predicted={len(predicted_rows)} "
                    f"skipped={len(skipped_rows)}",
                    flush=True,
                )

    return predicted_rows, skipped_rows, int(ckpt.get("args", {}).get("image_size", image_size))


def _infer_predictions_with_routing(
    records: Sequence[ImageRecord],
    rect_routes_by_group: Dict[str, Dict[str, object]],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]], Dict[str, int]]:
    buckets: Dict[Tuple[str, int], List[ImageRecord]] = {}
    route_type_counts_images: Dict[str, int] = {}

    for rec in records:
        route = rect_routes_by_group.get(rec.group_name)
        if route is None:
            raise RuntimeError(f"Routing rettangolo non disponibile per group '{rec.group_name}'.")
        ckpt = str(route.get("rect_model_checkpoint", "")).strip()
        model_size = int(route.get("rect_model_image_size", 320))
        if not ckpt:
            raise RuntimeError(f"Checkpoint rect vuoto per group '{rec.group_name}'.")
        key = (ckpt, model_size)
        buckets.setdefault(key, []).append(rec)
        route_type = str(route.get("rect_route_type", "global_no_specialized_vendor"))
        route_type_counts_images[route_type] = route_type_counts_images.get(route_type, 0) + 1

    predicted_all: List[Dict[str, object]] = []
    skipped_all: List[Dict[str, str]] = []
    total_buckets = len(buckets)
    for idx, ((ckpt_s, model_size), brecords) in enumerate(sorted(buckets.items()), start=1):
        ckpt = Path(ckpt_s)
        print(
            f"[rect-routing {idx}/{total_buckets}] model={ckpt.name} size={model_size} images={len(brecords)}",
            flush=True,
        )
        predicted_rows, skipped_rows, _ = _infer_predictions(
            records=brecords,
            checkpoint_path=ckpt,
            image_size=model_size,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            progress_prefix=f"[rect {idx}/{total_buckets} model={ckpt.name}]",
            progress_log_interval_batches=120,
        )
        predicted_all.extend(predicted_rows)
        skipped_all.extend(skipped_rows)

    for row in predicted_all:
        group_name = str(row.get("group_name", ""))
        route = rect_routes_by_group.get(group_name)
        if route is None:
            raise RuntimeError(f"Routing rettangolo non disponibile per group '{group_name}'.")
        row["rect_route_type"] = str(route.get("rect_route_type", "global_no_specialized_vendor"))
        row["rect_route_vendor"] = str(route.get("rect_route_vendor", ""))
        row["rect_model_checkpoint"] = str(route.get("rect_model_checkpoint", ""))
        row["rect_model_image_size"] = int(route.get("rect_model_image_size", 320))

    predicted_all.sort(key=lambda r: str(r.get("sample_id", "")))
    skipped_all.sort(key=lambda r: str(r.get("sample_id", "")))
    return predicted_all, skipped_all, route_type_counts_images


def _compute_global_rect(
    predicted_rows: Sequence[Dict[str, object]],
    inlier_iou_threshold: float,
) -> Tuple[Tuple[float, float, float, float], int]:
    boxes = [
        (
            float(r["pred_top_norm"]),
            float(r["pred_left_norm"]),
            float(r["pred_bottom_norm"]),
            float(r["pred_right_norm"]),
        )
        for r in predicted_rows
    ]
    if not boxes:
        raise RuntimeError("Nessuna predizione disponibile.")

    initial = _clamp_norm_box(
        _median([b[0] for b in boxes]),
        _median([b[1] for b in boxes]),
        _median([b[2] for b in boxes]),
        _median([b[3] for b in boxes]),
    )

    inliers = [b for b in boxes if _norm_box_iou(b, initial) >= inlier_iou_threshold]
    if len(inliers) < max(32, int(0.1 * len(boxes))):
        inliers = boxes

    final_box = _clamp_norm_box(
        _median([b[0] for b in inliers]),
        _median([b[1] for b in inliers]),
        _median([b[2] for b in inliers]),
        _median([b[3] for b in inliers]),
    )
    return final_box, len(inliers)


def _compute_group_global_rects(
    predicted_rows: Sequence[Dict[str, object]],
    inlier_iou_threshold: float,
) -> Tuple[Dict[str, Tuple[float, float, float, float]], Dict[str, int]]:
    by_group: Dict[str, List[Dict[str, object]]] = {}
    for row in predicted_rows:
        key = str(row["group_name"])
        by_group.setdefault(key, []).append(row)

    rects: Dict[str, Tuple[float, float, float, float]] = {}
    inliers: Dict[str, int] = {}
    for group_name, grows in by_group.items():
        rect, inlier_count = _compute_global_rect(
            predicted_rows=grows,
            inlier_iou_threshold=inlier_iou_threshold,
        )
        rects[group_name] = rect
        inliers[group_name] = inlier_count
    return rects, inliers


def _pick_common_resolution(rows: Sequence[Dict[str, object]]) -> Tuple[int, int]:
    counts: Dict[Tuple[int, int], int] = {}
    for row in rows:
        key = (int(row["width"]), int(row["height"]))
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return 1920, 1080
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _enrich_with_group_global(
    predicted_rows: Sequence[Dict[str, object]],
    group_global_rect_norm: Dict[str, Tuple[float, float, float, float]],
    vendor_by_group: Dict[str, Dict[str, float | str | int]],
) -> List[PredictionRow]:
    out: List[PredictionRow] = []
    for row in predicted_rows:
        width = int(row["width"])
        height = int(row["height"])
        group_name = str(row["group_name"])
        if group_name not in group_global_rect_norm:
            raise RuntimeError(f"Missing per-group global rect for group '{group_name}'.")
        global_rect_norm = group_global_rect_norm[group_name]
        vendor_info = vendor_by_group.get(group_name, {})
        vendor_predicted = str(vendor_info.get("vendor_predicted", "UNKNOWN"))
        vendor_top1_prob = float(vendor_info.get("vendor_top1_prob", 0.0))
        vendor_margin = float(vendor_info.get("vendor_margin_top1_top2", 0.0))
        vendor_vote_ratio = float(vendor_info.get("vendor_vote_ratio", 0.0))
        rect_route_type = str(row.get("rect_route_type", "global_no_specialized_vendor"))
        rect_route_vendor = str(row.get("rect_route_vendor", ""))
        rect_model_checkpoint = str(row.get("rect_model_checkpoint", ""))
        rect_model_image_size = int(row.get("rect_model_image_size", 0))
        global_top, global_left, global_bottom, global_right = _to_int_box(global_rect_norm, width, height)
        pred_box = (
            float(row["pred_top_norm"]),
            float(row["pred_left_norm"]),
            float(row["pred_bottom_norm"]),
            float(row["pred_right_norm"]),
        )
        iou = _norm_box_iou(pred_box, global_rect_norm)
        area = max(0.0, pred_box[2] - pred_box[0]) * max(0.0, pred_box[3] - pred_box[1])
        out.append(
            PredictionRow(
                sample_id=str(row["sample_id"]),
                image_path=str(row["image_path"]),
                rel_path=str(row["rel_path"]),
                group_name=group_name,
                orientation_name=str(row["orientation_name"]),
                vendor_predicted=vendor_predicted,
                vendor_top1_prob=vendor_top1_prob,
                vendor_margin_top1_top2=vendor_margin,
                vendor_vote_ratio=vendor_vote_ratio,
                rect_route_type=rect_route_type,
                rect_route_vendor=rect_route_vendor,
                rect_model_checkpoint=rect_model_checkpoint,
                rect_model_image_size=rect_model_image_size,
                width=width,
                height=height,
                pred_top=float(row["pred_top"]),
                pred_left=float(row["pred_left"]),
                pred_bottom=float(row["pred_bottom"]),
                pred_right=float(row["pred_right"]),
                pred_top_norm=float(row["pred_top_norm"]),
                pred_left_norm=float(row["pred_left_norm"]),
                pred_bottom_norm=float(row["pred_bottom_norm"]),
                pred_right_norm=float(row["pred_right_norm"]),
                pred_area_norm=float(area),
                global_iou=float(iou),
                global_top=int(global_top),
                global_left=int(global_left),
                global_bottom=int(global_bottom),
                global_right=int(global_right),
            )
        )
    return out


def _write_predictions_csv(rows: Sequence[PredictionRow], out_csv: Path) -> None:
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "sample_id",
                "image_path",
                "rel_path",
                "group_name",
                "orientation_name",
                "vendor_predicted",
                "vendor_top1_prob",
                "vendor_margin_top1_top2",
                "vendor_vote_ratio",
                "rect_route_type",
                "rect_route_vendor",
                "rect_model_checkpoint",
                "rect_model_image_size",
                "width",
                "height",
                "pred_top",
                "pred_left",
                "pred_bottom",
                "pred_right",
                "pred_top_norm",
                "pred_left_norm",
                "pred_bottom_norm",
                "pred_right_norm",
                "pred_area_norm",
                "global_iou",
                "global_top",
                "global_left",
                "global_bottom",
                "global_right",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.sample_id,
                    row.image_path,
                    row.rel_path,
                    row.group_name,
                    row.orientation_name,
                    row.vendor_predicted,
                    f"{row.vendor_top1_prob:.6f}",
                    f"{row.vendor_margin_top1_top2:.6f}",
                    f"{row.vendor_vote_ratio:.6f}",
                    row.rect_route_type,
                    row.rect_route_vendor,
                    row.rect_model_checkpoint,
                    row.rect_model_image_size,
                    row.width,
                    row.height,
                    f"{row.pred_top:.3f}",
                    f"{row.pred_left:.3f}",
                    f"{row.pred_bottom:.3f}",
                    f"{row.pred_right:.3f}",
                    f"{row.pred_top_norm:.6f}",
                    f"{row.pred_left_norm:.6f}",
                    f"{row.pred_bottom_norm:.6f}",
                    f"{row.pred_right_norm:.6f}",
                    f"{row.pred_area_norm:.6f}",
                    f"{row.global_iou:.6f}",
                    row.global_top,
                    row.global_left,
                    row.global_bottom,
                    row.global_right,
                ]
            )


def _select_review_rows(
    rows: Sequence[PredictionRow],
    max_images: int,
    per_group: int,
    seed: int,
) -> List[PredictionRow]:
    if not rows or max_images <= 0:
        return []

    groups: Dict[str, List[PredictionRow]] = {}
    for row in rows:
        groups.setdefault(row.group_name, []).append(row)

    selected: Dict[str, PredictionRow] = {}
    k_worst = max(1, per_group // 2)
    k_best = max(1, per_group - k_worst)
    for _, grows in groups.items():
        ordered = sorted(grows, key=lambda r: r.global_iou)
        for row in ordered[:k_worst]:
            selected[row.sample_id] = row
        for row in ordered[-k_best:]:
            selected[row.sample_id] = row

    out = list(selected.values())
    if len(out) > max_images:
        rng = random.Random(seed)
        out = rng.sample(out, k=max_images)
    elif len(out) < max_images and len(out) < len(rows):
        existing = {r.sample_id for r in out}
        remaining = [r for r in rows if r.sample_id not in existing]
        remaining_sorted = sorted(remaining, key=lambda r: r.global_iou)
        take = min(max_images - len(out), len(remaining_sorted))
        if take > 0:
            step = max(1, math.floor(len(remaining_sorted) / take))
            extra = remaining_sorted[::step][:take]
            out.extend(extra)

    return sorted(out, key=lambda r: r.sample_id)


def _overlay_name(row: PredictionRow) -> str:
    safe_rel = row.rel_path.replace("/", "__")
    return f"{row.sample_id}__{safe_rel}"


def _save_overlays(rows: Sequence[PredictionRow], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in rows:
        image_path = Path(row.image_path)
        if not image_path.exists():
            continue
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            draw = ImageDraw.Draw(image)
            pred_box = (row.pred_left, row.pred_top, row.pred_right, row.pred_bottom)
            global_box = (row.global_left, row.global_top, row.global_right, row.global_bottom)
            draw.rectangle(pred_box, outline=(255, 0, 0), width=3)    # red: per-image prediction
            draw.rectangle(global_box, outline=(0, 255, 255), width=3)  # cyan: single global rect
            out_path = output_dir / _overlay_name(row)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(out_path)
            count += 1
        except Exception:  # noqa: BLE001
            continue
    return count


def _build_html(
    review_rows: Sequence[PredictionRow],
    summary: Dict[str, object],
    review_html_path: Path,
) -> None:
    cards: List[str] = []
    for row in review_rows:
        cards.append(
            f"""<article class="card" data-sample-id="{_escape(row.sample_id)}">
  <img loading="lazy" src="overlays/{_escape(_overlay_name(row))}" alt="{_escape(row.sample_id)}">
  <div class="meta">
    <div class="title">{_escape(row.sample_id)}</div>
    <div>iou(pred/global): {row.global_iou:.4f}</div>
    <div>vendor(net): {_escape(row.vendor_predicted)} | p={row.vendor_top1_prob:.3f} | margin={row.vendor_margin_top1_top2:.3f} | vote={row.vendor_vote_ratio:.3f}</div>
    <div>rect route: {_escape(row.rect_route_type)} | route_vendor: {_escape(row.rect_route_vendor or "-")} | model_size={row.rect_model_image_size}</div>
    <div>rect checkpoint: {_escape(Path(row.rect_model_checkpoint).name if row.rect_model_checkpoint else "-")}</div>
    <div>group: {_escape(row.group_name)} | orientation_folder: {_escape(row.orientation_name or "-")}</div>
    <div>rel_path: {_escape(row.rel_path)}</div>
    <div>pred_px: T{int(round(row.pred_top))} L{int(round(row.pred_left))} B{int(round(row.pred_bottom))} R{int(round(row.pred_right))}</div>
    <div>global_px(this image): T{row.global_top} L{row.global_left} B{row.global_bottom} R{row.global_right}</div>
    <div class="decision">
      <label>decision</label>
      <select class="decision-select">
        <option value="">-</option>
        <option value="correct">correct</option>
        <option value="wrong">wrong</option>
      </select>
      <input class="note-input" type="text" placeholder="note">
    </div>
  </div>
</article>"""
        )

    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
    mode = str(summary.get("global_rect_mode", "per_group"))
    group_rects = summary.get("group_rects", {})
    num_groups = int(group_rects.get("num_groups", 0)) if isinstance(group_rects, dict) else 0
    inlier_threshold = float(group_rects.get("inlier_iou_threshold", 0.0)) if isinstance(group_rects, dict) else 0.0
    examples = group_rects.get("examples", []) if isinstance(group_rects, dict) else []
    rect_routing = summary.get("rect_routing", {})
    route_counts_groups = rect_routing.get("counts_by_group", {}) if isinstance(rect_routing, dict) else {}
    route_counts_groups_text = (
        ", ".join([f"{k}={int(v)}" for k, v in sorted(route_counts_groups.items())])
        if isinstance(route_counts_groups, dict) and route_counts_groups
        else "-"
    )
    examples_text = ""
    if isinstance(examples, list) and examples:
        lines = []
        for ex in examples[:8]:
            if not isinstance(ex, dict):
                continue
            g = str(ex.get("group_name", ""))
            rt = ex.get("rect_norm", {})
            if not isinstance(rt, dict):
                continue
            lines.append(
                f"- {g}: T={float(rt.get('top', 0.0)):.6f} "
                f"L={float(rt.get('left', 0.0)):.6f} "
                f"B={float(rt.get('bottom', 0.0)):.6f} "
                f"R={float(rt.get('right', 0.0)):.6f}"
            )
        if lines:
            examples_text = "\n".join(lines)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Ultrasound Rect Review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 14px; background: #11161d; color: #e8ebf0; }}
    h1 {{ margin: 0 0 8px; }}
    .summary {{ background: #1a222e; border: 1px solid #2a3444; border-radius: 8px; padding: 12px; margin-bottom: 12px; }}
    .summary pre {{ margin: 8px 0 0; white-space: pre-wrap; word-break: break-word; font-size: 12px; }}
    .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0 14px; }}
    button {{ background: #2f78ff; color: #fff; border: 0; border-radius: 6px; padding: 8px 10px; cursor: pointer; }}
    button.alt {{ background: #4a5568; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(420px, 1fr)); gap: 12px; }}
    .card {{ background: #1a222e; border: 1px solid #2a3444; border-radius: 8px; overflow: hidden; }}
    .card img {{ width: 100%; display: block; }}
    .meta {{ padding: 10px; font-size: 12px; line-height: 1.4; }}
    .title {{ font-weight: 700; margin-bottom: 4px; }}
    .decision {{ margin-top: 8px; display: grid; grid-template-columns: 54px 110px 1fr; gap: 8px; align-items: center; }}
    .decision select, .decision input {{ background: #0f141b; color: #e8ebf0; border: 1px solid #2d3949; border-radius: 6px; padding: 5px; }}
    @media (max-width: 960px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <h1>Ultrasound Rect Review</h1>
  <div class="summary">
    <div><b>Vendor recognition:</b> neural network per cartella acquisizione</div>
    <div><b>Global rectangle mode:</b> {mode} (per cartella acquisizione/group_name)</div>
    <div><b>Cartelle con rettangolo globale dedicato:</b> {num_groups}</div>
    <div><b>Inlier IoU threshold:</b> {inlier_threshold:.3f}</div>
    <div><b>Rect routing groups:</b> {_escape(route_counts_groups_text)}</div>
    {f'<pre>{_escape(examples_text)}</pre>' if examples_text else ''}
    <details>
      <summary>Summary JSON</summary>
      <pre>{_escape(summary_json)}</pre>
    </details>
  </div>
  <div class="toolbar">
    <button onclick="exportDecisions()">Export decisions.csv</button>
    <button class="alt" onclick="clearDecisions()">Reset decisions</button>
  </div>
  <div class="grid">
    {' '.join(cards)}
  </div>
  <script>
    function rows() {{
      const cards = document.querySelectorAll('.card');
      const out = [];
      cards.forEach((card) => {{
        const sampleId = card.dataset.sampleId || '';
        const decision = card.querySelector('.decision-select').value || '';
        const note = card.querySelector('.note-input').value || '';
        if (decision || note) {{
          out.push([sampleId, decision, note]);
        }}
      }});
      return out;
    }}
    function toCsv(data) {{
      const esc = (s) => '"' + String(s).replaceAll('"', '""') + '"';
      const lines = [['sample_id', 'decision', 'note'], ...data];
      return lines.map((r) => r.map(esc).join(',')).join('\\n');
    }}
    function exportDecisions() {{
      const data = rows();
      const blob = new Blob([toCsv(data)], {{ type: 'text/csv;charset=utf-8;' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'decisions.csv';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}
    function clearDecisions() {{
      document.querySelectorAll('.decision-select').forEach((x) => {{ x.value = ''; }});
      document.querySelectorAll('.note-input').forEach((x) => {{ x.value = ''; }});
    }}
  </script>
</body>
</html>
"""
    review_html_path.write_text(html_doc, encoding="utf-8")


def _stats(values: Iterable[float]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {"count": 0.0, "mean_iou": 0.0, "median_iou": 0.0, "min_iou": 0.0, "max_iou": 0.0}
    ordered = sorted(vals)
    return {
        "count": float(len(vals)),
        "mean_iou": float(sum(vals) / len(vals)),
        "median_iou": float(_median(ordered)),
        "min_iou": float(ordered[0]),
        "max_iou": float(ordered[-1]),
    }


def main() -> int:
    args = build_parser().parse_args()
    random.seed(args.seed)
    if not (0.0 <= float(args.rect_vendor_min_confidence) <= 1.0):
        raise ValueError("--rect-vendor-min-confidence deve essere tra 0 e 1.")

    input_dir = args.input_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    rect_vendor_map_path = args.rect_vendor_map.expanduser().resolve()
    vendor_checkpoint_path = args.vendor_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = output_dir / "predictions.csv"
    summary_json = output_dir / "summary.json"
    review_html = output_dir / "review.html"

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not vendor_checkpoint_path.exists():
        raise FileNotFoundError(f"Vendor checkpoint not found: {vendor_checkpoint_path}")

    ckpt_cpu = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_img_size = int(ckpt_cpu.get("args", {}).get("image_size", 320))
    image_size = int(args.image_size) if int(args.image_size) > 0 else ckpt_img_size

    print(f"Input dir: {input_dir}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Rect vendor map: {rect_vendor_map_path}", flush=True)
    print(f"Vendor checkpoint: {vendor_checkpoint_path}", flush=True)
    print(f"Image size model: {image_size}", flush=True)

    records = _find_images(input_dir=input_dir, limit_images=int(args.limit_images))
    if not records:
        raise RuntimeError(f"No images found under: {input_dir}")
    print(f"Images discovered: {len(records)}", flush=True)

    device = choose_device(args.device)
    print(f"Device: {device}", flush=True)

    vendor_by_group, vendor_summary = _infer_vendor_by_group(
        records=records,
        checkpoint_path=vendor_checkpoint_path,
        image_size_override=int(args.vendor_image_size),
        batch_size=int(args.vendor_batch_size),
        sample_per_group=int(args.vendor_sample_per_group),
        device=device,
        log_interval=int(args.vendor_log_interval),
    )
    print(f"Vendor recognized groups: {len(vendor_by_group)}", flush=True)

    rect_vendor_map = _load_rect_vendor_map(
        map_path=rect_vendor_map_path,
        disable_routing=bool(args.disable_rect_vendor_routing),
    )
    rect_routes_by_group, rect_route_counts_groups = _pick_rect_route_by_group(
        group_names=[r.group_name for r in records],
        vendor_by_group=vendor_by_group,
        global_checkpoint=checkpoint_path,
        global_image_size=image_size,
        vendor_rect_map=rect_vendor_map,
        image_size_override=int(args.image_size),
        vendor_min_confidence=float(args.rect_vendor_min_confidence),
    )
    print(
        "Rect routing groups: "
        + ", ".join([f"{k}={v}" for k, v in sorted(rect_route_counts_groups.items())]),
        flush=True,
    )

    predicted_raw, skipped_rows, rect_route_counts_images = _infer_predictions_with_routing(
        records=records,
        rect_routes_by_group=rect_routes_by_group,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
    )
    if not predicted_raw:
        raise RuntimeError("No valid predictions produced.")
    print(f"Images predicted: {len(predicted_raw)} | skipped: {len(skipped_rows)}", flush=True)

    group_rect_norm, group_inliers = _compute_group_global_rects(
        predicted_rows=predicted_raw,
        inlier_iou_threshold=float(args.inlier_iou_threshold),
    )
    print(f"Per-group global rects computed: {len(group_rect_norm)} groups", flush=True)

    predicted_rows = _enrich_with_group_global(
        predicted_rows=predicted_raw,
        group_global_rect_norm=group_rect_norm,
        vendor_by_group=vendor_by_group,
    )
    _write_predictions_csv(predicted_rows, predictions_csv)

    iou_all = [row.global_iou for row in predicted_rows]
    by_orientation: Dict[str, List[float]] = {}
    for row in predicted_rows:
        key = row.orientation_name if row.orientation_name else "UNKNOWN"
        by_orientation.setdefault(key, []).append(row.global_iou)

    by_group_iou: Dict[str, List[float]] = {}
    by_group_rows: Dict[str, List[Dict[str, object]]] = {}
    by_vendor_iou: Dict[str, List[float]] = {}
    by_rect_route_iou: Dict[str, List[float]] = {}
    for row in predicted_rows:
        by_group_iou.setdefault(row.group_name, []).append(row.global_iou)
        by_vendor_iou.setdefault(row.vendor_predicted, []).append(row.global_iou)
        by_rect_route_iou.setdefault(row.rect_route_type, []).append(row.global_iou)
    for row in predicted_raw:
        key = str(row["group_name"])
        by_group_rows.setdefault(key, []).append(row)

    review_rows: List[PredictionRow] = []
    generated_overlays = 0
    if bool(args.emit_quick_review_html):
        overlays_dir = output_dir / "overlays"
        review_rows = _select_review_rows(
            rows=predicted_rows,
            max_images=1200,
            per_group=4,
            seed=int(args.seed),
        )
        generated_overlays = _save_overlays(review_rows, overlays_dir)

    example_groups = sorted(group_rect_norm.keys())[:8]
    group_examples: List[Dict[str, object]] = []
    for group_name in example_groups:
        t, l, b, r = group_rect_norm[group_name]
        grows = by_group_rows.get(group_name, [])
        common_w, common_h = _pick_common_resolution(grows)
        ct, cl, cb, cr = _to_int_box((t, l, b, r), common_w, common_h)
        group_examples.append(
            {
                "group_name": group_name,
                "num_images": len(grows),
                "inlier_count": int(group_inliers.get(group_name, 0)),
                "rect_norm": {
                    "top": t,
                    "left": l,
                    "bottom": b,
                    "right": r,
                },
                "rect_common_resolution": {
                    "width": common_w,
                    "height": common_h,
                    "top": ct,
                    "left": cl,
                    "bottom": cb,
                    "right": cr,
                },
            }
        )

    group_rect_map_summary: Dict[str, Dict[str, object]] = {}
    for group_name, rect in sorted(group_rect_norm.items()):
        route_info = rect_routes_by_group.get(group_name, {})
        t, l, b, r = rect
        grows = by_group_rows.get(group_name, [])
        common_w, common_h = _pick_common_resolution(grows)
        ct, cl, cb, cr = _to_int_box((t, l, b, r), common_w, common_h)
        group_rect_map_summary[group_name] = {
            "num_images": len(grows),
            "inlier_count": int(group_inliers.get(group_name, 0)),
            "rect_norm": {
                "top": t,
                "left": l,
                "bottom": b,
                "right": r,
            },
            "rect_common_resolution": {
                "width": common_w,
                "height": common_h,
                "top": ct,
                "left": cl,
                "bottom": cb,
                "right": cr,
            },
            "rect_route_type": str(route_info.get("rect_route_type", "")),
            "rect_route_vendor": str(route_info.get("rect_route_vendor", "")),
            "rect_model_checkpoint": str(route_info.get("rect_model_checkpoint", "")),
            "rect_model_image_size": int(route_info.get("rect_model_image_size", 0)),
            "mean_iou_pred_vs_group_global": float(sum(by_group_iou.get(group_name, [0.0])) / max(1, len(by_group_iou.get(group_name, [])))),
        }

    summary: Dict[str, object] = {
        "input_dir": input_dir.as_posix(),
        "checkpoint": checkpoint_path.as_posix(),
        "rect_vendor_map_path": rect_vendor_map_path.as_posix(),
        "rect_vendor_map_loaded": {k: v.as_posix() for k, v in sorted(rect_vendor_map.items())},
        "rect_vendor_min_confidence": float(args.rect_vendor_min_confidence),
        "vendor_checkpoint": vendor_checkpoint_path.as_posix(),
        "device": str(device),
        "image_size_model": image_size,
        "images_requested": len(records),
        "images_predicted": len(predicted_rows),
        "images_skipped": len(skipped_rows),
        "global_rect_mode": "per_group",
        "group_rects": {
            "num_groups": len(group_rect_norm),
            "inlier_iou_threshold": float(args.inlier_iou_threshold),
            "inliers_total": int(sum(group_inliers.values())),
            "examples": group_examples,
        },
        "group_rects_by_group": group_rect_map_summary,
        "global_iou_stats": _stats(iou_all),
        "vendor_recognition": vendor_summary,
        "vendor_stats": {name: _stats(vals) for name, vals in sorted(by_vendor_iou.items())},
        "rect_routing": {
            "disabled": bool(args.disable_rect_vendor_routing),
            "counts_by_group": dict(sorted(rect_route_counts_groups.items())),
            "counts_by_image": dict(sorted(rect_route_counts_images.items())),
        },
        "rect_routing_stats": {name: _stats(vals) for name, vals in sorted(by_rect_route_iou.items())},
        "orientation_stats": {name: _stats(vals) for name, vals in sorted(by_orientation.items())},
        "review": {
            "quick_review_html_enabled": bool(args.emit_quick_review_html),
            "selected_rows": len(review_rows),
            "generated_overlays": generated_overlays,
        },
        "skipped_examples": [row["image_path"] for row in skipped_rows[:20]],
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if bool(args.emit_quick_review_html):
        _build_html(review_rows=review_rows, summary=summary, review_html_path=review_html)
    elif review_html.exists():
        review_html.unlink()

    print(f"Predictions CSV: {predictions_csv}", flush=True)
    print(f"Summary JSON: {summary_json}", flush=True)
    if bool(args.emit_quick_review_html):
        print(f"Review HTML: {review_html}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
