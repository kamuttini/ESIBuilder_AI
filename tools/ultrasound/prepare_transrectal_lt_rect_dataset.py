#!/usr/bin/env python3
"""Prepare a transrectal L/T dataset manifest using predicted ultrasound rect crops."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from train_ultrasound_rect_net import (
    RectRegressor,
    _clamp_box,
    box_iou,
    choose_device,
    normalize_box_order,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL_BY_PROBE_TYPE = {3: "L", 4: "T"}
CLASS_TARGET = {"L": 0, "T": 1}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SourceRow:
    image_path_raw: str
    image_path: Path
    split: str
    label_lt: str
    fss_probe_type: int
    fss_id_probe: str
    fss_id_echo: str
    manufacturer: str
    model_name: str
    dataset_folder: str
    fss_path: str
    source_name: str
    gt_left: float
    gt_top: float
    gt_right: float
    gt_bottom: float
    gt_available: bool


@dataclass(frozen=True)
class InferenceSample:
    source: SourceRow
    width: int
    height: int


class RectInferenceDataset(Dataset):
    def __init__(self, rows: Sequence[SourceRow], image_size: int) -> None:
        self.rows = list(rows)
        self.image_size = int(image_size)
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):  # type: ignore[override]
        row = self.rows[idx]
        with Image.open(row.image_path) as img:
            image = img.convert("RGB")
            width, height = image.size
            image = TF.resize(
                image,
                size=[self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.to_tensor(image)
            tensor = (tensor - self.mean) / self.std
        return tensor, InferenceSample(source=row, width=width, height=height)


def collate_infer(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    samples = [b[1] for b in batch]
    return images, samples


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _resolve_image_path(
    raw_path: str,
    dataset_root: Path,
    fallback_dataset_root: Optional[Path],
) -> Path:
    raw = (raw_path or "").strip()
    if not raw:
        return Path("")
    p = Path(raw).expanduser()
    candidates: List[Path] = []
    seen: set[str] = set()

    def _add(path_candidate: Path) -> None:
        key = path_candidate.as_posix()
        if key in seen:
            return
        seen.add(key)
        candidates.append(path_candidate)

    _add(p)
    if "/Dataset/" in raw:
        suffix = raw.split("/Dataset/", 1)[1]
        _add(dataset_root / suffix)
        if fallback_dataset_root is not None:
            _add(fallback_dataset_root / suffix)

    if "/data/Dataset/" in raw:
        suffix = raw.split("/data/Dataset/", 1)[1]
        _add(dataset_root / suffix)
        if fallback_dataset_root is not None:
            _add(fallback_dataset_root / suffix)

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    # Keep deterministic output path even if file is missing.
    if len(candidates) > 1:
        return candidates[1]
    return candidates[0]


def _load_source_rows(
    manifest_path: Path,
    dataset_root: Path,
    fallback_dataset_root: Optional[Path],
    max_samples: int,
    seed: int,
    exclude_keyword_re: Optional[re.Pattern[str]],
) -> Tuple[List[SourceRow], Dict[str, int], List[str]]:
    rows: List[SourceRow] = []
    stats = Counter()
    missing_examples: List[str] = []

    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            stats["rows_seen"] += 1
            probe_type = _safe_int(raw.get("fss_probe_type"), default=0)
            if probe_type not in LABEL_BY_PROBE_TYPE:
                stats["rows_skip_not_transrectal"] += 1
                continue

            if exclude_keyword_re is not None:
                text_blob = " | ".join(
                    [
                        str(raw.get("image_path") or ""),
                        str(raw.get("dataset_folder") or ""),
                        str(raw.get("model_name") or ""),
                        str(raw.get("fss_path") or ""),
                    ]
                )
                if exclude_keyword_re.search(text_blob):
                    stats["rows_skip_negative_proibite"] += 1
                    continue

            image_raw = str(raw.get("image_path") or "").strip()
            image_path = _resolve_image_path(
                raw_path=image_raw,
                dataset_root=dataset_root,
                fallback_dataset_root=fallback_dataset_root,
            )
            if not image_path.exists():
                stats["rows_skip_missing_image"] += 1
                if len(missing_examples) < 10:
                    missing_examples.append(image_path.as_posix())
                continue

            gt_left = _safe_float(raw.get("rect_canonical_left"), default=0.0)
            gt_top = _safe_float(raw.get("rect_canonical_top"), default=0.0)
            gt_right = _safe_float(raw.get("rect_canonical_right"), default=0.0)
            gt_bottom = _safe_float(raw.get("rect_canonical_bottom"), default=0.0)
            gt_available = bool(gt_right > gt_left and gt_bottom > gt_top)

            split = str(raw.get("split") or "").strip().lower()
            if split not in SPLITS:
                split = "train"
                stats["rows_fix_split_to_train"] += 1

            rows.append(
                SourceRow(
                    image_path_raw=image_raw,
                    image_path=image_path,
                    split=split,
                    label_lt=LABEL_BY_PROBE_TYPE[probe_type],
                    fss_probe_type=probe_type,
                    fss_id_probe=str(raw.get("fss_id_probe") or "").strip(),
                    fss_id_echo=str(raw.get("fss_id_echo") or "").strip(),
                    manufacturer=str(raw.get("manufacturer") or "UNKNOWN").strip() or "UNKNOWN",
                    model_name=str(raw.get("model_name") or "").strip(),
                    dataset_folder=str(raw.get("dataset_folder") or "").strip(),
                    fss_path=str(raw.get("fss_path") or "").strip(),
                    source_name=Path(image_path).name,
                    gt_left=gt_left,
                    gt_top=gt_top,
                    gt_right=gt_right,
                    gt_bottom=gt_bottom,
                    gt_available=gt_available,
                )
            )
            stats["rows_keep"] += 1

    if max_samples > 0 and len(rows) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_samples]
        rows.sort(key=lambda r: r.image_path.as_posix())
        stats["rows_subsampled"] = len(rows)
    return rows, dict(stats), missing_examples


def _checkpoint_image_size(checkpoint_path: Path, fallback_size: int) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    value = 0
    if isinstance(args, dict):
        value = _safe_int(args.get("image_size"), default=0)
    return value if value > 0 else int(fallback_size)


def _load_rect_model(checkpoint_path: Path, device: torch.device) -> RectRegressor:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = RectRegressor(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _iou_summary(values: Sequence[float]) -> Dict[str, float]:
    finite = [float(v) for v in values if v == v]
    if not finite:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "ge_050": 0.0,
            "ge_075": 0.0,
            "ge_090": 0.0,
        }
    ordered = sorted(finite)

    def q(prob: float) -> float:
        idx = int(round((len(ordered) - 1) * prob))
        idx = max(0, min(idx, len(ordered) - 1))
        return float(ordered[idx])

    return {
        "mean": float(sum(finite) / len(finite)),
        "median": float(statistics.median(finite)),
        "p10": q(0.10),
        "p25": q(0.25),
        "p75": q(0.75),
        "p90": q(0.90),
        "ge_050": float(sum(1 for v in finite if v >= 0.50) / len(finite)),
        "ge_075": float(sum(1 for v in finite if v >= 0.75) / len(finite)),
        "ge_090": float(sum(1 for v in finite if v >= 0.90) / len(finite)),
    }


def _write_manifest(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_path",
        "image_path_raw",
        "image_folder",
        "source_name",
        "split",
        "label_lt",
        "class_target",
        "label_evidence",
        "fss_probe_type",
        "fss_id_probe",
        "fss_id_echo",
        "manufacturer",
        "model_name",
        "dataset_folder",
        "fss_path",
        "image_width",
        "image_height",
        "gt_left",
        "gt_top",
        "gt_right",
        "gt_bottom",
        "gt_left_norm",
        "gt_top_norm",
        "gt_right_norm",
        "gt_bottom_norm",
        "rect_gt_source",
        "rect_gt_is_ultrasound_bbox",
        "pred_left",
        "pred_top",
        "pred_right",
        "pred_bottom",
        "pred_left_norm",
        "pred_top_norm",
        "pred_right_norm",
        "pred_bottom_norm",
        "iou_pred_vs_gt",
        "iou_pred_vs_orientation_rect",
        "rect_model_checkpoint",
        "rect_model_image_size",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _write_summary(path: Path, summary: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary_txt(path: Path, summary: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append(f"rows_total: {summary.get('rows_total', 0)}")
    lines.append(f"rows_with_gt: {summary.get('rows_with_gt', 0)}")
    lines.append(f"dataset_root: {summary.get('dataset_root', '')}")
    lines.append(f"fallback_dataset_root: {summary.get('fallback_dataset_root', '')}")
    lines.append(f"manifest_source: {summary.get('manifest_source', '')}")
    lines.append("")
    lines.append("rows_by_label:")
    for label, value in sorted((summary.get("rows_by_label") or {}).items()):
        lines.append(f"  - {label}: {value}")
    lines.append("")
    lines.append("rows_by_split_label:")
    rows_by_split_label = summary.get("rows_by_split_label") or {}
    for split in SPLITS:
        row = rows_by_split_label.get(split, {})
        lines.append(f"  - {split}: L={row.get('L', 0)} T={row.get('T', 0)}")
    lines.append("")
    lines.append("iou_overall:")
    iou_overall = summary.get("iou_overall") or {}
    for k in ("mean", "median", "p10", "p25", "p75", "p90", "ge_050", "ge_075", "ge_090"):
        lines.append(f"  - {k}: {iou_overall.get(k, 0.0):.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare transrectal L/T dataset from orientation manifest, predict ultrasound rect, "
            "and export a lightweight manifest for training/QA."
        )
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_line16_dataset/manifest_orientation_rect.csv"),
        help="Input manifest with fss_probe_type and canonical rect GT.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Volumes/SSD_esi1_n1"),
        help="Primary dataset root for resolving legacy /Dataset/ paths.",
    )
    parser.add_argument(
        "--fallback-dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "Dataset",
        help="Fallback dataset root if files are not on the selected volume.",
    )
    parser.add_argument(
        "--rect-checkpoint",
        type=Path,
        default=Path("artifacts/10_active_pipeline/pipeline_fss_head/models/rect_training_e40_run2/best_model.pt"),
        help="Rect detector checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/transrectal_lt_rect_dataset"),
        help="Output folder for manifest and summary files.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0=all rows, >0 random sample for smoke runs.",
    )
    parser.add_argument(
        "--fallback-image-size",
        type=int,
        default=384,
        help="Used only if checkpoint args do not contain image_size.",
    )
    parser.add_argument(
        "--exclude-keyword-regex",
        type=str,
        default=r"(?i)(negative|proibite)",
        help="Exclude rows matching this regex over path/folder/model/fss fields.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    random.seed(args.seed)

    source_manifest = args.source_manifest.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    fallback_dataset_root = args.fallback_dataset_root.expanduser().resolve()
    rect_checkpoint = args.rect_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_manifest.is_file():
        raise FileNotFoundError(f"Source manifest non trovato: {source_manifest}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root non trovato: {dataset_root}")
    if not rect_checkpoint.is_file():
        raise FileNotFoundError(f"Rect checkpoint non trovato: {rect_checkpoint}")

    exclude_keyword_re = re.compile(args.exclude_keyword_regex) if args.exclude_keyword_regex else None

    rows, load_stats, missing_examples = _load_source_rows(
        manifest_path=source_manifest,
        dataset_root=dataset_root,
        fallback_dataset_root=fallback_dataset_root if fallback_dataset_root.exists() else None,
        max_samples=int(args.max_samples),
        seed=int(args.seed),
        exclude_keyword_re=exclude_keyword_re,
    )
    if not rows:
        raise RuntimeError("Nessuna riga valida trovata dopo il filtering transrettale.")

    device = choose_device(args.device)
    image_size = _checkpoint_image_size(rect_checkpoint, fallback_size=int(args.fallback_image_size))
    model = _load_rect_model(rect_checkpoint, device=device)

    dataset = RectInferenceDataset(rows=rows, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_infer,
    )

    output_rows: List[Dict[str, object]] = []
    iou_values: List[float] = []
    iou_values_by_label: Dict[str, List[float]] = defaultdict(list)
    rows_by_split_label: Dict[str, Dict[str, int]] = {
        split: {"L": 0, "T": 0} for split in SPLITS
    }
    rows_by_label = Counter()

    with torch.no_grad():
        for images, samples in loader:
            images = images.to(device, non_blocking=True)
            pred = model(images)
            pred = normalize_box_order(pred).detach().cpu()

            for i, sample in enumerate(samples):
                source = sample.source
                width = int(sample.width)
                height = int(sample.height)

                pred_left_norm = float(pred[i, 0].item())
                pred_top_norm = float(pred[i, 1].item())
                pred_right_norm = float(pred[i, 2].item())
                pred_bottom_norm = float(pred[i, 3].item())

                pred_left, pred_top, pred_right, pred_bottom = _clamp_box(
                    pred_left_norm * width,
                    pred_top_norm * height,
                    pred_right_norm * width,
                    pred_bottom_norm * height,
                    width,
                    height,
                )

                if source.gt_available:
                    gt_left, gt_top, gt_right, gt_bottom = _clamp_box(
                        source.gt_left,
                        source.gt_top,
                        source.gt_right,
                        source.gt_bottom,
                        width,
                        height,
                    )
                    gt_left_norm = gt_left / width
                    gt_top_norm = gt_top / height
                    gt_right_norm = gt_right / width
                    gt_bottom_norm = gt_bottom / height

                    iou_t = box_iou(
                        torch.tensor(
                            [[pred_left_norm, pred_top_norm, pred_right_norm, pred_bottom_norm]],
                            dtype=torch.float32,
                        ),
                        torch.tensor(
                            [[gt_left_norm, gt_top_norm, gt_right_norm, gt_bottom_norm]],
                            dtype=torch.float32,
                        ),
                    )
                    iou = float(iou_t.item())
                    iou_values.append(iou)
                    iou_values_by_label[source.label_lt].append(iou)
                else:
                    gt_left, gt_top, gt_right, gt_bottom = 0.0, 0.0, 0.0, 0.0
                    gt_left_norm, gt_top_norm, gt_right_norm, gt_bottom_norm = 0.0, 0.0, 0.0, 0.0
                    iou = float("nan")

                rows_by_split_label[source.split][source.label_lt] += 1
                rows_by_label[source.label_lt] += 1

                output_rows.append(
                    {
                        "image_path": source.image_path.as_posix(),
                        "image_path_raw": source.image_path_raw,
                        "image_folder": source.image_path.parent.as_posix(),
                        "source_name": source.source_name,
                        "split": source.split,
                        "label_lt": source.label_lt,
                        "class_target": CLASS_TARGET[source.label_lt],
                        "label_evidence": f"fss_probe_type={source.fss_probe_type}",
                        "fss_probe_type": source.fss_probe_type,
                        "fss_id_probe": source.fss_id_probe,
                        "fss_id_echo": source.fss_id_echo,
                        "manufacturer": source.manufacturer,
                        "model_name": source.model_name,
                        "dataset_folder": source.dataset_folder,
                        "fss_path": source.fss_path,
                        "image_width": width,
                        "image_height": height,
                        "gt_left": f"{gt_left:.4f}",
                        "gt_top": f"{gt_top:.4f}",
                        "gt_right": f"{gt_right:.4f}",
                        "gt_bottom": f"{gt_bottom:.4f}",
                        "gt_left_norm": f"{gt_left_norm:.6f}",
                        "gt_top_norm": f"{gt_top_norm:.6f}",
                        "gt_right_norm": f"{gt_right_norm:.6f}",
                        "gt_bottom_norm": f"{gt_bottom_norm:.6f}",
                        "rect_gt_source": "orientation_line16.rect_canonical",
                        "rect_gt_is_ultrasound_bbox": 0,
                        "pred_left": f"{pred_left:.4f}",
                        "pred_top": f"{pred_top:.4f}",
                        "pred_right": f"{pred_right:.4f}",
                        "pred_bottom": f"{pred_bottom:.4f}",
                        "pred_left_norm": f"{pred_left_norm:.6f}",
                        "pred_top_norm": f"{pred_top_norm:.6f}",
                        "pred_right_norm": f"{pred_right_norm:.6f}",
                        "pred_bottom_norm": f"{pred_bottom_norm:.6f}",
                        "iou_pred_vs_gt": "" if iou != iou else f"{iou:.6f}",
                        "iou_pred_vs_orientation_rect": "" if iou != iou else f"{iou:.6f}",
                        "rect_model_checkpoint": rect_checkpoint.as_posix(),
                        "rect_model_image_size": image_size,
                    }
                )

    output_rows.sort(key=lambda r: (str(r["split"]), str(r["label_lt"]), str(r["image_path"])))
    manifest_out = output_dir / "manifest_transrectal_lt_rect.csv"
    summary_json = output_dir / "summary_transrectal_lt_rect.json"
    summary_txt = output_dir / "summary_transrectal_lt_rect.txt"
    _write_manifest(manifest_out, output_rows)

    summary: Dict[str, object] = {
        "rows_total": len(output_rows),
        "rows_with_gt": len(iou_values),
        "rows_by_label": dict(rows_by_label),
        "rows_by_split_label": rows_by_split_label,
        "iou_overall": _iou_summary(iou_values),
        "iou_by_label": {k: _iou_summary(v) for k, v in sorted(iou_values_by_label.items())},
        "load_stats": load_stats,
        "missing_examples": missing_examples,
        "manifest_source": source_manifest.as_posix(),
        "exclude_keyword_regex": args.exclude_keyword_regex,
        "dataset_root": dataset_root.as_posix(),
        "fallback_dataset_root": fallback_dataset_root.as_posix() if fallback_dataset_root.exists() else "",
        "rect_checkpoint": rect_checkpoint.as_posix(),
        "rect_image_size": image_size,
        "rect_gt_source": "orientation_line16.rect_canonical",
        "rect_gt_is_ultrasound_bbox": False,
        "notes": [
            "Rect GT loaded from orientation_line16 canonical fields.",
            "These GT rectangles are not guaranteed to be true ultrasound-box annotations.",
            "Use manual HTML review of predicted rect for crop validation.",
        ],
        "device": str(device),
        "seed": int(args.seed),
        "max_samples": int(args.max_samples),
    }
    _write_summary(summary_json, summary)
    _write_summary_txt(summary_txt, summary)

    print(f"Rows loaded: {len(rows)}", flush=True)
    print(f"Rows output: {len(output_rows)}", flush=True)
    print(f"Manifest: {manifest_out}", flush=True)
    print(f"Summary JSON: {summary_json}", flush=True)
    print(f"Summary TXT: {summary_txt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
