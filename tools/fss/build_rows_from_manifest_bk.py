#!/usr/bin/env python3
"""Build rows CSV for GT orientation debug from BK manifest + volume images.

This script scans manifest rows and resolves an image file per row from
`<volume_root>/<config_folder>/image_samples/`, then exports a rows CSV
compatible with build_scale_gt_orientation_debug_html.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

IMG_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

PATTERNS: Tuple[str, ...] = (
    # Explicit orientation files first (official orientation naming)
    "image_depth_find_no_flip_setup_{idx0}",
    "image_depth_find_flip_lr_setup_{idx0}",
    "image_depth_find_flip_ud_setup_{idx0}",
    "image_depth_find_flip_lrud_setup_{idx0}",
    # Fallback patterns
    "image_orientation_setup_{idx0}",
    "image_biplana_setup_{idx0}",
    "image_depth_value_setup_{idx0}",
    "image_CalGrid_depth_{idx0}",
)

ORIENTATION_NAMES: Dict[int, str] = {0: "NF", 1: "LR", 2: "UD", 3: "LRUD"}
IMAGE_PATTERNS: Dict[str, re.Pattern[str]] = {
    "depth_find_nf": re.compile(r"^image_depth_find_no_flip_setup_(\d+)\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$", re.IGNORECASE),
    "depth_find_lr": re.compile(r"^image_depth_find_flip_lr_setup_(\d+)\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$", re.IGNORECASE),
    "depth_find_ud": re.compile(r"^image_depth_find_flip_ud_setup_(\d+)\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$", re.IGNORECASE),
    "depth_find_lrud": re.compile(r"^image_depth_find_flip_lrud_setup_(\d+)\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$", re.IGNORECASE),
}
ORIENTATION_DIR_PATTERNS: Tuple[Tuple[int, re.Pattern[str]], ...] = (
    (3, re.compile(r"\b(lrud|fliplrud|flip\s*lr\s*ud|fliplr\s*ud|lr\s*ud)\b", re.IGNORECASE)),
    (1, re.compile(r"\b(lr|fliplr|flip\s*lr|left\s*right)\b", re.IGNORECASE)),
    (2, re.compile(r"\b(ud|flipud|flip\s*ud|upside\s*down)\b", re.IGNORECASE)),
    (0, re.compile(r"\b(nf|noflip|no\s*flip|flip\s*no)\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    split: str
    config_folder: str
    depth_index: int


@dataclass(frozen=True)
class SelectedRow:
    sample_id: str
    split: str
    config_folder: str
    depth_index: int
    image_path: str
    source_pattern: str
    orientation_source_type: str
    orientation_idx: int
    orientation_name: str


def _read_manifest(path: Path) -> List[ManifestRow]:
    out: List[ManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            sid = str(r.get("sample_id", "")).strip()
            cfg = str(r.get("config_folder", "")).strip()
            split = str(r.get("split", "")).strip().lower()
            if not sid or not cfg:
                continue
            try:
                depth = int(float(str(r.get("depth_index", "0")).strip()))
            except Exception:
                depth = 0
            out.append(ManifestRow(sample_id=sid, split=split, config_folder=cfg, depth_index=depth))
    return out


def _pick_existing(base_no_ext: Path) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = base_no_ext.with_suffix(ext)
        if p.exists() and p.is_file() and not p.name.startswith("._"):
            return p
    return None


def _normalize_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return ascii_value.lower()


def _match_image_name(name: str) -> Tuple[str, Optional[int], Optional[int]]:
    for source_type, regex in IMAGE_PATTERNS.items():
        m = regex.match(name)
        if not m:
            continue
        sample_idx = int(m.group(1))
        if source_type == "depth_find_nf":
            return source_type, 0, sample_idx
        if source_type == "depth_find_lr":
            return source_type, 1, sample_idx
        if source_type == "depth_find_ud":
            return source_type, 2, sample_idx
        if source_type == "depth_find_lrud":
            return source_type, 3, sample_idx
    return "other", None, None


def _infer_orientation_from_component(component: str) -> Optional[int]:
    normalized = re.sub(r"[^a-z0-9]+", " ", _normalize_name(component)).strip()
    if not normalized:
        return None
    for orientation_idx, regex in ORIENTATION_DIR_PATTERNS:
        if regex.search(normalized):
            return orientation_idx
    return None


def _infer_orientation_from_path(path: Path) -> Optional[int]:
    for part in reversed(path.parts[:-1]):
        orientation_idx = _infer_orientation_from_component(part)
        if orientation_idx is not None:
            return orientation_idx
    return None


def _official_orientation_hint(image_path: Path, config_folder: Path) -> Tuple[str, Optional[int], Optional[int]]:
    """Mirror official orientation inference used in orientation pipeline."""
    source_type, orientation_idx, sample_idx = _match_image_name(image_path.name)
    if orientation_idx is None:
        try:
            rel = image_path.relative_to(config_folder)
        except ValueError:
            rel = image_path
        orientation_idx = _infer_orientation_from_path(rel)
        if orientation_idx is not None and source_type == "other":
            source_type = "path_orientation_folder"
    return source_type, orientation_idx, sample_idx


def _resolve_image(volume_root: Path, cfg: str, depth_index: int) -> Tuple[Optional[Path], str]:
    idx0 = max(0, depth_index - 1)
    img_dir = volume_root / cfg / "image_samples"
    if not img_dir.exists():
        return None, "missing_image_samples_dir"
    for pat in PATTERNS:
        stem = pat.format(idx0=idx0)
        p = _pick_existing(img_dir / stem)
        if p is not None:
            return p, pat
    return None, "no_pattern_match"


def _evenly_pick(rows: Sequence[SelectedRow], n: int) -> List[SelectedRow]:
    if n <= 0 or n >= len(rows):
        return list(rows)
    if n == 1:
        return [rows[len(rows) // 2]]
    last = len(rows) - 1
    idxs = []
    for i in range(n):
        pos = round(i * last / (n - 1))
        idxs.append(pos)
    dedup = []
    seen = set()
    for i in idxs:
        if i not in seen:
            dedup.append(i)
            seen.add(i)
    while len(dedup) < n:
        for i in range(len(rows)):
            if i not in seen:
                dedup.append(i)
                seen.add(i)
                if len(dedup) >= n:
                    break
    return [rows[i] for i in dedup]


def _write_rows_csv(path: Path, rows: Sequence[SelectedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "split",
        "image_path",
        "score_after",
        "config_folder",
        "depth_index",
        "source_pattern",
        "orientation_source_type",
        "orientation_idx",
        "orientation_name",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        total = len(rows)
        for i, r in enumerate(rows):
            wr.writerow(
                {
                    "sample_id": r.sample_id,
                    "split": r.split,
                    "image_path": r.image_path,
                    # descending score keeps input order when downstream sorts by score_after desc
                    "score_after": float(total - i),
                    "config_folder": r.config_folder,
                    "depth_index": r.depth_index,
                    "source_pattern": r.source_pattern,
                    "orientation_source_type": r.orientation_source_type,
                    "orientation_idx": r.orientation_idx if r.orientation_idx >= 0 else "",
                    "orientation_name": r.orientation_name,
                }
            )


def _write_config_stats(path: Path, stats: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "config_folder",
        "manifest_rows",
        "resolved_rows",
        "selected_rows",
        "resolved_nf",
        "resolved_lr",
        "resolved_ud",
        "resolved_lrud",
        "resolved_unknown",
        "selected_nf",
        "selected_lr",
        "selected_ud",
        "selected_lrud",
        "selected_unknown",
        "missing_reason",
        "image_samples_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for r in stats:
            wr.writerow({k: r.get(k, "") for k in fields})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build rows CSV from BK manifest across all BK folders.")
    p.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("artifacts/32_scale_image_model_per_vendor_20260416/manifests/manifest_scale_bk.csv"),
    )
    p.add_argument("--volume-root", type=Path, default=Path("/Volumes/SSD_esi1_n1"))
    p.add_argument(
        "--out-rows-csv",
        type=Path,
        default=Path(
            "artifacts/34_scale_bbox_model_bk_20260421/bk_bbox_v2_canonical_segment_volumes/refined_ticks_zero_last_v2_dataset/rows_all_bk_from_manifest.csv"
        ),
    )
    p.add_argument(
        "--out-config-stats-csv",
        type=Path,
        default=Path(
            "artifacts/34_scale_bbox_model_bk_20260421/bk_bbox_v2_canonical_segment_volumes/refined_ticks_zero_last_v2_dataset/rows_all_bk_config_stats.csv"
        ),
    )
    p.add_argument(
        "--out-summary-json",
        type=Path,
        default=Path(
            "artifacts/34_scale_bbox_model_bk_20260421/bk_bbox_v2_canonical_segment_volumes/refined_ticks_zero_last_v2_dataset/rows_all_bk_summary.json"
        ),
    )
    p.add_argument(
        "--per-config-max",
        type=int,
        default=0,
        help="If >0, keep at most this many rows per config (evenly sampled by depth). 0 = keep all resolved rows.",
    )
    p.add_argument(
        "--only-orientation",
        type=str,
        default="ALL",
        choices=("ALL", "NF", "LR", "UD", "LRUD"),
        help="Keep only rows with this orientation label after official orientation inference.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    manifest_csv = args.manifest_csv.expanduser().resolve()
    volume_root = args.volume_root.expanduser().resolve()
    out_rows_csv = args.out_rows_csv.expanduser().resolve()
    out_config_stats_csv = args.out_config_stats_csv.expanduser().resolve()
    out_summary_json = args.out_summary_json.expanduser().resolve()
    per_config_max = int(args.per_config_max)
    only_orientation = str(args.only_orientation or "ALL").upper()

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")
    if not volume_root.exists():
        raise FileNotFoundError(f"Volume root not found: {volume_root}")

    manifest_rows = _read_manifest(manifest_csv)
    by_cfg: Dict[str, List[ManifestRow]] = defaultdict(list)
    for r in manifest_rows:
        by_cfg[r.config_folder].append(r)

    all_selected: List[SelectedRow] = []
    config_stats: List[Dict[str, object]] = []
    selected_orientation_counts: Dict[str, int] = {"NF": 0, "LR": 0, "UD": 0, "LRUD": 0, "UNKNOWN": 0}
    resolved_orientation_counts: Dict[str, int] = {"NF": 0, "LR": 0, "UD": 0, "LRUD": 0, "UNKNOWN": 0}

    for cfg in sorted(by_cfg.keys()):
        rows_cfg = sorted(by_cfg[cfg], key=lambda r: (r.depth_index, r.sample_id))
        resolved: List[SelectedRow] = []
        selected_before_cap: List[SelectedRow] = []
        missing_reason = ""
        img_samples = (volume_root / cfg / "image_samples").as_posix()

        for r in rows_cfg:
            p, src = _resolve_image(volume_root=volume_root, cfg=cfg, depth_index=r.depth_index)
            if p is None:
                if not missing_reason:
                    missing_reason = src
                continue
            orientation_source_type, orientation_idx_opt, _sample_idx = _official_orientation_hint(
                image_path=p,
                config_folder=volume_root / cfg,
            )
            orientation_idx = int(orientation_idx_opt) if orientation_idx_opt is not None else -1
            orientation_name = ORIENTATION_NAMES.get(orientation_idx, "")
            orientation_key = orientation_name if orientation_name else "UNKNOWN"
            resolved_orientation_counts[orientation_key] = int(resolved_orientation_counts.get(orientation_key, 0) + 1)
            resolved.append(
                SelectedRow(
                    sample_id=r.sample_id,
                    split=r.split,
                    config_folder=cfg,
                    depth_index=r.depth_index,
                    image_path=p.as_posix(),
                    source_pattern=src,
                    orientation_source_type=orientation_source_type,
                    orientation_idx=orientation_idx,
                    orientation_name=orientation_name,
                )
            )

        if only_orientation == "ALL":
            selected_before_cap = list(resolved)
        else:
            selected_before_cap = [row for row in resolved if row.orientation_name == only_orientation]
        selected = _evenly_pick(selected_before_cap, per_config_max) if per_config_max > 0 else selected_before_cap
        all_selected.extend(selected)
        for row in selected:
            selected_key = row.orientation_name if row.orientation_name else "UNKNOWN"
            selected_orientation_counts[selected_key] = int(selected_orientation_counts.get(selected_key, 0) + 1)

        def _cnt(rows_list: Sequence[SelectedRow], label: str) -> int:
            return sum(1 for x in rows_list if x.orientation_name == label)

        def _cnt_unknown(rows_list: Sequence[SelectedRow]) -> int:
            return sum(1 for x in rows_list if not x.orientation_name)

        config_stats.append(
            {
                "config_folder": cfg,
                "manifest_rows": len(rows_cfg),
                "resolved_rows": len(resolved),
                "selected_rows": len(selected),
                "resolved_nf": _cnt(resolved, "NF"),
                "resolved_lr": _cnt(resolved, "LR"),
                "resolved_ud": _cnt(resolved, "UD"),
                "resolved_lrud": _cnt(resolved, "LRUD"),
                "resolved_unknown": _cnt_unknown(resolved),
                "selected_nf": _cnt(selected, "NF"),
                "selected_lr": _cnt(selected, "LR"),
                "selected_ud": _cnt(selected, "UD"),
                "selected_lrud": _cnt(selected, "LRUD"),
                "selected_unknown": _cnt_unknown(selected),
                "missing_reason": missing_reason,
                "image_samples_dir": img_samples,
            }
        )

    _write_rows_csv(out_rows_csv, all_selected)
    _write_config_stats(out_config_stats_csv, config_stats)

    summary = {
        "manifest_csv": manifest_csv.as_posix(),
        "volume_root": volume_root.as_posix(),
        "configs_in_manifest": len(by_cfg),
        "rows_in_manifest": len(manifest_rows),
        "selected_rows": len(all_selected),
        "per_config_max": per_config_max,
        "only_orientation": only_orientation,
        "resolved_orientation_counts": resolved_orientation_counts,
        "selected_orientation_counts": selected_orientation_counts,
        "configs_with_zero_selected": sum(1 for r in config_stats if int(r["selected_rows"]) == 0),
        "configs_with_any_selected": sum(1 for r in config_stats if int(r["selected_rows"]) > 0),
        "out_rows_csv": out_rows_csv.as_posix(),
        "out_config_stats_csv": out_config_stats_csv.as_posix(),
    }
    out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    out_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
