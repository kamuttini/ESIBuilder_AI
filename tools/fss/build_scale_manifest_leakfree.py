#!/usr/bin/env python3
"""Build leak-free scale manifest splits from extracted SCALE_LINE rows.

Input format is expected from:
artifacts/21_scale_line_ssd_esi1_n1/scale_line_per_depth_strict.csv

Leak prevention constraints:
- same setup_id must stay in a single split
- same .fss file content hash must stay in a single split
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class RawRow:
    fss_path: str
    setup_id: str
    depth_index: int
    depth_mm: float
    x1: float
    x2: float
    y1: float
    y2: float
    length_mm: float
    tick_mm: float
    label_side: int


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}
        for item in items:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _safe_float(text: str) -> float:
    return float(text.strip())


def _safe_int(text: str) -> int:
    return int(float(text.strip()))


def _config_folder_from_path(path_str: str) -> str:
    p = Path(path_str)
    if p.parent.name == "DB_setup":
        return p.parent.parent.name
    return p.parent.name


def _read_video_size_from_fss(path_str: str) -> Tuple[int | None, int | None]:
    p = Path(path_str)
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None, None
    if len(lines) < 10:
        return None, None
    try:
        vx = int(lines[8].strip().replace("\r", ""))
        vy = int(lines[9].strip().replace("\r", ""))
        return vx, vy
    except Exception:
        return None, None


def load_rows(input_csv: Path) -> List[RawRow]:
    rows: List[RawRow] = []
    with input_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {
            "fss_path",
            "setup_id",
            "depth_index",
            "depth_mm",
            "x1",
            "x2",
            "y1",
            "y2",
            "length_mm",
            "tick_mm",
            "label_side",
        }
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise RuntimeError(f"Input CSV missing columns: {missing}")

        for row in reader:
            rows.append(
                RawRow(
                    fss_path=row["fss_path"],
                    setup_id=row["setup_id"],
                    depth_index=_safe_int(row["depth_index"]),
                    depth_mm=_safe_float(row["depth_mm"]),
                    x1=_safe_float(row["x1"]),
                    x2=_safe_float(row["x2"]),
                    y1=_safe_float(row["y1"]),
                    y2=_safe_float(row["y2"]),
                    length_mm=_safe_float(row["length_mm"]),
                    tick_mm=_safe_float(row["tick_mm"]),
                    label_side=_safe_int(row["label_side"]),
                )
            )
    return rows


def build_components(rows: Sequence[RawRow]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    fss_paths = sorted({r.fss_path for r in rows})
    uf = UnionFind(fss_paths)

    paths_by_setup: Dict[str, List[str]] = defaultdict(list)
    path_hash: Dict[str, str] = {}
    paths_by_hash: Dict[str, List[str]] = defaultdict(list)

    for path_str in fss_paths:
        setup_values = {r.setup_id for r in rows if r.fss_path == path_str and r.setup_id}
        setup_id = sorted(setup_values)[0] if setup_values else ""
        if setup_id:
            paths_by_setup[setup_id].append(path_str)

        content_hash = hashlib.sha256(Path(path_str).read_bytes()).hexdigest()
        path_hash[path_str] = content_hash
        paths_by_hash[content_hash].append(path_str)

    for group in paths_by_setup.values():
        for i in range(1, len(group)):
            uf.union(group[0], group[i])

    for group in paths_by_hash.values():
        for i in range(1, len(group)):
            uf.union(group[0], group[i])

    components: Dict[str, List[str]] = defaultdict(list)
    for p in fss_paths:
        components[uf.find(p)].append(p)

    return dict(components), path_hash


def assign_component_splits(
    components: Dict[str, List[str]],
    rows: Sequence[RawRow],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, str]:
    path_row_count: Counter[str] = Counter(r.fss_path for r in rows)
    comp_items: List[Tuple[str, List[str], int, str]] = []
    for comp_root, paths in components.items():
        count = sum(path_row_count[p] for p in paths)
        comp_sig = hashlib.sha256(("||".join(sorted(paths))).encode("utf-8")).hexdigest()
        comp_items.append((comp_root, paths, count, comp_sig))

    comp_items.sort(key=lambda x: (-x[2], x[3]))

    total_rows = sum(x[2] for x in comp_items)
    target = {
        "train": total_rows * train_ratio,
        "val": total_rows * val_ratio,
        "test": total_rows * test_ratio,
    }
    current = {"train": 0.0, "val": 0.0, "test": 0.0}
    split_order = ("train", "val", "test")
    comp_to_split: Dict[str, str] = {}

    for comp_root, _paths, row_count, _sig in comp_items:
        best_split = split_order[0]
        best_score = None
        for sp in split_order:
            t = target[sp]
            c = current[sp]
            if t <= 0:
                score = float("inf")
            else:
                score = c / t
            if best_score is None or score < best_score:
                best_score = score
                best_split = sp
        comp_to_split[comp_root] = best_split
        current[best_split] += float(row_count)

    return comp_to_split


def build_manifest(
    rows: Sequence[RawRow],
    components: Dict[str, List[str]],
    comp_to_split: Dict[str, str],
    path_hash: Dict[str, str],
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    path_to_root: Dict[str, str] = {}
    for root, paths in components.items():
        for p in paths:
            path_to_root[p] = root

    video_size_cache: Dict[str, Tuple[int | None, int | None]] = {}
    out_fields = [
        "sample_id",
        "split",
        "fss_group_id",
        "component_id",
        "fss_content_hash",
        "fss_path",
        "config_folder",
        "setup_id",
        "depth_index",
        "depth_mm",
        "x1",
        "x2",
        "y1",
        "y2",
        "video_x_size",
        "video_y_size",
        "x1_norm",
        "x2_norm",
        "y1_norm",
        "y2_norm",
        "length_px",
        "length_mm",
        "tick_mm",
        "label_side",
        "label_side_target",
        "is_inside_frame",
    ]

    out_all = output_dir / "manifest_scale_train_val_test_leakfree.csv"
    out_train = output_dir / "manifest_scale_train_leakfree.csv"
    out_val = output_dir / "manifest_scale_val_leakfree.csv"
    out_test = output_dir / "manifest_scale_test_leakfree.csv"

    fps = {
        "all": out_all.open("w", encoding="utf-8", newline=""),
        "train": out_train.open("w", encoding="utf-8", newline=""),
        "val": out_val.open("w", encoding="utf-8", newline=""),
        "test": out_test.open("w", encoding="utf-8", newline=""),
    }
    try:
        writers = {}
        for k, fh in fps.items():
            w = csv.DictWriter(fh, fieldnames=out_fields)
            w.writeheader()
            writers[k] = w

        row_counts: Counter[str] = Counter()
        group_counts: Counter[str] = Counter()
        side_counts: Counter[str] = Counter()

        unique_paths_by_split: Dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
        setup_split: Dict[str, set[str]] = defaultdict(set)
        hash_split: Dict[str, set[str]] = defaultdict(set)

        for r in rows:
            root = path_to_root[r.fss_path]
            split = comp_to_split[root]
            unique_paths_by_split[split].add(r.fss_path)
            if r.setup_id:
                setup_split[r.setup_id].add(split)
            hash_split[path_hash[r.fss_path]].add(split)

            if r.fss_path not in video_size_cache:
                video_size_cache[r.fss_path] = _read_video_size_from_fss(r.fss_path)
            vx, vy = video_size_cache[r.fss_path]

            x1_norm = (r.x1 / vx) if vx and vx > 0 else None
            x2_norm = (r.x2 / vx) if vx and vx > 0 else None
            y1_norm = (r.y1 / vy) if vy and vy > 0 else None
            y2_norm = (r.y2 / vy) if vy and vy > 0 else None
            length_px = ((r.x2 - r.x1) ** 2 + (r.y2 - r.y1) ** 2) ** 0.5

            label_side_target = 0 if r.label_side == -1 else 1 if r.label_side == 1 else None
            is_inside_frame = None
            if vx and vy:
                is_inside_frame = int(
                    0 <= r.x1 <= (vx - 1)
                    and 0 <= r.x2 <= (vx - 1)
                    and 0 <= r.y1 <= (vy - 1)
                    and 0 <= r.y2 <= (vy - 1)
                )

            out_row = {
                "sample_id": f"{r.setup_id}_{r.depth_index}",
                "split": split,
                "fss_group_id": r.fss_path,
                "component_id": root,
                "fss_content_hash": path_hash[r.fss_path],
                "fss_path": r.fss_path,
                "config_folder": _config_folder_from_path(r.fss_path),
                "setup_id": r.setup_id,
                "depth_index": r.depth_index,
                "depth_mm": r.depth_mm,
                "x1": r.x1,
                "x2": r.x2,
                "y1": r.y1,
                "y2": r.y2,
                "video_x_size": vx,
                "video_y_size": vy,
                "x1_norm": x1_norm,
                "x2_norm": x2_norm,
                "y1_norm": y1_norm,
                "y2_norm": y2_norm,
                "length_px": length_px,
                "length_mm": r.length_mm,
                "tick_mm": r.tick_mm,
                "label_side": r.label_side,
                "label_side_target": label_side_target,
                "is_inside_frame": is_inside_frame,
            }

            writers["all"].writerow(out_row)
            writers[split].writerow(out_row)

            row_counts[split] += 1
            side_counts[str(r.label_side)] += 1

        for split_name, paths in unique_paths_by_split.items():
            group_counts[split_name] = len(paths)

        setup_leak = sum(1 for s in setup_split.values() if len(s) > 1)
        hash_leak = sum(1 for s in hash_split.values() if len(s) > 1)

        summary = {
            "split_policy": "connected_components_on_setup_id_and_fss_content_hash",
            "groups_total": len({r.fss_path for r in rows}),
            "component_total": len(components),
            "groups_per_split": dict(group_counts),
            "rows_per_split": dict(row_counts),
            "rows_total": int(sum(row_counts.values())),
            "label_side_distribution": dict(side_counts),
            "leak_check": {
                "setup_ids_cross_split": int(setup_leak),
                "content_hash_cross_split": int(hash_leak),
            },
            "files": {
                "all": out_all.as_posix(),
                "train": out_train.as_posix(),
                "val": out_val.as_posix(),
                "test": out_test.as_posix(),
            },
        }

        (output_dir / "manifest_summary_leakfree.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary
    finally:
        for fh in fps.values():
            fh.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build leak-free split manifests for SCALE_LINE dataset.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("artifacts/21_scale_line_ssd_esi1_n1/scale_line_per_depth_strict.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/23_scale_manifest_ssd_esi1_n1_leakfree"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise RuntimeError("Ratios must sum to 1.0")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    rows = load_rows(input_csv)
    if not rows:
        raise RuntimeError("Input CSV has no rows")

    components, path_hash = build_components(rows)
    comp_to_split = assign_component_splits(
        components=components,
        rows=rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    summary = build_manifest(
        rows=rows,
        components=components,
        comp_to_split=comp_to_split,
        path_hash=path_hash,
        output_dir=output_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
