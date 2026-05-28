#!/usr/bin/env python3
"""Select max-zoom images from zoom-workspace export using horizontal segment length.

Input:
- zoom_workspace_interactive_export.json

Output:
- segment_candidates.csv
- max_zoom_by_group.csv
- max_zoom_pair_by_group_name.csv
- selection_summary.json

Rule:
- For each group_key (group+label), pick the segment with the largest length proxy.
- length_proxy = length_px if >0 else length_norm.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class GroupInfo:
    group_key: str
    group_name: str
    label: str
    selected_id: str
    status: str
    comment: str
    image_notes: Dict[str, str]


@dataclass(frozen=True)
class SegmentRow:
    image_id: str
    rel_path: str
    name: str
    x1: float
    x2: float
    y: float
    length_norm: float
    length_px: float
    source: str
    updated_at: str
    group_key: str
    group_name: str
    label: str
    group_status: str
    group_comment: str
    image_note: str
    length_proxy: float


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _norm_text(v: object) -> str:
    return str(v or "").strip()


def _load_export(path: Path) -> Dict[str, object]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError("Export JSON non valido: root deve essere oggetto.")
    return obj


def _load_groups(obj: Dict[str, object]) -> Dict[str, GroupInfo]:
    out: Dict[str, GroupInfo] = {}
    for raw in (obj.get("groups") or []):
        if not isinstance(raw, dict):
            continue
        gk = _norm_text(raw.get("group_key"))
        if not gk:
            continue
        notes_raw = raw.get("image_notes") or {}
        notes: Dict[str, str] = {}
        if isinstance(notes_raw, dict):
            for k, v in notes_raw.items():
                kk = _norm_text(k)
                if kk:
                    notes[kk] = _norm_text(v)
        out[gk] = GroupInfo(
            group_key=gk,
            group_name=_norm_text(raw.get("group_name")),
            label=_norm_text(raw.get("label")).lower(),
            selected_id=_norm_text(raw.get("selected_id")),
            status=_norm_text(raw.get("status")) or "pending",
            comment=_norm_text(raw.get("comment")),
            image_notes=notes,
        )
    return out


def _build_image_to_group(groups: Dict[str, GroupInfo]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for gk, g in groups.items():
        if g.selected_id:
            out.setdefault(g.selected_id, []).append(gk)
        for img_id in g.image_notes.keys():
            out.setdefault(img_id, []).append(gk)
    return out


def _select_unique_group(group_keys: List[str]) -> str:
    uniq = sorted(set([k for k in group_keys if k]))
    if len(uniq) == 1:
        return uniq[0]
    return ""


def _parse_segments(obj: Dict[str, object], groups: Dict[str, GroupInfo]) -> List[SegmentRow]:
    rows: List[SegmentRow] = []
    image_to_group = _build_image_to_group(groups)

    for raw in (obj.get("segments") or []):
        if not isinstance(raw, dict):
            continue
        image_id = _norm_text(raw.get("image_id"))
        if not image_id:
            continue
        rel_path = _norm_text(raw.get("rel_path"))
        name = _norm_text(raw.get("name"))
        x1 = _to_float(raw.get("x1"))
        x2 = _to_float(raw.get("x2"))
        y = _to_float(raw.get("y"))
        length_norm = abs(_to_float(raw.get("length_norm"), abs(x2 - x1)))
        length_px = _to_float(raw.get("length_px"), 0.0)
        source = _norm_text(raw.get("source"))
        updated_at = _norm_text(raw.get("updated_at"))

        gk = _norm_text(raw.get("group_key"))
        if not gk:
            gk = _select_unique_group(image_to_group.get(image_id, []))
        g = groups.get(gk) if gk else None
        image_note = g.image_notes.get(image_id, "") if g else ""

        length_proxy = float(length_px if length_px > 0 else length_norm)

        rows.append(
            SegmentRow(
                image_id=image_id,
                rel_path=rel_path,
                name=name,
                x1=x1,
                x2=x2,
                y=y,
                length_norm=length_norm,
                length_px=length_px,
                source=source,
                updated_at=updated_at,
                group_key=g.group_key if g else "",
                group_name=g.group_name if g else "",
                label=g.label if g else "",
                group_status=g.status if g else "",
                group_comment=g.comment if g else "",
                image_note=image_note,
                length_proxy=length_proxy,
            )
        )

    return rows


def _write_candidates_csv(rows: List[SegmentRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "group_key",
                "group_name",
                "label",
                "image_id",
                "rel_path",
                "name",
                "length_px",
                "length_norm",
                "length_proxy",
                "x1",
                "x2",
                "y",
                "source",
                "updated_at",
                "group_status",
                "group_comment",
                "image_note",
            ]
        )
        for r in sorted(rows, key=lambda x: (x.group_key, -x.length_proxy, x.image_id)):
            w.writerow(
                [
                    r.group_key,
                    r.group_name,
                    r.label,
                    r.image_id,
                    r.rel_path,
                    r.name,
                    f"{r.length_px:.6f}",
                    f"{r.length_norm:.6f}",
                    f"{r.length_proxy:.6f}",
                    f"{r.x1:.6f}",
                    f"{r.x2:.6f}",
                    f"{r.y:.6f}",
                    r.source,
                    r.updated_at,
                    r.group_status,
                    r.group_comment,
                    r.image_note,
                ]
            )


def _pick_max_by_group(rows: List[SegmentRow]) -> Dict[str, SegmentRow]:
    best: Dict[str, SegmentRow] = {}
    for r in rows:
        if not r.group_key:
            continue
        prev = best.get(r.group_key)
        if prev is None or r.length_proxy > prev.length_proxy:
            best[r.group_key] = r
    return best


def _write_best_by_group(best: Dict[str, SegmentRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "group_key",
                "group_name",
                "label",
                "max_zoom_image_id",
                "max_zoom_rel_path",
                "max_zoom_name",
                "segment_length_px",
                "segment_length_norm",
                "segment_length_proxy",
                "x1",
                "x2",
                "y",
                "group_status",
                "group_comment",
                "image_note",
            ]
        )
        for gk in sorted(best.keys()):
            r = best[gk]
            w.writerow(
                [
                    r.group_key,
                    r.group_name,
                    r.label,
                    r.image_id,
                    r.rel_path,
                    r.name,
                    f"{r.length_px:.6f}",
                    f"{r.length_norm:.6f}",
                    f"{r.length_proxy:.6f}",
                    f"{r.x1:.6f}",
                    f"{r.x2:.6f}",
                    f"{r.y:.6f}",
                    r.group_status,
                    r.group_comment,
                    r.image_note,
                ]
            )


def _write_pair_by_group_name(best: Dict[str, SegmentRow], out_csv: Path) -> None:
    by_name: Dict[str, Dict[str, SegmentRow]] = {}
    for r in best.values():
        if not r.group_name:
            continue
        by_name.setdefault(r.group_name, {})[r.label] = r
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "group_name",
                "su_image_id",
                "su_segment_length_px",
                "su_segment_length_norm",
                "giu_image_id",
                "giu_segment_length_px",
                "giu_segment_length_norm",
            ]
        )
        for gname in sorted(by_name.keys()):
            row = by_name[gname]
            su = row.get("su")
            giu = row.get("giu")
            w.writerow(
                [
                    gname,
                    su.image_id if su else "",
                    f"{su.length_px:.6f}" if su else "",
                    f"{su.length_norm:.6f}" if su else "",
                    giu.image_id if giu else "",
                    f"{giu.length_px:.6f}" if giu else "",
                    f"{giu.length_norm:.6f}" if giu else "",
                ]
            )


def run(args: argparse.Namespace) -> int:
    export_json = args.export_json.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not export_json.exists():
        raise FileNotFoundError(f"Export JSON non trovato: {export_json}")

    obj = _load_export(export_json)
    groups = _load_groups(obj)
    seg_rows = _parse_segments(obj, groups)

    assigned = [r for r in seg_rows if r.group_key]
    unassigned = [r for r in seg_rows if not r.group_key]
    best = _pick_max_by_group(assigned)

    candidates_csv = out_dir / "segment_candidates.csv"
    best_csv = out_dir / "max_zoom_by_group.csv"
    pair_csv = out_dir / "max_zoom_pair_by_group_name.csv"
    summary_json = out_dir / "selection_summary.json"

    _write_candidates_csv(seg_rows, candidates_csv)
    _write_best_by_group(best, best_csv)
    _write_pair_by_group_name(best, pair_csv)

    summary = {
        "export_json": export_json.as_posix(),
        "groups_in_export": len(groups),
        "segments_total": len(seg_rows),
        "segments_assigned_to_group": len(assigned),
        "segments_unassigned": len(unassigned),
        "groups_with_max_zoom_selected": len(best),
        "outputs": {
            "segment_candidates_csv": candidates_csv.as_posix(),
            "max_zoom_by_group_csv": best_csv.as_posix(),
            "max_zoom_pair_by_group_name_csv": pair_csv.as_posix(),
        },
        "unassigned_preview": [r.image_id for r in unassigned[:20]],
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Export JSON: {export_json}", flush=True)
    print(f"Segments total: {len(seg_rows)}", flush=True)
    print(f"Assigned: {len(assigned)} | Unassigned: {len(unassigned)}", flush=True)
    print(f"Groups selected: {len(best)}", flush=True)
    print(f"Candidates CSV: {candidates_csv}", flush=True)
    print(f"Max zoom CSV: {best_csv}", flush=True)
    print(f"Pair CSV: {pair_csv}", flush=True)
    print(f"Summary JSON: {summary_json}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Select max-zoom by horizontal segment from zoom workspace export.")
    p.add_argument(
        "--export-json",
        type=Path,
        default=Path("/Users/camilla/Downloads/zoom_workspace_interactive_export.json"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/segment_zoom_selection"),
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

