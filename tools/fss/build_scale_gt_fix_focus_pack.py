#!/usr/bin/env python3
"""Build a focused GT-fix pack for scale-line high-priority rows (P0/P1).

Input:
- artifacts/33_scale_gt_fix_queue/gt_fix_queue_all.csv

Output (default artifacts/34_scale_gt_fix_focus):
- focus_rows.csv
- by_setup.csv
- by_config.csv
- by_fss.csv
- review_batches.md
- summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class FocusRow:
    rank: int
    priority: str
    split: str
    sample_id: str
    config_folder: str
    setup_id: str
    depth_index: int
    score: float
    x_err_px: float
    y_err_px: float
    image_path: str
    fss_path: str
    scale_line_raw: str


def _f(value: str) -> float:
    return float((value or "").strip())


def _i(value: str) -> int:
    return int((value or "").strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build focused P0/P1 GT-fix pack for scale line.")
    p.add_argument(
        "--queue-csv",
        type=Path,
        default=Path("artifacts/33_scale_gt_fix_queue/gt_fix_queue_all.csv"),
        help="Input CSV produced by build_scale_gt_fix_queue.py",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/34_scale_gt_fix_focus"),
        help="Output directory for focus pack files",
    )
    p.add_argument(
        "--priorities",
        nargs="+",
        default=["P0", "P1"],
        help="Priorities to include (default: P0 P1)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="How many setups per review batch in review_batches.md",
    )
    p.add_argument(
        "--max-samples-per-setup",
        type=int,
        default=12,
        help="How many top samples to list per setup in review_batches.md",
    )
    return p.parse_args()


def load_focus_rows(path: Path, priorities: Sequence[str]) -> List[FocusRow]:
    wanted = set(priorities)
    out: List[FocusRow] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for d in reader:
            pr = (d.get("priority") or "").strip()
            if pr not in wanted:
                continue
            try:
                out.append(
                    FocusRow(
                        rank=_i(d.get("rank", "0")),
                        priority=pr,
                        split=(d.get("split") or "").strip(),
                        sample_id=(d.get("sample_id") or "").strip(),
                        config_folder=(d.get("config_folder") or "").strip(),
                        setup_id=(d.get("setup_id") or "").strip(),
                        depth_index=_i(d.get("depth_index", "-1")),
                        score=_f(d.get("score", "0")),
                        x_err_px=_f(d.get("x_err_px", "0")),
                        y_err_px=_f(d.get("y_err_px", "0")),
                        image_path=(d.get("image_path") or "").strip(),
                        fss_path=(d.get("fss_path") or "").strip(),
                        scale_line_raw=(d.get("scale_line_raw") or "").strip(),
                    )
                )
            except Exception:
                continue
    out.sort(key=lambda r: (0 if r.priority == "P0" else 1, -r.score, r.rank))
    return out


def write_focus_rows_csv(path: Path, rows: Sequence[FocusRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "rank",
                "priority",
                "split",
                "sample_id",
                "config_folder",
                "setup_id",
                "depth_index",
                "score",
                "x_err_px",
                "y_err_px",
                "image_path",
                "fss_path",
                "scale_line_raw",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.rank,
                    r.priority,
                    r.split,
                    r.sample_id,
                    r.config_folder,
                    r.setup_id,
                    r.depth_index,
                    f"{r.score:.4f}",
                    f"{r.x_err_px:.4f}",
                    f"{r.y_err_px:.4f}",
                    r.image_path,
                    r.fss_path,
                    r.scale_line_raw,
                ]
            )


def _severity(rows: Iterable[FocusRow]) -> float:
    total = 0.0
    for r in rows:
        # Weight P0 much higher than P1; keep score contribution to preserve ordering.
        base = 1000.0 if r.priority == "P0" else 300.0
        split_bonus = 60.0 if r.split == "test" else 0.0
        total += base + split_bonus + r.score
    return total


def aggregate(
    rows: Sequence[FocusRow],
    key_fn,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, ...], List[FocusRow]] = defaultdict(list)
    for r in rows:
        grouped[key_fn(r)].append(r)

    out: List[Dict[str, object]] = []
    for key, grp in grouped.items():
        p0 = sum(1 for r in grp if r.priority == "P0")
        p1 = sum(1 for r in grp if r.priority == "P1")
        test_n = sum(1 for r in grp if r.split == "test")
        scores = [r.score for r in grp]
        d: Dict[str, object] = {
            "rows": len(grp),
            "p0": p0,
            "p1": p1,
            "test_rows": test_n,
            "mean_score": sum(scores) / max(1, len(scores)),
            "max_score": max(scores) if scores else 0.0,
            "severity": _severity(grp),
        }
        if len(key) == 2:
            d["setup_id"] = key[0]
            d["config_folder"] = key[1]
        elif len(key) == 1:
            d["name"] = key[0]
        out.append(d)

    out.sort(
        key=lambda d: (
            -int(d["p0"]),
            -int(d["p1"]),
            -float(d["severity"]),
            -float(d["mean_score"]),
        )
    )
    return out


def write_dict_csv(path: Path, rows: Sequence[Dict[str, object]], header: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for d in rows:
            record = []
            for h in header:
                v = d.get(h, "")
                if isinstance(v, float):
                    record.append(f"{v:.4f}")
                else:
                    record.append(v)
            w.writerow(record)


def build_batches_md(
    path: Path,
    setup_rows: Sequence[Dict[str, object]],
    focus_rows: Sequence[FocusRow],
    batch_size: int,
    max_samples_per_setup: int,
) -> None:
    samples_by_setup: Dict[Tuple[str, str], List[FocusRow]] = defaultdict(list)
    for r in focus_rows:
        samples_by_setup[(r.setup_id, r.config_folder)].append(r)

    lines: List[str] = []
    lines.append("# Scale GT Fix - Piano Batch P0/P1")
    lines.append("")
    lines.append("Criterio ordinamento: numero P0, poi P1, poi severity aggregata (include peso split test).")
    lines.append("")

    if not setup_rows:
        lines.append("Nessun setup trovato per le priorità richieste.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    for i in range(0, len(setup_rows), max(1, batch_size)):
        batch_n = (i // max(1, batch_size)) + 1
        batch = setup_rows[i : i + max(1, batch_size)]
        lines.append(f"## Batch {batch_n}")
        lines.append("")
        for j, d in enumerate(batch, start=1):
            setup_id = str(d.get("setup_id", ""))
            cfg = str(d.get("config_folder", ""))
            lines.append(
                f"{j}. setup `{setup_id}` | P0={int(d.get('p0',0))} P1={int(d.get('p1',0))} | test={int(d.get('test_rows',0))} | severity={float(d.get('severity',0.0)):.1f}"
            )
            lines.append(f"   config: {cfg}")

            top_samples = samples_by_setup.get((setup_id, cfg), [])[: max(1, max_samples_per_setup)]
            for s in top_samples:
                lines.append(
                    f"   - [{s.priority}] rank={s.rank} score={s.score:.1f} split={s.split} depth={s.depth_index} sample={s.sample_id}"
                )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    queue_csv = args.queue_csv.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not queue_csv.exists():
        raise FileNotFoundError(f"queue CSV not found: {queue_csv}")

    focus_rows = load_focus_rows(queue_csv, args.priorities)

    focus_csv = out_dir / "focus_rows.csv"
    write_focus_rows_csv(focus_csv, focus_rows)

    setup_rows = aggregate(focus_rows, key_fn=lambda r: (r.setup_id, r.config_folder))
    config_rows = aggregate(focus_rows, key_fn=lambda r: (r.config_folder,))
    fss_rows = aggregate(focus_rows, key_fn=lambda r: (r.fss_path,))

    for row in config_rows:
        row["config_folder"] = row.pop("name", "")
    for row in fss_rows:
        row["fss_path"] = row.pop("name", "")

    write_dict_csv(
        out_dir / "by_setup.csv",
        setup_rows,
        ["setup_id", "config_folder", "rows", "p0", "p1", "test_rows", "mean_score", "max_score", "severity"],
    )
    write_dict_csv(
        out_dir / "by_config.csv",
        config_rows,
        ["config_folder", "rows", "p0", "p1", "test_rows", "mean_score", "max_score", "severity"],
    )
    write_dict_csv(
        out_dir / "by_fss.csv",
        fss_rows,
        ["fss_path", "rows", "p0", "p1", "test_rows", "mean_score", "max_score", "severity"],
    )

    build_batches_md(
        out_dir / "review_batches.md",
        setup_rows=setup_rows,
        focus_rows=focus_rows,
        batch_size=args.batch_size,
        max_samples_per_setup=args.max_samples_per_setup,
    )

    by_priority = defaultdict(int)
    by_split = defaultdict(int)
    for r in focus_rows:
        by_priority[r.priority] += 1
        by_split[r.split] += 1

    summary = {
        "input_queue_csv": queue_csv.as_posix(),
        "priorities": list(args.priorities),
        "rows_focus": len(focus_rows),
        "by_priority": dict(sorted(by_priority.items())),
        "by_split": dict(sorted(by_split.items())),
        "setups": len(setup_rows),
        "configs": len(config_rows),
        "fss_files": len(fss_rows),
        "outputs": {
            "focus_rows_csv": focus_csv.as_posix(),
            "by_setup_csv": (out_dir / "by_setup.csv").as_posix(),
            "by_config_csv": (out_dir / "by_config.csv").as_posix(),
            "by_fss_csv": (out_dir / "by_fss.csv").as_posix(),
            "review_batches_md": (out_dir / "review_batches.md").as_posix(),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
