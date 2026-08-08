#!/usr/bin/env python3
"""Read, triage and export the review feedback - the side Claude Code works from.

The UI writes ``feedback/inbox.jsonl``; this reads it. ``triage`` is the entry point of a
session: it groups the open entries by area and tag, names the owning module, and puts the
areas with the most evidence first. ``export`` is what closes the loop: a correction becomes a
row in the format an existing queue already consumes, so a comment ends up in a training set
instead of a backlog.

  python3 tools/review/feedback_cli.py triage
  python3 tools/review/feedback_cli.py list --area depth --status open
  python3 tools/review/feedback_cli.py show fb_20260808_143201_a3
  python3 tools/review/feedback_cli.py resolve fb_... --commit abc1234 --note "gate rifatto"
  python3 tools/review/feedback_cli.py export --target scale_corrections --out /tmp/corr.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from areas import AREA_BY_ID  # noqa: E402
from feedback_store import FeedbackStore  # noqa: E402

EXPORT_TARGETS = ["scale_corrections", "depth_review", "lr_seeds", "labels"]


def cmd_list(store: FeedbackStore, args: argparse.Namespace) -> int:
    filters = {k: getattr(args, k) for k in
               ("status", "area", "run_id", "folder", "image_id", "vendor", "kind", "verdict",
                "tag")}
    entries = store.query(**{k: v for k, v in filters.items() if v})
    if args.json:
        print(json.dumps(entries, ensure_ascii=False, indent=2))
        return 0
    if not entries:
        print("nessun feedback con questi filtri")
        return 0
    for entry in entries:
        target = entry.get("target") or {}
        context = entry.get("context") or {}
        where = " / ".join(str(v) for v in
                          (target.get("folder"), target.get("image_id")) if v)
        print(f"{entry['id']}  {entry['status']:<7} {entry['severity']:<8} "
              f"{entry['area']:<14} {str(entry.get('tag') or '-'):<34} "
              f"{context.get('vendor') or '-':<10} {where}")
        if entry.get("comment"):
            print(f"    {entry['comment']}")
        if entry.get("correction"):
            print(f"    correzione: {json.dumps(entry['correction'], ensure_ascii=False)}")
    print(f"\n{len(entries)} voci")
    return 0


def cmd_show(store: FeedbackStore, args: argparse.Namespace) -> int:
    entry = store.get(args.entry_id)
    if entry is None:
        print(f"non trovato: {args.entry_id}", file=sys.stderr)
        return 1
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    return 0


def cmd_resolve(store: FeedbackStore, args: argparse.Namespace) -> int:
    entry = store.resolve(args.entry_id, status=args.status, commit=args.commit or "",
                          note=args.note or "", by=args.by)
    if entry is None:
        print(f"non trovato: {args.entry_id}", file=sys.stderr)
        return 1
    print(f"{entry['id']} -> {entry['status']}")
    return 0


def cmd_stats(store: FeedbackStore, args: argparse.Namespace) -> int:
    print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_triage(store: FeedbackStore, args: argparse.Namespace) -> int:
    """What to work on, in order, with the evidence behind each line."""
    entries = [e for e in store.all() if str(e.get("status")) in {"open", "ack"}]
    if not entries:
        print("inbox vuota: niente da smaltire")
        return 0

    by_area: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for entry in entries:
        by_area[str(entry.get("area"))].append(entry)

    weight = {"blocking": 100, "major": 10, "minor": 2, "info": 1}
    ranked = sorted(
        by_area.items(),
        key=lambda kv: -sum(weight.get(str(e.get("severity")), 1) for e in kv[1]),
    )

    print(f"{len(entries)} voci aperte in {len(ranked)} aree\n")
    for area_id, items in ranked:
        area = AREA_BY_ID.get(area_id) or {}
        tags = Counter(str(e.get("tag") or "(testo libero)") for e in items)
        vendors = Counter(str((e.get("context") or {}).get("vendor") or "?") for e in items)
        severities = Counter(str(e.get("severity")) for e in items)
        corrections = sum(1 for e in items if e.get("correction"))
        print(f"== {area.get('label', area_id)} [{area_id}] — {len(items)} voci")
        if area.get("module"):
            print(f"   modulo: {area['module']}")
        print(f"   gravità: {dict(severities)}")
        print(f"   vendor:  {dict(vendors.most_common(6))}")
        print(f"   correzioni disponibili: {corrections}"
              + (f" → export {area['export']}" if area.get("export") else ""))
        for tag, count in tags.most_common(8):
            print(f"     {count:>3}x {tag}")
        free = [e for e in items if not e.get("tag") and e.get("comment")]
        for entry in free[:3]:
            print(f"     testo: {entry['comment'][:110]}")
        print()
    return 0


# --------------------------------------------------------------------------- export
def _corrections(store: FeedbackStore, area: str) -> List[Dict[str, object]]:
    return [e for e in store.all() if str(e.get("area")) == area and e.get("correction")]


def _image_path(entry: Dict[str, object]) -> str:
    """The path a training queue should point at: the file on the volume, not the run's symlink.

    The pipeline studies a symlinked copy inside the run dir, and that dir is disposable. The
    resolved original is recorded in the context exactly for this.
    """
    context = entry.get("context") or {}
    target = entry.get("target") or {}
    return str(context.get("image_path_original")  # type: ignore[union-attr]
               or target.get("image_path")  # type: ignore[union-attr]
               or context.get("image_path", ""))  # type: ignore[union-attr]


def _folder_name(entry: Dict[str, object]) -> str:
    target = entry.get("target") or {}
    return str(target.get("folder", ""))  # type: ignore[union-attr]


def _export_scale(store: FeedbackStore, out: Path) -> int:
    """CSV in the shape ``tools/scale/ingest_scale_corrections.py`` already reads.

    Only handles the operator actually placed become ``*_set`` flags: the ingest keeps the GT
    value for the rest, which is what stops a half-correction from overwriting good geometry.
    """
    fields = ["sample_id", "config_folder", "vendor", "depth_index", "image_path",
              "corr_x", "corr_y_zero", "corr_y_far", "zero_at",
              "x_set", "zero_set", "far_set", "added_labels", "flags", "comment"]
    rows: List[Dict[str, object]] = []
    for entry in _corrections(store, "scala"):
        correction = entry.get("correction") or {}
        target = entry.get("target") or {}
        context = entry.get("context") or {}
        prediction = entry.get("prediction") or {}
        y_zero = correction.get("y_zero", prediction.get("y_zero"))
        y_far = correction.get("y_far", prediction.get("y_far"))
        depth_mm = correction.get("depth_mm", prediction.get("depth_mm"))
        labels = ""
        if y_far is not None and depth_mm:
            # the ingest wants ``y_px:cm`` pairs
            labels = f"{float(y_far):.0f}:{float(depth_mm) / 10.0:g}"
        zero_at = ""
        if y_zero is not None and y_far is not None:
            zero_at = "bottom" if float(y_zero) > float(y_far) else "top"
        flags = [str(entry.get("tag") or "").split(".")[-1]] if entry.get("tag") else []
        if str(entry.get("tag")) == "scala.no_ruler":
            flags = ["noruler"]
        rows.append({
            "sample_id": entry.get("id"),
            "config_folder": _folder_name(entry),
            "vendor": context.get("vendor", ""),
            "depth_index": prediction.get("depth_index", ""),
            "image_path": _image_path(entry),
            "corr_x": correction.get("x", ""),
            "corr_y_zero": "" if y_zero is None else y_zero,
            "corr_y_far": "" if y_far is None else y_far,
            "zero_at": zero_at,
            "x_set": "1" if correction.get("x") is not None else "0",
            "zero_set": "1" if correction.get("y_zero") is not None else "0",
            "far_set": "1" if correction.get("y_far") is not None else "0",
            "added_labels": labels,
            "flags": "|".join(f for f in flags if f),
            "comment": entry.get("comment", ""),
        })
    return _write(out, fields, rows)


def _export_depth(store: FeedbackStore, out: Path) -> int:
    """CSV in the shape ``tools/depth/apply_rect_depth_review_export.py`` reads."""
    fields = ["image_path", "corrected_gt_box", "corrected_depth_mm", "predicted_depth_mm",
              "setup_id", "depth_index0", "flip_state", "review_tag", "review_comment",
              "config_folder", "vendor"]
    rows: List[Dict[str, object]] = []
    for entry in _corrections(store, "depth"):
        correction = entry.get("correction") or {}
        target = entry.get("target") or {}
        context = entry.get("context") or {}
        prediction = entry.get("prediction") or {}
        box = correction.get("box") or {}
        box_text = ""
        if box:
            box_text = "|".join(str(int(round(float(box[key]))))
                                for key in ("left", "top", "right", "bottom") if key in box)
        rows.append({
            "image_path": _image_path(entry),
            "corrected_gt_box": box_text,
            "corrected_depth_mm": correction.get("depth_mm", ""),
            "predicted_depth_mm": prediction.get("depth_mm", ""),
            "setup_id": "",
            "depth_index0": "",
            "flip_state": "",
            "review_tag": entry.get("tag", ""),
            "review_comment": entry.get("comment", ""),
            "config_folder": target.get("folder", ""),
            "vendor": context.get("vendor", ""),
        })
    return _write(out, fields, rows)


def _export_lr_seeds(store: FeedbackStore, out: Path) -> int:
    """JSON for ``--lr-marker-manual-seeds-file``: the marker box a human confirmed."""
    seeds: List[Dict[str, object]] = []
    for index, entry in enumerate(_corrections(store, "lr_marker"), start=1):
        correction = entry.get("correction") or {}
        target = entry.get("target") or {}
        context = entry.get("context") or {}
        box = correction.get("box") or {}
        if not box:
            continue
        seeds.append({
            "seed_index": index,
            "image_path": _image_path(entry),
            "rect": ",".join(str(int(round(float(box[key]))))
                             for key in ("left", "top", "right", "bottom") if key in box),
            "vendor": context.get("vendor", ""),
            "folder": target.get("folder", ""),
            "lr_label": correction.get("lr_label", ""),
            "note": entry.get("comment", ""),
            "feedback_id": entry.get("id"),
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": seeds}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(seeds)} seed -> {out}")
    return 0


def _export_labels(store: FeedbackStore, out: Path, area: Optional[str]) -> int:
    """Generic label CSV for the classifier areas (vendor, sonda, su/giù, L/T, rect, #13).

    Deliberately generic: these areas have no single existing queue, so this is the confirmed
    and corrected label plus the path, which is what any dataset-prep script needs, and it is
    also the regression set for the zero-regression rule.
    """
    fields = ["area", "verdict", "tag", "folder", "folder_path", "image_id", "image_path",
              "vendor", "predicted", "corrected", "comment", "severity", "feedback_id", "ts"]
    rows: List[Dict[str, object]] = []
    for entry in store.all():
        entry_area = str(entry.get("area"))
        if area and entry_area != area:
            continue
        if entry_area in {"ui", "orchestratore"}:
            continue
        if not entry.get("correction") and str(entry.get("verdict")) not in {"ok", "wrong"}:
            continue
        target = entry.get("target") or {}
        context = entry.get("context") or {}
        prediction = entry.get("prediction") or {}
        rows.append({
            "area": entry_area,
            "verdict": entry.get("verdict", ""),
            "tag": entry.get("tag", ""),
            "folder": target.get("folder", ""),
            "folder_path": context.get("folder_path_original") or target.get("folder_path", ""),
            "image_id": target.get("image_id", ""),
            "image_path": _image_path(entry),
            "vendor": context.get("vendor", ""),
            "predicted": json.dumps(
                {k: v for k, v in prediction.items() if not isinstance(v, (dict, list))},
                ensure_ascii=False),
            "corrected": json.dumps(entry.get("correction") or {}, ensure_ascii=False),
            "comment": entry.get("comment", ""),
            "severity": entry.get("severity", ""),
            "feedback_id": entry.get("id"),
            "ts": entry.get("ts"),
        })
    return _write(out, fields, rows)


def _write(out: Path, fields: List[str], rows: List[Dict[str, object]]) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"{len(rows)} righe -> {out}")
    return 0


def cmd_export(store: FeedbackStore, args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser()
    if args.target == "scale_corrections":
        return _export_scale(store, out)
    if args.target == "depth_review":
        return _export_depth(store, out)
    if args.target == "lr_seeds":
        return _export_lr_seeds(store, out)
    if args.target == "labels":
        return _export_labels(store, out, args.area)
    print(f"target sconosciuto: {args.target}", file=sys.stderr)
    return 2


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inbox", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="elenca i feedback")
    for name in ("status", "area", "run-id", "folder", "image-id", "vendor", "kind", "verdict",
                 "tag"):
        p_list.add_argument(f"--{name}", default=None)
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="una voce completa")
    p_show.add_argument("entry_id")
    p_show.set_defaults(func=cmd_show)

    p_resolve = sub.add_parser("resolve", help="marca una voce come risolta")
    p_resolve.add_argument("entry_id")
    p_resolve.add_argument("--status", default="done",
                           choices=["open", "ack", "done", "wontfix"])
    p_resolve.add_argument("--commit", default="")
    p_resolve.add_argument("--note", default="")
    p_resolve.add_argument("--by", default="claude")
    p_resolve.set_defaults(func=cmd_resolve)

    p_stats = sub.add_parser("stats", help="conteggi per area, tag, vendor")
    p_stats.set_defaults(func=cmd_stats)

    p_triage = sub.add_parser("triage", help="cosa lavorare, in ordine")
    p_triage.set_defaults(func=cmd_triage)

    p_export = sub.add_parser("export", help="porta le correzioni nelle code dei moduli")
    p_export.add_argument("--target", required=True, choices=EXPORT_TARGETS)
    p_export.add_argument("--out", required=True)
    p_export.add_argument("--area", default=None, help="solo per --target labels")
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    store = FeedbackStore(args.inbox) if args.inbox else FeedbackStore()
    return int(args.func(store, args))


if __name__ == "__main__":
    raise SystemExit(main())
