#!/usr/bin/env python3
"""Audit the scale ground truth taken from the legacy .fss files, per vendor.

Two outputs:

1. ``scale_gt_rows.csv`` - one row per (setup, depth index) with the parsed GT,
   the derived quantities (tick pitch, mm/px, side vs rect) and a ``status`` of
   ``clean`` / ``quarantine`` plus the failed check names.
2. ``scale_gt_profiles.json`` + ``scale_gt_profiles.md`` - per-vendor profile:
   which side the scale lives on, how far from the rect edge, which tick steps
   occur, tick pitch and mm/px ranges, depth ranges, image availability.

The profiles are the priors the per-vendor detectors are built on, so this must
run before any training or tuning.

Usage:
    python3 tools/scale/audit_scale_gt.py \
        --root /Volumes/SSD_esi1_n1 \
        --output-dir artifacts/37_scale_gt_audit_20260729
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scale_common import (  # noqa: E402
    Rect,
    ScaleEntry,
    ScaleSetup,
    SIDE_INSIDE,
    SIDE_LEFT,
    SIDE_RIGHT,
    find_depth_image,
    iter_scale_setups,
    side_vs_rect,
    signed_offset_from_rect,
)

# --------------------------------------------------------------------------- #
# coherence checks
# --------------------------------------------------------------------------- #

# The calibration identity length_px * pixel_ratio_y == depth_mm should hold to
# well under a pixel; we allow 2% to absorb legacy rounding.
CALIB_REL_TOL = 0.02
# A usable tick ladder needs a few pixels between ticks.
MIN_TICK_PITCH_PX = 4.0
# Below this the segment is too short to be a real scale.
MIN_LENGTH_PX = 20.0
# Plausible tick steps in cm seen on clinical scanners.
KNOWN_TICK_CM = (0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0)

EXCLUDE_TOKENS_FUSION = ("fusion",)
EXCLUDE_TOKENS_BAD = (
    "proibite",
    "proibita",
    "proibiti",
    "proibito",
    "prohibited",
    "sbagliat",
    "non usare",
    "non_usare",
    "da rifare",
)


@dataclass
class AuditRow:
    values: Dict[str, object]
    failures: List[str]

    @property
    def status(self) -> str:
        return "quarantine" if self.failures else "clean"


def _folder_flags(config_folder: str) -> Dict[str, int]:
    """Folder-name flags as 0/1.

    Deliberately ints, not bools: a bool round-trips through CSV as ``True``/``False``,
    so every downstream ``== "1"`` test silently fails and filters like ``--drop-fusion``
    become no-ops. All the other flags in this file are already 0/1.
    """
    t = config_folder.lower()
    return {
        "flag_fusion": int(any(tok in t for tok in EXCLUDE_TOKENS_FUSION)),
        "flag_suspect_folder": int(any(tok in t for tok in EXCLUDE_TOKENS_BAD)),
    }


def _pattern_of(stem: str, depth_index: int) -> str:
    from scale_common import IMAGE_STEM_PATTERNS

    for pattern in IMAGE_STEM_PATTERNS:
        if stem == pattern.format(i=depth_index):
            return pattern.replace("{i}", "N")
    return "other"


def _nearest_known_tick(tick_cm: float) -> Optional[float]:
    if tick_cm <= 0:
        return None
    return min(KNOWN_TICK_CM, key=lambda k: abs(k - tick_cm))


def audit_entry(
    setup: ScaleSetup,
    entry: ScaleEntry,
    check_images: bool,
) -> AuditRow:
    rect = setup.rect
    failures: List[str] = []

    # --- structural ---------------------------------------------------------
    if not entry.is_vertical:
        failures.append("not_vertical")
    if entry.length_px < MIN_LENGTH_PX:
        failures.append("too_short")
    if entry.length_mm <= 0:
        failures.append("length_mm_non_positive")
    if entry.tick_cm <= 0:
        failures.append("tick_non_positive")

    # --- inside the video frame --------------------------------------------
    w, h = setup.video_x_size, setup.video_y_size
    in_frame: Optional[bool] = None
    if w and h:
        in_frame = 0 <= entry.x < w and 0 <= entry.y_top < h and 0 < entry.y_bottom <= h
        if not in_frame:
            failures.append("outside_frame")

    # --- calibration identity ----------------------------------------------
    ratio_y = None
    if entry.depth_index < len(setup.pixel_ratio_y):
        ratio_y = setup.pixel_ratio_y[entry.depth_index]
    calib_length_mm = None
    calib_rel_err = None
    if ratio_y and ratio_y > 0 and entry.length_px > 0:
        calib_length_mm = entry.length_px * ratio_y
        calib_rel_err = (
            abs(calib_length_mm - entry.length_mm) / entry.length_mm if entry.length_mm else None
        )
        if calib_rel_err is not None and calib_rel_err > CALIB_REL_TOL:
            failures.append("calibration_mismatch")

    # --- depth vector relation (informative, NOT a quarantine reason) --------
    # VECT_DEPTH is the depth the scanner displayed; length_mm is only how far the
    # operator dragged the calibration line, so length_mm <= vect_depth normally and
    # the gap can be several ticks. Only the impossible direction is a failure.
    vect_depth_mm = None
    vect_minus_length = None
    coverage = None
    if entry.depth_index < len(setup.vect_depth):
        vect_depth_mm = setup.vect_depth[entry.depth_index]
        vect_minus_length = vect_depth_mm - entry.length_mm
        if vect_depth_mm > 0:
            coverage = entry.length_mm / vect_depth_mm
        if vect_minus_length < -1e-6:
            failures.append("length_above_vect_depth")
    else:
        failures.append("depth_index_out_of_vector")

    # --- tick ladder plausibility ------------------------------------------
    pitch = entry.tick_pitch_px
    n_intervals = entry.n_intervals
    if pitch is not None and pitch < MIN_TICK_PITCH_PX:
        failures.append("tick_pitch_too_small")
    if n_intervals is not None and abs(n_intervals - round(n_intervals)) > 0.02:
        failures.append("non_integer_intervals")
    nearest_tick = _nearest_known_tick(entry.tick_cm)
    if nearest_tick is not None and abs(nearest_tick - entry.tick_cm) > 1e-6:
        failures.append("unusual_tick_step")

    # --- geometry vs rect (flags only) --------------------------------------
    # RECT_ECHO is produced by a different block and, on multi-panel screenshots,
    # is sometimes drawn wide enough to swallow the scale column or is plainly
    # unreliable (Siemens, fusion layouts). A bad rect must not disqualify an
    # otherwise perfect scale label, so these are flags, not failures.
    side = side_vs_rect(entry.x, rect)
    offset = signed_offset_from_rect(entry.x, rect)
    y_overlap_frac = None
    if rect is not None and entry.length_px > 0:
        top = max(entry.y_top, rect.y1)
        bot = min(entry.y_bottom, rect.y2)
        y_overlap_frac = max(0.0, bot - top) / entry.length_px

    # --- image availability -------------------------------------------------
    image_path = None
    image_pattern = ""
    if check_images:
        img = find_depth_image(setup.fss_path, entry.depth_index)
        image_path = img.as_posix() if img else ""
        if img:
            image_pattern = _pattern_of(img.stem, entry.depth_index)
        else:
            failures.append("no_image")

    values: Dict[str, object] = {
        "sample_id": f"{setup.config_folder}|{setup.setup_id}|{entry.depth_index}",
        "config_folder": setup.config_folder,
        "vendor": setup.vendor,
        "id_echo": setup.id_echo,
        "id_probe": setup.id_probe,
        "setup_id": setup.setup_id,
        "fss_path": setup.fss_path.as_posix(),
        "depth_index": entry.depth_index,
        "length_mm": entry.length_mm,
        "vect_depth_mm": "" if vect_depth_mm is None else vect_depth_mm,
        "vect_minus_length_mm": "" if vect_minus_length is None else round(vect_minus_length, 4),
        "length_over_depth": "" if coverage is None else round(coverage, 4),
        "tick_cm": entry.tick_cm,
        "side_raw": entry.side_raw,
        "x": entry.x,
        "y_zero": entry.y_zero,
        "y_far": entry.y_far,
        "zero_at": entry.zero_at,
        "y_top": entry.y_top,
        "y_bottom": entry.y_bottom,
        "length_px": entry.length_px,
        "n_intervals": "" if n_intervals is None else round(n_intervals, 4),
        "tick_pitch_px": "" if pitch is None else round(pitch, 3),
        "mm_per_px_from_scale": "" if entry.mm_per_px is None else round(entry.mm_per_px, 6),
        "pixel_ratio_y": "" if ratio_y is None else ratio_y,
        "calib_length_mm": "" if calib_length_mm is None else round(calib_length_mm, 4),
        "calib_rel_err": "" if calib_rel_err is None else round(calib_rel_err, 5),
        "video_x_size": setup.video_x_size or "",
        "video_y_size": setup.video_y_size or "",
        "rect_x1": "" if rect is None else rect.x1,
        "rect_y1": "" if rect is None else rect.y1,
        "rect_x2": "" if rect is None else rect.x2,
        "rect_y2": "" if rect is None else rect.y2,
        "side_vs_rect": side,
        "offset_from_rect_px": "" if offset is None else round(offset, 2),
        "offset_from_rect_frac_w": (
            "" if offset is None or not setup.video_x_size else round(offset / setup.video_x_size, 5)
        ),
        "y_overlap_frac": "" if y_overlap_frac is None else round(y_overlap_frac, 4),
        "in_frame": "" if in_frame is None else int(in_frame),
        "image_path": "" if image_path is None else image_path,
        "image_pattern": image_pattern,
        "flag_no_rect": int(rect is None),
        "flag_y_range_off_rect": int(y_overlap_frac is not None and y_overlap_frac < 0.5),
        "flag_scale_inside_rect": int(side == SIDE_INSIDE),
    }
    values.update(_folder_flags(setup.config_folder))
    return AuditRow(values=values, failures=failures)


# --------------------------------------------------------------------------- #
# per-vendor profiles
# --------------------------------------------------------------------------- #


def _stats(vals: Sequence[float]) -> Dict[str, object]:
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return {"n": 0}
    s = sorted(vals)

    def q(p: float) -> float:
        if len(s) == 1:
            return s[0]
        idx = p * (len(s) - 1)
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    return {
        "n": len(s),
        "min": round(s[0], 3),
        "p05": round(q(0.05), 3),
        "median": round(statistics.median(s), 3),
        "p95": round(q(0.95), 3),
        "max": round(s[-1], 3),
    }


def build_profiles(rows: Sequence[AuditRow]) -> Dict[str, object]:
    by_vendor: Dict[str, List[AuditRow]] = defaultdict(list)
    for r in rows:
        by_vendor[str(r.values["vendor"])].append(r)

    profiles: Dict[str, object] = {}
    for vendor, vrows in sorted(by_vendor.items(), key=lambda kv: -len(kv[1])):
        clean = [r for r in vrows if r.status == "clean"]
        folders = {str(r.values["config_folder"]) for r in vrows}

        def col(rs: Sequence[AuditRow], key: str) -> List[float]:
            out = []
            for r in rs:
                v = r.values.get(key)
                if v == "" or v is None:
                    continue
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    continue
            return out

        side_counts = Counter(str(r.values["side_vs_rect"]) for r in clean)
        base = clean or vrows
        profiles[vendor] = {
            "rows_total": len(vrows),
            "rows_clean": len(clean),
            "rows_quarantine": len(vrows) - len(clean),
            "clean_ratio": round(len(clean) / len(vrows), 4) if vrows else 0.0,
            "config_folders": len(folders),
            "failure_counts": dict(Counter(f for r in vrows for f in r.failures).most_common()),
            "side_vs_rect": dict(side_counts.most_common()),
            "zero_at": dict(Counter(str(r.values["zero_at"]) for r in base).most_common()),
            "side_raw_values": dict(Counter(str(r.values["side_raw"]) for r in base).most_common()),
            "tick_cm_counts": dict(Counter(str(r.values["tick_cm"]) for r in base).most_common()),
            "offset_from_rect_px": _stats(col(base, "offset_from_rect_px")),
            "offset_from_rect_frac_w": _stats(col(base, "offset_from_rect_frac_w")),
            "tick_pitch_px": _stats(col(base, "tick_pitch_px")),
            "mm_per_px": _stats(col(base, "mm_per_px_from_scale")),
            "length_mm": _stats(col(base, "length_mm")),
            "vect_depth_mm": _stats(col(base, "vect_depth_mm")),
            "length_over_depth": _stats(col(base, "length_over_depth")),
            "length_px": _stats(col(base, "length_px")),
            "y_overlap_frac": _stats(col(base, "y_overlap_frac")),
            "video_sizes": dict(
                Counter(
                    f"{r.values['video_x_size']}x{r.values['video_y_size']}" for r in base
                ).most_common(6)
            ),
            "images_resolved": sum(1 for r in vrows if r.values.get("image_path")),
            "image_patterns": dict(
                Counter(str(r.values.get("image_pattern", "")) for r in vrows).most_common()
            ),
            "flags": {
                "no_rect": sum(int(r.values.get("flag_no_rect") or 0) for r in vrows),
                "y_range_off_rect": sum(
                    int(r.values.get("flag_y_range_off_rect") or 0) for r in vrows
                ),
                "scale_inside_rect": sum(
                    int(r.values.get("flag_scale_inside_rect") or 0) for r in vrows
                ),
                "fusion": sum(int(r.values.get("flag_fusion") or 0) for r in vrows),
                "suspect_folder": sum(int(r.values.get("flag_suspect_folder") or 0) for r in vrows),
            },
        }
    return profiles


def render_markdown(profiles: Dict[str, object], totals: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Audit GT scala (riga 21 `.fss`) per vendor")
    lines.append("")
    lines.append(
        f"Righe totali **{totals['rows_total']}** su **{totals['config_folders']}** cartelle di "
        f"configurazione, **{totals['rows_clean']}** clean "
        f"({totals['clean_ratio_pct']}%), **{totals['rows_quarantine']}** in quarantena."
    )
    lines.append("")
    lines.append("## Sintesi per vendor")
    lines.append("")
    lines.append(
        "| Vendor | Cartelle | Righe | Clean | Lato prevalente | Offset dal rect (px, mediana) "
        "| Passo tacche (px, mediana) | Tick cm prevalente | Immagini risolte |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for vendor, p in profiles.items():
        side = next(iter(p["side_vs_rect"].items()), ("n/d", 0))
        tick = next(iter(p["tick_cm_counts"].items()), ("n/d", 0))
        lines.append(
            f"| {vendor} | {p['config_folders']} | {p['rows_total']} | "
            f"{p['rows_clean']} ({round(100 * p['clean_ratio'], 1)}%) | "
            f"{side[0]} ({side[1]}) | {p['offset_from_rect_px'].get('median', 'n/d')} | "
            f"{p['tick_pitch_px'].get('median', 'n/d')} | {tick[0]} ({tick[1]}) | "
            f"{p['images_resolved']} |"
        )
    lines.append("")
    lines.append("## Motivi di quarantena (tutti i vendor)")
    lines.append("")
    for reason, n in totals["failure_counts"].items():
        lines.append(f"- `{reason}`: {n}")
    lines.append("")
    lines.append("## Dettaglio per vendor")
    for vendor, p in profiles.items():
        lines.append("")
        lines.append(f"### {vendor}")
        lines.append("")
        lines.append(f"- lato vs rect: `{p['side_vs_rect']}`")
        lines.append(f"- estremo che porta lo zero (`y1`): `{p['zero_at']}`")
        lines.append(f"- campo `side` in riga 21: `{p['side_raw_values']}`")
        lines.append(f"- passo tacche dichiarato (cm): `{p['tick_cm_counts']}`")
        lines.append(f"- offset dal bordo rect (px): `{p['offset_from_rect_px']}`")
        lines.append(f"- offset dal bordo rect (frazione larghezza): `{p['offset_from_rect_frac_w']}`")
        lines.append(f"- passo tacche (px): `{p['tick_pitch_px']}`")
        lines.append(f"- mm per pixel: `{p['mm_per_px']}`")
        lines.append(f"- lunghezza dichiarata del segmento (mm): `{p['length_mm']}`")
        lines.append(f"- depth visualizzata (mm): `{p['vect_depth_mm']}`")
        lines.append(
            f"- copertura `length_mm / vect_depth` (quanto l'operatore tirava la linea): "
            f"`{p['length_over_depth']}`"
        )
        lines.append(f"- lunghezza segmento (px): `{p['length_px']}`")
        lines.append(f"- risoluzioni video: `{p['video_sizes']}`")
        lines.append(f"- flag (non bloccanti): `{p['flags']}`")
        lines.append(f"- pattern immagine risolto: `{p['image_patterns']}`")
        if p["failure_counts"]:
            lines.append(f"- quarantena: `{p['failure_counts']}`")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

FIELDNAMES = [
    "sample_id",
    "config_folder",
    "vendor",
    "id_echo",
    "id_probe",
    "setup_id",
    "fss_path",
    "depth_index",
    "length_mm",
    "vect_depth_mm",
    "vect_minus_length_mm",
    "length_over_depth",
    "tick_cm",
    "side_raw",
    "x",
    "y_zero",
    "y_far",
    "zero_at",
    "y_top",
    "y_bottom",
    "length_px",
    "n_intervals",
    "tick_pitch_px",
    "mm_per_px_from_scale",
    "pixel_ratio_y",
    "calib_length_mm",
    "calib_rel_err",
    "video_x_size",
    "video_y_size",
    "rect_x1",
    "rect_y1",
    "rect_x2",
    "rect_y2",
    "side_vs_rect",
    "offset_from_rect_px",
    "offset_from_rect_frac_w",
    "y_overlap_frac",
    "in_frame",
    "image_path",
    "image_pattern",
    "flag_no_rect",
    "flag_y_range_off_rect",
    "flag_scale_inside_rect",
    "flag_fusion",
    "flag_suspect_folder",
    "status",
    "failures",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit the .fss scale ground truth per vendor.")
    p.add_argument("--root", type=Path, default=Path("/Volumes/SSD_esi1_n1"))
    p.add_argument(
        "--canonical-root",
        type=str,
        default="",
        help="Write paths as if the volume were mounted here. Use it when scanning the "
        "SSD through a different mount point than the one the team uses, so the CSVs "
        "stay portable (e.g. --root /mnt/SSD_esi1_n1 --canonical-root /Volumes/SSD_esi1_n1).",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit-setups", type=int, default=0, help="0 = no limit (smoke tests).")
    p.add_argument(
        "--no-image-check",
        action="store_true",
        help="Skip resolving the full-frame image for each depth index (faster).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root: Path = args.root
    if not root.is_dir():
        print(f"[error] root not found: {root}")
        return 2
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[AuditRow] = []
    setups_seen = 0
    setups_without_scale = 0
    for setup in iter_scale_setups(root):
        setups_seen += 1
        if args.limit_setups and setups_seen > args.limit_setups:
            setups_seen -= 1
            break
        if not setup.scale_entries:
            setups_without_scale += 1
            continue
        for entry in setup.scale_entries:
            rows.append(audit_entry(setup, entry, check_images=not args.no_image_check))
        if setups_seen % 25 == 0:
            print(f"[scan] setups={setups_seen} rows={len(rows)}", flush=True)

    if not rows:
        print("[error] no scale rows parsed")
        return 3

    if args.canonical_root:
        old = root.as_posix().rstrip("/")
        new = args.canonical_root.rstrip("/")
        rewritten = 0
        for r in rows:
            for key in ("fss_path", "image_path"):
                val = str(r.values.get(key) or "")
                if val.startswith(old):
                    r.values[key] = new + val[len(old) :]
                    rewritten += 1
        print(f"[audit] rewrote {rewritten} paths from {old} to {new}")

    csv_path = out_dir / "scale_gt_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in rows:
            payload = dict(r.values)
            payload["status"] = r.status
            payload["failures"] = ";".join(r.failures)
            writer.writerow(payload)

    clean_rows = [r for r in rows if r.status == "clean"]
    clean_path = out_dir / "scale_gt_rows_clean.csv"
    with clean_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in clean_rows:
            payload = dict(r.values)
            payload["status"] = r.status
            payload["failures"] = ""
            writer.writerow(payload)

    profiles = build_profiles(rows)
    totals = {
        "root": root.as_posix(),
        "setups_scanned": setups_seen,
        "setups_without_scale_line": setups_without_scale,
        "rows_total": len(rows),
        "rows_clean": len(clean_rows),
        "rows_quarantine": len(rows) - len(clean_rows),
        "clean_ratio_pct": round(100 * len(clean_rows) / len(rows), 2),
        "config_folders": len({str(r.values["config_folder"]) for r in rows}),
        "failure_counts": dict(Counter(f for r in rows for f in r.failures).most_common()),
        "vendors": len(profiles),
    }

    (out_dir / "scale_gt_profiles.json").write_text(
        json.dumps({"totals": totals, "profiles": profiles}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "scale_gt_profiles.md").write_text(
        render_markdown(profiles, totals), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(totals, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(totals, indent=2, ensure_ascii=False))
    print(f"[ok] {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
