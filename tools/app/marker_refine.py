"""Refine a marker position from an approximate click, and check its orientation group.

The user clicks roughly where the glyph is; the precise box comes from the **same matching the
detector uses**: the bundle's multi-scale `TM_CCOEFF_NORMED` over the vendor's template bank,
restricted to a window around the click. Nothing is re-invented, so a corrected position is
comparable with the ones the batch produced.

The group then follows the bundle's own rule (`_group_from_side_vertical`):

    left + su -> NF        right + su -> LR
    left + giu -> UD       right + giu -> LRUD

with `side` and `vertical` read off the marker's position inside the echo rectangle. That is the
verification the user asked for: the corrected point must land in the group the image claims.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

GROUP_ORDER = ("NF", "LR", "UD", "LRUD")

# The glyph changes size between machines and frames: matching only at the template's native
# size throws away the best overlap. Same ladder the pipeline recommends for --lr-marker-scales,
# with two intermediate steps because here we are refining, not scanning a whole folder.
SCALES = (0.75, 0.9, 1.0, 1.15, 1.3, 1.7, 2.2)
TOP_K_FOR_SCALES = 6


def _bundle(bundle_dir: Path):
    """Import the detector bundle as a module (it is a package on disk, not installed)."""
    root = Path(bundle_dir)
    if not (root / "orientation_marker_detector").is_dir():
        raise FileNotFoundError(f"bundle non trovato: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import orientation_marker_detector.detector as detector  # noqa: PLC0415

    return detector


def group_from_position(
    box: Dict[str, int], rect: Dict[str, int], detector
) -> Tuple[str, str, str]:
    """(side, vertical, group) of a marker box inside the echo rectangle."""
    centre_x = (int(box["left"]) + int(box["right"])) / 2.0
    centre_y = (int(box["top"]) + int(box["bottom"])) / 2.0
    rect_centre_x = (int(rect["left"]) + int(rect["right"])) / 2.0
    rect_centre_y = (int(rect["top"]) + int(rect["bottom"])) / 2.0
    side = "left" if centre_x < rect_centre_x else "right"
    vertical = "su" if centre_y < rect_centre_y else "giu"
    return side, vertical, detector._group_from_side_vertical(side, vertical)


def refine_from_click(
    *,
    image_path: Path,
    click: Tuple[int, int],
    rect: Dict[str, int],
    bundle_dir: Path,
    library_root: Path,
    vendor: str,
    preferred_template: str = "",
    window: int = 70,
    match_max_side: int = 560,
    scales: Sequence[float] = SCALES,
    own_templates: Sequence[Path] = (),
) -> Dict:
    """Best overlap of the vendor bank in a window around the click, across scales.

    Two passes, like the bundle's own selection: every template at its native size, then the
    extra scales only on the handful that already look promising. The winner is the highest
    `TM_CCOEFF_NORMED` of all the overlaps tried, and the runners-up travel back so the choice
    can be seen as a comparison instead of a single guess.
    """
    detector = _bundle(bundle_dir)
    templates: List = detector.load_vendor_templates(library_root, vendor=vendor)
    # I ritagli di questa cartella (il marker indicato a mano, i suggerimenti) entrano in
    # gara con la banca. Quando la banca ha scelto il glifo sbagliato — su una cartella GE
    # ha preso una scritta fissa dell'intestazione — cercare solo con lei riporterebbe
    # l'errore dentro la correzione.
    for extra_path in own_templates:
        loaded = detector._load_template_file(Path(extra_path), "orientation", "cartella")
        if loaded is not None:
            templates.insert(0, loaded)
    if not templates:
        return {"error": f"nessun template in banca per il vendor {vendor}"}

    if preferred_template:
        templates.sort(key=lambda t: 0 if Path(str(t.path)).name == preferred_template else 1)

    width, height, gray = detector._load_gray_cached(Path(image_path))
    x, y = int(click[0]), int(click[1])
    search_rect = detector._clip_rect(
        (y - window, x - window, y + window, x + window), width=width, height=height
    )
    if search_rect is None:
        return {"error": "il punto cliccato cade fuori dall'immagine"}

    tried: List[Dict] = []

    def attempt(template, scale: float) -> None:
        candidate = template
        if abs(scale - 1.0) > 1e-3:
            candidate = detector._rescale_template(template, scale)
            if candidate is None:
                return
        match = detector._match_template(gray, candidate, search_rect, "click", match_max_side)
        if match is None:
            return
        tried.append(
            {
                "score": float(match.score),
                "template": Path(str(template.path)).name,
                "scale": round(float(scale), 3),
                "size": [int(candidate.width), int(candidate.height)],
                "box": {
                    "top": int(match.box_abs[0]), "left": int(match.box_abs[1]),
                    "bottom": int(match.box_abs[2]), "right": int(match.box_abs[3]),
                },
            }
        )

    for template in templates:
        attempt(template, 1.0)
    if not tried:
        return {"error": "nessun match nella zona indicata: prova a cliccare piu' preciso"}

    extra = [float(s) for s in scales if abs(float(s) - 1.0) > 1e-3]
    if extra:
        best_names = [
            item["template"]
            for item in sorted(tried, key=lambda item: -item["score"])[:TOP_K_FOR_SCALES]
        ]
        for template in templates:
            if Path(str(template.path)).name not in best_names:
                continue
            for scale in extra:
                attempt(template, scale)

    tried.sort(key=lambda item: -item["score"])
    best = tried[0]
    box = best["box"]
    side, vertical, group = group_from_position(box, rect, detector)
    return {
        "box": box,
        "score": round(best["score"], 4),
        "template": best["template"],
        "template_size": best["size"],
        "scale": best["scale"],
        "overlaps_tried": len(tried),
        "runner_up": (
            {k: (round(tried[1][k], 4) if k == "score" else tried[1][k])
             for k in ("score", "template", "scale")}
            if len(tried) > 1 else None
        ),
        "candidates": [
            {"score": round(item["score"], 4), "template": item["template"], "scale": item["scale"]}
            for item in tried[:3]
        ],
        "side": side,
        "vertical": vertical,
        "group": group,
        "click": {"x": x, "y": y},
        "window": int(window),
        "offset_from_click": {
            "dx": int(round((box["left"] + box["right"]) / 2 - x)),
            "dy": int(round((box["top"] + box["bottom"]) / 2 - y)),
        },
    }


def envelope_of(boxes: Sequence[Dict[str, int]]) -> Optional[Dict[str, int]]:
    """The rectangle that encloses every marker position of a group."""
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return {
        "top": min(int(b["top"]) for b in boxes),
        "left": min(int(b["left"]) for b in boxes),
        "bottom": max(int(b["bottom"]) for b in boxes),
        "right": max(int(b["right"]) for b in boxes),
    }


# The batch builds an envelope only from the detections it trusts: rows left in `review` are
# out. Recomputing with them in would move every envelope at the first correction, which reads
# as "I fixed one image and all four boxes changed".
ACCEPTED_STATUS = {"ok", "corrected"}


def rebuild_groups(
    per_image: Sequence[Dict], corrections: Dict[str, Dict]
) -> Dict[str, Dict]:
    """Envelopes per group from the batch rows, with the user's corrections overriding them."""
    buckets: Dict[str, List[Dict[str, int]]] = {group: [] for group in GROUP_ORDER}
    for row in per_image:
        name = row.get("name") or row.get("image_id") or ""
        fixed = corrections.get(name)
        status = str((fixed or row).get("status") or "").strip().lower()
        if not fixed and status and status not in ACCEPTED_STATUS:
            continue
        group = (fixed or {}).get("group") or row.get("group") or ""
        box = (fixed or {}).get("box") or row.get("box")
        if group in buckets and box:
            buckets[group].append(box)
    # A correction on an image the batch never analysed still counts.
    for name, fixed in corrections.items():
        if any((row.get("name") == name) for row in per_image):
            continue
        group = fixed.get("group") or ""
        if group in buckets and fixed.get("box"):
            buckets[group].append(fixed["box"])

    out: Dict[str, Dict] = {}
    for group, boxes in buckets.items():
        envelope = envelope_of(boxes)
        if envelope:
            out[group] = {
                **envelope,
                "check": 1,
                "params": {"threshold": 0.0},
                "markers": len(boxes),
            }
    return out
