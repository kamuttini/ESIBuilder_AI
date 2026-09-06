"""Marker della cartella: si innesca sulla banca storica, poi si taglia e si usa il proprio.

Perche' in due fasi. A runtime ESI fa il match contro il PNG che la configurazione gli
consegna (`DB_echo/setup_NN/orientation_*.png`), non contro un template storico. Misurare la
confidenza contro la banca risponde quindi alla domanda sbagliata: dice quanto somiglia il
glifo di un'altra macchina. La domanda giusta e' quanto bene **il ritaglio che stiamo per
salvare** si sovrappone a tutte le immagini di questa cartella.

Quindi:

1. **innesco** — la banca del vendor, a piu' scale, su poche immagini di selezione: serve solo
   a trovare dove sta il marker;
2. **taglio** — dall'immagine col punteggio migliore si ritaglia il **riquadro esatto** del
   marker: quello e' il template del setup, ed e' il file che andra' in `db_echo`;
3. **rimatch** — tutte le immagini con quel ritaglio, a scala nativa (stessa macchina, stessa
   risoluzione): i punteggi diventano confrontabili fra loro e sono quelli che vedra' ESI;
4. **envelope** — dalle posizioni, i quattro box minimi di `#16`, uno per orientamento.

Misurato sul corpus legacy: i quattro `orientation_*.png` di un setup sono sempre **identici
fra loro** (un glifo, quattro copie) e la loro dimensione e' il riquadro del marker in quella
configurazione, senza margine — da 16x18 a 30x31 px nei dieci setup di `templates/DB_echo`.
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

GROUP_ORDER = ("NF", "LR", "UD", "LRUD")
SEED_SCALES = (0.75, 0.9, 1.0, 1.15, 1.3, 1.7, 2.2)
SEED_IMAGES = 6


def _bundle(bundle_dir: Path):
    root = Path(bundle_dir)
    if not (root / "orientation_marker_detector").is_dir():
        raise FileNotFoundError(f"bundle non trovato: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import orientation_marker_detector.detector as detector  # noqa: PLC0415

    return detector


def group_of(box: Dict[str, int], rect: Dict[str, int], detector) -> Tuple[str, str, str]:
    """(lato, verticale, gruppo) dalla posizione del marker dentro il rettangolo ecografico."""
    cx = (int(box["left"]) + int(box["right"])) / 2.0
    cy = (int(box["top"]) + int(box["bottom"])) / 2.0
    rcx = (int(rect["left"]) + int(rect["right"])) / 2.0
    rcy = (int(rect["top"]) + int(rect["bottom"])) / 2.0
    side = "left" if cx < rcx else "right"
    vertical = "su" if cy < rcy else "giu"
    return side, vertical, detector._group_from_side_vertical(side, vertical)


# --- fase 1+2: innesco sulla banca e taglio del marker della cartella -----
def cut_folder_template(
    *,
    images: Sequence[Path],
    bundle_dir: Path,
    library_root: Path,
    vendor: str,
    out_path: Path,
    seed_images: int = SEED_IMAGES,
    scales: Sequence[float] = SEED_SCALES,
) -> Dict:
    """Trova il marker con la banca e salva il ritaglio esatto: il template del setup."""
    detector = _bundle(bundle_dir)
    bank: List = detector.load_vendor_templates(library_root, vendor=vendor)
    if not bank:
        return {"error": f"nessun template in banca per il vendor {vendor}"}
    if not images:
        return {"error": "nessuna immagine da cui innescare"}

    step = max(1, len(images) // seed_images)
    seeds = list(images)[::step][:seed_images]

    best = None
    for image_path in seeds:
        try:
            width, height, gray = detector._load_gray_cached(Path(image_path))
        except Exception:
            continue
        whole = (0, 0, height - 1, width - 1)
        for template in bank:
            for scale in scales:
                candidate = template
                if abs(scale - 1.0) > 1e-3:
                    candidate = detector._rescale_template(template, scale)
                    if candidate is None:
                        continue
                match = detector._match_template(gray, candidate, whole, "seed", 720)
                if match is None:
                    continue
                if best is None or match.score > best["score"]:
                    best = {
                        "score": float(match.score),
                        "box": {
                            "top": int(match.box_abs[0]), "left": int(match.box_abs[1]),
                            "bottom": int(match.box_abs[2]), "right": int(match.box_abs[3]),
                        },
                        "image": Path(image_path),
                        "bank_template": Path(str(template.path)).name,
                        "scale": round(float(scale), 3),
                    }
    if best is None:
        return {"error": "nessun marker trovato nelle immagini di innesco"}

    # Riquadro esatto, senza margine: e' cosi' che sono i ritagli legacy.
    from PIL import Image

    box = best["box"]
    with Image.open(best["image"]) as raw:
        crop = raw.convert("RGB").crop(
            (box["left"], box["top"], box["right"] + 1, box["bottom"] + 1)
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)

    return {
        "template_path": str(out_path),
        "size": [crop.width, crop.height],
        "source_image": str(best["image"]),
        "source_box": box,
        "seed_score": round(best["score"], 4),
        "seed_bank_template": best["bank_template"],
        "seed_scale": best["scale"],
        "seed_images": len(seeds),
    }


# --- fase 3: rimatch di tutte le immagini con il template della cartella ---
@dataclass
class MarkerHit:
    name: str
    box: Optional[Dict[str, int]]
    score: float
    group: str
    side: str
    vertical: str
    scale: float = 1.0


def match_all(
    *,
    images: Sequence[Path],
    folder: Path,
    template_path: Path,
    rect: Dict[str, int],
    bundle_dir: Path,
    min_score: float = 0.55,
    search_margin: Optional[int] = 40,
    scales: Sequence[float] = (1.0,),
    progress=None,
) -> Dict:
    """Il template della cartella su tutte le immagini, a scala nativa.

    `search_margin` limita la ricerca al rettangolo ecografico piu' quel margine: senza limite
    un ritaglio piccolo trova "qualcosa" anche nelle schermate dove il marker non c'e'.
    Con `None` si cerca sul frame intero (serve per i marker che stanno fuori dal rect, come
    nei setup legacy 80 e 81).
    """
    detector = _bundle(bundle_dir)
    template = detector._load_template_file(Path(template_path), "orientation", "folder")
    if template is None:
        return {"error": f"ritaglio non utilizzabile come template: {template_path}"}

    rows: List[MarkerHit] = []
    folder = Path(folder)
    for index, image_path in enumerate(images):
        try:
            width, height, gray = detector._load_gray_cached(Path(image_path))
        except Exception:
            continue
        if search_margin is None:
            area = (0, 0, height - 1, width - 1)
        else:
            area = detector._clip_rect(
                (
                    int(rect["top"]) - search_margin, int(rect["left"]) - search_margin,
                    int(rect["bottom"]) + search_margin, int(rect["right"]) + search_margin,
                ),
                width=width, height=height,
            ) or (0, 0, height - 1, width - 1)
        match = None
        used_scale = 1.0
        for scale in scales:
            candidate = template
            if abs(scale - 1.0) > 1e-3:
                candidate = detector._rescale_template(template, scale)
                if candidate is None:
                    continue
            attempt = detector._match_template_exact(gray, candidate, area, "folder_template")
            if attempt is not None and (match is None or attempt.score > match.score):
                match = attempt
                used_scale = scale
        name = str(Path(image_path).relative_to(folder))
        if match is None:
            rows.append(MarkerHit(name, None, -1.0, "", "", ""))
        else:
            box = {
                "top": int(match.box_abs[0]), "left": int(match.box_abs[1]),
                "bottom": int(match.box_abs[2]), "right": int(match.box_abs[3]),
            }
            side, vertical, group = group_of(box, rect, detector)
            hit = MarkerHit(name, box, round(float(match.score), 4), group, side, vertical)
            hit.scale = round(float(used_scale), 3)  # type: ignore[attr-defined]
            rows.append(hit)
        if progress and index % 25 == 0:
            progress(index + 1, len(images))

    scored = [row for row in rows if row.score >= 0]
    accepted = [row for row in scored if row.score >= min_score]
    return {
        "rows": [row.__dict__ for row in rows],
        "images": len(rows),
        "matched": len(scored),
        "accepted": len(accepted),
        "min_score": min_score,
        "score_stats": {
            "min": round(min(r.score for r in scored), 4) if scored else None,
            "median": round(statistics.median(r.score for r in scored), 4) if scored else None,
            "max": round(max(r.score for r in scored), 4) if scored else None,
        },
    }


# --- fase 4: envelope per gruppo -----------------------------------------
def envelopes(rows: Sequence[Dict], min_score: float = 0.55) -> Dict[str, Dict]:
    """I quattro box minimi che contengono tutte le posizioni del marker, per orientamento."""
    buckets: Dict[str, List[Dict[str, int]]] = {group: [] for group in GROUP_ORDER}
    for row in rows:
        if not row.get("box") or float(row.get("score") or -1) < min_score:
            continue
        group = row.get("group") or ""
        if group in buckets:
            buckets[group].append(row["box"])

    out: Dict[str, Dict] = {}
    for group, boxes in buckets.items():
        if not boxes:
            continue
        out[group] = {
            "top": min(b["top"] for b in boxes),
            "left": min(b["left"] for b in boxes),
            "bottom": max(b["bottom"] for b in boxes),
            "right": max(b["right"] for b in boxes),
            "check": 1,
            "params": {"threshold": 0.0},
            "markers": len(boxes),
        }
    return out


# --- scelta del ritaglio per copertura, non per punteggio singolo ---------
def candidate_crops(
    *,
    images: Sequence[Path],
    bundle_dir: Path,
    library_root: Path,
    vendor: str,
    rect: Dict[str, int],
    out_dir: Path,
    seed_images: int = 8,
    scales: Sequence[float] = SEED_SCALES,
) -> List[Dict]:
    """Un ritaglio candidato per ogni immagine di innesco: il miglior match della banca."""
    detector = _bundle(bundle_dir)
    bank = detector.load_vendor_templates(library_root, vendor=vendor)
    if not bank or not images:
        return []
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(images) // seed_images)
    out: List[Dict] = []
    for index, image_path in enumerate(list(images)[::step][:seed_images]):
        try:
            width, height, gray = detector._load_gray_cached(Path(image_path))
        except Exception:
            continue
        area = detector._clip_rect(
            (rect["top"] - 40, rect["left"] - 40, rect["bottom"] + 40, rect["right"] + 40),
            width=width, height=height,
        ) or (0, 0, height - 1, width - 1)
        best = None
        for template in bank:
            for scale in scales:
                candidate = template
                if abs(scale - 1.0) > 1e-3:
                    candidate = detector._rescale_template(template, scale)
                    if candidate is None:
                        continue
                match = detector._match_template_exact(gray, candidate, area, "seed")
                if match is not None and (best is None or match.score > best[0]):
                    best = (float(match.score), match.box_abs, Path(str(template.path)).name, scale)
        if best is None:
            continue
        score, box, bank_name, scale = best
        top, left, bottom, right = box
        with Image.open(image_path) as raw:
            crop = raw.convert("RGB").crop((left, top, right + 1, bottom + 1))
        path = out_dir / f"candidate_{index:02d}.png"
        crop.save(path)
        out.append(
            {
                "path": str(path), "size": [crop.width, crop.height],
                "seed_score": round(score, 4), "seed_bank_template": bank_name,
                "seed_scale": round(float(scale), 3),
                "source_image": str(Path(image_path)),
                "source_box": {"top": top, "left": left, "bottom": bottom, "right": right},
            }
        )
    return out


def choose_by_coverage(
    *,
    candidates: Sequence[Dict],
    images: Sequence[Path],
    folder: Path,
    rect: Dict[str, int],
    bundle_dir: Path,
    sample: int = 60,
    min_score: float = 0.85,
) -> Dict:
    """Fra i candidati vince quello che aggancia piu' immagini, non quello col picco piu' alto."""
    if not candidates:
        return {"error": "nessun ritaglio candidato"}
    step = max(1, len(images) // sample)
    probe = list(images)[::step][:sample]
    scored = []
    for candidate in candidates:
        result = match_all(
            images=probe, folder=folder, template_path=Path(candidate["path"]),
            rect=rect, bundle_dir=bundle_dir, min_score=min_score, search_margin=40,
        )
        rows = [row for row in result["rows"] if row["score"] >= 0]
        if not rows:
            continue
        coverage = sum(1 for row in rows if row["score"] >= min_score) / len(rows)
        scored.append(
            {
                **candidate,
                "coverage": round(coverage, 4),
                "median": result["score_stats"]["median"],
                "probe_images": len(rows),
            }
        )
    if not scored:
        return {"error": "nessun candidato utilizzabile"}
    scored.sort(key=lambda item: (-item["coverage"], -(item["median"] or 0)))
    return {"chosen": scored[0], "candidates": scored}


# --- validazione: il ritaglio da consegnare, DENTRO gli envelope ----------
def validate_in_envelopes(
    *,
    images: Sequence[Path],
    folder: Path,
    template_path: Path,
    groups: Dict[str, Dict],
    group_of_image: Dict[str, str],
    bundle_dir: Path,
    min_score: float = 0.85,
    near_px: int = 12,
    reference: Optional[Dict[str, Dict]] = None,
    extra_templates: Sequence[Path] = (),
    progress=None,
) -> Dict:
    """Cosa fara' ESI: il ritaglio consegnato cercato dentro il box del suo gruppo.

    Non e' la stessa cosa della scoperta: la' serve il richiamo (il box deve contenere il
    marker), qui serve la precisione (il miglior match dentro il box deve *essere* il marker).
    """
    detector = _bundle(bundle_dir)
    template = detector._load_template_file(Path(template_path), "orientation", "folder")
    if template is None:
        return {"error": f"ritaglio non utilizzabile: {template_path}"}
    # I ritagli nati dalle correzioni umane si aggiungono a quello consegnato: ogni
    # suggerimento va provato anche sulle altre immagini, non solo su quella corretta.
    templates = [("consegnato", template)]
    for extra in extra_templates:
        loaded = detector._load_template_file(Path(extra), "orientation", "hint")
        if loaded is not None:
            templates.append((Path(extra).name, loaded))

    per_group: Dict[str, Dict] = {
        group: {"images": 0, "matched": 0, "on_marker": 0, "scores": []} for group in GROUP_ORDER
    }
    rows: List[Dict] = []
    folder = Path(folder)
    for image_path in images:
        name = str(Path(image_path).relative_to(folder))
        group = group_of_image.get(name) or ""
        if group and group in groups:
            candidates = [(group, groups[group])]
            assigned = False
        else:
            # Immagine senza gruppo, cioe' una di quelle in revisione: il marker puo' stare
            # in uno qualsiasi dei quattro box, e quale sia e' proprio la risposta che
            # cerchiamo. Si provano tutti e vince il migliore — che e' esattamente cio' che
            # fara' ESI. Prima queste immagini venivano saltate: restavano fuori dalla
            # validazione, quindi nessun suggerimento poteva recuperarle e l'unico modo di
            # sistemarle era correggerle a mano una per una.
            candidates = [(g, groups[g]) for g in GROUP_ORDER if g in groups]
            assigned = True
        if not candidates:
            continue
        try:
            width, height, gray = detector._load_gray_cached(Path(image_path))
        except Exception:
            continue
        match = None
        winner = ""
        won = candidates[0][0]
        for group_name, envelope in candidates:
            area = detector._clip_rect(
                (envelope["top"], envelope["left"], envelope["bottom"], envelope["right"]),
                width=width, height=height,
            )
            if not area:
                continue
            for label, candidate in templates:
                attempt = detector._match_template_exact(gray, candidate, area, "envelope")
                if attempt is not None and (match is None or attempt.score > match.score):
                    match = attempt
                    winner = label
                    won = group_name
        group = won
        stats = per_group[group]
        stats["images"] += 1
        if assigned:
            stats["assigned"] = stats.get("assigned", 0) + 1
        if match is None:
            rows.append({"name": name, "group": group, "score": -1.0, "box": None,
                         "assigned": assigned})
            continue
        box = {
            "top": int(match.box_abs[0]), "left": int(match.box_abs[1]),
            "bottom": int(match.box_abs[2]), "right": int(match.box_abs[3]),
        }
        score = round(float(match.score), 4)
        stats["scores"].append(score)
        if score >= min_score:
            stats["matched"] += 1
        near = None
        if reference and reference.get(name, {}).get("box"):
            ref = reference[name]["box"]
            near = max(abs(box[k] - ref[k]) for k in box) <= near_px
            if near:
                stats["on_marker"] += 1
        rows.append({"name": name, "group": group, "score": score, "box": box,
                     "on_marker": near, "template": winner, "assigned": assigned})
        if progress and len(rows) % 25 == 0:
            progress(len(rows), len(images))

    coverage = {}
    for group, stats in per_group.items():
        if not stats["images"]:
            continue
        coverage[group] = {
            "images": stats["images"],
            "matched": stats["matched"],
            "coverage": round(stats["matched"] / stats["images"], 4),
            "on_marker": stats["on_marker"],
            "assigned": stats.get("assigned", 0),
            "median": round(statistics.median(stats["scores"]), 4) if stats["scores"] else None,
            "min": round(min(stats["scores"]), 4) if stats["scores"] else None,
        }
    total = sum(s["images"] for s in per_group.values())
    matched = sum(s["matched"] for s in per_group.values())
    return {
        "rows": rows,
        "coverage_by_group": coverage,
        "coverage": round(matched / total, 4) if total else 0.0,
        "images": total,
        "min_score": min_score,
    }


def coverage_of(
    *,
    template_path: Path,
    images: Sequence[Path],
    folder: Path,
    rect: Dict[str, int],
    bundle_dir: Path,
    min_score: float,
    sample: int = 40,
) -> Dict:
    """Copertura di un ritaglio su un campione: serve a confrontare due candidati."""
    step = max(1, len(images) // sample)
    probe = list(images)[::step][:sample]
    result = match_all(
        images=probe, folder=folder, template_path=Path(template_path), rect=rect,
        bundle_dir=bundle_dir, min_score=min_score, search_margin=40,
    )
    rows = [row for row in result["rows"] if row["score"] >= 0]
    if not rows:
        return {"coverage": 0.0, "median": None, "images": 0}
    return {
        "coverage": round(sum(1 for r in rows if r["score"] >= min_score) / len(rows), 4),
        "median": result["score_stats"]["median"],
        "images": len(rows),
    }


def cut_at(image_path: Path, box: Dict[str, int], out_path: Path) -> Dict:
    """Ritaglia il riquadro esatto indicato: e' il glifo che l'utente ha confermato col click."""
    from PIL import Image

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as raw:
        crop = raw.convert("RGB").crop(
            (int(box["left"]), int(box["top"]), int(box["right"]) + 1, int(box["bottom"]) + 1)
        )
    crop.save(out_path)
    return {"path": str(out_path), "size": [crop.width, crop.height]}
