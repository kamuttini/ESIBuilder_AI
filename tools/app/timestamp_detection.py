"""Riconoscimento autonomo dell'area data/ora nei fotogrammi ecografici.

L'area non e' legata al vendor: tesseract legge un campione uniforme della cartella e le
letture compatibili con una data o un'ora vengono raggruppate per posizione. Una proposta
diventa affidabile solo se ricompare nello stesso punto su piu' immagini; in caso contrario
il box resta una proposta visibile, ma non viene usato automaticamente dalla deduplicazione.
"""

from __future__ import annotations

import csv
import io
import re
import shutil
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

MAX_SAMPLES = 8
OCR_WORKERS = 4
MIN_OCR_CONFIDENCE = 25.0

_TIME_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)")
_DATE_NUM_RE = re.compile(
    r"(?<!\d)(\d{1,4})([/.-])(\d{1,2})\2(\d{1,4})(?!\d)"
)
_MONTHS = (
    "gen|feb|mar|apr|mag|giu|lug|ago|set|ott|nov|dic|"
    "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
)
_DATE_WORD_RE = re.compile(
    rf"(?<!\d)\d{{1,2}}[/.-](?:{_MONTHS})[/.-]\d{{2,4}}(?!\d)", re.IGNORECASE
)


def _uniform_sample(paths: Sequence[Path], maximum: int = MAX_SAMPLES) -> List[Path]:
    ordered = sorted(Path(path) for path in paths if Path(path).is_file())
    if len(ordered) <= maximum:
        return ordered
    step = len(ordered) / maximum
    return [ordered[min(len(ordered) - 1, int(index * step))] for index in range(maximum)]


def _time_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for match in _TIME_RE.finditer(text):
        hour, minute, second = match.groups()
        if int(hour) <= 23 and int(minute) <= 59 and (second is None or int(second) <= 59):
            spans.append(match.span())
    return spans


def _date_spans(text: str) -> List[Tuple[int, int]]:
    spans = [match.span() for match in _DATE_WORD_RE.finditer(text)]
    for match in _DATE_NUM_RE.finditer(text):
        first, _separator, middle, last = match.groups()
        # Formati ammessi: gg/mm/aa(aa) e aaaa/mm/gg. Validare i campi evita che versioni
        # software come 27.00.10 diventino una data.
        if len(first) == 4:
            year, month, day = int(first), int(middle), int(last)
        else:
            day, month, year = int(first), int(middle), int(last)
        if 1 <= month <= 12 and 1 <= day <= 31 and (len(str(year)) in (2, 4)):
            spans.append(match.span())
    return spans


def _box_union(rows: Iterable[Dict]) -> Dict[str, int]:
    rows = list(rows)
    return {
        "left": min(int(row["left"]) for row in rows),
        "top": min(int(row["top"]) for row in rows),
        "right": max(int(row["left"]) + int(row["width"]) for row in rows),
        "bottom": max(int(row["top"]) + int(row["height"]) for row in rows),
    }


def _ocr_lines(path: Path) -> Tuple[Tuple[int, int], List[Dict]]:
    """Righe OCR che contengono una data o un'ora, con coordinate originali."""
    try:
        with Image.open(path) as image:
            size = image.size
        process = subprocess.run(
            # PSM 3 tiene data e ora sulla stessa riga anche quando sono molto in alto e
            # separate da pochi pixel (Esaote); PSM 11 tendeva a spezzare i secondi.
            ["tesseract", str(path), "stdout", "--psm", "3", "tsv"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return (0, 0), []
    if process.returncode != 0:
        return size, []

    grouped: Dict[Tuple[str, str, str], List[Dict]] = {}
    try:
        reader = csv.DictReader(io.StringIO(process.stdout), delimiter="\t")
        for row in reader:
            text = (row.get("text") or "").strip()
            try:
                confidence = float(row.get("conf") or -1)
                parsed = {
                    "text": text,
                    "confidence": confidence,
                    "left": int(row.get("left") or 0),
                    "top": int(row.get("top") or 0),
                    "width": int(row.get("width") or 0),
                    "height": int(row.get("height") or 0),
                }
            except (TypeError, ValueError):
                continue
            if (not text or confidence < MIN_OCR_CONFIDENCE
                    or parsed["width"] < 2 or parsed["height"] < 2):
                continue
            key = (row.get("block_num") or "", row.get("par_num") or "",
                   row.get("line_num") or "")
            grouped.setdefault(key, []).append(parsed)
    except csv.Error:
        return size, []

    candidates: List[Dict] = []
    for words in grouped.values():
        words.sort(key=lambda word: word["left"])
        compact_parts: List[str] = []
        word_spans: List[Tuple[int, int]] = []
        offset = 0
        for word in words:
            token = word["text"].replace("\\", "/")
            compact_parts.append(token)
            word_spans.append((offset, offset + len(token)))
            offset += len(token)
        compact = "".join(compact_parts)
        time_spans = _time_spans(compact)
        date_spans = _date_spans(compact)
        has_time = bool(time_spans)
        has_date = bool(date_spans)
        if not (has_time or has_date):
            continue
        matches = time_spans + date_spans
        relevant = [
            word for word, word_span in zip(words, word_spans)
            if any(word_span[0] < match[1] and match[0] < word_span[1] for match in matches)
        ]
        box = _box_union(relevant)
        box.update({
            "text": " ".join(word["text"] for word in relevant),
            "has_time": has_time,
            "has_date": has_date,
            "ocr_confidence": round(
                statistics.mean(word["confidence"] for word in relevant), 2
            ),
        })
        candidates.append(box)
    return size, candidates


def _near(first: Dict, second: Dict, width: int, height: int) -> bool:
    """Date e ora possono stare sulla stessa riga oppure su due righe sovrapposte."""
    horizontal_gap = max(first["left"], second["left"]) - min(first["right"], second["right"])
    vertical_gap = max(first["top"], second["top"]) - min(first["bottom"], second["bottom"])
    same_row = vertical_gap <= 0 and horizontal_gap <= max(20, round(width * 0.08))
    same_column = horizontal_gap <= 0 and vertical_gap <= max(16, round(height * 0.055))
    aligned_column = (
        abs(first["left"] - second["left"]) <= max(16, round(width * 0.025))
        and vertical_gap <= max(16, round(height * 0.055))
    )
    return same_row or same_column or aligned_column


def _merge_nearby(candidates: List[Dict], width: int, height: int) -> List[Dict]:
    pending = [dict(candidate) for candidate in candidates]
    merged: List[Dict] = []
    while pending:
        group = [pending.pop(0)]
        changed = True
        while changed:
            changed = False
            for candidate in list(pending):
                if any(_near(candidate, present, width, height) for present in group):
                    group.append(candidate)
                    pending.remove(candidate)
                    changed = True
        box = {
            "left": min(item["left"] for item in group),
            "top": min(item["top"] for item in group),
            "right": max(item["right"] for item in group),
            "bottom": max(item["bottom"] for item in group),
            "text": " · ".join(item["text"] for item in group),
            "has_time": any(item["has_time"] for item in group),
            "has_date": any(item["has_date"] for item in group),
            "ocr_confidence": round(statistics.mean(item["ocr_confidence"] for item in group), 2),
        }
        merged.append(box)
    return merged


def _same_position(candidate: Dict, cluster: Dict, width: int, height: int) -> bool:
    candidate_x = (candidate["left"] + candidate["right"]) / 2
    candidate_y = (candidate["top"] + candidate["bottom"]) / 2
    cluster_x = statistics.median(item["center_x"] for item in cluster["items"])
    cluster_y = statistics.median(item["center_y"] for item in cluster["items"])
    return (abs(candidate_x - cluster_x) <= max(24, width * 0.075)
            and abs(candidate_y - cluster_y) <= max(18, height * 0.045))


def _empty(reason: str, samples: int = 0) -> Dict:
    return {
        "box": None,
        "confidence": 0.0,
        "reliable": False,
        "source": "ocr",
        "support": 0,
        "samples": samples,
        "texts": [],
        "preview_path": None,
        "reason": reason,
    }


def detect_timestamp_box(paths: Sequence[Path], max_samples: int = MAX_SAMPLES) -> Dict:
    """Propone il rettangolo data/ora usando piu' fotogrammi della cartella."""
    sample = _uniform_sample(paths, max_samples)
    if not sample:
        return _empty("nessuna immagine da leggere")
    if shutil.which("tesseract") is None:
        return _empty("tesseract non disponibile: area dell'ora non riconosciuta", len(sample))

    with ThreadPoolExecutor(max_workers=min(OCR_WORKERS, len(sample))) as pool:
        readings = list(pool.map(_ocr_lines, sample))

    clusters: List[Dict] = []
    image_size: Optional[Tuple[int, int]] = None
    for sample_index, (path, reading) in enumerate(zip(sample, readings)):
        size, raw_candidates = reading
        width, height = size
        if width <= 0 or height <= 0:
            continue
        image_size = image_size or size
        for candidate in _merge_nearby(raw_candidates, width, height):
            item = dict(candidate)
            item.update({
                "sample_index": sample_index,
                "path": str(path),
                "center_x": (candidate["left"] + candidate["right"]) / 2,
                "center_y": (candidate["top"] + candidate["bottom"]) / 2,
            })
            cluster = next(
                (present for present in clusters
                 if _same_position(candidate, present, width, height)),
                None,
            )
            if cluster is None:
                clusters.append({"items": [item]})
            elif all(existing["sample_index"] != sample_index for existing in cluster["items"]):
                cluster["items"].append(item)

    if not clusters or image_size is None:
        return _empty("nessuna data o ora leggibile nel campione", len(sample))

    def rank(cluster: Dict) -> Tuple:
        items = cluster["items"]
        support = len(items)
        both = sum(bool(item["has_time"] and item["has_date"]) for item in items)
        kinds = int(any(item["has_time"] for item in items)) + int(
            any(item["has_date"] for item in items))
        confidence = statistics.mean(item["ocr_confidence"] for item in items)
        return support, kinds, both, confidence

    winner = max(clusters, key=rank)
    items = winner["items"]
    width, height = image_size
    raw_box = {
        side: int(round(statistics.median(item[side] for item in items)))
        for side in ("left", "top", "right", "bottom")
    }
    pad_x = max(5, round(width * 0.004))
    pad_y = max(4, round(height * 0.005))
    box = {
        "left": max(0, raw_box["left"] - pad_x),
        "top": max(0, raw_box["top"] - pad_y),
        "right": min(width, raw_box["right"] + pad_x),
        "bottom": min(height, raw_box["bottom"] + pad_y),
    }
    support = len(items)
    ratio = support / len(sample)
    has_time = any(item["has_time"] for item in items)
    has_date = any(item["has_date"] for item in items)
    ocr_confidence = statistics.mean(item["ocr_confidence"] for item in items) / 100.0
    kind_bonus = 1.0 if has_time and has_date else 0.82
    confidence = max(0.0, min(1.0, (0.65 * ratio + 0.35 * ocr_confidence) * kind_bonus))
    # Tre letture indipendenti e sovrapposte sono gia' una prova forte anche nelle
    # cartelle dove la scritta sparisce in alcune modalita' (su Esaote accade spesso).
    reliable = support >= 3 and ratio >= 0.35 and confidence >= 0.45
    if len(sample) == 1:
        reliable = ocr_confidence >= 0.80 and has_time and has_date
    texts = list(dict.fromkeys(item["text"] for item in items if item["text"]))[:5]
    reason = (
        f"data/ora letta nella stessa posizione su {support}/{len(sample)} immagini"
        if reliable else
        f"possibile data/ora su {support}/{len(sample)} immagini: controllare il riquadro"
    )
    best = max(items, key=lambda item: item["ocr_confidence"])
    return {
        "box": box,
        "confidence": round(confidence, 3),
        "reliable": reliable,
        "source": "ocr",
        "support": support,
        "samples": len(sample),
        "texts": texts,
        "preview_path": best["path"],
        "reason": reason,
    }
