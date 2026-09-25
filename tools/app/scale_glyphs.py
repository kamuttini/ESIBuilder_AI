"""Per-folder glyph reader for the depth label at the end of the ultrasound scale.

Tesseract on 18 px white labels fails as soon as something touches them: the green zoom
marker (`X3`), the blue `P` badge, a saturated echo strip. Within one folder the machine is
the same, so every character is drawn with the same pixels: the labels the OCR did read
teach the glyphs ("0".."9", "c", "m"), and the other images are read by isolating the white
components along the scale lane and matching them against those glyphs.

On prova_13 (Philips Affiniti 70) it recovered 23 of the 64 images the OCR missed, all
correct, and agreed with the OCR on the 77 it could compare. Labels cut by the bottom edge
of the screen are not read: only their top half exists.
"""
from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

GLYPH_W, GLYPH_H = 12, 20
THRESHOLDS = (150, 200, 235)  # low joins thin strokes, high separates the echo
MIN_SCORE = 0.7

Component = Tuple[int, int, int, int, np.ndarray]  # x, y, w, h, full-size mask


def _white(image_path: Path) -> Optional[np.ndarray]:
    """Darkest channel: white text stays bright, the green marker and blue badge go dark."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    return None if image is None else image.min(axis=2)


def _drop_long_lines(mask: np.ndarray, length: int = 36) -> np.ndarray:
    """Remove horizontal lines longer than any character, then re-stitch the characters.

    The bottom border of the ultrasound image is a white line that can run straight through
    the top of the label (prova_13: 96, 97, 51, 57, 63, 64, 69): it is as white as the digits
    and glued them into one blob. No character has a horizontal run that long. Where the
    line crossed a character, the pixels touching the character above or below come back.
    """
    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    if not lines.any():
        return mask
    rest = mask & (1 - lines)
    near = cv2.dilate(rest, np.ones((3, 1), np.uint8))
    return rest | (lines & near)


def _components(gray: np.ndarray, threshold: int) -> List[Component]:
    mask = _drop_long_lines((gray >= threshold).astype(np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    return [(int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3]), labels == i)
            for i in range(1, count) if stats[i, 3] >= 3]


def _normalised(comp: Component) -> np.ndarray:
    x, y, w, h, mask = comp
    return cv2.resize(mask[y:y + h, x:x + w].astype(np.float32), (GLYPH_W, GLYPH_H),
                      interpolation=cv2.INTER_AREA)


def _join(a: Component, b: Component) -> Component:
    x, y = min(a[0], b[0]), min(a[1], b[1])
    x2, y2 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return (x, y, x2 - x, y2 - y, a[4] | b[4])


def _split_wide(comps: Sequence[Component], height: float) -> List[Component]:
    """A blob of digit height much wider than a digit: split it at its empty columns."""
    out: List[Component] = []
    for comp in comps:
        x, y, w, h, mask = comp
        if not (0.8 * height <= h <= 1.3 * height and w > 1.3 * height):
            out.append(comp)
            continue
        sub = mask[y:y + h, x:x + w]
        columns = sub.any(axis=0)
        i = 0
        while i < w:
            if not columns[i]:
                i += 1
                continue
            j = i
            while j < w and columns[j]:
                j += 1
            rows = np.where(sub[:, i:j].any(axis=1))[0]
            part = np.zeros_like(mask)
            part[y:y + h, x + i:x + j] = sub[:, i:j]
            out.append((x + i, y + int(rows[0]), j - i, int(rows[-1] - rows[0] + 1), part))
            i = j
    return out


def _join_strokes(comps: Sequence[Component], height: float) -> List[Component]:
    """Consecutive thin strokes of lowercase height are one glyph ('m' at a high threshold)."""
    out: List[Component] = []
    group: List[Component] = []

    def close() -> None:
        if group:
            joined = group[0]
            for comp in group[1:]:
                joined = _join(joined, comp)
            out.append(joined)
            group.clear()

    for comp in sorted(_split_wide(comps, height), key=lambda c: c[0]):
        stroke = comp[2] <= 0.35 * height and 0.3 * height < comp[3] < 0.85 * height
        if stroke and group:
            prev = group[-1]
            if abs((prev[1] + prev[3]) - (comp[1] + comp[3])) <= 2 and comp[0] - (prev[0] + prev[2]) <= 3:
                group.append(comp)
                continue
        close()
        if stroke:
            group.append(comp)
        else:
            out.append(comp)
    close()
    return out


class GlyphBank:
    def __init__(self, glyphs: Dict[str, List[np.ndarray]], digit_height: float) -> None:
        self.glyphs = glyphs
        self.height = digit_height

    def classify(self, comp: Component) -> Tuple[Optional[str], float]:
        piece = _normalised(comp)
        a = piece - piece.mean()
        best: Tuple[Optional[str], float] = (None, -1.0)
        for char, templates in self.glyphs.items():
            for template in templates:
                b = template - template.mean()
                den = float(np.sqrt((a * a).sum() * (b * b).sum())) or 1.0
                score = float((a * b).sum() / den)
                if score > best[1]:
                    best = (char, score)
        return best


def learn(image_dir: Path, readings: Dict[str, Dict]) -> Optional[GlyphBank]:
    """Glyphs from labels already read ('4.5cm' -> components in x order)."""
    glyphs: Dict[str, List[np.ndarray]] = {}
    heights: List[int] = []
    for name, reading in readings.items():
        text = re.sub(r"[^0-9.cm]", "", str(reading.get("ocr_text") or "").lower())
        if not re.fullmatch(r"\d+(\.\d+)?cm", text) or not reading.get("box"):
            continue
        gray = _white(image_dir / name)
        if gray is None:
            continue
        box = reading["box"]
        x0, y0 = max(0, box["left"] - 6), max(0, box["top"] - 6)
        zone = gray[y0:box["bottom"] + 6, x0:box["right"] + 50]
        for threshold in THRESHOLDS:
            comps = _join_strokes(_components(zone, threshold), 18.0)
            tall = [c for c in comps if c[3] >= 10]
            if not tall:
                continue
            bottom = max(c[1] + c[3] for c in tall)
            row = sorted([c for c in comps if abs((c[1] + c[3]) - bottom) <= 2], key=lambda c: c[0])
            # The dot can vanish at a high threshold: align on whether it is really there,
            # otherwise every glyph after it is taught under the wrong name ('c' as '0').
            dot = any(c[2] <= 5 and c[3] <= 5 for c in row)
            expected = text if dot else text.replace(".", "")
            if len(row) < len(expected):
                continue
            pieces = row[:len(expected)]
            if any((ch == ".") != (c[2] <= 5 and c[3] <= 5) for ch, c in zip(expected, pieces)):
                continue  # something else sits in the row: do not learn from it
            for char, comp in zip(expected, pieces):
                if char == ".":
                    continue
                glyphs.setdefault(char, []).append(_normalised(comp))
                if char.isdigit() and threshold == THRESHOLDS[0]:
                    heights.append(comp[3])
    if not heights or len(glyphs) < 3:
        return None
    return GlyphBank({k: v[:36] for k, v in glyphs.items()}, statistics.median(heights))


def read(bank: GlyphBank, image_path: Path, lane: Tuple[float, float, float, float],
         decimals: Optional[int]) -> Optional[Tuple[str, Dict]]:
    """The lowest '<number>cm' label in the lane: (number text, box in image pixels)."""
    gray = _white(image_path)
    if gray is None:
        return None
    x0, y0, x1, y1 = (int(v) for v in lane)
    zone = gray[y0:y1, x0:x1]
    height = bank.height
    for threshold in THRESHOLDS:
        comps = _join_strokes(_components(zone, threshold), height)
        digits = [c for c in comps if 0.8 * height <= c[3] <= 1.2 * height and 3 <= c[2] <= height]
        labels = []
        for bottom in sorted({c[1] + c[3] for c in digits}):
            row = sorted([c for c in comps if abs(c[1] + c[3] - bottom) <= 2], key=lambda c: c[0])
            chains: List[List[Component]] = []
            current: List[Component] = []
            for comp in row:
                if current and comp[0] - (current[-1][0] + current[-1][2]) > 0.9 * height:
                    chains.append(current)
                    current = []
                current.append(comp)
            if current:
                chains.append(current)
            for chain in chains:
                text = _read_chain(bank, chain, decimals)
                match = re.search(r"(\d+(?:\.\d+)?)cm", text or "")
                if match:
                    labels.append((bottom, match.group(1), chain))
        if labels:
            bottom, number, chain = max(labels, key=lambda item: item[0])
            box = {"left": x0 + min(c[0] for c in chain), "top": y0 + min(c[1] for c in chain),
                   "right": x0 + max(c[0] + c[2] for c in chain), "bottom": y0 + bottom}
            return number, box
    return None


def _read_chain(bank: GlyphBank, chain: Sequence[Component], decimals: Optional[int]) -> Optional[str]:
    height = bank.height
    text, previous = "", None
    i = 0
    while i < len(chain):
        comp = chain[i]
        i += 1
        x, _y, w, h, _m = comp
        if w <= 0.3 * height and h <= 0.3 * height:
            text += "."
            previous = (x, w, ".")
            continue
        if w > 1.3 * height or h > 1.25 * height:
            return None
        char, score = bank.classify(comp)
        # not recognised: maybe a piece of a split glyph - try joining the next pieces
        joined = comp
        while score < MIN_SCORE and i < len(chain) and chain[i][0] - (joined[0] + joined[2]) <= 3:
            joined = _join(joined, chain[i])
            i += 1
            char2, score2 = bank.classify(joined)
            if score2 >= MIN_SCORE:
                char, score, comp = char2, score2, joined
        if score < MIN_SCORE or char is None:
            return None
        x, w = comp[0], comp[2]
        # an invisible dot: two digits much farther apart than usual
        if (decimals and previous and previous[2].isdigit() and char.isdigit() and "." not in text
                and 0.35 * height <= x - (previous[0] + previous[1]) <= 0.8 * height):
            text += "."
        text += char
        previous = (x, w, char)
    return text
