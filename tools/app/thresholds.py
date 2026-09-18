"""Match thresholds (the ``TH`` inside ``CH:TH:P1:P2:...``) for the template rows, the legacy way.

ESI does not compute anything at runtime: for every template row (#13, #14, #16, #17) it cuts the
search box out of the live frame, runs ``cv::matchTemplate`` with ``TM_SQDIFF`` on the stored PNG
and accepts the match when the minimum is **below** ``TH``. A zero threshold therefore means
"never", which is what the app wrote until now.

The old ESIBuilder derived ``TH`` on the *Thresholds* page (``WdgManageTamplatesThresholds``):
the operator listed positive frames (where the template must be found) and negative frames
(where it must not), the tool measured the worst positive and the best negative and put the
threshold in between. Here the two lists come from the project itself:

* #13 / #14 — positives: every usable frame of the folder; negatives: the forbidden screens and
  frames of other projects on disk;
* #16 — positives: the frames of that orientation group; negatives: the frames of the other groups;
* #17 — positives: the frames read at that depth; negatives: the frames read at the other depths.

``FE_TH_METHOD`` (MM=6) is reproduced from ``CvMatchTemplate::getMinValTh``: grey levels are
stretched from [P1, P2] to [0, 255] on every channel before ``TM_SQDIFF``.

Crops follow the DB_echo convention measured on the archive (echo_name 152x39 for a 157x44 box,
probe_name 46x16 for 51x21): the template is the box moved by (+3, +3) and shrunk by 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

METHOD_SQDIFF = 0
METHOD_TH = 6
TEMPLATE_INSET = 3  # legacy crop: origin +3, size -5
TEMPLATE_PAD = 3  # #17: template = tight digit box grown by this (legacy 19x17 around 11 px digits)
SEARCH_PAD = 7  # #17: search box = tight digit box grown by this (legacy 28x26)
DEFAULT_FRACTION = 0.5  # where TH sits between worst positive and best negative
NO_NEGATIVE_FACTOR = 3.0  # without negatives: a tolerance above the worst positive, in review


class FrameCache:
    """BGR frames loaded once; the same frame is scored by many templates."""

    def __init__(self) -> None:
        self._frames: Dict[str, Optional[np.ndarray]] = {}

    def get(self, path: Path) -> Optional[np.ndarray]:
        key = str(path)
        if key not in self._frames:
            img = cv2.imread(key, cv2.IMREAD_COLOR)
            self._frames[key] = img if img is not None and img.size else None
        return self._frames[key]


def load_image(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img if img is not None and img.size else None


def box_tuple(box: Dict) -> Tuple[int, int, int, int]:
    return (int(box["top"]), int(box["left"]), int(box["bottom"]), int(box["right"]))


def crop_template(img: np.ndarray, box: Dict, inset: int = TEMPLATE_INSET) -> Optional[np.ndarray]:
    """The PNG that goes into DB_echo: the box moved by (+inset, +inset), width/height -(2*inset-1)."""
    top, left, bottom, right = box_tuple(box)
    width, height = right - left, bottom - top
    w, h = width - (2 * inset - 1), height - (2 * inset - 1)
    if w < 3 or h < 3:
        return None
    y0, x0 = top + inset, left + inset
    if y0 < 0 or x0 < 0 or y0 + h > img.shape[0] or x0 + w > img.shape[1]:
        return None
    return img[y0:y0 + h, x0:x0 + w].copy()


def grow_box(box: Dict, margin: int, width: int, height: int) -> Dict:
    return {
        "top": max(0, int(box["top"]) - margin),
        "left": max(0, int(box["left"]) - margin),
        "bottom": min(height - 1, int(box["bottom"]) + margin) if height else int(box["bottom"]) + margin,
        "right": min(width - 1, int(box["right"]) + margin) if width else int(box["right"]) + margin,
    }


def union_box(boxes: Sequence[Dict]) -> Optional[Dict]:
    utili = [b for b in boxes if b]
    if not utili:
        return None
    return {
        "top": min(int(b["top"]) for b in utili), "left": min(int(b["left"]) for b in utili),
        "bottom": max(int(b["bottom"]) for b in utili), "right": max(int(b["right"]) for b in utili),
    }


def _stretch(img: np.ndarray, p1: float, p2: float) -> np.ndarray:
    """``convertTo(CV_8U, alpha, beta)`` of the legacy: [p1, p2] -> [0, 255], saturated."""
    alpha = 255.0 / (float(p2) - float(p1))
    beta = -float(p1) * alpha
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def match_min(img: np.ndarray, template: np.ndarray, box: Dict, method: int = METHOD_TH,
              p1: float = 20.0, p2: float = 120.0) -> Optional[float]:
    """Minimum TM_SQDIFF of the template inside the search box, as ESI will compute it.

    The legacy ``Rect`` has width ``right-left+1``: the box is inclusive on both ends."""
    top, left, bottom, right = box_tuple(box)
    if top < 0 or left < 0 or bottom >= img.shape[0] or right >= img.shape[1]:
        return None
    roi = img[top:bottom + 1, left:right + 1]
    if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
        return None
    if method == METHOD_TH:
        if p1 >= p2:
            return None
        roi, template = _stretch(roi, p1, p2), _stretch(template, p1, p2)
    result = cv2.matchTemplate(roi, template, cv2.TM_SQDIFF)
    return float(result.min())


def suggest_threshold(positives: Sequence[float], negatives: Sequence[float],
                      fraction: float = DEFAULT_FRACTION) -> Dict:
    """The legacy rule: between the worst positive and the best negative; overlap -> review."""
    if not positives:
        return {"threshold": 0.0, "status": "review", "reason": "nessun positivo misurabile",
                "worst_positive": None, "best_negative": None, "margin": None}
    worst = max(positives)
    if not negatives:
        return {"threshold": max(1.0, worst * NO_NEGATIVE_FACTOR), "status": "review",
                "reason": "nessun negativo: soglia di tolleranza sopra il peggior positivo",
                "worst_positive": worst, "best_negative": None, "margin": None}
    best = min(negatives)
    margin = best - worst
    if margin <= 0:
        return {"threshold": worst, "status": "review",
                "reason": "positivi e negativi si sovrappongono: il template non separa",
                "worst_positive": worst, "best_negative": best, "margin": margin}
    return {"threshold": worst + margin * fraction, "status": "ok", "reason": "",
            "worst_positive": worst, "best_negative": best, "margin": margin}


def spread(items: Sequence, limit: int) -> List:
    """Up to ``limit`` items spread over the sequence, so a sample does not come from one burst."""
    items = list(items)
    if limit <= 0 or len(items) <= limit:
        return items
    step = len(items) / float(limit)
    return [items[int(i * step)] for i in range(limit)]


@dataclass
class Task:
    """One template block to threshold."""

    key: str  # e.g. "13", "16:LR", "17:3:UD"
    line: int
    label: str
    template: np.ndarray
    template_name: str
    box: Dict
    positives: List[Path]
    negatives: List[Path]
    group: str = ""
    depth_index: Optional[int] = None
    depth_mm: Optional[float] = None
    method: int = METHOD_TH
    p1: float = 20.0
    p2: float = 120.0
    notes: List[str] = field(default_factory=list)
    synthetic_negative: Optional[float] = None  # used only when ``negatives`` is empty
    drop_weak_negatives: bool = False  # a 'negative' scoring like a positive is not a negative


def _scores(task: Task, paths: Sequence[Path], cache: FrameCache) -> Tuple[List[float], int]:
    values: List[float] = []
    skipped = 0
    for path in paths:
        img = cache.get(path)
        if img is None:
            skipped += 1
            continue
        score = match_min(img, task.template, task.box, task.method, task.p1, task.p2)
        if score is None:
            skipped += 1
            continue
        values.append(score)
    return values, skipped


def evaluate(task: Task, cache: FrameCache, fraction: float = DEFAULT_FRACTION) -> Dict:
    pos, skip_pos = _scores(task, task.positives, cache)
    neg, skip_neg = _scores(task, task.negatives, cache)
    dropped = 0
    if task.drop_weak_negatives and pos and neg:
        # Una schermata di servizio della stessa macchina mostra ancora il logo: per la #13 non
        # e' un negativo, e' un positivo travestito. Si scarta chi sta entro 10 volte il peggior
        # positivo, e si dice.
        limite = max(pos) * 10.0 + 1.0
        buoni = [v for v in neg if v > limite]
        dropped = len(neg) - len(buoni)
        neg = buoni
    if not neg and task.synthetic_negative is not None:
        neg = [float(task.synthetic_negative)]
        task.notes.append("negativo sintetico: template su sfondo nero")
    if dropped:
        task.notes.append(f"{dropped} negativi scartati perche' mostrano ancora il template")
    suggestion = suggest_threshold(pos, neg, fraction)
    return {
        "key": task.key, "line": task.line, "label": task.label, "group": task.group,
        "depth_index": task.depth_index, "depth_mm": task.depth_mm,
        "template": task.template_name,
        "template_size": [int(task.template.shape[1]), int(task.template.shape[0])],
        "box": {k: int(task.box[k]) for k in ("top", "left", "bottom", "right")},
        "n_positive": len(pos), "n_negative": len(neg),
        "skipped": skip_pos + skip_neg,
        "median_positive": float(np.median(pos)) if pos else None,
        "notes": list(task.notes),
        **suggestion,
    }


def recognise(frames: Sequence[Tuple[Path, str]], blocks: Dict[str, Dict], cache: FrameCache) -> Dict:
    """Simulate ESI on labelled frames: which blocks accept each frame?

    ``frames`` are (path, true_label); ``blocks`` maps label -> {"template", "box", "threshold",
    "method", "p1", "p2"}. A frame is *ok* when only its own label accepts it, *ambiguous* when
    its own and others do, *wrong* when only others do, *none* when nobody does."""
    counts = {"ok": 0, "ambiguous": 0, "wrong": 0, "none": 0, "unreadable": 0}
    failures: List[Dict] = []
    for path, truth in frames:
        img = cache.get(path)
        if img is None:
            counts["unreadable"] += 1
            continue
        accepted: List[str] = []
        for label, block in blocks.items():
            score = match_min(img, block["template"], block["box"], block.get("method", METHOD_TH),
                              block.get("p1", 20.0), block.get("p2", 120.0))
            if score is not None and block["threshold"] > 0 and score < block["threshold"]:
                accepted.append(label)
        if accepted == [truth]:
            counts["ok"] += 1
        elif truth in accepted:
            counts["ambiguous"] += 1
            failures.append({"frame": path.name, "truth": truth, "accepted": accepted})
        elif accepted:
            counts["wrong"] += 1
            failures.append({"frame": path.name, "truth": truth, "accepted": accepted})
        else:
            counts["none"] += 1
            failures.append({"frame": path.name, "truth": truth, "accepted": []})
    return {"counts": counts, "total": len(frames), "failures": failures[:40]}


def block_dict(box: Dict, threshold: float, method: int = METHOD_TH, p1: float = 20.0,
               p2: float = 120.0, channel: int = 7) -> Dict:
    """A template block in the shape ``fss_writer.TemplateBlock.from_dict`` expects."""
    return {
        "top": int(box["top"]), "left": int(box["left"]),
        "bottom": int(box["bottom"]), "right": int(box["right"]),
        "check": 1,
        "params": {"threshold": float(threshold), "channel": channel, "p1": p1, "p2": p2},
        "match_method": int(method),
    }


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def synthetic_black_negative(template: np.ndarray, method: int = METHOD_TH,
                             p1: float = 20.0, p2: float = 120.0) -> float:
    """The score the template gets on a black patch: what another machine's dark UI would give.

    Without a second machine on disk there is no real negative. On ultrasound UIs the place of
    the logo is black, or carries other text, on any other scanner: the template against black
    is a fair stand-in for the best negative (its value is the sum of the squared template)."""
    t = _stretch(template, p1, p2) if method == METHOD_TH else template
    return float(np.sum(t.astype(np.float64) ** 2))


def diff_box(frame: np.ndarray, references: Sequence[np.ndarray], exclude: Optional[Dict],
             min_diff: int = 60, min_area: int = 60, pad: int = 2) -> Optional[Dict]:
    """Where a forbidden screen differs from the normal frames, outside the echo rectangle.

    The mask is the pixel-wise *minimum* difference against several references, so anything
    that changes from frame to frame anyway (the image itself, the clock) drops out, and the
    largest connected component is what makes the screen recognisable: an alert, a banner, a
    menu. Returns its bounding box grown by ``pad``."""
    if not references:
        return None
    mask = None
    for ref in references:
        if ref.shape != frame.shape:
            continue
        d = np.abs(frame.astype(np.int16) - ref.astype(np.int16)).max(axis=2) > min_diff
        mask = d if mask is None else (mask & d)
    if mask is None:
        return None
    if exclude:
        top, left, bottom, right = box_tuple(exclude)
        mask[max(0, top):bottom + 1, max(0, left):right + 1] = False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    # Fra le componenti si preferisce quella dove la schermata proibita *ha* qualcosa che i
    # fotogrammi normali non hanno (un avviso, un banner), non quella dove le manca qualcosa
    # (il logo spento): un template di nero cercherebbe il nero, e il nero c'e' ovunque.
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref_grey = np.median(np.stack([cv2.cvtColor(r, cv2.COLOR_BGR2GRAY) for r in references
                                   if r.shape == frame.shape]), axis=0).astype(np.float32)
    present: List[Tuple[int, int, int, int, int]] = []
    absent: List[Tuple[int, int, int, int, int]] = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        comp = labels == i
        voce = (int(x), int(y), int(w), int(h), int(area))
        (present if float(grey[comp].mean()) > float(ref_grey[comp].mean()) else absent).append(voce)
    scelta = present or absent
    if not scelta:
        return None
    x, y, w, h, _ = max(scelta, key=lambda v: v[4])
    return {"top": max(0, y - pad), "left": max(0, x - pad),
            "bottom": min(frame.shape[0] - 1, y + h - 1 + pad),
            "right": min(frame.shape[1] - 1, x + w - 1 + pad)}
