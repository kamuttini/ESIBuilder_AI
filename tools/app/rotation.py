"""OSD rotation of a folder of acquisitions, with tesseract.

Same rule as the pipeline (`_estimate_folder_rotation_with_osd`): a uniform sample of frames
goes through `tesseract --psm 0`, the winning angle needs at least 2 votes and 60% of the
support, ties prefer 0 degrees. Without tesseract the folder is left unrotated and the reason
is reported, so the caller can flag it instead of silently pretending it is upright.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

MIN_VOTES = 2
MIN_RATIO = 0.60
ANGLES = (0, 90, 180, 270)
OSD_WORKERS = 4  # tesseract is a separate process, so this parallelises well

# No downscaling: measured on a 1920x1200 frame, OSD costs ~0.3s at full size, and shrinking
# to 1000 px makes tesseract give up with "Too few characters". Full frames it is.

_ROTATE_RE = re.compile(r"Rotate:\s*(\d+)")


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def _osd_angle(path: Path, timeout: float = 20.0) -> Optional[int]:
    """Clockwise rotation tesseract thinks is needed to make the text upright."""
    try:
        result = subprocess.run(
            ["tesseract", str(path), "stdout", "--psm", "0"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = _ROTATE_RE.search(result.stdout or "")
    if not match:
        return None
    angle = int(match.group(1)) % 360
    return angle if angle in ANGLES else None


def estimate_rotation(paths: Sequence[Path], max_samples: int = 12) -> Dict:
    if not tesseract_available():
        return {"angle": 0, "source": "osd_unavailable", "votes": {}, "reliable": False,
                "reason": "tesseract non disponibile: rotazione non stimata"}
    if not paths:
        return {"angle": 0, "source": "no_images", "votes": {}, "reliable": False,
                "reason": "nessuna immagine"}

    sample: List[Path] = list(paths)
    if len(sample) > max_samples:
        step = len(sample) / max_samples
        sample = [sample[int(index * step)] for index in range(max_samples)]

    votes = Counter()
    with ThreadPoolExecutor(max_workers=OSD_WORKERS) as pool:
        for angle in pool.map(_osd_angle, sample):
            if angle is not None:
                votes[angle] += 1

    total = sum(votes.values())
    if not total:
        return {"angle": 0, "source": "osd_no_votes", "votes": {}, "reliable": False,
                "reason": "OSD senza risposte utili"}

    best_angle, best_votes = max(votes.items(), key=lambda item: (item[1], item[0] == 0, -item[0]))
    ratio = best_votes / total
    if best_votes < MIN_VOTES or ratio < MIN_RATIO:
        return {"angle": 0, "source": "osd_low_support", "votes": dict(votes), "reliable": False,
                "reason": f"supporto debole ({best_votes}/{total} voti, {ratio:.0%})"}

    return {
        "angle": int(best_angle),
        "source": "osd",
        "votes": dict(votes),
        "ratio": round(ratio, 3),
        "images": total,
        "reliable": True,
        "reason": f"{best_votes}/{total} voti su {best_angle} gradi",
    }
