"""The needle detector used by the guides chain: Hough first, matched filter as a fallback.

Two methods were measured head to head on the same 48 frames, scored by how close their
choice lands to one of the configuration's legacy angles:

                    fires      best within 1 deg   best within 3 deg   top three within 3 deg
  Hough             36 / 48           50%                 72%                   78%
  matched filter    48 / 48           17%                 23%                   73%

So they fail in opposite ways. Hough thresholds first and therefore finds nothing on a quarter
of the frames, but when it does find something it ranks it well. The matched filter searches
every line of the frame and so always has an answer, but cannot tell a needle from a long
bright band: its shortlist is fine and its first choice is not.

Hence: Hough decides, and the matched filter is asked only when Hough is empty, where any
candidate beats none. Recall goes from 36 of 48 to 48 of 48 without touching the ranking on the
frames Hough already handles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_needle_line import candidates_in_frame as _hough  # noqa: E402
from detect_needle_radon import detect_in_frame as _radon  # noqa: E402


@dataclass
class Needle:
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    angle_deg: float
    method: str
    confidence: float

    @property
    def points(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return self.p1, self.p2


def detect(frame_gray: np.ndarray, rect: Tuple[int, int, int, int], top_k: int = 4,
           ) -> List[Needle]:
    """Needles in the frame, best first, in frame coordinates."""
    out: List[Needle] = []
    for found in _hough(frame_gray, rect, top_k=top_k):
        out.append(Needle(p1=found.p1, p2=found.p2, angle_deg=found.angle_deg,
                          method="hough", confidence=float(found.contrast)))
    if out:
        return out

    # Nothing survived the brightness cut: fall back, and say so, because a fallback
    # detection ranks worse and the caller should know which kind it is holding.
    for found in sorted(_radon(frame_gray, rect, top_k=top_k), key=lambda d: -d.support):
        out.append(Needle(p1=found.p1, p2=found.p2, angle_deg=found.angle_deg,
                          method="matched", confidence=float(found.support)))
    return out[:top_k]
