"""Pick the calibration frames of an acquisition with the classifier, not with folder names.

Folder names are what was available before the classifier existed, and they let through frames
that are not calibration material at all -- Camilla spotted one with no probe in water inside a
folder called `aghi`. On a frame like that there is no needle to find, so every measurement
taken from it is noise attributed to the detector.

This scores each frame the same way the classifier was trained (rect crop, letterboxed) and
keeps the ones above the accepted threshold, best first.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_needle_classifier import IMAGENET_MEAN, IMAGENET_STD, build_model, letterbox  # noqa: E402
from torchvision.transforms import functional as TF  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
SKIP_DIRS = {"$RECYCLE.BIN", "System Volume Information"}


class NeedleScorer:
    """The trained classifier, loaded once and kept."""

    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.image_size = int(checkpoint["image_size"])
        self.model = build_model(checkpoint["arch"], dropout=0.0)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    def _tensor(self, path: Path, rect: Tuple[int, int, int, int], margin_pct: float = 2.0):
        left, top, right, bottom = rect
        with Image.open(path) as raw:
            image = raw.convert("RGB")
            width, height = image.size
            left = max(0, min(int(left), width - 1))
            top = max(0, min(int(top), height - 1))
            right = max(left + 1, min(int(right), width))
            bottom = max(top + 1, min(int(bottom), height))
            mx = (right - left) * margin_pct / 100.0
            my = (bottom - top) * margin_pct / 100.0
            crop = image.crop((int(max(0, round(left - mx))), int(max(0, round(top - my))),
                               int(min(width, round(right + mx))),
                               int(min(height, round(bottom + my)))))
            canvas = letterbox(crop, self.image_size)
        return TF.normalize(TF.to_tensor(canvas), IMAGENET_MEAN, IMAGENET_STD)

    @torch.no_grad()
    def score(self, paths: Sequence[Path], rect: Tuple[int, int, int, int],
              batch: int = 16) -> List[float]:
        out: List[float] = []
        for start in range(0, len(paths), batch):
            chunk = list(paths[start:start + batch])
            tensors = []
            for path in chunk:
                try:
                    tensors.append(self._tensor(path, rect))
                except Exception:
                    tensors.append(torch.zeros(3, self.image_size, self.image_size))
            logits = self.model(torch.stack(tensors).to(self.device)).squeeze(1).float()
            out.extend(torch.sigmoid(logits).cpu().tolist())
        return out


def all_frames(acquisition: Path, size: Tuple[int, int], cap: int = 400) -> List[Path]:
    """Frames of the acquisition at the configuration's own resolution."""
    found: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(acquisition):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in sorted(filenames):
            if Path(name).suffix.lower() not in IMAGE_EXTS or name.startswith("._"):
                continue
            path = Path(dirpath) / name
            try:
                with Image.open(path) as img:
                    if img.size != size:
                        continue
            except Exception:
                continue
            found.append(path)
            if len(found) >= cap:
                return found
    return found


def calibration_frames(scorer: NeedleScorer, acquisition: Path, size: Tuple[int, int],
                       rect: Tuple[int, int, int, int], limit: int = 3,
                       threshold: float = 0.60, scan_cap: int = 160,
                       ) -> List[Tuple[Path, float]]:
    """The frames the classifier calls calibration material, best first."""
    frames = all_frames(acquisition, size, cap=scan_cap)
    if not frames:
        return []
    scores = scorer.score(frames, rect)
    ranked = sorted(zip(frames, scores), key=lambda pair: -pair[1])
    kept = [(path, score) for path, score in ranked if score >= threshold]
    return kept[:limit]


# --------------------------------------------------------------------- sonda
import re as _re

# A probe code mixes letters and digits: LA332, CA541, TLC3-13, E14CL4b, ML6-15, 12L-RS, 8848.
# Words that look like one but never are, mostly software and build numbers.
_NOT_A_PROBE = _re.compile(
    r"^(rev\d*|sw\d*|v\d+|f\d{6}|\d{1,2}|\d{4}|20\d{2}|r\d|bt\d+)$", _re.IGNORECASE)


def probe_tokens(config_name: str) -> List[str]:
    """Probe codes named in a configuration's folder name."""
    out: List[str] = []
    for token in _re.split(r"[\s,_/()]+", config_name):
        token = token.strip("-.")
        if len(token) < 3 or len(token) > 12:
            continue
        if not (_re.search(r"[A-Za-z]", token) and _re.search(r"\d", token)):
            continue
        if _NOT_A_PROBE.match(token):
            continue
        out.append(token)
    return out


def _normalise(text: str) -> str:
    return _re.sub(r"[^a-z0-9]", "", text.lower())


def frames_of_probe(frames: Sequence[Path], acquisition: Path, config_name: str
                    ) -> List[Path]:
    """Keep the frames whose sub-folder names the configuration's probe, when it does.

    One acquisition often covers several probes in sub-folders -- "0. LA332", "1. CA541",
    "2. LA523" -- and every configuration of that machine matched the whole acquisition, so a
    configuration for one probe was being measured on another probe's needles. Their guide
    angles differ, so the comparison against the legacy values was wrong in a way that looked
    like a detector error.

    If no sub-folder mentions any of the probes, nothing is filtered: the acquisition is
    single-probe and the frames are all there is.
    """
    tokens = [_normalise(t) for t in probe_tokens(config_name)]
    if not tokens:
        return list(frames)
    kept = [
        path for path in frames
        if any(token in _normalise(str(path.parent.relative_to(acquisition)))
               for token in tokens)
    ]
    return kept or list(frames)
