#!/usr/bin/env python3
"""Estimate if a single image is rotated and by how much (clockwise)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image


TESSERACT_OSD_ROTATE_RE = re.compile(r"Rotate:\s*(\d+)")
TESSERACT_OSD_CONF_RE = re.compile(r"Orientation confidence:\s*([0-9]*\.?[0-9]+)")
VALID_ROTATIONS = (0, 90, 180, 270)


def _run_tesseract_osd(image_path: Path, timeout_sec: float) -> Tuple[Optional[int], Optional[float], str]:
    try:
        proc = subprocess.run(
            [
                "tesseract",
                str(image_path),
                "stdout",
                "--psm",
                "0",
                "-l",
                "osd",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError:
        return None, None, "tesseract_not_found"
    except Exception as exc:
        return None, None, f"osd_error:{exc}"

    text = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    rotate_match = TESSERACT_OSD_ROTATE_RE.search(text)
    conf_match = TESSERACT_OSD_CONF_RE.search(text)
    conf_value: Optional[float] = None
    if conf_match:
        try:
            conf_value = float(conf_match.group(1))
        except ValueError:
            conf_value = None

    if not rotate_match:
        return None, conf_value, "osd_rotate_missing"
    rotate = int(rotate_match.group(1)) % 360
    if rotate not in VALID_ROTATIONS:
        return None, conf_value, f"osd_rotate_invalid:{rotate}"
    return rotate, conf_value, ""


def _run_tesseract_ocr_word_score(
    image_path: Path,
    rotate_deg_clockwise: int,
    timeout_sec: float,
    lang: str,
    psm: int,
    min_word_conf: float,
) -> Tuple[float, int, str]:
    rotate = int(rotate_deg_clockwise) % 360
    tmp_path: Optional[Path] = None
    source_path = image_path

    try:
        if rotate != 0:
            with Image.open(image_path) as img:
                rotated = img.convert("RGB").rotate(-rotate, expand=True)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                rotated.save(tmp_path, format="PNG")
            source_path = tmp_path

        proc = subprocess.run(
            [
                "tesseract",
                str(source_path),
                "stdout",
                "--oem",
                "1",
                "--psm",
                str(psm),
                "-l",
                lang,
                "tsv",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError:
        return 0.0, 0, "tesseract_not_found"
    except Exception as exc:
        return 0.0, 0, f"ocr_error:{exc}"
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    conf_sum = 0.0
    words = 0
    tsv = proc.stdout or ""
    for line in tsv.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        text = (parts[11] or "").strip()
        if not text:
            continue
        if not any(ch.isalnum() for ch in text):
            continue
        try:
            conf = float(parts[10])
        except ValueError:
            continue
        if conf < float(min_word_conf):
            continue
        conf_sum += conf
        words += 1
    return conf_sum, words, ""


def _evaluate_ocr_scores(
    image_path: Path,
    timeout_sec: float,
    lang: str,
    psm: int,
    min_word_conf: float,
    word_bonus: float,
) -> Tuple[Dict[int, float], Dict[int, int], str]:
    scores: Dict[int, float] = {}
    words: Dict[int, int] = {}
    for rotate_deg in VALID_ROTATIONS:
        conf_sum, word_count, err = _run_tesseract_ocr_word_score(
            image_path=image_path,
            rotate_deg_clockwise=rotate_deg,
            timeout_sec=timeout_sec,
            lang=lang,
            psm=psm,
            min_word_conf=min_word_conf,
        )
        if err:
            return {}, {}, err
        scores[rotate_deg] = float(conf_sum + float(word_count) * float(word_bonus))
        words[rotate_deg] = int(word_count)
    return scores, words, ""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check if a single image must be rotated.")
    p.add_argument("--image", type=Path, required=True, help="Input image path.")
    p.add_argument("--osd-timeout-sec", type=float, default=2.5)
    p.add_argument("--min-orientation-confidence", type=float, default=0.0)
    p.add_argument("--disable-ocr-validation", action="store_true")
    p.add_argument("--ocr-timeout-sec", type=float, default=3.0)
    p.add_argument("--ocr-lang", type=str, default="eng")
    p.add_argument("--ocr-psm", type=int, default=6)
    p.add_argument("--ocr-min-word-conf", type=float, default=35.0)
    p.add_argument("--ocr-score-word-bonus", type=float, default=4.0)
    p.add_argument("--ocr-validation-min-words", type=int, default=2)
    p.add_argument("--ocr-validation-min-score-delta", type=float, default=12.0)
    p.add_argument("--json-only", action="store_true", help="Print only JSON.")
    p.add_argument("--output-json", type=Path, default=None, help="Optional JSON output file.")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise RuntimeError(f"Image not found: {image_path}")
    if args.osd_timeout_sec <= 0:
        raise ValueError("--osd-timeout-sec must be > 0.")
    if args.ocr_timeout_sec <= 0:
        raise ValueError("--ocr-timeout-sec must be > 0.")
    if args.min_orientation_confidence < 0:
        raise ValueError("--min-orientation-confidence must be >= 0.")
    if args.ocr_min_word_conf < 0:
        raise ValueError("--ocr-min-word-conf must be >= 0.")
    if args.ocr_score_word_bonus < 0:
        raise ValueError("--ocr-score-word-bonus must be >= 0.")
    if args.ocr_validation_min_words < 0:
        raise ValueError("--ocr-validation-min-words must be >= 0.")
    if args.ocr_validation_min_score_delta < 0:
        raise ValueError("--ocr-validation-min-score-delta must be >= 0.")

    osd_rotate, osd_conf_opt, osd_warning = _run_tesseract_osd(
        image_path=image_path,
        timeout_sec=args.osd_timeout_sec,
    )
    osd_conf = float(osd_conf_opt or 0.0)

    candidate_rotate = 0
    if osd_rotate is not None and osd_conf >= float(args.min_orientation_confidence):
        candidate_rotate = int(osd_rotate)

    ocr_scores: Dict[int, float] = {}
    ocr_words: Dict[int, int] = {}
    ocr_best_rotate = candidate_rotate
    ocr_best_score = 0.0
    ocr_candidate_score = 0.0
    ocr_warning = ""
    final_rotate = candidate_rotate

    if not args.disable_ocr_validation:
        ocr_scores, ocr_words, ocr_warning = _evaluate_ocr_scores(
            image_path=image_path,
            timeout_sec=args.ocr_timeout_sec,
            lang=args.ocr_lang,
            psm=args.ocr_psm,
            min_word_conf=args.ocr_min_word_conf,
            word_bonus=args.ocr_score_word_bonus,
        )
        if ocr_warning == "tesseract_not_found":
            raise RuntimeError("tesseract not found")
        if not ocr_warning and ocr_scores:
            ocr_best_rotate = max(
                VALID_ROTATIONS,
                key=lambda deg: (
                    ocr_scores.get(deg, 0.0),
                    ocr_words.get(deg, 0),
                    deg == int(candidate_rotate),
                    deg == 0,
                    -deg,
                ),
            )
            ocr_best_score = float(ocr_scores.get(ocr_best_rotate, 0.0))
            ocr_candidate_score = float(ocr_scores.get(int(candidate_rotate), 0.0))
            delta = float(ocr_best_score - ocr_candidate_score)
            if (
                int(ocr_words.get(ocr_best_rotate, 0)) >= int(args.ocr_validation_min_words)
                and ocr_best_rotate != int(candidate_rotate)
                and delta >= float(args.ocr_validation_min_score_delta)
            ):
                final_rotate = int(ocr_best_rotate)

    result = {
        "image_path": image_path.as_posix(),
        "osd_rotate_deg": int(osd_rotate) if osd_rotate is not None else None,
        "osd_confidence": float(osd_conf),
        "osd_warning": osd_warning,
        "candidate_rotate_deg": int(candidate_rotate),
        "ocr_best_rotate_deg": int(ocr_best_rotate),
        "ocr_best_score": float(ocr_best_score),
        "ocr_candidate_score": float(ocr_candidate_score),
        "ocr_words": {str(k): int(v) for k, v in sorted(ocr_words.items())},
        "ocr_scores": {str(k): float(v) for k, v in sorted(ocr_scores.items())},
        "ocr_warning": ocr_warning,
        "final_rotate_deg": int(final_rotate),
        "is_rotated": bool(int(final_rotate) != 0),
        "status": "rotate" if int(final_rotate) != 0 else "ok",
        "message": (
            f"immagine ruotata: applicare {int(final_rotate)} gradi CW"
            if int(final_rotate) != 0
            else "immagine non ruotata"
        ),
    }

    if args.output_json is not None:
        out_path = args.output_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.json_only:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        print(result["message"], flush=True)
        print(f"final_rotate_deg={result['final_rotate_deg']}", flush=True)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

