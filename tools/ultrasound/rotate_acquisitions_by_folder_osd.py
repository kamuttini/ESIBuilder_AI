#!/usr/bin/env python3
"""Rotate acquisition images per folder with robust OSD + OCR validation.

Workflow:
1) scan each top-level folder under dataset-root
2) collect acquisition images matching timestamp_<vga|hdmi>_<WxH> pattern
3) run Tesseract OSD on multiple uniformly sampled frames
4) pick rotation only with minimum vote support
5) optionally validate candidate rotation with OCR readability over {0,90,180,270}
6) rotate all acquisition images in that folder by the selected correction angle
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
CAPTURE_FILENAME_PATTERN_RE = re.compile(
    r"^.+_(vga|hdmi)_(\d{3,5})[xX](\d{3,5})(?:[_\-].*)?$",
    flags=re.IGNORECASE,
)
TESSERACT_OSD_ROTATE_RE = re.compile(r"Rotate:\s*(\d+)")
TESSERACT_OSD_CONF_RE = re.compile(r"Orientation confidence:\s*([0-9]*\.?[0-9]+)")
VALID_ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True)
class FolderResult:
    folder: Path
    images_total: int
    sample_image: str
    osd_rotate_deg: int
    osd_confidence: float
    osd_votes_total: int
    osd_vote_ratio: float
    osd_samples_checked: int
    ocr_validation_used: int
    ocr_best_rotate_deg: int
    ocr_best_score: float
    ocr_candidate_score: float
    decision: str
    applied: int
    rotated_images: int
    errors: int
    warning: str


@dataclass(frozen=True)
class OsdEstimate:
    rotate_deg: Optional[int]
    confidence: float
    votes_total: int
    vote_ratio: float
    samples_checked: int
    sampled_paths: Tuple[Path, ...]
    warning: str
    tesseract_missing: bool


@dataclass(frozen=True)
class OcrValidation:
    final_rotate_deg: int
    used: int
    best_rotate_deg: int
    best_score: float
    candidate_score: float
    warning: str
    tesseract_missing: bool


def _collect_acquisition_images(folder: Path) -> List[Path]:
    out: List[Path] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not CAPTURE_FILENAME_PATTERN_RE.match(path.stem):
            continue
        out.append(path)
    out.sort()
    return out


def _select_uniform_subset(paths: Sequence[Path], limit: int) -> List[Path]:
    if not paths:
        return []
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    if limit == 1:
        return [paths[len(paths) // 2]]

    total = len(paths)
    selected_idx: List[int] = []
    seen = set()
    for i in range(limit):
        idx = round(i * (total - 1) / (limit - 1))
        if idx not in seen:
            selected_idx.append(idx)
            seen.add(idx)
    if len(selected_idx) < limit:
        for idx in range(total):
            if idx in seen:
                continue
            selected_idx.append(idx)
            seen.add(idx)
            if len(selected_idx) >= limit:
                break
    selected_idx.sort()
    return [paths[i] for i in selected_idx]


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


def _estimate_rotation_with_osd(
    image_paths: Sequence[Path],
    sample_count: int,
    timeout_sec: float,
    min_confidence: float,
    min_votes: int,
    min_ratio: float,
) -> OsdEstimate:
    sampled = _select_uniform_subset(image_paths, sample_count)
    if not sampled:
        return OsdEstimate(
            rotate_deg=None,
            confidence=0.0,
            votes_total=0,
            vote_ratio=0.0,
            samples_checked=0,
            sampled_paths=tuple(),
            warning="osd_no_samples",
            tesseract_missing=False,
        )

    votes: Counter[int] = Counter()
    conf_by_rotation: Dict[int, List[float]] = defaultdict(list)
    for path in sampled:
        rotate_deg, conf_opt, err = _run_tesseract_osd(path, timeout_sec=timeout_sec)
        if err == "tesseract_not_found":
            return OsdEstimate(
                rotate_deg=None,
                confidence=0.0,
                votes_total=int(sum(votes.values())),
                vote_ratio=0.0,
                samples_checked=len(sampled),
                sampled_paths=tuple(sampled),
                warning=err,
                tesseract_missing=True,
            )
        if rotate_deg is None:
            continue
        conf = float(conf_opt or 0.0)
        if conf < float(min_confidence):
            continue
        votes[int(rotate_deg)] += 1
        conf_by_rotation[int(rotate_deg)].append(conf)

    total_votes = int(sum(votes.values()))
    if total_votes <= 0:
        return OsdEstimate(
            rotate_deg=None,
            confidence=0.0,
            votes_total=0,
            vote_ratio=0.0,
            samples_checked=len(sampled),
            sampled_paths=tuple(sampled),
            warning="osd_no_votes",
            tesseract_missing=False,
        )

    best_rotation, best_count = max(
        votes.items(),
        key=lambda item: (item[1], item[0] == 0, -item[0]),
    )
    ratio = float(best_count / max(1, total_votes))
    best_conf = (
        float(sum(conf_by_rotation.get(best_rotation, [])) / max(1, len(conf_by_rotation.get(best_rotation, []))))
        if conf_by_rotation.get(best_rotation)
        else 0.0
    )
    if best_count < int(max(1, min_votes)) or ratio < float(min_ratio):
        return OsdEstimate(
            rotate_deg=None,
            confidence=best_conf,
            votes_total=total_votes,
            vote_ratio=ratio,
            samples_checked=len(sampled),
            sampled_paths=tuple(sampled),
            warning=(
                f"osd_low_support:best={best_rotation};votes={best_count}/{total_votes};"
                f"ratio={ratio:.3f};min_votes={int(max(1, min_votes))};min_ratio={float(min_ratio):.3f}"
            ),
            tesseract_missing=False,
        )

    return OsdEstimate(
        rotate_deg=int(best_rotation),
        confidence=best_conf,
        votes_total=total_votes,
        vote_ratio=ratio,
        samples_checked=len(sampled),
        sampled_paths=tuple(sampled),
        warning="",
        tesseract_missing=False,
    )


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


def _validate_rotation_with_ocr(
    image_paths: Sequence[Path],
    candidate_rotate_deg: int,
    sample_count: int,
    timeout_sec: float,
    lang: str,
    psm: int,
    min_word_conf: float,
    word_bonus: float,
    min_words: int,
    min_score_delta: float,
) -> OcrValidation:
    sampled = _select_uniform_subset(image_paths, sample_count)
    if not sampled:
        return OcrValidation(
            final_rotate_deg=int(candidate_rotate_deg),
            used=0,
            best_rotate_deg=int(candidate_rotate_deg),
            best_score=0.0,
            candidate_score=0.0,
            warning="ocr_no_samples",
            tesseract_missing=False,
        )

    scores: Dict[int, float] = {}
    word_counts: Dict[int, int] = {}
    for rotate_deg in VALID_ROTATIONS:
        conf_sum_total = 0.0
        words_total = 0
        for path in sampled:
            conf_sum, words, err = _run_tesseract_ocr_word_score(
                image_path=path,
                rotate_deg_clockwise=rotate_deg,
                timeout_sec=timeout_sec,
                lang=lang,
                psm=psm,
                min_word_conf=min_word_conf,
            )
            if err == "tesseract_not_found":
                return OcrValidation(
                    final_rotate_deg=int(candidate_rotate_deg),
                    used=len(sampled),
                    best_rotate_deg=int(candidate_rotate_deg),
                    best_score=0.0,
                    candidate_score=0.0,
                    warning=err,
                    tesseract_missing=True,
                )
            conf_sum_total += conf_sum
            words_total += words
        scores[rotate_deg] = float(conf_sum_total + float(words_total) * float(word_bonus))
        word_counts[rotate_deg] = int(words_total)

    best_deg = max(
        VALID_ROTATIONS,
        key=lambda deg: (
            scores.get(deg, 0.0),
            word_counts.get(deg, 0),
            deg == int(candidate_rotate_deg),
            deg == 0,
            -deg,
        ),
    )
    best_score = float(scores.get(best_deg, 0.0))
    candidate_score = float(scores.get(int(candidate_rotate_deg), 0.0))
    best_words = int(word_counts.get(best_deg, 0))
    delta = float(best_score - candidate_score)

    if best_words < int(max(0, min_words)):
        return OcrValidation(
            final_rotate_deg=int(candidate_rotate_deg),
            used=len(sampled),
            best_rotate_deg=int(best_deg),
            best_score=best_score,
            candidate_score=candidate_score,
            warning=f"ocr_no_text_support:best_words={best_words}",
            tesseract_missing=False,
        )

    if best_deg != int(candidate_rotate_deg) and delta >= float(min_score_delta):
        return OcrValidation(
            final_rotate_deg=int(best_deg),
            used=len(sampled),
            best_rotate_deg=int(best_deg),
            best_score=best_score,
            candidate_score=candidate_score,
            warning=(
                f"ocr_override:{int(candidate_rotate_deg)}->{int(best_deg)};"
                f"delta={delta:.2f};best_words={best_words}"
            ),
            tesseract_missing=False,
        )

    return OcrValidation(
        final_rotate_deg=int(candidate_rotate_deg),
        used=len(sampled),
        best_rotate_deg=int(best_deg),
        best_score=best_score,
        candidate_score=candidate_score,
        warning=f"ocr_confirm_or_weak_override:best={int(best_deg)};delta={delta:.2f};best_words={best_words}",
        tesseract_missing=False,
    )


def _rotate_image_in_place(path: Path, rotate_deg_clockwise: int) -> None:
    rotate = int(rotate_deg_clockwise) % 360
    if rotate == 0:
        return
    with Image.open(path) as img:
        fmt = img.format
        rotated = img.rotate(-rotate, expand=True)
        save_kwargs = {}
        if fmt:
            save_kwargs["format"] = fmt
        if (fmt or "").upper() in {"JPEG", "JPG"}:
            save_kwargs["quality"] = 95
        rotated.save(path, **save_kwargs)


def _write_results_csv(path: Path, rows: Iterable[FolderResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "folder_path",
                "folder_name",
                "images_total",
                "sample_image",
                "osd_rotate_deg",
                "osd_confidence",
                "osd_votes_total",
                "osd_vote_ratio",
                "osd_samples_checked",
                "ocr_validation_used",
                "ocr_best_rotate_deg",
                "ocr_best_score",
                "ocr_candidate_score",
                "decision",
                "applied",
                "rotated_images",
                "errors",
                "warning",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "folder_path": row.folder.as_posix(),
                    "folder_name": row.folder.name,
                    "images_total": row.images_total,
                    "sample_image": row.sample_image,
                    "osd_rotate_deg": row.osd_rotate_deg,
                    "osd_confidence": f"{row.osd_confidence:.4f}",
                    "osd_votes_total": row.osd_votes_total,
                    "osd_vote_ratio": f"{row.osd_vote_ratio:.4f}",
                    "osd_samples_checked": row.osd_samples_checked,
                    "ocr_validation_used": row.ocr_validation_used,
                    "ocr_best_rotate_deg": row.ocr_best_rotate_deg,
                    "ocr_best_score": f"{row.ocr_best_score:.4f}",
                    "ocr_candidate_score": f"{row.ocr_candidate_score:.4f}",
                    "decision": row.decision,
                    "applied": row.applied,
                    "rotated_images": row.rotated_images,
                    "errors": row.errors,
                    "warning": row.warning,
                }
            )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rotate acquisition images per folder via robust OSD.")
    p.add_argument("--dataset-root", type=Path, default=Path("data/Dataset L_T"))
    p.add_argument("--min-orientation-confidence", type=float, default=0.0)
    p.add_argument("--osd-timeout-sec", type=float, default=2.5)
    p.add_argument("--osd-samples-per-folder", type=int, default=5)
    p.add_argument("--osd-min-votes", type=int, default=2)
    p.add_argument("--osd-min-ratio", type=float, default=0.60)
    p.add_argument("--disable-ocr-validation", action="store_true")
    p.add_argument("--ocr-validation-samples", type=int, default=2)
    p.add_argument("--ocr-timeout-sec", type=float, default=3.0)
    p.add_argument("--ocr-lang", type=str, default="eng")
    p.add_argument("--ocr-psm", type=int, default=6)
    p.add_argument("--ocr-min-word-conf", type=float, default=35.0)
    p.add_argument("--ocr-score-word-bonus", type=float, default=4.0)
    p.add_argument("--ocr-validation-min-words", type=int, default=2)
    p.add_argument("--ocr-validation-min-score-delta", type=float, default=12.0)
    p.add_argument("--max-folders", type=int, default=0, help="0=all")
    p.add_argument("--dry-run", action="store_true", help="Do not modify files, only report decisions.")
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "artifacts/10_active_pipeline/pipeline_fss_head/runs/rotation_one_sample_by_folder/rotation_actions.csv"
        ),
    )
    p.add_argument("--log-interval", type=int, default=10)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.min_orientation_confidence < 0:
        raise ValueError("--min-orientation-confidence must be >= 0.")
    if args.osd_timeout_sec <= 0:
        raise ValueError("--osd-timeout-sec must be > 0.")
    if args.osd_samples_per_folder <= 0:
        raise ValueError("--osd-samples-per-folder must be > 0.")
    if args.osd_min_votes <= 0:
        raise ValueError("--osd-min-votes must be > 0.")
    if not (0.0 <= args.osd_min_ratio <= 1.0):
        raise ValueError("--osd-min-ratio must be in [0,1].")
    if args.ocr_validation_samples <= 0:
        raise ValueError("--ocr-validation-samples must be > 0.")
    if args.ocr_timeout_sec <= 0:
        raise ValueError("--ocr-timeout-sec must be > 0.")
    if args.ocr_min_word_conf < 0.0:
        raise ValueError("--ocr-min-word-conf must be >= 0.")
    if args.ocr_score_word_bonus < 0.0:
        raise ValueError("--ocr-score-word-bonus must be >= 0.")
    if args.ocr_validation_min_words < 0:
        raise ValueError("--ocr-validation-min-words must be >= 0.")
    if args.ocr_validation_min_score_delta < 0.0:
        raise ValueError("--ocr-validation-min-score-delta must be >= 0.")
    if args.max_folders < 0:
        raise ValueError("--max-folders must be >= 0.")

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise RuntimeError(f"Dataset root not found: {dataset_root}")

    folders = [p for p in sorted(dataset_root.iterdir()) if p.is_dir()]
    if args.max_folders > 0:
        folders = folders[: args.max_folders]
    if not folders:
        raise RuntimeError(f"No folders found in {dataset_root}")

    results: List[FolderResult] = []
    start = time.time()
    tesseract_missing = False
    for idx, folder in enumerate(folders, start=1):
        images = _collect_acquisition_images(folder)
        if not images:
            results.append(
                FolderResult(
                    folder=folder,
                    images_total=0,
                    sample_image="",
                    osd_rotate_deg=0,
                    osd_confidence=0.0,
                    osd_votes_total=0,
                    osd_vote_ratio=0.0,
                    osd_samples_checked=0,
                    ocr_validation_used=0,
                    ocr_best_rotate_deg=0,
                    ocr_best_score=0.0,
                    ocr_candidate_score=0.0,
                    decision="skip_no_images",
                    applied=0,
                    rotated_images=0,
                    errors=0,
                    warning="no_acquisition_images",
                )
            )
            continue

        osd = _estimate_rotation_with_osd(
            image_paths=images,
            sample_count=args.osd_samples_per_folder,
            timeout_sec=args.osd_timeout_sec,
            min_confidence=args.min_orientation_confidence,
            min_votes=args.osd_min_votes,
            min_ratio=args.osd_min_ratio,
        )
        sample_image = (
            osd.sampled_paths[len(osd.sampled_paths) // 2].as_posix()
            if osd.sampled_paths
            else images[len(images) // 2].as_posix()
        )
        if osd.tesseract_missing:
            tesseract_missing = True
            results.append(
                FolderResult(
                    folder=folder,
                    images_total=len(images),
                    sample_image=sample_image,
                    osd_rotate_deg=0,
                    osd_confidence=0.0,
                    osd_votes_total=osd.votes_total,
                    osd_vote_ratio=osd.vote_ratio,
                    osd_samples_checked=osd.samples_checked,
                    ocr_validation_used=0,
                    ocr_best_rotate_deg=0,
                    ocr_best_score=0.0,
                    ocr_candidate_score=0.0,
                    decision="skip_tesseract_missing",
                    applied=0,
                    rotated_images=0,
                    errors=0,
                    warning=osd.warning or "tesseract_not_found",
                )
            )
            break

        if osd.rotate_deg is None:
            decision = "skip_osd_low_support" if "osd_low_support" in osd.warning else "skip_osd_no_votes"
            results.append(
                FolderResult(
                    folder=folder,
                    images_total=len(images),
                    sample_image=sample_image,
                    osd_rotate_deg=0,
                    osd_confidence=osd.confidence,
                    osd_votes_total=osd.votes_total,
                    osd_vote_ratio=osd.vote_ratio,
                    osd_samples_checked=osd.samples_checked,
                    ocr_validation_used=0,
                    ocr_best_rotate_deg=0,
                    ocr_best_score=0.0,
                    ocr_candidate_score=0.0,
                    decision=decision,
                    applied=0,
                    rotated_images=0,
                    errors=0,
                    warning=osd.warning,
                )
            )
            continue

        candidate_rotate = int(osd.rotate_deg)
        ocr_validation = OcrValidation(
            final_rotate_deg=candidate_rotate,
            used=0,
            best_rotate_deg=candidate_rotate,
            best_score=0.0,
            candidate_score=0.0,
            warning="",
            tesseract_missing=False,
        )
        if candidate_rotate != 0 and not args.disable_ocr_validation:
            ocr_validation = _validate_rotation_with_ocr(
                image_paths=images,
                candidate_rotate_deg=candidate_rotate,
                sample_count=args.ocr_validation_samples,
                timeout_sec=args.ocr_timeout_sec,
                lang=args.ocr_lang,
                psm=args.ocr_psm,
                min_word_conf=args.ocr_min_word_conf,
                word_bonus=args.ocr_score_word_bonus,
                min_words=args.ocr_validation_min_words,
                min_score_delta=args.ocr_validation_min_score_delta,
            )
            if ocr_validation.tesseract_missing:
                tesseract_missing = True
                results.append(
                    FolderResult(
                        folder=folder,
                        images_total=len(images),
                        sample_image=sample_image,
                        osd_rotate_deg=candidate_rotate,
                        osd_confidence=osd.confidence,
                        osd_votes_total=osd.votes_total,
                        osd_vote_ratio=osd.vote_ratio,
                        osd_samples_checked=osd.samples_checked,
                        ocr_validation_used=ocr_validation.used,
                        ocr_best_rotate_deg=ocr_validation.best_rotate_deg,
                        ocr_best_score=ocr_validation.best_score,
                        ocr_candidate_score=ocr_validation.candidate_score,
                        decision="skip_tesseract_missing",
                        applied=0,
                        rotated_images=0,
                        errors=0,
                        warning=ocr_validation.warning or "tesseract_not_found",
                    )
                )
                break

        final_rotate = int(ocr_validation.final_rotate_deg)
        warning_parts: List[str] = []
        if osd.warning:
            warning_parts.append(osd.warning)
        if ocr_validation.warning:
            warning_parts.append(ocr_validation.warning)
        warning = ";".join(part for part in warning_parts if part)

        if final_rotate == 0:
            decision = "keep_0deg" if candidate_rotate == 0 else "keep_0deg_after_ocr_validation"
            results.append(
                FolderResult(
                    folder=folder,
                    images_total=len(images),
                    sample_image=sample_image,
                    osd_rotate_deg=candidate_rotate,
                    osd_confidence=osd.confidence,
                    osd_votes_total=osd.votes_total,
                    osd_vote_ratio=osd.vote_ratio,
                    osd_samples_checked=osd.samples_checked,
                    ocr_validation_used=ocr_validation.used,
                    ocr_best_rotate_deg=ocr_validation.best_rotate_deg,
                    ocr_best_score=ocr_validation.best_score,
                    ocr_candidate_score=ocr_validation.candidate_score,
                    decision=decision,
                    applied=0,
                    rotated_images=0,
                    errors=0,
                    warning=warning,
                )
            )
            continue

        rotated_images = 0
        errors = 0
        if not args.dry_run:
            for img_path in images:
                try:
                    _rotate_image_in_place(img_path, rotate_deg_clockwise=final_rotate)
                    rotated_images += 1
                except Exception:
                    errors += 1

        if args.dry_run:
            decision = (
                "would_rotate_all_in_folder_ocr_override"
                if final_rotate != candidate_rotate
                else "would_rotate_all_in_folder"
            )
        else:
            decision = (
                "rotate_all_in_folder_ocr_override"
                if final_rotate != candidate_rotate
                else "rotate_all_in_folder"
            )
        results.append(
            FolderResult(
                folder=folder,
                images_total=len(images),
                sample_image=sample_image,
                osd_rotate_deg=candidate_rotate,
                osd_confidence=osd.confidence,
                osd_votes_total=osd.votes_total,
                osd_vote_ratio=osd.vote_ratio,
                osd_samples_checked=osd.samples_checked,
                ocr_validation_used=ocr_validation.used,
                ocr_best_rotate_deg=ocr_validation.best_rotate_deg,
                ocr_best_score=ocr_validation.best_score,
                ocr_candidate_score=ocr_validation.candidate_score,
                decision=decision,
                applied=0 if args.dry_run else 1,
                rotated_images=0 if args.dry_run else rotated_images,
                errors=errors,
                warning=warning if errors == 0 else f"{warning};some_images_failed".strip(";"),
            )
        )

        if args.log_interval > 0 and (idx % args.log_interval == 0 or idx == len(folders)):
            elapsed = time.time() - start
            print(f"[{idx:04d}/{len(folders):04d}] elapsed={elapsed:.1f}s", flush=True)

    _write_results_csv(args.output_csv.expanduser().resolve(), results)

    rotated_folders = sum(1 for r in results if r.applied == 1)
    osd_skipped = sum(1 for r in results if r.decision.startswith("skip_osd_"))
    ocr_overrides = sum(1 for r in results if "ocr_override" in r.decision)
    print(f"Folders scanned: {len(results)}", flush=True)
    print(f"Folders rotated: {rotated_folders}", flush=True)
    print(f"Folders skipped by OSD: {osd_skipped}", flush=True)
    print(f"Folders with OCR override: {ocr_overrides}", flush=True)
    if tesseract_missing:
        print("Tesseract not found: run interrupted.", flush=True)
    print(f"Report CSV: {args.output_csv.expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
