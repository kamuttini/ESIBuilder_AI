#!/usr/bin/env python3
"""Utilities to parse and compare ESI .fss files.

Goal: provide a compatibility safety net while reimplementing ESIBuilder flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple


MIN_FSS_LINES = 23
MAX_FSS_LINES = 26

FIELD_NAMES: Dict[int, str] = {
    1: "VERSION",
    2: "ID_ECHO",
    3: "ID_PROBE",
    4: "PROBE_TYPE",
    5: "KIT_NEEDLE_GUIDE",
    6: "VIDEO_INPUT",
    7: "VIDEO_INPUT_SIZE_X",
    8: "VIDEO_INPUT_SIZE_Y",
    9: "VIDEO_X_SIZE",
    10: "VIDEO_Y_SIZE",
    11: "RECT_ECHO",
    12: "GROUP_ORIENTATION",
    13: "RECT_NAME_ECHO",
    14: "RECT_NAME_PROBE",
    15: "PROIBITED_SCREEN",
    16: "RECT_ORIENTATION",
    17: "RECT_DEPTH",
    18: "VECT_DEPTH",
    19: "PIXEL_RATIO_X",
    20: "PIXEL_RATIO_Y",
    21: "SCALE_LINE",
    22: "CENTRE_DISTANCE",
    23: "ANGLE",
    24: "RECT_TRANS",
    25: "ID_NEXT_PROBE",
    26: "BIPLANA_RECOGNITION_MODE",
}

_TOKEN_SPLIT_RE = re.compile(r"[|;,]")


@dataclass
class FssFile:
    path: Path
    lines: List[str]
    fields: Dict[int, str]

    @property
    def is_biplane(self) -> bool:
        return len(self.lines) >= 26


@dataclass
class FieldComparison:
    field_id: int
    field_name: str
    exact_equal: bool
    semantic_equal: bool
    reason: str
    reference: Optional[str]
    candidate: Optional[str]


@dataclass
class FssComparisonResult:
    reference_path: Path
    candidate_path: Path
    byte_identical: bool
    semantic_equal: bool
    line_count_equal: bool
    field_results: List[FieldComparison]

    @property
    def differing_fields(self) -> List[FieldComparison]:
        return [result for result in self.field_results if not result.semantic_equal]


def load_fss(path: str | Path) -> FssFile:
    file_path = Path(path).expanduser().resolve()
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if lines and lines[0].startswith("\ufeff"):
        lines[0] = lines[0].lstrip("\ufeff")

    if len(lines) < MIN_FSS_LINES:
        raise ValueError(
            f"{file_path} has {len(lines)} lines, expected at least {MIN_FSS_LINES}."
        )

    if len(lines) > MAX_FSS_LINES:
        # Keep current behavior permissive while warning callers via metadata.
        # We only map the standard line numbers for compatibility checks.
        pass

    fields: Dict[int, str] = {}
    for idx, line in enumerate(lines[:MAX_FSS_LINES], start=1):
        fields[idx] = line
    return FssFile(path=file_path, lines=lines, fields=fields)


def compare_fss(
    reference_path: str | Path,
    candidate_path: str | Path,
    float_tolerance: float = 1e-6,
) -> FssComparisonResult:
    reference = load_fss(reference_path)
    candidate = load_fss(candidate_path)

    ref_text = "\n".join(reference.lines)
    cand_text = "\n".join(candidate.lines)

    field_ids = sorted(set(reference.fields.keys()) | set(candidate.fields.keys()))
    field_results: List[FieldComparison] = []

    for field_id in field_ids:
        ref_value = reference.fields.get(field_id)
        cand_value = candidate.fields.get(field_id)
        field_name = FIELD_NAMES.get(field_id, f"FIELD_{field_id}")

        if ref_value is None or cand_value is None:
            field_results.append(
                FieldComparison(
                    field_id=field_id,
                    field_name=field_name,
                    exact_equal=False,
                    semantic_equal=False,
                    reason="Field missing in one file.",
                    reference=ref_value,
                    candidate=cand_value,
                )
            )
            continue

        if ref_value == cand_value:
            field_results.append(
                FieldComparison(
                    field_id=field_id,
                    field_name=field_name,
                    exact_equal=True,
                    semantic_equal=True,
                    reason="Exact match.",
                    reference=ref_value,
                    candidate=cand_value,
                )
            )
            continue

        semantic_equal, reason = _semantic_compare_line(
            ref_value, cand_value, float_tolerance
        )
        field_results.append(
            FieldComparison(
                field_id=field_id,
                field_name=field_name,
                exact_equal=False,
                semantic_equal=semantic_equal,
                reason=reason,
                reference=ref_value,
                candidate=cand_value,
            )
        )

    semantic_equal = all(result.semantic_equal for result in field_results)

    return FssComparisonResult(
        reference_path=reference.path,
        candidate_path=candidate.path,
        byte_identical=(ref_text == cand_text),
        semantic_equal=semantic_equal,
        line_count_equal=(len(reference.lines) == len(candidate.lines)),
        field_results=field_results,
    )


def _semantic_compare_line(
    ref_line: str, cand_line: str, float_tolerance: float
) -> Tuple[bool, str]:
    ref_tokens = _tokenize(ref_line)
    cand_tokens = _tokenize(cand_line)

    if len(ref_tokens) != len(cand_tokens):
        return (
            False,
            f"Different token count ({len(ref_tokens)} vs {len(cand_tokens)}).",
        )

    for idx, (ref_tok, cand_tok) in enumerate(zip(ref_tokens, cand_tokens)):
        ref_num = _to_float(ref_tok)
        cand_num = _to_float(cand_tok)

        if ref_num is not None and cand_num is not None:
            if abs(ref_num - cand_num) > float_tolerance:
                return (
                    False,
                    f"Numeric token mismatch at position {idx}: {ref_tok} vs {cand_tok}.",
                )
        else:
            if ref_tok != cand_tok:
                return (
                    False,
                    f"Text token mismatch at position {idx}: {ref_tok} vs {cand_tok}.",
                )

    return True, "Semantically equivalent."


def _tokenize(line: str) -> List[str]:
    pieces = _TOKEN_SPLIT_RE.split(line.strip())
    return [token.strip() for token in pieces if token.strip() != ""]


def _to_float(token: str) -> Optional[float]:
    try:
        return float(token)
    except ValueError:
        return None

