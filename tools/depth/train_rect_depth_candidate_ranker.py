#!/usr/bin/env python3
"""Train a lightweight ranker over RECT_DEPTH OCR candidate features."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


FEATURES = [
    "candidate_rank",
    "cluster_score",
    "token_count",
    "image_support",
    "unique_values",
    "cm_ratio",
    "mm_ratio",
    "depth_hint_ratio",
    "d_hint_ratio",
    "p_hint_ratio",
    "r_hint_ratio",
    "scale_hint_ratio",
    "fps_ips_ratio",
    "side_score",
    "accessory_score",
    "echo_center_penalty",
    "expected_ratio",
    "plausible_ratio",
    "ocr_conf",
    "snap_error_mm",
    "value_error_mm",
    "box_variant_expanded",
    "box_variant_embedded",
    "pred_width",
    "pred_height",
    "ocr_text_len",
    "ocr_digit_group_count",
    "ocr_has_fps_ips",
    "ocr_has_forbidden_marker",
    "ocr_has_hz_text",
    "ocr_has_percent",
    "ocr_has_mi_tis",
    "ocr_has_c2_text",
    "ocr_has_probe_model",
    "ocr_has_time_like_text",
    "ocr_bad_suffix_after_number",
    "ocr_has_letter_hint",
    "ocr_text_has_d",
    "ocr_text_has_p",
    "ocr_text_has_r",
    "ocr_text_has_depth",
    "ocr_d_left_of_number",
    "ocr_d_right_of_number",
    "ocr_has_cm_text",
    "ocr_has_mm_text",
    "ocr_multi_number",
    "ocr_single_depth_expr",
    "wide_text_box",
    "manual_prior_candidate",
    "manual_hint_present",
    "manual_hint_r",
    "manual_hint_p",
    "manual_hint_d",
    "manual_hint_depth",
    "manual_hint_cm",
    "manual_hint_mm",
    "manual_hint_scale",
    "manual_hint_match_score",
    "manual_hint_mismatch_score",
]


def _f(row: Dict[str, str], key: str) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return 0.0
        out = float(value)
        return out if math.isfinite(out) else 0.0
    except Exception:
        return 0.0


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _group_key(row: Dict[str, str]) -> str:
    return row.get("fss_path") or row.get("config_folder") or row.get("sample_key", "")


def _split_groups(rows: Sequence[Dict[str, str]], val_ratio: float) -> Tuple[set[str], set[str]]:
    groups = sorted({_group_key(r) for r in rows})
    val_count = max(1, int(round(len(groups) * val_ratio))) if len(groups) > 1 else 0
    val = set(groups[-val_count:]) if val_count else set()
    train = set(groups) - val
    return train, val


def _matrix(rows: Sequence[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([[_f(r, name) for name in FEATURES] for r in rows], dtype=np.float32)
    y = np.array([int(float(r.get("label", "0") or 0)) for r in rows], dtype=np.int32)
    return x, y


def _top1_metrics(rows: Sequence[Dict[str, str]], scores: Sequence[float]) -> Dict[str, float]:
    by_sample: Dict[str, List[Tuple[Dict[str, str], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_sample[row["sample_key"]].append((row, float(score)))
    total = len(by_sample)
    if total == 0:
        return {"samples": 0, "top1_accuracy": 0.0, "oracle_recall": 0.0, "mean_top1_center_px": 0.0}
    top1_ok = 0
    oracle_ok = 0
    top1_center = []
    for sample_rows in by_sample.values():
        top = max(sample_rows, key=lambda item: item[1])[0]
        top1_ok += int(int(top.get("label", "0")) == 1)
        oracle_ok += int(any(int(r.get("label", "0")) == 1 for r, _s in sample_rows))
        try:
            top1_center.append(float(top.get("center_error_px", "nan")))
        except Exception:
            pass
    return {
        "samples": total,
        "top1_accuracy": top1_ok / total,
        "oracle_recall": oracle_ok / total,
        "mean_top1_center_px": float(np.mean(top1_center)) if top1_center else 0.0,
    }


def train(args: argparse.Namespace) -> Dict[str, object]:
    rows = _load_rows(args.candidates_csv.expanduser().resolve())
    rows = [r for r in rows if r.get("label") in {"0", "1"}]
    train_groups, val_groups = _split_groups(rows, float(args.val_ratio))
    train_rows = [r for r in rows if _group_key(r) in train_groups]
    val_rows = [r for r in rows if _group_key(r) in val_groups]
    if not val_rows:
        val_rows = train_rows
    x_train, y_train = _matrix(train_rows)
    x_val, y_val = _matrix(val_rows)
    if len(set(y_train.tolist())) < 2:
        raise RuntimeError("Need both positive and negative candidate rows to train the ranker.")

    model = HistGradientBoostingClassifier(
        max_iter=int(args.max_iter),
        learning_rate=float(args.learning_rate),
        max_leaf_nodes=int(args.max_leaf_nodes),
        l2_regularization=float(args.l2_regularization),
        random_state=int(args.seed),
    )
    model.fit(x_train, y_train)
    val_scores = model.predict_proba(x_val)[:, 1]
    train_scores = model.predict_proba(x_train)[:, 1]

    def safe_auc(y, scores):
        return float(roc_auc_score(y, scores)) if len(set(y.tolist())) > 1 else None

    def safe_ap(y, scores):
        return float(average_precision_score(y, scores)) if len(set(y.tolist())) > 1 else None

    out_model = args.output_model.expanduser().resolve()
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, out_model)

    summary = {
        "rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "positive_rows": int(sum(int(r["label"]) for r in rows)),
        "features": FEATURES,
        "train_label_counts": dict(Counter(y_train.tolist())),
        "val_label_counts": dict(Counter(y_val.tolist())),
        "train_auc": safe_auc(y_train, train_scores),
        "val_auc": safe_auc(y_val, val_scores),
        "train_average_precision": safe_ap(y_train, train_scores),
        "val_average_precision": safe_ap(y_val, val_scores),
        "train_top1": _top1_metrics(train_rows, train_scores),
        "val_top1": _top1_metrics(val_rows, val_scores),
        "model": out_model.as_posix(),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight RECT_DEPTH OCR candidate ranker.")
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--max-iter", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--l2-regularization", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    summary = train(args)
    if args.summary_json:
        out = args.summary_json.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
