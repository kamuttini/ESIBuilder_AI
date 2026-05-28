#!/usr/bin/env python3
"""Train line #16 parameter model (B/CH/MM + TH/P1..P5).

The model is tabular and uses:
- categorical embeddings (manufacturer + probe identifiers)
- numeric geometry/setup features
- multi-task outputs:
  * classification: B, CH, MM
  * regression: TH, P1, P2, P3, P4, P5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REGRESSION_TARGETS: Tuple[str, ...] = ("threshold", "p1", "p2", "p3", "p4", "p5")
NUMERIC_FEATURES: Tuple[str, ...] = (
    "flip_idx",
    "flip_available",
    "fss_group_orientation",
    "fss_video_x",
    "fss_video_y",
    "rect_top_norm",
    "rect_left_norm",
    "rect_bottom_norm",
    "rect_right_norm",
    "rect_width_norm",
    "rect_height_norm",
    "rect_cx_norm",
    "rect_cy_norm",
    "rect_area_norm",
    "rect_aspect",
    "rect_diag_norm",
)
CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "manufacturer",
    "fss_id_echo",
    "fss_id_probe",
    "fss_probe_type",
)


@dataclass(frozen=True)
class ParamRow:
    sample_id: str
    split: str
    dataset_folder: str
    manufacturer: str
    model_name: str
    fss_path: str
    fss_id_echo: int
    fss_id_probe: int
    fss_probe_type: int
    fss_video_x: float
    fss_video_y: float
    fss_group_orientation: int
    flip_idx: int
    flip_available: int
    rect_top: float
    rect_left: float
    rect_bottom: float
    rect_right: float
    rect_width: float
    rect_height: float
    rect_cx_norm: float
    rect_cy_norm: float
    rect_area_norm: float
    b_value: int
    channel: int
    threshold: float
    p1: float
    p2: float
    p3: float
    p4: float
    p5: float
    mm: int


@dataclass
class FeaturePack:
    numeric: np.ndarray
    categorical: np.ndarray


@dataclass
class EncodedTargets:
    b: np.ndarray
    ch: np.ndarray
    mm: np.ndarray
    reg: np.ndarray


@dataclass
class TargetTransform:
    reg_mean: np.ndarray
    reg_std: np.ndarray


@dataclass
class CategoricalVocab:
    name: str
    stoi: Dict[str, int]
    itos: List[str]


@dataclass
class LabelMapping:
    b_values: List[int]
    ch_values: List[int]
    mm_values: List[int]


@dataclass
class Preprocessor:
    numeric_feature_names: Tuple[str, ...]
    categorical_feature_names: Tuple[str, ...]
    numeric_mean: np.ndarray
    numeric_std: np.ndarray
    categorical_vocabs: Dict[str, CategoricalVocab]
    label_mapping: LabelMapping
    target_transform: TargetTransform


class Line16ParamsDataset(Dataset):
    def __init__(
        self,
        features: FeaturePack,
        targets: EncodedTargets,
        row_ids: Sequence[str],
    ) -> None:
        self.numeric = torch.tensor(features.numeric, dtype=torch.float32)
        self.categorical = torch.tensor(features.categorical, dtype=torch.long)
        self.target_b = torch.tensor(targets.b, dtype=torch.long)
        self.target_ch = torch.tensor(targets.ch, dtype=torch.long)
        self.target_mm = torch.tensor(targets.mm, dtype=torch.long)
        self.target_reg = torch.tensor(targets.reg, dtype=torch.float32)
        self.row_ids = list(row_ids)

    def __len__(self) -> int:
        return int(self.numeric.shape[0])

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "numeric": self.numeric[idx],
            "categorical": self.categorical[idx],
            "target_b": self.target_b[idx],
            "target_ch": self.target_ch[idx],
            "target_mm": self.target_mm[idx],
            "target_reg": self.target_reg[idx],
            "row_id": self.row_ids[idx],
        }


class Line16ParamNet(nn.Module):
    def __init__(
        self,
        num_numeric_features: int,
        categorical_cardinalities: Dict[str, int],
        num_b_classes: int,
        num_ch_classes: int,
        num_mm_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cat_feature_names = list(categorical_cardinalities.keys())
        self.embeddings = nn.ModuleDict()
        emb_total = 0
        for name, cardinality in categorical_cardinalities.items():
            emb_dim = max(2, min(32, int(round(math.sqrt(cardinality) * 2.0))))
            self.embeddings[name] = nn.Embedding(cardinality, emb_dim)
            emb_total += emb_dim

        in_dim = num_numeric_features + emb_total
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.head_b = nn.Linear(128, max(1, num_b_classes))
        self.head_ch = nn.Linear(128, max(1, num_ch_classes))
        self.head_mm = nn.Linear(128, max(1, num_mm_classes))
        self.head_reg = nn.Linear(128, len(REGRESSION_TARGETS))

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        pieces = [numeric]
        for idx, name in enumerate(self.cat_feature_names):
            pieces.append(self.embeddings[name](categorical[:, idx]))
        x = torch.cat(pieces, dim=1)
        feat = self.backbone(x)
        return {
            "b": self.head_b(feat),
            "ch": self.head_ch(feat),
            "mm": self.head_mm(feat),
            "reg": self.head_reg(feat),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(prefer: Optional[str]) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _safe_int(value: str, default: int = 0) -> int:
    text = value.strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def _safe_float(value: str, default: float = 0.0) -> float:
    text = value.strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def load_rows(manifest_path: Path) -> List[ParamRow]:
    rows: List[ParamRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                ParamRow(
                    sample_id=row["sample_id"],
                    split=row["split"].strip().lower(),
                    dataset_folder=row["dataset_folder"],
                    manufacturer=row["manufacturer"],
                    model_name=row["model_name"],
                    fss_path=row["fss_path"],
                    fss_id_echo=_safe_int(row.get("fss_id_echo", "0"), default=0),
                    fss_id_probe=_safe_int(row.get("fss_id_probe", "0"), default=0),
                    fss_probe_type=_safe_int(row.get("fss_probe_type", "0"), default=0),
                    fss_video_x=max(1.0, _safe_float(row.get("fss_video_x", "0"), default=1.0)),
                    fss_video_y=max(1.0, _safe_float(row.get("fss_video_y", "0"), default=1.0)),
                    fss_group_orientation=_safe_int(row.get("fss_group_orientation", "4"), default=4),
                    flip_idx=_safe_int(row.get("flip_idx", "0"), default=0),
                    flip_available=_safe_int(row.get("flip_available", "1"), default=1),
                    rect_top=_safe_float(row.get("rect_top", "0"), default=0.0),
                    rect_left=_safe_float(row.get("rect_left", "0"), default=0.0),
                    rect_bottom=_safe_float(row.get("rect_bottom", "0"), default=0.0),
                    rect_right=_safe_float(row.get("rect_right", "0"), default=0.0),
                    rect_width=max(1.0, _safe_float(row.get("rect_width", "1"), default=1.0)),
                    rect_height=max(1.0, _safe_float(row.get("rect_height", "1"), default=1.0)),
                    rect_cx_norm=_safe_float(row.get("rect_cx_norm", "0"), default=0.0),
                    rect_cy_norm=_safe_float(row.get("rect_cy_norm", "0"), default=0.0),
                    rect_area_norm=_safe_float(row.get("rect_area_norm", "0"), default=0.0),
                    b_value=_safe_int(row.get("b_value", "1"), default=1),
                    channel=_safe_int(row.get("channel", "7"), default=7),
                    threshold=max(1e-9, _safe_float(row.get("threshold", "1"), default=1.0)),
                    p1=_safe_float(row.get("p1", "20"), default=20.0),
                    p2=_safe_float(row.get("p2", "120"), default=120.0),
                    p3=_safe_float(row.get("p3", "0"), default=0.0),
                    p4=_safe_float(row.get("p4", "0"), default=0.0),
                    p5=_safe_float(row.get("p5", "0"), default=0.0),
                    mm=_safe_int(row.get("mm", "6"), default=6),
                )
            )
    return rows


def deduplicate_rows(rows: Sequence[ParamRow]) -> List[ParamRow]:
    out: List[ParamRow] = []
    seen: set[Tuple[str, int]] = set()
    for row in rows:
        key = (row.fss_path, row.flip_idx)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _build_numeric_vector(row: ParamRow) -> List[float]:
    vx = max(1.0, row.fss_video_x)
    vy = max(1.0, row.fss_video_y)
    rect_top_norm = row.rect_top / vy
    rect_left_norm = row.rect_left / vx
    rect_bottom_norm = row.rect_bottom / vy
    rect_right_norm = row.rect_right / vx
    rect_width_norm = row.rect_width / vx
    rect_height_norm = row.rect_height / vy
    rect_aspect = row.rect_width / max(1.0, row.rect_height)
    rect_diag_norm = math.sqrt((row.rect_width ** 2) + (row.rect_height ** 2)) / math.sqrt((vx ** 2) + (vy ** 2))

    return [
        float(row.flip_idx),
        float(row.flip_available),
        float(row.fss_group_orientation),
        float(row.fss_video_x),
        float(row.fss_video_y),
        rect_top_norm,
        rect_left_norm,
        rect_bottom_norm,
        rect_right_norm,
        rect_width_norm,
        rect_height_norm,
        float(row.rect_cx_norm),
        float(row.rect_cy_norm),
        float(row.rect_area_norm),
        rect_aspect,
        rect_diag_norm,
    ]


def _build_categorical_dict(row: ParamRow) -> Dict[str, str]:
    return {
        "manufacturer": row.manufacturer.strip() or "UNKNOWN",
        "fss_id_echo": str(row.fss_id_echo),
        "fss_id_probe": str(row.fss_id_probe),
        "fss_probe_type": str(row.fss_probe_type),
    }


def _build_regression_target(row: ParamRow) -> List[float]:
    # Log transform keeps threshold stable across very large ranges.
    return [
        math.log10(max(1e-9, row.threshold)),
        row.p1,
        row.p2,
        row.p3,
        row.p4,
        row.p5,
    ]


def _fit_vocab(values: Sequence[str]) -> CategoricalVocab:
    counter: Dict[str, int] = {}
    for value in values:
        counter[value] = counter.get(value, 0) + 1
    ordered = ["__UNK__"] + [
        key for key, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    stoi = {token: idx for idx, token in enumerate(ordered)}
    return CategoricalVocab(name="", stoi=stoi, itos=ordered)


def _encode_categories(rows: Sequence[ParamRow], vocabs: Dict[str, CategoricalVocab]) -> np.ndarray:
    mat = np.zeros((len(rows), len(CATEGORICAL_FEATURES)), dtype=np.int64)
    for i, row in enumerate(rows):
        cat = _build_categorical_dict(row)
        for j, feat_name in enumerate(CATEGORICAL_FEATURES):
            vocab = vocabs[feat_name]
            mat[i, j] = vocab.stoi.get(cat[feat_name], 0)
    return mat


def _encode_numeric(rows: Sequence[ParamRow]) -> np.ndarray:
    arr = np.zeros((len(rows), len(NUMERIC_FEATURES)), dtype=np.float32)
    for i, row in enumerate(rows):
        arr[i, :] = np.array(_build_numeric_vector(row), dtype=np.float32)
    return arr


def _fit_numeric_stats(train_numeric: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_numeric.mean(axis=0)
    std = train_numeric.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _normalize_numeric(numeric: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((numeric - mean[None, :]) / std[None, :]).astype(np.float32)


def _fit_label_mapping(train_rows: Sequence[ParamRow]) -> LabelMapping:
    b_values = sorted({row.b_value for row in train_rows})
    ch_values = sorted({row.channel for row in train_rows})
    mm_values = sorted({row.mm for row in train_rows})
    return LabelMapping(
        b_values=b_values,
        ch_values=ch_values,
        mm_values=mm_values,
    )


def _encode_labels(rows: Sequence[ParamRow], values: List[int], getter) -> np.ndarray:
    lookup = {value: idx for idx, value in enumerate(values)}
    out = np.zeros((len(rows),), dtype=np.int64)
    for i, row in enumerate(rows):
        value = getter(row)
        if value not in lookup:
            raise RuntimeError(f"Label value {value} missing in training label mapping.")
        out[i] = lookup[value]
    return out


def _fit_target_transform(train_rows: Sequence[ParamRow]) -> TargetTransform:
    reg = np.array([_build_regression_target(row) for row in train_rows], dtype=np.float32)
    mean = reg.mean(axis=0)
    std = reg.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return TargetTransform(reg_mean=mean.astype(np.float32), reg_std=std.astype(np.float32))


def _encode_reg_targets(rows: Sequence[ParamRow], target_transform: TargetTransform) -> np.ndarray:
    reg = np.array([_build_regression_target(row) for row in rows], dtype=np.float32)
    reg = (reg - target_transform.reg_mean[None, :]) / target_transform.reg_std[None, :]
    return reg.astype(np.float32)


def _denormalize_reg(
    reg_pred_norm: np.ndarray,
    target_transform: TargetTransform,
) -> np.ndarray:
    raw = reg_pred_norm * target_transform.reg_std[None, :] + target_transform.reg_mean[None, :]
    out = raw.copy()
    out[:, 0] = np.power(10.0, out[:, 0])
    return out


def _prepare_preprocessor(
    train_rows: Sequence[ParamRow],
) -> Preprocessor:
    numeric_train = _encode_numeric(train_rows)
    num_mean, num_std = _fit_numeric_stats(numeric_train)

    vocabs: Dict[str, CategoricalVocab] = {}
    for name in CATEGORICAL_FEATURES:
        values = [_build_categorical_dict(row)[name] for row in train_rows]
        vocab = _fit_vocab(values)
        vocab.name = name
        vocabs[name] = vocab

    label_mapping = _fit_label_mapping(train_rows)
    target_transform = _fit_target_transform(train_rows)

    return Preprocessor(
        numeric_feature_names=NUMERIC_FEATURES,
        categorical_feature_names=CATEGORICAL_FEATURES,
        numeric_mean=num_mean,
        numeric_std=num_std,
        categorical_vocabs=vocabs,
        label_mapping=label_mapping,
        target_transform=target_transform,
    )


def _encode_split(rows: Sequence[ParamRow], preprocessor: Preprocessor) -> Tuple[FeaturePack, EncodedTargets]:
    numeric = _encode_numeric(rows)
    numeric = _normalize_numeric(numeric, preprocessor.numeric_mean, preprocessor.numeric_std)
    categorical = _encode_categories(rows, preprocessor.categorical_vocabs)

    targets = EncodedTargets(
        b=_encode_labels(rows, preprocessor.label_mapping.b_values, getter=lambda r: r.b_value),
        ch=_encode_labels(rows, preprocessor.label_mapping.ch_values, getter=lambda r: r.channel),
        mm=_encode_labels(rows, preprocessor.label_mapping.mm_values, getter=lambda r: r.mm),
        reg=_encode_reg_targets(rows, preprocessor.target_transform),
    )
    return FeaturePack(numeric=numeric, categorical=categorical), targets


def _build_loader(
    rows: Sequence[ParamRow],
    features: FeaturePack,
    targets: EncodedTargets,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = Line16ParamsDataset(
        features=features,
        targets=targets,
        row_ids=[row.sample_id for row in rows],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


def _compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = np.zeros((num_classes,), dtype=np.float64)
    present = counts > 0
    if present.any():
        weights[present] = counts[present].sum() / (counts[present] * present.sum())
        norm = weights[present].mean()
        if norm > 0:
            weights[present] /= norm
    if not present.all():
        weights[~present] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


def _unit_class_weights(num_classes: int) -> torch.Tensor:
    return torch.ones((num_classes,), dtype=torch.float32)


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> Tuple[float, int]:
    if logits.shape[1] <= 1:
        return 1.0, int(target.shape[0])
    pred = torch.argmax(logits, dim=1)
    correct = int((pred == target).sum().item())
    total = int(target.shape[0])
    return (correct / total if total > 0 else 0.0), total


def _balanced_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    if logits.shape[1] <= 1:
        return 1.0
    pred = torch.argmax(logits, dim=1)
    recalls: List[float] = []
    num_classes = logits.shape[1]
    for cls in range(num_classes):
        mask = target == cls
        denom = int(mask.sum().item())
        if denom == 0:
            continue
        numer = int((pred[mask] == cls).sum().item())
        recalls.append(numer / denom)
    if not recalls:
        return 0.0
    return float(sum(recalls) / len(recalls))


def _classification_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    if logits.shape[1] <= 1:
        return logits.new_zeros(())
    if class_weights is not None:
        return F.cross_entropy(logits, target, weight=class_weights)
    return F.cross_entropy(logits, target)


def run_epoch(
    model: Line16ParamNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    class_weights: Dict[str, Optional[torch.Tensor]],
    target_transform: TargetTransform,
    epoch: int,
    phase: str,
    cls_loss_weight: Dict[str, float],
    reg_loss_weight: float,
    log_interval: int,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_samples = 0
    loss_tot = 0.0
    loss_b_tot = 0.0
    loss_ch_tot = 0.0
    loss_mm_tot = 0.0
    loss_reg_tot = 0.0

    b_correct = 0
    ch_correct = 0
    mm_correct = 0

    mm_bal_acc_sum = 0.0

    reg_abs_err_sum = np.zeros((len(REGRESSION_TARGETS),), dtype=np.float64)
    reg_sq_err_sum = np.zeros((len(REGRESSION_TARGETS),), dtype=np.float64)

    step_count = len(loader)
    t0 = time.time()

    for step_idx, batch in enumerate(loader, start=1):
        numeric = batch["numeric"].to(device, non_blocking=True)
        categorical = batch["categorical"].to(device, non_blocking=True)
        target_b = batch["target_b"].to(device, non_blocking=True)
        target_ch = batch["target_ch"].to(device, non_blocking=True)
        target_mm = batch["target_mm"].to(device, non_blocking=True)
        target_reg = batch["target_reg"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(numeric, categorical)

            loss_b = _classification_loss(outputs["b"], target_b, class_weights["b"])
            loss_ch = _classification_loss(outputs["ch"], target_ch, class_weights["ch"])
            loss_mm = _classification_loss(outputs["mm"], target_mm, class_weights["mm"])
            loss_reg = F.smooth_l1_loss(outputs["reg"], target_reg)

            total_loss = (
                cls_loss_weight["b"] * loss_b
                + cls_loss_weight["ch"] * loss_ch
                + cls_loss_weight["mm"] * loss_mm
                + reg_loss_weight * loss_reg
            )

            if is_train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

        batch_size = int(numeric.shape[0])
        total_samples += batch_size

        loss_tot += float(total_loss.detach().cpu()) * batch_size
        loss_b_tot += float(loss_b.detach().cpu()) * batch_size
        loss_ch_tot += float(loss_ch.detach().cpu()) * batch_size
        loss_mm_tot += float(loss_mm.detach().cpu()) * batch_size
        loss_reg_tot += float(loss_reg.detach().cpu()) * batch_size

        b_acc, _ = _accuracy(outputs["b"], target_b)
        ch_acc, _ = _accuracy(outputs["ch"], target_ch)
        mm_acc, _ = _accuracy(outputs["mm"], target_mm)
        b_correct += int(round(b_acc * batch_size))
        ch_correct += int(round(ch_acc * batch_size))
        mm_correct += int(round(mm_acc * batch_size))

        mm_bal_acc_sum += _balanced_accuracy(outputs["mm"], target_mm) * batch_size

        pred_reg_norm = outputs["reg"].detach().cpu().numpy()
        true_reg_norm = target_reg.detach().cpu().numpy()
        pred_reg = _denormalize_reg(pred_reg_norm, target_transform)
        true_reg = _denormalize_reg(true_reg_norm, target_transform)

        abs_err = np.abs(pred_reg - true_reg)
        sq_err = (pred_reg - true_reg) ** 2
        reg_abs_err_sum += abs_err.sum(axis=0)
        reg_sq_err_sum += sq_err.sum(axis=0)

        if log_interval > 0 and (step_idx % log_interval == 0 or step_idx == step_count):
            elapsed = time.time() - t0
            progress = step_idx / max(1, step_count)
            eta = (elapsed / progress - elapsed) if progress > 0 else 0.0
            print(
                f"[Epoch {epoch:03d}][{phase}] step {step_idx}/{step_count} "
                f"({progress * 100:.1f}%) loss {loss_tot / max(1, total_samples):.4f} "
                f"mm_acc {mm_correct / max(1, total_samples):.4f} "
                f"elapsed {elapsed:.1f}s eta {max(0.0, eta):.1f}s",
                flush=True,
            )

    mean = lambda x: float(x / max(1, total_samples))
    metrics: Dict[str, float] = {
        "loss_total": mean(loss_tot),
        "loss_b": mean(loss_b_tot),
        "loss_ch": mean(loss_ch_tot),
        "loss_mm": mean(loss_mm_tot),
        "loss_reg": mean(loss_reg_tot),
        "acc_b": float(b_correct / max(1, total_samples)),
        "acc_ch": float(ch_correct / max(1, total_samples)),
        "acc_mm": float(mm_correct / max(1, total_samples)),
        "balanced_acc_mm": float(mm_bal_acc_sum / max(1, total_samples)),
        "samples": float(total_samples),
    }

    for idx, name in enumerate(REGRESSION_TARGETS):
        mae = reg_abs_err_sum[idx] / max(1, total_samples)
        rmse = math.sqrt(reg_sq_err_sum[idx] / max(1, total_samples))
        metrics[f"mae_{name}"] = float(mae)
        metrics[f"rmse_{name}"] = float(rmse)

    return metrics


def _save_test_predictions(
    path: Path,
    rows: Sequence[ParamRow],
    model: Line16ParamNet,
    loader: DataLoader,
    device: torch.device,
    preprocessor: Preprocessor,
) -> None:
    model.eval()
    b_values = preprocessor.label_mapping.b_values
    ch_values = preprocessor.label_mapping.ch_values
    mm_values = preprocessor.label_mapping.mm_values

    out_rows: List[Dict[str, object]] = []
    row_cursor = 0

    with torch.no_grad():
        for batch in loader:
            numeric = batch["numeric"].to(device, non_blocking=True)
            categorical = batch["categorical"].to(device, non_blocking=True)
            outputs = model(numeric, categorical)

            pred_b_idx = torch.argmax(outputs["b"], dim=1).detach().cpu().numpy()
            pred_ch_idx = torch.argmax(outputs["ch"], dim=1).detach().cpu().numpy()
            pred_mm_idx = torch.argmax(outputs["mm"], dim=1).detach().cpu().numpy()
            pred_reg = outputs["reg"].detach().cpu().numpy()
            pred_reg = _denormalize_reg(pred_reg, preprocessor.target_transform)

            batch_size = int(numeric.shape[0])
            for i in range(batch_size):
                row = rows[row_cursor]
                row_cursor += 1

                out_rows.append(
                    {
                        "sample_id": row.sample_id,
                        "dataset_folder": row.dataset_folder,
                        "flip_idx": row.flip_idx,
                        "true_b": row.b_value,
                        "pred_b": b_values[int(pred_b_idx[i])] if b_values else row.b_value,
                        "true_ch": row.channel,
                        "pred_ch": ch_values[int(pred_ch_idx[i])] if ch_values else row.channel,
                        "true_mm": row.mm,
                        "pred_mm": mm_values[int(pred_mm_idx[i])] if mm_values else row.mm,
                        "true_threshold": row.threshold,
                        "pred_threshold": float(pred_reg[i, 0]),
                        "true_p1": row.p1,
                        "pred_p1": float(pred_reg[i, 1]),
                        "true_p2": row.p2,
                        "pred_p2": float(pred_reg[i, 2]),
                        "true_p3": row.p3,
                        "pred_p3": float(pred_reg[i, 3]),
                        "true_p4": row.p4,
                        "pred_p4": float(pred_reg[i, 4]),
                        "true_p5": row.p5,
                        "pred_p5": float(pred_reg[i, 5]),
                    }
                )

    fields = list(out_rows[0].keys()) if out_rows else []
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)


def _serialize_preprocessor(preprocessor: Preprocessor) -> Dict[str, object]:
    vocabs = {}
    for name, vocab in preprocessor.categorical_vocabs.items():
        vocabs[name] = {
            "itos": vocab.itos,
        }
    return {
        "numeric_feature_names": list(preprocessor.numeric_feature_names),
        "categorical_feature_names": list(preprocessor.categorical_feature_names),
        "numeric_mean": preprocessor.numeric_mean.tolist(),
        "numeric_std": preprocessor.numeric_std.tolist(),
        "categorical_vocabs": vocabs,
        "label_mapping": {
            "b_values": list(preprocessor.label_mapping.b_values),
            "ch_values": list(preprocessor.label_mapping.ch_values),
            "mm_values": list(preprocessor.label_mapping.mm_values),
        },
        "target_transform": {
            "reg_mean": preprocessor.target_transform.reg_mean.tolist(),
            "reg_std": preprocessor.target_transform.reg_std.tolist(),
        },
    }


def _restore_preprocessor(blob: Dict[str, object]) -> Preprocessor:
    categorical_vocabs: Dict[str, CategoricalVocab] = {}
    for name, value in blob["categorical_vocabs"].items():
        itos = list(value["itos"])
        stoi = {token: idx for idx, token in enumerate(itos)}
        categorical_vocabs[name] = CategoricalVocab(name=name, stoi=stoi, itos=itos)

    label_mapping_data = blob["label_mapping"]
    target_transform_data = blob["target_transform"]

    return Preprocessor(
        numeric_feature_names=tuple(blob["numeric_feature_names"]),
        categorical_feature_names=tuple(blob["categorical_feature_names"]),
        numeric_mean=np.array(blob["numeric_mean"], dtype=np.float32),
        numeric_std=np.array(blob["numeric_std"], dtype=np.float32),
        categorical_vocabs=categorical_vocabs,
        label_mapping=LabelMapping(
            b_values=[int(x) for x in label_mapping_data["b_values"]],
            ch_values=[int(x) for x in label_mapping_data["ch_values"]],
            mm_values=[int(x) for x in label_mapping_data["mm_values"]],
        ),
        target_transform=TargetTransform(
            reg_mean=np.array(target_transform_data["reg_mean"], dtype=np.float32),
            reg_std=np.array(target_transform_data["reg_std"], dtype=np.float32),
        ),
    )


def encode_single_row_features(row: ParamRow, preprocessor: Preprocessor) -> FeaturePack:
    numeric = np.array([_build_numeric_vector(row)], dtype=np.float32)
    numeric = _normalize_numeric(numeric, preprocessor.numeric_mean, preprocessor.numeric_std)
    categorical = np.zeros((1, len(CATEGORICAL_FEATURES)), dtype=np.int64)
    cat = _build_categorical_dict(row)
    for j, name in enumerate(CATEGORICAL_FEATURES):
        vocab = preprocessor.categorical_vocabs[name]
        categorical[0, j] = vocab.stoi.get(cat[name], 0)
    return FeaturePack(numeric=numeric, categorical=categorical)


def decode_prediction(
    outputs: Dict[str, torch.Tensor],
    preprocessor: Preprocessor,
) -> Dict[str, float]:
    b_idx = int(torch.argmax(outputs["b"], dim=1)[0].item())
    ch_idx = int(torch.argmax(outputs["ch"], dim=1)[0].item())
    mm_idx = int(torch.argmax(outputs["mm"], dim=1)[0].item())

    reg_norm = outputs["reg"].detach().cpu().numpy()
    reg_raw = _denormalize_reg(reg_norm, preprocessor.target_transform)[0]

    return {
        "b_value": float(preprocessor.label_mapping.b_values[b_idx]),
        "channel": float(preprocessor.label_mapping.ch_values[ch_idx]),
        "mm": float(preprocessor.label_mapping.mm_values[mm_idx]),
        "threshold": float(reg_raw[0]),
        "p1": float(reg_raw[1]),
        "p2": float(reg_raw[2]),
        "p3": float(reg_raw[3]),
        "p4": float(reg_raw[4]),
        "p5": float(reg_raw[5]),
    }


def load_checkpoint_for_inference(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[Line16ParamNet, Preprocessor, Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    preprocessor = _restore_preprocessor(checkpoint["preprocessor"])

    model = Line16ParamNet(
        num_numeric_features=len(preprocessor.numeric_feature_names),
        categorical_cardinalities={
            name: len(preprocessor.categorical_vocabs[name].itos)
            for name in preprocessor.categorical_feature_names
        },
        num_b_classes=len(preprocessor.label_mapping.b_values),
        num_ch_classes=len(preprocessor.label_mapping.ch_values),
        num_mm_classes=len(preprocessor.label_mapping.mm_values),
        dropout=float(checkpoint["args"].get("dropout", 0.15)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, preprocessor, checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train multi-task model for line #16 params.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_line16_dataset/manifest_line16_params.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/30_models/line16_params_training"),
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--cls-weight-b", type=float, default=0.25)
    parser.add_argument("--cls-weight-ch", type=float, default=0.5)
    parser.add_argument("--cls-weight-mm", type=float, default=1.0)
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--disable-class-weights", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)

    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(manifest_path)
    if args.deduplicate:
        rows = deduplicate_rows(rows)

    split_rows = {
        "train": [row for row in rows if row.split == "train"],
        "val": [row for row in rows if row.split == "val"],
        "test": [row for row in rows if row.split == "test"],
    }
    if not split_rows["train"] or not split_rows["val"] or not split_rows["test"]:
        raise RuntimeError(
            "Train/val/test must contain at least one sample. "
            f"Counts: train={len(split_rows['train'])}, "
            f"val={len(split_rows['val'])}, test={len(split_rows['test'])}"
        )

    preprocessor = _prepare_preprocessor(split_rows["train"])

    train_features, train_targets = _encode_split(split_rows["train"], preprocessor)
    val_features, val_targets = _encode_split(split_rows["val"], preprocessor)
    test_features, test_targets = _encode_split(split_rows["test"], preprocessor)

    train_loader = _build_loader(
        rows=split_rows["train"],
        features=train_features,
        targets=train_targets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = _build_loader(
        rows=split_rows["val"],
        features=val_features,
        targets=val_targets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    test_loader = _build_loader(
        rows=split_rows["test"],
        features=test_features,
        targets=test_targets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    device = choose_device(args.device)
    print(f"Device: {device}", flush=True)
    print(
        "Samples train/val/test: "
        f"{len(split_rows['train'])}/{len(split_rows['val'])}/{len(split_rows['test'])}",
        flush=True,
    )

    model = Line16ParamNet(
        num_numeric_features=len(NUMERIC_FEATURES),
        categorical_cardinalities={
            name: len(preprocessor.categorical_vocabs[name].itos)
            for name in CATEGORICAL_FEATURES
        },
        num_b_classes=len(preprocessor.label_mapping.b_values),
        num_ch_classes=len(preprocessor.label_mapping.ch_values),
        num_mm_classes=len(preprocessor.label_mapping.mm_values),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    if args.disable_class_weights:
        class_weights = {
            "b": _unit_class_weights(len(preprocessor.label_mapping.b_values)).to(device),
            "ch": _unit_class_weights(len(preprocessor.label_mapping.ch_values)).to(device),
            "mm": _unit_class_weights(len(preprocessor.label_mapping.mm_values)).to(device),
        }
    else:
        class_weights = {
            "b": _compute_class_weights(train_targets.b, len(preprocessor.label_mapping.b_values)).to(device),
            "ch": _compute_class_weights(train_targets.ch, len(preprocessor.label_mapping.ch_values)).to(device),
            "mm": _compute_class_weights(train_targets.mm, len(preprocessor.label_mapping.mm_values)).to(device),
        }

    cls_loss_weight = {
        "b": float(args.cls_weight_b),
        "ch": float(args.cls_weight_ch),
        "mm": float(args.cls_weight_mm),
    }

    history: List[Dict[str, object]] = []
    best_val_loss = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            class_weights=class_weights,
            target_transform=preprocessor.target_transform,
            epoch=epoch,
            phase="train",
            cls_loss_weight=cls_loss_weight,
            reg_loss_weight=float(args.reg_weight),
            log_interval=args.log_interval,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            class_weights=class_weights,
            target_transform=preprocessor.target_transform,
            epoch=epoch,
            phase="val",
            cls_loss_weight=cls_loss_weight,
            reg_loss_weight=float(args.reg_weight),
            log_interval=args.log_interval,
        )
        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss_total']:.4f}, mm_acc {train_metrics['acc_mm']:.4f}, "
            f"mae_th {train_metrics['mae_threshold']:.1f} | "
            f"val loss {val_metrics['loss_total']:.4f}, mm_acc {val_metrics['acc_mm']:.4f}, "
            f"mae_th {val_metrics['mae_threshold']:.1f}",
            flush=True,
        )

        val_loss = float(val_metrics["loss_total"])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "preprocessor": _serialize_preprocessor(preprocessor),
                    "args": {
                        key: (str(value) if isinstance(value, Path) else value)
                        for key, value in vars(args).items()
                    },
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(
                    "Early stopping: no validation improvement for "
                    f"{args.early_stopping_patience} epochs.",
                    flush=True,
                )
                break

    if not best_model_path.exists():
        raise RuntimeError("Best model checkpoint missing.")

    model, preprocessor_loaded, _ = load_checkpoint_for_inference(best_model_path, device=device)

    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        optimizer=None,
        class_weights=class_weights,
        target_transform=preprocessor_loaded.target_transform,
        epoch=best_epoch if best_epoch > 0 else args.epochs,
        phase="test",
        cls_loss_weight=cls_loss_weight,
        reg_loss_weight=float(args.reg_weight),
        log_interval=args.log_interval,
    )

    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test": test_metrics,
        "history": history,
        "train_samples": len(split_rows["train"]),
        "val_samples": len(split_rows["val"]),
        "test_samples": len(split_rows["test"]),
        "device": str(device),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    test_pred_path = output_dir / "test_predictions.csv"
    _save_test_predictions(
        path=test_pred_path,
        rows=split_rows["test"],
        model=model,
        loader=test_loader,
        device=device,
        preprocessor=preprocessor_loaded,
    )

    print(f"Best model: {best_model_path}", flush=True)
    print(f"Metrics: {metrics_path}", flush=True)
    print(f"Test predictions: {test_pred_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
