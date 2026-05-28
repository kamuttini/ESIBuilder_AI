#!/usr/bin/env python3
"""Evaluate hybrid vendor recognition (CNN + OCR fusion) on HDMI frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import subprocess
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


STOP_TOKENS = {
    "MM",
    "CM",
    "DB",
    "FPS",
    "GAIN",
    "DEPTH",
    "DYN",
    "FREQ",
    "PRF",
    "BMODE",
    "COLOR",
    "PWR",
    "M",
    "B",
}
TOKEN_SANITIZE_RE = re.compile(r"[^A-Z0-9]+")
DEFAULT_VENDOR_HINT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "Hitachi": ("HITACHI", "ARIETTA", "ALOKA"),
    "Esaote": ("ESAOTE", "MYLAB"),
    "BK": ("BK", "BKMEDICAL", "FLEXFOCUS", "PROFOCUS", "SPECTO", "BK3000", "BK5000"),
    "GE": ("GE", "LOGIQ", "VOLUSON", "VERSANA"),
}


@dataclass(frozen=True)
class SampleRow:
    image_path: Path
    split: str
    manufacturer: str
    model_name: str


class VendorInferenceDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[SampleRow],
        label_to_idx: Dict[str, int],
        image_size: int,
    ) -> None:
        self.rows = list(rows)
        self.label_to_idx = label_to_idx
        self.image_size = image_size
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):  # type: ignore[override]
        row = self.rows[idx]
        with Image.open(row.image_path) as img:
            image = img.convert("RGB")
            image = TF.resize(
                image,
                size=[self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.to_tensor(image)
            tensor = self.normalize(tensor)

        target = self.label_to_idx[row.manufacturer]
        return tensor, target


class VendorClassifier(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
        self.backbone = backbone
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.30),
            torch.nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feats = self.backbone(images)
        return self.head(feats)


@dataclass
class OcrNaiveBayes:
    class_names: List[str]
    token_to_idx: Dict[str, int]
    log_priors: np.ndarray
    log_token_probs: np.ndarray

    def predict_proba(self, tokens: Sequence[str]) -> np.ndarray:
        logp = self.log_priors.copy()
        if self.log_token_probs.size == 0:
            return _softmax(logp)

        token_counts = Counter(token for token in tokens if token in self.token_to_idx)
        for token, count in token_counts.items():
            tok_idx = self.token_to_idx[token]
            logp += float(count) * self.log_token_probs[:, tok_idx]
        return _softmax(logp)


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


def load_manifest_rows(manifest_path: Path) -> List[SampleRow]:
    rows: List[SampleRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                SampleRow(
                    image_path=Path(row["image_path"]).expanduser().resolve(),
                    split=row["split"].strip().lower(),
                    manufacturer=row["manufacturer"].strip(),
                    model_name=row["model_name"].strip(),
                )
            )
    return rows


def subset_rows(rows: List[SampleRow], limit: int, seed: int) -> List[SampleRow]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    chosen = sorted(idx[:limit])
    return [rows[i] for i in chosen]


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits)
    ex = np.exp(x)
    return ex / np.clip(ex.sum(), a_min=1e-12, a_max=None)


def _compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
    idx_to_label: Sequence[str],
) -> Dict[str, object]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    support = confusion.sum(axis=1)
    pred_count = confusion.sum(axis=0)
    tp = np.diag(confusion)

    per_class: List[Dict[str, object]] = []
    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0
    classes_with_support = 0

    for i in range(num_classes):
        precision = float(tp[i] / pred_count[i]) if pred_count[i] > 0 else 0.0
        recall = float(tp[i] / support[i]) if support[i] > 0 else 0.0
        denom = precision + recall
        f1 = float(2.0 * precision * recall / denom) if denom > 0 else 0.0

        if support[i] > 0:
            macro_p += precision
            macro_r += recall
            macro_f1 += f1
            classes_with_support += 1

        per_class.append(
            {
                "manufacturer": idx_to_label[i],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support[i]),
            }
        )

    macro_div = max(1, classes_with_support)
    accuracy = float(tp.sum() / max(1, confusion.sum()))
    return {
        "accuracy": accuracy,
        "macro_precision": macro_p / macro_div,
        "macro_recall": macro_r / macro_div,
        "macro_f1": macro_f1 / macro_div,
        "classes_with_support": classes_with_support,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _write_per_class_csv(path: Path, per_class: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["manufacturer", "precision", "recall", "f1", "support"])
        for row in per_class:
            writer.writerow(
                [
                    row["manufacturer"],
                    f"{float(row['precision']):.6f}",
                    f"{float(row['recall']):.6f}",
                    f"{float(row['f1']):.6f}",
                    int(row["support"]),
                ]
            )


def _write_confusion_csv(path: Path, confusion: Sequence[Sequence[int]], labels: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, confusion):
            writer.writerow([label, *row])


def _normalize_token(raw: str) -> Optional[str]:
    value = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii").upper()
    value = TOKEN_SANITIZE_RE.sub("", value)
    if len(value) < 2:
        return None
    if value.isdigit():
        return None
    if value in STOP_TOKENS:
        return None
    return value


def _extract_tokens_with_tesseract(
    image_path: Path,
    lang: str,
    psm: int,
    conf_min: float,
    timeout_sec: float,
) -> List[str]:
    cmd = [
        "tesseract",
        str(image_path),
        "stdout",
        "-l",
        lang,
        "--psm",
        str(psm),
        "tsv",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return []

    if proc.returncode != 0 and not proc.stdout:
        return []

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []

    header = lines[0].split("\t")
    try:
        text_idx = header.index("text")
        conf_idx = header.index("conf")
    except ValueError:
        return []

    out_tokens: List[str] = []
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) <= max(text_idx, conf_idx):
            continue
        raw_text = cols[text_idx].strip()
        if not raw_text:
            continue
        try:
            conf_value = float(cols[conf_idx].strip())
        except ValueError:
            continue
        if conf_value < conf_min:
            continue
        token = _normalize_token(raw_text)
        if token:
            out_tokens.append(token)
    return out_tokens


def _load_ocr_cache(cache_path: Path) -> Dict[str, List[str]]:
    if not cache_path.exists():
        return {}
    cache: Dict[str, List[str]] = {}
    with cache_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_path = row.get("image_path")
            tokens = row.get("tokens")
            if isinstance(image_path, str) and isinstance(tokens, list):
                cache[image_path] = [str(tok) for tok in tokens]
    return cache


def _append_ocr_cache(cache_path: Path, image_path: str, tokens: Sequence[str]) -> None:
    with cache_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"image_path": image_path, "tokens": list(tokens)}, ensure_ascii=False) + "\n")


def collect_ocr_tokens(
    rows: Sequence[SampleRow],
    cache_path: Path,
    cache_key_mode: str,
    lang: str,
    psm: int,
    conf_min: float,
    timeout_sec: float,
    log_interval: int,
) -> List[List[str]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_ocr_cache(cache_path)
    result: List[List[str]] = []
    start = time.time()
    new_cache_entries = 0

    for idx, row in enumerate(rows, start=1):
        if cache_key_mode == "folder":
            image_dir = row.image_path.parent
            folder_path = image_dir.parent if image_dir.name == "image_samples" else image_dir
            key = f"folder::{folder_path.as_posix()}"
        else:
            key = row.image_path.as_posix()
        if key in cache:
            tokens = cache[key]
        else:
            tokens = _extract_tokens_with_tesseract(
                image_path=row.image_path,
                lang=lang,
                psm=psm,
                conf_min=conf_min,
                timeout_sec=timeout_sec,
            )
            cache[key] = tokens
            _append_ocr_cache(cache_path, key, tokens)
            new_cache_entries += 1
        result.append(tokens)

        if log_interval > 0 and (idx % log_interval == 0 or idx == len(rows)):
            elapsed = time.time() - start
            mean_tokens = float(sum(len(t) for t in result) / max(1, len(result)))
            print(
                f"OCR {idx}/{len(rows)} | elapsed {elapsed:.1f}s | "
                f"cache_add {new_cache_entries} | avg_tokens {mean_tokens:.2f}",
                flush=True,
            )

    return result


def fit_ocr_naive_bayes(
    docs_tokens: Sequence[Sequence[str]],
    labels_idx: Sequence[int],
    class_names: Sequence[str],
    min_df: int,
    max_df_ratio: float,
) -> OcrNaiveBayes:
    num_classes = len(class_names)
    num_docs = len(docs_tokens)
    if num_docs == 0:
        raise RuntimeError("Nessun documento OCR in train.")

    doc_freq = Counter()
    for tokens in docs_tokens:
        doc_freq.update(set(tokens))

    max_df = max(1, int(max_df_ratio * num_docs))
    vocab = sorted(token for token, df in doc_freq.items() if df >= min_df and df <= max_df)
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}

    class_doc_counts = np.zeros(num_classes, dtype=np.float64)
    for idx in labels_idx:
        class_doc_counts[idx] += 1.0
    class_doc_counts = np.clip(class_doc_counts, a_min=1.0, a_max=None)
    log_priors = np.log(class_doc_counts / class_doc_counts.sum())

    if not vocab:
        return OcrNaiveBayes(
            class_names=list(class_names),
            token_to_idx={},
            log_priors=log_priors,
            log_token_probs=np.zeros((num_classes, 0), dtype=np.float64),
        )

    token_counts = np.ones((num_classes, len(vocab)), dtype=np.float64)
    total_tokens = np.full(num_classes, fill_value=float(len(vocab)), dtype=np.float64)

    for tokens, label in zip(docs_tokens, labels_idx):
        counts = Counter(token for token in tokens if token in token_to_idx)
        for token, count in counts.items():
            tok_idx = token_to_idx[token]
            token_counts[label, tok_idx] += float(count)
            total_tokens[label] += float(count)

    log_token_probs = np.log(token_counts / total_tokens[:, None])
    return OcrNaiveBayes(
        class_names=list(class_names),
        token_to_idx=token_to_idx,
        log_priors=log_priors,
        log_token_probs=log_token_probs,
    )


def _augment_hint_keywords_from_ocr_model(
    ocr_model: OcrNaiveBayes,
    class_names: Sequence[str],
    hint_keyword_map: Dict[str, Tuple[str, ...]],
    topk: int,
    token_min_len: int,
) -> Dict[str, Tuple[str, ...]]:
    if topk <= 0 or ocr_model.log_token_probs.size == 0 or not hint_keyword_map:
        return hint_keyword_map

    idx_to_token = [None] * len(ocr_model.token_to_idx)
    for token, idx in ocr_model.token_to_idx.items():
        idx_to_token[idx] = token

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    updated: Dict[str, Tuple[str, ...]] = {}

    for cls, kws in hint_keyword_map.items():
        if cls not in class_to_idx:
            updated[cls] = kws
            continue

        cls_idx = class_to_idx[cls]
        others = [i for i in range(len(class_names)) if i != cls_idx]
        if not others:
            updated[cls] = kws
            continue

        scores = ocr_model.log_token_probs[cls_idx] - np.mean(ocr_model.log_token_probs[others], axis=0)
        ranked = np.argsort(scores)[::-1]

        new_keywords = set(kws)
        added = 0
        for tok_idx in ranked:
            token = idx_to_token[int(tok_idx)]
            if token is None:
                continue
            if len(token) < max(2, token_min_len):
                continue
            if token.isdigit():
                continue
            if token in STOP_TOKENS:
                continue
            if scores[int(tok_idx)] <= 0:
                continue
            if token not in new_keywords:
                new_keywords.add(token)
                added += 1
            if added >= topk:
                break
        updated[cls] = tuple(sorted(new_keywords))

    return updated


def evaluate_cnn_probs(
    rows: Sequence[SampleRow],
    model: torch.nn.Module,
    label_to_idx: Dict[str, int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[np.ndarray, List[int]]:
    dataset = VendorInferenceDataset(rows=rows, label_to_idx=label_to_idx, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    all_probs: List[np.ndarray] = []
    all_true: List[int] = []

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_true.extend(targets.tolist())

    stacked = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(label_to_idx)), dtype=np.float64)
    return stacked, all_true


def evaluate_probs(
    probs: np.ndarray,
    y_true: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, object]:
    if len(y_true) == 0:
        raise RuntimeError("Split vuoto: impossibile valutare metriche.")
    y_pred = probs.argmax(axis=1).tolist()
    return _compute_classification_metrics(y_true=y_true, y_pred=y_pred, num_classes=len(class_names), idx_to_label=class_names)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid vendor recognition (CNN + OCR fusion)."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/20_datasets/rect_dataset_with_capturemeta_v3/manifest_rect_echo.csv"),
        help="Manifest CSV generato da prepare_ultrasound_rect_dataset.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/10_active_pipeline/pipeline_fss_head/models/vendor_training_no_negative_v2_power/best_model.pt"),
        help="Checkpoint best model prodotto da train_ultrasound_vendor_classifier.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/40_outputs_eval/vendor_hybrid_eval"),
        help="Cartella output metriche e CSV.",
    )
    parser.add_argument("--image-size", type=int, default=0, help="Override image-size rete (0 = da checkpoint).")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--ocr-lang", type=str, default="eng")
    parser.add_argument("--ocr-psm", type=int, default=6)
    parser.add_argument("--ocr-conf-min", type=float, default=35.0)
    parser.add_argument("--ocr-timeout-sec", type=float, default=4.0)
    parser.add_argument("--ocr-min-df", type=int, default=2)
    parser.add_argument("--ocr-max-df-ratio", type=float, default=0.80)
    parser.add_argument(
        "--ocr-cache-key",
        type=str,
        default="image",
        choices=["image", "folder"],
        help="Chiave cache OCR: image (accurato) o folder (piu veloce).",
    )
    parser.add_argument(
        "--ocr-cache",
        type=Path,
        default=None,
        help="Path cache OCR jsonl (default: <output-dir>/ocr_cache.jsonl).",
    )
    parser.add_argument(
        "--fusion-alpha",
        type=float,
        default=-1.0,
        help="Peso CNN in [0,1]. Se <0, fa tuning automatico su validation.",
    )
    parser.add_argument("--alpha-grid-step", type=float, default=0.05)
    parser.add_argument(
        "--enable-ocr-hint",
        action="store_true",
        help="Abilita regola a soglia OCR con keyword vendor.",
    )
    parser.add_argument(
        "--ocr-hint-classes",
        type=str,
        default="Hitachi,Esaote",
        help="Classi su cui applicare OCR hint (CSV).",
    )
    parser.add_argument(
        "--ocr-hint-min-token-hits",
        type=int,
        default=1,
        help="Numero minimo keyword OCR per attivare hint.",
    )
    parser.add_argument(
        "--ocr-hint-min-ocr-prob",
        type=float,
        default=0.70,
        help="Probabilita minima OCR sulla classe hinted.",
    )
    parser.add_argument(
        "--ocr-hint-max-cnn-prob",
        type=float,
        default=1.0,
        help="Probabilita max CNN top1 per permettere l'hint.",
    )
    parser.add_argument(
        "--ocr-hint-alpha",
        type=float,
        default=0.35,
        help="Alpha CNN usato quando hint OCR e' attivo (piu basso => piu OCR).",
    )
    parser.add_argument(
        "--ocr-hint-force-boost",
        type=float,
        default=0.05,
        help="Boost additivo alla classe hinted dopo fusione (0 disabilita).",
    )
    parser.add_argument(
        "--ocr-hint-auto-topk",
        type=int,
        default=12,
        help="Numero token OCR discriminativi da aggiungere automaticamente per classe hinted.",
    )
    parser.add_argument(
        "--ocr-hint-token-min-len",
        type=int,
        default=3,
        help="Lunghezza minima token per auto-keyword OCR.",
    )
    parser.add_argument(
        "--hard-negatives-classes",
        type=str,
        default="Hitachi,Esaote",
        help="Classi focus per export hard negatives (CSV).",
    )
    parser.add_argument(
        "--hard-negatives-max-rows",
        type=int,
        default=3000,
        help="Massimo numero righe hard negatives da esportare.",
    )
    parser.add_argument("--ocr-log-interval", type=int, default=200)
    return parser


def _filter_rows_by_known_classes(rows: Iterable[SampleRow], known: Sequence[str]) -> Tuple[List[SampleRow], int]:
    known_set = set(known)
    out: List[SampleRow] = []
    dropped = 0
    for row in rows:
        if row.manufacturer in known_set:
            out.append(row)
        else:
            dropped += 1
    return out, dropped


def _build_alpha_grid(step: float) -> List[float]:
    step = max(0.001, float(step))
    values: List[float] = []
    alpha = 0.0
    while alpha < 1.0 + (step * 0.5):
        values.append(round(min(1.0, alpha), 4))
        alpha += step
    if values[-1] != 1.0:
        values.append(1.0)
    return sorted(set(values))


def _parse_csv_labels(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_hint_keyword_map(class_names: Sequence[str], hint_classes: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    class_set = set(class_names)
    out: Dict[str, Tuple[str, ...]] = {}
    for cls in hint_classes:
        if cls not in class_set:
            continue
        keywords = set(DEFAULT_VENDOR_HINT_KEYWORDS.get(cls, ()))
        for piece in re.split(r"[^A-Za-z0-9]+", cls):
            token = _normalize_token(piece)
            if token:
                keywords.add(token)
        if keywords:
            out[cls] = tuple(sorted(keywords))
    return out


def _normalize_probs_row(probs: np.ndarray) -> np.ndarray:
    row = np.clip(probs.astype(np.float64, copy=False), a_min=0.0, a_max=None)
    den = float(row.sum())
    if not math.isfinite(den) or den <= 0.0:
        row = np.full_like(row, fill_value=1.0 / max(1, row.shape[0]), dtype=np.float64)
    else:
        row = row / den
    return row


def _normalize_probs_matrix(probs: np.ndarray) -> np.ndarray:
    out = np.zeros_like(probs, dtype=np.float64)
    for i in range(probs.shape[0]):
        out[i] = _normalize_probs_row(probs[i])
    return out


def _fuse_with_ocr_hints(
    probs_cnn: np.ndarray,
    probs_ocr: np.ndarray,
    tokens_list: Sequence[Sequence[str]],
    class_names: Sequence[str],
    base_alpha: float,
    hint_keyword_map: Dict[str, Tuple[str, ...]],
    hint_min_token_hits: int,
    hint_min_ocr_prob: float,
    hint_max_cnn_prob: float,
    hint_alpha: float,
    hint_force_boost: float,
) -> Tuple[np.ndarray, List[Optional[str]], List[int], List[float]]:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    out = np.zeros_like(probs_cnn, dtype=np.float64)
    applied_labels: List[Optional[str]] = []
    matched_hits: List[int] = []
    used_alphas: List[float] = []

    for i, tokens in enumerate(tokens_list):
        base = _normalize_probs_row(base_alpha * probs_cnn[i] + (1.0 - base_alpha) * probs_ocr[i])
        fused = base
        best_label: Optional[str] = None
        best_hits = 0
        alpha_used = base_alpha

        if hint_keyword_map:
            token_set = set(tokens)
            for cls, keywords in hint_keyword_map.items():
                hits = len(token_set.intersection(keywords))
                if hits > best_hits:
                    best_hits = hits
                    best_label = cls

            if best_label is not None and best_hits >= hint_min_token_hits:
                hint_idx = class_to_idx[best_label]
                if (
                    float(probs_ocr[i, hint_idx]) >= hint_min_ocr_prob
                    and float(np.max(probs_cnn[i])) <= hint_max_cnn_prob
                ):
                    alpha_used = min(base_alpha, hint_alpha)
                    fused = _normalize_probs_row(alpha_used * probs_cnn[i] + (1.0 - alpha_used) * probs_ocr[i])
                    if hint_force_boost > 0.0:
                        boosted = fused.copy()
                        boosted[hint_idx] += hint_force_boost
                        fused = _normalize_probs_row(boosted)
                else:
                    best_label = None
                    best_hits = 0

        out[i] = fused
        applied_labels.append(best_label)
        matched_hits.append(best_hits)
        used_alphas.append(alpha_used)

    return out, applied_labels, matched_hits, used_alphas


def _write_hard_negatives_csv(
    out_path: Path,
    rows: Sequence[SampleRow],
    class_names: Sequence[str],
    y_true: Sequence[int],
    probs_cnn: np.ndarray,
    probs_ocr: np.ndarray,
    probs_fused: np.ndarray,
    tokens_list: Sequence[Sequence[str]],
    focus_labels: Sequence[str],
    max_rows: int,
) -> int:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    focus_idx = {class_to_idx[label] for label in focus_labels if label in class_to_idx}
    if not focus_idx:
        return 0

    records: List[Dict[str, object]] = []
    for i, row in enumerate(rows):
        true_idx = int(y_true[i])
        pred_fused = int(np.argmax(probs_fused[i]))
        if pred_fused == true_idx:
            continue
        if true_idx not in focus_idx or pred_fused not in focus_idx:
            continue

        pred_cnn = int(np.argmax(probs_cnn[i]))
        pred_ocr = int(np.argmax(probs_ocr[i]))
        margin = float(probs_fused[i, pred_fused] - probs_fused[i, true_idx])
        records.append(
            {
                "image_path": row.image_path.as_posix(),
                "model_name": row.model_name,
                "true_label": class_names[true_idx],
                "pred_fused": class_names[pred_fused],
                "pred_cnn": class_names[pred_cnn],
                "pred_ocr": class_names[pred_ocr],
                "prob_true_fused": float(probs_fused[i, true_idx]),
                "prob_pred_fused": float(probs_fused[i, pred_fused]),
                "margin_pred_minus_true": margin,
                "token_preview": " ".join(list(tokens_list[i])[:20]),
            }
        )

    records.sort(key=lambda r: float(r["margin_pred_minus_true"]), reverse=True)
    if max_rows > 0:
        records = records[:max_rows]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "image_path",
                "model_name",
                "true_label",
                "pred_fused",
                "pred_cnn",
                "pred_ocr",
                "prob_true_fused",
                "prob_pred_fused",
                "margin_pred_minus_true",
                "token_preview",
            ]
        )
        for rec in records:
            writer.writerow(
                [
                    rec["image_path"],
                    rec["model_name"],
                    rec["true_label"],
                    rec["pred_fused"],
                    rec["pred_cnn"],
                    rec["pred_ocr"],
                    f"{float(rec['prob_true_fused']):.6f}",
                    f"{float(rec['prob_pred_fused']):.6f}",
                    f"{float(rec['margin_pred_minus_true']):.6f}",
                    rec["token_preview"],
                ]
            )

    return len(records)


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)

    manifest_path = args.manifest.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ocr_cache_path = (
        args.ocr_cache.expanduser().resolve()
        if args.ocr_cache is not None
        else (output_dir / "ocr_cache.jsonl")
    )

    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint non trovato: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = ckpt.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise RuntimeError("Checkpoint senza class_names. Ritraina con script vendor aggiornato.")
    class_names = [str(x) for x in class_names]
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}

    image_size_from_ckpt = 0
    ckpt_args = ckpt.get("args")
    if isinstance(ckpt_args, dict):
        raw_size = ckpt_args.get("image_size")
        if isinstance(raw_size, int):
            image_size_from_ckpt = raw_size
        elif isinstance(raw_size, str) and raw_size.isdigit():
            image_size_from_ckpt = int(raw_size)
    image_size = args.image_size if args.image_size > 0 else (image_size_from_ckpt or 320)

    rows = load_manifest_rows(manifest_path)
    split_rows_raw = {
        "train": [row for row in rows if row.split == "train"],
        "val": [row for row in rows if row.split == "val"],
        "test": [row for row in rows if row.split == "test"],
    }
    split_rows: Dict[str, List[SampleRow]] = {}
    dropped_summary: Dict[str, int] = {}
    for split in ("train", "val", "test"):
        filtered, dropped = _filter_rows_by_known_classes(split_rows_raw[split], class_names)
        split_rows[split] = filtered
        dropped_summary[split] = dropped

    for split, limit in (
        ("train", args.max_train_samples),
        ("val", args.max_val_samples),
        ("test", args.max_test_samples),
    ):
        split_seed_offset = {"train": 11, "val": 23, "test": 37}[split]
        split_rows[split] = subset_rows(split_rows[split], limit=limit, seed=args.seed + split_seed_offset)

    if not split_rows["train"] or not split_rows["val"] or not split_rows["test"]:
        raise RuntimeError(
            "Train/val/test devono contenere almeno un campione. "
            f"Conteggi: train={len(split_rows['train'])}, val={len(split_rows['val'])}, test={len(split_rows['test'])}"
        )

    device = choose_device(args.device)
    print(f"Device: {device}", flush=True)
    print(
        f"Samples train/val/test: {len(split_rows['train'])}/{len(split_rows['val'])}/{len(split_rows['test'])}",
        flush=True,
    )
    print(f"Classi da checkpoint: {len(class_names)}", flush=True)
    print(
        f"Rows scartate per classe non nota (train/val/test): "
        f"{dropped_summary['train']}/{dropped_summary['val']}/{dropped_summary['test']}",
        flush=True,
    )

    # CNN probabilities
    model = VendorClassifier(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_probs_cnn, val_true = evaluate_cnn_probs(
        rows=split_rows["val"],
        model=model,
        label_to_idx=label_to_idx,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    test_probs_cnn, test_true = evaluate_cnn_probs(
        rows=split_rows["test"],
        model=model,
        label_to_idx=label_to_idx,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    # OCR extraction + OCR NB training
    print("OCR train split...", flush=True)
    train_tokens = collect_ocr_tokens(
        rows=split_rows["train"],
        cache_path=ocr_cache_path,
        cache_key_mode=args.ocr_cache_key,
        lang=args.ocr_lang,
        psm=args.ocr_psm,
        conf_min=args.ocr_conf_min,
        timeout_sec=args.ocr_timeout_sec,
        log_interval=args.ocr_log_interval,
    )
    print("OCR val split...", flush=True)
    val_tokens = collect_ocr_tokens(
        rows=split_rows["val"],
        cache_path=ocr_cache_path,
        cache_key_mode=args.ocr_cache_key,
        lang=args.ocr_lang,
        psm=args.ocr_psm,
        conf_min=args.ocr_conf_min,
        timeout_sec=args.ocr_timeout_sec,
        log_interval=args.ocr_log_interval,
    )
    print("OCR test split...", flush=True)
    test_tokens = collect_ocr_tokens(
        rows=split_rows["test"],
        cache_path=ocr_cache_path,
        cache_key_mode=args.ocr_cache_key,
        lang=args.ocr_lang,
        psm=args.ocr_psm,
        conf_min=args.ocr_conf_min,
        timeout_sec=args.ocr_timeout_sec,
        log_interval=args.ocr_log_interval,
    )

    train_labels_idx = [label_to_idx[row.manufacturer] for row in split_rows["train"]]
    ocr_model = fit_ocr_naive_bayes(
        docs_tokens=train_tokens,
        labels_idx=train_labels_idx,
        class_names=class_names,
        min_df=args.ocr_min_df,
        max_df_ratio=args.ocr_max_df_ratio,
    )
    print(f"OCR vocab size: {len(ocr_model.token_to_idx)}", flush=True)

    val_probs_ocr = np.vstack([ocr_model.predict_proba(tokens) for tokens in val_tokens])
    test_probs_ocr = np.vstack([ocr_model.predict_proba(tokens) for tokens in test_tokens])

    hint_classes = _parse_csv_labels(args.ocr_hint_classes)
    hint_keyword_map = _build_hint_keyword_map(class_names, hint_classes) if args.enable_ocr_hint else {}
    if args.enable_ocr_hint:
        hint_keyword_map = _augment_hint_keywords_from_ocr_model(
            ocr_model=ocr_model,
            class_names=class_names,
            hint_keyword_map=hint_keyword_map,
            topk=max(0, args.ocr_hint_auto_topk),
            token_min_len=max(2, args.ocr_hint_token_min_len),
        )
        print(
            "OCR hint attivo su classi: "
            + (", ".join(sorted(hint_keyword_map.keys())) if hint_keyword_map else "(nessuna classe valida)"),
            flush=True,
        )
        for cls in sorted(hint_keyword_map):
            preview = ", ".join(list(hint_keyword_map[cls])[:12])
            print(f"  hint keywords {cls}: {preview}", flush=True)

    # Fuse alpha on validation (or use manual)
    alpha_candidates: List[float]
    if 0.0 <= args.fusion_alpha <= 1.0:
        alpha_candidates = [float(args.fusion_alpha)]
    else:
        alpha_candidates = _build_alpha_grid(args.alpha_grid_step)

    alpha_search_rows: List[Dict[str, object]] = []
    best_alpha = alpha_candidates[0]
    best_val_macro_f1 = -math.inf
    best_val_acc = -math.inf
    best_val_probs_fused_base: Optional[np.ndarray] = None
    best_val_probs_fused: Optional[np.ndarray] = None
    best_val_hint_labels: List[Optional[str]] = []
    best_val_hint_hits: List[int] = []

    for alpha in alpha_candidates:
        val_probs_fused_base = _normalize_probs_matrix(alpha * val_probs_cnn + (1.0 - alpha) * val_probs_ocr)
        val_metrics_base_tmp = evaluate_probs(val_probs_fused_base, val_true, class_names)

        if args.enable_ocr_hint and hint_keyword_map:
            val_probs_fused, val_hint_labels, val_hint_hits, _ = _fuse_with_ocr_hints(
                probs_cnn=val_probs_cnn,
                probs_ocr=val_probs_ocr,
                tokens_list=val_tokens,
                class_names=class_names,
                base_alpha=alpha,
                hint_keyword_map=hint_keyword_map,
                hint_min_token_hits=max(1, args.ocr_hint_min_token_hits),
                hint_min_ocr_prob=max(0.0, min(1.0, args.ocr_hint_min_ocr_prob)),
                hint_max_cnn_prob=max(0.0, min(1.0, args.ocr_hint_max_cnn_prob)),
                hint_alpha=max(0.0, min(1.0, args.ocr_hint_alpha)),
                hint_force_boost=max(0.0, args.ocr_hint_force_boost),
            )
        else:
            val_probs_fused = val_probs_fused_base
            val_hint_labels = [None] * len(val_true)
            val_hint_hits = [0] * len(val_true)

        val_metrics_tmp = evaluate_probs(val_probs_fused, val_true, class_names)
        curr_f1 = float(val_metrics_tmp["macro_f1"])
        curr_acc = float(val_metrics_tmp["accuracy"])
        hints_applied = sum(label is not None for label in val_hint_labels)
        alpha_search_rows.append(
            {
                "alpha": alpha,
                "val_macro_f1_base": float(val_metrics_base_tmp["macro_f1"]),
                "val_accuracy_base": float(val_metrics_base_tmp["accuracy"]),
                "val_macro_f1_fused": curr_f1,
                "val_accuracy_fused": curr_acc,
                "val_hints_applied": hints_applied,
                "val_hint_avg_hits": float(sum(val_hint_hits) / max(1, len(val_hint_hits))),
            }
        )

        if (curr_f1 > best_val_macro_f1) or (
            math.isclose(curr_f1, best_val_macro_f1) and curr_acc > best_val_acc
        ):
            best_val_macro_f1 = curr_f1
            best_val_acc = curr_acc
            best_alpha = alpha
            best_val_probs_fused_base = val_probs_fused_base
            best_val_probs_fused = val_probs_fused
            best_val_hint_labels = val_hint_labels
            best_val_hint_hits = val_hint_hits

    assert best_val_probs_fused is not None
    assert best_val_probs_fused_base is not None
    print(
        f"Best fusion alpha (CNN weight): {best_alpha:.3f} | "
        f"val macro-F1: {best_val_macro_f1:.4f} | val acc: {best_val_acc:.4f}",
        flush=True,
    )

    test_probs_fused_base = _normalize_probs_matrix(best_alpha * test_probs_cnn + (1.0 - best_alpha) * test_probs_ocr)
    if args.enable_ocr_hint and hint_keyword_map:
        test_probs_fused, test_hint_labels, test_hint_hits, _ = _fuse_with_ocr_hints(
            probs_cnn=test_probs_cnn,
            probs_ocr=test_probs_ocr,
            tokens_list=test_tokens,
            class_names=class_names,
            base_alpha=best_alpha,
            hint_keyword_map=hint_keyword_map,
            hint_min_token_hits=max(1, args.ocr_hint_min_token_hits),
            hint_min_ocr_prob=max(0.0, min(1.0, args.ocr_hint_min_ocr_prob)),
            hint_max_cnn_prob=max(0.0, min(1.0, args.ocr_hint_max_cnn_prob)),
            hint_alpha=max(0.0, min(1.0, args.ocr_hint_alpha)),
            hint_force_boost=max(0.0, args.ocr_hint_force_boost),
        )
    else:
        test_probs_fused = test_probs_fused_base
        test_hint_labels = [None] * len(test_true)
        test_hint_hits = [0] * len(test_true)

    val_metrics_cnn = evaluate_probs(val_probs_cnn, val_true, class_names)
    val_metrics_ocr = evaluate_probs(val_probs_ocr, val_true, class_names)
    val_metrics_fused_base = evaluate_probs(best_val_probs_fused_base, val_true, class_names)
    val_metrics_fused = evaluate_probs(best_val_probs_fused, val_true, class_names)

    test_metrics_cnn = evaluate_probs(test_probs_cnn, test_true, class_names)
    test_metrics_ocr = evaluate_probs(test_probs_ocr, test_true, class_names)
    test_metrics_fused_base = evaluate_probs(test_probs_fused_base, test_true, class_names)
    test_metrics_fused = evaluate_probs(test_probs_fused, test_true, class_names)

    val_hint_counter = Counter(label for label in best_val_hint_labels if label is not None)
    test_hint_counter = Counter(label for label in test_hint_labels if label is not None)

    metrics = {
        "checkpoint": str(checkpoint_path),
        "image_size": image_size,
        "class_names": class_names,
        "num_classes": len(class_names),
        "samples": {
            "train": len(split_rows["train"]),
            "val": len(split_rows["val"]),
            "test": len(split_rows["test"]),
        },
        "dropped_rows_unknown_class": dropped_summary,
        "ocr": {
            "lang": args.ocr_lang,
            "psm": args.ocr_psm,
            "conf_min": args.ocr_conf_min,
            "timeout_sec": args.ocr_timeout_sec,
            "min_df": args.ocr_min_df,
            "max_df_ratio": args.ocr_max_df_ratio,
            "vocab_size": len(ocr_model.token_to_idx),
            "cache_path": str(ocr_cache_path),
            "cache_key_mode": args.ocr_cache_key,
        },
        "fusion": {
            "alpha_candidates": [float(row["alpha"]) for row in alpha_search_rows],
            "alpha_search": alpha_search_rows,
            "best_alpha": best_alpha,
            "ocr_hint_enabled": args.enable_ocr_hint,
            "ocr_hint_classes": hint_classes,
            "ocr_hint_keywords": {k: list(v) for k, v in sorted(hint_keyword_map.items())},
            "ocr_hint_settings": {
                "min_token_hits": max(1, args.ocr_hint_min_token_hits),
                "min_ocr_prob": max(0.0, min(1.0, args.ocr_hint_min_ocr_prob)),
                "max_cnn_prob": max(0.0, min(1.0, args.ocr_hint_max_cnn_prob)),
                "hint_alpha": max(0.0, min(1.0, args.ocr_hint_alpha)),
                "hint_force_boost": max(0.0, args.ocr_hint_force_boost),
                "auto_topk": max(0, args.ocr_hint_auto_topk),
                "auto_token_min_len": max(2, args.ocr_hint_token_min_len),
            },
            "val_hints_applied": int(sum(label is not None for label in best_val_hint_labels)),
            "test_hints_applied": int(sum(label is not None for label in test_hint_labels)),
            "val_hints_applied_by_class": dict(sorted(val_hint_counter.items())),
            "test_hints_applied_by_class": dict(sorted(test_hint_counter.items())),
            "val_hint_avg_hits": float(sum(best_val_hint_hits) / max(1, len(best_val_hint_hits))),
            "test_hint_avg_hits": float(sum(test_hint_hits) / max(1, len(test_hint_hits))),
        },
        "val": {
            "cnn": val_metrics_cnn,
            "ocr": val_metrics_ocr,
            "fused_base": val_metrics_fused_base,
            "fused": val_metrics_fused,
        },
        "test": {
            "cnn": test_metrics_cnn,
            "ocr": test_metrics_ocr,
            "fused_base": test_metrics_fused_base,
            "fused": test_metrics_fused,
        },
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV outputs
    alpha_csv = output_dir / "fusion_alpha_search.csv"
    with alpha_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "alpha_cnn",
                "val_macro_f1_base",
                "val_accuracy_base",
                "val_macro_f1_fused",
                "val_accuracy_fused",
                "val_hints_applied",
                "val_hint_avg_hits",
            ]
        )
        for row in alpha_search_rows:
            writer.writerow(
                [
                    f"{float(row['alpha']):.4f}",
                    f"{float(row['val_macro_f1_base']):.6f}",
                    f"{float(row['val_accuracy_base']):.6f}",
                    f"{float(row['val_macro_f1_fused']):.6f}",
                    f"{float(row['val_accuracy_fused']):.6f}",
                    int(row["val_hints_applied"]),
                    f"{float(row['val_hint_avg_hits']):.6f}",
                ]
            )

    _write_per_class_csv(output_dir / "test_per_manufacturer_metrics_cnn.csv", test_metrics_cnn["per_class"])
    _write_per_class_csv(output_dir / "test_per_manufacturer_metrics_ocr.csv", test_metrics_ocr["per_class"])
    _write_per_class_csv(output_dir / "test_per_manufacturer_metrics_fused_base.csv", test_metrics_fused_base["per_class"])
    _write_per_class_csv(output_dir / "test_per_manufacturer_metrics_fused.csv", test_metrics_fused["per_class"])
    _write_confusion_csv(output_dir / "test_confusion_matrix_fused_base.csv", test_metrics_fused_base["confusion_matrix"], class_names)
    _write_confusion_csv(output_dir / "test_confusion_matrix_fused.csv", test_metrics_fused["confusion_matrix"], class_names)

    focus_labels = _parse_csv_labels(args.hard_negatives_classes)
    hard_slug = "_".join(
        re.sub(r"[^A-Za-z0-9]+", "", label).lower() for label in focus_labels if label.strip()
    ) or "focus"
    hard_negatives_path = output_dir / f"hard_negatives_{hard_slug}.csv"
    hard_negatives_rows = _write_hard_negatives_csv(
        out_path=hard_negatives_path,
        rows=split_rows["test"],
        class_names=class_names,
        y_true=test_true,
        probs_cnn=test_probs_cnn,
        probs_ocr=test_probs_ocr,
        probs_fused=test_probs_fused,
        tokens_list=test_tokens,
        focus_labels=focus_labels,
        max_rows=max(0, args.hard_negatives_max_rows),
    )
    metrics["hard_negatives"] = {
        "focus_labels": focus_labels,
        "rows_written": hard_negatives_rows,
        "path": str(hard_negatives_path),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "Test metrics | "
        f"CNN macro-F1 {test_metrics_cnn['macro_f1']:.4f} | "
        f"OCR macro-F1 {test_metrics_ocr['macro_f1']:.4f} | "
        f"FUSED_BASE macro-F1 {test_metrics_fused_base['macro_f1']:.4f} | "
        f"FUSED macro-F1 {test_metrics_fused['macro_f1']:.4f}",
        flush=True,
    )
    print(f"Metriche complete: {metrics_path}", flush=True)
    print(f"Alpha search: {alpha_csv}", flush=True)
    print(f"Per-class fused: {output_dir / 'test_per_manufacturer_metrics_fused.csv'}", flush=True)
    print(
        f"Hard negatives ({','.join(focus_labels) if focus_labels else 'n/a'}): "
        f"{hard_negatives_rows} righe in {hard_negatives_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
