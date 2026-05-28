#!/usr/bin/env python3
"""Train a tabular MLP to regress scale geometry targets.

Targets:
- x1_norm, x2_norm, y1_norm, y2_norm
- length_mm, tick_mm

This baseline is useful to validate manifest quality and data consistency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

FEATURE_NUMERIC: Tuple[str, ...] = (
    "depth_index",
    "depth_mm",
    "video_x_size",
    "video_y_size",
)
FEATURE_CATEGORICAL: Tuple[str, ...] = ("config_folder",)
TARGETS: Tuple[str, ...] = (
    "x1_norm",
    "x2_norm",
    "y1_norm",
    "y2_norm",
    "length_mm",
    "tick_mm",
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    split: str
    config_folder: str
    video_x_size: float
    video_y_size: float
    features_num: np.ndarray
    target: np.ndarray


class TabularDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        cat_stoi: Dict[str, int],
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
    ) -> None:
        self.row_ids = [s.sample_id for s in samples]
        x = np.stack([s.features_num for s in samples]).astype(np.float32)
        y = np.stack([s.target for s in samples]).astype(np.float32)
        self.video_x = np.array([s.video_x_size for s in samples], dtype=np.float32)
        self.video_y = np.array([s.video_y_size for s in samples], dtype=np.float32)

        x_std_safe = x_std.copy()
        x_std_safe[x_std_safe == 0.0] = 1.0
        y_std_safe = y_std.copy()
        y_std_safe[y_std_safe == 0.0] = 1.0

        self.x = torch.from_numpy((x - x_mean) / x_std_safe).float()
        self.y = torch.from_numpy((y - y_mean) / y_std_safe).float()
        self.cat = torch.tensor(
            [cat_stoi.get(s.config_folder, 0) for s in samples], dtype=torch.long
        )

        self.y_mean = y_mean.astype(np.float32)
        self.y_std = y_std_safe.astype(np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "x": self.x[idx],
            "cat": self.cat[idx],
            "y": self.y[idx],
            "video_x": self.video_x[idx],
            "video_y": self.video_y[idx],
            "row_id": self.row_ids[idx],
        }


class ScaleMLP(nn.Module):
    def __init__(self, in_num: int, cat_cardinality: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        emb_dim = max(2, min(32, int(round(math.sqrt(cat_cardinality) * 2.0))))
        self.emb = nn.Embedding(cat_cardinality, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(in_num + emb_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = torch.cat([x_num, self.emb(x_cat)], dim=1)
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(prefer: str | None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _f(text: str) -> float:
    return float(text.strip())


def _i(text: str) -> int:
    return int(float(text.strip()))


def load_samples(manifest: Path) -> List[Sample]:
    rows: List[Sample] = []
    with manifest.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        needed = {
            "sample_id",
            "split",
            "config_folder",
            "depth_index",
            "depth_mm",
            "video_x_size",
            "video_y_size",
            "x1_norm",
            "x2_norm",
            "y1_norm",
            "y2_norm",
            "length_mm",
            "tick_mm",
        }
        if not needed.issubset(set(reader.fieldnames or [])):
            missing = sorted(needed.difference(set(reader.fieldnames or [])))
            raise RuntimeError(f"Manifest missing columns: {missing}")

        for row in reader:
            split = (row.get("split") or "").strip().lower()
            if split not in {"train", "val", "test"}:
                continue
            try:
                depth_index = _f(row["depth_index"])
                depth_mm = _f(row["depth_mm"])
                video_x = _f(row["video_x_size"])
                video_y = _f(row["video_y_size"])
                target = np.array(
                    [
                        _f(row["x1_norm"]),
                        _f(row["x2_norm"]),
                        _f(row["y1_norm"]),
                        _f(row["y2_norm"]),
                        _f(row["length_mm"]),
                        _f(row["tick_mm"]),
                    ],
                    dtype=np.float32,
                )
                features = np.array([depth_index, depth_mm, video_x, video_y], dtype=np.float32)
            except Exception:
                continue

            rows.append(
                Sample(
                    sample_id=row["sample_id"],
                    split=split,
                    config_folder=(row.get("config_folder") or "__UNK__").strip() or "__UNK__",
                    video_x_size=video_x,
                    video_y_size=video_y,
                    features_num=features,
                    target=target,
                )
            )
    return rows


def split_samples(samples: Sequence[Sample]) -> Dict[str, List[Sample]]:
    out: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}
    for s in samples:
        out[s.split].append(s)
    return out


def build_cat_vocab(train_samples: Sequence[Sample]) -> Dict[str, int]:
    values = sorted({s.config_folder for s in train_samples})
    stoi = {"__UNK__": 0}
    for i, v in enumerate(values, start=1):
        stoi[v] = i
    return stoi


def make_loader(ds: TabularDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total = 0.0
    count = 0
    loss_fn = nn.MSELoss()
    for batch in loader:
        x = batch["x"].to(device)
        cat = batch["cat"].to(device)
        y = batch["y"].to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(x, cat)
        loss = loss_fn(pred, y)
        if is_train:
            loss.backward()
            optimizer.step()
        bs = int(x.shape[0])
        total += float(loss.item()) * bs
        count += bs
    return total / max(1, count)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> Dict[str, float]:
    model.eval()
    preds = []
    trues = []
    video_x = []
    video_y = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            cat = batch["cat"].to(device)
            out = model(x, cat).cpu().numpy()
            y = batch["y"].cpu().numpy()
            preds.append(out)
            trues.append(y)
            video_x.append(batch["video_x"].cpu().numpy())
            video_y.append(batch["video_y"].cpu().numpy())

    if not preds:
        return {"loss_mse": float("nan")}

    p = np.concatenate(preds, axis=0)
    t = np.concatenate(trues, axis=0)
    vx = np.concatenate(video_x, axis=0)
    vy = np.concatenate(video_y, axis=0)

    p_real = p * y_std + y_mean
    t_real = t * y_std + y_mean
    abs_err = np.abs(p_real - t_real)

    mse = float(np.mean((p - t) ** 2))
    metrics = {
        "loss_mse": mse,
        "mae_x1_norm": float(np.mean(abs_err[:, 0])),
        "mae_x2_norm": float(np.mean(abs_err[:, 1])),
        "mae_y1_norm": float(np.mean(abs_err[:, 2])),
        "mae_y2_norm": float(np.mean(abs_err[:, 3])),
        "mae_length_mm": float(np.mean(abs_err[:, 4])),
        "mae_tick_mm": float(np.mean(abs_err[:, 5])),
        "mae_x_px_avg": float(np.mean((abs_err[:, 0] * vx + abs_err[:, 1] * vx) * 0.5)),
        "mae_y_px_avg": float(np.mean((abs_err[:, 2] * vy + abs_err[:, 3] * vy) * 0.5)),
    }
    return metrics


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train tabular MLP baseline for scale line geometry.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/23_scale_manifest_ssd_esi1_n1_leakfree/manifest_scale_train_val_test_leakfree.csv"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/30_models/scale_geometry_tabular_v1"),
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")
    return p


def main() -> int:
    args = parser().parse_args()
    set_seed(args.seed)

    manifest = args.manifest.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    samples = load_samples(manifest)
    split = split_samples(samples)
    if not split["train"] or not split["val"] or not split["test"]:
        raise RuntimeError(
            f"Empty split. Counts train/val/test: {len(split['train'])}/{len(split['val'])}/{len(split['test'])}"
        )

    cat_stoi = build_cat_vocab(split["train"])
    train_x = np.stack([s.features_num for s in split["train"]], axis=0)
    train_y = np.stack([s.target for s in split["train"]], axis=0)
    x_mean = train_x.mean(axis=0).astype(np.float32)
    x_std = train_x.std(axis=0).astype(np.float32)
    y_mean = train_y.mean(axis=0).astype(np.float32)
    y_std = train_y.std(axis=0).astype(np.float32)

    ds_train = TabularDataset(split["train"], cat_stoi, x_mean, x_std, y_mean, y_std)
    ds_val = TabularDataset(split["val"], cat_stoi, x_mean, x_std, y_mean, y_std)
    ds_test = TabularDataset(split["test"], cat_stoi, x_mean, x_std, y_mean, y_std)
    ld_train = make_loader(ds_train, batch_size=args.batch_size, shuffle=True)
    ld_val = make_loader(ds_val, batch_size=max(256, args.batch_size), shuffle=False)
    ld_test = make_loader(ds_test, batch_size=max(256, args.batch_size), shuffle=False)

    device = choose_device(args.device if args.device else None)
    model = ScaleMLP(
        in_num=len(FEATURE_NUMERIC),
        cat_cardinality=len(cat_stoi),
        out_dim=len(TARGETS),
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_epoch = 0
    wait = 0
    history: List[Dict[str, float]] = []
    best_model_path = out_dir / "best_model.pt"
    last_model_path = out_dir / "last_model.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss = _run_epoch(model, ld_train, device, opt)
        val_loss = _run_epoch(model, ld_val, device, None)
        history.append({"epoch": float(epoch), "train_loss": tr_loss, "val_loss": val_loss})
        print(f"epoch {epoch:03d} train_loss={tr_loss:.6f} val_loss={val_loss:.6f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stop at epoch {epoch}", flush=True)
                break

    torch.save(model.state_dict(), last_model_path)
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    val_metrics = evaluate(model, ld_val, device, y_mean, y_std)
    test_metrics = evaluate(model, ld_test, device, y_mean, y_std)

    preprocess = {
        "feature_numeric": list(FEATURE_NUMERIC),
        "feature_categorical": list(FEATURE_CATEGORICAL),
        "targets": list(TARGETS),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "cat_stoi": cat_stoi,
    }
    (out_dir / "preprocess.json").write_text(json.dumps(preprocess, indent=2), encoding="utf-8")

    summary = {
        "manifest": manifest.as_posix(),
        "output_dir": out_dir.as_posix(),
        "device": str(device),
        "epochs_requested": int(args.epochs),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "split_counts": {
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_model": best_model_path.as_posix(),
        "last_model": last_model_path.as_posix(),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
