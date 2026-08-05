#!/usr/bin/env python3
"""Train the per-vendor scale localiser: 1-D heatmaps for the ruler column and the zero.

The network replaces the *detection* stage of ``detect_scale_ladder.py``, which is where
the block actually fails: after the setup consensus was added, essentially all remaining
loss came from frames where no tick ladder was found at all (35/95 rows on BK, 50/91 on
Esaote — whole setups with nothing to interpolate from). Thresholding cannot handle a
ruler drawn in dark grey over bright anatomy; a learned response can.

It does **not** replace the OCR: the numbers printed beside the ruler remain the only
direct evidence of the absolute scale, and the ``log_span_mm`` head here is a fallback and
a cross-check, not the primary source.

Heads
-----
* ``x`` heatmap over image columns  -> which column is the ruler
* ``y`` heatmap over image rows     -> where the zero tick is
* ``log_span_mm`` scalar            -> log(mm_per_px * image_height), resize invariant
* ``direction`` 2 classes           -> zero at top / zero at bottom

Run on the Mac training environment (MPS); this file only needs torch at train time, so
``--check-data`` works anywhere and validates the risky part (paths, resize, target
coordinates) without it.

Usage
-----
    # validate the data pipeline without torch
    python3 tools/scale/train_scale_heatmap.py --check-data \
        --manifest artifacts/39_scale_heatmap_dataset_20260729/manifests/manifest_scale_heatmap_bk.csv

    # train one vendor
    OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/train_scale_heatmap.py \
        --manifest artifacts/39_scale_heatmap_dataset_20260729/manifests/manifest_scale_heatmap_bk.csv \
        --output-dir artifacts/40_scale_heatmap_models_20260729/bk \
        --epochs 40 --device auto

    # train every vendor that has enough data
    OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/train_scale_heatmap.py \
        --dataset-dir artifacts/39_scale_heatmap_dataset_20260729 \
        --output-root artifacts/40_scale_heatmap_models_20260729 --all-eligible
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from heatmap_codec import (  # noqa: E402
    decode_prediction,
    encode_gaussian,
    span_mm_to_mm_per_px,
)

# Bin counts. x gets more than y because the x tolerance is tighter (6 px vs 8 px) and
# some vendors are 1920 wide: measured on the real GT, 256 x-bins leave a worst-case
# quantisation error of 4.0 px, which eats most of the budget before the model even
# makes a mistake. 512 bins bring that to ~2 px.
X_BINS = 512
Y_BINS = 256
SIGMA_BINS = 2.0

# Same tolerances as tools/scale/eval_scale_detector.py, so numbers are comparable.
TOL_X_PX = 6.0
TOL_Y_PX = 8.0
TOL_MM_REL = 0.02

# The tolerance that actually matters for x. The network's job is to say *which column*
# the ruler is in; the classical tick detector then refines inside a band of this
# half-width around it (``prior_band_px`` in detect_scale_ladder.py) and is accurate to
# ~2 px once pointed at the right ladder. Demanding 6 px from the network is asking it to
# do the classical stage's job — and it cannot, because the resize destroys the ticks:
# at 1920 wide downscaled to 512, one input pixel is 3.75 real px and a tick shrinks from
# 8x2 px to 2.1x0.9 px.
TOL_X_BAND_PX = 70.0


# --------------------------------------------------------------------------- #
# data (numpy only, so --check-data needs no torch)
# --------------------------------------------------------------------------- #


@dataclass
class Sample:
    image_path: str
    image_w: int
    image_h: int
    x_norm: float
    y_zero_norm: float
    log_span_mm: float
    direction: int
    mm_per_px: float
    sample_id: str
    vendor: str
    config_folder: str


def load_manifest(path: Path, split: Optional[str], remap: Sequence[str]) -> List[Sample]:
    out: List[Sample] = []
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if split and r.get("split") != split:
            continue
        img = r.get("image_path", "")
        for rule in remap:
            if "=" in rule:
                old, new = rule.split("=", 1)
                if img.startswith(old):
                    img = new + img[len(old) :]
                    break
        try:
            out.append(
                Sample(
                    image_path=img,
                    image_w=int(r["image_w"]),
                    image_h=int(r["image_h"]),
                    x_norm=float(r["x_norm"]),
                    y_zero_norm=float(r["y_zero_norm"]),
                    log_span_mm=float(r["log_span_mm"]),
                    direction=int(r["direction"]),
                    mm_per_px=float(r["mm_per_px"]),
                    sample_id=r.get("sample_id", ""),
                    vendor=r.get("vendor", ""),
                    config_folder=r.get("config_folder", ""),
                )
            )
        except (KeyError, ValueError):
            continue
    return out


class ScaleHeatmapData:
    """Loads a frame, resizes it, and builds the four targets.

    Augmentation is deliberately limited to what preserves the semantics:

    * brightness/contrast jitter — the whole point is robustness to a dim ruler;
    * small translations, with the targets shifted to match;
    * **no flips**. A horizontal flip would move the ruler to the other side and a
      vertical one would invert the direction, so both would teach the model the opposite
      of what the label says.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        in_w: int = 512,
        in_h: int = 512,
        augment: bool = False,
        shift_frac: float = 0.04,
        seed: int = 0,
    ) -> None:
        self.samples = list(samples)
        self.in_w, self.in_h = in_w, in_h
        self.augment = augment
        self.shift_frac = shift_frac
        self.rng = random.Random(seed)
        # Indices whose frame could not be read; evaluation skips these.
        self.unreadable: set = set()

    def __len__(self) -> int:
        return len(self.samples)

    def load_raw(self, index: int) -> Optional[np.ndarray]:
        img = cv2.imread(self.samples[index].image_path, cv2.IMREAD_GRAYSCALE)
        return img

    def _read_with_retry(self, index: int, attempts: int = 3) -> Optional[np.ndarray]:
        """Read one frame, retrying briefly before giving up.

        Reads come from an external SSD over a whole training run (tens of thousands of
        them), so a transient failure is a question of when, not if. Previously any such
        failure raised out of a DataLoader worker and killed the run — a 1.5 hour job lost
        to one hiccup, with no resume. Now it retries, then degrades gracefully.
        """
        path = self.samples[index].image_path
        for attempt in range(attempts):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
            if attempt + 1 < attempts:
                time.sleep(0.25 * (attempt + 1))
        return None

    def __getitem__(self, index: int):
        s = self.samples[index]
        img = self._read_with_retry(index)
        valid = img is not None
        if not valid:
            # Substitute a neighbour so training carries on, and mark the item invalid so
            # evaluation ignores it instead of scoring a frame it never actually saw.
            self.unreadable.add(index)
            if len(self.unreadable) <= 5:
                print(f"[data] unreadable after retries, substituting: {s.image_path}", flush=True)
            alt = (index + 1) % max(1, len(self.samples))
            img = self._read_with_retry(alt, attempts=1)
            if img is None:
                img = np.zeros((self.in_h, self.in_w), dtype=np.uint8)
        img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_AREA)
        x_norm, y_norm = s.x_norm, s.y_zero_norm

        if self.augment:
            if self.shift_frac > 0:
                dx = self.rng.uniform(-self.shift_frac, self.shift_frac)
                dy = self.rng.uniform(-self.shift_frac, self.shift_frac)
                M = np.float32([[1, 0, dx * self.in_w], [0, 1, dy * self.in_h]])
                img = cv2.warpAffine(
                    img, M, (self.in_w, self.in_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                x_norm = float(np.clip(x_norm + dx, 0.0, 0.999))
                y_norm = float(np.clip(y_norm + dy, 0.0, 0.999))
            gain = self.rng.uniform(0.75, 1.3)
            bias = self.rng.uniform(-14, 14)
            img = np.clip(img.astype(np.float32) * gain + bias, 0, 255)
        img = img.astype(np.float32) / 255.0

        return {
            "image": img[None, :, :],  # (1, H, W)
            "x_heat": encode_gaussian(x_norm, X_BINS, SIGMA_BINS).astype(np.float32),
            "y_heat": encode_gaussian(y_norm, Y_BINS, SIGMA_BINS).astype(np.float32),
            "log_span_mm": np.float32(s.log_span_mm),
            "direction_class": np.int64(0 if s.direction > 0 else 1),
            "index": index,
            "valid": bool(valid),
        }


# --------------------------------------------------------------------------- #
# model (torch)
# --------------------------------------------------------------------------- #


def build_model(width: int = 48):
    import torch
    import torch.nn as nn

    class ConvBlock(nn.Sequential):
        def __init__(self, cin: int, cout: int, stride: int = 2) -> None:
            super().__init__(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

    class AxisHead(nn.Module):
        """Collapse one image axis, then reason 1-D along the other.

        Mean and max pooling are concatenated on purpose: the mean carries "is there
        generally something bright in this column", the max carries "is there a single
        very bright thin thing" — which is what a tick is.
        """

        def __init__(self, cin: int, n_bins: int, axis: int) -> None:
            super().__init__()
            self.axis = axis  # 2 = collapse H (x head), 3 = collapse W (y head)
            self.n_bins = n_bins
            self.net = nn.Sequential(
                nn.Conv1d(2 * cin, 128, 5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 128, 5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 1, 3, padding=1),
            )

        def forward(self, feat):
            pooled = torch.cat([feat.mean(dim=self.axis), feat.amax(dim=self.axis)], dim=1)
            logits = self.net(pooled)  # (B, 1, L)
            logits = torch.nn.functional.interpolate(
                logits, size=self.n_bins, mode="linear", align_corners=False
            )
            return logits.squeeze(1)  # (B, n_bins)

    class ScaleHeatmapNet(nn.Module):
        def __init__(self, w: int = width) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                ConvBlock(1, w, stride=2),
                ConvBlock(w, w * 2, stride=2),
                ConvBlock(w * 2, w * 3, stride=2),
            )  # stride 8
            c = w * 3
            self.x_head = AxisHead(c, X_BINS, axis=2)
            self.y_head = AxisHead(c, Y_BINS, axis=3)
            # The vertical profile is what makes the direction learnable at all. A plain
            # AdaptiveAvgPool2d(1) averages over the whole frame and therefore destroys
            # exactly the up/down asymmetry that "is the zero at the top or the bottom"
            # consists of: with it, this head could only ever output a constant, and it
            # did — measured direction accuracy equalled the majority-class share of each
            # split to the second decimal (Esaote val 91.73%, BK val 23.81%).
            self.v_profile_bins = 16
            self.global_head = nn.Sequential(
                nn.Linear(c + c * self.v_profile_bins, 192),
                nn.ReLU(inplace=True),
                nn.Linear(192, 3),  # [log_span_mm, dir_logit_top, dir_logit_bottom]
            )

        def forward(self, image):
            feat = self.stem(image)
            # global context (scale-ish cues) ...
            g_global = feat.mean(dim=(2, 3))  # (B, C)
            # ... plus an explicit, order-preserving vertical profile
            v = feat.mean(dim=3)  # (B, C, H): collapse width, keep height
            v = torch.nn.functional.adaptive_avg_pool1d(v, self.v_profile_bins)
            g = self.global_head(torch.cat([g_global, v.flatten(1)], dim=1))
            return {
                "x_logits": self.x_head(feat),
                "y_logits": self.y_head(feat),
                "log_span_mm": g[:, 0],
                "direction_logits": g[:, 1:],
            }

    return ScaleHeatmapNet()


def pick_device(requested: str):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #


def _collate(batch):
    import torch

    return {
        "image": torch.from_numpy(np.stack([b["image"] for b in batch])),
        "x_heat": torch.from_numpy(np.stack([b["x_heat"] for b in batch])),
        "y_heat": torch.from_numpy(np.stack([b["y_heat"] for b in batch])),
        "log_span_mm": torch.from_numpy(np.stack([b["log_span_mm"] for b in batch])),
        "direction_class": torch.from_numpy(np.stack([b["direction_class"] for b in batch])),
        "index": [b["index"] for b in batch],
        "valid": [b["valid"] for b in batch],
    }


def soft_cross_entropy(logits, target):
    import torch

    logp = torch.log_softmax(logits, dim=1)
    return -(target * logp).sum(dim=1).mean()


def evaluate(model, data: ScaleHeatmapData, device, batch_size: int) -> Dict[str, object]:
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    err_x: List[float] = []
    err_y: List[float] = []
    rel_mm: List[float] = []
    dir_ok = 0
    strict = 0
    n = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device))
            xh = torch.softmax(out["x_logits"], dim=1).cpu().numpy()
            yh = torch.softmax(out["y_logits"], dim=1).cpu().numpy()
            ls = out["log_span_mm"].cpu().numpy()
            dl = out["direction_logits"].cpu().numpy()
            for k, idx in enumerate(batch["index"]):
                if not batch["valid"][k]:
                    continue  # frame unreadable: do not score a prediction on a blank
                s = data.samples[idx]
                d = decode_prediction(xh[k], yh[k], float(ls[k]), dl[k], s.image_w, s.image_h)
                ex = abs(d["x"] - s.x_norm * s.image_w)
                ey = abs(d["y_zero"] - s.y_zero_norm * s.image_h)
                em = abs(d["mm_per_px"] - s.mm_per_px) / s.mm_per_px
                err_x.append(ex)
                err_y.append(ey)
                rel_mm.append(em)
                ok_dir = d["direction"] == s.direction
                dir_ok += int(ok_dir)
                strict += int(ex <= TOL_X_PX and ey <= TOL_Y_PX and em <= TOL_MM_REL and ok_dir)
                n += 1
    if n == 0:
        return {"rows": 0}

    def agg(v: Sequence[float]) -> Dict[str, float]:
        s = sorted(v)
        return {
            "median": round(statistics.median(s), 4),
            "mean": round(statistics.fmean(s), 4),
            "p90": round(s[min(len(s) - 1, int(0.9 * (len(s) - 1)))], 4),
            "max": round(s[-1], 4),
        }

    x_band = 100 * sum(1 for v in err_x if v <= TOL_X_BAND_PX) / n
    y_tol = 100 * sum(1 for v in err_y if v <= TOL_Y_PX) / n
    dir_pct = 100 * dir_ok / n
    return {
        "rows": n,
        "rows_unreadable": len(data.unreadable),
        "err_x_px": agg(err_x),
        "err_y_zero_px": agg(err_y),
        "rel_err_mm_per_px": agg(rel_mm),
        "direction_acc": round(dir_ok / n, 4),
        "x_within_band_pct": round(x_band, 2),
        "x_within_tol_pct": round(100 * sum(1 for v in err_x if v <= TOL_X_PX) / n, 2),
        "y_within_tol_pct": round(y_tol, 2),
        "calib_within_tol_pct": round(100 * sum(1 for v in rel_mm if v <= TOL_MM_REL) / n, 2),
        "strict_ok_pct": round(100 * strict / n, 2),
        # What the checkpoint is selected on: the three things the network is responsible
        # for. Selecting on strict_ok would select on the calibration, which is the OCR's
        # job and sits at 2-5% — effectively picking an epoch at random.
        "handoff_score": round((x_band + y_tol + dir_pct) / 3.0, 3),
    }


def train_one(
    manifest: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    import torch
    from torch.utils.data import DataLoader

    device = pick_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        s: load_manifest(manifest, s, args.path_remap) for s in ("train", "val", "test")
    }
    for name, rows in splits.items():
        print(f"[data] {name}: {len(rows)} rows")
    if len(splits["train"]) < 20 or len(splits["val"]) < 5:
        return {"status": "skipped_not_enough_rows", "rows_by_split": {k: len(v) for k, v in splits.items()}}

    train_data = ScaleHeatmapData(
        splits["train"], args.image_width, args.image_height, augment=True, seed=args.seed
    )
    val_data = ScaleHeatmapData(splits["val"], args.image_width, args.image_height)
    test_data = ScaleHeatmapData(splits["test"], args.image_width, args.image_height)

    torch.manual_seed(args.seed)
    model = build_model(args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    huber = torch.nn.SmoothL1Loss(beta=0.1)

    # Balance the direction classes. Esaote is 95% "zero at top", so an unweighted loss is
    # minimised by ignoring the minority class entirely — which, combined with the pooling
    # bug fixed above, is how this head ended up constant. BK, at 58/42, barely needs it;
    # the weighting is computed per vendor so both cases are handled by the same code.
    n_top = sum(1 for s in splits["train"] if s.direction > 0)
    n_bottom = len(splits["train"]) - n_top
    if args.balance_direction and n_top > 0 and n_bottom > 0:
        w = torch.tensor(
            [len(splits["train"]) / (2.0 * n_top), len(splits["train"]) / (2.0 * n_bottom)],
            dtype=torch.float32,
            device=device,
        )
        print(f"[data] direction train split: top={n_top} bottom={n_bottom}, class weights={w.tolist()}")
    else:
        w = None
        print(f"[data] direction train split: top={n_top} bottom={n_bottom}, unweighted")
    ce = torch.nn.CrossEntropyLoss(weight=w)

    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=args.num_workers,
    )

    log_path = output_dir / "train_log.jsonl"
    best_score = -1.0
    best_metrics: Dict[str, object] = {}
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses: List[float] = []
            for batch in loader:
                image = batch["image"].to(device)
                out = model(image)
                loss = (
                    soft_cross_entropy(out["x_logits"], batch["x_heat"].to(device))
                    + soft_cross_entropy(out["y_logits"], batch["y_heat"].to(device))
                    + args.w_span * huber(out["log_span_mm"], batch["log_span_mm"].to(device))
                    + args.w_dir * ce(out["direction_logits"], batch["direction_class"].to(device))
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
            sched.step()

            val = evaluate(model, val_data, device, args.batch_size)
            score = float(val.get("handoff_score", 0.0))
            record = {
                "epoch": epoch,
                "train_loss": round(statistics.fmean(losses), 5) if losses else None,
                "val": val,
                "elapsed_sec": round(time.time() - t0, 1),
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(
                f"[epoch {epoch:3d}] loss={record['train_loss']} "
                f"handoff={score:.1f} x_band={val.get('x_within_band_pct')}% "
                f"x_tol={val.get('x_within_tol_pct')}% y_tol={val.get('y_within_tol_pct')}% "
                f"dir={val.get('direction_acc')} calib={val.get('calib_within_tol_pct')}%",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_metrics = {"epoch": epoch, "val": val}
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "width": args.width,
                        "x_bins": X_BINS,
                        "y_bins": Y_BINS,
                        "image_width": args.image_width,
                        "image_height": args.image_height,
                        "epoch": epoch,
                        "val": val,
                    },
                    output_dir / "best_model.pt",
                )

    # Final test with the best checkpoint.
    ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test = evaluate(model, test_data, device, args.batch_size)

    summary = {
        "status": "ok",
        "manifest": manifest.as_posix(),
        "device": str(device),
        "epochs": args.epochs,
        "image_size": [args.image_width, args.image_height],
        "x_bins": X_BINS,
        "y_bins": Y_BINS,
        "rows_by_split": {k: len(v) for k, v in splits.items()},
        "folders_by_split": {
            k: len({s.config_folder for s in v}) for k, v in splits.items()
        },
        "best": best_metrics,
        "test": test,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(test, indent=2, ensure_ascii=False))
    return summary


# --------------------------------------------------------------------------- #
# data check (no torch)
# --------------------------------------------------------------------------- #


def eval_only(
    checkpoint: Path,
    manifest: Path,
    split: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Score an existing checkpoint against an arbitrary manifest split.

    Needed for any honest comparison between two training runs: the two models must be
    measured on the *same* rows. Without this, the "with fusion" arm would be scored on a
    test set that includes fusion rows and the "without fusion" arm on one that does not,
    and the difference would say nothing about which training set is better.
    """
    import torch

    device = pick_device(args.device)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model(int(ckpt.get("width", args.width)))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    rows = load_manifest(manifest, split, args.path_remap)
    data = ScaleHeatmapData(
        rows,
        int(ckpt.get("image_width", args.image_width)),
        int(ckpt.get("image_height", args.image_height)),
    )
    metrics = evaluate(model, data, device, args.batch_size)
    out = {
        "checkpoint": checkpoint.as_posix(),
        "trained_epoch": ckpt.get("epoch"),
        "manifest": manifest.as_posix(),
        "split": split,
        "rows": len(rows),
        "folders": len({r.config_folder for r in rows}),
        "metrics": metrics,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def check_data(manifest: Path, args: argparse.Namespace) -> int:
    """Validate paths, resize and target construction without touching torch."""
    problems: Dict[str, int] = {}
    checked = 0
    per_split: Dict[str, int] = {}
    for split in ("train", "val", "test"):
        rows = load_manifest(manifest, split, args.path_remap)
        per_split[split] = len(rows)
        data = ScaleHeatmapData(rows, args.image_width, args.image_height)
        step = max(1, len(rows) // max(1, args.check_samples))
        for i in range(0, len(rows), step):
            s = rows[i]
            try:
                item = data[i]
            except FileNotFoundError:
                problems["image_unreadable"] = problems.get("image_unreadable", 0) + 1
                continue
            img = item["image"]
            if img.shape != (1, args.image_height, args.image_width):
                problems["wrong_shape"] = problems.get("wrong_shape", 0) + 1
            # the declared frame size must match the real file, or every target is wrong
            raw = data.load_raw(i)
            if raw is not None and (raw.shape[1] != s.image_w or raw.shape[0] != s.image_h):
                problems["frame_size_mismatch"] = problems.get("frame_size_mismatch", 0) + 1
            # round-trip the targets through the decoder
            d = decode_prediction(
                item["x_heat"],
                item["y_heat"],
                float(item["log_span_mm"]),
                np.array([1.0, 0.0]) if s.direction > 0 else np.array([0.0, 1.0]),
                s.image_w,
                s.image_h,
            )
            if abs(d["x"] - s.x_norm * s.image_w) > TOL_X_PX:
                problems["x_quantisation_over_tol"] = problems.get("x_quantisation_over_tol", 0) + 1
            if abs(d["y_zero"] - s.y_zero_norm * s.image_h) > TOL_Y_PX:
                problems["y_quantisation_over_tol"] = problems.get("y_quantisation_over_tol", 0) + 1
            if abs(d["mm_per_px"] - s.mm_per_px) / s.mm_per_px > 1e-4:
                problems["mm_per_px_roundtrip"] = problems.get("mm_per_px_roundtrip", 0) + 1
            if d["direction"] != s.direction:
                problems["direction_roundtrip"] = problems.get("direction_roundtrip", 0) + 1
            checked += 1
    print(
        json.dumps(
            {
                "manifest": manifest.as_posix(),
                "rows_by_split": per_split,
                "samples_checked": checked,
                "problems": problems,
            },
            indent=2,
        )
    )
    if problems.get("image_unreadable") and problems["image_unreadable"] >= max(1, checked):
        # Every single image failed: almost always a mount-point mismatch rather than
        # corrupt files, so say so instead of leaving the user to guess.
        sample = next(
            (
                s.image_path
                for split in ("train", "val", "test")
                for s in load_manifest(manifest, split, args.path_remap)
            ),
            "",
        )
        print(
            "\n[hint] nessuna immagine leggibile: il manifest e' stato generato con un\n"
            f"       punto di mount diverso. Primo percorso nel manifest:\n         {sample}\n"
            "       Rigenera il dataset con --canonical-root sull'audit, oppure passa\n"
            "       --path-remap VECCHIO_PREFISSO=/Volumes/ a questo comando."
        )
    return 0 if not problems else 1


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the per-vendor scale heatmap localiser.")
    p.add_argument("--manifest", type=Path, help="Single-vendor manifest CSV.")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--dataset-dir", type=Path, help="Dataset root for --all-eligible.")
    p.add_argument("--output-root", type=Path)
    p.add_argument(
        "--all-eligible",
        action="store_true",
        help="Train every vendor flagged eligible in the dataset summary.json.",
    )
    p.add_argument("--image-width", type=int, default=512)
    p.add_argument("--image-height", type=int, default=512)
    p.add_argument("--width", type=int, default=48, help="Backbone base channel count.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--w-span", type=float, default=2.0, help="Weight of the log_span_mm loss.")
    p.add_argument("--w-dir", type=float, default=1.0, help="Weight of the direction loss.")
    p.add_argument(
        "--balance-direction",
        action="store_true",
        default=True,
        help="Weight the direction classes inversely to their frequency (default on).",
    )
    p.add_argument(
        "--no-balance-direction",
        dest="balance_direction",
        action="store_false",
        help="Disable the direction class weighting.",
    )
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--path-remap", action="append", default=[])
    p.add_argument("--check-data", action="store_true", help="Validate the data pipeline, no torch.")
    p.add_argument("--check-samples", type=int, default=40)
    p.add_argument(
        "--eval-only",
        type=Path,
        default=None,
        help="Score this checkpoint against --manifest instead of training. Use it to "
        "compare two runs on identical rows.",
    )
    p.add_argument("--eval-split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--eval-out",
        type=Path,
        default=None,
        help="Write the --eval-only result to this JSON file.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.check_data:
        manifests: List[Path] = []
        if args.manifest:
            manifests = [args.manifest]
        elif args.dataset_dir:
            # Check every vendor that will actually be trained, plus any other manifest
            # present, so one command covers the whole dataset.
            summary_path = args.dataset_dir / "summary.json"
            eligible = []
            if summary_path.is_file():
                eligible = json.loads(summary_path.read_text(encoding="utf-8")).get(
                    "vendors_eligible_own_model", []
                )
            slugs = [v.lower().replace(" ", "_") for v in eligible]
            man_dir = args.dataset_dir / "manifests"
            manifests = [
                man_dir / f"manifest_scale_heatmap_{slug}.csv"
                for slug in slugs
                if (man_dir / f"manifest_scale_heatmap_{slug}.csv").is_file()
            ]
            if not manifests:
                manifests = sorted(man_dir.glob("manifest_scale_heatmap_*.csv"))
        if not manifests:
            print("[error] --check-data needs --manifest <file.csv> or --dataset-dir <dir>")
            return 2

        worst = 0
        for m in manifests:
            print(f"\n===== {m.name} =====")
            worst = max(worst, check_data(m, args))
        if len(manifests) > 1:
            print(
                f"\n[check-data] {len(manifests)} manifest controllati: "
                + ("NESSUN PROBLEMA" if worst == 0 else "PROBLEMI RILEVATI, vedi sopra")
            )
        return worst

    if args.eval_only is not None:
        if not args.manifest:
            print("[error] --eval-only needs --manifest")
            return 2
        result = eval_only(args.eval_only, args.manifest, args.eval_split, args)
        if args.eval_out:
            args.eval_out.parent.mkdir(parents=True, exist_ok=True)
            args.eval_out.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return 0

    if args.all_eligible:
        if not args.dataset_dir or not args.output_root:
            print("[error] --all-eligible needs --dataset-dir and --output-root")
            return 2
        summary = json.loads((args.dataset_dir / "summary.json").read_text(encoding="utf-8"))
        results: Dict[str, object] = {}
        for vendor in summary.get("vendors_eligible_own_model", []):
            slug = vendor.lower().replace(" ", "_")
            manifest = args.dataset_dir / "manifests" / f"manifest_scale_heatmap_{slug}.csv"
            if not manifest.is_file():
                results[vendor] = {"status": "manifest_missing"}
                continue
            print(f"\n===== {vendor} =====", flush=True)
            results[vendor] = train_one(manifest, args.output_root / slug, args)
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "run_summary.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for vendor, res in results.items():
            test = res.get("test", {}) if isinstance(res, dict) else {}
            print(
                f"  {vendor:12s} {res.get('status')}  "
                f"test strict={test.get('strict_ok_pct')}% "
                f"x_tol={test.get('x_within_tol_pct')}% dir={test.get('direction_acc')}"
            )
        return 0

    if not args.manifest or not args.output_dir:
        print("[error] need --manifest and --output-dir (or --all-eligible)")
        return 2
    train_one(args.manifest, args.output_dir, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
