#!/usr/bin/env python3
"""Train the ranker that decides which candidate line is the needle.

The classical detector proposes reasonably and chooses badly: on about half the frames the
needle is among its candidates but not first. That is a ranking problem, and ranking is exactly
what a small model can learn from few examples -- each sample is a narrow strip with the
candidate laid flat in the middle, so the decision is about texture (a bright ridge with
reverberation against a band edge or empty water), not about geometry.

The split is by acquisition, not by patch: the same needle appears as several jittered variants
and the same frame yields many candidates, so any finer split would be scoring the model on
what it memorised.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class PatchDataset(Dataset):
    def __init__(self, samples: Sequence[dict], root: Path, train: bool) -> None:
        self.samples = list(samples)
        self.root = root
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import cv2

        record = self.samples[index]
        image = cv2.imread(str(self.root / record["file"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((48, 160), dtype=np.uint8)
        image = image.astype(np.float32) / 255.0

        if self.train:
            if random.random() < 0.5:
                image = image[:, ::-1].copy()          # a needle read backwards is the same needle
            if random.random() < 0.5:
                image = image[::-1, :].copy()          # and reverberation can be on either side
            if random.random() < 0.7:
                image = np.clip(image * random.uniform(0.75, 1.3)
                                + random.uniform(-0.06, 0.06), 0.0, 1.0)
        tensor = torch.from_numpy(image).unsqueeze(0)
        return tensor, torch.tensor(float(record["label"]))


class Ranker(nn.Module):
    """Small on purpose: a thousand patches cannot feed anything bigger."""

    def __init__(self) -> None:
        super().__init__()
        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def split_by_config(samples: Sequence[dict], val_fraction: float, seed: int
                    ) -> Tuple[List[dict], List[dict]]:
    by_config: Dict[str, List[dict]] = defaultdict(list)
    for sample in samples:
        by_config[sample.get("config", "?")].append(sample)
    configs = sorted(by_config)
    random.Random(seed).shuffle(configs)
    target = max(1, int(round(len(configs) * val_fraction)))
    val_configs = set(configs[:target])
    train = [s for c, group in by_config.items() if c not in val_configs for s in group]
    val = [s for c in val_configs for s in by_config[c]]
    return train, val


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = args.dataset.expanduser().resolve()
    samples = json.loads((root / "index.json").read_text(encoding="utf-8"))["samples"]
    train_rows, val_rows = split_by_config(samples, args.val_fraction, args.seed)
    print(f"addestramento {len(train_rows)} campioni, validazione {len(val_rows)} "
          f"(divisi per configurazione, non per fotogramma)")
    print(f"  positivi in val: {sum(r['label'] for r in val_rows)}/{len(val_rows)}")

    device = torch.device(args.device)
    train_loader = DataLoader(PatchDataset(train_rows, root, True), batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(PatchDataset(val_rows, root, False), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    model = Ranker().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = {"auc": -1.0, "epoch": -1}
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = F.binary_cross_entropy_with_logits(model(images).squeeze(1), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss) * images.size(0)
        scheduler.step()

        model.eval()
        scores, truth = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                probs = torch.sigmoid(model(images.to(device)).squeeze(1))
                scores.extend(probs.cpu().tolist())
                truth.extend(labels.tolist())
        scores_np, truth_np = np.array(scores), np.array(truth)
        order = np.argsort(scores_np)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(order) + 1)
        pos, neg = truth_np.sum(), (1 - truth_np).sum()
        auc = ((ranks[truth_np == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)) if pos and neg else float("nan")
        accuracy = float(((scores_np >= 0.5) == (truth_np == 1)).mean())
        if auc > best["auc"]:
            best = {"auc": float(auc), "epoch": epoch, "accuracy": accuracy}
            torch.save({"model_state_dict": model.state_dict(), "auc": float(auc),
                        "epoch": epoch, "args": vars(args) | {"dataset": str(root)}},
                       out / "best_model.pt")
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoca {epoch:3d}  loss {total/max(1,len(train_rows)):.4f}  "
                  f"val AUC {auc:.4f}  acc {accuracy:.3f}")

    (out / "metrics.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"\nmigliore: epoca {best['epoch']}, AUC {best['auc']:.4f}, accuratezza {best['accuracy']:.3f}")
    print(f"modello: {out / 'best_model.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
