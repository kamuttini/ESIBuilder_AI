#!/usr/bin/env python3
"""Run SU/GIU inference on test split and build an HTML report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

LABELS = ("su", "giu")


@dataclass(frozen=True)
class TestRow:
    sample_id: str
    image_path: Path
    true_label: str
    true_idx: int


class BinaryOrientationClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


def choose_device(requested: str | None) -> torch.device:
    if requested:
        dev = requested.strip().lower()
        if dev == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if dev.startswith("cuda") and torch.cuda.is_available():
            return torch.device(dev)
        if dev == "cpu":
            return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_test_rows(crop_manifest: Path) -> List[TestRow]:
    rows: List[TestRow] = []
    with crop_manifest.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"sample_id", "split", "label", "image_path"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Crop manifest missing columns: {sorted(missing)}")
        for row in reader:
            split = str(row.get("split", "")).strip().lower()
            label = str(row.get("label", "")).strip().lower()
            if split != "test" or label not in LABELS:
                continue
            image_path = Path(str(row.get("image_path", ""))).expanduser().resolve()
            if not image_path.exists():
                continue
            rows.append(
                TestRow(
                    sample_id=str(row.get("sample_id", "")).strip(),
                    image_path=image_path,
                    true_label=label,
                    true_idx=0 if label == "su" else 1,
                )
            )
    if not rows:
        raise RuntimeError(f"No valid test rows found in {crop_manifest}")
    return rows


def preprocess(image_path: Path, image_size: int) -> torch.Tensor:
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    image = TF.resize(
        image,
        size=[image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    x = TF.to_tensor(image)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
    x = (x - mean) / std
    return x


def batch_chunks(items: Sequence[TestRow], batch_size: int) -> List[Sequence[TestRow]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def safe_rel(from_dir: Path, to_path: Path) -> str:
    try:
        return to_path.relative_to(from_dir).as_posix()
    except Exception:  # noqa: BLE001
        return os.path.relpath(to_path.as_posix(), start=from_dir.as_posix())


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def build_html(
    rows: List[Dict[str, object]],
    summary: Dict[str, object],
    output_path: Path,
) -> None:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    stats = {
        "num_test": int(summary["num_test"]),
        "accuracy": float(summary["accuracy"]),
        "balanced_accuracy": float(summary["balanced_accuracy"]),
        "errors": int(summary["errors"]),
        "correct": int(summary["correct"]),
        "confusion": summary["confusion"],
    }
    stats_json = json.dumps(stats, ensure_ascii=False).replace("</", "<\\/")
    html_text = f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>SU/GIU Test Inference Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #0f1722; color: #e7edf5; }}
    .wrap {{ max-width: 1600px; margin: 0 auto; padding: 16px; }}
    h1 {{ margin: 0 0 6px; }}
    .muted {{ color: #9fb0c3; margin-bottom: 12px; }}
    .toolbar {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }}
    .btn {{
      border: 1px solid #2b3a4f;
      border-radius: 8px;
      background: #172333;
      color: #e7edf5;
      padding: 8px 10px;
      cursor: pointer;
    }}
    .btn.active {{ background: #2563eb; border-color: #2563eb; }}
    .search {{
      background: #121c2a;
      color: #e7edf5;
      border: 1px solid #2b3a4f;
      border-radius: 8px;
      padding: 8px 10px;
      min-width: 280px;
    }}
    .chips {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }}
    .chip {{ border: 1px solid #2b3a4f; border-radius: 999px; padding: 4px 8px; background: #121c2a; font-size: 12px; color: #c7d4e4; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(250px, 1fr)); gap: 10px; }}
    .card {{ background: #121c2a; border: 1px solid #2b3a4f; border-radius: 8px; overflow: hidden; }}
    .card.err {{ border-color: #ef4444; }}
    .card.ok {{ border-color: #22c55e; }}
    .thumb {{ display: block; background: #000; }}
    .thumb img {{ width: 100%; display: block; }}
    .meta {{ padding: 8px 10px; font-size: 12px; line-height: 1.35; }}
    .line {{ color: #c7d4e4; }}
    .bad {{ color: #fca5a5; font-weight: 700; }}
    .good {{ color: #86efac; font-weight: 700; }}
    @media (max-width: 1300px) {{ .grid {{ grid-template-columns: repeat(3, minmax(250px, 1fr)); }} }}
    @media (max-width: 980px) {{ .grid {{ grid-template-columns: repeat(2, minmax(220px, 1fr)); }} }}
    @media (max-width: 700px) {{ .grid {{ grid-template-columns: 1fr; }} .search {{ min-width: 0; width: 100%; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>SU/GIU Test Inference Report</h1>
    <div class="muted">Modello su crop del rettangolo ecografico. Risultati sullo split test.</div>
    <div class="chips" id="stats"></div>
    <div class="toolbar">
      <button class="btn active" data-filter="all">Tutti</button>
      <button class="btn" data-filter="errors">Solo errori</button>
      <button class="btn" data-filter="ok">Solo corretti</button>
      <input id="search" class="search" type="text" placeholder="Cerca sample_id o label...">
    </div>
    <div class="grid" id="grid"></div>
  </div>

  <script>
    const ROWS = {payload};
    const STATS = {stats_json};
    let currentFilter = 'all';

    function esc(v) {{
      return String(v)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    function renderStats() {{
      const c = STATS.confusion || [[0,0],[0,0]];
      const out = [
        `test: ${{STATS.num_test}}`,
        `acc: ${{(100*STATS.accuracy).toFixed(2)}}%`,
        `bal acc: ${{(100*STATS.balanced_accuracy).toFixed(2)}}%`,
        `corretti: ${{STATS.correct}}`,
        `errori: ${{STATS.errors}}`,
        `conf [SU->SU,SU->GIU; GIU->SU,GIU->GIU]: [${{c[0][0]}},${{c[0][1]}}; ${{c[1][0]}},${{c[1][1]}}]`
      ];
      document.getElementById('stats').innerHTML = out.map((x) => `<span class="chip">${{esc(x)}}</span>`).join('');
    }}

    function attachFallback(root) {{
      root.querySelectorAll('img[data-fallback-src]').forEach((img) => {{
        img.addEventListener('error', () => {{
          if (img.dataset.fallbackUsed === '1') return;
          const fallback = img.getAttribute('data-fallback-src') || '';
          if (!fallback) return;
          img.dataset.fallbackUsed = '1';
          img.setAttribute('src', fallback);
        }});
      }});
    }}

    function render() {{
      const q = String(document.getElementById('search').value || '').trim().toLowerCase();
      const rows = ROWS.filter((r) => {{
        if (currentFilter === 'errors' && r.correct) return false;
        if (currentFilter === 'ok' && !r.correct) return false;
        if (!q) return true;
        const blob = `${{r.sample_id}} ${{r.true_label}} ${{r.pred_label}}`.toLowerCase();
        return blob.includes(q);
      }});

      rows.sort((a,b) => {{
        if (a.correct !== b.correct) return a.correct ? 1 : -1;
        return Number(a.confidence) - Number(b.confidence);
      }});

      const grid = document.getElementById('grid');
      grid.innerHTML = rows.map((r) => `
        <article class="card ${{r.correct ? 'ok' : 'err'}}">
          <a class="thumb" href="${{esc(r.image_uri)}}" target="_blank" rel="noopener noreferrer">
            <img loading="lazy" src="${{esc(r.image_src_rel)}}" data-fallback-src="${{esc(r.image_uri)}}" alt="${{esc(r.sample_id)}}">
          </a>
          <div class="meta">
            <div><b>${{esc(r.sample_id)}}</b></div>
            <div class="line">true: <b>${{esc(r.true_label)}}</b> | pred: <b>${{esc(r.pred_label)}}</b></div>
            <div class="line">p(su): ${{Number(r.prob_su).toFixed(4)}} | p(giu): ${{Number(r.prob_giu).toFixed(4)}} | conf: ${{Number(r.confidence).toFixed(4)}}</div>
            <div class="${{r.correct ? 'good' : 'bad'}}">${{r.correct ? 'corretto' : 'errore'}}</div>
          </div>
        </article>
      `).join('');
      attachFallback(grid);
    }}

    document.querySelectorAll('.btn[data-filter]').forEach((btn) => {{
      btn.addEventListener('click', () => {{
        currentFilter = btn.getAttribute('data-filter') || 'all';
        document.querySelectorAll('.btn[data-filter]').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        render();
      }});
    }});
    document.getElementById('search').addEventListener('input', render);

    renderStats();
    render();
  </script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    crop_manifest = args.crop_manifest.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not crop_manifest.exists():
        raise FileNotFoundError(f"Crop manifest not found: {crop_manifest}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    rows = load_test_rows(crop_manifest)
    device = choose_device(args.device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    image_size = int(args.image_size) if int(args.image_size) > 0 else int(ckpt.get("args", {}).get("image_size", 256))
    batch_size = max(1, int(args.batch_size))

    model = BinaryOrientationClassifier().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_rows: List[Dict[str, object]] = []
    confusion = [[0, 0], [0, 0]]
    correct = 0

    with torch.no_grad():
        for chunk in batch_chunks(rows, batch_size):
            x_list = [preprocess(r.image_path, image_size=image_size) for r in chunk]
            x = torch.stack(x_list, dim=0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).detach().cpu()
            pred_idx = torch.argmax(probs, dim=1).numpy().tolist()

            for i, row in enumerate(chunk):
                pi = int(pred_idx[i])
                ti = int(row.true_idx)
                p_su = float(probs[i, 0].item())
                p_giu = float(probs[i, 1].item())
                conf = p_su if pi == 0 else p_giu
                is_ok = pi == ti
                confusion[ti][pi] += 1
                if is_ok:
                    correct += 1
                out_rows.append(
                    {
                        "sample_id": row.sample_id,
                        "image_path": row.image_path.as_posix(),
                        "image_uri": row.image_path.as_uri(),
                        "image_src_rel": safe_rel(output_dir, row.image_path),
                        "true_label": row.true_label,
                        "true_idx": ti,
                        "pred_label": LABELS[pi],
                        "pred_idx": pi,
                        "prob_su": p_su,
                        "prob_giu": p_giu,
                        "confidence": conf,
                        "correct": is_ok,
                    }
                )

    n = len(out_rows)
    acc = float(correct / max(1, n))
    recall_su = float(confusion[0][0] / max(1, confusion[0][0] + confusion[0][1]))
    recall_giu = float(confusion[1][1] / max(1, confusion[1][0] + confusion[1][1]))
    bal_acc = float((recall_su + recall_giu) / 2.0)
    errors = n - correct

    predictions_csv = output_dir / "test_predictions.csv"
    with predictions_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "sample_id",
                "image_path",
                "true_label",
                "true_idx",
                "pred_label",
                "pred_idx",
                "prob_su",
                "prob_giu",
                "confidence",
                "correct",
            ]
        )
        for r in out_rows:
            writer.writerow(
                [
                    r["sample_id"],
                    r["image_path"],
                    r["true_label"],
                    r["true_idx"],
                    r["pred_label"],
                    r["pred_idx"],
                    f"{float(r['prob_su']):.8f}",
                    f"{float(r['prob_giu']):.8f}",
                    f"{float(r['confidence']):.8f}",
                    1 if bool(r["correct"]) else 0,
                ]
            )

    summary = {
        "checkpoint": checkpoint.as_posix(),
        "crop_manifest": crop_manifest.as_posix(),
        "device": str(device),
        "image_size": image_size,
        "batch_size": batch_size,
        "num_test": n,
        "correct": correct,
        "errors": errors,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "recall_su": recall_su,
        "recall_giu": recall_giu,
        "confusion": confusion,
        "predictions_csv": predictions_csv.as_posix(),
        "html_report": (output_dir / "index.html").as_posix(),
    }
    (output_dir / "test_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    build_html(rows=out_rows, summary=summary, output_path=output_dir / "index.html")

    print(f"Test rows: {n}", flush=True)
    print(f"Accuracy: {acc:.4f}", flush=True)
    print(f"Balanced accuracy: {bal_acc:.4f}", flush=True)
    print(f"Predictions CSV: {predictions_csv}", flush=True)
    print(f"HTML report: {output_dir / 'index.html'}", flush=True)
    print(f"Summary JSON: {output_dir / 'test_summary.json'}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer SU/GIU on test crop split and build HTML report.")
    parser.add_argument(
        "--crop-manifest",
        type=Path,
        default=Path("artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/cropped_dataset/crop_manifest.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/model_su_giu_rect/best_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/test_inference_html"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=0, help="Override image size. 0 = use checkpoint value.")
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
