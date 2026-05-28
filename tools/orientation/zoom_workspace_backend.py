#!/usr/bin/env python3
"""Backend locale per Zoom Workspace: calcolo rect rete (singolo + globale)."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional

from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import HTTPException
from werkzeug.routing import Map, Rule
from werkzeug.serving import run_simple
from werkzeug.wrappers import Request, Response


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HTML = ROOT / "artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/zoom_workspace_interactive/index.html"
INFER_SCRIPT = ROOT / "tools/review_html/infer_ultrasound_rect_folder_html.py"
DEFAULT_PYTHON = Path("/opt/anaconda3/bin/python")
DEFAULT_RECT_CHECKPOINT = ROOT / "artifacts/10_active_pipeline/pipeline_fss_head/models/rect_training_e40_run2/best_model.pt"
DEFAULT_VENDOR_CHECKPOINT = (
    ROOT / "artifacts/10_active_pipeline/pipeline_fss_head/models/vendor_training_no_negative_v2_power/best_model.pt"
)
DEFAULT_RECT_VENDOR_MAP = ROOT / "artifacts/10_active_pipeline/pipeline_fss_head/maps/vendor_rect_map_bk_only.json"
DEFAULT_ORIENTATION_CHECKPOINT = (
    ROOT / "artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/model_su_giu_rect/best_model.pt"
)


def _json_response(payload: Dict[str, object], status: int = 200) -> Response:
    resp = Response(
        json.dumps(payload, ensure_ascii=False),
        status=status,
        content_type="application/json; charset=utf-8",
    )
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


def _text_response(text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> Response:
    resp = Response(text, status=status, content_type=content_type)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


def _normalize_path(p: str) -> str:
    return str(p or "").strip().replace("\\", "/").lstrip("./").lower()


def _safe_rel_path(rel: str) -> Optional[Path]:
    rel_norm = str(rel or "").strip().replace("\\", "/")
    if not rel_norm:
        return None
    parts = [part for part in PurePosixPath(rel_norm).parts if part not in {"", ".", ".."}]
    if not parts:
        return None
    return Path(*parts)


def _to_float(value: object) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    if x != x:  # NaN
        return None
    return x


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _rect_norm_from_edges(
    left: object,
    top: object,
    right: object,
    bottom: object,
    width: object,
    height: object,
) -> Optional[Dict[str, float]]:
    l = _to_float(left)
    t = _to_float(top)
    r = _to_float(right)
    b = _to_float(bottom)
    if l is None or t is None or r is None or b is None:
        return None
    x1 = min(l, r)
    y1 = min(t, b)
    x2 = max(l, r)
    y2 = max(t, b)

    w = _to_float(width)
    h = _to_float(height)
    if w is not None and h is not None and w > 1.0 and h > 1.0:
        x = x1 / w
        y = y1 / h
        rw = (x2 - x1) / w
        rh = (y2 - y1) / h
    else:
        # maybe already normalized
        if not (0.0 <= x1 <= 1.2 and 0.0 <= y1 <= 1.2 and 0.0 <= x2 <= 1.2 and 0.0 <= y2 <= 1.2):
            return None
        x = x1
        y = y1
        rw = x2 - x1
        rh = y2 - y1

    rw = max(0.001, rw)
    rh = max(0.001, rh)
    x = _clamp01(x)
    y = _clamp01(y)
    if x + rw > 1.0:
        rw = max(0.001, 1.0 - x)
    if y + rh > 1.0:
        rh = max(0.001, 1.0 - y)
    return {"x": _clamp01(x), "y": _clamp01(y), "w": _clamp01(rw), "h": _clamp01(rh)}


def _median_rect(rects: Iterable[Dict[str, float]]) -> Optional[Dict[str, float]]:
    arr = list(rects)
    if not arr:
        return None
    x = statistics.median([r["x"] for r in arr])
    y = statistics.median([r["y"] for r in arr])
    w = statistics.median([r["w"] for r in arr])
    h = statistics.median([r["h"] for r in arr])
    out = _rect_norm_from_edges(x, y, x + w, y + h, 1.0, 1.0)
    return out


def _parse_predictions_csv(pred_csv: Path) -> Dict[str, object]:
    per_image: Dict[str, Dict[str, float]] = {}
    per_image_by_basename: Dict[str, Dict[str, float]] = {}
    global_by_group_lists: Dict[str, List[Dict[str, float]]] = {}
    entries: List[Dict[str, object]] = []

    rows = 0
    with pred_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows += 1
            rel_path = str(row.get("rel_path") or row.get("image_path") or "").strip().replace("\\", "/")
            image_path_raw = str(row.get("image_path") or "").strip()
            key = _normalize_path(rel_path)
            if not key:
                continue

            width = row.get("width")
            height = row.get("height")

            pred = _rect_norm_from_edges(
                row.get("pred_left") or row.get("pred_x1") or row.get("pred_left_norm"),
                row.get("pred_top") or row.get("pred_y1") or row.get("pred_top_norm"),
                row.get("pred_right") or row.get("pred_x2") or row.get("pred_right_norm"),
                row.get("pred_bottom") or row.get("pred_y2") or row.get("pred_bottom_norm"),
                width,
                height,
            )
            if pred is not None:
                per_image[key] = pred
                bname = Path(key).name
                if bname and bname not in per_image_by_basename:
                    per_image_by_basename[bname] = pred

            group_name = str(row.get("group_name") or "").strip()
            if not group_name:
                # fallback first path segment
                parts = Path(key).parts
                group_name = parts[0] if parts else ""

            grect = _rect_norm_from_edges(
                row.get("global_left") or row.get("global_x1") or row.get("global_left_norm"),
                row.get("global_top") or row.get("global_y1") or row.get("global_top_norm"),
                row.get("global_right") or row.get("global_x2") or row.get("global_right_norm"),
                row.get("global_bottom") or row.get("global_y2") or row.get("global_bottom_norm"),
                width,
                height,
            )
            if grect is not None and group_name:
                global_by_group_lists.setdefault(group_name, []).append(grect)

            entries.append(
                {
                    "key": key,
                    "basename": Path(key).name,
                    "rel_path": rel_path,
                    "image_path": image_path_raw,
                    "group_name": group_name,
                    "pred_rect": pred,
                    "global_rect": grect,
                }
            )

    global_by_group: Dict[str, Dict[str, float]] = {}
    for g, rects in global_by_group_lists.items():
        med = _median_rect(rects)
        if med is not None:
            global_by_group[g] = med

    return {
        "rows": rows,
        "per_image": per_image,
        "per_image_by_basename": per_image_by_basename,
        "global_by_group": global_by_group,
        "entries": entries,
    }


def _resolve_entry_image_path(entry: Dict[str, object], input_dir: Path) -> Optional[Path]:
    cands: List[Path] = []
    rel_raw = str(entry.get("rel_path") or "").strip().replace("\\", "/")
    image_path_raw = str(entry.get("image_path") or "").strip()

    if image_path_raw:
        p = Path(image_path_raw).expanduser()
        if p.is_absolute():
            cands.append(p)
        else:
            cands.append((input_dir / p).resolve())

    if rel_raw:
        rel_safe = _safe_rel_path(rel_raw)
        if rel_safe is not None:
            cands.append((input_dir / rel_safe).resolve())

    seen: set[str] = set()
    for cand in cands:
        ckey = cand.as_posix()
        if ckey in seen:
            continue
        seen.add(ckey)
        if cand.exists() and cand.is_file():
            return cand
    return None


def _crop_rgb_by_rect_norm(image, rect: Optional[Dict[str, float]]):  # noqa: ANN001
    if rect is None:
        return image
    try:
        w, h = image.size
    except Exception:  # noqa: BLE001
        return image
    if w < 2 or h < 2:
        return image

    x = _to_float(rect.get("x")) if isinstance(rect, dict) else None
    y = _to_float(rect.get("y")) if isinstance(rect, dict) else None
    rw = _to_float(rect.get("w")) if isinstance(rect, dict) else None
    rh = _to_float(rect.get("h")) if isinstance(rect, dict) else None
    if x is None or y is None or rw is None or rh is None:
        return image
    x = _clamp01(x)
    y = _clamp01(y)
    rw = _clamp01(max(0.001, rw))
    rh = _clamp01(max(0.001, rh))
    if x + rw > 1.0:
        rw = max(0.001, 1.0 - x)
    if y + rh > 1.0:
        rh = max(0.001, 1.0 - y)

    left = int(round(x * w))
    top = int(round(y * h))
    right = int(round((x + rw) * w))
    bottom = int(round((y + rh) * h))

    left = max(0, min(w - 2, left))
    top = max(0, min(h - 2, top))
    right = max(left + 1, min(w, right))
    bottom = max(top + 1, min(h, bottom))
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def _write_uploaded_files(files: List[FileStorage], rel_paths: List[str], dst_root: Path) -> int:
    written = 0
    for idx, fs in enumerate(files):
        rel = rel_paths[idx] if idx < len(rel_paths) else (fs.filename or "")
        rel_safe = _safe_rel_path(rel)
        if rel_safe is None:
            continue
        out = dst_root / rel_safe
        out.parent.mkdir(parents=True, exist_ok=True)
        fs.save(out)
        written += 1
    return written


def _run_inference(
    *,
    input_dir: Path,
    output_dir: Path,
    python_bin: Path,
    checkpoint: Optional[Path],
    vendor_checkpoint: Optional[Path],
    rect_vendor_map: Optional[Path],
    batch_size: int,
    num_workers: int,
    limit_images: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(python_bin),
        str(INFER_SCRIPT),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(max(1, int(batch_size))),
        "--num-workers",
        str(max(0, int(num_workers))),
        "--vendor-log-interval",
        "20",
    ]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    if vendor_checkpoint is not None:
        cmd += ["--vendor-checkpoint", str(vendor_checkpoint)]
    if rect_vendor_map is not None:
        cmd += ["--rect-vendor-map", str(rect_vendor_map)]
    if int(limit_images) > 0:
        cmd += ["--limit-images", str(int(limit_images))]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc


class ZoomWorkspaceApp:
    def __init__(
        self,
        *,
        html_path: Path,
        python_bin: Path,
        checkpoint: Optional[Path],
        vendor_checkpoint: Optional[Path],
        rect_vendor_map: Optional[Path],
        batch_size: int,
        num_workers: int,
        limit_images: int,
        orientation_checkpoint: Optional[Path],
        orientation_batch_size: int,
        orientation_image_size: int,
        orientation_device: Optional[str],
    ) -> None:
        self.url_map = Map(
            [
                Rule("/", endpoint="index"),
                Rule("/health", endpoint="health"),
                Rule("/api/network-infer", endpoint="network_infer", methods=["POST", "OPTIONS"]),
            ]
        )
        self.html_path = html_path
        self.python_bin = python_bin
        self.checkpoint = checkpoint
        self.vendor_checkpoint = vendor_checkpoint
        self.rect_vendor_map = rect_vendor_map
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit_images = limit_images
        self.orientation_checkpoint = orientation_checkpoint
        self.orientation_batch_size = max(1, int(orientation_batch_size))
        self.orientation_image_size = int(orientation_image_size)
        self.orientation_device = str(orientation_device).strip() if orientation_device else ""
        self._orientation_runtime: Optional[Dict[str, object]] = None

    def wsgi_app(self, environ, start_response):  # noqa: ANN001
        request = Request(environ)
        adapter = self.url_map.bind_to_environ(environ)
        try:
            endpoint, values = adapter.match()
            if endpoint == "index":
                resp = self.handle_index(request)
            elif endpoint == "health":
                resp = self.handle_health(request)
            elif endpoint == "network_infer":
                resp = self.handle_network_infer(request)
            else:
                resp = _json_response({"ok": False, "error": f"Unknown endpoint: {endpoint}"}, status=404)
        except HTTPException as exc:
            resp = _json_response({"ok": False, "error": str(exc)}, status=exc.code or 500)
        return resp(environ, start_response)

    def handle_index(self, request: Request) -> Response:  # noqa: ARG002
        if not self.html_path.exists():
            return _text_response(f"Missing HTML: {self.html_path}", status=404)
        return _text_response(self.html_path.read_text(encoding="utf-8"), content_type="text/html; charset=utf-8")

    def handle_health(self, request: Request) -> Response:  # noqa: ARG002
        return _json_response(
            {
                "ok": True,
                "html": self.html_path.as_posix(),
                "python": str(self.python_bin),
                "infer_script": INFER_SCRIPT.as_posix(),
                "orientation_checkpoint": self.orientation_checkpoint.as_posix() if self.orientation_checkpoint else "",
            }
        )

    def _choose_torch_device(self, torch_mod, requested: str) -> object:  # noqa: ANN001
        req = str(requested or "").strip().lower()
        if req:
            if req == "cpu":
                return torch_mod.device("cpu")
            if req == "mps" and hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
                return torch_mod.device("mps")
            if req.startswith("cuda") and torch_mod.cuda.is_available():
                return torch_mod.device(req)
        if hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
            return torch_mod.device("mps")
        if torch_mod.cuda.is_available():
            return torch_mod.device("cuda")
        return torch_mod.device("cpu")

    def _load_orientation_runtime(self) -> tuple[Optional[Dict[str, object]], str]:
        if self._orientation_runtime is not None:
            return self._orientation_runtime, ""
        if self.orientation_checkpoint is None:
            return None, "orientation checkpoint non configurato."
        if not self.orientation_checkpoint.exists():
            return None, f"orientation checkpoint non trovato: {self.orientation_checkpoint}"

        try:
            import torch
            import torch.nn as nn
            from PIL import Image
            from torchvision.models import resnet18
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms import functional as TF
        except Exception as exc:  # noqa: BLE001
            return None, f"import dipendenze orientation fallito: {exc}"

        try:
            class BinaryOrientationClassifier(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    model = resnet18(weights=None)
                    in_features = model.fc.in_features
                    model.fc = nn.Linear(in_features, 2)
                    self.model = model

                def forward(self, x):  # noqa: ANN001
                    return self.model(x)

            device = self._choose_torch_device(torch, self.orientation_device)
            ckpt = torch.load(self.orientation_checkpoint, map_location=device, weights_only=False)
            model_state = ckpt.get("model_state_dict")
            if not isinstance(model_state, dict):
                return None, "checkpoint orientation privo di model_state_dict."
            image_size = int(self.orientation_image_size) if self.orientation_image_size > 0 else int(
                ckpt.get("args", {}).get("image_size", 256)
            )

            model = BinaryOrientationClassifier().to(device)
            model.load_state_dict(model_state)
            model.eval()

            self._orientation_runtime = {
                "torch": torch,
                "Image": Image,
                "TF": TF,
                "InterpolationMode": InterpolationMode,
                "model": model,
                "device": device,
                "image_size": int(max(32, image_size)),
                "mean": torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1),
                "std": torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1),
            }
            return self._orientation_runtime, ""
        except Exception as exc:  # noqa: BLE001
            return None, f"caricamento rete orientation fallito: {exc}"

    def _infer_orientation_from_entries(
        self,
        *,
        entries: List[Dict[str, object]],
        input_dir: Path,
        global_by_group: Dict[str, Dict[str, float]],
    ) -> Dict[str, object]:
        out: Dict[str, object] = {
            "per_image": {},
            "per_image_by_basename": {},
            "summary": {
                "requested": int(len(entries)),
                "predicted": 0,
                "missing_image": 0,
                "decode_failed": 0,
                "label_counts": {"su": 0, "giu": 0},
                "checkpoint": self.orientation_checkpoint.as_posix() if self.orientation_checkpoint else "",
                "device": "",
                "image_size": 0,
                "error": "",
            },
        }
        if not entries:
            return out

        runtime, runtime_err = self._load_orientation_runtime()
        if runtime is None:
            out["summary"]["error"] = runtime_err
            return out

        torch_mod = runtime["torch"]
        Image = runtime["Image"]
        TF = runtime["TF"]
        InterpolationMode = runtime["InterpolationMode"]
        model = runtime["model"]
        device = runtime["device"]
        image_size = int(runtime["image_size"])
        mean = runtime["mean"]
        std = runtime["std"]
        batch_size = max(1, int(self.orientation_batch_size))

        out["summary"]["device"] = str(device)
        out["summary"]["image_size"] = image_size

        per_image: Dict[str, Dict[str, object]] = {}
        per_image_by_basename: Dict[str, Dict[str, object]] = {}

        chunk_meta: List[tuple[str, str]] = []
        chunk_tensor: List[object] = []

        def flush_chunk() -> None:
            if not chunk_tensor:
                return
            x = torch_mod.stack(chunk_tensor, dim=0).to(device)
            with torch_mod.no_grad():
                logits = model(x)
                probs = torch_mod.softmax(logits, dim=1).detach().cpu()
            for i, (key, bname) in enumerate(chunk_meta):
                p_su = float(probs[i, 0].item())
                p_giu = float(probs[i, 1].item())
                pred_idx = 0 if p_su >= p_giu else 1
                pred_label = "su" if pred_idx == 0 else "giu"
                confidence = p_su if pred_idx == 0 else p_giu
                pred = {
                    "label": pred_label,
                    "confidence": confidence,
                    "prob_su": p_su,
                    "prob_giu": p_giu,
                    "source": "orientation_net_rect_crop",
                }
                per_image[key] = pred
                if bname and bname not in per_image_by_basename:
                    per_image_by_basename[bname] = pred
                out["summary"]["predicted"] = int(out["summary"]["predicted"]) + 1
                counts = out["summary"]["label_counts"]
                counts[pred_label] = int(counts.get(pred_label, 0)) + 1
            chunk_meta.clear()
            chunk_tensor.clear()

        for entry in entries:
            key = str(entry.get("key") or "").strip()
            if not key:
                continue

            img_path = _resolve_entry_image_path(entry, input_dir=input_dir)
            if img_path is None:
                out["summary"]["missing_image"] = int(out["summary"]["missing_image"]) + 1
                continue

            try:
                with Image.open(img_path) as img:
                    image = img.convert("RGB")
                    rect = entry.get("pred_rect")
                    if rect is None:
                        g_name = str(entry.get("group_name") or "").strip()
                        if g_name and g_name in global_by_group:
                            rect = global_by_group[g_name]
                        elif entry.get("global_rect") is not None:
                            rect = entry.get("global_rect")
                    crop = _crop_rgb_by_rect_norm(image, rect)
                    crop = TF.resize(
                        crop,
                        size=[image_size, image_size],
                        interpolation=InterpolationMode.BILINEAR,
                        antialias=True,
                    )
                    x = TF.to_tensor(crop)
                    x = (x - mean) / std
            except Exception:  # noqa: BLE001
                out["summary"]["decode_failed"] = int(out["summary"]["decode_failed"]) + 1
                continue

            chunk_meta.append((key, Path(key).name))
            chunk_tensor.append(x)
            if len(chunk_tensor) >= batch_size:
                flush_chunk()

        flush_chunk()
        out["per_image"] = per_image
        out["per_image_by_basename"] = per_image_by_basename
        return out

    def handle_network_infer(self, request: Request) -> Response:
        if request.method == "OPTIONS":
            return _json_response({"ok": True})

        if not INFER_SCRIPT.exists():
            return _json_response({"ok": False, "error": f"Infer script not found: {INFER_SCRIPT}"}, status=500)
        if not self.python_bin.exists():
            return _json_response({"ok": False, "error": f"Python bin not found: {self.python_bin}"}, status=500)

        input_dir_raw = str(request.form.get("input_dir", "")).strip()
        uploaded_files = list(request.files.getlist("files"))
        rel_paths = [str(x) for x in request.form.getlist("rel_paths")]

        with tempfile.TemporaryDirectory(prefix="zoom_ws_backend_") as tmp_s:
            tmp_root = Path(tmp_s)
            use_input_dir: Optional[Path] = None
            source_label = ""

            if input_dir_raw:
                cand = Path(input_dir_raw).expanduser().resolve()
                if not cand.exists() or not cand.is_dir():
                    return _json_response({"ok": False, "error": f"input_dir non valida: {cand}"}, status=400)
                use_input_dir = cand
                source_label = cand.as_posix()
            else:
                if not uploaded_files:
                    return _json_response(
                        {"ok": False, "error": "Nessun input: specifica input_dir oppure files+rel_paths."},
                        status=400,
                    )
                upload_root = tmp_root / "input"
                upload_root.mkdir(parents=True, exist_ok=True)
                written = _write_uploaded_files(uploaded_files, rel_paths, upload_root)
                if written <= 0:
                    return _json_response({"ok": False, "error": "Upload ricevuto ma nessun file valido scritto."}, status=400)
                use_input_dir = upload_root
                source_label = f"uploaded:{written}"

            out_dir = tmp_root / "infer_out"
            out_dir.mkdir(parents=True, exist_ok=True)

            proc = _run_inference(
                input_dir=use_input_dir,
                output_dir=out_dir,
                python_bin=self.python_bin,
                checkpoint=self.checkpoint,
                vendor_checkpoint=self.vendor_checkpoint,
                rect_vendor_map=self.rect_vendor_map,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                limit_images=self.limit_images,
            )

            if proc.returncode != 0:
                return _json_response(
                    {
                        "ok": False,
                        "error": "Inferenza rete fallita.",
                        "returncode": proc.returncode,
                        "stdout_tail": proc.stdout[-4000:],
                        "stderr_tail": proc.stderr[-4000:],
                    },
                    status=500,
                )

            pred_csv = out_dir / "predictions.csv"
            if not pred_csv.exists():
                return _json_response(
                    {
                        "ok": False,
                        "error": f"predictions.csv non trovato: {pred_csv}",
                        "stdout_tail": proc.stdout[-4000:],
                        "stderr_tail": proc.stderr[-4000:],
                    },
                    status=500,
                )

            parsed = _parse_predictions_csv(pred_csv)
            summary_obj: Dict[str, object] = {}
            summary_json = out_dir / "summary.json"
            if summary_json.exists():
                try:
                    summary_obj = json.loads(summary_json.read_text(encoding="utf-8"))
                except Exception:
                    summary_obj = {}

            orientation = self._infer_orientation_from_entries(
                entries=list(parsed.get("entries", []) or []),
                input_dir=use_input_dir,
                global_by_group=dict(parsed.get("global_by_group", {}) or {}),
            )

            return _json_response(
                {
                    "ok": True,
                    "source": source_label,
                    "rows": int(parsed.get("rows", 0) or 0),
                    "per_image": parsed.get("per_image", {}),
                    "per_image_by_basename": parsed.get("per_image_by_basename", {}),
                    "global_by_group": parsed.get("global_by_group", {}),
                    "orientation_per_image": orientation.get("per_image", {}),
                    "orientation_per_image_by_basename": orientation.get("per_image_by_basename", {}),
                    "orientation_summary": orientation.get("summary", {}),
                    "stdout_tail": proc.stdout[-2000:],
                    "summary": {
                        "rect_routing": summary_obj.get("rect_routing", {}),
                        "vendor_recognition": summary_obj.get("vendor_recognition", {}),
                        "group_rects": summary_obj.get("group_rects", {}),
                    },
                }
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backend locale per zoom workspace (rete rect).")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--html", type=Path, default=DEFAULT_HTML)
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_RECT_CHECKPOINT,
    )
    p.add_argument(
        "--vendor-checkpoint",
        type=Path,
        default=DEFAULT_VENDOR_CHECKPOINT,
    )
    p.add_argument(
        "--rect-vendor-map",
        type=Path,
        default=DEFAULT_RECT_VENDOR_MAP,
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit-images", type=int, default=0)
    p.add_argument(
        "--orientation-checkpoint",
        type=Path,
        default=DEFAULT_ORIENTATION_CHECKPOINT,
        help="Checkpoint rete binaria su/giu (classificazione su crop rettangolo).",
    )
    p.add_argument("--orientation-batch-size", type=int, default=96)
    p.add_argument("--orientation-image-size", type=int, default=0, help="Override size input orientation. 0=da ckpt.")
    p.add_argument("--orientation-device", type=str, default=None, help="cpu | mps | cuda[:idx] (auto se omesso).")
    return p


def main() -> int:
    args = build_parser().parse_args()

    html_path = args.html.expanduser().resolve()
    python_bin = args.python.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    vendor_checkpoint = args.vendor_checkpoint.expanduser().resolve() if args.vendor_checkpoint else None
    rect_vendor_map = args.rect_vendor_map.expanduser().resolve() if args.rect_vendor_map else None
    orientation_checkpoint = args.orientation_checkpoint.expanduser().resolve() if args.orientation_checkpoint else None

    app = ZoomWorkspaceApp(
        html_path=html_path,
        python_bin=python_bin,
        checkpoint=checkpoint,
        vendor_checkpoint=vendor_checkpoint,
        rect_vendor_map=rect_vendor_map,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        limit_images=int(args.limit_images),
        orientation_checkpoint=orientation_checkpoint,
        orientation_batch_size=int(args.orientation_batch_size),
        orientation_image_size=int(args.orientation_image_size),
        orientation_device=args.orientation_device,
    )

    print(f"Zoom backend listening on http://{args.host}:{args.port}")
    print(f"HTML: {html_path}")
    print(f"Infer script: {INFER_SCRIPT}")
    if orientation_checkpoint:
        print(f"Orientation checkpoint: {orientation_checkpoint}")
    run_simple(
        hostname=args.host,
        port=int(args.port),
        application=app.wsgi_app,
        threaded=True,
        use_reloader=False,
        use_debugger=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
