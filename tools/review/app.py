#!/usr/bin/env python3
"""Revisione snella - local web tool to review the .fss pipeline and send back feedback.

Three screens:
1. home: run history + folder picker;
2. run: stage-by-stage progress, live, folder by folder;
3. review: one folder, every module's output on every image, with comments and corrections.

The server is thin on purpose. It launches the official pipeline, reads what the pipeline
writes, and appends feedback to ``feedback/inbox.jsonl``. It never re-implements a prediction:
if a number is on this page, some module in the pipeline produced it.

Avvio:
  OldSoftwareEsiBuilder/.venv-mps/bin/python tools/review/app.py --port 8790
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from flask import Flask, jsonify, request, send_file, send_from_directory  # noqa: E402
from PIL import Image  # noqa: E402

import areas as areas_mod  # noqa: E402
from feedback_store import FeedbackStore, derive_mm_per_px  # noqa: E402
from orchestrator import VendorKnowledge  # noqa: E402
from runner import (  # noqa: E402
    RUNS_ROOT,
    STAGE_LABELS,
    STAGE_ORDER,
    RunManager,
    known_dataset_roots,
    list_candidate_folders,
    python_bin,
    set_python_bin,
)
from study import FolderStudy, folder_coherence  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPE_MODELS = REPO_ROOT / "artifacts/10_active_pipeline/pipeline_fss_head/models"
CACHE_DIR = REPO_ROOT / "artifacts" / "72_review_snella" / "cache"

app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")
manager = RunManager()
store = FeedbackStore()
_allow_roots: List[Path] = []
_knowledge_cache: Dict[str, object] = {}


# --------------------------------------------------------------------------- paths
def _allowed(path: Path) -> bool:
    """Only serve files under a root we know about: a dataset root, a run dir, artifacts."""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    roots = [RUNS_ROOT, REPO_ROOT / "artifacts", *_allow_roots, *known_dataset_roots()]
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except (ValueError, OSError):
            continue
    return False


def _cached_jpeg(key: str) -> Path:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    out = CACHE_DIR / digest[:2] / f"{digest}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _classes_from_confusion(path: Path) -> List[str]:
    """Class names without loading a checkpoint: they are the confusion matrix's header."""
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
    except (OSError, StopIteration):
        return []
    return [h.strip() for h in header[1:] if h.strip()]


def _vendor_classes() -> List[str]:
    for name in ("vendor_training_no_negative_v2_power",):
        found = _classes_from_confusion(
            PIPE_MODELS / name / "test_confusion_matrix_from_best.csv")
        if found:
            return found
    return []


def _probe_classes() -> List[str]:
    directory = PIPE_MODELS / "probe_training_no_negative_v1"
    found = _classes_from_confusion(directory / "test_confusion_matrix.csv")
    if found:
        return found
    rows = []
    metrics = directory / "test_per_probe_metrics.csv"
    if metrics.is_file():
        with metrics.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                label = str(row.get("probe") or row.get("class") or "").strip()
                if label:
                    rows.append(label)
    return rows


# --------------------------------------------------------------------------- pages
@app.get("/")
def index() -> object:
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/catalog")
def api_catalog() -> object:
    payload = areas_mod.catalog()
    payload["stage_order"] = STAGE_ORDER
    payload["stage_labels"] = STAGE_LABELS
    payload["options_sources"] = {
        "vendor_classes": _vendor_classes(),
        "probe_classes": _probe_classes(),
    }
    return jsonify(payload)


@app.get("/api/roots")
def api_roots() -> object:
    """Roots worth offering in the picker: the volumes, plus whatever past runs used."""
    candidates: List[Path] = []
    volumes = Path("/Volumes")
    if volumes.is_dir():
        for entry in sorted(volumes.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                candidates.append(entry)
                for child in sorted(entry.iterdir())[:40]:
                    if child.is_dir() and not child.name.startswith("."):
                        candidates.append(child)
    for entry in manager.history(limit=50):
        root = str(entry.get("dataset_root", ""))
        if root:
            candidates.append(Path(root))
    candidates.extend(_allow_roots)

    seen: Dict[str, Dict[str, object]] = {}
    for path in candidates:
        key = path.as_posix()
        if key in seen or not path.is_dir():
            continue
        seen[key] = {"path": key, "name": path.name or key}
    return jsonify({"roots": list(seen.values())})


@app.get("/api/folders")
def api_folders() -> object:
    root = request.args.get("root", "").strip()
    if not root:
        return jsonify({"error": "root mancante"}), 400
    path = Path(root).expanduser()
    if not path.is_dir():
        return jsonify({"error": f"non è una cartella: {root}"}), 400
    return jsonify({"root": path.as_posix(), "folders": list_candidate_folders(path)})


# --------------------------------------------------------------------------- runs
@app.get("/api/runs")
def api_runs() -> object:
    counts: Dict[str, int] = {}
    for entry in store.all():
        run_id = str((entry.get("target") or {}).get("run_id", ""))  # type: ignore[union-attr]
        if run_id:
            counts[run_id] = counts.get(run_id, 0) + 1
    history = manager.history()
    for entry in history:
        entry["feedback_count"] = counts.get(str(entry["run_id"]), 0)
    active = manager.active()
    return jsonify({"runs": history, "active": active.run_id if active else None})


@app.post("/api/runs")
def api_run_start() -> object:
    payload = request.get_json(silent=True) or {}
    root = str(payload.get("dataset_root", "")).strip()
    names = [str(n) for n in (payload.get("folders") or []) if str(n).strip()]
    options = dict(payload.get("options") or {})
    if not root or not names:
        return jsonify({"error": "servono dataset_root e almeno una cartella"}), 400
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        return jsonify({"error": f"non è una cartella: {root}"}), 400
    folders = []
    for name in names:
        candidate = root_path / name
        if not candidate.is_dir():
            return jsonify({"error": f"cartella inesistente: {name}"}), 400
        folders.append({"name": name, "path": candidate.as_posix()})
    try:
        run = manager.start(root_path, folders, options)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"run_id": run.run_id, "run": run.manifest()})


@app.get("/api/runs/<run_id>")
def api_run(run_id: str) -> object:
    run = manager.get(run_id)
    if run is None:
        return jsonify({"error": "run non trovata"}), 404
    return jsonify({"run": run.manifest(), "event_count": run.event_count})


@app.get("/api/runs/<run_id>/events")
def api_run_events(run_id: str) -> object:
    run = manager.get(run_id)
    if run is None:
        return jsonify({"error": "run non trovata"}), 404
    since = int(request.args.get("since", 0) or 0)
    events = run.events_since(since)
    return jsonify({
        "events": events,
        "next": since + len(events),
        "state": run.state,
        "folders": [job.to_dict() for job in run.jobs],
    })


@app.post("/api/runs/<run_id>/cancel")
def api_run_cancel(run_id: str) -> object:
    run = manager.get(run_id)
    if run is None:
        return jsonify({"error": "run non trovata"}), 404
    run.cancel()
    return jsonify({"ok": True, "state": run.state})


@app.get("/api/runs/<run_id>/folders/<slug>/study")
def api_study(run_id: str, slug: str) -> object:
    run = manager.get(run_id)
    if run is None:
        return jsonify({"error": "run non trovata"}), 404
    job = next((j for j in run.jobs if j.slug == slug), None)
    if job is None:
        return jsonify({"error": "cartella non trovata nella run"}), 404
    study = FolderStudy(run.folders_dir / slug)
    if not study.available:
        return jsonify({
            "ready": False,
            "state": job.state,
            "stages": job.stages,
            "message": "la cartella non ha ancora prodotto risultati",
        })
    folder = study.folder_summary()
    images = study.images()
    feedback = store.query(run_id=run_id, folder=job.name)
    return jsonify({
        "ready": True,
        "state": job.state,
        "run_id": run_id,
        "slug": slug,
        "folder": folder,
        "images": images,
        "folder_checks": folder_coherence(folder, images),
        "stages": job.stages,
        "feedback": feedback,
        "run_dir": (run.folders_dir / slug).as_posix(),
    })


# --------------------------------------------------------------------------- media
@app.get("/api/image")
def api_image() -> object:
    raw = request.args.get("path", "")
    path = Path(raw).expanduser()
    if not path.is_file() or not _allowed(path):
        return jsonify({"error": "immagine non disponibile"}), 404
    return send_file(path.as_posix())


@app.get("/api/thumb")
def api_thumb() -> object:
    raw = request.args.get("path", "")
    width = max(64, min(1600, int(request.args.get("w", 320) or 320)))
    path = Path(raw).expanduser()
    if not path.is_file() or not _allowed(path):
        return jsonify({"error": "immagine non disponibile"}), 404
    cache = _cached_jpeg(f"thumb::{path.as_posix()}::{width}::{path.stat().st_mtime_ns}")
    if not cache.is_file():
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                ratio = width / float(img.width)
                img = img.resize((width, max(1, int(img.height * ratio))), Image.LANCZOS)
                img.save(cache, format="JPEG", quality=82)
        except (OSError, ValueError) as exc:
            return jsonify({"error": f"thumbnail non generata: {exc}"}), 500
    return send_file(cache.as_posix(), mimetype="image/jpeg")


@app.get("/api/crop")
def api_crop() -> object:
    """A zoom on one box: how the operator actually judges a depth label or a ruler."""
    raw = request.args.get("path", "")
    path = Path(raw).expanduser()
    if not path.is_file() or not _allowed(path):
        return jsonify({"error": "immagine non disponibile"}), 404
    try:
        left = float(request.args.get("left", 0))
        top = float(request.args.get("top", 0))
        right = float(request.args.get("right", 0))
        bottom = float(request.args.get("bottom", 0))
    except ValueError:
        return jsonify({"error": "coordinate non valide"}), 400
    pad = max(0, min(400, int(request.args.get("pad", 24) or 24)))
    zoom = max(1.0, min(8.0, float(request.args.get("zoom", 3) or 3)))
    if right <= left or bottom <= top:
        return jsonify({"error": "box vuoto"}), 400
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            box = (
                max(0, int(left - pad)),
                max(0, int(top - pad)),
                min(img.width, int(right + pad)),
                min(img.height, int(bottom + pad)),
            )
            crop = img.crop(box)
            crop = crop.resize(
                (max(1, int(crop.width * zoom)), max(1, int(crop.height * zoom))),
                Image.LANCZOS,
            )
            buffer = io.BytesIO()
            crop.save(buffer, format="JPEG", quality=88)
            buffer.seek(0)
    except (OSError, ValueError) as exc:
        return jsonify({"error": f"crop non generato: {exc}"}), 500
    return send_file(buffer, mimetype="image/jpeg")


# --------------------------------------------------------------------------- feedback
@app.get("/api/feedback")
def api_feedback_list() -> object:
    filters = {k: request.args.get(k) for k in
               ("status", "area", "run_id", "folder", "image_id", "vendor", "kind", "verdict",
                "tag")}
    filters = {k: v for k, v in filters.items() if v}
    entries = store.query(**filters)  # type: ignore[arg-type]
    return jsonify({"entries": entries, "stats": store.stats()})


@app.post("/api/feedback")
def api_feedback_add() -> object:
    payload = request.get_json(silent=True) or {}
    area = str(payload.get("area", "")).strip()
    if area not in areas_mod.AREA_BY_ID:
        return jsonify({"error": f"area sconosciuta: {area}"}), 400
    scope = str(payload.get("scope", "folder")).strip() or "folder"
    tag = str(payload.get("tag", "") or "").strip()
    comment = str(payload.get("comment", "") or "").strip()
    correction = dict(payload.get("correction") or {})
    if not comment and not tag and not correction:
        return jsonify({"error": "un feedback vuoto non serve a nessuno"}), 400

    suggestion = areas_mod.suggestion(area, tag) if tag else None
    verdict = str(payload.get("verdict") or (suggestion or {}).get("verdict") or "").strip()
    severity = str(payload.get("severity") or (suggestion or {}).get("severity")
                   or "major").strip()
    if not comment and suggestion:
        comment = str(suggestion.get("text", ""))

    target = dict(payload.get("target") or {})
    context, prediction, provenance = _auto_context(target, area, payload)

    if area == "scala" and correction:
        derived = derive_mm_per_px(correction)
        if derived is not None:
            correction["mm_per_px_derived"] = derived

    kind = str(payload.get("kind") or ("correction" if correction else
                                       ("confirm" if verdict == "ok" else "comment")))
    entry = store.append(
        area=area, kind=kind, scope=scope, comment=comment, tag=tag, verdict=verdict,
        severity=severity, target=target, prediction=prediction, correction=correction,
        context=context, provenance=provenance,
    )
    return jsonify({"entry": entry})


@app.post("/api/feedback/<entry_id>/resolve")
def api_feedback_resolve(entry_id: str) -> object:
    payload = request.get_json(silent=True) or {}
    entry = store.resolve(
        entry_id,
        status=str(payload.get("status", "done")),
        commit=str(payload.get("commit", "")),
        note=str(payload.get("note", "")),
        by=str(payload.get("by", "camilla")),
    )
    if entry is None:
        return jsonify({"error": "feedback non trovato"}), 404
    return jsonify({"entry": entry})


@app.delete("/api/feedback/<entry_id>")
def api_feedback_delete(entry_id: str) -> object:
    if not store.delete(entry_id):
        return jsonify({"error": "feedback non trovato"}), 404
    return jsonify({"ok": True})


def _auto_context(target: Dict[str, object], area: str,
                  payload: Dict[str, object]) -> tuple:
    """Fill in the context from the run on disk, so the operator only writes the comment.

    A comment whose thresholds, confidences and checkpoints have to be reconstructed by
    reopening the run is a comment nobody acts on. Everything here is read, not typed.
    """
    context: Dict[str, object] = {}
    prediction: Dict[str, object] = dict(payload.get("prediction") or {})
    provenance: Dict[str, object] = {}

    run_id = str(target.get("run_id", "") or "")
    slug = str(target.get("slug", "") or "")
    image_key = str(target.get("image_id", "") or "")
    if not run_id or not slug:
        return context, prediction, provenance

    run = manager.get(run_id)
    if run is None:
        return context, prediction, provenance
    run_dir = run.folders_dir / slug
    study = FolderStudy(run_dir)
    if not study.available:
        return context, prediction, provenance

    folder = study.folder_summary()
    context.update({
        "vendor": (folder.get("vendor") or {}).get("value", ""),
        "vendor_conf": (folder.get("vendor") or {}).get("confidence"),
        "vendor_threshold": (folder.get("vendor") or {}).get("threshold"),
        "probe_id": (folder.get("probe") or {}).get("value", ""),
        "probe_conf": (folder.get("probe") or {}).get("confidence"),
        "line_11": (folder.get("rect") or {}).get("line_11", ""),
        "line_11_method": (folder.get("rect") or {}).get("method", ""),
        "rotation_deg": (folder.get("rotazione") or {}).get("deg"),
        "images_studied": (folder.get("dedup") or {}).get("images_unique"),
        "folder_status": folder.get("status", ""),
        "pipeline_options": run.options,
    })
    provenance.update({
        "run_dir": run_dir.as_posix(),
        "pipeline_output": (run_dir / "pipeline_output").as_posix(),
        "step_checks": (run_dir / "step_checks.json").as_posix(),
        "folder_csv": (run_dir / "pipeline_output" / "folder_fss_head_predictions.csv").as_posix(),
    })
    # The pipeline works through a symlinked copy of the folder; a training queue must point at
    # the real files on the volume, which outlive the run dir.
    original_folder = str((study.step_checks or {}).get("input_original_folder", "") or "")
    if original_folder:
        context["folder_path_original"] = original_folder

    area_summary = folder.get(area)
    if isinstance(area_summary, dict) and not prediction:
        prediction = {k: v for k, v in area_summary.items() if k != "per_depth"}

    if image_key:
        for image in study.images():
            if str(image.get("image_id")) == image_key or str(image.get("key")) == image_key.lower():
                context["image_size"] = [image.get("image_width"), image.get("image_height")]
                context["image_checks"] = image.get("checks")
                context["image_path"] = image.get("image_path")
                try:
                    resolved = Path(str(image.get("image_path", ""))).resolve()
                    if resolved.is_file():
                        context["image_path_original"] = resolved.as_posix()
                except OSError:
                    pass
                area_image = image.get(area if area != "rect" else "rect_per_image")
                if isinstance(area_image, dict):
                    prediction = dict(area_image)
                # The neighbours matter: a depth is judged with its rect and its ruler.
                context["neighbours"] = {
                    key: image.get(key)
                    for key in ("depth", "scala", "su_giu", "lt", "lr_marker",
                                "rect_per_image", "rect_vs_folder")
                    if key != area and image.get(key)
                }
                break

    return context, prediction, provenance


# --------------------------------------------------------------------------- orchestrator
@app.get("/api/orchestrator")
def api_orchestrator() -> object:
    if not _knowledge_cache.get("payload") or request.args.get("refresh"):
        knowledge = VendorKnowledge()
        knowledge.add_all_runs()
        knowledge.add_feedback(store)
        _knowledge_cache["payload"] = knowledge.build()
    return jsonify(_knowledge_cache["payload"])


@app.post("/api/orchestrator/write")
def api_orchestrator_write() -> object:
    knowledge = VendorKnowledge()
    knowledge.add_all_runs()
    knowledge.add_feedback(store)
    path = knowledge.write()
    _knowledge_cache["payload"] = knowledge.build()
    return jsonify({"ok": True, "path": path.as_posix()})


# --------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--inbox", type=Path, default=None,
                        help="File JSONL dei feedback (default feedback/inbox.jsonl).")
    parser.add_argument("--allow-root", action="append", type=Path, default=[],
                        help="Radice aggiuntiva da cui servire immagini. Ripetibile.")
    parser.add_argument("--python-bin", default=None,
                        help="Python con cui lanciare la pipeline (default: il venv .venv-mps).")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    global store
    if args.inbox:
        store = FeedbackStore(args.inbox)
    _allow_roots.extend(Path(p).expanduser() for p in args.allow_root)
    set_python_bin(args.python_bin)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Revisione snella su http://{args.host}:{args.port}", flush=True)
    print(f"Feedback in {store.path}", flush=True)
    print(f"Pipeline con {python_bin()}", flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
