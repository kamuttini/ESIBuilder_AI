"""Depth, scale and orientation-marker stages, run the way the pipeline runs them.

These three are launched as **subprocesses** with a `pipeline_context.json`, exactly as
`predict_fss_head_from_acquisitions.py` does. It is deliberate, not laziness:

* the scale detector needs cv2 and Tesseract, and a folder whose OCR wedges must cost one
  folder marked ``review``, not the whole app;
* the context JSON left on disk is what makes a run reproducible by hand, and what the review
  pages already know how to read;
* no logic is duplicated, so the app and the pipeline cannot drift apart.

The frame context handed to the scale stage is woven from what the earlier stages already
know (up/down, rect, depth) with the same rules as the pipeline's `_build_scale_frames_context`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPTH_SCRIPT = REPO_ROOT / "tools" / "depth" / "predict_rect_depth_autonomous.py"
RANKER_RELATIVE = Path("artifacts/24_rect_depth_hybrid/candidate_ranker_v11_bk_reviews/ranker.joblib")
SCALE_SCRIPT = REPO_ROOT / "tools" / "scale" / "predict_scale_from_pipeline.py"
SCALE_STUDY_SCRIPT = REPO_ROOT / "tools" / "scale" / "study_scale_folder.py"
MARKER_SCRIPT = REPO_ROOT / "tools" / "orientation" / "predict_marker_envelopes_batch.py"

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def stage_dir(output_root: Path, stage: str, folder: Path, index: int = 0) -> Path:
    """Same layout as the pipeline: NNNN_<slug>_<hash> under the stage name."""
    slug = _SAFE_RE.sub("_", folder.name).strip("_")[:60] or f"folder_{index:04d}"
    digest = hashlib.sha1(folder.as_posix().encode("utf-8")).hexdigest()[:8]
    target = Path(output_root) / stage / f"{index:04d}_{slug}_{digest}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def rect_to_line11(rect: Dict) -> str:
    return "{top}|{left}|{bottom}|{right}|".format(**{k: int(rect[k]) for k in rect})


def rect_to_cli(rect: Dict) -> str:
    """left,top,right,bottom, the order the depth module expects."""
    return f"{int(rect['left'])},{int(rect['top'])},{int(rect['right'])},{int(rect['bottom'])}"


def _run(cmd: List[str], log_path: Path, timeout: Optional[float]) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        log_path.write_text(str(error.stdout or error), encoding="utf-8")
        return -1, f"timeout dopo {timeout}s"
    except Exception as error:  # noqa: BLE001
        log_path.write_text(str(error), encoding="utf-8")
        return -1, str(error)
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    return proc.returncode, ""


def _checkout_roots() -> List[Path]:
    """This checkout first, then the main one when we are inside a git worktree.

    `artifacts/` is never committed, so in a worktree it simply does not exist and every
    default path the modules compute from their own location points at nothing.
    """
    roots = [REPO_ROOT]
    marker = REPO_ROOT / ".git"
    if marker.is_file():
        try:
            text = marker.read_text(encoding="utf-8").strip()
        except OSError:
            return roots
        if text.startswith("gitdir:"):
            gitdir = Path(text.split(":", 1)[1].strip())
            for parent in gitdir.parents:
                if parent.name == ".git":
                    roots.append(parent.parent)
                    break
    return roots


def depth_ranker_path() -> str:
    """The tabular ranker the depth module uses as a prior, or "" if it is not on disk."""
    for root in _checkout_roots():
        candidate = root / RANKER_RELATIVE
        if candidate.is_file():
            return candidate.as_posix()
    return ""


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{str(k): ("" if v is None else str(v)) for k, v in row.items()}
                for row in csv.DictReader(handle)]


# --- depth ----------------------------------------------------------------
def run_depth(
    *,
    folder: Path,
    output_root: Path,
    python_bin: str,
    vendor: str,
    probe_id: str,
    rect: Dict,
    probe_model: str = "",
    video_size: Sequence[int],
    rotation: int = 0,
    max_images: int = 24,
    timeout: float = 900.0,
) -> Dict:
    if not DEPTH_SCRIPT.is_file():
        return {"status": "error", "error": f"script assente: {DEPTH_SCRIPT}", "rows": []}

    target = stage_dir(output_root, "rect_depth_autonomous", folder)
    context = {
        "folder_path": Path(folder).as_posix(),
        "vendor_predicted": vendor or "",
        "manufacturer": vendor or "",
        "predicted_probe_id": str(probe_id or ""),
        "predicted_probe": str(probe_model or probe_id or ""),
        "probe_name": str(probe_model or ""),
        "line_11_rect_echo": rect_to_line11(rect),
        "rect_echo_cli_left_top_right_bottom": rect_to_cli(rect),
        "video_x": int(video_size[0] or 0),
        "video_y": int(video_size[1] or 0),
        "rotation_deg_clockwise": int(rotation),
        "pipeline_stage": "app_rect_depth_autonomous",
    }
    (target / "pipeline_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cmd = [
        python_bin, DEPTH_SCRIPT.as_posix(),
        "--folder", Path(folder).as_posix(),
        "--output-dir", target.as_posix(),
        "--context-json", (target / "pipeline_context.json").as_posix(),
        "--max-images", str(int(max_images)),
    ]
    if vendor:
        cmd += ["--vendor", str(vendor)]
    # The module matches the probe by *model* (TLC3-13, 8848, 18L5...): the numeric
    # anagrafica id says nothing to its profile rules.
    probe_arg = str(probe_model or probe_id or "")
    if probe_arg:
        cmd += ["--probe", probe_arg]
    cmd += ["--rect-echo", rect_to_cli(rect)]
    ranker = depth_ranker_path()
    if ranker:
        cmd += ["--ranker-model", ranker]

    code, error = _run(cmd, target / "pipeline_subprocess.log", timeout)
    rows = _read_csv(target / "rect_depth_autonomous_predictions.csv")
    summary = {}
    summary_path = target / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}

    depths = sorted({
        round(float(row["depth_mm"]), 2)
        for row in rows
        if str(row.get("status", "")).lower() in ("accepted", "review")
        and _float(row.get("depth_mm")) not in (None, 0.0)
    })
    accepted = sum(1 for row in rows if str(row.get("status", "")).lower() == "accepted")
    return {
        "status": "ok" if code == 0 and rows else "error",
        "error": error or ("" if code == 0 else f"returncode={code}"),
        "output_dir": target.as_posix(),
        "images": len(rows),
        "accepted": accepted,
        "acceptance_ratio": round(accepted / len(rows), 4) if rows else 0.0,
        "depths_mm": depths,
        "ranker_model": str(summary.get("ranker_model") or ""),
        "folder_strategy": summary.get("folder_strategy") or {},
        "summary": summary,
        "rows": rows,
    }


def _float(value) -> Optional[float]:  # noqa: ANN001
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


# --- scale ----------------------------------------------------------------
def build_scale_frames(
    *,
    images: Sequence[Path],
    su_giu_rows: Sequence[Dict],
    depth_rows: Sequence[Dict],
    rect: Dict,
) -> List[Dict]:
    """Weave up/down, rect and depth per frame. Same rules as the pipeline, joined by path."""
    sugiu_by_path = {str(row.get("image_path", "")): row for row in su_giu_rows}

    depth_by_path: Dict[str, Dict] = {}
    for row in depth_rows:
        path = str(row.get("image_path", "") or "")
        if not path:
            continue
        status = str(row.get("status", "") or "").strip().lower()
        mode = str(row.get("mode", "") or "").strip()
        rank = ({"accepted": 2, "review": 1}.get(status, 0), 1 if mode == "direct_label" else 0)
        previous = depth_by_path.get(path)
        if previous is None or rank > previous["_rank"]:
            depth_by_path[path] = {**row, "_rank": rank}

    folder_rect = [
        float(rect["left"]), float(rect["top"]), float(rect["right"]), float(rect["bottom"])
    ]

    frames: List[Dict] = []
    for path in images:
        key = Path(path).as_posix()
        sugiu_row = sugiu_by_path.get(key, {})
        depth_row = depth_by_path.get(key, {})

        rect_ltrb = None
        try:
            rect_ltrb = [
                float(sugiu_row["crop_left"]), float(sugiu_row["crop_top"]),
                float(sugiu_row["crop_right"]), float(sugiu_row["crop_bottom"]),
            ]
        except (KeyError, TypeError, ValueError):
            rect_ltrb = list(folder_rect)

        depth_mm = _float(depth_row.get("depth_mm")) if depth_row else None
        if depth_mm is not None and depth_mm <= 0:
            depth_mm = None
        depth_status = str(depth_row.get("status", "") or "")
        if depth_status.strip().lower() == "reject":
            depth_mm = None
        depth_mode = str(depth_row.get("mode", "") or "")

        frames.append(
            {
                "image_path": key,
                "sugiu": str(sugiu_row.get("pred_label", "") or ""),
                "sugiu_conf": _float(sugiu_row.get("confidence")) or 0.0,
                # No marker rows here: the label is the network's, and it is said so.
                "sugiu_source": "rete" if sugiu_row.get("pred_label") else "",
                "marker_score": None,
                "orientation_group": "",
                "rect_ltrb": rect_ltrb,
                "depth_mm": depth_mm,
                "depth_from_interface": depth_mode == "direct_label",
                "depth_mode": depth_mode,
                "depth_status": depth_status,
                "depth_box": [
                    _float(depth_row.get(k)) for k in ("left", "top", "right", "bottom")
                ] if depth_row else None,
                "depth_ocr_text": str(depth_row.get("ocr_text", "") or ""),
            }
        )
    return frames


def run_scale(
    *,
    folder: Path,
    output_root: Path,
    python_bin: str,
    vendor: str,
    vendor_confidence: float,
    probe_id: str,
    rect: Dict,
    video_size: Sequence[int],
    frames: Sequence[Dict],
    rotation: int = 0,
    max_frames: int = 40,
    timeout: float = 900.0,
) -> Dict:
    if not SCALE_SCRIPT.is_file():
        return {"status": "error", "error": f"script assente: {SCALE_SCRIPT}"}

    target = stage_dir(output_root, "scale", folder)
    context = {
        "folder_path": Path(folder).as_posix(),
        "vendor_predicted": vendor or "",
        "vendor_confidence": float(vendor_confidence or 0.0),
        "predicted_probe_id": str(probe_id or ""),
        "line_11_rect_echo": rect_to_line11(rect),
        "video_x": int(video_size[0] or 0),
        "video_y": int(video_size[1] or 0),
        "rotation_deg_clockwise": int(rotation),
        "pipeline_stage": "app_scale_line21",
        "frames": list(frames),
    }
    (target / "pipeline_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    cmd = [
        python_bin, SCALE_SCRIPT.as_posix(),
        "--context-json", (target / "pipeline_context.json").as_posix(),
        "--output-dir", target.as_posix(),
        "--max-frames", str(int(max_frames)),
    ]
    code, error = _run(cmd, target / "pipeline_subprocess.log", timeout)

    summary = {}
    summary_path = target / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    per_depth = _read_csv(target / "scale_per_depth.csv")
    lines = {
        key: summary.get(key, "")
        for key in (
            "line_18_vect_depth", "line_19_pixel_ratio_x",
            "line_20_pixel_ratio_y", "line_21_scale_line",
        )
    }
    return {
        "status": "ok" if code == 0 and any(lines.values()) else "error",
        "error": error or ("" if code == 0 else f"returncode={code}"),
        "output_dir": target.as_posix(),
        "lines": lines,
        "depths_total": summary.get("depths_total"),
        "depths_accepted": summary.get("depths_accepted"),
        "acceptance_ratio": summary.get("acceptance_ratio"),
        "review_reasons": summary.get("review_reasons") or [],
        "profile": summary.get("profile", ""),
        "per_depth": per_depth,
        "summary": summary,
    }


# --- studio della scala: il righello, tacca per tacca ----------------------
def run_scale_study(
    *,
    context_dir: Path,
    python_bin: str,
    vendor: str = "",
    max_images: int = 14,
    corrections: Optional[Path] = None,
    only: Optional[List[str]] = None,
    timeout: float = 1800.0,
) -> Dict:
    """Lo studio del righello: dove sta, le tacche, lo zero, il passo, i numeri.

    E' il modulo autonomo (`detect_scale_ladder` + i controlli incrociati di
    `study_scale_folder`), non lo stadio che consolida `#18-#21`: qui non serve la depth,
    perche' due etichette lette a due altezze danno gia' `mm_per_px` e lo zero.

    Gli si passa il contesto che lo stadio della scala ha gia' scritto — stessi fotogrammi,
    stesso rettangolo, stesso orientamento — con `--from-pipeline`, cosi' i due guardano le
    stesse immagini e i loro esiti sono confrontabili.
    """
    if not SCALE_STUDY_SCRIPT.is_file():
        return {"status": "error", "error": f"script assente: {SCALE_STUDY_SCRIPT}"}
    contesto = Path(context_dir) / "pipeline_context.json"
    if not contesto.is_file():
        return {"status": "error", "error": f"contesto assente: {contesto}"}

    target = Path(context_dir) / "studio"
    target.mkdir(parents=True, exist_ok=True)
    dati = target / "scale_study.json"
    cmd = [
        python_bin, SCALE_STUDY_SCRIPT.as_posix(),
        "--from-pipeline", contesto.as_posix(),
        "--data-json", dati.as_posix(),
        "--out", (target / "scale_study.html").as_posix(),
        "--max-images", str(int(max_images)),
    ]
    if vendor:
        cmd += ["--vendor", str(vendor)]
    if only:
        # I fotogrammi da studiare, scelti da chi chiama: il righello si costruisce una
        # volta per depth, su un orientamento solo, e studiare anche gli altri tre e'
        # lavoro in piu' che non aggiunge evidenza.
        elenco = target / "frames_to_study.json"
        elenco.write_text(json.dumps(list(only), ensure_ascii=False), encoding="utf-8")
        cmd += ["--only-file", elenco.as_posix()]
    if corrections is not None and Path(corrections).is_file():
        # Le correzioni dell'operatore rientrano nel calcolo: e' cosi' che il modulo impara
        # dagli errori, e una colonna corretta su un fotogramma fa da ancora per tutta la
        # cartella (`_corr_anchor_x`).
        cmd += ["--corrections", Path(corrections).as_posix()]
    code, error = _run(cmd, target / "study_subprocess.log", timeout)

    payload: Dict = {}
    if dati.is_file():
        try:
            payload = json.loads(dati.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    frames = payload.get("frames") or []
    per_stato: Dict[str, int] = {}
    for frame in frames:
        stato = str(frame.get("status") or "?")
        per_stato[stato] = per_stato.get(stato, 0) + 1
    return {
        "status": "ok" if code == 0 and frames else "error",
        "error": error or ("" if code == 0 else f"returncode={code}"),
        "output_dir": target.as_posix(),
        "data_json": dati.as_posix() if dati.is_file() else "",
        "html": (target / "scale_study.html").as_posix(),
        "frames": len(frames),
        "by_status": per_stato,
        "vendor": payload.get("vendor", ""),
        "zone": payload.get("zone"),
        "corrected_frames": sum(1 for f in frames if f.get("corr_applied")),
    }


# --- orientamento: marker + envelope #16 ----------------------------------
def run_marker_envelopes(
    *,
    folder: Path,
    output_root: Path,
    python_bin: str,
    bundle_dir: Optional[Path] = None,
    library_root: Optional[Path] = None,
    vendor: str = "",
    exclusion_rect: Optional[Dict] = None,
    max_images: int = 0,
    timeout: float = 1200.0,
) -> Dict:
    """The marker runner works on a dataset root: point it at the parent and pin the folder.

    ``bundle_dir`` and ``library_root`` must be passed explicitly: the script's defaults are
    relative to the repo root, and in a git worktree the artifacts live in the main checkout.
    """
    if not MARKER_SCRIPT.is_file():
        return {"status": "error", "error": f"script assente: {MARKER_SCRIPT}"}
    if bundle_dir is not None and not Path(bundle_dir).exists():
        return {"status": "error", "error": f"bundle marker assente: {bundle_dir}", "rows": []}

    folder = Path(folder)
    target = stage_dir(output_root, "orientation_marker", folder)
    # Il runner scrive CSV incrementali per poter riprendere i batch con ``--resume``.
    # Dall'app, pero', «Rifai» significa una misura nuova: riusare la stessa directory
    # accodava ogni giro ai precedenti. Dopo una divisione L/T restavano perfino le righe
    # dell'altro piano, e gli envelope venivano calcolati sull'unione delle due run.
    shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin, MARKER_SCRIPT.as_posix(),
        "--dataset-root", folder.parent.as_posix(),
        "--output-dir", target.as_posix(),
        "--folder-regex", f"^{re.escape(folder.name)}$",
        "--max-folders", "1",
        "--max-images-per-folder", str(int(max_images)),
    ]
    if bundle_dir is not None:
        cmd += ["--bundle-dir", Path(bundle_dir).as_posix()]
    if library_root is not None and Path(library_root).exists():
        cmd += ["--library-root", Path(library_root).as_posix()]
    if vendor:
        # The vendor comes from the classifier, not from the folder name.
        cmd += ["--vendor", str(vendor)]
    if exclusion_rect:
        cmd += ["--exclusion-rect", rect_to_line11(exclusion_rect)]
    code, error = _run(cmd, target / "pipeline_subprocess.log", timeout)

    envelopes = _read_csv(target / "marker_envelopes.csv")
    if not envelopes:
        for candidate in sorted(target.glob("*.csv")):
            envelopes = _read_csv(candidate)
            if envelopes:
                break
    return {
        "status": "ok" if code == 0 and envelopes else "error",
        "error": error or ("" if code == 0 else f"returncode={code}"),
        "output_dir": target.as_posix(),
        "rows": envelopes,
        "files": [p.name for p in sorted(target.glob("*"))],
    }


# --- dalle righe testuali del modulo scala alle strutture del writer ------
def parse_scale_lines(lines: Dict[str, str]) -> Dict:
    """`line_18..line_21` arrivano come testo `.fss`: qui diventano il valore dello step.

    Il writer vuole liste e dizionari, non stringhe: convertendo qui, l'output del modulo
    scala entra nel documento senza passaggi manuali.
    """

    def vector(raw: str) -> List[float]:
        return [float(part) for part in str(raw or "").split("|") if part.strip()]

    scale_lines: List[Dict] = []
    for entry in str(lines.get("line_21_scale_line") or "").split(";"):
        parts = [part for part in entry.split("|") if part.strip()]
        if len(parts) < 7:
            continue
        scale_lines.append(
            {
                "x1": int(float(parts[0])), "x2": int(float(parts[1])),
                "y1": int(float(parts[2])), "y2": int(float(parts[3])),
                "length_mm": float(parts[4]), "tick_mm": float(parts[5]),
                "label_side": int(float(parts[6])),
            }
        )
    return {
        "depths": vector(lines.get("line_18_vect_depth")),
        "pixel_ratio_x": vector(lines.get("line_19_pixel_ratio_x")),
        "pixel_ratio_y": vector(lines.get("line_20_pixel_ratio_y")),
        "scale_lines": scale_lines,
    }


def parse_marker_envelopes(rows: Sequence[Dict], group_orientation: int = 4) -> Dict:
    """Envelope per gruppo -> valore dello step orientamento (#12 + #16).

    Un gruppo non trovato non e' un errore: significa che quell'orientamento **non esiste** in
    questa configurazione. Si registra fra gli orientamenti disponibili della pagina codici
    (`vbOrientationAvailable` del legacy) e, per non lasciare la riga monca, il posto vuoto
    viene riempito con la copia di un gruppo esistente -- la stessa convenzione che il formato
    documenta per le righe a quattro gruppi ("quelle non esistenti saranno riempite con una di
    quelle esistenti"), e quello che i progetti legacy a un solo orientamento hanno in #16.
    """
    order = ("NF", "LR", "UD", "LRUD")
    by_group: Dict[str, Dict] = {}
    for row in rows:
        group = str(row.get("group", "") or "").strip().upper()
        box = str(row.get("envelope_box", "") or "")
        parts = [part for part in box.split("|") if part.strip()]
        if group not in order or len(parts) < 4:
            continue
        top, left, bottom, right = (int(float(value)) for value in parts[:4])
        by_group[group] = {
            "top": top, "left": left, "bottom": bottom, "right": right,
            "check": 1, "params": {"threshold": 0.0},
            "markers": int(float(row.get("n_markers") or 0)),
        }

    found = [group for group in order if group in by_group]
    missing = [group for group in order if group not in by_group]
    value: Dict = {
        "group_orientation": int(group_orientation),
        "groups": by_group,
        "found_groups": found,
        "missing_groups": missing,
        "orientation_available": {group: group in by_group for group in order},
    }
    if not found:
        return value

    donor = found[0]
    filled: List[str] = []
    blocks = []
    for group in order:
        source = by_group.get(group)
        if source is None:
            source = by_group[donor]
            filled.append(group)
        blocks.append(
            {key: source[key] for key in ("top", "left", "bottom", "right", "check", "params")}
        )
    value["blocks"] = blocks
    value["filled_groups"] = filled
    value["filled_from"] = donor if filled else ""
    return value
