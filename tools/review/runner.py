#!/usr/bin/env python3
"""Run registry and folder-by-folder orchestration.

One subprocess per folder, on purpose. The official pipeline writes its CSVs only when the
*last* folder is done, so a batch run shows nothing until the end; one folder per process
means the review page for folder 1 is ready while folder 2 is still going. The cost is
reloading the models per folder, which on a review-sized selection is worth the interactivity.

Progress comes from two channels on the child's stdout:
- ``##EVENT`` lines from ``run_pipeline_single_folder_safe.py`` (its own lifecycle);
- ``##STAGE`` lines from the pipeline's ``--stage-events`` (one per stage per folder).

Both are appended to ``events.jsonl`` in the run dir, so a reloaded page replays the run
instead of losing it.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "artifacts" / "72_review_snella" / "runs"
SAFE_RUNNER = REPO_ROOT / "tools" / "ultrasound" / "run_pipeline_single_folder_safe.py"
VENV_PYTHON = REPO_ROOT / "OldSoftwareEsiBuilder" / ".venv-mps" / "bin" / "python"

# The stages the UI draws as a row of lights, in pipeline order.
STAGE_ORDER = ["dedup", "rotazione", "vendor", "probe", "rect", "orientamento", "depth",
               "scala", "folder_done"]

STAGE_LABELS = {
    "dedup": "Deduplicazione",
    "rotazione": "Rotazione",
    "vendor": "Vendor",
    "probe": "Sonda",
    "rect": "Rettangolo",
    "orientamento": "Orientamento",
    "depth": "Depth",
    "scala": "Scala",
    "folder_done": "Testa .fss",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(value: str) -> str:
    out = _SLUG_RE.sub("_", str(value).strip()).strip("_")
    return out or "folder"


def _now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


_PYTHON_OVERRIDE: Optional[str] = None


def set_python_bin(path: Optional[str]) -> None:
    global _PYTHON_OVERRIDE
    _PYTHON_OVERRIDE = str(path) if path else None


def _main_repo_root() -> Optional[Path]:
    """The main checkout, when we are running inside a git worktree.

    The venv with torch lives there, not in the worktree, so a tool started from a worktree
    must be able to reach it or every run dies on ``import torch``.
    """
    git_file = REPO_ROOT / ".git"
    if not git_file.is_file():
        return None
    try:
        text = git_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    git_dir = Path(text.split(":", 1)[1].strip())
    # <main>/.git/worktrees/<name> -> <main>
    for parent in git_dir.parents:
        if parent.name == ".git":
            return parent.parent
    return None


def python_bin() -> str:
    if _PYTHON_OVERRIDE:
        return _PYTHON_OVERRIDE
    candidates = [VENV_PYTHON]
    main_root = _main_repo_root()
    if main_root is not None:
        candidates.append(main_root / "OldSoftwareEsiBuilder" / ".venv-mps" / "bin" / "python")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.as_posix()
    return "python3"


def list_candidate_folders(root: Path, limit: int = 4000) -> List[Dict[str, object]]:
    """Immediate subdirectories of a dataset root, with a cheap image count.

    Cheap on purpose: counting images of every folder on an external volume would make the
    first screen unusable. It stops at 200 files per folder and says "200+".
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        return []
    out: List[Dict[str, object]] = []
    for path in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_dir() or path.name.startswith((".", "$", "_")):
            continue
        if path.name == "System Volume Information":
            continue
        count = 0
        capped = False
        try:
            for child in path.rglob("*"):
                if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
                    count += 1
                    if count >= 200:
                        capped = True
                        break
        except OSError:
            pass
        out.append({
            "name": path.name,
            "path": path.as_posix(),
            "images": count,
            "images_capped": capped,
        })
        if len(out) >= limit:
            break
    return out


_roots_cache: Dict[str, object] = {"ts": 0.0, "roots": []}


def known_dataset_roots(max_age: float = 20.0) -> List[Path]:
    """Dataset roots of every run on disk, read from ``run.json`` only.

    Cached and manifest-only on purpose: this is called once per served image, and loading a
    full ``Run`` (events included) per thumbnail would make the grid crawl.
    """
    import time as _time

    now = _time.time()
    if now - float(_roots_cache["ts"]) < max_age:  # type: ignore[arg-type]
        return list(_roots_cache["roots"])  # type: ignore[arg-type]
    roots: List[Path] = []
    if RUNS_ROOT.is_dir():
        for path in sorted(RUNS_ROOT.iterdir()):
            manifest = path / "run.json"
            if not manifest.is_file():
                continue
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            root = str(payload.get("dataset_root", "") or "")
            if root:
                roots.append(Path(root))
    _roots_cache["ts"] = now
    _roots_cache["roots"] = roots
    return list(roots)


class FolderJob:
    def __init__(self, name: str, path: Path, run_dir: Path) -> None:
        self.name = name
        self.path = path
        self.slug = _slug(name)
        self.run_dir = run_dir
        self.state = "pending"          # pending | running | done | failed | cancelled
        self.error = ""
        self.started: Optional[str] = None
        self.finished: Optional[str] = None
        self.stages: Dict[str, Dict[str, object]] = {}
        self.sample_image: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "path": self.path.as_posix(),
            "slug": self.slug,
            "run_dir": self.run_dir.as_posix(),
            "state": self.state,
            "error": self.error,
            "started": self.started,
            "finished": self.finished,
            "stages": self.stages,
            "sample_image": self.sample_image,
        }


class Run:
    def __init__(self, run_id: str, dataset_root: Path, options: Dict[str, object]) -> None:
        self.run_id = run_id
        self.dataset_root = dataset_root
        self.options = options
        self.dir = RUNS_ROOT / run_id
        self.folders_dir = self.dir / "folders"
        self.events_path = self.dir / "events.jsonl"
        self.jobs: List[FolderJob] = []
        self.state = "pending"          # pending | running | done | cancelled | failed
        self.created = _now()
        self.finished: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._events: List[Dict[str, object]] = []

    # ---------------------------------------------------------------- persistence
    def manifest(self) -> Dict[str, object]:
        return {
            "run_id": self.run_id,
            "created": self.created,
            "finished": self.finished,
            "state": self.state,
            "dataset_root": self.dataset_root.as_posix(),
            "options": self.options,
            "folders": [job.to_dict() for job in self.jobs],
        }

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "run.json").write_text(
            json.dumps(self.manifest(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, run_dir: Path) -> Optional["Run"]:
        manifest_path = Path(run_dir) / "run.json"
        if not manifest_path.is_file():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        run = cls(
            str(payload.get("run_id", Path(run_dir).name)),
            Path(str(payload.get("dataset_root", ""))),
            dict(payload.get("options") or {}),
        )
        run.created = str(payload.get("created", ""))
        run.finished = payload.get("finished")
        run.state = str(payload.get("state", "done"))
        # A run interrupted by a server restart is not running any more, whatever it claimed.
        if run.state == "running":
            run.state = "interrupted"
        for item in payload.get("folders") or []:
            job = FolderJob(str(item.get("name", "")), Path(str(item.get("path", ""))), run.folders_dir)
            job.slug = str(item.get("slug", job.slug))
            job.state = str(item.get("state", "pending"))
            if job.state == "running":
                job.state = "interrupted"
            job.error = str(item.get("error", ""))
            job.started = item.get("started")
            job.finished = item.get("finished")
            job.stages = dict(item.get("stages") or {})
            job.sample_image = str(item.get("sample_image", ""))
            run.jobs.append(job)
        if run.events_path.is_file():
            with run.events_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        run._events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return run

    # ---------------------------------------------------------------- events
    def _record(self, event: Dict[str, object]) -> None:
        with self._lock:
            event = dict(event)
            event.setdefault("seq", len(self._events))
            event["seq"] = len(self._events)
            self._events.append(event)
            self.dir.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def events_since(self, since: int) -> List[Dict[str, object]]:
        with self._lock:
            return [e for e in self._events if int(e.get("seq", 0)) >= since]

    @property
    def event_count(self) -> int:
        with self._lock:
            return len(self._events)

    # ---------------------------------------------------------------- lifecycle
    def cancel(self) -> None:
        self._cancel.set()
        proc = self._proc
        if proc and proc.poll() is None:
            proc.terminate()

    def start(self, folders: List[Dict[str, str]]) -> None:
        self.folders_dir.mkdir(parents=True, exist_ok=True)
        for item in folders:
            self.jobs.append(FolderJob(item["name"], Path(item["path"]), self.folders_dir))
        self.state = "running"
        self.save()
        thread = threading.Thread(target=self._worker, name=f"review-run-{self.run_id}",
                                  daemon=True)
        thread.start()

    def _worker(self) -> None:
        try:
            for job in self.jobs:
                if self._cancel.is_set():
                    job.state = "cancelled"
                    continue
                self._run_folder(job)
                self.save()
        finally:
            if self._cancel.is_set():
                self.state = "cancelled"
            elif any(j.state == "failed" for j in self.jobs):
                self.state = "failed"
            else:
                self.state = "done"
            self.finished = _now()
            self.save()
            self._record({"type": "run_state", "state": self.state})

    def _command(self, job: FolderJob) -> List[str]:
        opts = self.options
        cmd = [
            python_bin(),
            SAFE_RUNNER.as_posix(),
            "--input-folder", job.path.as_posix(),
            "--work-root", self.folders_dir.as_posix(),
            "--run-name", job.slug,
            "--stage-events",
            "--sample-per-folder", str(int(opts.get("sample_per_folder", 40) or 40)),
            "--rect-depth-max-images", str(int(opts.get("depth_max_images", 0) or 0)),
            "--scale-max-frames", str(int(opts.get("scale_max_frames", 48) or 48)),
            "--low-confidence-policy", "review",
        ]
        if not bool(opts.get("previews", True)):
            cmd.append("--no-generated-images")
        if bool(opts.get("disable_depth", False)):
            cmd.append("--disable-rect-depth-autonomous")
        if bool(opts.get("disable_scale", False)):
            cmd.append("--disable-scale-stage")
        if bool(opts.get("disable_lt", False)):
            cmd.append("--disable-lt-rect-classifier")
        return cmd

    def _run_folder(self, job: FolderJob) -> None:
        job.state = "running"
        job.started = _now()
        self._record({"type": "folder_started", "folder": job.name, "slug": job.slug})
        self.save()

        cmd = self._command(job)
        log_dir = self.dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{job.slug}.log"

        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                cwd=REPO_ROOT.as_posix(),
            )
        except OSError as exc:
            job.state = "failed"
            job.error = f"impossibile avviare la pipeline: {exc}"
            job.finished = _now()
            self._record({"type": "folder_failed", "folder": job.name, "error": job.error})
            return

        assert self._proc.stdout is not None
        with log_path.open("w", encoding="utf-8") as log:
            for line in self._proc.stdout:
                log.write(line)
                # Flushed per line: this log is what you read when a run looks stuck, and a
                # buffered file would only tell you about it once the run is over.
                log.flush()
                self._handle_line(job, line.rstrip("\n"))
        self._proc.wait()
        code = self._proc.returncode
        self._proc = None

        if self._cancel.is_set():
            job.state = "cancelled"
        elif code != 0:
            job.state = "failed"
            job.error = f"la pipeline è uscita con codice {code} (log: {log_path.name})"
            self._record({"type": "folder_failed", "folder": job.name, "error": job.error})
        else:
            job.state = "done"
            self._record({"type": "folder_done", "folder": job.name, "slug": job.slug})
        job.finished = _now()

    def _handle_line(self, job: FolderJob, line: str) -> None:
        if line.startswith("##STAGE "):
            try:
                payload = json.loads(line[len("##STAGE "):])
            except json.JSONDecodeError:
                return
            stage = str(payload.get("stage", ""))
            if stage:
                job.stages[stage] = payload
                self._record({"type": "stage", "folder": job.name, "slug": job.slug,
                              "stage": stage, "payload": payload})
            return
        if line.startswith("##EVENT "):
            try:
                payload = json.loads(line[len("##EVENT "):])
            except json.JSONDecodeError:
                return
            self._record({"type": "runner_event", "folder": job.name, "slug": job.slug,
                          "payload": payload})
            return
        text = line.strip()
        if text:
            self._record({"type": "log", "folder": job.name, "line": text[:400]})


class RunManager:
    """In-process registry. One run at a time: the models want the whole device."""

    def __init__(self) -> None:
        self._runs: Dict[str, Run] = {}
        self._lock = threading.Lock()
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    def new_run_id(self) -> str:
        return datetime.now().strftime("run_%Y%m%d_%H%M%S")

    def active(self) -> Optional[Run]:
        for run in self._runs.values():
            if run.state == "running":
                return run
        return None

    def start(self, dataset_root: Path, folders: List[Dict[str, str]],
              options: Dict[str, object]) -> Run:
        with self._lock:
            if self.active() is not None:
                raise RuntimeError("C'è già una run in corso: attendi o annullala.")
            run = Run(self.new_run_id(), Path(dataset_root), options)
            self._runs[run.run_id] = run
        run.start(folders)
        return run

    def get(self, run_id: str) -> Optional[Run]:
        run = self._runs.get(run_id)
        if run is not None:
            return run
        loaded = Run.load(RUNS_ROOT / run_id)
        if loaded is not None:
            self._runs[loaded.run_id] = loaded
        return loaded

    def history(self, limit: int = 60) -> List[Dict[str, object]]:
        """Every run on disk, newest first, with a one-glance outcome per run."""
        out: List[Dict[str, object]] = []
        if not RUNS_ROOT.is_dir():
            return out
        for path in sorted(RUNS_ROOT.iterdir(), reverse=True):
            if not path.is_dir():
                continue
            run = self._runs.get(path.name) or Run.load(path)
            if run is None:
                continue
            jobs = run.jobs
            out.append({
                "run_id": run.run_id,
                "created": run.created,
                "finished": run.finished,
                "state": run.state,
                "dataset_root": run.dataset_root.as_posix(),
                "folders_total": len(jobs),
                "folders_done": sum(1 for j in jobs if j.state == "done"),
                "folders_failed": sum(1 for j in jobs if j.state == "failed"),
                "folder_names": [j.name for j in jobs],
                "options": run.options,
            })
            if len(out) >= limit:
                break
        return out
