#!/usr/bin/env python3
"""Append-only feedback inbox: one JSON line per comment or correction.

The store lives in the repo (``feedback/inbox.jsonl``) so it travels with git and Claude Code
reads it without any export step. Append-only because a review session is a log, not a state:
a later "actually it was fine" is a new entry that supersedes the old one, and both are
evidence. The mutable part is ``status`` / ``resolution``, rewritten in place by ``resolve``.

Every entry carries, besides what the operator wrote:
- ``prediction``: what the module said, verbatim, at the moment of the comment;
- ``correction``: what the operator says it should be;
- ``context``: vendor, confidences, thresholds, checkpoints, sources - filled in by the
  server, never typed by hand, because a comment whose context has to be reconstructed by
  reopening the run is a comment nobody will act on;
- ``provenance``: run dir and CSV row, so any claim here can be re-derived from disk.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INBOX = REPO_ROOT / "feedback" / "inbox.jsonl"

SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _new_id() -> str:
    return f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"


class _FileLock:
    """Lock for concurrent appends (the UI can fire several saves at once).

    ``O_CREAT | O_EXCL`` on a sibling file: no dependency, works on macOS and Windows, and a
    stale lock older than ``stale_after`` is broken instead of hanging the request.
    """

    def __init__(self, target: Path, timeout: float = 5.0, stale_after: float = 30.0) -> None:
        self.path = target.with_suffix(target.suffix + ".lock")
        self.timeout = timeout
        self.stale_after = stale_after
        self._fd: Optional[int] = None

    def __enter__(self) -> "_FileLock":
        deadline = time.time() + self.timeout
        while True:
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    if age > self.stale_after:
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.time() > deadline:
                    # Losing the lock must not lose the comment: proceed unlocked. A duplicated
                    # or interleaved line is recoverable, a dropped correction is not.
                    return self
                time.sleep(0.05)

    def __exit__(self, *_exc: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self.path.unlink(missing_ok=True)


class FeedbackStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else DEFAULT_INBOX
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ read
    def all(self) -> List[Dict[str, object]]:
        if not self.path.is_file():
            return []
        out: List[Dict[str, object]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
        return out

    def query(
        self,
        *,
        status: Optional[str] = None,
        area: Optional[str] = None,
        run_id: Optional[str] = None,
        folder: Optional[str] = None,
        image_id: Optional[str] = None,
        vendor: Optional[str] = None,
        kind: Optional[str] = None,
        verdict: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        def keep(entry: Dict[str, object]) -> bool:
            target = entry.get("target") or {}
            context = entry.get("context") or {}
            if status and str(entry.get("status", "")) != status:
                return False
            if area and str(entry.get("area", "")) != area:
                return False
            if kind and str(entry.get("kind", "")) != kind:
                return False
            if verdict and str(entry.get("verdict", "")) != verdict:
                return False
            if tag and str(entry.get("tag", "")) != tag:
                return False
            if run_id and str(target.get("run_id", "")) != run_id:  # type: ignore[union-attr]
                return False
            if folder and str(target.get("folder", "")) != folder:  # type: ignore[union-attr]
                return False
            if image_id and str(target.get("image_id", "")) != image_id:  # type: ignore[union-attr]
                return False
            if vendor and str(context.get("vendor", "")) != vendor:  # type: ignore[union-attr]
                return False
            return True

        return [e for e in self.all() if keep(e)]

    def get(self, entry_id: str) -> Optional[Dict[str, object]]:
        for entry in self.all():
            if str(entry.get("id")) == entry_id:
                return entry
        return None

    # ----------------------------------------------------------------- write
    def append(
        self,
        *,
        area: str,
        kind: str,
        scope: str,
        comment: str = "",
        tag: str = "",
        verdict: str = "",
        severity: str = "major",
        target: Optional[Dict[str, object]] = None,
        prediction: Optional[Dict[str, object]] = None,
        correction: Optional[Dict[str, object]] = None,
        context: Optional[Dict[str, object]] = None,
        provenance: Optional[Dict[str, object]] = None,
        author: str = "camilla",
    ) -> Dict[str, object]:
        entry: Dict[str, object] = {
            "id": _new_id(),
            "schema": SCHEMA_VERSION,
            "ts": _now_iso(),
            "author": author,
            "kind": kind,
            "area": area,
            "scope": scope,
            "tag": tag,
            "verdict": verdict,
            "severity": severity,
            "status": "open",
            "comment": comment,
            "target": target or {},
            "prediction": prediction or {},
            "correction": correction or {},
            "context": context or {},
            "provenance": provenance or {},
            "resolution": {"by": None, "commit": None, "note": None, "ts": None},
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with _FileLock(self.path):
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return entry

    def resolve(
        self,
        entry_id: str,
        *,
        status: str = "done",
        commit: str = "",
        note: str = "",
        by: str = "claude",
    ) -> Optional[Dict[str, object]]:
        """Rewrite one entry's status in place (the only mutable part of the log)."""
        entries = self.all()
        found: Optional[Dict[str, object]] = None
        for entry in entries:
            if str(entry.get("id")) == entry_id:
                entry["status"] = status
                entry["resolution"] = {
                    "by": by,
                    "commit": commit or None,
                    "note": note or None,
                    "ts": _now_iso(),
                }
                found = entry
                break
        if found is None:
            return None
        with _FileLock(self.path):
            tmp = self.path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            tmp.replace(self.path)
        return found

    def delete(self, entry_id: str) -> bool:
        """Remove an entry (an operator's mis-click should not become permanent noise)."""
        entries = self.all()
        kept = [e for e in entries if str(e.get("id")) != entry_id]
        if len(kept) == len(entries):
            return False
        with _FileLock(self.path):
            tmp = self.path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                for entry in kept:
                    handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            tmp.replace(self.path)
        return True

    # ----------------------------------------------------------------- stats
    def stats(self) -> Dict[str, object]:
        entries = self.all()
        by_area: Dict[str, Dict[str, int]] = {}
        by_tag: Dict[str, int] = {}
        by_vendor_area: Dict[str, Dict[str, int]] = {}
        for entry in entries:
            area = str(entry.get("area", "?"))
            status = str(entry.get("status", "open"))
            verdict = str(entry.get("verdict", ""))
            slot = by_area.setdefault(area, {"total": 0, "open": 0, "wrong": 0, "ok": 0})
            slot["total"] += 1
            if status == "open":
                slot["open"] += 1
            if verdict == "wrong":
                slot["wrong"] += 1
            if verdict == "ok":
                slot["ok"] += 1
            tag = str(entry.get("tag", "")).strip()
            if tag:
                by_tag[tag] = by_tag.get(tag, 0) + 1
            vendor = str((entry.get("context") or {}).get("vendor", "")).strip()  # type: ignore[union-attr]
            if vendor:
                vslot = by_vendor_area.setdefault(vendor, {})
                vslot[area] = vslot.get(area, 0) + 1
        return {
            "total": len(entries),
            "open": sum(1 for e in entries if str(e.get("status")) == "open"),
            "by_area": by_area,
            "by_tag": dict(sorted(by_tag.items(), key=lambda kv: -kv[1])),
            "by_vendor_area": by_vendor_area,
        }


def derive_mm_per_px(correction: Dict[str, object]) -> Optional[float]:
    """mm per pixel from a scala correction, only when it is actually determined.

    Zero, far end and the value at the far end: three numbers, one answer. With anything less
    the quantity is guessed, and a guessed calibration poisons the GT it would feed.
    """
    try:
        y_zero = float(correction.get("y_zero"))  # type: ignore[arg-type]
        y_far = float(correction.get("y_far"))  # type: ignore[arg-type]
        depth_mm = float(correction.get("depth_mm"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    span = abs(y_far - y_zero)
    if span < 1.0 or depth_mm <= 0:
        return None
    return round(depth_mm / span, 6)


if __name__ == "__main__":
    store = FeedbackStore()
    print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
