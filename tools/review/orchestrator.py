#!/usr/bin/env python3
"""Per-vendor knowledge built from every review run, plus what it implies for the stages.

This is the seed of the general orchestrator. The idea it implements: a stage that only sees
its own image is weaker than one that also knows what this vendor's machines have always
looked like. So we accumulate, per vendor:

- the geometry the folders agree on (rect, ruler column, calibration range);
- which strategy each stage actually used, and how often it worked;
- where the human said the stage was wrong, by tag, so the weak spot is named, not guessed;
- which cross-area checks fail together, which is what tells us two stages should be talking.

Every number carries its support count. A prior with support 2 is a curiosity; with support 40
it is a rule, and only the caller can decide where the line is - so we never hide it.

Nothing here changes the pipeline. It produces the evidence a future orchestrator would need,
and the suggestions a human can approve one at a time.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from areas import AREA_BY_ID
from feedback_store import FeedbackStore
from runner import RUNS_ROOT, Run
from study import FolderStudy, folder_coherence

REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = REPO_ROOT / "artifacts" / "72_review_snella" / "orchestrator"


def _median(values: List[float]) -> Optional[float]:
    return round(statistics.median(values), 3) if values else None


def _spread(values: List[float]) -> Optional[float]:
    """Half of the middle 90% - a robust "how much does this wobble" for a prior."""
    if len(values) < 3:
        return None
    ordered = sorted(values)
    lo = ordered[int(0.05 * (len(ordered) - 1))]
    hi = ordered[int(0.95 * (len(ordered) - 1))]
    return round((hi - lo) / 2.0, 3)


class VendorKnowledge:
    def __init__(self) -> None:
        self.folders: List[Dict[str, object]] = []
        self.feedback: List[Dict[str, object]] = []

    # ------------------------------------------------------------------ ingest
    def add_run(self, run: Run) -> int:
        added = 0
        for job in run.jobs:
            run_dir = run.folders_dir / job.slug
            study = FolderStudy(run_dir)
            if not study.available:
                continue
            folder = study.folder_summary()
            images = study.images()
            self.folders.append({
                "run_id": run.run_id,
                "folder": folder.get("folder_name") or job.name,
                "folder_path": folder.get("folder_path") or job.path.as_posix(),
                "run_dir": run_dir.as_posix(),
                "summary": folder,
                "images_count": len(images),
                "folder_checks": folder_coherence(folder, images),
                "image_checks": [c for image in images for c in image.get("checks", [])],
                "depth_modes": Counter(str((i.get("depth") or {}).get("mode", ""))
                                       for i in images if (i.get("depth") or {}).get("mode")),
                "scala_rows": [i.get("scala") or {} for i in images if i.get("scala")],
                "rect_rows": [i.get("rect_per_image") or {} for i in images
                              if i.get("rect_per_image")],
            })
            added += 1
        return added

    def add_all_runs(self) -> int:
        total = 0
        if not RUNS_ROOT.is_dir():
            return 0
        for path in sorted(RUNS_ROOT.iterdir()):
            if not path.is_dir():
                continue
            run = Run.load(path)
            if run is not None:
                total += self.add_run(run)
        return total

    def add_feedback(self, store: Optional[FeedbackStore] = None) -> int:
        self.feedback = (store or FeedbackStore()).all()
        return len(self.feedback)

    # ------------------------------------------------------------------ build
    def build(self) -> Dict[str, object]:
        by_vendor: Dict[str, Dict[str, object]] = {}
        for entry in self.folders:
            summary = entry["summary"]  # type: ignore[index]
            vendor = str((summary.get("vendor") or {}).get("value", "") or "?")  # type: ignore[union-attr]
            slot = by_vendor.setdefault(vendor, _empty_vendor(vendor))
            _accumulate_folder(slot, entry, summary)  # type: ignore[arg-type]

        for entry in self.feedback:
            context = entry.get("context") or {}
            vendor = str(context.get("vendor", "") or "?")  # type: ignore[union-attr]
            slot = by_vendor.setdefault(vendor, _empty_vendor(vendor))
            _accumulate_feedback(slot, entry)

        vendors = {name: _finalize_vendor(slot) for name, slot in sorted(by_vendor.items())}
        return {
            "generated": _now(),
            "folders_seen": len(self.folders),
            "feedback_seen": len(self.feedback),
            "vendors": vendors,
            "cross_vendor": _cross_vendor(vendors),
            "suggestions": _suggestions(vendors),
        }

    def write(self, path: Optional[Path] = None) -> Path:
        payload = self.build()
        out = path or (KNOWLEDGE_DIR / "vendor_knowledge.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
        return out


def _now() -> str:
    from datetime import datetime

    return datetime.now().replace(microsecond=0).isoformat()


def _empty_vendor(name: str) -> Dict[str, object]:
    return {
        "vendor": name,
        "folders": [],
        "video_sizes": Counter(),
        "probe_ids": Counter(),
        "rect_by_size": defaultdict(list),
        "rect_methods": Counter(),
        "depth_modes": Counter(),
        "depth_values": [],
        "depth_acceptance": [],
        "scala_profiles": Counter(),
        "scala_ruler_x_by_size": defaultdict(list),
        "scala_mm_per_px_by_size": defaultdict(list),
        "scala_directions": Counter(),
        "scala_calib_sources": Counter(),
        "scala_acceptance": [],
        "su_giu": Counter(),
        "lt": Counter(),
        "lr_marker_sources": Counter(),
        "checks": Counter(),
        "check_pairs": Counter(),
        "feedback_tags": Counter(),
        "feedback_areas": Counter(),
        "feedback_verdicts": Counter(),
        "corrections": [],
    }


def _accumulate_folder(slot: Dict[str, object], entry: Dict[str, object],
                       summary: Dict[str, object]) -> None:
    slot["folders"].append({  # type: ignore[union-attr]
        "folder": entry.get("folder"),
        "run_id": entry.get("run_id"),
        "status": summary.get("status"),
        "images": entry.get("images_count"),
    })

    rect = summary.get("rect") or {}
    size_key = _size_key(summary)
    slot["video_sizes"][size_key] += 1  # type: ignore[index]
    if isinstance(rect.get("box"), dict):
        slot["rect_by_size"][size_key].append(rect["box"])  # type: ignore[index]
    if rect.get("method"):
        slot["rect_methods"][str(rect["method"])] += 1  # type: ignore[index]

    probe = summary.get("probe") or {}
    if probe.get("value"):
        slot["probe_ids"][str(probe["value"])] += 1  # type: ignore[index]

    depth = summary.get("depth") or {}
    for mode, count in (entry.get("depth_modes") or {}).items():  # type: ignore[union-attr]
        slot["depth_modes"][mode] += count  # type: ignore[index]
    for value in depth.get("unique_depths") or []:
        try:
            slot["depth_values"].append(float(value))  # type: ignore[union-attr]
        except (TypeError, ValueError):
            continue
    if depth.get("acceptance_ratio") is not None:
        slot["depth_acceptance"].append(float(depth["acceptance_ratio"]))  # type: ignore[union-attr]

    scala = summary.get("scala") or {}
    if scala.get("profile"):
        slot["scala_profiles"][str(scala["profile"])] += 1  # type: ignore[index]
    if scala.get("acceptance_ratio") is not None:
        slot["scala_acceptance"].append(float(scala["acceptance_ratio"]))  # type: ignore[union-attr]
    for row in entry.get("scala_rows") or []:  # type: ignore[union-attr]
        if row.get("x") is not None:
            slot["scala_ruler_x_by_size"][size_key].append(float(row["x"]))  # type: ignore[index]
        if row.get("mm_per_px"):
            slot["scala_mm_per_px_by_size"][size_key].append(float(row["mm_per_px"]))  # type: ignore[index]
        if row.get("direction"):
            slot["scala_directions"][str(row["direction"])] += 1  # type: ignore[index]
        if row.get("calib_source"):
            slot["scala_calib_sources"][str(row["calib_source"])] += 1  # type: ignore[index]

    su_giu = summary.get("su_giu") or {}
    if su_giu.get("majority"):
        slot["su_giu"][str(su_giu["majority"])] += 1  # type: ignore[index]
    lt = summary.get("lt") or {}
    if lt.get("majority"):
        slot["lt"][str(lt["majority"])] += 1  # type: ignore[index]
    lr = summary.get("lr_marker") or {}
    if lr.get("source"):
        slot["lr_marker_sources"][str(lr["source"])] += 1  # type: ignore[index]

    for check in list(entry.get("folder_checks") or []) + list(entry.get("image_checks") or []):
        if str(check.get("level")) in {"warn", "error"}:
            slot["checks"][str(check.get("id"))] += 1  # type: ignore[index]
            areas = str(check.get("areas", ""))
            if "+" in areas:
                slot["check_pairs"][areas] += 1  # type: ignore[index]


def _accumulate_feedback(slot: Dict[str, object], entry: Dict[str, object]) -> None:
    area = str(entry.get("area", "?"))
    slot["feedback_areas"][area] += 1  # type: ignore[index]
    slot["feedback_verdicts"][str(entry.get("verdict", ""))] += 1  # type: ignore[index]
    tag = str(entry.get("tag", "") or "").strip()
    if tag:
        slot["feedback_tags"][tag] += 1  # type: ignore[index]
    if entry.get("correction"):
        slot["corrections"].append({  # type: ignore[union-attr]
            "id": entry.get("id"),
            "area": area,
            "tag": tag,
            "correction": entry.get("correction"),
            "target": entry.get("target"),
        })


def _size_key(summary: Dict[str, object]) -> str:
    rect = summary.get("rect") or {}
    box = rect.get("box") if isinstance(rect, dict) else None
    if isinstance(box, dict):
        # Not the image size, but a stable bucket for folders that share a capture geometry.
        return f"{box.get('right')}x{box.get('bottom')}"
    return "?"


def _finalize_vendor(slot: Dict[str, object]) -> Dict[str, object]:
    rect_priors = {}
    for size, boxes in (slot["rect_by_size"] or {}).items():  # type: ignore[union-attr]
        if not boxes:
            continue
        rect_priors[size] = {
            "support": len(boxes),
            "top": _median([float(b["top"]) for b in boxes if b.get("top") is not None]),
            "left": _median([float(b["left"]) for b in boxes if b.get("left") is not None]),
            "bottom": _median([float(b["bottom"]) for b in boxes if b.get("bottom") is not None]),
            "right": _median([float(b["right"]) for b in boxes if b.get("right") is not None]),
        }

    ruler = {}
    for size, values in (slot["scala_ruler_x_by_size"] or {}).items():  # type: ignore[union-attr]
        if not values:
            continue
        ruler[size] = {"support": len(values), "median": _median(values),
                       "spread": _spread(values)}

    calib = {}
    for size, values in (slot["scala_mm_per_px_by_size"] or {}).items():  # type: ignore[union-attr]
        if not values:
            continue
        calib[size] = {"support": len(values), "median": _median(values),
                       "spread": _spread(values), "min": round(min(values), 4),
                       "max": round(max(values), 4)}

    return {
        "vendor": slot["vendor"],
        "folders_seen": len(slot["folders"]),  # type: ignore[arg-type]
        "folders": slot["folders"],
        "probe_ids": dict(slot["probe_ids"]),  # type: ignore[arg-type]
        "rect": {"priors_by_geometry": rect_priors, "methods": dict(slot["rect_methods"])},  # type: ignore[arg-type]
        "depth": {
            "modes": dict(slot["depth_modes"]),  # type: ignore[arg-type]
            "values_seen": sorted({round(v, 1) for v in slot["depth_values"]}),  # type: ignore[union-attr]
            "acceptance_median": _median(list(slot["depth_acceptance"])),  # type: ignore[arg-type]
        },
        "scala": {
            "profiles": dict(slot["scala_profiles"]),  # type: ignore[arg-type]
            "ruler_x_by_geometry": ruler,
            "mm_per_px_by_geometry": calib,
            "directions": dict(slot["scala_directions"]),  # type: ignore[arg-type]
            "calib_sources": dict(slot["scala_calib_sources"]),  # type: ignore[arg-type]
            "acceptance_median": _median(list(slot["scala_acceptance"])),  # type: ignore[arg-type]
        },
        "orientation": {
            "su_giu": dict(slot["su_giu"]),  # type: ignore[arg-type]
            "lt": dict(slot["lt"]),  # type: ignore[arg-type]
            "lr_marker_sources": dict(slot["lr_marker_sources"]),  # type: ignore[arg-type]
        },
        "coherence": {
            "checks": dict(slot["checks"].most_common()),  # type: ignore[union-attr]
            "area_pairs": dict(slot["check_pairs"].most_common()),  # type: ignore[union-attr]
        },
        "human": {
            "by_area": dict(slot["feedback_areas"]),  # type: ignore[arg-type]
            "by_tag": dict(slot["feedback_tags"].most_common()),  # type: ignore[union-attr]
            "verdicts": dict(slot["feedback_verdicts"]),  # type: ignore[arg-type]
            "corrections": slot["corrections"],
        },
    }


def _cross_vendor(vendors: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """What is a vendor problem and what is a module problem.

    A tag that appears for one vendor points at a profile; the same tag across most vendors
    points at the module. Telling those two apart is most of the triage work.
    """
    tag_vendors: Dict[str, List[str]] = defaultdict(list)
    check_vendors: Dict[str, List[str]] = defaultdict(list)
    for name, data in vendors.items():
        for tag in (data.get("human") or {}).get("by_tag", {}):  # type: ignore[union-attr]
            tag_vendors[tag].append(name)
        for check in (data.get("coherence") or {}).get("checks", {}):  # type: ignore[union-attr]
            check_vendors[check].append(name)
    total = max(1, len(vendors))
    return {
        "module_wide_tags": {t: v for t, v in tag_vendors.items() if len(v) >= max(2, total // 2)},
        "vendor_specific_tags": {t: v for t, v in tag_vendors.items() if len(v) == 1},
        "module_wide_checks": {c: v for c, v in check_vendors.items()
                               if len(v) >= max(2, total // 2)},
        "vendor_specific_checks": {c: v for c, v in check_vendors.items() if len(v) == 1},
    }


def _suggestions(vendors: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    """Concrete, checkable proposals - each with the evidence that produced it."""
    out: List[Dict[str, object]] = []

    def add(kind: str, vendor: str, area: str, text: str, support: int,
            evidence: object = None) -> None:
        out.append({"kind": kind, "vendor": vendor, "area": area, "text": text,
                    "support": support, "evidence": evidence})

    for name, data in vendors.items():
        if name == "?":
            continue
        scala = data.get("scala") or {}
        for geometry, prior in (scala.get("ruler_x_by_geometry") or {}).items():  # type: ignore[union-attr]
            if prior.get("support", 0) >= 8 and prior.get("spread") is not None:
                if float(prior["spread"]) <= 25.0:
                    add("prior", name, "scala",
                        f"Righello sempre attorno a x={prior['median']:.0f} "
                        f"(±{prior['spread']:.0f} px) sulla geometria {geometry}: usare come "
                        f"banda invece del fallback a tutto schermo",
                        int(prior["support"]), prior)

        for geometry, prior in (scala.get("mm_per_px_by_geometry") or {}).items():  # type: ignore[union-attr]
            if prior.get("support", 0) >= 8 and prior.get("median"):
                add("prior", name, "scala",
                    f"mm_per_px atteso {prior['median']} su {geometry} "
                    f"(min {prior['min']}, max {prior['max']}): una calibrazione fuori da "
                    f"questo intervallo è evidenza debole",
                    int(prior["support"]), prior)

        depth = data.get("depth") or {}
        modes = depth.get("modes") or {}
        if modes:
            total = sum(modes.values())  # type: ignore[union-attr]
            top_mode, top_count = max(modes.items(), key=lambda kv: kv[1])  # type: ignore[union-attr]
            if total >= 20 and top_count / total >= 0.90:
                add("prior", name, "depth",
                    f"La depth arriva dalla modalità '{top_mode}' nel "
                    f"{top_count / total * 100:.0f}% dei frame: le altre modalità richiedono "
                    f"evidenza più forte su questo vendor", int(total), modes)

        rect = data.get("rect") or {}
        methods = rect.get("methods") or {}
        if methods.get("segment_top_red_rect") and methods.get("median_rect_fallback"):
            add("attenzione", name, "rect",
                "Su questo vendor il #11 arriva sia dal rosso sia dalla mediana: "
                "controllare se il gate dell'override è troppo permissivo",
                int(sum(methods.values())), methods)  # type: ignore[union-attr]

        human = data.get("human") or {}
        for tag, count in (human.get("by_tag") or {}).items():  # type: ignore[union-attr]
            if count >= 3:
                area_id = tag.split(".", 1)[0]
                module = str((AREA_BY_ID.get(area_id) or {}).get("module", ""))
                add("lavoro", name, area_id,
                    f"{count} segnalazioni '{tag}': è il punto debole di questo vendor"
                    + (f" → {module}" if module else ""), int(count))

        coherence = (data.get("coherence") or {}).get("area_pairs") or {}
        for pair, count in coherence.items():  # type: ignore[union-attr]
            if count >= 5:
                add("collaborazione", name, pair,
                    f"{count} immagini in cui {pair.replace('+', ' e ')} non si accordano: "
                    f"le due aree dovrebbero scambiarsi l'evidenza invece di decidere da sole",
                    int(count))

    out.sort(key=lambda item: -int(item.get("support", 0)))
    return out


def build_knowledge() -> Dict[str, object]:
    knowledge = VendorKnowledge()
    knowledge.add_all_runs()
    knowledge.add_feedback()
    return knowledge.build()


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    knowledge = VendorKnowledge()
    folders = knowledge.add_all_runs()
    entries = knowledge.add_feedback()
    path = knowledge.write()
    print(f"cartelle lette: {folders}, feedback letti: {entries}")
    print(f"scritto: {path}")
