"""Project model: project.json, step state machine and dependency-driven invalidation.

The step list and the dependency graph follow docs/specifiche_app_esibuilder_ai_2026-08-26.md.
Framework free on purpose: the Flask layer in server.py only calls into this module.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from constants import BIPLANE_PROBE_TYPES, ID_CHECK_ORIENTATION, VIDEO_HDMI
from fss_writer import FssDocument, ScaleLine, TemplateBlock

SCHEMA_VERSION = 1

# Step status values.
EMPTY = "empty"
PROPOSED = "proposed"
CONFIRMED = "confirmed"
CORRECTED = "corrected"
MANUAL = "manual"
BLOCKED = "blocked"
STALE = "stale"

DONE_STATES = (CONFIRMED, CORRECTED, MANUAL)


@dataclass(frozen=True)
class Step:
    id: str
    label: str
    legacy_page: str
    fss_lines: tuple[int, ...]
    depends_on: tuple[str, ...]
    optional_when: Optional[str] = None  # "single_probe" -> only relevant for biplane probes


# Step 0 is the import: dedup, rotation and the recognition stages run there, before the
# user sees anything, so the codes page opens already filled in.
STEPS: tuple[Step, ...] = (
    Step("import", "Import e analisi", "-", (6, 7, 8, 9, 10), ()),
    Step("codes", "Progetto e codici", "WdgCreateProject", (2, 3, 4, 5, 25, 26), ("import",)),
    # Tre cose distinte, tre schermate: il vendor col suo template (#13), la sonda col suo
    # (#14), e il rettangolo ecografico (#11). Stavano insieme perche' il legacy aveva una
    # pagina sola (PAGE_US), ma si confermano con criteri diversi e il rettangolo e' quello
    # da cui dipende tutta la geometria a valle.
    Step("vendor", "Ecografo e vendor", "PAGE_US", (13,), ("import",)),
    Step("probe", "Sonda", "PAGE_US", (14,), ("import",)),
    Step("rect", "Rettangolo ecografico", "PAGE_US", (11,), ("import",)),
    Step("biplana", "Biplana", "PAGE_BIPLANA", (24, 25, 26), ("codes", "probe"),
         optional_when="single_probe"),
    Step("proibited", "Schermate proibite", "PAGE_PROIBITED", (15,), ("rect",)),
    Step("orientation", "Orientamento", "PAGE_ORIENTATION", (12, 16), ("rect",)),
    Step("depth_scale", "Depth e scala", "PAGE_DEPTH_VALUE", (18, 19, 20, 21),
         ("rect", "orientation")),
    # Lo studio del righello ha una schermata sua, dopo la depth: si guarda un fotogramma
    # per volta — colonna, zero, tacche, numeri — e si corregge. Non possiede righe del
    # `.fss`: le sue correzioni tornano nel modulo, ed e' lo stadio della scala a
    # consolidare #19-#21 dentro `depth_scale`.
    Step("scale_study", "Scala: righello e tacche", "PAGE_DEPTH_VALUE", (), ("depth_scale",)),
    Step("depth_find", "Ricerca depth", "PAGE_DEPTH_FIND", (17,), ("depth_scale",)),
    Step("thresholds", "Soglie", "PAGE_FSS_THRESHOLDS", (),
         ("vendor", "probe", "rect", "proibited", "orientation", "depth_find")),
    Step("guides", "Linee guida", "PAGE_CALIBRATION", (22, 23), ("codes", "depth_scale")),
    Step("generate", "Generazione e verifica", "-", (), ("thresholds", "guides")),
)

STEPS_BY_ID = {step.id: step for step in STEPS}

# Which steps a code field really invalidates. Editing an ID changes nothing geometric;
# changing the sample size invalidates every coordinate in the project (spec, grafo delle
# dipendenze). Keys missing from this map invalidate nothing.
GEOMETRY_STEPS = tuple(
    step.id for step in STEPS if step.fss_lines and step.id not in ("codes", "import")
)
CODE_IMPACT: Dict[str, tuple[str, ...]] = {
    "probe_type": ("biplana", "guides"),
    "echo_model": (),
    "probe_model": (),
    "group_orientation": ("orientation",),
    "orientation_available": ("orientation",),
    "biplana_test_method": ("biplana",),
    "id_fss_next_probe": ("biplana",),
    "image_sample_size": GEOMETRY_STEPS,
    "video_input_size": GEOMETRY_STEPS,
}


# Cosa apparteneva a chi, quando "Ecografo e sonda" era una schermata sola.
ECHO_PROBE_SPLIT: Dict[str, tuple[str, ...]] = {
    "vendor": ("rect_name_echo", "rect_name_echo_source", "rect_name_echo_agreement",
               "rect_name_echo_reason", "preview_image"),
    "probe": ("rect_name_probe", "preview_image"),
    "rect": ("rect_echo", "margin_percent", "rect_model", "rect_source",
             "rect_agreement_iou", "preview_image"),
}


def _split_echo_probe(data: Dict) -> bool:
    """Progetti scritti quando i tre step erano uno solo: il valore si divide in tre.

    Si fa al caricamento e una volta sola: dopo, `echo_probe` non esiste piu'. I campi che
    non appartengono a nessuno dei tre (non ce ne sono oggi, ma un domani) restano nel
    passo del rettangolo, che e' quello che porta la geometria.
    """
    steps = data.get("steps") or {}
    vecchio = steps.pop("echo_probe", None)
    if not vecchio:
        return False
    valore = vecchio.get("value") or {}
    noti = {chiave for chiavi in ECHO_PROBE_SPLIT.values() for chiave in chiavi}
    for step_id, chiavi in ECHO_PROBE_SPLIT.items():
        pezzo = {k: valore[k] for k in chiavi if k in valore}
        if step_id == "rect":
            pezzo.update({k: v for k, v in valore.items() if k not in noti})
        steps[step_id] = {
            **{k: v for k, v in vecchio.items() if k not in ("value", "depends_on")},
            "value": pezzo,
            "depends_on": ["import"],
            # senza dati proprii lo step torna vuoto: non si eredita uno stato confermato
            "status": vecchio.get("status", "empty") if pezzo else "empty",
        }
    data["steps"] = steps
    return True


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return slug or "progetto"


def default_codes() -> Dict:
    return {
        "project_name": "",
        "fss_version": "",  # open point: which version ESI accepts
        "echo_model": "",  # key into the ECO/FSS sheets of encoding_struct
        "probe_model": "",  # key into the PROBE/FSS sheets
        "id_setup": 0,
        "id_echo": 0,
        "id_probe": 0,
        "id_ndg": 0,
        "probe_type": 0,
        "video_type": VIDEO_HDMI,
        "video_input_size": [0, 0],
        "image_sample_size": [0, 0],
        "group_orientation": ID_CHECK_ORIENTATION,
        "orientation_available": {"NF": True, "LR": True, "UD": True, "LRUD": True},
        "biplana_test_method": None,
        "id_fss_next_probe": None,
    }


def default_source() -> Dict:
    return {
        "folder": "",
        "images_total_raw": 0,
        "images_total": 0,
        "duplicates_removed": 0,
        "rotation_applied": 0,
        "rotation_source": "not_run",
        # Eccezioni manuali: una cartella puo' avere pochi frame arrivati gia' ruotati.
        # La chiave e' il nome relativo del frame, il valore l'angolo orario da applicare.
        "rotation_overrides": {},
        "resize_factor": 1.0,
        "native_size": [0, 0],
    }


def expand_rect(
    rect: Dict, margin_percent: Optional[Dict], width: int = 0, height: int = 0
) -> Dict:
    """The rectangle plus the margin the user asked for, clamped to the image.

    The margin is a percentage of the rectangle's own side, added on both sides: at 0 the
    result is the echo rectangle itself, which is what gets written to #11.
    """
    top, left = int(rect["top"]), int(rect["left"])
    bottom, right = int(rect["bottom"]), int(rect["right"])
    margin = margin_percent or {}
    dx = int(round((float(margin.get("x") or 0) / 100.0) * (right - left)))
    dy = int(round((float(margin.get("y") or 0) / 100.0) * (bottom - top)))
    out = {
        "top": max(0, top - dy),
        "left": max(0, left - dx),
        "bottom": bottom + dy,
        "right": right + dx,
    }
    if height:
        out["bottom"] = min(int(height) - 1, out["bottom"])
    if width:
        out["right"] = min(int(width) - 1, out["right"])
    return out


def _empty_step(step: Step) -> Dict:
    return {
        "status": EMPTY,
        "source": None,
        "value": {},
        "confidence": None,
        "user_edited": False,
        "depends_on": list(step.depends_on),
        "ts": None,
    }


class Project:
    """A configuration project on disk: project.json plus the generated files."""

    def __init__(self, root: Path, data: Dict):
        self.root = Path(root)
        self.data = data

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def create(cls, projects_root: Path, name: str) -> "Project":
        projects_root = Path(projects_root)
        base = _slug(name)
        root = projects_root / base
        suffix = 2
        while root.exists():
            root = projects_root / f"{base}_{suffix}"
            suffix += 1
        codes = default_codes()
        codes["project_name"] = name
        data = {
            "schema_version": SCHEMA_VERSION,
            "project_id": root.name,
            "created_at": _now(),
            "updated_at": _now(),
            "codes": codes,
            "source": default_source(),
            "steps": {step.id: _empty_step(step) for step in STEPS},
            "runs": [],
        }
        project = cls(root, data)
        root.mkdir(parents=True, exist_ok=True)
        project.save()
        return project

    @classmethod
    def load(cls, root: Path) -> "Project":
        root = Path(root)
        data = json.loads((root / "project.json").read_text(encoding="utf-8"))
        # Tolerate projects written before a step was added to the registry.
        data.setdefault("steps", {})
        migrato = _split_echo_probe(data)
        for step in STEPS:
            data["steps"].setdefault(step.id, _empty_step(step))
        project = cls(root, data)
        if migrato:
            # si scrive subito: la divisione e' definitiva, non va rifatta a ogni lettura
            project.save()
        return project

    def save(self) -> None:
        self.data["updated_at"] = _now()
        path = self.root / "project.json"
        path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    # -- accessors ---------------------------------------------------------
    @property
    def codes(self) -> Dict:
        return self.data["codes"]

    @property
    def source(self) -> Dict:
        return self.data["source"]

    @property
    def steps(self) -> Dict:
        return self.data["steps"]

    @property
    def is_biplane(self) -> bool:
        return int(self.codes.get("probe_type") or 0) in BIPLANE_PROBE_TYPES

    def step_is_relevant(self, step: Step) -> bool:
        if step.optional_when == "single_probe":
            return self.is_biplane
        return True

    # -- state machine -----------------------------------------------------
    def dependents_of(self, step_id: str) -> List[str]:
        """Every step that transitively depends on ``step_id``."""
        found: List[str] = []
        frontier = [step_id]
        while frontier:
            current = frontier.pop()
            for step in STEPS:
                if current in step.depends_on and step.id not in found:
                    found.append(step.id)
                    frontier.append(step.id)
        return found

    def set_step(
        self,
        step_id: str,
        value: Dict,
        status: str = CONFIRMED,
        source: str = "user",
        confidence: Optional[float] = None,
    ) -> List[str]:
        """Store a step value and mark its dependents stale. Returns the stale step ids."""
        if step_id not in STEPS_BY_ID:
            raise KeyError(step_id)
        entry = self.steps[step_id]
        entry.update(
            {
                "status": status,
                "source": source,
                "value": value,
                "confidence": confidence,
                "user_edited": source == "user",
                "ts": _now(),
            }
        )
        stale = self._mark_stale(self.dependents_of(step_id))
        self.save()
        return stale

    def _mark_stale(self, step_ids: Iterable[str]) -> List[str]:
        stale: List[str] = []
        for step_id in step_ids:
            entry = self.steps[step_id]
            if entry["status"] in DONE_STATES:
                entry["status"] = STALE
                stale.append(step_id)
        return stale

    def update_codes(self, changes: Dict) -> List[str]:
        """Codes stay editable at any time (principio P4): only what depends on them goes stale."""
        touched = {key for key, value in changes.items() if self.codes.get(key) != value}
        self.codes.update(changes)

        affected: List[str] = []
        for key in touched:
            affected.extend(CODE_IMPACT.get(key, ()))
        stale = self._mark_stale(sorted(set(affected)))
        self.steps["codes"].update(
            {"status": CONFIRMED, "source": "user", "user_edited": True, "ts": _now()}
        )
        self.save()
        return sorted(set(stale))

    # -- reporting ---------------------------------------------------------
    def status_report(self) -> List[Dict]:
        report = []
        for step in STEPS:
            entry = self.steps[step.id]
            report.append(
                {
                    "id": step.id,
                    "label": step.label,
                    "legacy_page": step.legacy_page,
                    "fss_lines": list(step.fss_lines),
                    "depends_on": list(step.depends_on),
                    "status": entry["status"],
                    "source": entry.get("source"),
                    "relevant": self.step_is_relevant(step),
                    "has_value": bool(entry.get("value")),
                }
            )
        return report

    # -- .fss assembly -----------------------------------------------------
    def step_value(self, step_id: str) -> Dict:
        return (self.steps.get(step_id) or {}).get("value") or {}

    def build_document(self) -> FssDocument:
        codes = self.codes
        imported = self.step_value("import")
        vendor_step = self.step_value("vendor")
        probe_step = self.step_value("probe")
        rect_step = self.step_value("rect")
        proibited = self.step_value("proibited")
        orientation = self.step_value("orientation")
        depth_scale = self.step_value("depth_scale")
        depth_find = self.step_value("depth_find")
        guides = self.step_value("guides")
        biplana = self.step_value("biplana")

        def blocks(raw) -> List[TemplateBlock]:
            return [TemplateBlock.from_dict(item) for item in (raw or [])]

        def block_groups(raw) -> List[List[TemplateBlock]]:
            return [blocks(group) for group in (raw or [])]

        doc = FssDocument(
            version=str(codes.get("fss_version") or ""),
            id_echo=int(codes.get("id_echo") or 0),
            id_probe=int(codes.get("id_probe") or 0),
            probe_type=int(codes.get("probe_type") or 0),
            kit_needle_guide=int(codes.get("id_ndg") or 0),
            video_input=int(imported.get("video_input", codes.get("video_type") or 0)),
            video_input_size=tuple(
                imported.get("video_input_size") or codes.get("video_input_size") or (0, 0)
            ),
            video_size=tuple(
                imported.get("image_sample_size") or codes.get("image_sample_size") or (0, 0)
            ),
            group_orientation=int(
                orientation.get("group_orientation", codes.get("group_orientation") or 0)
            ),
            vect_depth=[float(v) for v in depth_scale.get("depths") or []],
            pixel_ratio_x=[float(v) for v in depth_scale.get("pixel_ratio_x") or []],
            pixel_ratio_y=[float(v) for v in depth_scale.get("pixel_ratio_y") or []],
            scale_lines=[ScaleLine.from_dict(s) for s in depth_scale.get("scale_lines") or []],
            centre_distance=[[float(v) for v in row] for row in guides.get("centre_distance") or []],
            angles=[float(v) for v in guides.get("angles") or []],
            id_next_probe=(
                biplana.get("id_next_probe")
                if biplana.get("id_next_probe") is not None
                else codes.get("id_fss_next_probe")
            ),
            biplana_recognition_mode=(
                biplana.get("recognition_mode")
                if biplana.get("recognition_mode") is not None
                else codes.get("biplana_test_method")
            ),
        )

        rect = rect_step.get("rect_echo")
        if rect:
            # #11 carries the rectangle *plus* the margin: at 0 the two coincide.
            final = expand_rect(
                rect,
                rect_step.get("margin_percent"),
                width=doc.video_size[0],
                height=doc.video_size[1],
            )
            doc.rect_echo = (
                final["top"],
                final["left"],
                final["bottom"],
                final["right"],
            )
        if vendor_step.get("rect_name_echo"):
            doc.rect_name_echo = TemplateBlock.from_dict(vendor_step["rect_name_echo"])
        if probe_step.get("rect_name_probe"):
            doc.rect_name_probe = TemplateBlock.from_dict(probe_step["rect_name_probe"])
        doc.proibited_screen = block_groups(proibited.get("groups"))
        doc.rect_orientation = blocks(orientation.get("blocks"))
        doc.rect_depth = block_groups(depth_find.get("groups"))
        doc.rect_trans = blocks(biplana.get("rect_trans"))
        return doc

    # -- immagini deduplicate ---------------------------------------------
    DEDUP_FILE = "dedup_images.json"
    DEDUP_LINKS = "input_dedup"

    def save_dedup_images(self, names: List[str]) -> None:
        (self.root / self.DEDUP_FILE).write_text(
            json.dumps({"count": len(names), "names": names}, ensure_ascii=False),
            encoding="utf-8",
        )

    def dedup_names(self) -> List[str]:
        path = self.root / self.DEDUP_FILE
        if not path.is_file():
            return []
        try:
            return list(json.loads(path.read_text(encoding="utf-8")).get("names") or [])
        except (json.JSONDecodeError, OSError):
            return []

    def rotation(self) -> int:
        """Quanti gradi in senso orario servono per raddrizzare le immagini."""
        try:
            return int(self.source.get("rotation_applied") or 0) % 360
        except (TypeError, ValueError):
            return 0

    def dedup_images(self) -> List[Path]:
        """Absolute paths of the working frames. Empty when the import has not run yet."""
        base = self.working_dir()
        if base is None or not base.is_dir():
            return []
        return [base / name for name in self.dedup_names()]

    def working_dir(self) -> Optional[Path]:
        """Da dove leggono tutti: lo specchio se c'e', altrimenti la cartella originale."""
        return self.dedup_link_dir() or (Path(self.source.get("folder") or "") or None)

    def dedup_link_dir(self, progress=None) -> Optional[Path]:  # noqa: ANN001
        """A mirror of the folder with only the unique frames.

        The depth and marker modules scan a folder themselves: they cannot be handed a list.
        Pointing them at this mirror is how they too see only the deduplicated images - the
        same trick `run_pipeline_single_folder_safe.py` uses with its `input_ref/`.

        Quando la cartella va ruotata, lo specchio contiene **copie gia' ruotate** invece di
        link. Misurare la rotazione e poi darla in pasto ai moduli come un numero non
        funziona: le reti non la ricevono affatto, e il modulo della scala si rifiuta di
        lavorare su un contesto ruotato (`rotation_not_supported`). Ruotare i pixel una volta
        sola, qui, e' cio' che la rende vera per tutti - e le coordinate che finiscono nel
        `.fss` sono quelle del fotogramma raddrizzato, che e' quello che ESI vedra'.
        """
        names = self.dedup_names()
        folder = Path(self.source.get("folder") or "")
        if not names or not folder.is_dir():
            return None
        angolo = self.rotation()
        override_grezzo = self.source.get("rotation_overrides") or {}
        overrides = {}
        for nome, valore in override_grezzo.items():
            try:
                gradi = int(valore) % 360
            except (TypeError, ValueError):
                continue
            if gradi in (0, 90, 180, 270):
                overrides[str(nome)] = gradi
        target = self.root / self.DEDUP_LINKS
        marker = target / ".built_from"
        stamp = f"{folder}\n{len(names)}\nrot={angolo}\noverrides={json.dumps(overrides, sort_keys=True)}"
        if marker.is_file() and marker.read_text(encoding="utf-8") == stamp:
            return target

        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        for indice, name in enumerate(names, start=1):
            destinazione = target / name
            destinazione.parent.mkdir(parents=True, exist_ok=True)
            source = folder / name
            try:
                rotazione = overrides.get(name, angolo)
                if rotazione:
                    from PIL import Image  # noqa: PLC0415

                    with Image.open(source) as immagine:
                        # `angle` e' la rotazione oraria che raddrizza: PIL ruota antiorario.
                        immagine.rotate(-rotazione, expand=True).save(destinazione)
                else:
                    destinazione.symlink_to(source)
            except OSError:
                continue
            if progress is not None and (indice % 25 == 0 or indice == len(names)):
                progress(indice, len(names))
        marker.write_text(stamp, encoding="utf-8")
        return target

    def fss_path(self) -> Path:
        id_setup = int(self.codes.get("id_setup") or 0)
        name = f"setup_{id_setup}.fss" if id_setup else "setup.fss"
        return self.root / "out" / name


def list_projects(projects_root: Path) -> List[Dict]:
    projects_root = Path(projects_root)
    if not projects_root.exists():
        return []
    out = []
    for path in sorted(projects_root.iterdir()):
        if not (path / "project.json").exists():
            continue
        try:
            project = Project.load(path)
        except (json.JSONDecodeError, OSError):
            continue
        relevant = [step for step in STEPS if project.step_is_relevant(step)]
        done = sum(1 for step in relevant if project.steps[step.id]["status"] in DONE_STATES)
        out.append(
            {
                "project_id": project.data["project_id"],
                "name": project.codes.get("project_name") or project.data["project_id"],
                "updated_at": project.data.get("updated_at"),
                "steps_done": done,
                "steps_total": len(relevant),
            }
        )
    return sorted(out, key=lambda item: item["updated_at"] or "", reverse=True)
