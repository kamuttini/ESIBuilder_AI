#!/usr/bin/env python3
"""ESIBuilder AI - versione base dell'app di configurazione.

Wizard locale a step confermati: si crea un progetto, si importa una cartella di
acquisizioni, ogni step propone/accetta un valore e alla fine si genera il file `.fss`
con il quality gate verso un `.fss` legacy di riferimento.

Questa e' la versione base: il wizard, lo stato del progetto con invalidazione a valle,
l'import e il writer `.fss` sono reali. Gli stadi di inferenza dei modelli non sono
ancora agganciati: gli step relativi accettano valori inseriti a mano.

Avvio:
  OldSoftwareEsiBuilder/.venv-mps/bin/python tools/app/server.py --port 8800
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from flask import Flask, jsonify, request, send_file, send_from_directory  # noqa: E402

import constants  # noqa: E402
from datetime import datetime  # noqa: E402

from anagrafica import Anagrafica, default_path  # noqa: E402
from importer import IMAGE_SUFFIXES, import_folder, resize_proposal, scan_folder  # noqa: E402
from inference import Engine, ModelPaths, sample_paths  # noqa: E402
from rotation import estimate_rotation  # noqa: E402
import marker_refine  # noqa: E402
import orientation_marker as om  # noqa: E402
import stages as stages_mod  # noqa: E402
from project import STEPS, Project, expand_rect, list_projects  # noqa: E402

REPO_ROOT = HERE.parents[1]
COMPARE_FSS = REPO_ROOT / "tools" / "fss" / "compare_fss.py"
DEFAULT_PROJECTS_ROOT = REPO_ROOT / "artifacts" / "80_app_projects"
DEFAULT_ACTIVE_PIPELINE = REPO_ROOT / "artifacts" / "10_active_pipeline" / "pipeline_fss_head"

app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")
_projects_root = DEFAULT_PROJECTS_ROOT
_anagrafica_path: Optional[Path] = None
_anagrafica: Optional[Anagrafica] = None


_engine: Optional[Engine] = None
_models_root: Optional[Path] = None
_jobs: Dict[str, Dict] = {}
_jobs_lock = threading.Lock()


def _models_dir() -> Path:
    """Checkpoints are read-only artifacts: in a git worktree they live in the main checkout."""
    if _models_root and Path(_models_root).exists():
        return Path(_models_root)
    for candidate in (DEFAULT_ACTIVE_PIPELINE, *(
        parent / "artifacts" / "10_active_pipeline" / "pipeline_fss_head"
        for parent in REPO_ROOT.parents
    )):
        if candidate.exists():
            return candidate
    return DEFAULT_ACTIVE_PIPELINE


def _inference_engine() -> Engine:
    """One engine per process: torch and the checkpoints load on first use."""
    global _engine
    if _engine is None:
        paths = ModelPaths.from_active_pipeline(_models_dir())
        missing = paths.missing()
        if missing:
            raise FileNotFoundError("checkpoint mancanti: " + ", ".join(missing))
        _engine = Engine(paths)
    return _engine


def _job_update(job_id: str, **fields) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def _fill_codes_from_analysis(project: Project, analysis: Dict) -> Dict:
    """Codes the analysis can answer. Never overwrites something the user already set."""
    codes = project.codes
    filled: Dict[str, str] = {}

    def put(key: str, value, source: str) -> None:
        current = codes.get(key)
        empty = current in (None, "", 0) or (isinstance(current, list) and not any(current))
        if value not in (None, "", 0) and empty:
            codes[key] = value
            filled[key] = source

    vendor = (analysis.get("vendor") or {}).get("vendor")
    probe = analysis.get("probe") or {}
    registry = analysis.get("registry") or {}
    plane = (analysis.get("plane") or {}).get("plane")

    put("echo_model", vendor, "rete vendor")
    put("probe_model", registry.get("model"), "anagrafica da ID sonda")
    put("id_probe", probe.get("probe_id"), "rete sonda")

    types = registry.get("probe_types") or []
    if registry.get("biplane"):
        if plane in ("L", "T"):
            put("probe_type", 3 if plane == "L" else 4, f"piano {plane} dalla rete L/T")
    elif types:
        put("probe_type", types[0], "anagrafica da ID sonda")

    combination = analysis.get("combination") or {}
    # ID ECHO, ID NDG e ID setup si compilano solo se la riga e' univoca: la rete vendor da'
    # la marca, e la stessa sonda vive su decine di modelli diversi.
    if not combination.get("ambiguous"):
        for key in ("id_echo", "id_ndg", "id_setup"):
            put(key, combination.get(key), "anagrafica (foglio FSS)")

    project.save()
    return filled


def _box_nel_fotogramma(box: Dict, size: Sequence[int]) -> bool:
    """Un box misurato su un fotogramma di dimensioni diverse non e' recuperabile."""
    try:
        larghezza, altezza = int(size[0]), int(size[1])
    except (TypeError, ValueError, IndexError):
        return True  # senza la dimensione non si puo' dire: si tiene
    if larghezza <= 0 or altezza <= 0:
        return True
    try:
        return (0 <= int(box["left"]) < int(box["right"]) <= larghezza
                and 0 <= int(box["top"]) < int(box["bottom"]) <= altezza)
    except (KeyError, TypeError, ValueError):
        return False


def _campione_stadi(payload: Dict) -> int:
    """Quante immagini per i tre moduli in coda all'import. `stages: false` li salta."""
    if payload.get("stages") is False:
        return 0
    return int(payload.get("sample_stages") or 12)


def _run_import_analysis(job_id: str, project_id: str, folder: str, sample: int,
                         sample_stadi: int = 12) -> None:
    """Step 0: dedup, rotation, vendor, probe, rect and L/T, before the user sees anything."""
    try:
        project = _project(project_id)

        _job_update(job_id, stage="scansione e dedup")
        imported = import_folder(Path(folder))
        # The deduplicated list is the working set from here on: nets, modules, viewer.
        project.save_dedup_images(imported.pop("kept_names", []))
        project.source.update(
            {
                key: imported[key]
                for key in (
                    "folder", "images_total_raw", "images_total", "duplicates_removed",
                    "rotation_applied", "rotation_source", "resize_factor", "native_size",
                )
            }
        )
        project.set_step("import", imported, status="proposed", source="import")

        project = _project(project_id)
        images = project.dedup_images()
        picked = sample_paths(images, sample)
        if not picked:
            raise FileNotFoundError("nessuna immagine da analizzare")

        _job_update(job_id, stage="rotazione (OSD)", total=len(picked))
        rotation = estimate_rotation(picked)
        angolo = int(rotation["angle"] or 0) % 360
        imported["rotation_applied"] = angolo
        imported["rotation_source"] = rotation["source"]
        imported["rotation_detail"] = rotation
        # Va scritto **e salvato** ora: piu' sotto il progetto viene riletto da disco, e in
        # memoria si perdeva. Da qui in poi lo specchio di lavoro contiene le immagini gia'
        # raddrizzate, e tutto - reti comprese - legge da li'.
        project.source["rotation_applied"] = angolo
        project.source["rotation_source"] = rotation["source"]
        project.save()
        if angolo:
            _job_update(job_id, stage=f"raddrizzo le immagini di {angolo} gradi", done=0,
                        total=len(project.dedup_names()))
            project.dedup_link_dir(
                progress=lambda fatte, quante: _job_update(job_id, done=fatte, total=quante)
            )
            larghezza, altezza = imported.get("native_size") or [0, 0]
            if angolo in (90, 270):
                # Il fotogramma raddrizzato ha i lati scambiati, e sono quelli che finiscono
                # nel `.fss`: e' quello che ESI vedra'.
                for chiave in ("native_size", "image_sample_size", "video_input_size"):
                    valore = imported.get(chiave)
                    if valore and len(valore) == 2:
                        imported[chiave] = [valore[1], valore[0]]
                project.source["native_size"] = [altezza, larghezza]
            project.save()
            project = _project(project_id)
            images = project.dedup_images()
            picked = sample_paths(images, sample)

        engine = _inference_engine()
        _job_update(job_id, stage="riconoscimento ecografo")
        vendor = engine.predict_vendor(picked)

        _job_update(job_id, stage="riconoscimento sonda")
        probe = engine.predict_probe(picked)

        registry: Dict = {}
        combination: Dict = {}
        try:
            anagrafica = _registry()
            if probe.get("probe_id") is not None:
                registry = anagrafica.probe_by_id(probe["probe_id"])
            if vendor.get("vendor") and registry.get("model"):
                resolved = anagrafica.resolve(vendor["vendor"], registry["model"])
                combination = dict(resolved.get("proposal") or {})
                combination["sources"] = resolved.get("sources") or {}
                combination["ambiguous"] = resolved.get("ambiguous", False)
                combination["ambiguous_ids"] = resolved.get("ambiguous_ids") or []
        except (FileNotFoundError, KeyError):
            registry = registry or {}

        _job_update(job_id, stage="rettangolo ecografico")
        rect = engine.predict_rect(
            picked, vendor.get("vendor"), vendor.get("confidence"),
            progress=lambda done, total: _job_update(job_id, done=done, total=total),
        )
        box = rect.get("rect_echo")

        _job_update(job_id, stage="template ecografo (#13)")
        line13 = engine.predict_line13(picked, vendor.get("vendor"))

        plane: Dict = {}
        if box and registry.get("biplane"):
            _job_update(job_id, stage="piano L/T")
            plane = engine.predict_lt(picked, box)

        analysis = {
            "vendor": vendor,
            "probe": probe,
            "registry": registry,
            "combination": combination,
            # The box is kept here as well: it is the value the "restore the network proposal"
            # button goes back to after the user has edited the boxes by hand.
            "rect": {
                k: rect.get(k)
                for k in ("source", "model", "agreement_iou", "spread_px", "images", "rect_echo")
            },
            "line13": line13,
            "plane": plane,
            "rotation": rotation,
            "device": engine.device,
            "images": len(picked),
        }

        project = _project(project_id)
        project.data["analysis"] = analysis
        project.steps["import"]["value"] = imported
        project.source["rotation_applied"] = angolo
        project.source["rotation_source"] = rotation["source"]
        project.save()

        # Il nome e' relativo allo specchio di lavoro, non alla cartella originale: da quando
        # c'e' una rotazione le immagini analizzate stanno li'.
        base_lavoro = project.working_dir() or Path(folder)
        anteprima = None
        if rect.get("boxes"):
            scelta = Path(rect["boxes"][0]["path"])
            try:
                anteprima = str(scelta.relative_to(base_lavoro))
            except ValueError:
                anteprima = scelta.name

        # --- rettangolo ecografico (#11): la geometria da cui dipende tutto il resto
        rect_value = dict(project.step_value("rect"))
        rect_value.update(
            {
                "rect_echo": box,
                "rect_source": rect.get("source"),
                "rect_model": rect.get("model"),
                "rect_agreement_iou": rect.get("agreement_iou"),
                "preview_image": anteprima,
            }
        )
        project.set_step(
            "rect", rect_value, status="proposed", source="model",
            confidence=rect.get("agreement_iou"),
        )

        # --- ecografo (#13): il template col nome della macchina
        vendor_value = dict(project.step_value("vendor"))
        vendor_value["preview_image"] = anteprima
        # Il box si teneva solo perche' c'era gia', e una rianalisi non lo toccava mai. Cosi'
        # dopo una rotazione restava quello misurato sul fotogramma storto — su una cartella
        # BK finiva a `left 1058` in un fotogramma largo 1024, cioe' fuori dall'immagine, e
        # nella sezione ecografo non si vedeva piu' niente. Si conserva solo se l'utente l'ha
        # messo a mano *e* sta ancora dentro al fotogramma.
        fotogramma = imported.get("image_sample_size") or imported.get("native_size") or []
        attuale = vendor_value.get("rect_name_echo")
        a_mano = bool((project.steps.get("vendor") or {}).get("user_edited"))
        fuori = bool(attuale) and not _box_nel_fotogramma(attuale, fotogramma)
        if line13.get("box") and (not attuale or not a_mano or fuori):
            # The box is the proposal; TH stays 0 until the thresholds step computes it.
            vendor_value["rect_name_echo"] = {
                **line13["box"],
                "check": 1,
                "params": {"threshold": 0.0},
            }
            vendor_value["rect_name_echo_source"] = line13.get("source")
            vendor_value["rect_name_echo_agreement"] = line13.get("agreement_iou")
            vendor_value["rect_name_echo_replaced"] = (
                "il box corretto a mano cadeva fuori dal fotogramma: rifatto dalla rete"
                if fuori and a_mano else ""
            )
        vendor_value["rect_name_echo_reason"] = line13.get("reason")
        project.set_step(
            "vendor", vendor_value, status="proposed", source="model",
            confidence=line13.get("agreement_iou"),
        )

        # --- sonda (#14): il template col nome della sonda, oggi senza rete
        probe_value = dict(project.step_value("probe"))
        probe_value["preview_image"] = anteprima
        project.set_step(
            "probe", probe_value,
            status="proposed" if probe_value.get("rect_name_probe") else "empty",
            source="model",
        )

        project = _project(project_id)
        filled = _fill_codes_from_analysis(project, analysis)
        project.data["analysis"]["codes_filled"] = filled
        project.steps["import"]["status"] = "confirmed"
        project.save()

        esito = {"import": imported, "analysis": analysis, "codes_filled": filled}
        if sample_stadi and box:
            # Un solo comando porta fino in fondo: i tre moduli girano di seguito, senza che
            # l'utente debba premere «calcola» in ogni sezione. Il rettangolo appena trovato
            # e' la loro precondizione; se manca ci si ferma qui, dicendolo.
            _run_advanced_stages(job_id, project_id, sample_stadi, base_result=esito)
            return
        if sample_stadi and not box:
            esito["stages_skipped"] = (
                "senza rettangolo ecografico i moduli non possono girare: sistemalo nella "
                "sezione «Rettangolo ecografico», poi ricalcola"
            )
        _job_update(job_id, status="done", stage="fatto", result=esito)
    except Exception as error:  # noqa: BLE001 - surfaced to the user as job error
        _job_update(job_id, status="error", stage="errore", error=str(error))


def _require_folder(project: Project) -> Path:
    """A folder that was imported but is no longer reachable is a different problem from a
    project with no folder at all: on external drives it usually just means "not mounted"."""
    raw = (project.source.get("folder") or "").strip()
    if not raw:
        raise FileNotFoundError("nessuna cartella importata: parti dallo step Import e analisi")
    folder = Path(raw)
    if not folder.is_dir():
        volume = ""
        parts = folder.parts
        if len(parts) > 2 and parts[1] == "Volumes":
            volume = f" Il volume '{parts[2]}' sembra non montato."
        raise FileNotFoundError(f"cartella non raggiungibile: {folder}.{volume}")
    return folder


def _righe_depth_per_nome(project: Project, righe: Sequence[Dict]) -> List[Dict]:
    """Le righe del modulo con il nome relativo, come le vuole il resto dell'app."""
    base = project.dedup_link_dir() or Path(project.source.get("folder") or "")
    fuori: List[Dict] = []
    for riga in righe:
        percorso = Path(str(riga.get("image_path") or ""))
        try:
            nome = str(percorso.relative_to(base))
        except ValueError:
            nome = percorso.name

        def numero(chiave: str) -> Optional[float]:
            try:
                return float(riga.get(chiave) or "")
            except (TypeError, ValueError):
                return None

        lati = [numero(k) for k in ("top", "left", "bottom", "right")]
        box = None
        if all(v is not None for v in lati):
            box = {"top": int(lati[0]), "left": int(lati[1]),
                   "bottom": int(lati[2]), "right": int(lati[3])}
        fuori.append({"name": nome, "mode": str(riga.get("mode") or ""),
                      "status": str(riga.get("status") or ""), "box": box,
                      "depth_mm": numero("depth_mm"),
                      "ocr_text": str(riga.get("ocr_text") or "")})
    return fuori


def _fotogrammi_per_depth(
    project: Project, base: Path, letture: Dict[str, Dict],
    righe_modulo: Sequence[Dict], per_valore: int = 6,
) -> Tuple[List[Path], List[Dict]]:
    """Fotogrammi scelti **per valore di depth**, non a campione piatto.

    La scala vuole piu' fotogrammi che concordino sullo stesso valore; un campione uniforme
    sulla cartella gliene dava uno per depth. Qui si prendono fino a `per_valore` immagini per
    ogni depth distinta, distribuite nel gruppo per non pescarle tutte dalla stessa
    acquisizione.
    """
    if not letture:
        # Nessuna rilettura: si resta alle righe del modulo, com'era prima.
        percorsi = [Path(r["image_path"]) for r in righe_modulo if r.get("image_path")]
        return percorsi, list(righe_modulo)

    per_depth: Dict[float, List[str]] = {}
    for nome, lettura in sorted(letture.items()):
        valore = lettura.get("depth_mm")
        if valore is None or valore <= 0:
            continue
        per_depth.setdefault(float(valore), []).append(nome)

    scelti: List[str] = []
    for valore in sorted(per_depth):
        gruppo = per_depth[valore]
        if len(gruppo) <= per_valore:
            scelti.extend(gruppo)
            continue
        passo = len(gruppo) / per_valore
        scelti.extend(gruppo[int(indice * passo)] for indice in range(per_valore))

    righe = [
        {
            "image_path": (base / nome).as_posix(),
            "depth_mm": letture[nome]["depth_mm"],
            "status": "accepted",
            "mode": "direct_label",
            "ocr_text": letture[nome].get("ocr_text", ""),
            **{k: letture[nome]["box"][k] for k in ("left", "top", "right", "bottom")},
        }
        for nome in scelti
    ]
    return [base / nome for nome in scelti], righe


def _run_advanced_stages(
    job_id: str, project_id: str, sample: int, marker_min_score: float = 0.55,
    base_result: Optional[Dict] = None,
) -> None:
    """Orientation marker, depth and scale: the three modules, run as the pipeline runs them."""
    try:
        project = _project(project_id)
        folder = _require_folder(project)
        rect = project.step_value("rect").get("rect_echo")
        if not rect:
            raise ValueError("serve prima il rettangolo ecografico (step «Rettangolo ecografico»)")

        analysis = project.data.get("analysis") or {}
        vendor = ((analysis.get("vendor") or {}).get("vendor") or "")
        vendor_conf = float((analysis.get("vendor") or {}).get("confidence") or 0.0)
        probe_id = (analysis.get("probe") or {}).get("probe_id")
        imported = project.step_value("import")
        video_size = imported.get("image_sample_size") or [0, 0]
        # Le immagini che i moduli ricevono sono gia' raddrizzate (lo specchio di lavoro le
        # contiene ruotate), quindi il contesto deve dire zero. Dirgli l'angolo vero
        # significherebbe farglielo applicare due volte, e il modulo della scala si rifiuta
        # proprio di lavorare su un contesto ruotato.
        rotation = 0
        # #11 carries the rectangle plus its margin: the modules must see the same box ESI will.
        final_rect = expand_rect(
            rect, project.step_value("rect").get("margin_percent"),
            width=video_size[0], height=video_size[1],
        )

        out_root = project.root / "stages"
        python_bin = sys.executable
        results: Dict = {}
        # The subprocess modules scan a folder: they get the mirror with only unique frames.
        scan_folder_for_modules = project.dedup_link_dir() or folder
        base_moduli = scan_folder_for_modules
        results["images_source"] = {
            "deduplicated": scan_folder_for_modules != folder,
            "folder": str(scan_folder_for_modules),
            "unique_images": len(project.dedup_names()),
        }

        # The marker runs on EVERY deduplicated image, not on a sample: the envelope of a
        # group is only as good as the markers it saw, and the viewer must be able to show a
        # box on any image the user scrolls to.
        unique_count = len(project.dedup_names())
        _job_update(
            job_id,
            stage=f"orientamento: marker su tutte le {unique_count or '?'} immagini",
        )
        artifacts = _models_dir().parents[1]
        marker = stages_mod.run_marker_envelopes(
            folder=scan_folder_for_modules, output_root=out_root, python_bin=python_bin,
            bundle_dir=artifacts / "41_orientation_marker_detector_bundle",
            library_root=_models_dir() / "lr_marker_vendor_template_library",
            vendor=vendor, max_images=0, timeout=3600.0,
        )
        results["marker"] = {k: marker.get(k) for k in ("status", "error", "output_dir")}
        if marker.get("rows"):
            # Il lavoro umano si legge subito: correzioni e ritagli dei suggerimenti
            # servono sia per gli envelope sia per la validazione, che vengono prima della
            # costruzione del nuovo valore dello step.
            previous = project.step_value("orientation")
            corrections_before = dict(previous.get("corrections") or {})
            # --- SCOPERTA: le detection della banca costruiscono gli envelope. Conta il
            # richiamo: se il box non contiene il marker, ESI non lo trovera' mai.
            # Tenute per nome: una riga per immagine, cosi' una correzione la sostituisce
            # invece di aggiungersi (altrimenti la posizione vecchia resta nell'unione e il
            # box non si stringe).
            rows_by_name: Dict[str, Dict] = {
                row["name"]: {"group": row["group"], "box": row["box"]}
                for row in _marker_rows(Path(marker["output_dir"]))
                if row.get("box") and str(row.get("status") or "").lower() == "ok"
            }

            # --- RITAGLIO: il glifo di questa cartella, scelto per copertura e non per
            # punteggio singolo. E' il file che andra' in DB_echo, quindi e' quello che ESI
            # usera' davvero.
            _job_update(job_id, stage="orientamento: ritaglio del marker della cartella")
            useful = [
                path for path in scan_folder(scan_folder_for_modules)
                if not re.search(r"Thumbs\.db|Software Release|System Info|proibite",
                                 str(path), re.IGNORECASE)
            ]
            template_info: Dict = {}
            override = previous.get("marker_override") or {}
            override_rows: Dict[str, Dict] = {}
            if override.get("path") and Path(override["path"]).is_file():
                # L'utente ha indicato il marker a mano: il ricalcolo non torna alla banca,
                # che su questa cartella aveva scelto il glifo sbagliato. Le posizioni, i
                # gruppi e gli envelope si rifanno tutti dal suo ritaglio.
                _job_update(job_id, stage="orientamento: uso il marker che hai indicato")
                soglia = float(marker_min_score)
                template_info = {
                    **{k: override[k] for k in ("path", "size", "source_image") if k in override},
                    "from_user_marker": True,
                }
                own = om.match_all(
                    images=useful, folder=scan_folder_for_modules,
                    template_path=Path(override["path"]), rect=final_rect,
                    bundle_dir=artifacts / "41_orientation_marker_detector_bundle",
                    min_score=soglia, search_margin=60,
                )
                agganciate = [
                    row for row in own["rows"]
                    if row.get("box") and (row.get("score") or 0) >= soglia
                ]
                rows_by_name = {
                    row["name"]: {"group": row["group"], "box": row["box"]}
                    for row in agganciate
                }
                override_rows = {
                    row["name"]: {"score": row["score"], "group": row["group"],
                                  "box": row["box"]}
                    for row in agganciate
                }
            else:
                try:
                    candidates = om.candidate_crops(
                        images=useful, bundle_dir=artifacts / "41_orientation_marker_detector_bundle",
                        library_root=_models_dir() / "lr_marker_vendor_template_library",
                        vendor=vendor, rect=final_rect,
                        out_dir=project.root / "templates" / "candidates", seed_images=6,
                    )
                    chosen = om.choose_by_coverage(
                        candidates=candidates, images=useful, folder=scan_folder_for_modules,
                        rect=final_rect, bundle_dir=artifacts / "41_orientation_marker_detector_bundle",
                        sample=40,
                    )
                    if "chosen" in chosen:
                        source = Path(chosen["chosen"]["path"])
                        final = project.root / "templates" / "orientation_marker.png"
                        final.parent.mkdir(parents=True, exist_ok=True)
                        final.write_bytes(source.read_bytes())
                        template_info = {**chosen["chosen"], "path": str(final)}
                        # le detection del ritaglio allargano l'envelope dove la banca non arriva
                        own = om.match_all(
                            images=useful, folder=scan_folder_for_modules,
                            template_path=final, rect=final_rect,
                            bundle_dir=artifacts / "41_orientation_marker_detector_bundle",
                            min_score=0.85, search_margin=40,
                        )
                        for row in own["rows"]:
                            if row.get("box") and row["score"] >= 0.85:
                                # riempie i buchi: dove la banca non e' arrivata ci mette la
                                # sua detection, dove c'e' gia' non aggiunge un secondo box
                                # per la stessa immagine
                                rows_by_name.setdefault(
                                    row["name"], {"group": row["group"], "box": row["box"]}
                                )
                except (FileNotFoundError, ImportError) as error:
                    template_info = {"error": str(error)}

            # le correzioni umane vincono sull'immagine corrispondente, anche dopo un
            # ricalcolo: sostituiscono la detection del modulo, non le si affiancano
            for name, fixed in corrections_before.items():
                if fixed.get("box") and fixed.get("group"):
                    rows_by_name[name] = {"group": fixed["group"], "box": fixed["box"]}
            group_of_image = {name: row["group"] for name, row in rows_by_name.items()}
            groups = om.envelopes(
                [{"group": row["group"], "box": row["box"], "score": 1.0}
                 for row in rows_by_name.values()],
                min_score=0.0,
            )

            # --- VALIDAZIONE: il ritaglio consegnato, cercato DENTRO ogni envelope. E' la
            # previsione di cosa fara' ESI, e da qui esce la copertura per gruppo.
            validation: Dict = {}
            if template_info.get("path") and groups:
                _job_update(job_id, stage="orientamento: validazione del ritaglio negli envelope")
                validation = om.validate_in_envelopes(
                    images=useful, folder=scan_folder_for_modules,
                    template_path=Path(template_info["path"]), groups=groups,
                    group_of_image=group_of_image,
                    bundle_dir=artifacts / "41_orientation_marker_detector_bundle",
                    min_score=float(marker_min_score),
                    extra_templates=[
                        Path(h) for h in (previous.get("hint_paths") or {}).values()
                        if Path(h).is_file()
                    ],
                )

            order = marker_refine.GROUP_ORDER
            found = [g for g in order if g in groups]
            missing = [g for g in order if g not in groups]
            # Il ricalcolo non deve cancellare il lavoro umano: correzioni e ritagli dei
            # suggerimenti sopravvivono al rifacimento dello stadio.
            orientation_value: Dict = {
                "corrections": corrections_before,
                "marker_override": override,
                "marker_rows_override": override_rows,
                # La diagnosi "questo marker non si muove" e' l'unico modo automatico di
                # accorgersi che la banca ha scelto un elemento fisso dell'interfaccia.
                "marker_warning": _marker_warning(
                    list(override_rows.values()) if override_rows
                    else [r for r in _marker_rows(Path(marker["output_dir"]))
                          if str(r.get("status") or "").lower() == "ok"],
                    groups,
                ),
                "hint_paths": {
                    name: path for name, path in (previous.get("hint_paths") or {}).items()
                    if name in corrections_before and Path(path).is_file()
                },
                "hint_templates": [
                    path for name, path in (previous.get("hint_paths") or {}).items()
                    if name in corrections_before and Path(path).is_file()
                ],
                "group_orientation": project.codes.get("group_orientation") or 4,
                "groups": groups,
                "found_groups": found,
                "missing_groups": missing,
                "orientation_available": {g: g in groups for g in order},
                "folder_template": template_info,
                "validation": {
                    k: validation.get(k) for k in ("coverage", "coverage_by_group", "images", "min_score")
                },
                "validation_rows": {
                    row["name"]: {"score": row["score"], "group": row["group"], "box": row["box"]}
                    for row in (validation.get("rows") or [])
                },
            }
            if found:
                donor = found[0]
                filled, blocks = [], []
                for g in order:
                    source_box = groups.get(g)
                    if source_box is None:
                        source_box = groups[donor]
                        filled.append(g)
                    blocks.append({k: source_box[k] for k in ("top", "left", "bottom", "right", "check", "params")})
                orientation_value["blocks"] = blocks
                orientation_value["filled_groups"] = filled
                orientation_value["filled_from"] = donor if filled else ""

            results["marker"]["groups"] = found
            results["marker"]["missing_groups"] = missing
            results["marker"]["filled_groups"] = orientation_value.get("filled_groups") or []
            results["marker"]["filled_from"] = orientation_value.get("filled_from") or ""
            results["marker"]["folder_template"] = template_info
            results["marker"]["validation"] = orientation_value["validation"]

            project = _project(project_id)
            if orientation_value["orientation_available"]:
                project.codes["orientation_available"] = orientation_value["orientation_available"]
                results["marker"]["orientation_available"] = orientation_value["orientation_available"]
            project.set_step(
                "orientation", orientation_value,
                status="proposed" if orientation_value.get("blocks") else "blocked",
                source="model",
            )
            project.save()

        _job_update(job_id, stage="depth: lettura dalla scala ecografica")
        depth = stages_mod.run_depth(
            folder=scan_folder_for_modules, output_root=out_root, python_bin=python_bin,
            vendor=vendor, probe_id=str(probe_id or ""),
            probe_model=str(project.codes.get("probe_model") or ""), rect=final_rect,
            video_size=video_size, rotation=rotation, max_images=sample,
        )
        results["depth"] = {
            k: depth.get(k)
            for k in ("status", "error", "images", "accepted", "acceptance_ratio", "depths_mm",
                      "output_dir", "ranker_model", "folder_strategy")
        }

        # --- la depth su tutta la cartella, che e' cio' di cui la scala ha bisogno ------
        # Il modulo della scala misura il righello *per valore di depth*, e per accettare un
        # valore vuole vedere piu' fotogrammi che concordano. Con un campione piatto ogni
        # depth compariva una volta sola e ogni riga usciva `setup_too_small_for_consensus`.
        # Rileggere la depth dentro al riquadro gia' trovato costa 0.4 s per immagine, contro
        # i minuti della generazione dei candidati: si fa su tutta la cartella, e poi la
        # scala riceve piu' fotogrammi per ogni depth.
        righe_depth = list(depth.get("rows") or [])
        letture: Dict[str, Dict] = {}
        riferimento = _riquadro_depth_di_riferimento(_righe_depth_per_nome(project, righe_depth))
        if riferimento is not None:
            nome_rif, box_rif, testo_rif, valore_rif = riferimento
            numero = _rileggi_nel_riquadro(base_moduli / nome_rif, box_rif)
            if numero is not None:
                fattore = _fattore_unita(testo_rif, valore_rif, numero["value"])
                tutti = list(project.dedup_names())
                _job_update(job_id, stage=f"depth: la stessa etichetta su {len(tutti)} immagini",
                            done=0, total=len(tutti))
                letture, falliti = _leggi_riquadro_su(
                    base_moduli, box_rif, fattore, tutti,
                    progress=lambda fatte, quante: _job_update(job_id, done=fatte),
                )
                if letture:
                    def salva_letture(_p: Project, value: Dict) -> Dict:
                        value["depth_box_template"] = {
                            "box": box_rif, "from": nome_rif, "unit_factor": fattore,
                            "scope": "auto", "applied": len(letture), "targets": len(tutti),
                            "failed": falliti,
                            "at": datetime.now().isoformat(timespec="seconds"),
                        }
                        value["depth_box_reads"] = letture
                        return value

                    _write_step(project_id, "depth_scale", salva_letture,
                                status="proposed", source="model")
                    results["depth"]["box_reads"] = len(letture)
                    results["depth"]["box_failed"] = len(falliti)

        _job_update(job_id, stage="scala: righello e righe #18-#21")
        engine = _inference_engine()
        # Same base as the depth module: the scale stage joins su/giu and depth rows by path,
        # so both must see the images through the same folder (the deduplicated mirror).
        images, righe_depth = _fotogrammi_per_depth(
            project, base_moduli, letture, righe_depth, sample
        )
        su_giu = engine.predict_su_giu(images, final_rect)
        results["su_giu"] = {
            k: su_giu.get(k) for k in ("counts", "majority", "images", "mean_confidence")
        }
        frames = stages_mod.build_scale_frames(
            images=images, su_giu_rows=su_giu.get("rows") or [],
            depth_rows=righe_depth, rect=final_rect,
        )
        gruppi_immagine = _groups_of_images(project)
        per_nome = {}
        for gruppo, elenco in (gruppi_immagine or {}).items():
            for nome in elenco:
                per_nome[nome] = gruppo
        for frame in frames:
            try:
                nome = str(Path(frame["image_path"]).relative_to(base_moduli))
            except ValueError:
                nome = Path(frame["image_path"]).name
            # L'orientamento e' il dato che dice alla scala da che parte guardare: era
            # sempre vuoto, benche' il marker fosse gia' girato.
            frame["orientation_group"] = per_nome.get(nome, "")
        scale = stages_mod.run_scale(
            folder=scan_folder_for_modules, output_root=out_root, python_bin=python_bin,
            vendor=vendor, vendor_confidence=vendor_conf, probe_id=str(probe_id or ""),
            rect=final_rect, video_size=video_size, frames=frames, max_frames=len(frames),
            rotation=rotation,
        )
        results["scale"] = {
            k: scale.get(k)
            for k in ("status", "error", "depths_total", "depths_accepted", "acceptance_ratio",
                      "review_reasons", "profile", "output_dir", "lines")
        }

        # Lo studio del righello: modulo autonomo, non ha bisogno della depth. Gira sullo
        # stesso contesto dello stadio, cosi' i due guardano gli stessi fotogrammi ed e'
        # possibile confrontarli tacca per tacca.
        if scale.get("output_dir"):
            _job_update(job_id, stage="scala: studio del righello, tacca per tacca")
            studio = stages_mod.run_scale_study(
                context_dir=Path(scale["output_dir"]), python_bin=python_bin,
                vendor=vendor, max_images=max(14, sample),
            )
            results["scale_study"] = {
                k: studio.get(k)
                for k in ("status", "error", "frames", "by_status", "vendor", "zone",
                          "output_dir", "data_json")
            }

        parsed = stages_mod.parse_scale_lines(scale.get("lines") or {})
        if parsed.get("depths"):
            project = _project(project_id)
            complete = bool(parsed["pixel_ratio_x"] and parsed["pixel_ratio_y"] and parsed["scale_lines"])
            project.set_step(
                "depth_scale",
                {**parsed, "depth_module": results["depth"], "scale_module": results["scale"]},
                status="proposed" if complete else "blocked",
                source="model",
            )

        project = _project(project_id)
        project.data.setdefault("analysis", {})["stages"] = results
        project.save()
        _job_update(job_id, status="done", stage="fatto",
                    result={**(base_result or {}), "stages": results})
    except Exception as error:  # noqa: BLE001
        _job_update(job_id, status="error", stage="errore", error=str(error))


@app.post("/api/projects/<project_id>/analyze_stages")
def api_analyze_stages(project_id: str):
    """Run the orientation, depth and scale modules on the imported folder."""
    _project(project_id)
    payload = _payload()
    sample = int(payload.get("sample") or 12)
    # Soglia della validazione: la tolleranza legacy e' molto piu' lenta di quanto sembri
    # (fino a 0.368 di correlazione in setup_53), quindi 0.85 escluderebbe match buoni.
    min_score = float(payload.get("marker_min_score") or 0.55)
    return jsonify(
        {"job_id": _start_job(_run_advanced_stages, project_id, sample, min_score)}
    )


def _start_job(target, *args) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"id": job_id, "status": "running", "stage": "avvio", "done": 0, "total": 0}
    threading.Thread(target=target, args=(job_id, *args), daemon=True).start()
    return job_id


@app.post("/api/projects/<project_id>/analyze")
def api_analyze(project_id: str):
    """Re-run step 0 on the folder already imported."""
    project = _project(project_id)
    folder = project.source.get("folder") or ""
    if not folder:
        return jsonify({"error": "importa prima una cartella"}), 400
    payload = _payload()
    sample = int(payload.get("sample") or 24)
    return jsonify({"job_id": _start_job(_run_import_analysis, project_id, folder, sample,
                                         _campione_stadi(payload))})


@app.get("/api/jobs/<job_id>")
def api_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "job sconosciuto"}), 404
    return jsonify(job)


def _registry(reload: bool = False) -> Anagrafica:
    """The encoding_struct workbook, loaded once and reloaded after a write."""
    global _anagrafica
    if _anagrafica is None or reload:
        if _anagrafica_path is None or not Path(_anagrafica_path).exists():
            raise FileNotFoundError("encoding_struct")
        _anagrafica = Anagrafica(Path(_anagrafica_path))
    return _anagrafica


def _project(project_id: str) -> Project:
    root = _projects_root / project_id
    if not (root / "project.json").exists():
        raise FileNotFoundError(project_id)
    return Project.load(root)


def _payload() -> Dict:
    return request.get_json(silent=True) or {}


@app.errorhandler(FileNotFoundError)
def _not_found(error):  # noqa: ANN001
    return jsonify({"error": f"progetto non trovato: {error}"}), 404


@app.errorhandler(NotADirectoryError)
def _bad_folder(error):  # noqa: ANN001
    return jsonify({"error": f"cartella non valida: {error}"}), 400


# -- pagine ----------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# -- metadati --------------------------------------------------------------
@app.get("/api/meta")
def api_meta():
    return jsonify(
        {
            "steps": [
                {
                    "id": step.id,
                    "label": step.label,
                    "legacy_page": step.legacy_page,
                    "fss_lines": list(step.fss_lines),
                    "depends_on": list(step.depends_on),
                }
                for step in STEPS
            ],
            "probe_types": constants.PROBE_TYPE_LABELS,
            "video_inputs": constants.VIDEO_INPUT_LABELS,
            "group_orientations": constants.GROUP_ORIENTATION_LABELS,
            "biplana_modes": constants.BIPLANA_MODE_LABELS,
            "orientation_keys": list(constants.ORIENTATION_KEYS),
            "match_defaults": {
                "channel": constants.DEFAULT_CHANNEL,
                "match_method": constants.DEFAULT_MATCH_METHOD,
                "p1": constants.DEFAULT_P1,
                "p2": constants.DEFAULT_P2,
            },
            "rect_limit": [constants.RECT_MAX_WIDTH, constants.RECT_MAX_HEIGHT],
            "projects_root": str(_projects_root),
            "anagrafica": str(_anagrafica_path) if _anagrafica_path else None,
        }
    )


# -- progetti --------------------------------------------------------------
@app.get("/api/projects")
def api_projects():
    return jsonify({"projects": list_projects(_projects_root)})


@app.post("/api/projects")
def api_project_create():
    name = (_payload().get("name") or "").strip()
    if not name:
        return jsonify({"error": "il nome del progetto e' obbligatorio"}), 400
    project = Project.create(_projects_root, name)
    return jsonify({"project_id": project.data["project_id"]})


@app.get("/api/projects/<project_id>")
def api_project(project_id: str):
    project = _project(project_id)
    return jsonify(
        {
            "project": project.data,
            "status": project.status_report(),
            "is_biplane": project.is_biplane,
            "fss_path": str(project.fss_path()),
        }
    )


@app.post("/api/projects/<project_id>/codes")
def api_project_codes(project_id: str):
    project = _project(project_id)
    stale = project.update_codes(_payload().get("codes") or {})
    return jsonify({"stale": stale, "status": project.status_report()})


# --- sfoglia le cartelle -----------------------------------------------------
# Il percorso si incollava a mano. Il browser non puo' dare il percorso vero di una cartella
# scelta con `<input type=file>` — restituisce nomi relativi — ma il server gira sulla stessa
# macchina dell'utente, quindi e' lui a sfogliare il disco. Funziona uguale su Windows.


def _radici() -> List[Dict]:
    """Da dove si parte: casa, i volumi montati, e le cartelle dei progetti gia' importati."""
    voci: List[Dict] = []
    visti = set()

    def aggiungi(percorso: Path, etichetta: str, tipo: str) -> None:
        try:
            if not percorso.is_dir():
                return
        except OSError:
            return
        chiave = percorso.as_posix()
        if chiave in visti:
            return
        visti.add(chiave)
        voci.append({"path": chiave, "label": etichetta, "kind": tipo})

    aggiungi(Path.home(), "Home", "home")
    for nome in ("Desktop", "Documents", "Downloads"):
        aggiungi(Path.home() / nome, nome, "home")
    if sys.platform == "darwin":
        volumi = Path("/Volumes")
        if volumi.is_dir():
            for voce in sorted(volumi.iterdir()):
                aggiungi(voce, voce.name, "volume")
    elif sys.platform.startswith("win"):
        for lettera in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            aggiungi(Path(f"{lettera}:\\"), f"Disco {lettera}:", "volume")
    else:
        for base in (Path("/media"), Path("/mnt")):
            if base.is_dir():
                for voce in sorted(base.iterdir()):
                    aggiungi(voce, voce.name, "volume")

    for progetto in list_projects(_projects_root):
        try:
            dati = _project(progetto["project_id"])
        except Exception:  # noqa: BLE001
            continue
        cartella = str((dati.source or {}).get("folder") or "")
        if cartella:
            # La cartella madre e' piu' utile della cartella stessa: da li' si sceglie
            # l'acquisizione successiva, che di solito e' la sorella accanto.
            aggiungi(Path(cartella).parent, Path(cartella).parent.name, "recente")
            aggiungi(Path(cartella), Path(cartella).name, "recente")
    return voci


@app.get("/api/browse")
def api_browse():
    """Le sottocartelle di un percorso, con quante immagini contengono direttamente."""
    grezzo = (request.args.get("path") or "").strip()
    if not grezzo:
        return jsonify({"roots": _radici(), "path": "", "parent": None, "entries": []})

    cartella = Path(grezzo).expanduser()
    try:
        cartella = cartella.resolve()
    except OSError as errore:
        return jsonify({"error": str(errore)}), 400
    if not cartella.is_dir():
        return jsonify({"error": f"non e' una cartella: {cartella}"}), 404

    voci: List[Dict] = []
    immagini_qui = 0
    try:
        with os.scandir(cartella) as elenco:
            for voce in elenco:
                if voce.name.startswith("."):
                    continue
                try:
                    if voce.is_dir(follow_symlinks=True):
                        voci.append({"name": voce.name,
                                     "path": (cartella / voce.name).as_posix()})
                    elif Path(voce.name).suffix.lower() in IMAGE_SUFFIXES:
                        immagini_qui += 1
                except OSError:
                    continue
    except PermissionError:
        return jsonify({"error": f"non ho i permessi per leggere {cartella}"}), 403

    # Quante immagini ci sono dentro a ciascuna sottocartella: solo il livello diretto, che
    # su un disco esterno con migliaia di file una conta ricorsiva costerebbe secondi.
    for voce in voci:
        conta = 0
        figlie = 0
        try:
            with os.scandir(voce["path"]) as dentro:
                for figlia in dentro:
                    if figlia.name.startswith("."):
                        continue
                    if figlia.is_dir(follow_symlinks=False):
                        figlie += 1
                    elif Path(figlia.name).suffix.lower() in IMAGE_SUFFIXES:
                        conta += 1
        except OSError:
            conta = -1
        voce["images"] = conta
        voce["folders"] = figlie
    voci.sort(key=lambda v: v["name"].lower())

    genitore = cartella.parent
    return jsonify({
        "path": cartella.as_posix(),
        "parent": genitore.as_posix() if genitore != cartella else None,
        "name": cartella.name or cartella.as_posix(),
        "images_here": immagini_qui,
        "entries": voci,
        "roots": _radici(),
    })


@app.post("/api/projects/<project_id>/import")
def api_project_import(project_id: str):
    """Step 0: import plus the whole recognition chain, as one background job."""
    _project(project_id)
    payload = _payload()
    folder = (payload.get("folder") or "").strip()
    if not folder:
        return jsonify({"error": "indica la cartella delle acquisizioni"}), 400
    if not Path(folder).expanduser().is_dir():
        return jsonify({"error": f"cartella non valida: {folder}"}), 400
    sample = int(payload.get("sample") or 24)
    return jsonify({"job_id": _start_job(_run_import_analysis, project_id, folder, sample,
                                         _campione_stadi(payload))})


@app.post("/api/projects/<project_id>/steps/<step_id>")
def api_project_step(project_id: str, step_id: str):
    project = _project(project_id)
    payload = _payload()
    try:
        stale = project.set_step(
            step_id,
            payload.get("value") or {},
            status=payload.get("status") or "confirmed",
            source=payload.get("source") or "user",
            confidence=payload.get("confidence"),
        )
    except KeyError:
        return jsonify({"error": f"step sconosciuto: {step_id}"}), 400
    return jsonify({"stale": stale, "status": project.status_report()})


@app.post("/api/projects/<project_id>/resize_check")
def api_resize_check(project_id: str):
    """Spec sezione 11: propose a resize only when the rectangle overflows the ESI screen."""
    project = _project(project_id)
    rect = _payload().get("rect_echo") or {}
    if not rect:
        return jsonify({"error": "manca il rettangolo"}), 400
    sample = project.step_value("import").get("image_sample_size") or project.codes.get(
        "image_sample_size"
    )
    return jsonify({"proposal": resize_proposal(rect, list(sample or [0, 0]))})


# -- anagrafica (encoding_struct) -----------------------------------------
@app.get("/api/anagrafica")
def api_anagrafica():
    registry = _registry()
    return jsonify(
        {
            "path": str(registry.path),
            "probe_types": registry.probe_types(),
            "combinations": len(registry.fss_rows()),
        }
    )


@app.post("/api/anagrafica/resolve")
def api_anagrafica_resolve():
    payload = _payload()
    registry = _registry()
    return jsonify(
        registry.resolve(
            (payload.get("echo_model") or "").strip(),
            (payload.get("probe_model") or "").strip(),
        )
    )


@app.post("/api/anagrafica/candidates")
def api_anagrafica_candidates():
    """Setups registered for a probe: the list the user picks the machine model from."""
    payload = _payload()
    probe_id = payload.get("probe_id")
    if probe_id is None:
        return jsonify({"error": "serve probe_id"}), 400
    probe_type = payload.get("probe_type")
    rows = _registry().combinations_for_probe(
        int(probe_id), int(probe_type) if probe_type not in (None, "") else None
    )
    return jsonify({"candidates": rows, "count": len(rows)})


@app.post("/api/anagrafica/new_row")
def api_anagrafica_new_row():
    """Create the missing row in encoding_struct. A backup is taken before saving."""
    payload = _payload()
    registry = _registry()
    kind = (payload.get("kind") or "").strip()
    try:
        if kind == "echo":
            result = registry.add_echo((payload.get("model") or "").strip())
        elif kind == "probe":
            result = registry.add_probe((payload.get("model") or "").strip())
        elif kind == "combination":
            result = registry.add_combination(payload.get("entry") or {})
        else:
            return jsonify({"error": f"tipo di riga sconosciuto: {kind}"}), 400
    except (RuntimeError, KeyError) as error:
        return jsonify({"error": str(error)}), 400
    _registry(reload=True)
    return jsonify(result)


# -- orientamento: envelope, template e detection per immagine -------------
def _marker_stage_dir(project: Project) -> Optional[Path]:
    raw = (((project.data.get("analysis") or {}).get("stages") or {}).get("marker") or {}).get(
        "output_dir"
    )
    path = Path(raw) if raw else None
    return path if path and path.is_dir() else None


@app.get("/api/projects/<project_id>/orientation")
def api_orientation(project_id: str):
    """What the marker stage found: the four envelopes, the chosen template, per-image boxes."""
    project = _project(project_id)
    stage = _marker_stage_dir(project)
    if stage is None:
        return jsonify({"error": "lancia prima i moduli dallo step Orientamento"}), 400

    def read(name: str) -> List[Dict[str, str]]:
        path = stage / name
        if not path.is_file():
            return []
        import csv as _csv

        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in _csv.DictReader(handle)]

    def box(raw: str) -> Optional[Dict[str, int]]:
        parts = [p for p in str(raw or "").split("|") if p.strip()]
        if len(parts) < 4:
            return None
        top, left, bottom, right = (int(float(v)) for v in parts[:4])
        return {"top": top, "left": left, "bottom": bottom, "right": right}

    # The runner appends to its CSVs across runs: the last row for a key is the current one.
    envelopes = {}
    for row in read("folder_envelopes.csv"):
        group = str(row.get("group", "")).strip().upper()
        parsed = box(row.get("envelope_box", ""))
        if group and parsed:
            envelopes[group] = {**parsed, "markers": int(float(row.get("n_markers") or 0))}

    # Gli envelope dello step sono la verita' corrente: li ha scritti l'ultimo passaggio
    # (modulo, correzione o marker indicato a mano). Il CSV del batch invece si accoda fra
    # le run, quindi conta anche i marker di run vecchie e non conosce le correzioni.
    stored = project.step_value("orientation")
    corrections = dict(stored.get("corrections") or {})
    if stored.get("groups"):
        envelopes = {
            group: {**data, "markers": data.get("markers", 0)}
            for group, data in stored["groups"].items()
        }

    summary = (read("folder_summary.csv") or [{}])[-1]
    folder_template = (stored.get("folder_template") or {})
    validation = (stored.get("validation") or {})
    validation_rows = (stored.get("validation_rows") or {})
    by_name: Dict[str, Dict] = {}
    override_rows = stored.get("marker_rows_override") or {}
    if override_rows:
        # Il marker indicato dall'utente ha le sue posizioni: quelle del batch appartengono
        # al glifo che ha sbagliato, mostrarle sarebbe mostrare l'errore.
        for name, row in override_rows.items():
            by_name[name] = {
                "name": name, "group": row.get("group", ""), "status": "ok",
                "score": row.get("score"), "box": row.get("box"),
                "template": "marker indicato a mano", "scope": "override",
            }
    else:
        for row in read("per_image_predictions.csv"):
            name = row.get("image_id", "")
            by_name[name] = {
                "name": name,
                "group": row.get("pred_group", ""),
                "status": row.get("status", ""),
                "score": round(float(row.get("match_score") or 0.0), 4),
                "box": box(row.get("marker_box_abs", "")),
                "template": row.get("template_name", ""),
                "scope": row.get("search_scope", ""),
            }
    for name, fixed in corrections.items():
        row = by_name.get(name) or {"name": name, "template": fixed.get("template", ""), "scope": "click"}
        row.update(
            {
                "group": fixed.get("group", row.get("group", "")),
                "box": fixed.get("box"),
                "score": fixed.get("score"),
                "status": "corrected",
                "corrected": True,
            }
        )
        by_name[name] = row
    # Il punteggio e la posizione che contano sono quelli della validazione (il ritaglio
    # consegnato cercato dentro l'envelope): e' la previsione di cosa fara' ESI. Mostrare il
    # box della scoperta col punteggio della validazione faceva sembrare vecchio il ritaglio.
    # Le immagini corrette restano dove le ha messe l'utente.
    for row_name, row in by_name.items():
        v = validation_rows.get(row_name)
        if not v or row.get("corrected"):
            continue
        row["score_batch"] = row.get("score")
        row["score"] = v.get("score")
        row["box_batch"] = row.get("box")
        if v.get("box"):
            row["box"] = v["box"]
    per_image = list(by_name.values())

    # The runner skips these on purpose: forbidden/freeze screens carry no orientation marker.
    marker_excluded = r"Thumbs\.db|Software Release|System Info|proibite"
    return jsonify(
        {
            "envelopes": envelopes,
            "corrections": len(corrections),
            "corrections_detail": [
                {"name": name, **{k: v for k, v in fixed.items() if k != "note"}}
                for name, fixed in sorted(
                    corrections.items(), key=lambda item: item[1].get("ts") or "", reverse=True
                )
            ],
            "excluded_pattern": marker_excluded,
            # Il ritaglio della cartella e' quello che finira' in DB_echo: e' il protagonista.
            "folder_template": {
                **{k: v for k, v in folder_template.items() if k != "path"},
                "url": (
                    f"/api/projects/{project_id}/orientation/folder_template"
                    if folder_template.get("path")
                    else None
                ),
            },
            "validation": validation,
            "marker_warning": stored.get("marker_warning") or {},
            "marker_override": (
                {
                    **{k: v for k, v in (stored.get("marker_override") or {}).items()
                       if k != "path"},
                    "url": f"/api/projects/{project_id}/orientation/folder_template",
                }
                if (stored.get("marker_override") or {}).get("path") else {}
            ),
            "template": {
                "name": summary.get("template_selected", ""),
                "vendor": summary.get("vendor", ""),
                "url": (
                    f"/api/projects/{project_id}/orientation/template"
                    if summary.get("template_selected")
                    else None
                ),
            },
            "summary": {
                key: summary.get(key)
                for key in ("status", "n_images", "review_rate", "marker_found_rate",
                            "groups_found", "dominant_resolution", "n_sugiu_corrected")
            },
            "per_image": per_image,
        }
    )


def _final_rect(project: Project) -> Optional[Dict[str, int]]:
    """The rectangle the modules saw: base plus margin, as it goes into #11."""
    value = project.step_value("rect")
    rect = value.get("rect_echo")
    if not rect:
        return None
    size = project.step_value("import").get("image_sample_size") or [0, 0]
    return expand_rect(rect, value.get("margin_percent"), width=size[0], height=size[1])


def _marker_rows(stage: Path) -> List[Dict]:
    import csv as _csv

    path = stage / "per_image_predictions.csv"
    if not path.is_file():
        return []
    out: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in _csv.DictReader(handle):
            parts = [p for p in str(row.get("marker_box_abs") or "").split("|") if p.strip()]
            box = None
            if len(parts) >= 4:
                top, left, bottom, right = (int(float(v)) for v in parts[:4])
                box = {"top": top, "left": left, "bottom": bottom, "right": right}
            out[row.get("image_id", "")] = {
                "name": row.get("image_id", ""),
                "group": row.get("pred_group", ""),
                "status": row.get("status", ""),
                "score": round(float(row.get("match_score") or 0.0), 4),
                "box": box,
                "template": row.get("template_name", ""),
                "scope": row.get("search_scope", ""),
            }
    return list(out.values())


@app.post("/api/projects/<project_id>/orientation/refine")
def api_orientation_refine(project_id: str):
    """From an approximate click to the precise marker box, with the group check."""
    project = _project(project_id)
    stage = _marker_stage_dir(project)
    payload = _payload()
    name = (payload.get("name") or "").strip()
    # Lo specchio di lavoro: e' li' che stanno le immagini su cui i moduli hanno misurato.
    folder = project.working_dir() or Path("")
    rect = _final_rect(project)
    if stage is None or not folder.is_dir() or not name or rect is None:
        return jsonify({"error": "servono la run del marker, la cartella e il rettangolo"}), 400
    try:
        x = int(payload["x"])
        y = int(payload["y"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "coordinate del click non valide"}), 400

    vendor = ((project.data.get("analysis") or {}).get("vendor") or {}).get("vendor") or ""
    current = next((row for row in _marker_rows(stage) if row["name"] == name), None)
    artifacts = _models_dir().parents[1]
    try:
        result = marker_refine.refine_from_click(
            image_path=folder / name,
            click=(x, y),
            rect=rect,
            bundle_dir=artifacts / "41_orientation_marker_detector_bundle",
            library_root=_models_dir() / "lr_marker_vendor_template_library",
            vendor=vendor,
            preferred_template=(current or {}).get("template", ""),
            window=int(payload.get("window") or 70),
        )
    except (FileNotFoundError, ImportError) as error:
        return jsonify({"error": str(error)}), 400
    if "error" in result:
        return jsonify(result), 200

    result["current"] = current
    result["group_before"] = (current or {}).get("group") or ""
    result["group_changes"] = bool(result["group_before"]) and result["group"] != result["group_before"]
    result["rect"] = rect
    return jsonify(result)


def _rebuild_orientation(project: Project, value: Dict) -> Dict:
    """Envelope e gruppo per immagine dalle detection correnti.

    Una riga per immagine, e la correzione **sostituisce** quella del modulo. Prima la
    aggiungeva: la posizione vecchia restava nell'unione, quindi correggere un marker che
    faceva da vertice non stringeva il box. E poiche' una riga sola cambia gruppo insieme
    alla posizione, correggere un marker sposta il conto anche fra i gruppi: quello che lo
    perde si stringe, quello che lo prende si allarga.
    """
    contributors = _envelope_contributors(project, value)
    discovery: List[Dict] = []
    group_of_image: Dict[str, str] = {}
    for group, rows in contributors.items():
        for row in rows:
            discovery.append({"group": group, "box": row["box"], "score": 1.0})
            group_of_image[row["name"]] = group
    return {"groups": om.envelopes(discovery, min_score=0.0), "group_of_image": group_of_image}


def _click_to_image(folder: Path, name: str, payload: Dict) -> Dict:
    """Dal click sullo schermo alle coordinate dell'immagine, usando la dimensione vera.

    Il client mandava le coordinate gia' convertite, assumendo la dimensione del campione:
    se quell'assunzione era sbagliata il punto finiva fuori dall'immagine. Ora il client
    manda dove ha cliccato e quanto e' grande l'immagine sullo schermo, e la conversione la
    fa qui, dove la dimensione reale del file e' un fatto.
    """
    path = Path(folder) / name
    try:
        from PIL import Image

        with Image.open(path) as raw:
            width, height = raw.size
    except Exception:
        width = height = 0

    display_w = float(payload.get("display_width") or 0)
    display_h = float(payload.get("display_height") or 0)
    if display_w > 0 and display_h > 0 and width and height:
        x = int(round(float(payload.get("cx", payload.get("x", 0))) * width / display_w))
        y = int(round(float(payload.get("cy", payload.get("y", 0))) * height / display_h))
        source = "convertito dal server"
    else:
        x = int(float(payload.get("x", 0)))
        y = int(float(payload.get("y", 0)))
        source = "coordinate immagine dal client"

    inside = 0 <= x < (width or 10 ** 6) and 0 <= y < (height or 10 ** 6)
    return {
        "x": max(0, min(x, (width or 1) - 1)),
        "y": max(0, min(y, (height or 1) - 1)),
        "inside": inside,
        "image_size": [width, height],
        "source": source,
        "raw": {k: payload.get(k) for k in ("x", "y", "cx", "cy", "display_width", "display_height")},
    }


# ---------------------------------------------------------------------------
# Correzione del marker in due tempi. Il click risponde subito: raffina e salva quella
# singola immagine, che e' aritmetica su una finestra di 140 px. La parte lenta (riprovare
# i ritagli su tutte le immagini della cartella) corre in background e si fonde con le
# correzioni che arrivano nel frattempo: N correzioni non fanno N passaggi, ne fanno uno.
# Aspettare la cartella intera prima di poter correggere l'immagine successiva rendeva
# questo tool piu' lento del vecchio software, che e' esattamente cio' che deve battere.
# ---------------------------------------------------------------------------

MARKER_EXCLUDED = r"Thumbs\.db|Software Release|System Info|proibite"

_consolidation: Dict[str, Dict] = {}
_consolidation_lock = threading.Lock()
_step_write_lock = threading.Lock()


def _marker_context(project: Project) -> Dict:
    """Cio' che serve a ogni operazione sul marker: run, cartella, rect, vendor, banca."""
    stage = _marker_stage_dir(project)
    rect = _final_rect(project)
    if stage is None or rect is None:
        raise ValueError("servono la run del marker e il rettangolo ecografico")
    analysis = project.data.get("analysis") or {}
    return {
        "stage": stage,
        "folder": _require_folder(project),
        "rect": rect,
        "vendor": ((analysis.get("vendor") or {}).get("vendor") or ""),
        "bundle": _models_dir().parents[1] / "41_orientation_marker_detector_bundle",
    }


def _refine_click(project: Project, name: str, click: Dict, window: int) -> Dict:
    """Dal click approssimativo alla posizione esatta, per questa sola immagine."""
    ctx = _marker_context(project)
    if not click.get("inside"):
        raise ValueError(
            f"il punto ({click['raw']}) cade fuori dall'immagine "
            f"{click['image_size'][0]}x{click['image_size'][1]}"
        )
    stored = project.step_value("orientation")
    current_row = next((r for r in _marker_rows(ctx["stage"]) if r["name"] == name), None)
    # Il ritaglio della cartella (indicato a mano o scelto dal modulo) entra in gara con la
    # banca: e' quello che ESI usera' davvero.
    propri = [
        Path(candidate) for candidate in (
            (stored.get("marker_override") or {}).get("path"),
            (stored.get("folder_template") or {}).get("path"),
        ) if candidate and Path(candidate).is_file()
    ]
    refined = marker_refine.refine_from_click(
        image_path=ctx["folder"] / name,
        click=(int(click["x"]), int(click["y"])),
        rect=ctx["rect"],
        bundle_dir=ctx["bundle"],
        library_root=_models_dir() / "lr_marker_vendor_template_library",
        vendor=ctx["vendor"],
        preferred_template=(current_row or {}).get("template", ""),
        window=window,
        own_templates=propri,
    )
    if "error" in refined:
        raise ValueError(refined["error"])
    refined["group_before"] = (current_row or {}).get("group") or ""
    refined["score_before"] = (current_row or {}).get("score")
    refined["click_source"] = click.get("source")
    refined["image_size"] = click.get("image_size")
    return refined


def _fill_blocks(value: Dict, groups: Dict) -> Dict:
    """Gruppi trovati, mancanti e i quattro blocchi di #17, dal donatore se serve."""
    order = marker_refine.GROUP_ORDER
    found = [g for g in order if g in groups]
    value.update(
        {
            "groups": groups,
            "found_groups": found,
            "missing_groups": [g for g in order if g not in groups],
            "orientation_available": {g: g in groups for g in order},
        }
    )
    if found:
        donor = found[0]
        filled, blocks = [], []
        for group in order:
            source = groups.get(group)
            if source is None:
                source = groups[donor]
                filled.append(group)
            blocks.append(
                {k: source[k] for k in ("top", "left", "bottom", "right", "check", "params")}
            )
        value["blocks"] = blocks
        value["filled_groups"] = filled
        value["filled_from"] = donor if filled else ""
    return value


def _write_orientation(
    project_id: str, mutate, status: str = "corrected", source: str = "user"
) -> Dict:
    """Read-modify-write serializzato: click e rielaborazione scrivono lo stesso step.

    Senza questo la rielaborazione, che parte da una fotografia dello step, riscriverebbe
    sopra le correzioni salvate mentre lavorava — cioe' le farebbe sparire.
    """
    with _step_write_lock:
        project = _project(project_id)
        value = mutate(project, dict(project.step_value("orientation")))
        project.codes["orientation_available"] = value.get("orientation_available") or {}
        project.set_step("orientation", value, status=status, source=source)
        return value


def _save_correction(project_id: str, name: str, refined: Dict) -> Dict:
    """La correzione e' salvata e visibile subito, senza aspettare la cartella."""

    def mutate(project: Project, value: Dict) -> Dict:
        corrections = dict(value.get("corrections") or {})
        corrections[name] = {
            "box": refined["box"],
            "group": refined["group"],
            "score": refined["score"],
            "template": refined["template"],
            "click": refined["click"],
            "group_before": refined["group_before"],
            "source": "user",
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        value["corrections"] = corrections
        # Gruppo dell'immagine ed envelope si ricalcolano ora: sono minimi e massimi su
        # dei box, costano microsecondi. Il resto lo fa la rielaborazione in background.
        rebuilt = _rebuild_orientation(project, value)
        value = _fill_blocks(value, rebuilt["groups"])
        rows = dict(value.get("validation_rows") or {})
        rows[name] = {
            "score": refined["score"], "group": refined["group"],
            "box": refined["box"], "template": "correzione",
        }
        value["validation_rows"] = rows
        return value

    return _write_orientation(project_id, mutate)


def _consolidate_marker(project_id: str, progress) -> Dict:
    """Un passaggio sulla cartella con tutte le correzioni note, non una per una."""
    project = _project(project_id)
    ctx = _marker_context(project)
    stored = dict(project.step_value("orientation"))
    corrections = dict(stored.get("corrections") or {})
    min_score = float((stored.get("validation") or {}).get("min_score") or 0.55)
    before_rows = dict(stored.get("validation_rows") or {})
    probe_folder = project.dedup_link_dir() or ctx["folder"]
    useful = [
        path for path in scan_folder(probe_folder)
        if not re.search(MARKER_EXCLUDED, str(path), re.IGNORECASE)
    ]

    progress("ritaglio i marker corretti", 0, 0)
    # Un ritaglio per correzione, tagliato una volta sola: il nome viene dall'immagine,
    # cosi' un secondo passaggio lo ritrova invece di rifarlo.
    # Solo le correzioni vive: se una viene annullata il suo ritaglio esce dalla gara,
    # altrimenti continuerebbe a essere provato su tutte le immagini per sempre.
    hint_paths = {
        name: path for name, path in (stored.get("hint_paths") or {}).items()
        if name in corrections
    }
    for name, fixed in corrections.items():
        known = hint_paths.get(name)
        if known and Path(known).is_file():
            continue
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem)[:40]
        cut = om.cut_at(
            ctx["folder"] / name, fixed["box"],
            project.root / "templates" / "candidates" / f"hint_{stem}.png",
        )
        hint_paths[name] = cut["path"]
    hints = [Path(p) for p in hint_paths.values() if Path(p).is_file()]

    progress("ricalcolo i quattro envelope", 0, 0)
    rebuilt = _rebuild_orientation(project, stored)
    groups = rebuilt["groups"]

    template_info = dict(stored.get("folder_template") or {})
    validation: Dict = {}
    if template_info.get("path") and groups:
        label = f"riprovo {len(hints) + 1} ritagli su ogni immagine"
        validation = om.validate_in_envelopes(
            images=useful, folder=probe_folder,
            template_path=Path(template_info["path"]), groups=groups,
            group_of_image=rebuilt["group_of_image"], bundle_dir=ctx["bundle"],
            min_score=min_score, extra_templates=hints,
            progress=lambda done, total: progress(label, done, total),
        )

    # Chi diventa il ritaglio consegnato si decide con la misura della validazione, dentro
    # gli envelope: vince chi aggancia piu' immagini. A ESI ne consegniamo uno solo.
    wins: Dict[str, int] = {}
    for row in (validation.get("rows") or []):
        if row.get("template"):
            wins[row["template"]] = wins.get(row["template"], 0) + 1
    best_label = max(wins, key=wins.get) if wins else ""
    promoted = ""
    if best_label and best_label != "consegnato":
        source = next((p for p in hints if p.name == best_label), None)
        if source is not None:
            final = project.root / "templates" / "orientation_marker.png"
            final.write_bytes(source.read_bytes())
            from PIL import Image as _Image

            with _Image.open(final) as image:
                size = [image.width, image.height]
            template_info = {
                **template_info, "path": str(final), "size": size,
                "promoted_from": best_label, "wins": wins.get(best_label),
                "from_correction": True,
            }
            promoted = best_label

    after_rows = {
        row["name"]: {
            "score": row["score"], "group": row["group"], "box": row["box"],
            "template": row.get("template", ""),
        }
        for row in (validation.get("rows") or [])
    }

    def mutate(project: Project, value: Dict) -> Dict:
        # Una correzione arrivata mentre giravamo non era in gara: il suo punteggio resta
        # quello del click, altrimenti la riga sembrerebbe peggiorata fino al giro dopo.
        rows = dict(after_rows)
        for name, fixed in (value.get("corrections") or {}).items():
            if name not in corrections:
                rows[name] = {
                    "score": fixed.get("score"), "group": fixed.get("group"),
                    "box": fixed.get("box"), "template": "correzione",
                }
        merged = _fill_blocks(dict(value), groups)
        merged.update(
            {
                "folder_template": template_info,
                "hint_paths": hint_paths,
                "hint_templates": [str(p) for p in hints],
                "validation": {
                    k: validation.get(k)
                    for k in ("coverage", "coverage_by_group", "images", "min_score")
                },
                "validation_rows": rows,
            }
        )
        return merged

    _write_orientation(project_id, mutate)

    recovered, worsened = [], []
    for name, after in after_rows.items():
        before = (before_rows.get(name) or {}).get("score")
        if before is None or after.get("score") is None:
            continue
        gain = after["score"] - before
        if gain > 0.02:
            recovered.append({
                "name": name, "before": before, "after": after["score"],
                "crossed": before < min_score <= after["score"],
                "template": after.get("template", ""),
            })
        elif gain < -0.02:
            worsened.append({"name": name, "before": before, "after": after["score"]})
    recovered.sort(key=lambda item: item["after"] - item["before"], reverse=True)
    return {
        "corrections": len(corrections),
        "hints": len(hints),
        "threshold": min_score,
        "review_before": sum(1 for r in before_rows.values() if (r.get("score") or -1) < min_score),
        "review_after": sum(1 for r in after_rows.values() if (r.get("score") or -1) < min_score),
        "recovered": recovered[:8],
        "recovered_count": len(recovered),
        "crossed_count": sum(1 for item in recovered if item.get("crossed")),
        "worsened": worsened[:5],
        "worsened_count": len(worsened),
        "wins": wins,
        "promoted": promoted,
        "coverage": validation.get("coverage"),
        "coverage_by_group": validation.get("coverage_by_group") or {},
        "template_size": template_info.get("size"),
        "ts": datetime.now().isoformat(timespec="seconds"),
    }


def _consolidation_state(project_id: str) -> Dict:
    with _consolidation_lock:
        return dict(_consolidation.get(project_id) or {"state": "idle", "runs": 0, "queued": 0})


def _consolidation_worker(project_id: str) -> None:
    """Un solo lavoratore per progetto, che si fonde: le correzioni arrivate durante un
    passaggio non ne accodano altrettanti, ne fanno ripartire uno con tutte dentro."""

    def progress(stage: str, done: int = 0, total: int = 0) -> None:
        with _consolidation_lock:
            state = _consolidation.get(project_id)
            if state is not None:
                state.update({"stage": stage, "done": done, "total": total})

    while True:
        try:
            result = _consolidate_marker(project_id, progress)
            with _consolidation_lock:
                state = _consolidation[project_id]
                state.update({"last": result, "error": "", "runs": state.get("runs", 0) + 1})
        except Exception as error:  # noqa: BLE001
            with _consolidation_lock:
                state = _consolidation[project_id]
                state.update({"error": str(error), "runs": state.get("runs", 0) + 1})
        with _consolidation_lock:
            state = _consolidation[project_id]
            if state.get("pending"):
                state.update({"pending": False, "queued": 0,
                              "stage": "riparto con le correzioni nuove"})
                continue
            state.update({"state": "idle", "stage": "", "done": 0, "total": 0, "queued": 0})
            return


def _schedule_consolidation(project_id: str) -> Dict:
    with _consolidation_lock:
        state = _consolidation.setdefault(project_id, {"state": "idle", "runs": 0})
        if state.get("state") == "running":
            state["pending"] = True
            state["queued"] = state.get("queued", 0) + 1
            return dict(state)
        state.update({"state": "running", "stage": "avvio", "pending": False,
                      "queued": 0, "done": 0, "total": 0})
    threading.Thread(target=_consolidation_worker, args=(project_id,), daemon=True).start()
    return _consolidation_state(project_id)


@app.post("/api/projects/<project_id>/orientation/quick_fix")
def api_orientation_quick_fix(project_id: str):
    """Il click risponde subito; la cartella si rielabora dopo, senza bloccare nulla."""
    project = _project(project_id)
    payload = _payload()
    name = (payload.get("name") or "").strip()
    if not name:
        return jsonify({"error": "manca l'immagine"}), 400
    # Lo specchio di lavoro: e' l'immagine che l'utente ha davanti, ed e' la sua dimensione
    # quella con cui va convertito il click.
    click = _click_to_image(project.working_dir() or Path(""), name, payload)
    try:
        refined = _refine_click(project, name, click, int(payload.get("window") or 70))
    except ValueError as error:
        return jsonify({"error": str(error), "click": click}), 400
    value = _save_correction(project_id, name, refined)
    return jsonify(
        {
            "saved": True,
            "click": click,
            "refined": refined,
            "corrections": len(value.get("corrections") or {}),
            "groups": {
                group: {k: box.get(k) for k in ("top", "left", "bottom", "right", "markers")}
                for group, box in (value.get("groups") or {}).items()
            },
            "missing_groups": value.get("missing_groups") or [],
            "background": _schedule_consolidation(project_id),
        }
    )


@app.get("/api/projects/<project_id>/orientation/consolidation")
def api_orientation_consolidation(project_id: str):
    """Come va la rielaborazione in background, e com'e' andato l'ultimo passaggio."""
    _project(project_id)
    return jsonify(_consolidation_state(project_id))


def _marker_warning(rows: List[Dict], groups: Dict) -> Dict:
    """Il marker sbagliato ha una firma: non si muove.

    Un marker di orientamento **deve** cambiare posizione col gruppo. Se il candidato
    scelto cade sempre nello stesso punto, o assegna un solo gruppo a tutta la cartella,
    quello che ha trovato e' un elemento fisso dell'interfaccia (un logo, l'ora, una
    scritta di intestazione), non il marker. E' successo su una cartella GE: la scritta
    "RGP" dell'intestazione matchava a 0.6106 identico su tutte le 136 immagini.
    """
    utili = [r for r in rows if r.get("box")]
    if len(utili) < 8:
        return {}
    posizioni = {(r["box"]["top"], r["box"]["left"]) for r in utili}
    punteggi = {round(float(r.get("score") or 0), 4) for r in utili}
    motivi = []
    if len(posizioni) == 1:
        motivi.append(
            f"cade sempre nello stesso punto su {len(utili)} immagini "
            f"(top|left {utili[0]['box']['top']}|{utili[0]['box']['left']})"
        )
    if len(punteggi) == 1:
        motivi.append(f"ha punteggio identico ({punteggi.pop()}) su tutte le immagini")
    if len(groups) == 1 and len(utili) >= 20:
        motivi.append(f"assegna un solo gruppo ({next(iter(groups))}) a tutta la cartella")
    if not motivi:
        return {}
    return {
        "static": True,
        "reasons": motivi,
        "message": "il marker trovato non si muove: probabilmente e' un elemento fisso "
                   "dell'interfaccia, non il marker di orientamento",
    }


def _inside_rect(box: Dict[str, int], rect: Dict[str, int], tol: int = 8) -> bool:
    """Il centro del marker cade dentro il rettangolo ecografico (con un filo di tolleranza)."""
    cx = (int(box["left"]) + int(box["right"])) / 2.0
    cy = (int(box["top"]) + int(box["bottom"])) / 2.0
    return (rect["left"] - tol <= cx <= rect["right"] + tol
            and rect["top"] - tol <= cy <= rect["bottom"] + tol)


def _drop_outliers(rows: List[Dict], rect: Dict[str, int]) -> Tuple[List[Dict], int]:
    """Via i match fuori dal rettangolo, ma solo se sono l'eccezione.

    Un marker che casca fuori dal rect, quando tutti gli altri stanno dentro, e' un falso
    match: su una cartella GE uno di questi (in basso a destra, sotto il rect) vinceva
    0.649 contro 0.63 del marker vero e allargava l'envelope del gruppo fino a includerlo,
    portandosi dietro anche la riassegnazione. Se invece la maggioranza sta fuori — succede
    nei setup storici 80 e 81, dove il marker e' fuori dall'immagine ecografica — non si
    butta niente: e' la cartella a dire dove sta il marker, non noi.
    """
    dentro = [row for row in rows if row.get("box") and _inside_rect(row["box"], rect)]
    if dentro and len(dentro) >= 0.8 * len(rows):
        return dentro, len(rows) - len(dentro)
    return rows, 0


def _run_marker_override(
    job_id: str, project_id: str, name: str, box: Dict[str, int], min_score: float
) -> None:
    """Il marker indicato a mano diventa IL marker della cartella, e il modulo riparte.

    Non passa dalla banca: quando la banca ha scelto il glifo sbagliato, ripescarla
    significherebbe ritrovare lo stesso errore. Il ritaglio e' quello che l'utente ha
    disegnato, e da lui si rifanno posizioni, gruppi, envelope e validazione.
    """
    try:
        project = _project(project_id)
        ctx = _marker_context(project)
        stored = dict(project.step_value("orientation"))
        probe_folder = project.dedup_link_dir() or ctx["folder"]
        useful = [
            path for path in scan_folder(probe_folder)
            if not re.search(MARKER_EXCLUDED, str(path), re.IGNORECASE)
        ]

        _job_update(job_id, stage="1/4 ritaglio il marker che hai indicato")
        crop = om.cut_at(
            ctx["folder"] / name, box, project.root / "templates" / "marker_override.png"
        )

        _job_update(job_id, stage=f"2/4 lo cerco su tutte le {len(useful)} immagini", total=len(useful))
        found = om.match_all(
            images=useful, folder=probe_folder, template_path=Path(crop["path"]),
            rect=ctx["rect"], bundle_dir=ctx["bundle"], min_score=min_score,
            search_margin=60,
            progress=lambda done, total: _job_update(
                job_id, stage=f"2/4 lo cerco su tutte le {total} immagini",
                done=done, total=total),
        )
        rows = found.get("rows") or []
        agganciate = [r for r in rows if r.get("box") and (r.get("score") or 0) >= min_score]
        if not agganciate:
            _job_update(
                job_id, status="error", stage="errore",
                error=(f"il ritaglio indicato non si ritrova in nessuna immagine "
                       f"(soglia {min_score}): prova a disegnarlo piu' stretto sul glifo"),
            )
            return

        _job_update(job_id, stage="3/5 ricostruisco i quattro envelope dalle posizioni trovate")
        agganciate, fuori_rect = _drop_outliers(agganciate, ctx["rect"])
        groups_liberi = om.envelopes(
            [{"group": r["group"], "box": r["box"], "score": r["score"]} for r in agganciate],
            min_score=min_score,
        )
        gruppi_liberi = {r["name"]: r["group"] for r in agganciate}

        # Secondo passaggio. La ricerca libera dentro il rect prende, su qualche immagine,
        # un falso match lontano dal marker (su una cartella GE: un angolo in basso a
        # destra che vinceva 0.649 contro 0.63 del marker vero, e quell'immagine finiva nel
        # gruppo sbagliato). Cercare invece **dentro i quattro envelope** e tenere il
        # migliore riassegna il gruppo dove il marker puo' stare davvero — ed e' anche
        # quello che fara' ESI.
        _job_update(job_id, stage="4/5 riassegno i gruppi cercando dentro i quattro envelope")
        validation = om.validate_in_envelopes(
            images=useful, folder=probe_folder, template_path=Path(crop["path"]),
            groups=groups_liberi, group_of_image={}, bundle_dir=ctx["bundle"],
            min_score=min_score,
            progress=lambda done, total: _job_update(
                job_id, stage="4/5 riassegno i gruppi cercando dentro i quattro envelope",
                done=done, total=total),
        )
        riassegnate, fuori_rect_2 = _drop_outliers(
            [row for row in (validation.get("rows") or [])
             if row.get("box") and (row.get("score") or 0) >= min_score],
            ctx["rect"],
        )
        _job_update(job_id, stage="5/5 envelope definitivi dalle posizioni riassegnate")
        groups = om.envelopes(
            [{"group": r["group"], "box": r["box"], "score": r["score"]} for r in riassegnate],
            min_score=min_score,
        ) or groups_liberi
        agganciate = riassegnate or agganciate
        spostate = sum(
            1 for r in riassegnate if gruppi_liberi.get(r["name"], r["group"]) != r["group"]
        )

        gruppi_per_immagine: Dict[str, int] = {}
        for row in agganciate:
            gruppi_per_immagine[row["group"]] = gruppi_per_immagine.get(row["group"], 0) + 1

        def mutate(project: Project, value: Dict) -> Dict:
            merged = _fill_blocks(dict(value), groups)
            merged.update(
                {
                    # Le correzioni fatte sul marker sbagliato non valgono piu': indicavano
                    # dov'era un glifo che non stiamo piu' cercando. Si buttano, dicendolo.
                    "corrections": {},
                    "hint_paths": {},
                    "hint_templates": [],
                    "marker_override": {
                        "path": str(crop["path"]), "size": crop["size"],
                        "source_image": name, "box": dict(box),
                        "ts": datetime.now().isoformat(timespec="seconds"),
                    },
                    "folder_template": {
                        "path": str(crop["path"]), "size": crop["size"],
                        "source_image": name, "source_box": dict(box),
                        "coverage": validation.get("coverage"),
                        "median": (validation.get("coverage_by_group") or {}),
                        "from_user_marker": True,
                    },
                    "marker_warning": _marker_warning(agganciate, groups),
                    "validation": {
                        k: validation.get(k)
                        for k in ("coverage", "coverage_by_group", "images", "min_score")
                    },
                    "validation_rows": {
                        row["name"]: {
                            "score": row["score"], "group": row["group"], "box": row["box"],
                            "template": row.get("template", ""),
                        }
                        for row in (validation.get("rows") or [])
                    },
                    "marker_rows_override": {
                        row["name"]: {"score": row["score"], "group": row["group"],
                                      "box": row["box"]}
                        for row in agganciate
                    },
                }
            )
            return merged

        value = _write_orientation(project_id, mutate)
        _job_update(
            job_id, status="done", stage="fatto",
            result={
                "crop": crop,
                "matched": len(agganciate),
                "images": len(useful),
                "groups_first_pass": [g for g in marker_refine.GROUP_ORDER if g in groups_liberi],
                "reassigned": spostate,
                "dropped_outside_rect": fuori_rect + fuori_rect_2,
                "groups_found": value.get("found_groups") or [],
                "missing_groups": value.get("missing_groups") or [],
                "by_group": gruppi_per_immagine,
                "coverage": validation.get("coverage"),
                "coverage_by_group": validation.get("coverage_by_group") or {},
                "warning": value.get("marker_warning") or {},
                "corrections_dropped": len(stored.get("corrections") or {}),
                "legacy_size_hint": (
                    "" if 10 <= crop["size"][0] <= 40 and 10 <= crop["size"][1] <= 40
                    else f"i ritagli orientation_*.png delle configurazioni storiche vanno da "
                         f"16x18 a 30x31: questo e' {crop['size'][0]}x{crop['size'][1]}"
                ),
            },
        )
    except Exception as error:  # noqa: BLE001
        _job_update(job_id, status="error", stage="errore", error=str(error))


# Un envelope e' l'unione dei box del gruppo: quattro immagini, una per lato, ne fissano
# l'estensione. Vederle serve a due cose: capire perche' un box e' grande, e accorgersi
# quando e' una detection sbagliata a gonfiarlo (su GE un falso match sotto il rect
# allargava l'envelope LRUD di 200 px, e con esso il gruppo assegnato a un'altra immagine).
_ENVELOPE_SIDES = (
    ("top", "alto", False),
    ("left", "sinistra", False),
    ("bottom", "basso", True),
    ("right", "destra", True),
)


def _envelope_contributors(project: Project, value: Dict) -> Dict[str, List[Dict]]:
    """Le detection che hanno formato ogni envelope, per gruppo, col nome dell'immagine."""
    stage = _marker_stage_dir(project)
    override = value.get("marker_rows_override") or {}
    if override:
        rows = [
            {"name": name, "group": row.get("group"), "box": row.get("box"),
             "score": row.get("score"), "source": "marker indicato a mano"}
            for name, row in override.items() if row.get("box") and row.get("group")
        ]
    elif stage is not None:
        rows = [
            {"name": row["name"], "group": row["group"], "box": row["box"],
             "score": row.get("score"), "source": row.get("template") or "banca"}
            for row in _marker_rows(stage)
            if row.get("box") and str(row.get("status") or "").lower() == "ok"
        ]
    else:
        rows = []
    per_name = {row["name"]: row for row in rows}
    for name, fixed in (value.get("corrections") or {}).items():
        if fixed.get("box") and fixed.get("group"):
            per_name[name] = {
                "name": name, "group": fixed["group"], "box": fixed["box"],
                "score": fixed.get("score"), "source": "correzione", "corrected": True,
            }
    out: Dict[str, List[Dict]] = {}
    for row in per_name.values():
        out.setdefault(row["group"], []).append(row)
    return out


def _envelope_limits(rows: List[Dict], validation_rows: Optional[Dict] = None) -> Dict:
    """Chi fissa ciascun lato, di quanto si stringerebbe togliendolo, e quanto e' isolato."""
    centri_x = sorted(((r["box"]["left"] + r["box"]["right"]) / 2.0) for r in rows)
    centri_y = sorted(((r["box"]["top"] + r["box"]["bottom"]) / 2.0) for r in rows)
    mediana_x = centri_x[len(centri_x) // 2]
    mediana_y = centri_y[len(centri_y) // 2]

    limiti: Dict[str, Dict] = {}
    stretto: Dict[str, int] = {}
    for side, etichetta, massimo in _ENVELOPE_SIDES:
        ordinati = sorted(rows, key=lambda r: r["box"][side], reverse=massimo)
        estremo = ordinati[0]["box"][side]
        # il secondo valore *diverso*: e' quanto misurerebbe il lato senza chi lo fissa
        secondo = next((r["box"][side] for r in ordinati if r["box"][side] != estremo), estremo)
        insieme = [r for r in ordinati if r["box"][side] == estremo]
        prima = insieme[0]
        limiti[side] = {
            "side": etichetta,
            "value": estremo,
            "second": secondo,
            "gain": abs(secondo - estremo),
            "shared": len(insieme),
            "image": prima["name"],
            "score": prima.get("score"),
            "box": prima["box"],
            "source": prima.get("source") or "",
            "corrected": bool(prima.get("corrected")),
            # Due punteggi, e sono due cose diverse: quello della **scoperta** ha deciso
            # questa posizione (ed e' quello che dice se il bordo e' fidato), quello della
            # **validazione** e' il match del ritaglio consegnato dentro l'envelope, cioe'
            # quello che si vede accanto all'immagine nel resto della pagina.
            "score_validation": (
                (validation_rows or {}).get(prima["name"], {}).get("score")
            ),
            # quanto e' lontano dal grosso del gruppo: un estremo isolato e' sospetto
            "distance_from_median": int(round(max(
                abs((prima["box"]["left"] + prima["box"]["right"]) / 2.0 - mediana_x),
                abs((prima["box"]["top"] + prima["box"]["bottom"]) / 2.0 - mediana_y),
            ))),
            "others": [
                {"name": r["name"], "score": r.get("score")} for r in insieme[1:4]
            ],
        }
        stretto[side] = secondo
    return {
        "markers": len(rows),
        "limits": limiti,
        "median_centre": [int(round(mediana_x)), int(round(mediana_y))],
        "tight": stretto,
    }


def _bank_vendor_dir(vendor: str) -> Path:
    """La cartella della banca per questo vendor, con il nome che ha davvero sul disco.

    I nomi non seguono la maiuscola del riconoscimento: la rete dice `GE`, la banca ha
    `Ge`. Si cerca senza badare a maiuscole e spazi; se non c'e' si crea col nome
    capitalizzato, come le altre.
    """
    library = _models_dir() / "lr_marker_vendor_template_library"
    chiave = re.sub(r"[^a-z0-9]", "", (vendor or "").lower())
    if not chiave:
        raise ValueError("vendor non riconosciuto: la banca e' organizzata per vendor")
    if library.is_dir():
        for candidate in sorted(library.iterdir()):
            if candidate.is_dir() and re.sub(r"[^a-z0-9]", "", candidate.name.lower()) == chiave:
                return candidate
    return library / vendor.strip().title()


@app.get("/api/projects/<project_id>/orientation/bank")
def api_orientation_bank(project_id: str):
    """Cosa c'e' in banca per questo vendor, e cosa verrebbe aggiunto."""
    project = _project(project_id)
    vendor = ((project.data.get("analysis") or {}).get("vendor") or {}).get("vendor") or ""
    value = project.step_value("orientation")
    shipped = (value.get("folder_template") or {}).get("path")
    try:
        vendor_dir = _bank_vendor_dir(vendor)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    esistenti = sorted(vendor_dir.glob("marker_*.png")) if vendor_dir.is_dir() else []
    digest_presenti = {}
    for path in esistenti:
        digest_presenti[hashlib.sha1(path.read_bytes()).hexdigest()] = path.name
    proprio = ""
    dimensione = None
    if shipped and Path(shipped).is_file():
        proprio = hashlib.sha1(Path(shipped).read_bytes()).hexdigest()
        from PIL import Image

        with Image.open(shipped) as raw:
            dimensione = [raw.width, raw.height]
    return jsonify(
        {
            "vendor": vendor,
            "dir": str(vendor_dir),
            "exists": vendor_dir.is_dir(),
            "templates": len(esistenti),
            "next_name": f"marker_{len(esistenti) + 1:03d}.png",
            "shipped": {
                "available": bool(proprio),
                "size": dimensione,
                "already_in_bank": digest_presenti.get(proprio, ""),
                "from_user_marker": bool((value.get("folder_template") or {}).get("from_user_marker")),
                "source_image": (value.get("folder_template") or {}).get("source_image", ""),
            },
        }
    )


@app.post("/api/projects/<project_id>/orientation/bank")
def api_orientation_bank_add(project_id: str):
    """Versa il ritaglio di questa cartella nella banca storica del suo vendor.

    E' il modo in cui una configurazione fatta a mano insegna qualcosa alla prossima: la
    banca e' costruita dagli `orientation_*.png` delle configurazioni storiche, e questo
    ritaglio e' esattamente la stessa cosa — il glifo vero di un'acquisizione. Si scrive
    con la convenzione della banca (`marker_NNN.png` piu' la riga nei manifest), e non si
    aggiunge due volte lo stesso file: il digest lo dice.
    """
    project = _project(project_id)
    payload = _payload()
    vendor = (
        payload.get("vendor")
        or ((project.data.get("analysis") or {}).get("vendor") or {}).get("vendor")
        or ""
    )
    value = project.step_value("orientation")
    name = (payload.get("name") or "").strip()
    if name:
        # il ritaglio di un'immagine precisa, tagliato dove si vede il marker
        box = _current_marker_box(project, name)
        folder = _require_folder(project)
        if box is None:
            return jsonify({"error": "questa immagine non ha un marker"}), 400
        sorgente = om.cut_at(
            folder / name, box, project.root / "templates" / "bank_candidate.png"
        )["path"]
        origine = str((folder / name).resolve())
        box_origine = box
    else:
        sorgente = (value.get("folder_template") or {}).get("path") or ""
        origine = (value.get("folder_template") or {}).get("source_image") or ""
        box_origine = (value.get("folder_template") or {}).get("source_box") or {}
        if origine:
            origine = str((_require_folder(project) / origine).resolve())
    if not sorgente or not Path(sorgente).is_file():
        return jsonify({"error": "nessun ritaglio da versare: manca il template della cartella"}), 400

    try:
        vendor_dir = _bank_vendor_dir(vendor)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    dati = Path(sorgente).read_bytes()
    digest = hashlib.sha1(dati).hexdigest()
    vendor_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(vendor_dir.glob("marker_*.png")):
        if hashlib.sha1(path.read_bytes()).hexdigest() == digest:
            return jsonify(
                {
                    "added": False,
                    "reason": f"questo ritaglio e' gia' in banca come {path.name}",
                    "vendor": vendor_dir.name,
                    "existing": path.name,
                }
            ), 409

    from PIL import Image
    import statistics as _stats

    with Image.open(sorgente) as raw:
        grigio = raw.convert("L")
        larghezza, altezza = grigio.width, grigio.height
        pixel = list(grigio.getdata())
    media = sum(pixel) / len(pixel)
    scarto = _stats.pstdev(pixel) if len(pixel) > 1 else 0.0

    esistenti = sorted(vendor_dir.glob("marker_*.png"))
    numero = 1 + max(
        (int(re.sub(r"[^0-9]", "", path.stem) or 0) for path in esistenti), default=0
    )
    destinazione = vendor_dir / f"marker_{numero:03d}.png"
    destinazione.write_bytes(dati)

    voce = {
        "vendor": vendor_dir.name,
        "template_path": str(destinazione),
        "source_path": origine,
        "config_dir": str(_require_folder(project)),
        "width": larghezza,
        "height": altezza,
        "mean": media,
        "std": scarto,
        "digest": digest,
        "best_similarity_to_kept": -1.0,
        # da dove viene: una configurazione fatta con questa app, non una storica
        "added_by": "esibuilder_ai",
        "project_id": project_id,
        "source_box": box_origine or None,
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }

    manifest = vendor_dir / "manifest.json"
    voci = []
    if manifest.is_file():
        try:
            voci = json.loads(manifest.read_text())
        except Exception:  # noqa: BLE001
            voci = []
    if not isinstance(voci, list):
        voci = []
    voci.append(voce)
    manifest.write_text(json.dumps(voci, indent=2, ensure_ascii=False))

    globale = vendor_dir.parent / "manifest.csv"
    if globale.is_file():
        import csv as _csv

        with globale.open("r", encoding="utf-8", newline="") as handle:
            intestazione = next(_csv.reader(handle), [])
        if intestazione:
            with globale.open("a", encoding="utf-8", newline="") as handle:
                _csv.writer(handle).writerow(
                    [voce.get(colonna, "") for colonna in intestazione]
                )

    return jsonify(
        {
            "added": True,
            "vendor": vendor_dir.name,
            "name": destinazione.name,
            "path": str(destinazione),
            "size": [larghezza, altezza],
            "templates": len(voci),
            "source_image": origine,
        }
    )


@app.get("/api/projects/<project_id>/orientation/limits")
def api_orientation_limits(project_id: str):
    """Per ogni orientamento: i marker ai vertici, quelli che hanno deciso l'estensione."""
    project = _project(project_id)
    value = project.step_value("orientation")
    groups = value.get("groups") or {}
    contributors = _envelope_contributors(project, value)
    out = {}
    for group in marker_refine.GROUP_ORDER:
        rows = contributors.get(group) or []
        if not rows:
            continue
        envelope = groups.get(group) or {}
        misura = _envelope_limits(rows, value.get("validation_rows") or {})
        stretto = misura["tight"]
        out[group] = {
            **misura,
            "envelope": {k: envelope.get(k) for k in ("top", "left", "bottom", "right")},
            "size": (
                [envelope["right"] - envelope["left"] + 1, envelope["bottom"] - envelope["top"] + 1]
                if envelope else None
            ),
            "tight_size": [
                stretto["right"] - stretto["left"] + 1, stretto["bottom"] - stretto["top"] + 1
            ],
        }
    return jsonify({"groups": out, "order": list(marker_refine.GROUP_ORDER)})


@app.post("/api/projects/<project_id>/orientation/marker_override")
def api_orientation_marker_override(project_id: str):
    """Segnala che il marker trovato e' sbagliato e indica quello giusto, con un rettangolo."""
    project = _project(project_id)
    payload = _payload()
    name = (payload.get("name") or "").strip()
    if not name:
        return jsonify({"error": "manca l'immagine"}), 400
    box = payload.get("box") or {}
    if not all(side in box for side in ("top", "left", "bottom", "right")):
        # Il rettangolo arriva in pixel dello schermo, come il click: la conversione la fa
        # il server, che conosce la dimensione vera del file.
        folder = project.working_dir() or Path("")
        angoli = [
            _click_to_image(folder, name, {
                "cx": payload.get(f"x{i}"), "cy": payload.get(f"y{i}"),
                "display_width": payload.get("display_width"),
                "display_height": payload.get("display_height"),
            })
            for i in (1, 2)
        ]
        if not all(a.get("image_size", [0])[0] for a in angoli):
            return jsonify({"error": "serve il rettangolo del marker"}), 400
        box = {
            "top": min(a["y"] for a in angoli), "left": min(a["x"] for a in angoli),
            "bottom": max(a["y"] for a in angoli), "right": max(a["x"] for a in angoli),
        }
    box = {side: int(round(float(box[side]))) for side in ("top", "left", "bottom", "right")}
    if box["right"] - box["left"] < 6 or box["bottom"] - box["top"] < 6:
        return jsonify({"error": "il rettangolo e' troppo piccolo: almeno 6x6 px"}), 400
    stored = project.step_value("orientation")
    min_score = float(
        payload.get("min_score") or (stored.get("validation") or {}).get("min_score") or 0.55
    )
    return jsonify(
        {"job_id": _start_job(_run_marker_override, project_id, name, box, min_score)}
    )


@app.post("/api/projects/<project_id>/orientation/reprocess")
def api_orientation_reprocess(project_id: str):
    """Rielabora ora la cartella con le correzioni che ci sono, senza toccarne altre."""
    _project(project_id)
    return jsonify(_schedule_consolidation(project_id))


@app.post("/api/projects/<project_id>/orientation/correct")
def api_orientation_correct(project_id: str):
    """Salva un marker corretto dandone il box, e ricalcola gli envelope di conseguenza.

    Stessa strada della correzione col click (`quick_fix`): la correzione sostituisce la
    detection dell'immagine, i quattro envelope si rifanno subito, e il resto della
    cartella si rielabora in background.
    """
    project = _project(project_id)
    payload = _payload()
    name = (payload.get("name") or "").strip()
    box = payload.get("box") or {}
    group = (payload.get("group") or "").strip().upper()
    if not name or group not in marker_refine.GROUP_ORDER:
        return jsonify({"error": "correzione incompleta"}), 400
    if not all(side in box for side in ("top", "left", "bottom", "right")):
        return jsonify({"error": "box incompleto"}), 400

    def mutate(project: Project, value: Dict) -> Dict:
        corrections = dict(value.get("corrections") or {})
        corrections[name] = {
            "box": {side: int(box[side]) for side in ("top", "left", "bottom", "right")},
            "group": group,
            "score": payload.get("score"),
            "template": payload.get("template", ""),
            "note": payload.get("note", ""),
            "click": payload.get("click"),
            "group_before": payload.get("group_before", ""),
            "source": "user",
            "ts": payload.get("ts") or datetime.now().isoformat(timespec="seconds"),
        }
        value["corrections"] = corrections
        rebuilt = _rebuild_orientation(project, value)
        value = _fill_blocks(value, rebuilt["groups"])
        rows = dict(value.get("validation_rows") or {})
        rows[name] = {
            "score": payload.get("score"), "group": group,
            "box": corrections[name]["box"], "template": "correzione",
        }
        value["validation_rows"] = rows
        return value

    value = _write_orientation(project_id, mutate)
    return jsonify(
        {
            "saved": True,
            "corrections": len(value.get("corrections") or {}),
            "groups": {
                g: {k: b.get(k) for k in ("top", "left", "bottom", "right", "markers")}
                for g, b in (value.get("groups") or {}).items()
            },
            "missing_groups": value.get("missing_groups") or [],
            "background": _schedule_consolidation(project_id),
        }
    )


@app.post("/api/projects/<project_id>/orientation/correction/delete")
def api_orientation_correction_delete(project_id: str):
    """Annulla una correzione: l'immagine torna a quello che diceva il modulo.

    Anche qui gli envelope si rifanno con la regola di sempre — una riga per immagine, e
    senza quella correzione il gruppo che la conteneva torna com'era. Il ritaglio nato da
    lei esce dalla gara dei template.
    """
    project = _project(project_id)
    name = (_payload().get("name") or "").strip()
    presenti = dict(project.step_value("orientation").get("corrections") or {})
    if name not in presenti:
        return jsonify({"error": "correzione non trovata"}), 404
    restanti = len(presenti) - 1

    def mutate(project: Project, value: Dict) -> Dict:
        corrections = dict(value.get("corrections") or {})
        corrections.pop(name, None)
        value["corrections"] = corrections
        value["hint_paths"] = {
            key: path for key, path in (value.get("hint_paths") or {}).items() if key != name
        }
        rebuilt = _rebuild_orientation(project, value)
        value = _fill_blocks(value, rebuilt["groups"])
        # il punteggio dell'immagine lo rifa' la rielaborazione, con gli envelope nuovi
        value["validation_rows"] = {
            key: row for key, row in (value.get("validation_rows") or {}).items()
            if key != name
        }
        return value

    value = _write_orientation(
        project_id, mutate,
        status="corrected" if restanti else "proposed",
        source="user" if restanti else "model",
    )
    return jsonify(
        {
            "removed": name,
            "corrections": len(value.get("corrections") or {}),
            "groups": {
                g: {k: b.get(k) for k in ("top", "left", "bottom", "right", "markers")}
                for g, b in (value.get("groups") or {}).items()
            },
            "background": _schedule_consolidation(project_id),
        }
    )


def _current_marker_box(project: Project, name: str) -> Optional[Dict[str, int]]:
    """Il box del marker che si vede per questa immagine, con la stessa priorita' di
    `/orientation`: la correzione dell'utente, poi la validazione (dove il ritaglio
    consegnato viene trovato dentro l'envelope), poi il marker indicato a mano, poi la
    scoperta del batch. Serve a tenere allineati il rettangolo disegnato, il ritaglio
    mostrato e quello che finisce in banca."""
    value = project.step_value("orientation")
    fixed = (value.get("corrections") or {}).get(name)
    if fixed and fixed.get("box"):
        return {side: int(fixed["box"][side]) for side in ("top", "left", "bottom", "right")}
    for sorgente in ("validation_rows", "marker_rows_override"):
        row = (value.get(sorgente) or {}).get(name)
        if row and row.get("box"):
            return {side: int(row["box"][side]) for side in ("top", "left", "bottom", "right")}
    stage = _marker_stage_dir(project)
    if stage is not None:
        riga = next((r for r in _marker_rows(stage) if r["name"] == name), None)
        if riga and riga.get("box"):
            return {side: int(riga["box"][side]) for side in ("top", "left", "bottom", "right")}
    return None


@app.get("/api/projects/<project_id>/orientation/crop")
def api_orientation_crop(project_id: str):
    """The crop of one acquisition at its marker box.

    This is the glyph the configuration actually has to carry: a legacy setup folder holds
    `DB_echo/setup_NN/orientation_0..3.png`, and the vendor bank was built exactly from those.
    The bank template is only the reference that matched.
    """
    project = _project(project_id)
    stage = _marker_stage_dir(project)
    # Lo specchio di lavoro: e' li' che stanno le immagini su cui i moduli hanno misurato.
    folder = project.working_dir() or Path("")
    name = request.args.get("name") or ""
    if stage is None or not folder.is_dir() or not name:
        return jsonify({"error": "ritaglio non disponibile"}), 404

    box = _current_marker_box(project, name)
    if box is None:
        return jsonify({"error": "nessun marker in questa immagine"}), 404
    box = [box["top"], box["left"], box["bottom"], box["right"]]

    path = (folder / name).resolve()
    if not path.is_file():
        return jsonify({"error": "immagine non trovata"}), 404

    scale = max(1, min(16, int(request.args.get("scale") or 6)))
    pad = max(0, min(40, int(request.args.get("pad") or 0)))
    try:
        from PIL import Image

        with Image.open(path) as raw:
            image = raw.convert("RGB")
            top, left, bottom, right = box
            crop = image.crop(
                (
                    max(0, left - pad), max(0, top - pad),
                    min(image.width, right + pad), min(image.height, bottom + pad),
                )
            )
            if scale > 1:
                crop = crop.resize(
                    (crop.width * scale, crop.height * scale), resample=Image.NEAREST
                )
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")
    except Exception as error:  # noqa: BLE001
        return jsonify({"error": str(error)}), 500


@app.get("/api/projects/<project_id>/orientation/folder_template")
def api_orientation_folder_template(project_id: str):
    """Il ritaglio della cartella: il PNG che andra' in DB_echo/setup_NN/orientation_*.png."""
    project = _project(project_id)
    raw = (project.step_value("orientation").get("folder_template") or {}).get("path")
    if not raw or not Path(raw).is_file():
        return jsonify({"error": "ritaglio della cartella non ancora prodotto"}), 404
    return send_file(Path(raw), mimetype="image/png")


@app.get("/api/projects/<project_id>/orientation/template")
def api_orientation_template(project_id: str):
    """The marker template picked from the vendor bank, as an image."""
    project = _project(project_id)
    stage = _marker_stage_dir(project)
    if stage is None:
        return jsonify({"error": "nessuna run del marker"}), 404
    import csv as _csv

    summary_path = stage / "folder_summary.csv"
    if not summary_path.is_file():
        return jsonify({"error": "summary del marker assente"}), 404
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary = next(iter(_csv.DictReader(handle)), {})
    name = (summary.get("template_selected") or "").strip()
    vendor = (summary.get("vendor") or "").strip()
    if not name or not vendor:
        return jsonify({"error": "template non registrato"}), 404

    library = _models_dir() / "lr_marker_vendor_template_library"
    # The bank folder is not always spelled like the vendor class (GE lives under "Ge").
    for candidate in (vendor, vendor.capitalize(), vendor.upper(), vendor.lower()):
        path = library / candidate / name
        if path.is_file():
            return send_file(path, mimetype="image/png")
    return jsonify({"error": f"template {name} non trovato per {vendor}"}), 404


# ---------------------------------------------------------------------------
# La catena del rettangolo ecografico.
#
# I moduli si tengono per mano: l'orientamento si studia DENTRO il rettangolo, ma il
# rettangolo si rifinisce CON l'orientamento; il piano L/T si riconosce dentro il
# rettangolo e serve al rettangolo per cercare il segmento piu' lungo nel piano giusto.
# Non e' un cerchio vizioso se lo si percorre a giri: un rettangolo provvisorio basta a far
# girare orientamento e piano, e con quelli il rettangolo si stringe. Ogni giro si mostra
# all'utente con i numeri prima e dopo, e si applica solo se lui conferma.
# ---------------------------------------------------------------------------

RECT_PASSES = (
    {
        "id": "rete",
        "label": "1 · rettangolo dalla rete",
        "needs": (),
        "what": "mediana dei box predetti su un campione di immagini: basta per far girare "
                "orientamento e piano dentro un rettangolo provvisorio",
    },
    {
        "id": "specularita",
        "label": "2 · specularita' sui quattro orientamenti",
        "needs": ("orientation",),
        "what": "un rettangolo per gruppo, poi il confronto fra i gruppi: identici o "
                "speculari rispetto al centro dell'immagine? Le diagonali dicono di quanto "
                "si scostano, e il consenso diventa il rettangolo raffinato",
    },
    {
        "id": "segmento",
        "label": "3 · segmento piu' lungo per piano L/T",
        # il piano non si aspetta: lo si riconosce immagine per immagine dentro il
        # rettangolo provvisorio, ed e' proprio quello che serve per non mescolare L e T
        "needs": ("orientation",),
        "what": "la corda piu' larga del ventaglio, cercata nel piano giusto: fissa "
                "l'ampiezza orizzontale. NF/LR contro UD/LRUD, che devono essere speculari",
    },
    {
        "id": "depth",
        "label": "4 · incrocio con la depth",
        "needs": ("orientation", "plane", "depth"),
        "what": "la scala e la depth dicono dove finisce l'immagine utile: da usare per "
                "chiudere il rettangolo in verticale",
    },
)


def _groups_of_images(project: Project) -> Dict[str, str]:
    """Il gruppo di orientamento per immagine, con la solita priorita' delle sorgenti."""
    value = project.step_value("orientation")
    out: Dict[str, str] = {}
    stage = _marker_stage_dir(project)
    if stage is not None:
        for row in _marker_rows(stage):
            if row.get("group") and str(row.get("status") or "").lower() == "ok":
                out[row["name"]] = row["group"]
    for sorgente in ("marker_rows_override", "validation_rows"):
        for name, row in (value.get(sorgente) or {}).items():
            if row.get("group"):
                out[name] = row["group"]
    for name, fixed in (value.get("corrections") or {}).items():
        if fixed.get("group"):
            out[name] = fixed["group"]
    return out


def _mirror_rect(rect: Dict[str, int], group: str, width: int, height: int) -> Dict[str, int]:
    """Il rettangolo riportato nel sistema di NF ribaltando rispetto al centro dell'immagine."""
    top, left = int(rect["top"]), int(rect["left"])
    bottom, right = int(rect["bottom"]), int(rect["right"])
    if group in ("LR", "LRUD") and width:
        left, right = width - 1 - right, width - 1 - left
    if group in ("UD", "LRUD") and height:
        top, bottom = height - 1 - bottom, height - 1 - top
    return {"top": top, "left": left, "bottom": bottom, "right": right}


def _rect_consensus(rects: Sequence[Dict[str, int]]) -> Optional[Dict[str, int]]:
    if not rects:
        return None
    return {
        side: int(round(statistics.median([float(r[side]) for r in rects])))
        for side in ("top", "left", "bottom", "right")
    }


def _rect_residual(rect: Dict[str, int], reference: Dict[str, int]) -> Dict:
    """Scarto fra due rettangoli, letto sulle diagonali: due punti bastano a definirli."""
    dx1 = rect["left"] - reference["left"]
    dy1 = rect["top"] - reference["top"]
    dx2 = rect["right"] - reference["right"]
    dy2 = rect["bottom"] - reference["bottom"]
    return {
        "corner_top_left": [dx1, dy1],
        "corner_bottom_right": [dx2, dy2],
        "diag_start_px": round((dx1 ** 2 + dy1 ** 2) ** 0.5, 1),
        "diag_end_px": round((dx2 ** 2 + dy2 ** 2) ** 0.5, 1),
        "max_side_px": max(abs(dx1), abs(dy1), abs(dx2), abs(dy2)),
    }


def _run_rect_specularity(job_id: str, project_id: str, per_group: int) -> None:
    """Un rettangolo per gruppo di orientamento, e il confronto fra i quattro.

    Non si assume quale sia l'invariante: si misurano **entrambe** le ipotesi — i quattro
    rettangoli identici fra loro, oppure speculari rispetto al centro dell'immagine — e si
    riporta quale delle due tiene. E' la misura a dire la regola.
    """
    try:
        project = _project(project_id)
        folder = _require_folder(project)
        gruppi = _groups_of_images(project)
        if not gruppi:
            raise ValueError(
                "serve prima l'orientamento: senza i gruppi non c'e' niente da confrontare"
            )
        imported = project.step_value("import")
        width, height = (imported.get("image_sample_size") or [0, 0])[:2]
        if not width or not height:
            raise ValueError("manca la dimensione dell'immagine campione")

        analysis = project.data.get("analysis") or {}
        vendor = ((analysis.get("vendor") or {}).get("vendor") or "")
        vendor_conf = float((analysis.get("vendor") or {}).get("confidence") or 0.0)
        base = project.dedup_link_dir() or folder

        per_gruppo: Dict[str, List[Path]] = {}
        for name, group in gruppi.items():
            if group in marker_refine.GROUP_ORDER:
                per_gruppo.setdefault(group, []).append(base / name)
        for group in per_gruppo:
            immagini = sorted(per_gruppo[group])
            passo = max(1, len(immagini) // per_group)
            per_gruppo[group] = immagini[::passo][:per_group]

        engine = _inference_engine()
        misure: Dict[str, Dict] = {}
        for indice, (group, immagini) in enumerate(sorted(per_gruppo.items())):
            _job_update(
                job_id,
                stage=f"rettangolo del gruppo {group} ({len(immagini)} immagini)",
                done=indice, total=len(per_gruppo),
            )
            esito = engine.predict_rect(
                [p for p in immagini if p.exists()], vendor=vendor, vendor_confidence=vendor_conf
            )
            box = esito.get("rect_echo")
            if not box:
                continue
            misure[group] = {
                "rect": box,
                "images": len(esito.get("boxes") or []),
                "agreement_iou": esito.get("agreement_iou"),
                "mirrored": _mirror_rect(box, group, width, height),
            }
        if len(misure) < 2:
            raise ValueError("servono almeno due gruppi con un rettangolo per confrontarli")

        _job_update(job_id, stage="confronto fra i gruppi", done=len(per_gruppo), total=len(per_gruppo))
        # Ipotesi A: i quattro rettangoli sono lo stesso riquadro di schermo.
        cons_identici = _rect_consensus([m["rect"] for m in misure.values()])
        # Ipotesi B: sono speculari rispetto al centro dell'immagine.
        cons_speculari = _rect_consensus([m["mirrored"] for m in misure.values()])
        for group, m in misure.items():
            m["residual_identity"] = _rect_residual(m["rect"], cons_identici)
            m["residual_mirror"] = _rect_residual(m["mirrored"], cons_speculari)

        peggiore_identita = max(m["residual_identity"]["max_side_px"] for m in misure.values())
        peggiore_specularita = max(m["residual_mirror"]["max_side_px"] for m in misure.values())
        # Sotto specularita' il consenso vive nel sistema di NF: per tornare in coordinate
        # schermo va ribaltato indietro, ma per un rettangolo simmetrico le due cose
        # coincidono — ed e' proprio quello che la misura verifica.
        vincitore = "identita" if peggiore_identita <= peggiore_specularita else "specularita"
        proposta = cons_identici if vincitore == "identita" else cons_speculari

        # La domanda che conta per la scala: il rettangolo torna su se stesso se lo si
        # ribalta rispetto al centro dell'immagine? Se no, una scala ribaltata non cade
        # dentro lo stesso riquadro, ed e' il controllo che il requisito chiedeva.
        specchio = _mirror_rect(proposta, "LRUD", width, height)
        asimmetria = _rect_residual(specchio, proposta)
        dx = ((proposta["left"] + proposta["right"]) / 2.0) - ((width - 1) / 2.0)
        dy = ((proposta["top"] + proposta["bottom"]) / 2.0) - ((height - 1) / 2.0)
        centro = {
            "dx": round(dx, 1),
            "dy": round(dy, 1),
            # di quanto si sposta il rettangolo se lo si ribalta: il doppio dello scostamento
            # dal centro. Zero vuol dire che il ribaltamento lo lascia dov'e', quindi una
            # scala ribaltata cade ancora dentro lo stesso riquadro.
            "shift_lr": round(abs(dx) * 2, 1),
            "shift_ud": round(abs(dy) * 2, 1),
            # riferimento misurato sulle dieci configurazioni storiche in templates/DB_setup
            "legacy_note": "sulle dieci storiche: orizzontale 0-128 px, verticale 40-303 px",
        }

        corrente = project.step_value("rect").get("rect_echo")
        _job_update(
            job_id, status="done", stage="fatto",
            result={
                "per_group": misure,
                "consensus_identity": cons_identici,
                "consensus_mirror": cons_speculari,
                "worst_identity_px": peggiore_identita,
                "worst_mirror_px": peggiore_specularita,
                "hypothesis": vincitore,
                "self_mirror": specchio,
                "asymmetry": asimmetria,
                "off_centre": centro,
                "image_size": [width, height],
                "proposal": proposta,
                "current": corrente,
                "delta": _rect_residual(proposta, corrente) if corrente else None,
                "groups": sorted(misure),
                "per_group_images": {g: len(v) for g, v in per_gruppo.items()},
            },
        )
        # la proposta resta nello step: si applica solo quando l'utente conferma
        def mutate(project: Project, value: Dict) -> Dict:
            catena = dict(value.get("chain") or {})
            passi = dict(catena.get("passes") or {})
            passi["specularita"] = {
                "state": "proposed",
                "proposal": proposta,
                "hypothesis": vincitore,
                "worst_identity_px": peggiore_identita,
                "worst_mirror_px": peggiore_specularita,
                "off_centre": centro,
                "asymmetry_px": asimmetria["max_side_px"],
                "per_group": {g: m["rect"] for g, m in misure.items()},
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            catena["passes"] = passi
            value["chain"] = catena
            return value

        _write_step(project_id, "rect", mutate)
    except Exception as error:  # noqa: BLE001
        _job_update(job_id, status="error", stage="errore", error=str(error))


def _write_step(project_id: str, step_id: str, mutate, status: str = "proposed",
                source: str = "model") -> Dict:
    """Read-modify-write serializzato per uno step qualunque."""
    with _step_write_lock:
        project = _project(project_id)
        value = mutate(project, dict(project.step_value(step_id)))
        project.set_step(step_id, value, status=status, source=source)
        return value


# I due assi di specularita': il contenuto si ribalta, non il riquadro. NF e LR mostrano lo
# stesso ventaglio specchiato sull'asse verticale, UD e LRUD lo stesso sull'orizzontale.
SEGMENT_PAIRS = (("NF", "LR", "x"), ("UD", "LRUD", "x"), ("NF", "UD", "y"))


def _segment_estimator():
    """Il detector del segmento vive nei tool della pipeline: si importa, non si riscrive."""
    root = REPO_ROOT / "tools" / "ultrasound"
    if not root.is_dir():
        raise FileNotFoundError(f"tools/ultrasound non trovato: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from rect_red_pipeline import estimate_horizontal_segment_from_path  # noqa: PLC0415

    return estimate_horizontal_segment_from_path


def _mirror_segment(seg: Dict, axis: str) -> Dict:
    """Il segmento riportato nell'altro orientamento: le coordinate sono normalizzate."""
    if axis == "x":
        return {**seg, "x1": 1.0 - float(seg["x2"]), "x2": 1.0 - float(seg["x1"])}
    return {**seg, "y": 1.0 - float(seg["y"])}




def _costruzione_sugli_assi(misure: Dict[str, Dict], width: int, height: int) -> Dict:
    """Il rettangolo costruito **sugli assi**, contenendo tutte le corde.

    - l'asse verticale e' il centro delle corde: il rettangolo ci sta simmetrico attorno,
      quindi il margine a destra e quello a sinistra sono uguali per costruzione;
    - l'asse orizzontale sta a meta' fra la corda **piu' alta** dei gruppi «su» (NF, LR) e
      la **piu' bassa** dei gruppi «giu» (UD, LRUD): le due estreme cadono esattamente sui
      bordi e tutte le altre stanno dentro;
    - le mezze estensioni sono la distanza dall'asse all'estremo piu' lontano; il margine lo
      decide l'utente e si aggiunge uguale sui due lati.

    Qui si restituisce la costruzione **senza margine**: cosi' lo slider puo' muoverlo senza
    tornare al server, e il rettangolo si ridisegna mentre lo trascini.
    """
    corde = []
    for group, m in misure.items():
        for seg in (m.get("all") or ([m["segment"]] if m.get("segment") else [])):
            corde.append({**seg, "group": group})
    if not corde:
        return {}

    centri = [(float(s["x1"]) + float(s["x2"])) / 2.0 * width for s in corde]
    asse_x = statistics.median(centri)
    def lontananza(s: Dict) -> float:
        return max(abs(float(s["x1"]) * width - asse_x), abs(float(s["x2"]) * width - asse_x))
    piu_lontano = max(corde, key=lontananza)
    mezza_x = lontananza(piu_lontano)

    su = [s for s in corde if s["group"] in ("NF", "LR")]
    giu = [s for s in corde if s["group"] in ("UD", "LRUD")]
    asse_y = mezza_y = None
    alta = bassa = None
    if su and giu:
        alta = min(su, key=lambda s: float(s["y"]))
        bassa = max(giu, key=lambda s: float(s["y"]))
        y_alta, y_bassa = float(alta["y"]) * height, float(bassa["y"]) * height
        asse_y = (y_alta + y_bassa) / 2.0
        mezza_y = abs(y_bassa - y_alta) / 2.0

    return {
        "axis_x": round(asse_x, 1),
        "half_x": round(mezza_x, 1),
        # quanto ballano i centri delle corde: se e' molto, l'asse e' incerto e la
        # simmetria costa slack da una parte
        "centre_spread_px": round(max(centri) - min(centri), 1),
        "axis_y": round(asse_y, 1) if asse_y is not None else None,
        "half_y": round(mezza_y, 1) if mezza_y is not None else None,
        "chords": len(corde),
        "widest": {"image": piu_lontano.get("image"), "group": piu_lontano.get("group"),
                   "x1": round(float(piu_lontano["x1"]) * width, 1),
                   "x2": round(float(piu_lontano["x2"]) * width, 1)},
        "highest": ({"image": alta.get("image"), "group": alta.get("group"),
                     "y": round(float(alta["y"]) * height, 1)} if alta else None),
        "lowest": ({"image": bassa.get("image"), "group": bassa.get("group"),
                    "y": round(float(bassa["y"]) * height, 1)} if bassa else None),
        "image_size": [width, height],
    }


def _segments_analysis(
    misure: Dict[str, Dict], width: int, height: int, corrente: Dict[str, int],
    rect_norm: Optional[Dict] = None,
) -> Dict:
    """Dalle corde per gruppo alle coppie speculari, all'ampiezza, alla proposta.

    Sta a parte perche' la fanno in due: il giro 3, che le corde le trova; e la
    correzione a mano, che ne cambia una e ha bisogno che tutto il resto si rifaccia
    senza ricalcolare niente con le reti — sono minimi, massimi e differenze.
    """
    # Come nel giro 2 non si assume l'invariante: si misurano tutte e due le ipotesi —
    # le due corde nella stessa posizione, oppure specchiate rispetto al centro
    # dell'immagine — e vince quella che tiene. Sulla prima cartella GE ha vinto
    # l'identita' (0 px contro 245): il ventaglio e' simmetrico attorno al proprio asse,
    # quindi la sua corda piu' larga non si sposta quando il contenuto si ribalta.
    coppie = {}
    for primo, secondo, asse in SEGMENT_PAIRS:
        a, b = misure.get(primo), misure.get(secondo)
        if not a or not b:
            continue
        sa, sb = a["segment"], b["segment"]
        specchiato = _mirror_segment(sb, asse)

        def scarto(altro: Dict) -> Dict:
            if asse == "x":
                d1 = abs(float(altro["x1"]) - float(sa["x1"])) * width
                d2 = abs(float(altro["x2"]) - float(sa["x2"])) * width
            else:
                d1 = d2 = abs(float(altro["y"]) - float(sa["y"])) * height
            return {"start_px": round(d1, 1), "end_px": round(d2, 1),
                    "max_px": round(max(d1, d2), 1)}

        identita, specularita = scarto(sb), scarto(specchiato)
        coppie[f"{primo}-{secondo}"] = {
            "axis": asse,
            "mirrored": specchiato,
            "identity": identita,
            "mirror": specularita,
            "hypothesis": ("identita" if identita["max_px"] <= specularita["max_px"]
                           else "specularita"),
            "start_px": min(identita["start_px"], specularita["start_px"]),
            "end_px": min(identita["end_px"], specularita["end_px"]),
            "max_px": min(identita["max_px"], specularita["max_px"]),
            "length_delta_px": round(
                abs(float(sb["length_px"]) - float(sa["length_px"])), 1),
        }

    # L'ampiezza orizzontale la fissa l'unione delle corde, portate nel sistema di NF
    # secondo l'ipotesi che ha vinto: il rettangolo deve contenere il punto in cui il
    # ventaglio e' piu' largo, in ogni orientamento.
    specchia = any(
        c["hypothesis"] == "specularita" for c in coppie.values() if c["axis"] == "x"
    )
    nel_sistema_nf = []
    for group, m in misure.items():
        seg = m["segment"]
        nel_sistema_nf.append(
            _mirror_segment(seg, "x") if (specchia and group in ("LR", "LRUD")) else seg
        )
    piu_larga = max(nel_sistema_nf, key=lambda s: float(s["x2"]) - float(s["x1"]))
    margine = 0.02
    x1 = min(float(s["x1"]) for s in nel_sistema_nf)
    x2 = max(float(s["x2"]) for s in nel_sistema_nf)
    sinistra = max(0, int(round((x1 - margine) * width)))
    destra = min(width - 1, int(round((x2 + margine) * width)))
    proposta = {**corrente, "left": sinistra, "right": destra}

    # Il confronto ha senso dentro un piano solo: se i gruppi cadono su piani diversi
    # (succede con le sonde biplana, dove la cartella contiene sia L sia T) le corde non
    # sono confrontabili e la proposta non e' affidabile. Meglio dirlo che tacerlo.
    piani_per_gruppo = {g: m.get("plane") for g, m in misure.items()}
    piani_distinti = {p for p in piani_per_gruppo.values() if p}
    conflitto = len(piani_distinti) > 1

    return {
        "per_group": misure,
        "construction": _costruzione_sugli_assi(misure, width, height),
        "planes_by_group": piani_per_gruppo,
        "plane_conflict": conflitto,
        "plane_note": (
            "i gruppi cadono su piani diversi ("
            + ", ".join(f"{g}:{p}" for g, p in sorted(piani_per_gruppo.items()) if p)
            + "): le corde non sono confrontabili e la proposta non e' affidabile. "
              "Questa cartella contiene due configurazioni, una per piano."
            if conflitto else ""
        ),
        "pairs": coppie,
        "widest": piu_larga,
        "union": {"x1": x1, "x2": x2},
        "mirrored_frame": specchia,
        "margin": margine,
        "proposal": proposta,
        "current": corrente,
        "delta": _rect_residual(proposta, corrente),
        "image_size": [width, height],
        "rect_norm": rect_norm,
    }


def _run_rect_segments(job_id: str, project_id: str, per_group: int) -> None:
    """Il segmento piu' lungo per gruppo, e la specularita' fra le coppie.

    E' il contenuto a ribaltarsi, non il riquadro: la corda piu' larga del ventaglio in NF
    deve cadere sulla corda di LR specchiata rispetto all'asse verticale, e quella di UD su
    LRUD. Lo scarto fra le due, in pixel, dice se il rettangolo e' al posto giusto — e la
    corda piu' larga fissa l'ampiezza orizzontale.
    """
    try:
        stimatore = _segment_estimator()
        project = _project(project_id)
        folder = _require_folder(project)
        gruppi = _groups_of_images(project)
        if not gruppi:
            raise ValueError("serve prima l'orientamento")
        imported = project.step_value("import")
        width, height = (imported.get("image_sample_size") or [0, 0])[:2]
        corrente = project.step_value("rect").get("rect_echo")
        if not width or not height or not corrente:
            raise ValueError("servono il rettangolo provvisorio e la dimensione dell'immagine")
        rect_norm = {
            "x": corrente["left"] / float(width),
            "y": corrente["top"] / float(height),
            "w": (corrente["right"] - corrente["left"] + 1) / float(width),
            "h": (corrente["bottom"] - corrente["top"] + 1) / float(height),
        }
        base = project.dedup_link_dir() or folder
        engine = _inference_engine()

        per_gruppo: Dict[str, List[Path]] = {}
        for name, group in gruppi.items():
            if group in marker_refine.GROUP_ORDER:
                per_gruppo.setdefault(group, []).append(base / name)
        for group in per_gruppo:
            immagini = sorted(per_gruppo[group])
            passo = max(1, len(immagini) // per_group)
            per_gruppo[group] = [p for p in immagini[::passo][:per_group] if p.exists()]

        misure: Dict[str, Dict] = {}
        for indice, (group, immagini) in enumerate(sorted(per_gruppo.items())):
            _job_update(job_id, stage=f"segmento nel gruppo {group}",
                        done=indice, total=len(per_gruppo))
            etichetta = "su" if group in ("NF", "LR") else "giu"
            trovati = []
            for path in immagini:
                seg = stimatore(
                    image_path=path, rect_norm=rect_norm, orientation_label=etichetta,
                )
                if not seg:
                    continue
                piano = engine.predict_lt([path], corrente).get("plane")
                trovati.append({
                    "image": str(path.relative_to(base)),
                    "plane": piano,
                    "x1": float(seg["x1"]), "x2": float(seg["x2"]), "y": float(seg["y"]),
                    "length_norm": float(seg["length_norm"]),
                    "length_px": round(float(seg["length_norm"]) * width, 1),
                })
            if not trovati:
                continue
            # il piano dominante del gruppo: la corda va cercata dentro un piano solo
            piani: Dict[str, int] = {}
            for s in trovati:
                if s["plane"]:
                    piani[s["plane"]] = piani.get(s["plane"], 0) + 1
            dominante = max(piani, key=piani.get) if piani else None
            nel_piano = [s for s in trovati if not dominante or s["plane"] == dominante]
            migliore = max(nel_piano, key=lambda s: s["length_norm"])
            misure[group] = {
                "segment": migliore,
                "plane": dominante,
                "planes": piani,
                "tried": len(immagini),
                "found": len(trovati),
                "median_length_px": round(
                    statistics.median([s["length_px"] for s in nel_piano]), 1),
                # tutte le corde del gruppo: servono per gli estremi, cioe' per costruire un
                # rettangolo che le contenga tutte e non solo quella rappresentativa
                "all": nel_piano,
            }

        if not misure:
            raise ValueError("nessun segmento trovato: il rettangolo provvisorio e' plausibile?")

        risultato = _segments_analysis(misure, width, height, corrente, rect_norm)
        _job_update(job_id, status="done", stage="fatto", result=risultato)

        def mutate(project: Project, value: Dict) -> Dict:
            catena = dict(value.get("chain") or {})
            passi = dict(catena.get("passes") or {})
            passi["segmento"] = {
                "state": "proposed",
                "proposal": risultato["proposal"],
                "pairs": {k: v["max_px"] for k, v in risultato["pairs"].items()},
                "per_group": {g: m["segment"]["length_px"] for g, m in misure.items()},
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            catena["passes"] = passi
            value["chain"] = catena
            value["study"] = {"segments": risultato}
            return value

        _write_step(project_id, "rect", mutate)
    except Exception as error:  # noqa: BLE001
        _job_update(job_id, status="error", stage="errore", error=str(error))


def _images_by_group(project: Project) -> Dict[str, List[str]]:
    """Le immagini della cartella divise per orientamento, in ordine."""
    out: Dict[str, List[str]] = {}
    for name, group in _groups_of_images(project).items():
        if group in marker_refine.GROUP_ORDER:
            out.setdefault(group, []).append(name)
    return {group: sorted(nomi) for group, nomi in out.items()}


@app.get("/api/projects/<project_id>/rect/chain")
def api_rect_chain(project_id: str):
    """Lo stato della catena: cosa e' pronto, cosa aspetta cosa, e cosa cambierebbe."""
    project = _project(project_id)
    rect_value = project.step_value("rect")
    orientation = project.step_value("orientation")
    analysis = project.data.get("analysis") or {}
    disponibile = {
        "orientation": bool(orientation.get("groups")),
        "plane": bool((analysis.get("plane") or {}).get("plane")),
        "depth": bool(project.step_value("depth_scale").get("depths")),
    }
    stato_passi = (rect_value.get("chain") or {}).get("passes") or {}
    passi = []
    for spec in RECT_PASSES:
        salvato = stato_passi.get(spec["id"]) or {}
        mancano = [n for n in spec["needs"] if not disponibile.get(n)]
        if spec["id"] == "rete":
            stato = "done" if rect_value.get("rect_echo") else "available"
        elif spec["id"] == "depth":
            stato = "todo"
        elif mancano:
            stato = "waiting"
        else:
            stato = salvato.get("state") or "available"
        passi.append({
            **{k: spec[k] for k in ("id", "label", "what")},
            "needs": list(spec["needs"]),
            "missing": mancano,
            "state": stato,
            "saved": salvato,
        })
    return jsonify(
        {
            "rect": rect_value.get("rect_echo"),
            "margin_percent": rect_value.get("margin_percent"),
            "study": rect_value.get("study") or {},
            "preview_image": rect_value.get("preview_image"),
            # tutte le immagini, divise per orientamento: lo studio si guarda su una sola,
            # ma va controllato su tutte
            "images_by_group": _images_by_group(project),
            "available": disponibile,
            "groups": sorted((orientation.get("groups") or {}).keys()),
            "plane": (analysis.get("plane") or {}).get("plane"),
            "passes": passi,
            "history": (rect_value.get("chain") or {}).get("history") or [],
        }
    )


def _symmetric_rect(rect: Dict[str, int], axis_x, axis_y, width: int, height: int) -> Dict:
    """Il rettangolo allargato finche' e' simmetrico attorno all'asse: ribaltarlo lo lascia
    dov'e', quindi cio' che ci si disegna dentro si ribalta dentro. Si allunga il lato
    corto, non si accorcia il lungo: accorciare taglierebbe via immagine utile."""
    out = dict(rect)
    if axis_x is not None:
        lato = max(axis_x - rect["left"], rect["right"] - axis_x)
        out["left"] = max(0, int(round(axis_x - lato)))
        out["right"] = min(width - 1, int(round(axis_x + lato)))
    if axis_y is not None:
        lato = max(axis_y - rect["top"], rect["bottom"] - axis_y)
        out["top"] = max(0, int(round(axis_y - lato)))
        out["bottom"] = min(height - 1, int(round(axis_y + lato)))
    return out


@app.post("/api/projects/<project_id>/rect/segment")
def api_rect_segment(project_id: str):
    """Correggi a mano la corda di un gruppo, e rifai i conti che ne dipendono.

    La corda e' il dato di partenza di tutto il giro 3: se il detector l'ha presa storta,
    correggerla deve valere piu' del suo risultato. Il ricalcolo non richiama nessuna rete
    — coppie speculari, ampiezza e proposta sono minimi, massimi e differenze — quindi
    risponde subito.
    """
    project = _project(project_id)
    payload = _payload()
    group = (payload.get("group") or "").strip().upper()
    if group not in marker_refine.GROUP_ORDER:
        return jsonify({"error": "gruppo non valido"}), 400
    study = ((project.step_value("rect").get("study") or {}).get("segments")) or {}
    misure = dict(study.get("per_group") or {})
    if group not in misure:
        return jsonify({"error": f"nessuna corda da correggere per {group}: lancia il giro 3"}), 400
    width, height = study.get("image_size") or [0, 0]
    corrente = project.step_value("rect").get("rect_echo")
    if not width or not corrente:
        return jsonify({"error": "manca il rettangolo o la dimensione"}), 400

    if payload.get("reset"):
        precedente = dict(misure[group].get("segment") or {})
        originale = precedente.get("original")
        if not originale:
            return jsonify({"error": "questa corda non e' stata corretta"}), 400
        # si ripristina la geometria, non si butta il resto: immagine e piano restano quelli
        nuova = {**precedente, **originale}
        for chiave in ("corrected", "original", "ts"):
            nuova.pop(chiave, None)
    else:
        try:
            x1_px = float(payload["x1"])
            x2_px = float(payload["x2"])
            y_px = float(payload["y"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "servono x1, x2 e y in pixel dell'immagine"}), 400
        if abs(x2_px - x1_px) < 8:
            return jsonify({"error": "la corda e' troppo corta: almeno 8 px"}), 400
        precedente = dict(misure[group]["segment"])
        nuova = {
            **precedente,
            "x1": max(0.0, min(1.0, min(x1_px, x2_px) / width)),
            "x2": max(0.0, min(1.0, max(x1_px, x2_px) / width)),
            "y": max(0.0, min(1.0, y_px / height)),
            "corrected": True,
            "ts": datetime.now().isoformat(timespec="seconds"),
            # l'originale si tiene: una correzione deve poter essere annullata
            "original": precedente.get("original") or {
                k: precedente[k] for k in ("x1", "x2", "y", "length_norm", "length_px")
            },
        }
        nuova["length_norm"] = nuova["x2"] - nuova["x1"]
        nuova["length_px"] = round(nuova["length_norm"] * width, 1)
        # se l'hai corretta guardando un'altra immagine, e' quella che vale come riferimento
        if payload.get("image"):
            nuova["image"] = str(payload["image"])
    tutte = list(misure[group].get("all") or [])
    riferimento = (misure[group].get("segment") or {}).get("image")
    sostituite = False
    for indice, voce in enumerate(tutte):
        if voce.get("image") == riferimento:
            tutte[indice] = nuova
            sostituite = True
            break
    if not sostituite:
        tutte.append(nuova)
    misure[group] = {**misure[group], "segment": nuova, "all": tutte}

    risultato = _segments_analysis(misure, width, height, corrente, study.get("rect_norm"))

    def mutate(project: Project, value: Dict) -> Dict:
        catena = dict(value.get("chain") or {})
        passi = dict(catena.get("passes") or {})
        passi["segmento"] = {
            **(passi.get("segmento") or {}),
            "state": "proposed",
            "proposal": risultato["proposal"],
            "pairs": {k: v["max_px"] for k, v in risultato["pairs"].items()},
            "corrected_groups": sorted(
                g for g, m in misure.items() if (m.get("segment") or {}).get("corrected")
            ),
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        catena["passes"] = passi
        value["chain"] = catena
        value["study"] = {**(value.get("study") or {}), "segments": risultato}
        return value

    _write_step(project_id, "rect", mutate)
    return jsonify(
        {
            "saved": True,
            "group": group,
            "segment": nuova,
            "pairs": risultato["pairs"],
            "union": risultato["union"],
            "proposal": risultato["proposal"],
            "corrected_groups": sorted(
                g for g, m in misure.items() if (m.get("segment") or {}).get("corrected")
            ),
        }
    )


@app.get("/api/projects/<project_id>/rect/axes")
def api_rect_axes(project_id: str):
    """Gli assi di ribaltamento misurati da due feature indipendenti, e i candidati.

    L'asse non si assume: si misura. Dalle **corde** (il punto medio della corda piu' larga
    per la x, il punto medio fra la posizione su e giu' per la y) e dai **marker** di
    orientamento (il punto medio fra i due gruppi della coppia). Se le due feature
    concordano l'asse e' fidato; dove divergono, quella divergenza e' l'errore che non va
    nascosto — sulle cartelle provate: 2-14 px in orizzontale, 42-49 px in verticale.
    """
    project = _project(project_id)
    rect = project.step_value("rect").get("rect_echo")
    imported = project.step_value("import")
    width, height = (imported.get("image_sample_size") or [0, 0])[:2]
    if not rect or not width:
        return jsonify({"error": "serve il rettangolo provvisorio"}), 400

    study = ((project.step_value("rect").get("study") or {}).get("segments")) or {}
    catena = (project.step_value("rect").get("chain") or {}).get("passes") or {}

    assi: Dict[str, Optional[float]] = {
        "immagine_x": (width - 1) / 2.0,
        "immagine_y": (height - 1) / 2.0,
        "rect_x": (rect["left"] + rect["right"]) / 2.0,
        "rect_y": (rect["top"] + rect["bottom"]) / 2.0,
        "corde_x": None, "corde_y": None, "marker_x": None, "marker_y": None,
    }
    if study.get("union"):
        assi["corde_x"] = (study["union"]["x1"] + study["union"]["x2"]) / 2.0 * width
    ys = sorted({round(m["segment"]["y"], 4) for m in (study.get("per_group") or {}).values()})
    if len(ys) >= 2:
        assi["corde_y"] = (min(ys) + max(ys)) / 2.0 * height

    # i marker: la mediana del centro per gruppo, poi il punto medio delle coppie
    gruppi = _groups_of_images(project)
    value = project.step_value("orientation")
    righe = value.get("marker_rows_override") or value.get("validation_rows") or {}
    centri: Dict[str, List[Tuple[float, float]]] = {}
    for name, row in righe.items():
        box, group = row.get("box"), row.get("group") or gruppi.get(name)
        if box and group:
            centri.setdefault(group, []).append(
                ((box["left"] + box["right"]) / 2.0, (box["top"] + box["bottom"]) / 2.0)
            )
    mediani = {
        g: (statistics.median([p[0] for p in v]), statistics.median([p[1] for p in v]))
        for g, v in centri.items()
    }
    coppie_x = [(a, b) for a, b in (("NF", "LR"), ("UD", "LRUD")) if a in mediani and b in mediani]
    coppie_y = [(a, b) for a, b in (("NF", "UD"), ("LR", "LRUD")) if a in mediani and b in mediani]
    if coppie_x:
        assi["marker_x"] = statistics.mean(
            [(mediani[a][0] + mediani[b][0]) / 2.0 for a, b in coppie_x])
    if coppie_y:
        assi["marker_y"] = statistics.mean(
            [(mediani[a][1] + mediani[b][1]) / 2.0 for a, b in coppie_y])

    accordo = {
        "x_px": (round(abs(assi["corde_x"] - assi["marker_x"]), 1)
                 if assi["corde_x"] is not None and assi["marker_x"] is not None else None),
        "y_px": (round(abs(assi["corde_y"] - assi["marker_y"]), 1)
                 if assi["corde_y"] is not None and assi["marker_y"] is not None else None),
    }

    limite = [constants.RECT_MAX_WIDTH, constants.RECT_MAX_HEIGHT]

    def candidato(nome: str, box: Optional[Dict], nota: str) -> Optional[Dict]:
        if not box:
            return None
        larghezza = box["right"] - box["left"]
        altezza = box["bottom"] - box["top"]
        # Il centro del rettangolo cade sull'asse di ribaltamento? E' la domanda che decide
        # se una scala ribaltata ricadra' dentro: le due mezze larghezze dicono di quanto
        # il rettangolo pende da una parte.
        centro_x = (box["left"] + box["right"]) / 2.0
        centro_y = (box["top"] + box["bottom"]) / 2.0
        misura_asse = None
        if assi["corde_x"] is not None:
            misura_asse = {
                "axis_x": assi["corde_x"],
                "centre_x": round(centro_x, 1),
                "off_axis_x": round(centro_x - assi["corde_x"], 1),
                "half_left": round(assi["corde_x"] - box["left"], 1),
                "half_right": round(box["right"] - assi["corde_x"], 1),
            }
            if assi["corde_y"] is not None:
                misura_asse.update({
                    "axis_y": assi["corde_y"],
                    "centre_y": round(centro_y, 1),
                    "off_axis_y": round(centro_y - assi["corde_y"], 1),
                    "half_top": round(assi["corde_y"] - box["top"], 1),
                    "half_bottom": round(box["bottom"] - assi["corde_y"], 1),
                })
        return {
            "id": nome, "rect": box, "note": nota,
            "size": [larghezza, altezza],
            "over_limit": larghezza > limite[0] or altezza > limite[1],
            "delta": _rect_residual(box, rect),
            "axis_fit": misura_asse,
        }

    # Nomi in chiaro e una riga che dice *perche'* uno lo sceglieresti: "consenso" e
    # "simmetrico_corde" non dicono niente a chi deve decidere.
    fidato = (accordo["x_px"] is not None and accordo["x_px"] <= 15
              and accordo["y_px"] is not None and accordo["y_px"] <= 15)
    candidati = [
        candidato("attuale", rect,
                  "quello salvato adesso, che i moduli stanno usando"),
        candidato("consenso", (catena.get("specularita") or {}).get("proposal"),
                  "la mediana dei rettangoli visti nei quattro orientamenti: toglie il "
                  "rumore della singola immagine"),
        candidato("ampiezza", (catena.get("segmento") or {}).get("proposal"),
                  "allargato fino alla corda piu' larga del ventaglio: sotto questo il "
                  "rettangolo taglia via immagine utile"
                  + (
                      " — unione delle corde dei quattro orientamenti da x "
                      f"{round(study['union']['x1'] * width)} a x "
                      f"{round(study['union']['x2'] * width)}, piu' un margine del "
                      f"{round(study.get('margin', 0.02) * 100)}% "
                      f"({round(study.get('margin', 0.02) * width)} px per lato); "
                      "alto e basso restano quelli di adesso"
                      if study.get("union") else ""
                  )),
        candidato("simmetrico_corde",
                  _symmetric_rect(rect, assi["corde_x"], assi["corde_y"], width, height),
                  "simmetrico attorno all'asse del ventaglio: ribaltandolo resta dov'e', "
                  "quindi una scala ribaltata ricade dentro"),
        candidato("simmetrico_marker",
                  _symmetric_rect(rect, assi["marker_x"], assi["marker_y"], width, height),
                  "simmetrico attorno all'asse dei marker di orientamento"),
    ]
    ETICHETTE = {
        "attuale": "Quello di adesso",
        "consenso": "Media dei quattro orientamenti",
        "ampiezza": "Allargato fino alla corda piu' larga",
        "simmetrico_corde": "Simmetrico sull'asse del ventaglio",
        "simmetrico_marker": "Simmetrico sull'asse dei marker",
    }
    for c in candidati:
        if not c:
            continue
        c["label"] = ETICHETTE.get(c["id"], c["id"])
        avvisi = []
        if c["over_limit"]:
            avvisi.append(f"supera il limite ESI {limite[0]}x{limite[1]}")
        if c["id"] == "simmetrico_marker" and not fidato:
            avvisi.append(
                "sconsigliato: l'asse dei marker non concorda con quello delle corde "
                f"({accordo['y_px']} px sulla verticale)"
            )
        c["warnings"] = avvisi
    return jsonify(
        {
            "image_size": [width, height],
            "rect": rect,
            "axes": {k: (round(v, 1) if v is not None else None) for k, v in assi.items()},
            "agreement": accordo,
            "marker_centres": {g: [round(c[0], 1), round(c[1], 1)] for g, c in mediani.items()},
            "esi_limit": limite,
            "candidates": [c for c in candidati if c],
        }
    )


@app.post("/api/projects/<project_id>/rect/refine")
def api_rect_refine(project_id: str):
    """Lancia un giro di raffinamento. Non applica niente: propone e basta."""
    _project(project_id)
    passo = (_payload().get("pass") or "").strip()
    lavori = {"specularita": _run_rect_specularity, "segmento": _run_rect_segments}
    if passo not in lavori:
        return jsonify({"error": f"raffinamento non disponibile: {passo or '(vuoto)'}"}), 400
    per_group = int(_payload().get("per_group") or 8)
    return jsonify({"job_id": _start_job(lavori[passo], project_id, per_group)})


@app.post("/api/projects/<project_id>/rect/apply")
def api_rect_apply(project_id: str):
    """Applica la proposta di un giro, dopo che l'utente l'ha guardata."""
    _project(project_id)
    passo = (_payload().get("pass") or "").strip()

    esplicito = _payload().get("rect")

    def mutate(project: Project, value: Dict) -> Dict:
        catena = dict(value.get("chain") or {})
        passi = dict(catena.get("passes") or {})
        salvato = dict(passi.get(passo) or {})
        proposta = esplicito or salvato.get("proposal")
        if not proposta or not all(
            side in proposta for side in ("top", "left", "bottom", "right")
        ):
            raise ValueError("nessuna proposta da applicare: lancia prima il raffinamento")
        prima = value.get("rect_echo")
        value["rect_echo"] = {k: int(proposta[k]) for k in ("top", "left", "bottom", "right")}
        salvato["state"] = "applied" if not esplicito else salvato.get("state", "proposed")
        salvato["applied_at"] = datetime.now().isoformat(timespec="seconds")
        passi[passo] = salvato
        catena["passes"] = passi
        catena["history"] = [
            *(catena.get("history") or []),
            {"pass": passo, "from": prima, "to": value["rect_echo"],
             "at": salvato["applied_at"]},
        ]
        value["chain"] = catena
        return value

    try:
        value = _write_step(project_id, "rect", mutate, status="corrected", source="user")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    return jsonify({"rect": value.get("rect_echo"), "history": (value.get("chain") or {}).get("history")})


# I due modi in cui la depth si legge, e come si vedono sull'immagine.
DEPTH_MODES = {
    "direct_label": {"label": "letta dall'etichetta", "color": "#3fb950",
                     "what": "il numero e' scritto a schermo: e' stato letto li'"},
    "scale": {"label": "dedotta dalla scala", "color": "#40d0ff",
              "what": "il numero viene dalla scala ecografica: l'ultima tacca leggibile"},
    "numeric_accessory": {"label": "da un numero accessorio", "color": "#d29922",
                          "what": "nessuna delle due: un numero trovato altrove sullo schermo"},
}


def _depth_stage_dir(project: Project) -> Optional[Path]:
    root = project.root / "stages" / "rect_depth_autonomous"
    if not root.is_dir():
        return None
    cartelle = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)
    for cartella in reversed(cartelle):
        if (cartella / "rect_depth_autonomous_predictions.csv").is_file():
            return cartella
    return None


def _depth_module_rows(project: Project) -> Tuple[List[Dict], Optional[Path]]:
    """Le righe come le ha scritte il modulo, senza correzioni sopra."""
    stage = _depth_stage_dir(project)
    if stage is None:
        return [], None
    folder = Path(project.source.get("folder") or "")
    base = project.dedup_link_dir() or folder

    import csv as _csv

    righe: List[Dict] = []
    with (stage / "rect_depth_autonomous_predictions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in _csv.DictReader(handle):
            percorso = Path(row.get("image_path") or "")
            try:
                nome = str(percorso.relative_to(base))
            except ValueError:
                nome = percorso.name
            def numero(chiave: str) -> Optional[float]:
                try:
                    return float(row.get(chiave) or "")
                except (TypeError, ValueError):
                    return None
            box = None
            lati = [numero(k) for k in ("top", "left", "bottom", "right")]
            if all(v is not None for v in lati):
                box = {"top": int(lati[0]), "left": int(lati[1]),
                       "bottom": int(lati[2]), "right": int(lati[3])}
            righe.append({
                "name": nome,
                "status": row.get("status", ""),
                "mode": row.get("mode", ""),
                "depth_mm": numero("depth_mm"),
                "score": numero("score"),
                "box": box,
                "ocr_text": row.get("ocr_text", ""),
                "reason": row.get("reason", ""),
                "candidates": numero("candidates"),
                # le due strade, sempre visibili: cosi' si vede *perche'* ha scelto quella
                "direct": {"score": numero("best_direct_score"), "text": row.get("best_direct_text", "")},
                "scale": {"score": numero("best_scale_score"), "text": row.get("best_scale_text", ""),
                          "value_mm": numero("best_scale_value_mm")},
            })
    return righe, stage


@app.get("/api/projects/<project_id>/scale/study")
def api_scale_study(project_id: str):
    """Lo studio del righello, fotogramma per fotogramma: tacche, zero, passo, numeri.

    E' il modulo autonomo della scala (`detect_scale_ladder` piu' i controlli incrociati di
    `study_scale_folder`): trova la colonna del righello, ne stacca le tacche, distingue lo
    zero dal fondo, ricava il passo e legge i numeri con l'OCR. Non usa la depth: due
    etichette lette a due altezze danno gia' `mm_per_px` e lo zero.
    """
    project = _project(project_id)
    stadi = ((project.data.get("analysis") or {}).get("stages") or {})
    percorso = str((stadi.get("scale_study") or {}).get("data_json") or "")
    if not percorso or not Path(percorso).is_file():
        return jsonify({"error": "lo studio della scala non e' ancora stato fatto",
                        "frames": []}), 404
    try:
        dati = json.loads(Path(percorso).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as errore:
        return jsonify({"error": f"studio illeggibile: {errore}", "frames": []}), 500

    base = project.working_dir() or Path("")
    for frame in dati.get("frames") or []:
        percorso_frame = Path(str(frame.get("path") or ""))
        try:
            frame["name"] = str(percorso_frame.relative_to(base))
        except ValueError:
            frame["name"] = percorso_frame.name
    correzioni = project.step_value("depth_scale").get("scale_study_corrections") or {}
    dati["corrections"] = correzioni
    dati["stage"] = stadi.get("scale") or {}
    return jsonify(dati)


@app.get("/api/projects/<project_id>/depth")
def api_depth(project_id: str):
    """La depth immagine per immagine: valore, metodo e riquadro da cui e' stata presa."""
    project = _project(project_id)
    righe, stage = _depth_module_rows(project)
    if stage is None:
        return jsonify({"error": "la depth non e' ancora stata calcolata", "rows": []}), 404

    valore_step = project.step_value("depth_scale")
    # Il riquadro stretto vale per tutta la cartella: dove e' stato riletto, vince sul modulo.
    letture = dict(valore_step.get("depth_box_reads") or {})
    modello = valore_step.get("depth_box_template") or None
    nella_run = {r["name"] for r in righe}
    for r in righe:
        lettura = letture.get(r["name"])
        if not lettura:
            continue
        r.update({"depth_mm": lettura.get("depth_mm"), "box": lettura.get("box"),
                  "ocr_text": lettura.get("ocr_text", ""), "from_box": True})
    # Le immagini fuori dal campione del modulo: la depth ce l'hanno solo grazie al riquadro.
    modo_riquadro = ""
    if modello:
        sorgente = next((r for r in righe if r["name"] == modello.get("from")), None)
        modo_riquadro = (sorgente or {}).get("mode") or "direct_label"
    for nome, lettura in letture.items():
        if nome in nella_run:
            continue
        righe.append({
            "name": nome, "status": "box", "mode": modo_riquadro,
            "depth_mm": lettura.get("depth_mm"), "score": None,
            "box": lettura.get("box"), "ocr_text": lettura.get("ocr_text", ""),
            "reason": "letta nel riquadro di cartella", "candidates": None,
            "direct": {}, "scale": {}, "from_box": True, "in_run": False,
        })
    righe.sort(key=lambda r: r["name"])
    correzioni = valore_step.get("depth_corrections") or {}
    for nome, fix in correzioni.items():
        for r in righe:
            if r["name"] == nome:
                r.update({"depth_mm": fix.get("depth_mm"), "corrected": True,
                          "status": "corrected", "note": fix.get("note", "")})

    valori: Dict[str, int] = {}
    for r in righe:
        if r["depth_mm"] is not None:
            chiave = f"{r['depth_mm']:g}"
            valori[chiave] = valori.get(chiave, 0) + 1
    return jsonify(
        {
            "rows": righe,
            "box_template": modello,
            "box_applications": valore_step.get("depth_box_applications") or [],
            "box_modes": list(DEPTH_BOX_MODES),
            "modes": DEPTH_MODES,
            "by_mode": {m: sum(1 for r in righe if r["mode"] == m) for m in DEPTH_MODES},
            "by_status": {
                s: sum(1 for r in righe if r["status"] == s)
                for s in ("accepted", "review", "reject", "corrected", "box")
            },
            "values_mm": valori,
            "images_total": len(project.dedup_images()),
            "stage_dir": str(stage),
        }
    )


# --- il riquadro della depth letta dall'interfaccia ------------------------
# Quando la depth e' scritta a schermo la label sta sempre nello stesso posto: cambia solo
# quanto e' larga, perche' le cifre crescono. Un riquadro stretto sul solo numero e' quindi
# valido per tutta la cartella, e da li' il valore si rilegge immagine per immagine.
# Sulla scala no: il numero cambia posto ad ogni depth, e propagare un riquadro sarebbe falso.
DEPTH_BOX_MODES = ("direct_label", "numeric_accessory")


def _tesseract_words():
    """L'OCR vive nel modulo depth: si importa, non si riscrive."""
    root = REPO_ROOT / "tools" / "depth"
    if not root.is_dir():
        raise FileNotFoundError(f"tools/depth non trovato: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from rect_depth_hybrid import run_tesseract_tsv_region  # noqa: PLC0415

    return run_tesseract_tsv_region


def _numero_scritto(testo: str) -> Optional[float]:
    """Il numero dentro a un token OCR, con la virgola trattata come punto."""
    trovato = re.search(r"\d+(?:[.,]\d+)?", str(testo or ""))
    if not trovato:
        return None
    try:
        valore = float(trovato.group(0).replace(",", "."))
    except ValueError:
        return None
    return valore if valore > 0 else None


def _stringi_sui_pixel(image_path: Path, box: Dict, caratteri: int) -> Dict:
    """Porta il riquadro sulle sole cifre, contandole.

    L'OCR restituisce riquadri di parola generosi e disuguali fra una passata e l'altra: su
    Esaote Nine il riquadro di `43` arriva a coprire anche il `*mm` che segue. Il riquadro che
    va a match deve contenere solo il numero, altrimenti prende contesto che cambia da
    un'immagine all'altra.

    Lo spazio non basta a separarli — fra `162` e `*mm` ci sono 5 px, fra le cifre 3-4 — ma il
    numero di caratteri si sa gia' dall'OCR: si tengono i primi gruppi di colonne accese, uno
    per carattere.
    """
    from PIL import Image  # noqa: PLC0415

    try:
        with Image.open(image_path) as immagine:
            ritaglio = immagine.convert("L").crop(
                (int(box["left"]), int(box["top"]), int(box["right"]), int(box["bottom"]))
            )
        larghezza, altezza = ritaglio.size
        if larghezza < 2 or altezza < 2:
            return box
        px = ritaglio.load()
        valori = [px[x, y] for y in range(altezza) for x in range(larghezza)]
        massimo = max(valori)
        # Su una riga selezionata l'inchiostro e' scuro su barra chiara: se piu' di meta' del
        # ritaglio e' luminoso, sono i pixel *scuri* a disegnare le cifre.
        chiaro = sum(1 for v in valori if v >= 140) > len(valori) / 2
        if chiaro:
            soglia = min(140, max(40, min(valori) + 45))
            acceso = lambda v: v <= soglia  # noqa: E731
        else:
            soglia = max(70, min(155, massimo - 35))
            acceso = lambda v: v >= soglia  # noqa: E731
        colonne = [x for x in range(larghezza) if any(acceso(px[x, y]) for y in range(altezza))]
    except Exception:  # noqa: BLE001
        return box
    if not colonne:
        return box

    gruppi: List[Tuple[int, int]] = []
    inizio = precedente = colonne[0]
    for colonna in colonne[1:]:
        if colonna - precedente > 1:
            gruppi.append((inizio, precedente))
            inizio = colonna
        precedente = colonna
    gruppi.append((inizio, precedente))
    # Un carattere puo' spezzarsi in due gruppi; se i gruppi sono meno dei caratteri si
    # tengono tutti, che e' comunque il riquadro dell'OCR gia' rifilato.
    tenuti = gruppi[:caratteri] if caratteri and len(gruppi) >= caratteri else gruppi
    da, a = tenuti[0][0], tenuti[-1][1]
    righe = [y for y in range(altezza) if any(acceso(px[x, y]) for x in range(da, a + 1))]
    if not righe:
        return box
    return {
        "left": int(box["left"] + da),
        "right": int(box["left"] + a + 1),
        "top": int(box["top"] + righe[0]),
        "bottom": int(box["top"] + righe[-1] + 1),
    }


def _rileggi_nel_riquadro(image_path: Path, box: Dict, timeout: float = 8.0) -> Optional[Dict]:
    """Rileggi il numero dentro al riquadro, lasciandogli spazio per crescere di cifre.

    Il riquadro che l'utente stringe e' quello di *una* immagine: altrove il numero puo'
    avere una cifra in piu' o in meno. Quindi si cerca in una banda piu' larga e si torna
    il riquadro stretto sul numero che si e' davvero letto, non quello di partenza.
    """
    parole_di = _tesseract_words()
    larghezza = max(1.0, float(box["right"] - box["left"]))
    altezza = max(1.0, float(box["bottom"] - box["top"]))
    centro_x = (box["left"] + box["right"]) / 2.0
    centro_y = (box["top"] + box["bottom"]) / 2.0
    # Il ritaglio OCR ha un minimo: sotto 40x28 px il modulo lo rifiuta e non legge niente.
    # Senza questo, piu' l'utente stringeva il riquadro piu' la rilettura falliva in silenzio,
    # cioe' l'esatto contrario di quello che serve.
    mezza_x = max(larghezza / 2.0 + max(24.0, 1.2 * larghezza), 26.0)
    mezza_y = max(altezza / 2.0 + max(3.0, 0.30 * altezza), 18.0)
    banda = (centro_x - mezza_x, centro_y - mezza_y, centro_x + mezza_x, centro_y + mezza_y)
    candidati: List[Dict] = []
    # `invert` serve alle righe selezionate dell'interfaccia, dove il numero e' scuro su barra
    # chiara: su `prova` le immagini `_trans` hanno la riga della depth evidenziata, e senza
    # questa passata leggevano «646» invece di 15, 20, 23...
    for variante in ("base", "invert"):
        for psm in ("7", "11", "6"):
            parole, _ = parole_di(
                image_path, timeout=timeout, max_side=1800, crop_box=banda, psm=psm,
                preprocess=variante, char_whitelist="0123456789.,",
            )
            candidati.extend(_pesca_numeri(parole, box, altezza, variante))
    if not candidati:
        return None
    candidati.sort(key=lambda c: (c["distanza"], 0 if c["variante"] == "base" else 1))
    scelto = candidati[0]
    stessi = [
        c for c in candidati
        if c["value"] == scelto["value"] and abs(c["box"]["left"] - scelto["box"]["left"]) <= 6
    ]
    stretto = min(stessi, key=lambda c: (c["box"]["right"] - c["box"]["left"])
                  * (c["box"]["bottom"] - c["box"]["top"]))
    cifre = re.search(r"\d+(?:[.,]\d+)?", str(stretto["text"] or ""))
    gioco = max(2, int(round(0.25 * (box["bottom"] - box["top"]))))
    fascia = dict(stretto["box"])
    fascia["top"] = max(fascia["top"], box["top"] - gioco)
    fascia["bottom"] = min(fascia["bottom"], box["bottom"] + gioco)
    if fascia["bottom"] - fascia["top"] < 4:
        fascia = dict(stretto["box"])
    return {**stretto,
            "box": _stringi_sui_pixel(image_path, fascia,
                                      len(cifre.group(0)) if cifre else 0)}


def _pesca_numeri(parole, box: Dict, altezza: float, variante: str) -> List[Dict]:
    """I token che sono un numero e stanno sulla riga del riquadro."""
    centro_y = (box["top"] + box["bottom"]) / 2.0
    fuori: List[Dict] = []
    for parola in parole:
        valore = _numero_scritto(parola.text)
        if valore is None:
            continue
        parola_centro = parola.top + parola.height / 2.0
        if abs(parola_centro - centro_y) > altezza:
            continue
        fuori.append({
            "value": valore,
            "box": {"top": int(round(parola.top)), "left": int(round(parola.left)),
                    "bottom": int(round(parola.top + parola.height)),
                    "right": int(round(parola.left + parola.width))},
            "text": parola.text,
            "variante": variante,
            # il numero comincia dove l'utente ha messo il bordo sinistro: e' l'ancora
            "distanza": abs(parola.left - box["left"]) + abs(parola_centro - centro_y),
        })
    return fuori


def _fattore_unita(testo: str, misura_mm: Optional[float], numero: Optional[float]) -> float:
    """mm o cm: prima si guarda l'unita' scritta, poi il rapporto col valore del modulo."""
    basso = str(testo or "").lower()
    if "mm" in basso:
        return 1.0
    if "cm" in basso:
        return 10.0
    if misura_mm and numero:
        rapporto = misura_mm / numero
        if 0.8 <= rapporto <= 1.25:
            return 1.0
        if 8.0 <= rapporto <= 12.5:
            return 10.0
    return 1.0


@app.post("/api/projects/<project_id>/depth/box")
def api_depth_box(project_id: str):
    """Stringi il riquadro su un'immagine e rileggi la depth su tutte quelle con etichetta."""
    project = _project(project_id)
    payload = _payload()
    if payload.get("reset"):
        corrente = project.step_value("depth_scale")
        if not corrente.get("depth_box_template") and not corrente.get("depth_box_reads"):
            # Niente da togliere: non si marca lo step come corretto dall'utente per nulla.
            return jsonify({"applied": 0, "reset": False, "nothing": True})

        def azzera(_project: Project, value: Dict) -> Dict:
            value.pop("depth_box_template", None)
            value.pop("depth_box_applications", None)
            value.pop("depth_box_reads", None)
            return value
        _write_step(project_id, "depth_scale", azzera, status="corrected", source="user")
        return jsonify({"applied": 0, "reset": True})

    nome = str(payload.get("name") or "").strip()
    grezzo = payload.get("box") or {}
    try:
        box = {k: int(round(float(grezzo[k]))) for k in ("top", "left", "bottom", "right")}
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "serve un riquadro completo (top/left/bottom/right)"}), 400
    if box["right"] - box["left"] < 3 or box["bottom"] - box["top"] < 3:
        return jsonify({"error": "il riquadro e' troppo piccolo"}), 400

    righe, stage = _depth_module_rows(project)
    if stage is None:
        return jsonify({"error": "la depth non e' ancora stata calcolata"}), 404
    per_nome = {r["name"]: r for r in righe}
    riferimento = per_nome.get(nome)
    if riferimento is None:
        # Il modulo gira su un campione: l'immagine di riferimento puo' benissimo starne
        # fuori, e' comunque una della cartella. Serve pero' che la cartella sia una di
        # quelle in cui la depth si legge dall'interfaccia.
        if nome not in set(project.dedup_names()):
            return jsonify({"error": f"immagine non nella cartella: {nome}"}), 400
        interfaccia = [r for r in righe if r["mode"] in DEPTH_BOX_MODES]
        if not interfaccia:
            return jsonify({
                "error": "in questa cartella la depth viene dalla scala: il riquadro non si "
                         "puo' propagare, il numero cambia posto ad ogni immagine",
            }), 400
        riferimento = {"name": nome, "mode": interfaccia[0]["mode"], "ocr_text": "",
                       "depth_mm": None}
    elif riferimento["mode"] not in DEPTH_BOX_MODES:
        return jsonify({
            "error": "questa depth viene dalla scala: il riquadro non si puo' propagare, "
                     "il numero cambia posto ad ogni immagine",
        }), 400
    # Se il riquadro di cartella e' gia' stato applicato, l'unita' e' quella decisa allora:
    # il riferimento nuovo puo' non avere un valore del modulo con cui ricavarla.
    modello_attuale = project.step_value("depth_scale").get("depth_box_template") or {}

    base = project.dedup_link_dir() or Path(project.source.get("folder") or "")
    letto = _rileggi_nel_riquadro(base / nome, box)
    if letto is None:
        return jsonify({"error": "in quel riquadro non si legge nessun numero"}), 400
    fattore = _fattore_unita(riferimento.get("ocr_text"), riferimento.get("depth_mm"), letto["value"])
    if riferimento.get("depth_mm") is None and not str(riferimento.get("ocr_text") or ""):
        fattore = float(modello_attuale.get("unit_factor") or fattore)

    # "tutte" e' il default: la label sta nell'interfaccia di *ogni* fotogramma, non solo di
    # quelli che il modulo aveva campionato. Rileggere un numero dentro a un riquadro noto
    # costa 0.4 s per immagine, contro i minuti della generazione dei candidati.
    scope = str(payload.get("scope") or "all")
    elenco = [str(n) for n in (payload.get("names") or [])] if scope == "names" else []
    return jsonify({
        "job_id": _start_job(_run_depth_box, project_id, nome, box, fattore, scope, elenco),
        "unit_factor": fattore,
    })


def _leggi_riquadro_su(base: Path, box: Dict, fattore: float, nomi: Sequence[str],
                       progress=None) -> Tuple[Dict[str, Dict], List[str]]:  # noqa: ANN001
    """La rilettura nel riquadro su un elenco di immagini, in parallelo. 0.4 s l'una."""
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    letture: Dict[str, Dict] = {}
    falliti: List[str] = []
    fatte = 0

    def leggi(n: str) -> Tuple[str, Optional[Dict]]:
        return n, _rileggi_nel_riquadro(base / n, box)

    with ThreadPoolExecutor(max_workers=6) as pool:
        for n, esito in pool.map(leggi, nomi):
            fatte += 1
            if progress is not None and (fatte % 10 == 0 or fatte == len(nomi)):
                progress(fatte, len(nomi))
            if esito is None:
                falliti.append(n)
                continue
            letture[n] = {"box": esito["box"],
                          "depth_mm": round(esito["value"] * fattore, 2),
                          "ocr_text": esito["text"]}
    return letture, falliti


def _riquadro_depth_di_riferimento(righe: Sequence[Dict]) -> Optional[Tuple[str, Dict, str, Optional[float]]]:
    """Da quale immagine propagare: il riquadro piu' stretto fra quelli letti dall'interfaccia.

    Il piu' stretto perche' e' quello che contiene meno contesto oltre al numero, ed e' il
    contesto che cambia da un'immagine all'altra.
    """
    def utili(stato: str) -> List[Dict]:
        return [r for r in righe
                if r.get("mode") in DEPTH_BOX_MODES and r.get("box")
                and (not stato or str(r.get("status") or "") == stato)]

    candidati = utili("accepted") or utili("")
    if not candidati:
        return None

    def area(r: Dict) -> int:
        b = r["box"]
        return (b["right"] - b["left"]) * (b["bottom"] - b["top"])

    scelto = min(candidati, key=area)
    return scelto["name"], scelto["box"], str(scelto.get("ocr_text") or ""), scelto.get("depth_mm")


def _run_depth_box(job_id: str, project_id: str, nome: str, box: Dict,
                   fattore: float, scope: str, elenco: Sequence[str] = ()) -> None:
    """Rilegge la depth nel riquadro, immagine per immagine, in parallelo."""
    try:
        project = _project(project_id)
        base = project.dedup_link_dir() or Path(project.source.get("folder") or "")
        righe, _stage = _depth_module_rows(project)
        if scope == "run":
            nomi = [r["name"] for r in righe if r["mode"] in DEPTH_BOX_MODES]
        elif scope == "names":
            # Una cartella puo' avere piu' di un'interfaccia: su `prova` le immagini `_trans`
            # hanno la riga della depth 22 px piu' in basso delle altre. Un riquadro solo non
            # puo' servirle tutte, quindi se ne applica un secondo al sottoinsieme scelto.
            conosciuti = set(project.dedup_names())
            nomi = [n for n in elenco if n in conosciuti]
        else:
            nomi = list(project.dedup_names())
        if nome not in nomi:
            nomi.append(nome)

        _job_update(job_id, stage=f"rilettura nel riquadro su {len(nomi)} immagini",
                    total=len(nomi), done=0)
        letture, falliti = _leggi_riquadro_su(
            base, box, fattore, nomi,
            progress=lambda fatte, quante: _job_update(job_id, done=fatte),
        )

        def salva(_project: Project, value: Dict) -> Dict:
            applicazione = {
                "box": box, "from": nome, "unit_factor": fattore, "scope": scope,
                "applied": len(letture), "targets": len(nomi), "failed": falliti,
                "at": datetime.now().isoformat(timespec="seconds"),
            }
            value["depth_box_template"] = applicazione
            storico = [a for a in (value.get("depth_box_applications") or [])
                       if a.get("scope") != scope or scope == "names"]
            value["depth_box_applications"] = storico + [applicazione]
            # Le letture sono per immagine: una seconda applicazione su un sottoinsieme
            # aggiorna le sue e lascia stare le altre.
            precedenti = dict(value.get("depth_box_reads") or {}) if scope == "names" else {}
            precedenti.update(letture)
            value["depth_box_reads"] = precedenti
            return value

        _write_step(project_id, "depth_scale", salva, status="corrected", source="user")
        _job_update(job_id, status="done", stage="fatto", done=len(nomi),
                    result={"applied": len(letture), "targets": len(nomi),
                            "failed": len(falliti),
                            "values_mm": sorted({v["depth_mm"] for v in letture.values()})})
    except Exception as error:  # noqa: BLE001
        _job_update(job_id, status="error", stage="errore", error=str(error))


@app.get("/api/projects/<project_id>/depth/crop")
def api_depth_crop(project_id: str):
    """La striscia di interfaccia attorno alla depth, ingrandita, col riquadro disegnato.

    Serve a guardare *dove* si e' fermato il riquadro senza aprire l'immagine intera: a
    grandezza naturale il numero e' alto 13 px e non si giudica.
    """
    project = _project(project_id)
    nome = request.args.get("name") or ""
    base = project.dedup_link_dir() or Path(project.source.get("folder") or "")
    if not nome or not base.is_dir():
        return jsonify({"error": "immagine non disponibile"}), 404
    percorso = (base / nome).resolve()
    if not percorso.is_file():
        return jsonify({"error": "immagine non disponibile"}), 404

    letture = project.step_value("depth_scale").get("depth_box_reads") or {}
    box = (letture.get(nome) or {}).get("box")
    if box is None:
        righe, _stage = _depth_module_rows(project)
        box = next((r["box"] for r in righe if r["name"] == nome), None)
    if not box:
        return jsonify({"error": "nessun riquadro per questa immagine"}), 404

    zoom = max(1, min(8, int(request.args.get("zoom") or 3)))
    # `raw` serve all'editor: li' il riquadro e' un overlay HTML con le maniglie, quindi
    # l'immagine deve arrivare pulita. La finestra puo' arrivare dal client, cosi' resta
    # ferma mentre si trascina il riquadro dentro di essa.
    grezzo = request.args.get("raw") in ("1", "true", "si")
    from PIL import Image, ImageDraw  # noqa: PLC0415

    def _intero(chiave: str) -> Optional[int]:
        try:
            return int(float(request.args[chiave]))
        except (KeyError, TypeError, ValueError):
            return None

    with Image.open(percorso) as grezza:
        immagine = grezza.convert("RGB")
        # Poco margine in verticale: le righe di interfaccia sopra e sotto distano una
        # ventina di pixel, e mostrarne una fetta tagliata a meta' e' solo rumore.
        finestra = [_intero(k) for k in ("x0", "y0", "x1", "y1")]
        if all(v is not None for v in finestra):
            x0, y0, x1, y1 = finestra
        else:
            x0, y0 = box["left"] - 60, box["top"] - 7
            x1, y1 = box["right"] + 120, box["bottom"] + 7
        x0 = max(0, min(int(x0), immagine.size[0] - 2))
        y0 = max(0, min(int(y0), immagine.size[1] - 2))
        x1 = max(x0 + 2, min(int(x1), immagine.size[0]))
        y1 = max(y0 + 2, min(int(y1), immagine.size[1]))
        ritaglio = immagine.crop((x0, y0, x1, y1))
        ritaglio = ritaglio.resize((ritaglio.width * zoom, ritaglio.height * zoom), Image.NEAREST)
        if not grezzo:
            disegno = ImageDraw.Draw(ritaglio)
            disegno.rectangle(
                [(box["left"] - x0) * zoom - 1, (box["top"] - y0) * zoom - 1,
                 (box["right"] - x0) * zoom, (box["bottom"] - y0) * zoom],
                outline=(63, 185, 80), width=2,
            )
    buffer = io.BytesIO()
    ritaglio.save(buffer, format="PNG")
    buffer.seek(0)
    risposta = send_file(buffer, mimetype="image/png")
    risposta.headers["X-Crop-Window"] = f"{x0},{y0},{x1},{y1}"
    return risposta


@app.post("/api/projects/<project_id>/depth/run")
def api_depth_run(project_id: str):
    """Rilancia la depth, volendo su tutte le immagini e non su un campione."""
    _project(project_id)
    sample = int(_payload().get("sample") or 0)
    return jsonify({"job_id": _start_job(_run_depth_only, project_id, sample)})


def _run_depth_only(job_id: str, project_id: str, sample: int) -> None:
    """Solo la depth, senza rifare marker e scala: e' l'unica che si sta guardando."""
    try:
        project = _project(project_id)
        folder = _require_folder(project)
        rect = _final_rect(project)
        if rect is None:
            raise ValueError("serve prima il rettangolo ecografico")
        analysis = project.data.get("analysis") or {}
        imported = project.step_value("import")
        immagini = len(project.dedup_images())
        quante = sample if sample > 0 else immagini
        _job_update(job_id, stage=f"depth su {quante} immagini", total=quante)
        esito = stages_mod.run_depth(
            folder=project.dedup_link_dir() or folder,
            output_root=project.root / "stages",
            python_bin=sys.executable,
            vendor=((analysis.get("vendor") or {}).get("vendor") or ""),
            probe_id=str((analysis.get("probe") or {}).get("probe_id") or ""),
            probe_model=str(project.codes.get("probe_model") or ""),
            rect=rect,
            video_size=imported.get("image_sample_size") or [0, 0],
            rotation=0,  # lo specchio di lavoro le ha gia' raddrizzate
            max_images=quante,
            timeout=3600.0,
        )
        if esito.get("status") != "ok":
            raise ValueError(esito.get("error") or "la depth non e' andata a buon fine")
        analisi = dict(project.data.get("analysis") or {})
        stadi = dict(analisi.get("stages") or {})
        stadi["depth"] = {k: esito.get(k) for k in
                          ("status", "images", "accepted", "acceptance_ratio", "depths_mm",
                           "output_dir", "ranker_model", "folder_strategy")}
        analisi["stages"] = stadi
        project.data["analysis"] = analisi
        project.save()
        _job_update(job_id, status="done", stage="fatto",
                    result={k: esito.get(k) for k in
                            ("status", "images", "accepted", "acceptance_ratio", "depths_mm")})
    except Exception as error:  # noqa: BLE001
        _job_update(job_id, status="error", stage="errore", error=str(error))


@app.post("/api/projects/<project_id>/depth/correct")
def api_depth_correct(project_id: str):
    """Correggi a mano la depth di un'immagine."""
    project = _project(project_id)
    payload = _payload()
    name = (payload.get("name") or "").strip()
    if not name:
        return jsonify({"error": "manca l'immagine"}), 400

    def mutate(project: Project, value: Dict) -> Dict:
        correzioni = dict(value.get("depth_corrections") or {})
        if payload.get("reset"):
            correzioni.pop(name, None)
        else:
            try:
                valore = float(payload.get("depth_mm"))
            except (TypeError, ValueError):
                raise ValueError("serve la depth in millimetri")
            correzioni[name] = {
                "depth_mm": valore,
                "note": payload.get("note", ""),
                "source": "user",
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
        value["depth_corrections"] = correzioni
        return value

    try:
        value = _write_step(project_id, "depth_scale", mutate, status="corrected", source="user")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    return jsonify({"saved": True, "corrections": len(value.get("depth_corrections") or {})})


# -- .fss ------------------------------------------------------------------
@app.get("/api/projects/<project_id>/preview")
def api_preview(project_id: str):
    project = _project(project_id)
    document = project.build_document()
    lines = document.render_lines()
    rect_step = project.step_value("rect")
    return jsonify(
        {
            "rect_base": rect_step.get("rect_echo"),
            "margin_percent": rect_step.get("margin_percent"),
            "text": document.render(),
            "lines": [
                {"n": index + 1, "name": _line_name(index + 1), "value": value}
                for index, value in enumerate(lines)
            ],
            "problems": document.validate(),
            "is_biplane": document.is_biplane,
        }
    )


@app.post("/api/projects/<project_id>/generate")
def api_generate(project_id: str):
    project = _project(project_id)
    document = project.build_document()
    problems = document.validate()
    if problems and not _payload().get("force"):
        return jsonify({"written": False, "problems": problems}), 200
    target = project.fss_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document.render(), encoding="utf-8")
    project.set_step("generate", {"path": str(target), "problems": problems}, source="app")
    return jsonify({"written": True, "path": str(target), "problems": problems})


@app.post("/api/projects/<project_id>/compare")
def api_compare(project_id: str):
    """Quality gate: compare_fss.py against a legacy reference. Exit 0 = compatibile."""
    project = _project(project_id)
    reference = (_payload().get("reference") or "").strip()
    if not reference:
        return jsonify({"error": "indica il .fss legacy di riferimento"}), 400
    candidate = project.fss_path()
    if not candidate.exists():
        return jsonify({"error": "genera prima il file .fss"}), 400
    result = subprocess.run(
        [sys.executable, str(COMPARE_FSS), reference, str(candidate)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return jsonify(
        {
            "returncode": result.returncode,
            "compatible": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )


# -- immagini --------------------------------------------------------------
_image_lists: Dict[str, List[str]] = {}


@app.get("/api/projects/<project_id>/images")
def api_images(project_id: str):
    """Every frame of the folder, so the full-screen viewer can scroll through them."""
    project = _project(project_id)
    folder = Path(project.source.get("folder") or "")
    if not folder.is_dir():
        return jsonify({"error": "importa prima una cartella"}), 400
    names = project.dedup_names()
    if not names:
        if request.args.get("refresh") or str(folder) not in _image_lists:
            _image_lists[str(folder)] = [
                str(path.relative_to(folder)) for path in scan_folder(folder)
            ]
        names = _image_lists[str(folder)]
    return jsonify(
        {
            "count": len(names),
            "deduplicated": bool(project.dedup_names()),
            "names": names,
            "sample_size": project.step_value("import").get("image_sample_size"),
        }
    )

@app.get("/api/projects/<project_id>/image")
def api_image(project_id: str):
    project = _project(project_id)
    # Lo specchio di lavoro, non la cartella originale: e' li' che stanno le immagini
    # raddrizzate, ed e' su quelle che sono stati misurati i riquadri che ci disegniamo sopra.
    folder = project.working_dir() or Path(project.source.get("folder") or "")
    name = request.args.get("name") or ""
    if not folder.is_dir() or not name:
        return jsonify({"error": "immagine non disponibile"}), 404
    path = (folder / name).resolve()
    if not path.is_file() or folder.resolve() not in path.parents:
        return jsonify({"error": "percorso non consentito"}), 403
    width = int(request.args.get("w") or 320)
    box = request.args.get("box")  # "top,left,bottom,right" in original coordinates
    box2 = request.args.get("box2")  # second box, drawn in a different colour (#13 template)
    try:
        from PIL import Image, ImageDraw

        with Image.open(path) as raw:
            image = raw.convert("RGB")
            original = image.size
            image.thumbnail((width, width))
            scale_x = image.size[0] / original[0]
            scale_y = image.size[1] / original[1]
            draw = ImageDraw.Draw(image)
            for raw, colour in ((box, (255, 96, 64)), (box2, (64, 208, 255))):
                if not raw:
                    continue
                try:
                    top, left, bottom, right = [float(v) for v in raw.split(",")]
                except (ValueError, TypeError):
                    continue
                draw.rectangle(
                    [left * scale_x, top * scale_y, right * scale_x, bottom * scale_y],
                    outline=colour,
                    width=max(2, image.size[0] // 260),
                )
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=82)
        buffer.seek(0)
        return send_file(buffer, mimetype="image/jpeg")
    except Exception:
        return send_file(path)


def _line_name(number: int) -> str:
    return constants.FSS_LINE_NAMES.get(number, "?")


def main() -> int:
    global _projects_root, _anagrafica_path, _models_root
    parser = argparse.ArgumentParser(description="ESIBuilder AI - app di configurazione (base)")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--projects-root", type=Path, default=DEFAULT_PROJECTS_ROOT)
    parser.add_argument(
        "--encoding-struct",
        type=Path,
        default=None,
        help="anagrafica xlsx (default: il piu' recente encoding_struct*.xlsx nella radice del repo)",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=DEFAULT_ACTIVE_PIPELINE,
        help="cartella della pipeline attiva con i checkpoint (default: artifacts/10_active_pipeline/pipeline_fss_head)",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    _projects_root = Path(args.projects_root).expanduser().resolve()
    _projects_root.mkdir(parents=True, exist_ok=True)
    _anagrafica_path = (
        Path(args.encoding_struct).expanduser().resolve()
        if args.encoding_struct
        else default_path(REPO_ROOT)
    )
    _models_root = Path(args.models_root).expanduser().resolve()
    print(f"progetti in: {_projects_root}")
    print(f"anagrafica: {_anagrafica_path or 'NON TROVATA'}")
    print(f"modelli: {_models_dir()}")
    print(f"apri: http://{args.host}:{args.port}/")
    try:
        app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    except OSError as error:
        print(f"impossibile aprire la porta {args.port}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
