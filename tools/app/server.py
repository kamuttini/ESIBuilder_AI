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
import shutil
import statistics
import subprocess
import sys
import threading
import time
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
from importer import (  # noqa: E402
    IMAGE_SUFFIXES, deduplicate, import_folder, resize_proposal, scan_folder,
)
from inference import Engine, ModelPaths, sample_paths  # noqa: E402
from rotation import estimate_rotation  # noqa: E402
from timestamp_detection import detect_timestamp_box  # noqa: E402
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


def _run_import_analysis(job_id: str, project_id: str, folder: str, sample: int) -> None:
    """Step 0: dedup, rotation, vendor, probe, rect and L/T, before the user sees anything."""
    try:
        project = _project(project_id)
        sorgente_prima = dict(project.source)
        # Una scelta manuale e' piu' forte dell'OSD e deve sopravvivere a «rianalizza».
        # Vale soltanto per la stessa cartella: importare un'altra acquisizione ricomincia
        # correttamente dal riconoscimento automatico.
        rotazione_manual = (
            int(sorgente_prima.get("rotation_applied") or 0) % 360
            if sorgente_prima.get("rotation_source") == "user"
            and str(sorgente_prima.get("folder") or "") == str(folder) else None
        )
        override_rotazione = dict(sorgente_prima.get("rotation_overrides") or {}) \
            if str(sorgente_prima.get("folder") or "") == str(folder) else {}

        _job_update(job_id, stage="scansione e dedup bit a bit")
        # Quello che appartiene al progetto e non alla cartella sopravvive al rilancio:
        # l'area dell'orologio, i piani riconosciuti, le correzioni. Riscrivere lo step
        # daccapo li perdeva - e con i piani si perdeva anche lo sdoppiamento, che tornava
        # a mettere tutte le immagini nel progetto della L.
        prima = project.step_value("import") or {}
        da_tenere = {k: prima[k] for k in
                     ("timestamp_box", "timestamp_detection", "timestamp_disabled",
                      "planes", "plane_corrections", "plane_counts")
                     if prima.get(k) is not None}
        # La dedup a meno dell'orologio si fa **dopo** la rotazione, piu' sotto: l'area la
        # si indica su un'anteprima gia' dritta, e confrontarla su immagini storte
        # guarderebbe un altro pezzo di schermo.
        imported = import_folder(Path(folder))
        imported.update(da_tenere)
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
        rotation = ( {
            "angle": rotazione_manual, "source": "user", "reliable": True,
            "reason": "rotazione scelta manualmente", "votes": {},
        } if rotazione_manual is not None else estimate_rotation(picked) )
        angolo = int(rotation["angle"] or 0) % 360
        imported["rotation_applied"] = angolo
        imported["rotation_source"] = rotation["source"]
        imported["rotation_detail"] = rotation
        # Va scritto **e salvato** ora: piu' sotto il progetto viene riletto da disco, e in
        # memoria si perdeva. Da qui in poi lo specchio di lavoro contiene le immagini gia'
        # raddrizzate, e tutto - reti comprese - legge da li'.
        project.source["rotation_applied"] = angolo
        project.source["rotation_source"] = rotation["source"]
        project.source["rotation_overrides"] = override_rotazione
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

        # Terza tappa: la dedup a meno dell'orologio, sullo specchio **gia' dritto**. E'
        # qui e non prima perche' l'area la si indica su un'anteprima raddrizzata: su
        # immagini storte quel rettangolo cadrebbe su un altro pezzo di schermo.
        area_orologio = imported.get("timestamp_box")
        if not area_orologio and not imported.get("timestamp_disabled"):
            _job_update(job_id, stage="riconoscimento automatico di data e ora")
            rilevamento = _timestamp_preview(project, detect_timestamp_box(images))
            imported["timestamp_detection"] = rilevamento
            if rilevamento.get("reliable") and rilevamento.get("box"):
                area_orologio = dict(rilevamento["box"])
                imported["timestamp_box"] = area_orologio
                imported["timestamp_disabled"] = False
        if area_orologio:
            base_lavoro = project.working_dir() or Path(folder)
            nomi = project.dedup_names()
            _job_update(job_id, stage="dedup a meno dell'orologio", done=0, total=len(nomi))
            tenute, scartate = deduplicate(
                [base_lavoro / n for n in nomi], area_orologio,
                progress=lambda fatte, quante: _job_update(job_id, done=fatte, total=quante),
            )
            rimasti = sorted(str(Path(x).relative_to(base_lavoro)) for x in tenute)
            if len(rimasti) < len(nomi):
                project.save_dedup_images(rimasti)
                doppie = dict(imported.get("duplicates") or {})
                doppie["timestamp"] = [d for d in scartate if d["kind"] == "solo_timestamp"]
                imported["duplicates"] = doppie
                imported["duplicates_removed"] = (
                    len(doppie.get("identical") or []) + len(doppie.get("timestamp") or []))
                imported["images_total"] = len(rimasti)
                project.source["images_total"] = len(rimasti)
                project.save()
                specchio = project.root / project.DEDUP_LINKS
                if specchio.exists():
                    shutil.rmtree(specchio, ignore_errors=True)
                project.dedup_link_dir()
                project = _project(project_id)
                images = project.dedup_images()
                picked = sample_paths(images, sample)

        # E quando questo progetto e' gia' un piano (dopo lo sdoppiamento), da qui in poi
        # lavora **solo sulle sue** immagini: rileggere la cartella intera rimetterebbe
        # dentro quelle dell'altro piano, ed e' esattamente quello che succedeva.
        piano = str(project.source.get("plane") or "")
        if piano in ("L", "T"):
            etichette = _piani_salvati(project)
            suoi = [n for n in project.dedup_names()
                    if (etichette.get(n) or {}).get("plane") == piano]
            if suoi and len(suoi) < len(project.dedup_names()):
                project.save_dedup_images(sorted(suoi))
                imported["images_total"] = len(suoi)
                project.source["images_total"] = len(suoi)
                project.save()
                specchio = project.root / project.DEDUP_LINKS
                if specchio.exists():
                    shutil.rmtree(specchio, ignore_errors=True)
                project.dedup_link_dir()
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
        project.source["rotation_overrides"] = override_rotazione
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

        # Il seguito non richiede un secondo clic: parte in un job proprio, cosi' questa
        # risposta puo' mostrare subito il piano L/T. Se la cartella contiene entrambi i
        # piani, il job si ferma ordinatamente alla divisione; dopo lo split riparte su
        # ciascuna meta'. I risultati confermati/corretti dall'utente non vengono scelti
        # dal pianificatore automatico.
        automatic_job = _schedule_auto_pipeline(project_id)
        esito = {
            "import": imported,
            "analysis": analysis,
            "codes_filled": filled,
            "advanced_job_id": automatic_job,
            "next": ("riconoscimento piano L/T e moduli avviati automaticamente"
                     if automatic_job else "moduli gia' confermati: nessun ricalcolo automatico"),
        }
        _job_update(job_id, status="done", stage="fatto", result=esito)
    except Exception as error:  # noqa: BLE001 - surfaced to the user as job error
        _job_update(job_id, status="error", stage="errore", error=str(error))


def _giri_rettangolo(job_id: str, project_id: str, per_group: int = 6) -> Dict:
    """I giri 2 e 3 del rettangolo, fatti girare da soli dopo l'orientamento.

    Sono le due misure che l'orientamento rende possibili: la specularita' fra i quattro
    gruppi e la corda piu' larga del ventaglio nel piano giusto. Nessuno dei due tocca il
    rettangolo - scrivono una proposta in `chain.passes`, e applicarla resta un gesto
    dell'utente. Per questo si possono fare senza chiedere: costano una decina di secondi
    l'uno e riempiono una sezione che altrimenti l'utente trova vuota, senza capire se il
    modulo non sa rispondere o se semplicemente non e' partito.

    Un giro gia' fatto non si rifa': una proposta guardata, o corretta a mano, non va persa
    perche' si e' ricalcolata l'analisi.
    """
    esito: Dict[str, str] = {}
    giri = (
        ("specularita", _run_rect_specularity, "rettangolo: specularita' fra i quattro gruppi"),
        ("segmento", _run_rect_segments, "rettangolo: la corda piu' larga per piano"),
    )
    for passo, funzione, etichetta in giri:
        try:
            fatti = ((_project(project_id).step_value("rect").get("chain") or {})
                     .get("passes") or {})
        except FileNotFoundError:
            return esito
        if fatti.get(passo):
            esito[passo] = "gia' fatto"
            continue
        _job_update(job_id, stage=etichetta)
        # Un id suo: queste funzioni chiudono il job quando finiscono, e chiudere quello
        # grosso a meta' strada farebbe credere all'interfaccia che sia tutto finito.
        interno = uuid.uuid4().hex[:12]
        with _jobs_lock:
            _jobs[interno] = {"id": interno, "status": "running", "stage": "avvio",
                              "done": 0, "total": 0}
        try:
            funzione(interno, project_id, per_group)
        finally:
            with _jobs_lock:
                stato = dict(_jobs.pop(interno, {}))
        esito[passo] = "fatto" if stato.get("status") == "done" else (
            stato.get("error") or "non riuscito")
    return esito


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


def _advanced_stages_block_reason(project: Project) -> str:
    """Perche' i moduli devono ancora aspettare la verifica o la divisione del piano."""
    # Dopo lo sdoppiamento ciascuna meta' dichiara il proprio piano ed e' pronta. Vale sia
    # per la L rimasta nel progetto originale sia per la T derivata.
    if str(project.source.get("plane") or "") in ("L", "T"):
        return ""

    conteggi = (project.step_value("import").get("plane_counts") or {})
    if int(conteggi.get("L") or 0) and int(conteggi.get("T") or 0):
        return (
            "prima dividi le immagini L e T in due progetti: orientamento, depth e scala "
            "devono girare separatamente"
        )
    if project.is_biplane and not conteggi:
        return (
            "prima riconosci e controlla il piano L/T di ogni immagine nella sezione Import"
        )
    return ""


def _marker_nel_rettangolo(righe: Dict[str, Dict], rect: Dict,
                           margine: int = 60) -> Tuple[Dict[str, Dict], int]:
    """Solo le detection che cadono nel rettangolo ecografico, col suo margine.

    La banca cerca il glifo sulla meta' alta o bassa del frame **intero**. Su Esaote Nine
    (prova 5) prendeva una lettera della data in alto a destra, con 0.70 di punteggio, su
    ventotto immagini: l'envelope del gruppo NF diventava 752x195 px - mezzo schermo -
    invece dei venti pixel del pallino rosso. Un envelope cosi' e' un invito a sbagliare,
    perche' ESI dentro ci cerchera' il marker.

    Il rettangolo e' quello dell'abbozzo - lo studio raffinato viene dopo, perche' dipende
    proprio dall'orientamento - e per questo il taglio e' largo e non e' assoluto: se di un
    gruppo non resta niente si tengono le sue righe com'erano. Ci sono setup legacy, l'80 e
    l'81, dove il marker sta davvero fuori dal rettangolo, e perderli sarebbe peggio del
    rumore.
    """
    try:
        limiti = {
            "top": int(rect["top"]) - margine, "left": int(rect["left"]) - margine,
            "bottom": int(rect["bottom"]) + margine, "right": int(rect["right"]) + margine,
        }
    except (KeyError, TypeError, ValueError):
        return righe, 0

    def dentro(box: Dict) -> bool:
        cx = (int(box["left"]) + int(box["right"])) / 2.0
        cy = (int(box["top"]) + int(box["bottom"])) / 2.0
        return (limiti["left"] <= cx <= limiti["right"]
                and limiti["top"] <= cy <= limiti["bottom"])

    per_gruppo: Dict[str, List[str]] = {}
    for nome, riga in righe.items():
        per_gruppo.setdefault(str(riga.get("group") or ""), []).append(nome)

    tenute: Dict[str, Dict] = {}
    scartate = 0
    for gruppo, nomi in per_gruppo.items():
        buone = [n for n in nomi if dentro(righe[n]["box"])]
        if not buone:
            # Tutto il gruppo e' fuori: o il marker sta davvero li', o il rettangolo e'
            # sbagliato. In tutti e due i casi buttare via il gruppo e' la mossa peggiore.
            tenute.update({n: righe[n] for n in nomi})
            continue
        tenute.update({n: righe[n] for n in buone})
        scartate += len(nomi) - len(buone)
    return tenute, scartate


def _marker_vendor_exclusion(project: Project, margin_fraction: float = 0.75) -> Optional[Dict]:
    """Il template ecografo #13 non puo' mai essere scambiato per il marker."""
    box = project.step_value("vendor").get("rect_name_echo") or {}
    try:
        top, left = int(box["top"]), int(box["left"])
        bottom, right = int(box["bottom"]), int(box["right"])
    except (KeyError, TypeError, ValueError):
        return None
    width, height = project.step_value("import").get("image_sample_size") or [0, 0]
    dx = int(round((right - left + 1) * float(margin_fraction)))
    dy = int(round((bottom - top + 1) * float(margin_fraction)))
    return {
        "top": max(0, top - dy),
        "left": max(0, left - dx),
        "bottom": min(int(height) - 1, bottom + dy) if height else bottom + dy,
        "right": min(int(width) - 1, right + dx) if width else right + dx,
    }


MODULI_AVANZATI = ("orientamento", "depth", "scala")


def _righe_depth_dal_disco(project: Project) -> List[Dict]:
    """Le righe del modulo depth come le ha lasciate l'ultima volta che e' girato."""
    stage = _depth_stage_dir(project)
    if stage is None:
        return []
    import csv as _csv

    percorso = stage / "rect_depth_autonomous_predictions.csv"
    if not percorso.is_file():
        return []
    with percorso.open("r", encoding="utf-8", newline="") as handle:
        return list(_csv.DictReader(handle))


def _run_advanced_stages(
    job_id: str, project_id: str, sample: int, marker_min_score: float = 0.55,
    base_result: Optional[Dict] = None,
    quali: Sequence[str] = MODULI_AVANZATI,
) -> None:
    """Orientation marker, depth and scale, run as the pipeline runs them - one at a time.

    `quali` dice quali far girare: ognuno ha il suo comando nella sua sezione, e quello che
    non gira resta com'era, artefatti e valore dello step compresi."""
    try:
        project = _project(project_id)
        piano_bloccato = _advanced_stages_block_reason(project)
        if piano_bloccato:
            raise ValueError(piano_bloccato)
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
        marker_exclusion = _marker_vendor_exclusion(project)

        out_root = project.root / "stages"
        python_bin = sys.executable
        gia_fatti: Dict = dict((project.data.get("analysis") or {}).get("stages") or {})
        results: Dict = {}
        # The subprocess modules scan a folder: they get the mirror with only unique frames.
        scan_folder_for_modules = project.dedup_link_dir() or folder
        base_moduli = scan_folder_for_modules
        results["images_source"] = {
            "deduplicated": scan_folder_for_modules != folder,
            "folder": str(scan_folder_for_modules),
            "unique_images": len(project.dedup_names()),
        }

        # I tre moduli si lanciano uno per uno: ognuno ha il suo comando nella sua
        # sezione. Rifare l'orientamento non deve costringere a rifare anche la scala,
        # che sono minuti di lavoro su cose che non sono cambiate.
        if "orientamento" in quali:
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
                vendor=vendor, exclusion_rect=marker_exclusion, max_images=0, timeout=3600.0,
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
                useful_names = {
                    str(path.relative_to(scan_folder_for_modules)) for path in useful
                }
                rows_by_name = {
                    name: row for name, row in rows_by_name.items() if name in useful_names
                }
                # E dentro al rettangolo: la banca cerca sulla meta' alta o bassa del frame
                # intero, e cio' che trova fuori dall'immagine ecografica non e' il marker.
                rows_by_name, fuori_rect = _marker_nel_rettangolo(rows_by_name, final_rect)
                results["marker"]["outside_rect"] = fuori_rect
                template_info: Dict = {}
                override = previous.get("marker_override") or {}
                override_rows: Dict[str, Dict] = {}
                consegnate: Dict[str, Dict] = {}
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
                        exclusion_rect=marker_exclusion,
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
                            exclusion_rect=marker_exclusion,
                            out_dir=project.root / "templates" / "candidates", seed_images=6,
                        )
                        chosen = om.choose_by_coverage(
                            candidates=candidates, images=useful, folder=scan_folder_for_modules,
                            rect=final_rect, exclusion_rect=marker_exclusion,
                            bundle_dir=artifacts / "41_orientation_marker_detector_bundle",
                            sample=40,
                        )
                        if "chosen" in chosen:
                            source = Path(chosen["chosen"]["path"])
                            final = project.root / "templates" / "orientation_marker.png"
                            final.parent.mkdir(parents=True, exist_ok=True)
                            final.write_bytes(source.read_bytes())
                            template_info = {**chosen["chosen"], "path": str(final)}
                            # --- GLI ENVELOPE SONO DI CHI VERRA' CONSEGNATO ------------
                            # ESI cerchera' *questo* ritaglio, lo stesso in tutti e quattro i
                            # gruppi (nelle dieci configurazioni storiche i quattro
                            # `orientation_N.png` sono byte per byte identici). Se l'envelope
                            # lo costruisce la banca, che usa i suoi template, descrive dove
                            # sta un altro glifo: su prova 6 (GE Logiq E9) il ritaglio scelto
                            # e' «E9» ma l'envelope NF cadeva sulla riga di sopra, quella di
                            # «LOGIQ», e la validazione dava copertura 0.0 su ventiquattro
                            # immagini. Un envelope dove il marker non c'e' e' peggio di
                            # nessun envelope: ESI ci cerchera' dentro e non trovera' mai.
                            _job_update(job_id, stage="orientamento: dove si ritrova il ritaglio consegnato")
                            own = om.match_all(
                                images=useful, folder=scan_folder_for_modules,
                                template_path=final, rect=final_rect,
                                bundle_dir=artifacts / "41_orientation_marker_detector_bundle",
                                min_score=float(marker_min_score), search_margin=60,
                                exclusion_rect=marker_exclusion,
                            )
                            consegnate = {
                                row["name"]: {"score": row["score"], "group": row["group"],
                                              "box": row["box"]}
                                for row in own["rows"]
                                if row.get("box") and (row.get("score") or 0) >= float(marker_min_score)
                            }
                            if consegnate:
                                rows_by_name = {
                                    nome: {"group": r["group"], "box": r["box"]}
                                    for nome, r in consegnate.items()
                                }
                                results["marker"]["envelope_source"] = "ritaglio consegnato"
                                results["marker"]["delivered_hits"] = len(consegnate)
                            else:
                                # Non si ritrova da nessuna parte: meglio gli envelope della
                                # banca che nessun envelope, ma va detto.
                                results["marker"]["envelope_source"] = "banca (il ritaglio non si ritrova)"
                    except (FileNotFoundError, ImportError) as error:
                        template_info = {"error": str(error)}

                # le correzioni umane vincono sull'immagine corrispondente, anche dopo un
                # ricalcolo: sostituiscono la detection del modulo, non le si affiancano
                for name, fixed in corrections_before.items():
                    if name in useful_names and fixed.get("box") and fixed.get("group"):
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
                        exclusion_rect=marker_exclusion,
                        min_score=float(marker_min_score),
                        extra_templates=[
                            Path(path) for name, path in (previous.get("hint_paths") or {}).items()
                            if name in useful_names and Path(path).is_file()
                        ],
                    )

                order = marker_refine.GROUP_ORDER
                found = [g for g in order if g in groups]
                missing = [g for g in order if g not in groups]
                # Il ricalcolo non deve cancellare il lavoro umano: correzioni e ritagli dei
                # suggerimenti sopravvivono al rifacimento dello stadio.
                orientation_value: Dict = {
                    "corrections": corrections_before,
                    # Anche le detection che hai gia' guardato e buttato: sono lavoro tuo
                    # come le correzioni, e un «Rifai l'orientamento» le rimetteva tutte in
                    # gioco - a ricominciare daccapo dallo stesso rumore.
                    "refused": dict(previous.get("refused") or {}),
                    "marker_override": override,
                    "marker_rows_override": override_rows,
                    # Dove si ritrova il ritaglio consegnato: sono queste le posizioni da cui
                    # rifare gli envelope quando arriva una correzione, se no il primo
                    # ritocco li riporterebbe a quelli della banca.
                    "marker_rows_delivered": consegnate,
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

            # I due giri della catena che dipendono solo dall'orientamento appena fatto. Non
            # applicano niente: propongono, e la scelta resta sua. Ma se non girano qui, la
            # sezione del rettangolo resta vuota finche' qualcuno non preme un tasto - ed e'
            # proprio quello che non deve succedere.
            results["rect_chain"] = _giri_rettangolo(job_id, project_id)
        if "depth" in quali:
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
            tutti = list(project.dedup_names())
            _job_update(job_id, stage="depth: quale etichetta regge su tutta la cartella")
            scelta = _riferimento_che_regge(
                base_moduli, _righe_depth_per_nome(project, righe_depth), tutti,
            )
            results["depth"]["box_reference"] = "" if scelta is None else scelta[0]
            if scelta is None:
                # Nessuna etichetta regge: si porta via anche quella di un giro precedente,
                # altrimenti la sezione continua a mostrare le sue letture come se valessero.
                def butta_propagazione(_p: Project, value: Dict) -> Dict:
                    if (value.get("depth_box_template") or {}).get("scope") == "auto":
                        value.pop("depth_box_template", None)
                        value.pop("depth_box_reads", None)
                    return value

                _write_step(project_id, "depth_scale", butta_propagazione,
                            status="proposed", source="model")
            if scelta is not None:
                nome_rif, box_rif, fattore, quota = scelta
                _job_update(job_id, stage=f"depth: la stessa etichetta su {len(tutti)} immagini",
                            done=0, total=len(tutti))
                letture, falliti = _leggi_riquadro_su(
                    base_moduli, box_rif, fattore, tutti,
                    progress=lambda fatte, quante: _job_update(job_id, done=fatte),
                )
                # Dove l'etichetta non c'e' - un fotogramma di un'altra schermata - resta
                # comunque un numero, ed e' un numero qualsiasi. Fuori dai millimetri di una
                # profondita' e' meglio nessuna lettura che una sbagliata.
                fuori_scala = [n for n, l in letture.items()
                               if not 5.0 <= float(l.get("depth_mm") or 0) <= 500.0]
                for nome in fuori_scala:
                    letture.pop(nome, None)
                falliti = list(falliti) + fuori_scala
                if letture:
                    def salva_letture(_p: Project, value: Dict) -> Dict:
                        value["depth_box_template"] = {
                            "box": box_rif, "from": nome_rif, "unit_factor": fattore,
                            "scope": "auto", "applied": len(letture), "targets": len(tutti),
                            "failed": falliti, "checked_ratio": round(quota, 2),
                            "at": datetime.now().isoformat(timespec="seconds"),
                        }
                        value["depth_box_reads"] = letture
                        return value

                    _write_step(project_id, "depth_scale", salva_letture,
                                status="proposed", source="model")
                    results["depth"]["box_reads"] = len(letture)
                    results["depth"]["box_failed"] = len(falliti)

        if "scala" in quali:
            if "depth" not in quali:
                # La scala vive di cio' che la depth ha letto. Lanciata da sola se le
                # riprende dal progetto - sono le stesse letture che si vedono nella
                # sezione, correzioni comprese - invece di rifare la depth da capo.
                letture = dict(
                    project.step_value("depth_scale").get("depth_box_reads") or {})
                righe_depth = _righe_depth_dal_disco(project)
                if not letture and not righe_depth:
                    raise ValueError(
                        "la scala ha bisogno della depth: lanciala prima dalla sua sezione")
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
                    corrections=_write_scale_corrections(_project(project_id)),
                )
                results["scale_study"] = {
                    k: studio.get(k)
                    for k in ("status", "error", "frames", "by_status", "vendor", "zone",
                              "output_dir", "data_json")
                }

            parsed = stages_mod.parse_scale_lines(scale.get("lines") or {})
            if parsed.get("depths"):
                complete = bool(parsed["pixel_ratio_x"] and parsed["pixel_ratio_y"] and parsed["scale_lines"])

                def scrivi_scala(_p: Project, value: Dict) -> Dict:
                    # Si **aggiorna**, non si sostituisce: nello stesso step vivono anche la
                    # rilettura della depth su tutta la cartella e le correzioni dell'utente, e
                    # riscrivendo il valore per intero sparivano — la depth tornava alle dodici
                    # immagini del campione appena finiva lo stadio della scala.
                    # La depth puo' essere di un giro precedente: lanciata da sola, la
                    # scala non l'ha rifatta e il suo esito e' quello gia' registrato.
                    value.update({**parsed,
                                  "depth_module": results.get("depth") or gia_fatti.get("depth"),
                                  "scale_module": results["scale"]})
                    return value

                _write_step(project_id, "depth_scale", scrivi_scala,
                            status="proposed" if complete else "blocked", source="model")

        project = _project(project_id)
        # Si **aggiunge** a quello che c'era: chi ha rifatto solo l'orientamento deve
        # ritrovare intatti gli esiti di depth e scala, non un riquadro vuoto al loro posto.
        stages = {**gia_fatti, **results}
        project.data.setdefault("analysis", {})["stages"] = stages
        project.save()
        _job_update(job_id, status="done", stage="fatto",
                    result={**(base_result or {}), "stages": stages})
    except Exception as error:  # noqa: BLE001
        _job_update(job_id, status="error", stage="errore", error=str(error))


@app.post("/api/projects/<project_id>/analyze_stages")
def api_analyze_stages(project_id: str):
    """Run one of the three modules - or all of them - once the plane is resolved."""
    project = _project(project_id)
    piano_bloccato = _advanced_stages_block_reason(project)
    if piano_bloccato:
        return jsonify({"error": piano_bloccato}), 409
    payload = _payload()
    # Ogni modulo si lancia dalla sua sezione. Senza indicazioni girano tutti, com'era.
    quali = [str(nome) for nome in (payload.get("stages") or MODULI_AVANZATI)]
    sconosciuti = [nome for nome in quali if nome not in MODULI_AVANZATI]
    if sconosciuti or not quali:
        return jsonify({"error": f"moduli sconosciuti: {', '.join(sconosciuti) or 'nessuno'}"}), 400
    sample = int(payload.get("sample") or 12)
    # Soglia della validazione: la tolleranza legacy e' molto piu' lenta di quanto sembri
    # (fino a 0.368 di correlazione in setup_53), quindi 0.85 escluderebbe match buoni.
    min_score = float(payload.get("marker_min_score") or 0.55)
    return jsonify(
        {"job_id": _start_job(_run_advanced_stages, project_id, sample, min_score, None, quali)}
    )


def _start_job(target, *args) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"id": job_id, "status": "running", "stage": "avvio", "done": 0, "total": 0}
    threading.Thread(target=target, args=(job_id, *args), daemon=True).start()
    return job_id


@app.post("/api/projects/<project_id>/analyze")
def api_analyze(project_id: str):
    """Re-run the initial analysis, stopping before orientation, depth and scale."""
    project = _project(project_id)
    folder = project.source.get("folder") or ""
    if not folder:
        return jsonify({"error": "importa prima una cartella"}), 400
    payload = _payload()
    sample = int(payload.get("sample") or 24)
    return jsonify({"job_id": _start_job(_run_import_analysis, project_id, folder, sample)})


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


@app.after_request
def _niente_cache(risposta):  # noqa: ANN001
    """La pagina e gli script non si mettono in cache.

    Il browser teneva `index.html` e con lui la lista degli script: aggiungendone uno nuovo
    (`scale_view.js`) la pagina restava senza, la funzione della sezione non esisteva e il
    pannello si fermava senza dire niente. Qui l'app la si ricarica di continuo mentre si
    lavora: la cache non fa guadagnare nulla e nasconde le modifiche.
    """
    if request.path == "/" or request.path.startswith("/static/"):
        risposta.headers["Cache-Control"] = "no-store, must-revalidate"
    return risposta


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


@app.delete("/api/projects/<project_id>")
def api_project_delete(project_id: str):
    """Butta via un progetto: lo studio, non le immagini.

    Cancella la cartella del progetto sotto la radice dei progetti - `project.json`, lo
    specchio di lavoro, i template ritagliati, gli artefatti dei moduli. La cartella delle
    acquisizioni non si tocca: quella non e' nostra.

    Il controllo che sia davvero **dentro** la radice non e' pedanteria: qui si cancella
    ricorsivamente, e un percorso che esce dalla radice sarebbe un disastro silenzioso.
    """
    project = _project(project_id)
    radice = Path(_projects_root).resolve()
    cartella = Path(project.root).resolve()
    if cartella.parent != radice or cartella == radice:
        return jsonify({"error": "questo progetto non sta dove dovrebbe: non lo tocco"}), 400

    # Se e' la meta' di uno sdoppiamento, l'altra meta' resta orfana: lo si dice a lei, e si
    # toglie il riferimento invece di lasciarlo puntare al vuoto.
    orfani = []
    for altro in list_projects(_projects_root):
        if altro.get("project_id") == project_id:
            continue
        try:
            vicino = _project(altro["project_id"])
        except FileNotFoundError:
            continue
        cambiato = False
        if str(vicino.source.get("split_into") or "") == project_id:
            vicino.source.pop("split_into", None)
            cambiato = True
        if str(vicino.source.get("derived_from") or "") == project_id:
            vicino.source["derived_from"] = ""
            cambiato = True
        if cambiato:
            vicino.save()
            orfani.append(vicino.root.name)

    shutil.rmtree(cartella, ignore_errors=True)
    return jsonify({"deleted": project_id, "unlinked": orfani})


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
    stages_blocked_reason = _advanced_stages_block_reason(project)
    return jsonify(
        {
            "project": project.data,
            "status": project.status_report(),
            "is_biplane": project.is_biplane,
            "advanced_stages": {
                "ready": not stages_blocked_reason,
                "blocked_reason": stages_blocked_reason,
            },
            "split_pending": _divisione_da_riapplicare(project),
            # La selezione della rotazione mostra solo un lotto: non si trasferiscono migliaia
            # di nomi nel JSON di ogni refresh, ma si puo' comunque correggere una o piu'
            # immagini alla volta (le altre si raggiungono dalla cartella).
            "rotation_images": project.dedup_names()[:240],
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
    """Initial analysis through the rough rectangle and L/T proposal."""
    _project(project_id)
    payload = _payload()
    folder = (payload.get("folder") or "").strip()
    if not folder:
        return jsonify({"error": "indica la cartella delle acquisizioni"}), 400
    if not Path(folder).expanduser().is_dir():
        return jsonify({"error": f"cartella non valida: {folder}"}), 400
    sample = int(payload.get("sample") or 24)
    return jsonify({"job_id": _start_job(_run_import_analysis, project_id, folder, sample)})


@app.post("/api/projects/<project_id>/import/rotation")
def api_import_rotation(project_id: str):
    """Correggi la rotazione dell'intera cartella o di frame selezionati.

    La riscrittura riguarda soltanto lo specchio di lavoro del progetto; i file originali
    restano sempre intatti. Poi si rilancia la pipeline, che aggiorna solo le proposte non
    confermate e conserva le eccezioni per immagine.
    """
    project = _project(project_id)
    payload = _payload()
    try:
        angle = int(payload.get("angle")) % 360
    except (TypeError, ValueError):
        return jsonify({"error": "scegli 0°, 90°, 180° o 270°"}), 400
    if angle not in (0, 90, 180, 270):
        return jsonify({"error": "scegli 0°, 90°, 180° o 270°"}), 400
    names = [str(n) for n in (payload.get("names") or []) if str(n)]
    disponibili = set(project.dedup_names())
    sconosciuti = [name for name in names if name not in disponibili]
    if sconosciuti:
        return jsonify({"error": "una delle immagini selezionate non appartiene al progetto"}), 400
    if names:
        overrides = dict(project.source.get("rotation_overrides") or {})
        for name in names:
            overrides[name] = angle
        project.source["rotation_overrides"] = overrides
    else:
        # Senza selezione e' una scelta per tutta la cartella; 0 e' un'opzione esplicita,
        # non un fallimento del riconoscimento.
        project.source["rotation_applied"] = angle
        project.source["rotation_source"] = "user"
        project.source["rotation_overrides"] = {}
    project.save()
    folder = str(project.source.get("folder") or "")
    if not Path(folder).is_dir():
        return jsonify({"error": "cartella non raggiungibile"}), 400
    return jsonify({"job_id": _start_job(_run_import_analysis, project_id, folder, 24)})


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


@app.post("/api/projects/<project_id>/steps/<step_id>/confirm")
def api_step_confirm(project_id: str, step_id: str):
    """Conferma uno step **senza toccarne il valore**.

    `POST /steps/<id>` sostituisce il valore con quello che il pannello ha in mano, ed e'
    giusto per gli step che si compilano a mano. Per gli step scritti dai moduli e'
    un'arma puntata ai piedi: dentro a `depth_scale` vivono la rilettura su tutta la
    cartella, i riquadri stretti e le correzioni, e il pannello ne ha una copia vecchia di
    quando e' stato disegnato. Confermare non deve poter cancellare niente.
    """
    project = _project(project_id)
    if step_id not in project.steps:
        return jsonify({"error": f"step sconosciuto: {step_id}"}), 400
    annulla = bool(_payload().get("reset"))
    _write_step(project_id, step_id, lambda _p, v: v,
                status="proposed" if annulla else "confirmed",
                source="model" if annulla else "user")
    return jsonify({"confirmed": not annulla, "status": _project(project_id).status_report()})


# --- il piano L/T, e i due progetti che ne nascono ------------------------
#
# Una cartella di acquisizioni puo' contenere due piani: la sonda e' la stessa, ma le
# immagini in L e quelle in T sono due studi diversi e vanno in due configurazioni diverse.
# Finche' stanno insieme ogni modulo le mescola: il rettangolo esce mediato fra due ventagli,
# la depth su due scale, l'orientamento su due geometrie.
#
# Il piano si riconosce **dopo** dedup e rotazione, perche' la rete guarda dentro al
# rettangolo e il rettangolo si legge su immagini gia' dritte.

STEP_EREDITATI = ("codes", "vendor", "probe", "rect")
# Della scala si eredita quello che e' della macchina, non quello che e' del fotogramma:
# il rapporto pixel/mm e la riga della scala restano, le depth lette immagine per immagine
# no - quelle immagini nel progetto nuovo non ci sono nemmeno.
CHIAVI_SCALA_EREDITATE = ("pixel_ratio_x", "pixel_ratio_y", "scale_lines", "scale_module")


def _piani_salvati(project: Project) -> Dict[str, str]:
    """Il piano di ogni immagine: quello riconosciuto, con sopra le correzioni a mano."""
    valore = project.step_value("import")
    piani = dict((valore.get("planes") or {}))
    for nome, scelto in (valore.get("plane_corrections") or {}).items():
        piani[nome] = {"plane": scelto, "confidence": None, "source": "user"}
    return piani


def _divisione_da_riapplicare(project: Project) -> Dict:
    """Quante immagini stanno nel progetto sbagliato rispetto alla divisione di adesso.

    Correggere un piano scrive la correzione e basta: i due progetti restano com'erano
    finche' non si riconferma la divisione. Senza dirlo, una correzione sembra fatta e
    invece le immagini dell'altro piano continuano a girare nei moduli - e' cosi' che
    cinquantasei T sono rimaste nel progetto L di Esaote.
    """
    gemello = str(project.source.get("split_into") or project.source.get("derived_from") or "")
    mio = str(project.source.get("plane") or "")
    if not gemello or mio not in ("L", "T"):
        return {}
    piani = _piani_salvati(project)
    if not piani:
        return {}
    qui = set(project.dedup_names())
    try:
        la = set(_project(gemello).dedup_names())
    except FileNotFoundError:
        return {}
    def piano_di(nome: str) -> str:
        return str((piani.get(nome) or {}).get("plane") or "")

    # Le immagini senza piano stanno con la L, che e' il progetto principale.
    altro = "T" if mio == "L" else "L"
    da_mandare = sorted(n for n in qui if piano_di(n) == altro)
    da_prendere = sorted(n for n in la if (piano_di(n) or "L") == mio)
    if not da_mandare and not da_prendere:
        return {}
    return {"to_send": len(da_mandare), "to_take": len(da_prendere),
            "examples": [n.split("/")[-1] for n in (da_mandare + da_prendere)[:3]]}


@app.post("/api/projects/<project_id>/planes")
def api_planes(project_id: str):
    """Riconosci il piano L/T di ogni immagine della cartella."""
    _project(project_id)
    return jsonify({"job_id": _start_job(_run_planes, project_id)})


def _predict_planes(job_id: str, project_id: str) -> Dict[str, int]:
    """Classifica ogni frame L/T e restituisce i conteggi, senza chiudere il job chiamante."""
    project = _project(project_id)
    rect = project.step_value("rect").get("rect_echo")
    if not rect:
        raise ValueError("serve prima il rettangolo ecografico: la rete L/T guarda li' dentro")
    base = project.working_dir()
    immagini = project.dedup_images()
    if not immagini:
        raise ValueError("nessuna immagine da classificare")
    _job_update(job_id, stage=f"piano L/T su {len(immagini)} immagini", total=len(immagini))
    righe = _inference_engine().predict_lt_each(
        immagini, rect,
        progress=lambda fatte, quante: _job_update(job_id, done=fatte, total=quante),
    )
    piani: Dict[str, Dict] = {}
    for riga in righe:
        percorso = Path(riga["path"])
        try:
            nome = str(percorso.relative_to(base))
        except ValueError:
            nome = percorso.name
        piani[nome] = {"plane": riga.get("plane"), "confidence": riga.get("confidence")}
    conteggi: Dict[str, int] = {}
    for voce in piani.values():
        chiave = voce.get("plane") or "?"
        conteggi[chiave] = conteggi.get(chiave, 0) + 1

    def mutate(_p: Project, value: Dict) -> Dict:
        # Una correzione fatta fra un riconoscimento e l'altro resta la voce piu' forte.
        value["planes"] = piani
        value["plane_counts"] = conteggi
        return value

    _write_step(project_id, "import", mutate, status="proposed", source="model")
    return conteggi


def _run_planes(job_id: str, project_id: str) -> None:
    try:
        conteggi = _predict_planes(job_id, project_id)
        _job_update(job_id, status="done", stage="fatto", result={"counts": conteggi})
    except Exception as error:  # noqa: BLE001
        _job_update(job_id, status="error", stage="errore", error=str(error))


def _automatic_stages(project: Project) -> List[str]:
    """Gli stadi che possono essere aggiornati senza sostituire lavoro gia' confermato.

    `user_edited` sopravvive allo stato `stale`: e' importante, perche' una conferma
    diventa stale quando cambia il rettangolo ma non autorizza il programma a cancellarla.
    """
    if _advanced_stages_block_reason(project):
        return []
    orientamento = project.steps.get("orientation") or {}
    depth = project.steps.get("depth_scale") or {}
    auto_orientation = not bool(orientamento.get("user_edited"))
    auto_depth = not bool(depth.get("user_edited"))
    stages: List[str] = []
    if auto_orientation:
        stages.append("orientamento")
    if auto_depth:
        stages.extend(("depth", "scala"))
    return stages


_automatic_pipeline_lock = threading.Lock()
_automatic_pipeline_active: set[str] = set()


def _run_auto_pipeline(job_id: str, project_id: str) -> None:
    """Piano L/T, poi i moduli non confermati; tutto senza dover premere altri pulsanti."""
    with _automatic_pipeline_lock:
        if project_id in _automatic_pipeline_active:
            _job_update(job_id, status="done", stage="gia' in esecuzione", result={"skipped": True})
            return
        _automatic_pipeline_active.add(project_id)
    try:
        project = _project(project_id)
        # Il riconoscimento L/T e' utile anche sulle sonde singole come controllo visivo;
        # solo una sonda biplana con entrambi i piani blocca i moduli in attesa dello split.
        _predict_planes(job_id, project_id)
        project = _project(project_id)
        reason = _advanced_stages_block_reason(project)
        if reason:
            _job_update(job_id, status="done", stage="attendo divisione L/T", result={"blocked": reason})
            return
        stages = _automatic_stages(project)
        if not stages:
            _job_update(job_id, status="done", stage="moduli gia' confermati", result={"skipped": True})
            return
        _run_advanced_stages(job_id, project_id, 12, 0.55, {"automatic": True}, stages)
    except Exception as error:  # noqa: BLE001
        _job_update(job_id, status="error", stage="errore", error=str(error))
    finally:
        with _automatic_pipeline_lock:
            _automatic_pipeline_active.discard(project_id)


def _schedule_auto_pipeline(project_id: str) -> Optional[str]:
    """Accoda la continuazione automatica; il worker coalesce le richieste concorrenti."""
    try:
        _project(project_id)
    except FileNotFoundError:
        return None
    return _start_job(_run_auto_pipeline, project_id)


@app.post("/api/projects/<project_id>/planes/correct")
def api_planes_correct(project_id: str):
    """Correggi a mano il piano di una o piu' immagini.

    Quando la divisione sbaglia, di solito sbaglia in blocco - un'acquisizione intera
    finita dalla parte storta - e correggerle una per una e' N volte lo stesso gesto.
    """
    _project(project_id)
    payload = _payload()
    nomi = [str(n).strip() for n in (payload.get("names") or []) if str(n).strip()]
    if not nomi:
        uno = str(payload.get("name") or "").strip()
        nomi = [uno] if uno else []
    piano = str(payload.get("plane") or "").strip().upper()
    if not nomi:
        return jsonify({"error": "manca l'immagine"}), 400
    if not payload.get("reset") and piano not in ("L", "T"):
        return jsonify({"error": "il piano puo' essere L o T"}), 400

    def mutate(_p: Project, value: Dict) -> Dict:
        fatte = dict(value.get("plane_corrections") or {})
        for nome in nomi:
            if payload.get("reset"):
                fatte.pop(nome, None)
            else:
                fatte[nome] = piano
        value["plane_corrections"] = fatte
        return value

    _write_step(project_id, "import", mutate, status="corrected", source="user")
    return jsonify({"saved": True, "count": len(nomi)})


@app.post("/api/projects/<project_id>/split")
def api_split_planes(project_id: str):
    """Da una cartella con due piani, due progetti: la L resta qui, la T nasce accanto.

    Il progetto nuovo non riparte da zero: eredita quello che e' della **macchina** e non
    del piano - i codici, il vendor col suo template, la sonda, il rettangolo, e della scala
    il rapporto pixel/mm. E' una copia, non un legame: da qui in poi le due configurazioni
    vivono per conto loro, e resta scritto da dove viene.
    """
    project = _project(project_id)
    piani = _piani_salvati(project)
    if not piani:
        return jsonify({"error": "il piano non e' ancora stato riconosciuto"}), 400

    # L'insieme da dividere non e' «le immagini di questo progetto»: dopo il primo
    # sdoppiamento qui dentro ci sono solo le L, e riconfermare avrebbe diviso quelle -
    # buttando via le T invece di rimetterle al loro posto. L'insieme sono le immagini dei
    # due progetti insieme, che e' la cartella deduplicata.
    tutte = list(project.dedup_names())
    gia_esistente = str(project.source.get("split_into") or "")
    if gia_esistente:
        try:
            tutte = sorted(set(tutte) | set(_project(gia_esistente).dedup_names()))
        except FileNotFoundError:
            tutte = sorted(set(tutte) | set(piani.keys()))
    nomi = {"L": [], "T": []}
    senza = []
    for nome in tutte:
        piano = (piani.get(nome) or {}).get("plane")
        if piano in ("L", "T"):
            nomi[piano].append(nome)
        else:
            senza.append(nome)
    if not nomi["L"] or not nomi["T"]:
        return jsonify({"error": "in questa cartella c'e' un piano solo: non c'e' niente da "
                                 "sdoppiare"}), 400

    # Le immagini senza piano restano con la L, che e' il progetto principale: buttarle
    # sarebbe peggio che tenerle dove si possono ancora guardare.
    nomi["L"].extend(senza)

    # Se lo sdoppiamento c'e' gia' stato, si **aggiorna** quello che c'e' invece di aprire
    # un terzo progetto. Correggere due immagini e ritrovarsi tre configurazioni sarebbe
    # peggio del problema che si stava sistemando.
    nome_base = str(project.codes.get("project_name") or project.root.name)
    gia = str(project.source.get("split_into") or "")
    nuovo = None
    if gia:
        try:
            nuovo = _project(gia)
        except FileNotFoundError:
            nuovo = None
    aggiornato = nuovo is not None
    if nuovo is None:
        nuovo = Project.create(_projects_root, f"{nome_base} T")
    nuovo.source.update({
        **{k: v for k, v in project.source.items() if k != "split_into"},
        "plane": "T",
        "images_total": len(nomi["T"]),
        "derived_from": project.root.name,
        "derived_at": datetime.now().isoformat(timespec="seconds"),
    })
    nuovo.codes.update({k: v for k, v in project.codes.items() if k != "project_name"})
    nuovo.codes.setdefault("project_name", f"{nome_base} T")
    nuovo.save()
    nuovo.save_dedup_images(sorted(nomi["T"]))

    def _import_del_piano(valore: Dict, elenco: Sequence[str]) -> Dict:
        """Il riepilogo della cartella riscritto per il piano che quel progetto ha davvero.

        Senza, dopo lo sdoppiamento la scheda continuava a dire «56 immagini tenute» in un
        progetto che ne ha 28 - il numero della cartella, non del progetto.
        """
        fuori = dict(valore)
        fuori["images_total"] = len(elenco)
        fuori["images"] = [{"name": n} for n in list(elenco)[:60]]
        fuori["images_listed"] = min(len(elenco), 60)
        return fuori

    def _anteprima_valida(valore: Dict, disponibili: Sequence[str]) -> Dict:
        """L'anteprima deve essere un'immagine che quel progetto ha davvero.

        Dopo lo sdoppiamento quella di prima puo' essere finita dall'altra parte: il
        pannello mostrerebbe un buco, e i riquadri disegnati sopra non si vedrebbero.
        """
        attuale = valore.get("preview_image")
        if attuale and attuale in disponibili:
            return valore
        if disponibili:
            valore = dict(valore)
            valore["preview_image"] = disponibili[0]
        return valore

    nomi_t = sorted(nomi["T"])
    for step_id in STEP_EREDITATI:
        if step_id == "codes":
            continue
        # Su un progetto che esiste gia' si eredita solo cio' che non e' ancora stato
        # guardato: quello che ha gia' sistemato lei di la' vale piu' della copia di qua.
        suo = nuovo.step_value(step_id) or {}
        stato_suo = str((nuovo.steps.get(step_id) or {}).get("status") or "")
        if aggiornato and suo and stato_suo not in ("", "empty", "proposed"):
            continue
        valore = dict(project.step_value(step_id) or {})
        if valore:
            nuovo.set_step(step_id, _anteprima_valida(valore, nomi_t),
                           status="proposed", source="ereditato")
    nuovo.set_step("import", _import_del_piano(project.step_value("import") or {}, nomi_t),
                   status="proposed", source="ereditato")
    scala = project.step_value("depth_scale") or {}
    ereditata = {k: scala[k] for k in CHIAVI_SCALA_EREDITATE if scala.get(k)}
    stato_scala = str((nuovo.steps.get("depth_scale") or {}).get("status") or "")
    if ereditata and not (aggiornato and stato_scala not in ("", "empty", "proposed")):
        nuovo.set_step("depth_scale", ereditata, status="proposed", source="ereditato")
    nuovo.save()
    # Lo specchio di lavoro va rifatto da zero: aggiornando una divisione gia' fatta le
    # immagini sono altre, e il timbro `.built_from` cambierebbe solo se cambia il numero.
    specchio_t = nuovo.root / nuovo.DEDUP_LINKS
    if specchio_t.exists():
        shutil.rmtree(specchio_t, ignore_errors=True)
    nuovo.dedup_link_dir()

    # E la L resta qui, con le sue sole immagini - anteprime comprese: anche di qua quella
    # di prima puo' essere una che ora sta di la'.
    project = _project(project_id)
    nomi_l = sorted(nomi["L"])
    for step_id in STEP_EREDITATI:
        if step_id == "codes":
            continue
        valore = dict(project.step_value(step_id) or {})
        if valore and valore.get("preview_image") not in nomi_l:
            stato = (project.steps.get(step_id) or {})
            project.set_step(step_id, _anteprima_valida(valore, nomi_l),
                             status=stato.get("status") or "proposed",
                             source=stato.get("source") or "model")
    project.save_dedup_images(nomi_l)
    stato_import = (project.steps.get("import") or {})
    project.set_step("import", _import_del_piano(project.step_value("import") or {}, nomi_l),
                     status=stato_import.get("status") or "proposed",
                     source=stato_import.get("source") or "import")
    project.source.update({"plane": "L", "images_total": len(nomi["L"]),
                           "split_into": nuovo.root.name})
    project.save()
    specchio = project.root / project.DEDUP_LINKS
    if specchio.exists():
        shutil.rmtree(specchio, ignore_errors=True)
    project.dedup_link_dir()

    automatic_jobs = {
        "L": _schedule_auto_pipeline(project_id),
        "T": _schedule_auto_pipeline(nuovo.root.name),
    }
    return jsonify({"created": nuovo.root.name, "updated": aggiornato,
                    "L": len(nomi["L"]), "T": len(nomi["T"]),
                    "without_plane": len(senza),
                    "inherited": list(STEP_EREDITATI) + (["scala"] if ereditata else []),
                    "advanced_jobs": automatic_jobs})


@app.post("/api/projects/<project_id>/import/timestamp")
def api_import_timestamp(project_id: str):
    """L'area dell'orologio, e la deduplicazione rifatta ignorandola.

    Due fotogrammi della stessa scena presi a un secondo di distanza differiscono in ogni
    byte del file e in nessun pixel tranne l'ora. Finche' non si dice dove sta quell'ora,
    nessuna misura li trova uguali - e restano tutti e due, a pesare due volte in ogni
    mediana dei moduli.
    """
    project = _project(project_id)
    payload = _payload()
    folder = (project.source.get("folder") or "").strip()
    if not folder or not Path(folder).is_dir():
        return jsonify({"error": "cartella non raggiungibile"}), 400
    if payload.get("reset"):
        box = None
    else:
        grezzo = payload.get("box") or {}
        try:
            box = {k: int(round(float(grezzo[k]))) for k in ("top", "left", "bottom", "right")}
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "serve un riquadro completo (top/left/bottom/right)"}), 400
        if box["right"] - box["left"] < 4 or box["bottom"] - box["top"] < 4:
            return jsonify({"error": "l'area dell'orologio e' troppo piccola"}), 400
    rilevamento = {
        "box": dict(box) if box else None,
        "confidence": 1.0,
        "reliable": bool(box),
        "source": "user" if box else "user_disabled",
        "support": None,
        "samples": None,
        "texts": [],
        "reason": ("area corretta dall'utente" if box
                   else "riconoscimento automatico disattivato dall'utente"),
    }
    return jsonify({
        "job_id": _start_job(
            _run_dedup, project_id, folder, box, rilevamento, box is None
        )
    })


@app.post("/api/projects/<project_id>/import/timestamp/detect")
def api_import_timestamp_detect(project_id: str):
    """Rilegge data/ora e, quando la proposta e' affidabile, rifà la dedup."""
    project = _project(project_id)
    folder = (project.source.get("folder") or "").strip()
    if not folder or not Path(folder).is_dir():
        return jsonify({"error": "cartella non raggiungibile"}), 400
    return jsonify({"job_id": _start_job(_run_timestamp_detection, project_id, folder)})


def _timestamp_preview(project: Project, detection: Dict) -> Dict:
    """Converte il percorso OCR assoluto nel nome relativo accettato dall'API immagini."""
    result = dict(detection)
    path = result.pop("preview_path", None)
    if not path:
        return result
    base = project.working_dir()
    try:
        result["preview_image"] = str(Path(path).relative_to(base))
    except (TypeError, ValueError):
        result["preview_image"] = Path(path).name
    return result


def _run_timestamp_detection(job_id: str, project_id: str, folder: str) -> None:
    """Trova l'orologio sullo specchio raddrizzato; applica solo proposte affidabili."""
    try:
        project = _project(project_id)
        images = project.dedup_images()
        if not images:
            raise FileNotFoundError("nessuna immagine da leggere: rifai prima l'import")
        _job_update(job_id, stage="riconoscimento automatico di data e ora")
        detection = _timestamp_preview(project, detect_timestamp_box(images))
        box = detection.get("box") if detection.get("reliable") else None
        if box:
            _run_dedup(job_id, project_id, folder, box, detection, False)
            return

        def mutate(_project_value: Project, value: Dict) -> Dict:
            if value.get("timestamp_box"):
                # Una nuova lettura incerta non deve disattivare un'area gia' applicata.
                # Si conserva la provenienza dell'area attiva e si registra solo il tentativo.
                active = dict(value.get("timestamp_detection") or {})
                active["last_attempt"] = detection
                value["timestamp_detection"] = active
            else:
                value["timestamp_detection"] = detection
            value["timestamp_disabled"] = False
            return value

        _write_step(project_id, "import", mutate, status="proposed", source="model")
        _job_update(job_id, status="done", stage="proposta da controllare", result={
            "applied": False,
            "detection": detection,
            "kept": len(project.dedup_names()),
        })
    except Exception as error:  # noqa: BLE001
        _job_update(job_id, status="error", stage="errore", error=str(error))


def _run_dedup(job_id: str, project_id: str, folder: str, box: Optional[Dict],
               detection: Optional[Dict] = None, disabled: bool = False) -> None:
    """Rifa' solo la deduplicazione, senza toccare il resto dell'analisi."""
    try:
        # Prima si ricostruisce l'insieme degli esatti dalla sorgente, poi si applica il box
        # allo specchio raddrizzato. Il box e' disegnato sull'immagine dritta: usarlo sui file
        # originali era sbagliato per ogni cartella con rotazione diversa da zero.
        _job_update(job_id, stage="rileggo le immagini e tolgo le copie identiche")
        esito = import_folder(Path(folder))
        project = _project(project_id)
        names = list(esito.pop("kept_names", []) or [])
        plane = str(project.source.get("plane") or "")
        if plane in ("L", "T"):
            labels = _piani_salvati(project)
            own = [name for name in names if (labels.get(name) or {}).get("plane") == plane]
            if own:
                names = own
        project.save_dedup_images(names)
        # Lo specchio di lavoro va rifatto: contiene una copia (o un link) per immagine
        # tenuta, e adesso ne sono di meno.
        specchio = project.root / project.DEDUP_LINKS
        if specchio.exists():
            shutil.rmtree(specchio, ignore_errors=True)
        project.dedup_link_dir(
            progress=lambda fatte, quante: _job_update(
                job_id, stage="raddrizzo le immagini da confrontare", done=fatte, total=quante)
        )

        timestamp_removed: List[Dict] = []
        if box:
            project = _project(project_id)
            base = project.working_dir()
            names = project.dedup_names()
            _job_update(job_id, stage="confronto le immagini senza data e ora",
                        done=0, total=len(names))
            kept, removed = deduplicate(
                [base / name for name in names], box,
                progress=lambda fatte, quante: _job_update(
                    job_id, stage="confronto le immagini senza data e ora",
                    done=fatte, total=quante),
            )
            final_names = sorted(str(Path(path).relative_to(base)) for path in kept)
            timestamp_removed = [row for row in removed if row["kind"] == "solo_timestamp"]
            project.save_dedup_images(final_names)
            names = final_names
            if specchio.exists():
                shutil.rmtree(specchio, ignore_errors=True)

        duplicates = dict(esito.get("duplicates") or {})
        duplicates["timestamp"] = timestamp_removed
        esito["duplicates"] = duplicates
        esito["images_total"] = len(names)
        esito["duplicates_removed"] = (
            len(duplicates.get("identical") or []) + len(timestamp_removed)
        )

        def mutate(_p: Project, value: Dict) -> Dict:
            value.update({
                "timestamp_box": dict(box) if box else None,
                "timestamp_detection": detection or value.get("timestamp_detection") or {},
                "timestamp_disabled": bool(disabled),
                "duplicates": esito.get("duplicates") or {},
                "duplicates_removed": esito.get("duplicates_removed", 0),
                "images_total": esito.get("images_total", 0),
                "images_total_raw": esito.get("images_total_raw", 0),
            })
            return value

        source = "model" if (detection or {}).get("source") == "ocr" else "user"
        _write_step(project_id, "import", mutate, status="proposed", source=source)
        progetto = _project(project_id)
        progetto.source.update({
            "images_total": esito.get("images_total", 0),
            "duplicates_removed": esito.get("duplicates_removed", 0),
        })
        progetto.save()
        progetto.dedup_link_dir()      # ricostruisce lo specchio, ruotato se serve
        doppie = esito.get("duplicates") or {}
        # L'insieme dei frame e' cambiato: tutte le proposte a valle che non sono state
        # confermate vanno aggiornate sul nuovo insieme. Il job separato evita di tenere la
        # UI bloccata mentre l'utente guarda il nuovo conteggio della dedup.
        automatic_job = _schedule_auto_pipeline(project_id)
        _job_update(job_id, status="done", stage="fatto", result={
            "applied": bool(box),
            "detection": detection or {},
            "kept": esito.get("images_total", 0),
            "identical": len(doppie.get("identical") or []),
            "timestamp": len(doppie.get("timestamp") or []),
            "advanced_job_id": automatic_job,
        })
    except Exception as error:  # noqa: BLE001
        _job_update(job_id, status="error", stage="errore", error=str(error))


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
        return jsonify({"error": "lancia prima l'orientamento dal suo step"}), 400

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
    # Gli envelope si rifanno da cio' che c'e' adesso - detection del ritaglio consegnato,
    # correzioni, rifiuti - e se sono cambiati si riscrivono. Mostrarne di diversi da quelli
    # salvati vorrebbe dire far vedere una cosa e scriverne un'altra in #16: gli envelope
    # dello step sono quelli che finiscono nel file.
    rifatti = _rebuild_orientation(project, stored)["groups"]
    if rifatti and rifatti != (stored.get("groups") or {}):
        def allinea(_p: Project, value: Dict) -> Dict:
            return _fill_blocks(dict(value), rifatti)

        # Riallineare non e' un gesto dell'utente: lo stato dello step resta quello che era,
        # se no aprire la sezione lo marcherebbe come corretto a mano.
        stato_ora = project.steps.get("orientation") or {}
        stored = _write_orientation(
            project_id, allinea,
            status=str(stato_ora.get("status") or "proposed"),
            source=str(stato_ora.get("source") or "model"),
        )
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
    # Solo le immagini di questo progetto: gli artefatti del marker sono di quando e'
    # girato, e dopo uno sdoppiamento elencano ancora quelle dell'altro piano.
    mie = _sue_immagini(project)
    per_image = [r for r in by_name.values() if not mie or r.get("name") in mie]
    # Le detection rifiutate restano visibili - servono per poterle rimettere - ma marcate:
    # non fanno envelope e non contano come marker trovato.
    rifiutate = set(stored.get("refused") or {})
    for row in per_image:
        if row.get("name") in rifiutate:
            row["refused"] = True
    # Chi non entra negli envelope perche' da' torto a una correzione: si vede, e si puo'
    # sempre correggere anche quella immagine. Tacerlo vorrebbe dire far sparire venti
    # detection senza dire perche'.
    contributori = {
        riga["name"]
        for elenco in _envelope_contributors(project, stored).values() for riga in elenco
    }
    scontente = 0
    for row in per_image:
        if (row.get("box") and not row.get("refused") and not row.get("corrected")
                and row.get("name") not in contributori):
            row["disagrees"] = True
            scontente += 1

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
            "refused": sorted(rifiutate),
            "disagreeing": scontente,
            # Il ritaglio della cartella e' quello che finira' in DB_echo: e' il protagonista.
            "folder_template": {
                **{k: v for k, v in folder_template.items() if k != "path"},
                "url": (
                    f"/api/projects/{project_id}/orientation/folder_template"
                    if folder_template.get("path")
                    else None
                ),
                # Il PNG sta sempre allo stesso indirizzo: senza sapere **quando** e' stato
                # riscritto, il browser continuerebbe a mostrare quello di prima proprio
                # dopo averlo cambiato.
                "mtime": (
                    int(Path(folder_template["path"]).stat().st_mtime)
                    if folder_template.get("path") and Path(folder_template["path"]).is_file()
                    else 0
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
            exclusion_rect=_marker_vendor_exclusion(project),
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
        "exclusion": _marker_vendor_exclusion(project),
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
        exclusion_rect=ctx["exclusion"],
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
            exclusion_rect=ctx["exclusion"],
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

    # --- UN MARKER SOLO PER TUTTE LE IMMAGINI ---------------------------------
    # Alla gara dei ritagli partecipano il consegnato e quelli nati dalle correzioni: serve
    # a scoprire se uno di loro aggancia meglio, e a promuoverlo. Ma da li' in poi il match
    # deve tornare a essere uno: quello promosso, cercato su tutte le immagini. Se gli
    # envelope li facessero i ritagli mescolati, descriverebbero dove stanno glifi diversi -
    # e ESI di ritagli ne usa uno.
    consegnate: Dict[str, Dict] = {}
    if template_info.get("path") and Path(template_info["path"]).is_file():
        progress("ricerco il ritaglio consegnato su tutte le immagini", 0, len(useful))
        finale = om.match_all(
            images=useful, folder=probe_folder,
            template_path=Path(template_info["path"]), rect=ctx["rect"],
            bundle_dir=ctx["bundle"], min_score=min_score, search_margin=60,
            exclusion_rect=ctx["exclusion"],
            progress=lambda fatte, quante: progress(
                "ricerco il ritaglio consegnato su tutte le immagini", fatte, quante),
        )
        consegnate = {
            row["name"]: {"score": row["score"], "group": row["group"], "box": row["box"]}
            for row in (finale.get("rows") or [])
            if row.get("box") and (row.get("score") or 0) >= min_score
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
        if consegnate:
            value = dict(value)
            value["marker_rows_delivered"] = consegnate
        # Gli envelope si rifanno con le posizioni nuove del ritaglio consegnato, non con
        # quelle del giro precedente.
        groups_finali = (_rebuild_orientation(project, value)["groups"] if consegnate
                         else groups)
        merged = _fill_blocks(dict(value), groups_finali)
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
        # Gli envelope sono stati aggiornati: ora le proposte a valle possono leggere la
        # correzione. Il planner lascia intatti gli step confermati dall'utente.
        _schedule_auto_pipeline(project_id)
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
    job_id: str, project_id: str, name: str, box: Dict[str, int], min_score: float,
    stretta: bool = False,
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
            search_margin=60, exclusion_rect=ctx["exclusion"],
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
            exclusion_rect=ctx["exclusion"],
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
            # Le correzioni fatte sul marker **sbagliato** non valgono piu': indicavano
            # dov'era un glifo che non stiamo piu' cercando. Ma stringere e' un'altra cosa -
            # il glifo e' lo stesso, si e' solo tolto il contorno - e li' buttarle sarebbe
            # buttare lavoro suo. Si spostano invece dello stesso scarto del ritaglio: se il
            # riquadro si e' stretto di 4 px a sinistra, la posizione corretta si sposta di 4.
            vecchie = dict(value.get("corrections") or {})
            tenute: Dict[str, Dict] = {}
            if stretta and vecchie:
                prima = ((value.get("marker_override") or {}).get("box")
                         or (value.get("folder_template") or {}).get("source_box") or {})
                dx = int(box["left"]) - int(prima.get("left", box["left"]))
                dy = int(box["top"]) - int(prima.get("top", box["top"]))
                for nome_fix, fix in vecchie.items():
                    voce = dict(fix)
                    scatola = voce.get("box")
                    if isinstance(scatola, dict) and all(
                            k in scatola for k in ("top", "left", "bottom", "right")):
                        voce["box"] = {
                            "left": scatola["left"] + dx, "right": scatola["right"] + dx,
                            "top": scatola["top"] + dy, "bottom": scatola["bottom"] + dy,
                        }
                    tenute[nome_fix] = voce
            merged.update(
                {
                    "corrections": tenute,
                    # Stessa regola delle correzioni: stringere non cambia il glifo, quindi
                    # cio' che avevi buttato resta buttato; cambiarlo si', e allora quelle
                    # detection erano di un altro glifo e non vogliono dire piu' niente.
                    "refused": dict(value.get("refused") or {}) if stretta else {},
                    "hint_paths": {} if not stretta else (value.get("hint_paths") or {}),
                    "hint_templates": [] if not stretta else (value.get("hint_templates") or []),
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
                "corrections_dropped": 0 if stretta else len(stored.get("corrections") or {}),
                "corrections_kept": len(value.get("corrections") or {}) if stretta else 0,
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


def _detection_in_disaccordo(righe: Dict[str, Dict],
                             correzioni: Dict[str, Dict]) -> Dict[str, Dict]:
    """Via le detection che danno torto a una correzione, nel suo stesso gruppo.

    Correggere un marker diceva una cosa sola: «in *questa* immagine sta qui». L'envelope
    pero' e' l'unione di tutte le detection del gruppo, quindi bastava che le altre venti
    fossero sul posto sbagliato perche' la correzione non cambiasse niente - ed e' la cosa
    che si sente di piu': indichi il marker giusto e non succede niente.

    Qui una correzione vale per il suo gruppo: le detection che stanno su un'**altra riga
    dello schermo** vengono messe da parte. La riga, non il punto: il marker scorre in
    orizzontale col bordo dell'immagine ecografica (su prova 6 di trecento pixel), mentre
    l'altezza e' quella che distingue il marker vero da un falso aggancio - su prova 6 il
    ritaglio «E9» contro la riga di «LOGIQ», sedici pixel piu' su.

    Restano sempre le correzioni: sono loro il metro.
    """
    if not correzioni:
        return righe
    per_gruppo: Dict[str, List[Dict]] = {}
    for fix in correzioni.values():
        per_gruppo.setdefault(str(fix["group"]), []).append(fix["box"])
    fuori: Dict[str, Dict] = {}
    for nome, riga in righe.items():
        if riga.get("corrected"):
            fuori[nome] = riga
            continue
        modelli = per_gruppo.get(str(riga.get("group") or ""))
        box = riga.get("box")
        if not modelli or not box:
            fuori[nome] = riga
            continue
        centro = (int(box["top"]) + int(box["bottom"])) / 2.0
        vicino = False
        for atteso in modelli:
            altezza = max(6.0, float(int(atteso["bottom"]) - int(atteso["top"])))
            suo = (int(atteso["top"]) + int(atteso["bottom"])) / 2.0
            if abs(centro - suo) <= max(10.0, 1.2 * altezza):
                vicino = True
                break
        if vicino:
            fuori[nome] = riga
    return fuori


def _envelope_contributors(project: Project, value: Dict) -> Dict[str, List[Dict]]:
    """Le detection che hanno formato ogni envelope, per gruppo, col nome dell'immagine."""
    stage = _marker_stage_dir(project)
    override = value.get("marker_rows_override") or {}
    consegnate = value.get("marker_rows_delivered") or {}
    if override:
        rows = [
            {"name": name, "group": row.get("group"), "box": row.get("box"),
             "score": row.get("score"), "source": "marker indicato a mano"}
            for name, row in override.items() if row.get("box") and row.get("group")
        ]
    elif consegnate:
        # Le posizioni del ritaglio che va in DB_echo: e' quello che ESI cerchera', quindi
        # e' quello che deve disegnare gli envelope - anche dopo una correzione.
        rows = [
            {"name": name, "group": row.get("group"), "box": row.get("box"),
             "score": row.get("score"), "source": "ritaglio consegnato"}
            for name, row in consegnate.items() if row.get("box") and row.get("group")
        ]
    elif stage is not None:
        rows = [
            {"name": row["name"], "group": row["group"], "box": row["box"],
             "score": row.get("score"), "source": row.get("template") or "banca"}
            for row in _marker_rows(stage)
            if row.get("box") and str(row.get("status") or "").lower() == "ok"
        ]
        # Le stesse detection che il modulo scarta perche' cadono fuori dal rettangolo. Il
        # filtro sta anche qui, e non solo dentro alla run, perche' gli envelope si rifanno
        # da questa funzione ogni volta che arriva una correzione: senza, una cartella gia'
        # analizzata continuerebbe a portarsi dietro il rumore fino al prossimo «Rifai».
        rettangolo = _final_rect(project)
        if rettangolo:
            per_nome, _fuori = _marker_nel_rettangolo(
                {r["name"]: r for r in rows}, rettangolo)
            rows = list(per_nome.values())
    else:
        rows = []
    per_name = {row["name"]: row for row in rows}
    # Le detection che lei ha guardato e rifiutate: non sono il marker, e finche' restano
    # dentro tengono aperto un envelope - a volte un gruppo intero - che non esiste.
    for nome in (value.get("refused") or {}):
        per_name.pop(nome, None)
    correzioni = {
        nome: fix for nome, fix in (value.get("corrections") or {}).items()
        if fix.get("box") and fix.get("group")
    }
    for name, fixed in correzioni.items():
        per_name[name] = {
            "name": name, "group": fixed["group"], "box": fixed["box"],
            "score": fixed.get("score"), "source": "correzione", "corrected": True,
        }
    per_name = _detection_in_disaccordo(per_name, correzioni)
    # Solo le immagini che questo progetto ha: gli artefatti del marker sono di quando e'
    # girato, e dopo uno sdoppiamento contengono ancora quelle dell'altro piano.
    mie = _sue_immagini(project)
    out: Dict[str, List[Dict]] = {}
    for row in per_name.values():
        if mie and row["name"] not in mie:
            continue
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


def _senza_bordo(image_path: Path, box: Dict[str, int],
                 margine: int = 1) -> Optional[Dict]:
    """Il riquadro del marker portato sul glifo, buttando via il nero attorno.

    Il template matching consegna un rettangolo che contiene il glifo, non che gli aderisce:
    attorno resta un bordo di sfondo, e quel bordo cambia da un'immagine all'altra - e' la
    parte che rende il ritaglio meno riconoscibile proprio dove serve.

    Lo sfondo di queste interfacce e' scuro e il glifo chiaro, ma non sempre: si guarda
    quale dei due prevale nel ritaglio e si prende l'altro come inchiostro. Poi si tiene la
    scatola che contiene l'inchiostro, con un pixel di respiro - senza, il match si aggrappa
    ai bordi del glifo e diventa fragile.

    Torna None quando non c'e' niente da togliere o quando quel che resta e' troppo poco per
    essere un glifo: meglio lasciare il riquadro com'e' che consegnarne uno vuoto.
    """
    from PIL import Image  # noqa: PLC0415

    try:
        with Image.open(image_path) as grezza:
            ritaglio = grezza.convert("L").crop(
                (int(box["left"]), int(box["top"]), int(box["right"]), int(box["bottom"]))
            )
    except Exception:  # noqa: BLE001
        return None
    larghezza, altezza = ritaglio.size
    if larghezza < 6 or altezza < 6:
        return None
    px = ritaglio.load()
    valori = [px[x, y] for y in range(altezza) for x in range(larghezza)]
    basso, alto = min(valori), max(valori)
    if alto - basso < 25:
        return None      # tutto uguale: non c'e' nessun glifo da isolare
    soglia = (basso + alto) / 2.0
    # Chiaro su scuro o scuro su chiaro: comanda quello che occupa meno spazio, che e'
    # l'inchiostro. Uno sfondo che copre meno di meta' del ritaglio non e' uno sfondo.
    chiari = sum(1 for v in valori if v > soglia)
    inchiostro_chiaro = chiari <= len(valori) / 2
    acceso = ((lambda v: v > soglia) if inchiostro_chiaro else (lambda v: v <= soglia))

    colonne = [x for x in range(larghezza) if any(acceso(px[x, y]) for y in range(altezza))]
    righe = [y for y in range(altezza) if any(acceso(px[x, y]) for x in range(larghezza))]
    if not colonne or not righe:
        return None
    nuovo = {
        "left": int(box["left"] + max(0, colonne[0] - margine)),
        "right": int(box["left"] + min(larghezza, colonne[-1] + 1 + margine)),
        "top": int(box["top"] + max(0, righe[0] - margine)),
        "bottom": int(box["top"] + min(altezza, righe[-1] + 1 + margine)),
    }
    if nuovo["right"] - nuovo["left"] < 6 or nuovo["bottom"] - nuovo["top"] < 6:
        return None
    tolti = {
        "left": nuovo["left"] - int(box["left"]), "top": nuovo["top"] - int(box["top"]),
        "right": int(box["right"]) - nuovo["right"], "bottom": int(box["bottom"]) - nuovo["bottom"],
    }
    if not any(tolti.values()):
        return None
    return {"box": nuovo, "trimmed": tolti,
            "ink": round(sum(1 for v in valori if acceso(v)) / len(valori), 3),
            "ink_light": inchiostro_chiaro}


@app.get("/api/projects/<project_id>/orientation/marker_tight")
def api_orientation_marker_tight(project_id: str):
    """Il riquadro del marker senza il bordo di sfondo, da guardare prima di applicarlo."""
    project = _project(project_id)
    folder = project.working_dir() or Path("")
    nome = (request.args.get("name") or "").strip()
    percorso = _immagine_nella_cartella(folder, nome)
    if percorso is None:
        return jsonify({"error": f"immagine non nella cartella: {nome or '(vuoto)'}"}), 400
    chiesto = request.args.get("box") or ""
    if chiesto:
        try:
            lati = [int(round(float(v))) for v in chiesto.split("|")]
            box = {"top": lati[0], "left": lati[1], "bottom": lati[2], "right": lati[3]}
        except (TypeError, ValueError, IndexError):
            return jsonify({"error": "riquadro illeggibile: serve top|left|bottom|right"}), 400
    else:
        box = _current_marker_box(project, nome)
        if box is None:
            return jsonify({"error": "nessun marker in questa immagine"}), 404
    esito = _senza_bordo(percorso, box)
    if esito is None:
        return jsonify({"box": box, "before": box, "changed": False,
                        "reason": "il bordo scuro non c'e' o quel che resterebbe e' troppo poco"})
    return jsonify({**esito, "before": box, "changed": True})


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
    # Stringere non e' cambiare glifo: si dichiara, perche' cambia cosa succede alle
    # correzioni gia' fatte.
    stretta = bool(payload.get("narrow"))
    return jsonify(
        {"job_id": _start_job(_run_marker_override, project_id, name, box, min_score, stretta)}
    )


@app.post("/api/projects/<project_id>/orientation/reprocess")
def api_orientation_reprocess(project_id: str):
    """Rielabora ora la cartella con le correzioni che ci sono, senza toccarne altre."""
    _project(project_id)
    return jsonify(_schedule_consolidation(project_id))


@app.post("/api/projects/<project_id>/orientation/refuse")
def api_orientation_refuse(project_id: str):
    """«Questo non e' il marker»: la detection si butta, l'immagine resta senza.

    Correggere un marker vale per una immagine. Ma un glifo sbagliato non sbaglia una volta
    sola: su prova 5 il ritaglio si ritrovava su sette immagini nello stesso punto del
    pannello di destra, sempre a 0.62, e quelle sette inventavano un gruppo LR che nella
    cartella non c'e'. Correggerle una per una non le toglieva di mezzo: l'envelope e' la
    loro unione, e finche' ci sono dentro il gruppo resta.

    Qui si buttano tutte insieme. Non si cancella niente su disco: la riga del modulo resta
    dov'e', semplicemente non fa piu' envelope.
    """
    project = _project(project_id)
    payload = _payload()
    nomi = [str(n).strip() for n in (payload.get("names") or []) if str(n).strip()]
    if not nomi:
        uno = (payload.get("name") or "").strip()
        nomi = [uno] if uno else []
    if not nomi:
        return jsonify({"error": "manca l'immagine"}), 400
    annulla = bool(payload.get("reset"))

    def mutate(project: Project, value: Dict) -> Dict:
        rifiutate = dict(value.get("refused") or {})
        adesso = datetime.now().isoformat(timespec="seconds")
        for nome in nomi:
            if annulla:
                rifiutate.pop(nome, None)
            else:
                rifiutate[nome] = {"ts": adesso, "note": payload.get("note", "")}
        value["refused"] = rifiutate
        rebuilt = _rebuild_orientation(project, value)
        value = _fill_blocks(value, rebuilt["groups"])
        return value

    value = _write_orientation(project_id, mutate)
    return jsonify(
        {
            "saved": True,
            "refused": len(value.get("refused") or {}),
            "groups": {
                g: {k: b.get(k) for k in ("top", "left", "bottom", "right", "markers")}
                for g, b in (value.get("groups") or {}).items()
            },
            "missing_groups": value.get("missing_groups") or [],
        }
    )


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

    # Un riquadro esplicito serve a vedere il ritaglio **prima** di applicarlo: mentre si
    # stringe il marker, quello salvato e' ancora il vecchio.
    chiesto = request.args.get("box") or ""
    if chiesto:
        try:
            box = [int(round(float(v))) for v in chiesto.split("|")]
        except (TypeError, ValueError):
            return jsonify({"error": "riquadro illeggibile: serve top|left|bottom|right"}), 400
        if len(box) != 4:
            return jsonify({"error": "riquadro incompleto: serve top|left|bottom|right"}), 400
    else:
        trovato = _current_marker_box(project, name)
        if trovato is None:
            return jsonify({"error": "nessun marker in questa immagine"}), 404
        box = [trovato["top"], trovato["left"], trovato["bottom"], trovato["right"]]

    path = _immagine_nella_cartella(folder, name)
    if path is None:
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
        "label": "4 · corde sulle immagini a depth piu' bassa",
        # il piano non si aspetta: dopo lo sdoppiamento il progetto e' gia' di un piano solo,
        # e prima lo si riconosce immagine per immagine dentro al rettangolo provvisorio
        "needs": ("orientation", "depth"),
        "what": "a depth bassa la macchina ingrandisce, e la corda piu' larga del ventaglio "
                "arriva davvero al bordo: per ogni orientamento si rimisura sulle immagini "
                "con la depth confermata piu' bassa, e i bordi si posano su una misura",
    },
)


def _sue_immagini(project: Project) -> set:
    """I nomi che questo progetto ha davvero.

    Gli artefatti dei moduli sono di **quando sono girati**: dopo uno sdoppiamento contengono
    ancora le immagini dell'altro piano, e ogni sezione che li legge tal quali le rimette in
    scena - immagini che di questo progetto non fanno piu' parte. Il confronto con l'elenco
    di adesso e' cio' che tiene le sezioni oneste senza aspettare che i moduli rigirino.
    """
    return set(project.dedup_names())


def _groups_of_images(project: Project) -> Dict[str, str]:
    """Il gruppo di orientamento per immagine, con la solita priorita' delle sorgenti."""
    value = project.step_value("orientation")
    mie = _sue_immagini(project)
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
    return {n: g for n, g in out.items() if not mie or n in mie}


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


def _run_rect_depth(job_id: str, project_id: str, per_group: int) -> None:
    """Giro 4: le corde misurate sulle immagini con la depth **piu' bassa**.

    A depth bassa la macchina ingrandisce: lo stesso ventaglio occupa piu' pixel, e la sua
    corda piu' larga arriva davvero al bordo del rettangolo. Il giro del segmento le
    immagini le prendeva a passo fisso nella cartella, senza sapere quali fossero le piu'
    informative - e su una cartella dove le depth vanno da 18 a 120 mm significa misurare
    la corda su un'immagine rimpicciolita e chiudere il rettangolo troppo stretto.

    Adesso lo dice la depth confermata: per ogni orientamento si prendono le immagini con
    il valore piu' basso. La costruzione resta quella di sempre - l'asse verticale a meta'
    fra la corda piu' alta di NF/LR e la piu' bassa di UD/LRUD, le due estreme sui bordi -
    ma le corde sono misurate dove si vedono meglio.
    """
    try:
        project = _project(project_id)
        confermate = _depth_confermate(project)
        if not confermate:
            raise ValueError(
                "serve la depth confermata: vai nella sezione depth, controlla i valori e "
                "premi «Conferma la depth»"
            )
        gruppi = _groups_of_images(project)
        if not gruppi:
            raise ValueError("serve prima l'orientamento")
        folder = _require_folder(project)
        imported = project.step_value("import")
        width, height = (imported.get("image_sample_size") or [0, 0])[:2]
        corrente = project.step_value("rect").get("rect_echo")
        if not width or not height or not corrente:
            raise ValueError("servono il rettangolo di adesso e la dimensione dell'immagine")
        rect_norm = {
            "x": corrente["left"] / float(width),
            "y": corrente["top"] / float(height),
            "w": (corrente["right"] - corrente["left"] + 1) / float(width),
            "h": (corrente["bottom"] - corrente["top"] + 1) / float(height),
        }
        base = project.dedup_link_dir() or folder
        stimatore = _segment_estimator()
        engine = _inference_engine()

        # Per gruppo: le immagini con la depth piu' bassa fra quelle confermate.
        scelte: Dict[str, Dict] = {}
        for nome, gruppo in gruppi.items():
            if gruppo not in marker_refine.GROUP_ORDER or nome not in confermate:
                continue
            scelte.setdefault(gruppo, {}).setdefault(confermate[nome], []).append(nome)
        if not scelte:
            raise ValueError(
                "nessuna immagine ha insieme un orientamento e una depth confermata"
            )

        misure: Dict[str, Dict] = {}
        for indice, (gruppo, per_valore) in enumerate(sorted(scelte.items())):
            minima = min(per_valore)
            nomi = sorted(per_valore[minima])[:max(1, per_group)]
            _job_update(job_id, stage=f"corde nel gruppo {gruppo} a {minima:g} mm",
                        done=indice, total=len(scelte))
            etichetta = "su" if gruppo in ("NF", "LR") else "giu"
            trovati = []
            for nome in nomi:
                percorso = base / nome
                if not percorso.exists():
                    continue
                seg = stimatore(image_path=percorso, rect_norm=rect_norm,
                                orientation_label=etichetta)
                if not seg:
                    continue
                trovati.append({
                    "image": nome,
                    "plane": engine.predict_lt([percorso], corrente).get("plane"),
                    "x1": float(seg["x1"]), "x2": float(seg["x2"]), "y": float(seg["y"]),
                    "length_norm": float(seg["length_norm"]),
                    "length_px": round(float(seg["length_norm"]) * width, 1),
                    "depth_mm": minima,
                })
            if not trovati:
                continue
            piani: Dict[str, int] = {}
            for s in trovati:
                if s["plane"]:
                    piani[s["plane"]] = piani.get(s["plane"], 0) + 1
            dominante = max(piani, key=piani.get) if piani else None
            nel_piano = [s for s in trovati if not dominante or s["plane"] == dominante]
            misure[gruppo] = {
                "segment": max(nel_piano, key=lambda s: s["length_norm"]),
                "plane": dominante,
                "planes": piani,
                "depth_mm": minima,
                "depths_available": sorted(per_valore),
                "tried": len(nomi),
                "found": len(trovati),
                "median_length_px": round(
                    statistics.median([s["length_px"] for s in nel_piano]), 1),
                "all": nel_piano,
            }

        if not misure:
            raise ValueError("nessuna corda misurata sulle immagini a depth piu' bassa")

        risultato = _segments_analysis(misure, width, height, corrente, rect_norm)
        risultato["chosen_by"] = "depth"
        risultato["depth_by_group"] = {g: m["depth_mm"] for g, m in misure.items()}
        _job_update(job_id, status="done", stage="fatto", result=risultato)

        def mutate(project: Project, value: Dict) -> Dict:
            catena = dict(value.get("chain") or {})
            passi = dict(catena.get("passes") or {})
            passi["depth"] = {
                "state": "proposed",
                "proposal": risultato["proposal"],
                "pairs": {k: v["max_px"] for k, v in risultato["pairs"].items()},
                "per_group": {g: m["segment"]["length_px"] for g, m in misure.items()},
                "depth_by_group": risultato["depth_by_group"],
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            catena["passes"] = passi
            value["chain"] = catena
            # Lo studio delle corde diventa questo: e' lo stesso studio, misurato dove si
            # vede meglio. La sezione del rettangolo mostra sempre l'ultimo.
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
        # Non «c'e' una depth», ma «la depth e' confermata»: e' quella che sceglie le
        # immagini su cui rimisurare le corde.
        "depth": bool(_depth_confermate(project)),
    }
    stato_passi = (rect_value.get("chain") or {}).get("passes") or {}
    passi = []
    for spec in RECT_PASSES:
        salvato = stato_passi.get(spec["id"]) or {}
        mancano = [n for n in spec["needs"] if not disponibile.get(n)]
        if spec["id"] == "rete":
            stato = "done" if rect_value.get("rect_echo") else "available"
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
    lavori = {"specularita": _run_rect_specularity, "segmento": _run_rect_segments,
              "depth": _run_rect_depth}
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
    return jsonify({
        "rect": value.get("rect_echo"),
        "history": (value.get("chain") or {}).get("history"),
        "advanced_job_id": _schedule_auto_pipeline(project_id),
    })


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


# --- studio della scala: le correzioni dell'operatore ----------------------
# Il modulo le rilegge a ogni run (`--corrections`) e le applica *dopo* la detection, cosi'
# la pagina mostra sempre anche cosa avrebbe detto da solo. Una colonna corretta su un
# fotogramma fa da ancora per tutta la cartella. E' in questo senso che impara: la verita'
# non si perde, rientra nel calcolo ogni volta.
SCALE_CORRECTION_FIELDS = ("x", "y_zero", "y_far", "zero_end", "ticks_add", "ticks_del",
                           "ticks", "pitch", "nums", "depth_mm")


# I passi che una macchina ecografica disegna davvero fra una tacca e l'altra. Non e' un
# elenco arbitrario: e' quello che si legge sui righelli, e serve da rete quando il passo
# ricavato dai pixel cade vicino a uno di questi ma non esatto.
PASSI_PLAUSIBILI_MM = (1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0)


def _griglia_tacche(y_zero: float, y_far: float, passo_px: float,
                    aggancia_al_fondo: bool = False) -> List[float]:
    """Le tacche da uno zero, a passo costante, fino al fondo.

    Dentro a un fotogramma il passo **non cambia**: e' la sola cosa che il righello promette
    sempre. Quindi non si cercano le tacche una per una - se ne conosce una e le altre
    seguono. L'ultima si include se ci sta dentro mezzo passo dal fondo: e' il fondo scala,
    e sarebbe strano perderlo per un pixel.

    `aggancia_al_fondo` quando il fondo e' un dato e non una stima - tipicamente perche'
    viene dalla depth confermata. Allora il passo si aggiusta di quel poco che serve a far
    cadere l'ultima tacca **esattamente** sul fondo: 575 px in 10 passi fanno 57.5, non 58
    con l'ultima tacca cinque pixel oltre la fine della barra.
    """
    if not passo_px or passo_px <= 0:
        return []
    verso = 1.0 if y_far >= y_zero else -1.0
    lunghezza = abs(y_far - y_zero)
    quante = int(round(lunghezza / passo_px))
    if quante < 1 or quante > 200:
        return []
    if aggancia_al_fondo and quante:
        passo_px = lunghezza / quante
    tacche = [round(y_zero + verso * k * passo_px, 1) for k in range(quante + 1)]
    # Se l'ultima sfora il fondo di piu' di mezzo passo, non e' una tacca: e' un troppo.
    if abs(tacche[-1] - y_zero) - lunghezza > 0.5 * passo_px:
        tacche.pop()
    return tacche


def _passo_da_tacca(y_zero: float, tacca: float, passo_atteso: Optional[float] = None) -> float:
    """Il passo dedotto da **una** tacca indicata a mano.

    Di solito e' la prima dopo lo zero, e allora il passo e' la distanza. Ma se lei ne
    segna una piu' lontana - capita, e' quella che si vede meglio - la distanza e' un
    multiplo: si divide per il numero di passi piu' vicino, usando il passo atteso quando
    c'e' (quello della cartella, o quello che il rilevatore aveva trovato).
    """
    distanza = abs(float(tacca) - float(y_zero))
    if distanza <= 0:
        return 0.0
    if passo_atteso and passo_atteso > 0:
        quanti = max(1, int(round(distanza / passo_atteso)))
        return distanza / quanti
    return distanza


def _passo_vicino(passo_mm: float) -> Tuple[float, float]:
    """Il passo plausibile piu' vicino, e di quanto ci si discosta (in parti di uno)."""
    migliore = min(PASSI_PLAUSIBILI_MM, key=lambda v: abs(v - passo_mm))
    return migliore, abs(migliore - passo_mm) / migliore


def _righello_con_depth(y_zero: float, y_far: Optional[float], passo_px: Optional[float],
                        depth_mm: Optional[float],
                        mm_per_px_etichette: Optional[float] = None,
                        etichetta_max_mm: Optional[float] = None) -> Dict:
    """Il righello riletto alla luce della depth confermata, dicendo **cosa** e' sbagliato.

    Ci sono due misure indipendenti della stessa cosa. Le tacche danno un passo in pixel; se
    l'OCR ha letto dei numeri, danno anche i millimetri per pixel. La barra da' una
    lunghezza, e la depth confermata dice quanti millimetri deve valere. Se le due non
    tornano, una delle due e' sbagliata - e sapere quale e' tutto il punto.

    Sono le etichette a decidere, quando ci sono: sono una misura, non un'ipotesi. Su
    `prova_4` dicono 4, 6, 8 cm a 115 px di distanza l'una dall'altra, cioe' 0.174 mm/px;
    con quella scala la barra fino a 750 px varrebbe 113 mm, mentre la depth dice 100. Non
    e' il passo a essere sbagliato: e' il fondo, che il rilevatore ha tirato troppo in giu'.
    Senza etichette si fa il contrario, perche' resta solo la barra.
    """
    esito: Dict = {}
    if y_far is not None:
        esito["span_px"] = round(abs(float(y_far) - float(y_zero)), 1)
    if passo_px:
        esito["pitch_px"] = round(float(passo_px), 2)
    if not depth_mm or float(depth_mm) <= 0:
        esito["verdict"] = "senza depth confermata non c'e' niente con cui confrontare"
        return esito
    depth_mm = float(depth_mm)
    esito["depth_mm"] = depth_mm

    if mm_per_px_etichette and mm_per_px_etichette > 0:
        mm_per_px = float(mm_per_px_etichette)
        esito["source"] = "etichette"
    elif esito.get("span_px"):
        mm_per_px = depth_mm / esito["span_px"]
        esito["source"] = "barra"
    else:
        esito["verdict"] = "manca la barra"
        return esito
    esito["mm_per_px"] = round(mm_per_px, 5)

    # Il fondo che la depth impone: e' un conto, non una stima.
    verso = 1.0
    if y_far is not None and float(y_far) < float(y_zero):
        verso = -1.0
    span_atteso = depth_mm / mm_per_px
    esito["span_expected_px"] = round(span_atteso, 1)
    if esito.get("span_px"):
        scarto_barra = abs(esito["span_px"] - span_atteso) / span_atteso
        esito["span_off"] = round(scarto_barra, 3)
        if esito["source"] == "etichette" and scarto_barra > 0.03:
            esito["y_far_suggested"] = round(float(y_zero) + verso * span_atteso, 1)

    if passo_px and passo_px > 0:
        passo_mm = float(passo_px) * mm_per_px
        vicino, scarto = _passo_vicino(passo_mm)
        esito.update({"step_mm_raw": round(passo_mm, 3), "step_mm": vicino,
                      "step_off": round(scarto, 3)})
        if scarto > 0.02:
            esito["pitch_suggested"] = round(vicino / mm_per_px, 2)
        esito["ticks_expected"] = round(depth_mm / vicino, 2)

    scarti = [esito.get("span_off") or 0.0, esito.get("step_off") or 0.0]
    peggiore = max(scarti)
    if peggiore <= 0.02:
        esito["verdict"] = "torna"
    elif peggiore <= 0.25:
        esito["verdict"] = "da correggere"
    else:
        esito["verdict"] = "non torna"
    # Il fattore dieci non si vede dalla barra - quella puo' essere tirata lunga - ma dai
    # numeri stampati sul righello: una depth piu' piccola del numero piu' grande scritto
    # sulla scala e' impossibile, e dieci volte tanto la rimette al suo posto.
    if etichetta_max_mm and etichetta_max_mm > depth_mm * 1.05:
        # Il fattore dieci si racconta solo se regge: il numero piu' grande stampato sta
        # vicino al fondo scala, quindi con una depth dieci volte tanto deve caderci dentro
        # e non troppo sotto. Un `90` con depth 40 non diventa credibile a 400 - li' e'
        # l'etichetta a essere sbagliata, ed e' un'altra storia.
        if depth_mm * 4.0 <= etichetta_max_mm <= depth_mm * 10.0 * 1.05:
            esito["unit_hint"] = (
                f"sul righello c'e' scritto {etichetta_max_mm:g} mm, piu' della depth "
                f"confermata ({depth_mm:g} mm): la depth sembra in centimetri, "
                f"dovrebbe essere {depth_mm * 10:g} mm")
        else:
            esito["unit_hint"] = (
                f"sul righello c'e' scritto {etichetta_max_mm:g} mm, piu' della depth "
                f"confermata ({depth_mm:g} mm): una delle due e' sbagliata")
        if esito.get("verdict") == "torna":
            esito["verdict"] = "da correggere"
    return esito


def _scale_corrections_path(project: Project) -> Path:
    return project.root / "scale_study_corrections.json"


def _depth_confermate(project: Project) -> Dict[str, float]:
    """Le depth che l'utente ha confermato, e solo quelle.

    Il modulo della scala distingue una depth *letta* da una depth *dell'operatore*, e la
    seconda la tratta come l'evidenza piu' forte che ha. La distinzione conta davvero: su
    `prova_2` la depth letta dice 20 mm su fotogrammi che il righello misura a 40 e 80, e
    propagarla farebbe danno. Quella confermata invece e' verita' e va usata.

    Vale come confermata: un valore riscritto a mano; le letture nate da un riquadro che
    l'utente ha applicato lui (`scope` diverso da `auto`); oppure tutte, se lo step della
    depth risulta confermato o corretto dall'utente.
    """
    valore = project.step_value("depth_scale")
    stato = (project.steps.get("depth_scale") or {})
    confermate: Dict[str, float] = {}

    letture = valore.get("depth_box_reads") or {}
    modello = valore.get("depth_box_template") or {}
    tutto_confermato = (
        bool(stato.get("user_edited"))
        and str(stato.get("status") or "") in ("confirmed", "corrected")
    )
    if letture and (tutto_confermato or str(modello.get("scope") or "") not in ("", "auto")):
        for nome, lettura in letture.items():
            misura = lettura.get("depth_mm")
            if misura:
                confermate[nome] = float(misura)

    for nome, fix in (valore.get("depth_corrections") or {}).items():
        misura = fix.get("depth_mm")
        if misura:
            confermate[nome] = float(misura)
    return confermate


def _write_scale_corrections(project: Project) -> Path:
    """Scrive le correzioni nel formato del modulo: {cartella: {note, frames: {nome: {...}}}}.

    Le depth confermate entrano qui come `depth_mm` per fotogramma: e' il canale che il modulo
    ha per la depth dell'operatore, quindi non serve inventarne un altro.
    """
    valore = project.step_value("scale_study")
    base = project.working_dir() or Path(project.source.get("folder") or "")
    chiave = base.name or "cartella"
    frames: Dict[str, Dict] = {
        nome: dict(voce) for nome, voce in (valore.get("corrections") or {}).items()
    }
    for nome, misura in _depth_confermate(project).items():
        voce = frames.setdefault(nome, {})
        voce.setdefault("depth_mm", misura)
    payload = {chiave: {"note": str(valore.get("note") or ""), "frames": frames}}
    percorso = _scale_corrections_path(project)
    percorso.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return percorso


def _righello_noto(frame: Dict, correzioni: Dict) -> bool:
    """Un fotogramma su cui il righello si sa: accettato dal modulo, o corretto a mano."""
    if not frame.get("x") or frame.get("y_zero") is None:
        return False
    return bool(frame.get("status") == "accepted" or correzioni.get(frame.get("name")))


def _suggerisci_righelli(frames: Sequence[Dict], correzioni: Dict,
                         depth_confermate: Optional[Dict[str, float]] = None) -> Dict[str, Dict]:
    """Dove il righello non si trova, proporlo a partire da dove si sa.

    Si usano solo le grandezze che i fotogrammi noti misurano bene, e la depth non e' fra
    queste: su questa cartella `E_depth_interface` dice 20 mm su fotogrammi che il nome
    chiama 40 e 80, mentre il righello misurato dal modulo li da' a 39.9 e 79.8 mm — cioe'
    la lettura del righello e' giusta e quella della depth no. Un modello che si appoggiasse
    alla depth propagherebbe l'errore.

    Le grandezze buone sono geometriche e di cartella: la macchina disegna il righello sempre
    nello stesso posto e sempre lungo lo stesso, e le tacche hanno sempre lo stesso passo.
      * colonna e zero: mediana dei noti;
      * lunghezza della barra in pixel: mediana dei noti (qui 511-550 px, molto stabile);
      * passo in pixel: mediana dei noti, da cui `mm_per_px = passo_mm / passo_px`.
    Lo scarto fra i noti viene riportato: e' la misura di quanto fidarsi della proposta.
    """
    depth_confermate = depth_confermate or {}
    noti = [f for f in frames if _righello_noto(f, correzioni)]
    if not noti:
        return {}

    def mediana(valori: Sequence[float]) -> Optional[float]:
        puliti = [float(v) for v in valori if v not in (None, "")]
        return statistics.median(puliti) if puliti else None

    x_cartella = mediana([f["x"] for f in noti])
    colonne = [float(f["x"]) for f in noti]
    scarto_x = max(colonne) - min(colonne)

    versi = [str(f.get("B_zero_end") or "") for f in noti if f.get("B_zero_end")]
    verso = max(set(versi), key=versi.count) if versi else "bottom"
    stessi = [f for f in noti if str(f.get("B_zero_end") or "") == verso] or noti
    y_zero_cartella = mediana([f["y_zero"] for f in stessi])

    lunghezze = [abs(float(f["y_far"]) - float(f["y_zero"]))
                 for f in stessi if f.get("y_far") is not None]
    lunghezza = mediana(lunghezze)
    scarto_lunghezza = (max(lunghezze) - min(lunghezze)) if lunghezze else 0.0

    passo_px = mediana([f["pitch"] for f in noti if f.get("pitch")])
    passo_mm = mediana([f["D_step_mm"] for f in noti if f.get("D_step_mm")]) or 10.0

    if x_cartella is None or y_zero_cartella is None or not lunghezza:
        return {}

    direzione = 1 if verso == "top" else -1
    motivo = (f"colonna e zero dalla mediana di {len(noti)} fotogrammi noti; "
              f"barra lunga {lunghezza:.0f} px (scarto {scarto_lunghezza:.0f} px)")
    if scarto_x > 12.0:
        motivo += "; colonne note discordi, la colonna e' meno affidabile"
    if passo_px:
        motivo += f"; passo {passo_px:.0f} px = {passo_mm:g} mm"

    proposte: Dict[str, Dict] = {}
    for frame in frames:
        if _righello_noto(frame, correzioni) or frame.get("x"):
            continue
        lunghezza_frame = lunghezza
        motivo_frame = motivo
        confermata = depth_confermate.get(str(frame.get("name")))
        if confermata and passo_px:
            # Una depth confermata dall'utente e' l'unico dato esterno affidabile: dice quanti
            # millimetri deve coprire la barra, e con il passo si sa a quanti pixel corrispondono.
            lunghezza_frame = confermata / (passo_mm / passo_px)
            motivo_frame = (f"lunghezza dalla depth che hai confermato ({confermata:g} mm) "
                            f"col passo di {passo_mm:g} mm ogni {passo_px:.0f} px; "
                            f"colonna e zero dalla mediana di {len(noti)} noti")
            if scarto_x > 12.0:
                motivo_frame += "; colonne note discordi"
        y_far = y_zero_cartella + direzione * lunghezza_frame
        altezza = float(frame.get("h") or 0)
        if altezza and not (0 <= y_far <= altezza):
            continue
        tacche: List[float] = []
        if passo_px:
            quante = int(lunghezza_frame / passo_px)
            tacche = [round(y_zero_cartella + direzione * k * passo_px, 1)
                      for k in range(quante + 1)]
        proposte[str(frame.get("name"))] = {
            "x": round(x_cartella, 1),
            "y_zero": round(y_zero_cartella, 1),
            "y_far": round(y_far, 1),
            "zero_end": verso,
            "ticks": tacche,
            "mm_per_px": round(passo_mm / passo_px, 5) if passo_px else None,
            "step_mm": passo_mm,
            "from": len(noti),
            "spread_px": round(scarto_lunghezza, 1),
            "depth_confirmed_mm": confermata,
            "reason": motivo_frame,
        }
    return proposte


@app.get("/api/projects/<project_id>/scale/study/suggestions")
def api_scale_study_suggestions(project_id: str):
    """I righelli proposti per i fotogrammi in cui non e' stato trovato."""
    project = _project(project_id)
    stadi = ((project.data.get("analysis") or {}).get("stages") or {})
    percorso = str((stadi.get("scale_study") or {}).get("data_json") or "")
    if not percorso or not Path(percorso).is_file():
        return jsonify({"error": "lo studio della scala non e' ancora stato fatto",
                        "suggestions": {}}), 404
    dati = json.loads(Path(percorso).read_text(encoding="utf-8"))
    base = project.working_dir() or Path("")
    for frame in dati.get("frames") or []:
        percorso_frame = Path(str(frame.get("path") or ""))
        try:
            frame["name"] = str(percorso_frame.relative_to(base))
        except ValueError:
            frame["name"] = percorso_frame.name
    correzioni = project.step_value("scale_study").get("corrections") or {}
    confermate = _depth_confermate(project)
    proposte = _suggerisci_righelli(dati.get("frames") or [], correzioni, confermate)
    return jsonify({"suggestions": proposte,
                    "known": sum(1 for f in dati.get("frames") or []
                                 if _righello_noto(f, correzioni)),
                    "depth_confirmed": len(confermate)})


@app.post("/api/projects/<project_id>/scale/study/correct")
def api_scale_study_correct(project_id: str):
    """Correggi colonna, zero, fondo, tacche o numeri su un fotogramma."""
    project = _project(project_id)
    payload = _payload()
    nome = str(payload.get("name") or "").strip()
    if not nome:
        return jsonify({"error": "manca l'immagine"}), 400

    def mutate(_p: Project, value: Dict) -> Dict:
        correzioni = dict(value.get("corrections") or {})
        if payload.get("reset"):
            correzioni.pop(nome, None)
        else:
            voce = dict(correzioni.get(nome) or {})
            for campo in SCALE_CORRECTION_FIELDS:
                if campo in payload:
                    if payload[campo] in (None, ""):
                        voce.pop(campo, None)
                    else:
                        voce[campo] = payload[campo]
            voce["ts"] = datetime.now().isoformat(timespec="seconds")
            if not any(k in voce for k in SCALE_CORRECTION_FIELDS):
                correzioni.pop(nome, None)
            else:
                correzioni[nome] = voce
        value["corrections"] = correzioni
        if "note" in payload:
            value["note"] = str(payload.get("note") or "")
        return value

    valore = _write_step(project_id, "scale_study", mutate, status="corrected", source="user")
    _write_scale_corrections(_project(project_id))
    return jsonify({"saved": True, "corrections": len(valore.get("corrections") or {})})


def _dati_studio(project: Project) -> Optional[Dict]:
    """I dati dell'ultimo studio, coi nomi relativi come li usa il resto dell'app."""
    stadi = ((project.data.get("analysis") or {}).get("stages") or {})
    percorso = str((stadi.get("scale_study") or {}).get("data_json") or "")
    if not percorso or not Path(percorso).is_file():
        return None
    try:
        dati = json.loads(Path(percorso).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    base = project.working_dir() or Path("")
    for frame in dati.get("frames") or []:
        percorso_frame = Path(str(frame.get("path") or ""))
        try:
            frame["name"] = str(percorso_frame.relative_to(base))
        except ValueError:
            frame["name"] = percorso_frame.name
    # Solo i fotogrammi di questo progetto: lo studio puo' essere di prima dello sdoppiamento.
    mie = _sue_immagini(project)
    if mie:
        dati["frames"] = [f for f in (dati.get("frames") or []) if f.get("name") in mie]
    return dati


def _righello_effettivo(frame: Dict, correzione: Optional[Dict]) -> Dict:
    """Il righello com'e' adesso: quello che ha trovato il modulo, con sopra le correzioni."""
    c = correzione or {}
    def numero(chiave: str, dal_frame: str) -> Optional[float]:
        if c.get(chiave) not in (None, ""):
            return float(c[chiave])
        valore = frame.get(dal_frame)
        return float(valore) if valore not in (None, "") else None

    tacche = [float(t) for t in (c.get("ticks") or frame.get("ticks") or [])]
    if not c.get("ticks") and (c.get("ticks_add") or c.get("ticks_del")):
        via = {round(float(v)) for v in (c.get("ticks_del") or [])}
        tacche = sorted([t for t in tacche if round(t) not in via]
                        + [float(v) for v in (c.get("ticks_add") or [])])
    passo = None
    if len(tacche) > 1:
        salti = sorted(tacche[k + 1] - tacche[k] for k in range(len(tacche) - 1))
        passo = salti[len(salti) // 2]
    return {
        "x": numero("x", "x"),
        "y_zero": numero("y_zero", "y_zero"),
        "y_far": numero("y_far", "y_far"),
        "zero_end": str(c.get("zero_end") or frame.get("B_zero_end") or "bottom"),
        "ticks": tacche,
        "pitch": passo if passo else (float(frame["pitch"]) if frame.get("pitch") else None),
        # La scala del rilevatore vale per **la sua** scala di tacche. Se quella e' stata
        # sostituita, quel numero non descrive piu' niente: si torna a misurare la barra
        # contro la depth, che e' l'altra strada e resta valida.
        "mm_per_px": (None if c.get("ticks")
                      else (float(frame["mm_per_px"]) if frame.get("mm_per_px") else None)),
        "step_mm": float(frame["D_step_mm"]) if frame.get("D_step_mm") else None,
        "max_label_mm": ((frame.get("DE_check") or {}).get("max_label_mm")),
    }


def _etichette_sulle_tacche(frame: Dict, tacche: Sequence[float],
                            tolleranza: float = 6.0) -> Dict:
    """Quante delle scritte lette dall'OCR cadono su una tacca.

    E' la verifica indipendente: le tacche le abbiamo generate da uno zero e un passo, i
    numeri li ha letti l'OCR guardando i pixel. Se cadono uno sull'altro, due strade diverse
    sono arrivate allo stesso posto.

    Non e' un voto sul righello, e' un voto su **entrambi**: su prova_4 tre fotogrammi con
    depth diverse riportano "9 cm" alla stessa identica altezza, il che e' impossibile e
    dice che a sbagliare e' la lettura, non la tacca.
    """
    etichette = [(float(y), float(v)) for y, v in (frame.get("labels") or [])]
    if not etichette or not tacche:
        return {"ok": 0, "total": len(etichette)}
    fuori: List[Dict] = []
    dentro = 0
    for y, valore in etichette:
        vicina = min(tacche, key=lambda t: abs(t - y))
        if abs(vicina - y) <= tolleranza:
            dentro += 1
        else:
            fuori.append({"y": round(y, 1), "cm": valore, "nearest": round(vicina, 1),
                          "off": round(vicina - y, 1)})
    return {"ok": dentro, "total": len(etichette), "off": fuori[:4]}


def _controlla_righello(frame: Dict, righello: Dict, depth_mm: Optional[float]) -> Dict:
    """Il controllo del righello con la depth confermata, coi dati gia' pescati dal frame."""
    return _righello_con_depth(
        righello.get("y_zero") or 0.0, righello.get("y_far"), righello.get("pitch"),
        depth_mm, righello.get("mm_per_px"), righello.get("max_label_mm"),
    )


@app.post("/api/projects/<project_id>/scale/study/ticks")
def api_scale_study_ticks(project_id: str):
    """Da **una** tacca, tutte le altre: dentro a un fotogramma il passo non cambia.

    Si puo' dare il passo in tre modi, ed e' sempre lo stesso conto: una tacca (la distanza
    dallo zero), il passo in pixel, o il passo in millimetri insieme alla depth confermata.
    L'ultimo e' il piu' forte, perche' non dipende da cosa il rilevatore ha visto.
    """
    project = _project(project_id)
    payload = _payload()
    nome = str(payload.get("name") or "").strip()
    dati = _dati_studio(project)
    if dati is None:
        return jsonify({"error": "lo studio della scala non e' ancora stato fatto"}), 404
    frame = next((f for f in (dati.get("frames") or []) if f.get("name") == nome), None)
    if frame is None:
        return jsonify({"error": f"fotogramma non nello studio: {nome or '(vuoto)'}"}), 400

    correzioni = project.step_value("scale_study").get("corrections") or {}
    righello = _righello_effettivo(frame, correzioni.get(nome))
    for chiave in ("x", "y_zero", "y_far", "zero_end"):
        if payload.get(chiave) not in (None, ""):
            righello[chiave] = (str(payload[chiave]) if chiave == "zero_end"
                                else float(payload[chiave]))
    if righello.get("y_zero") is None:
        return jsonify({"error": "serve almeno lo zero del righello"}), 400

    depth = _depth_confermate(project).get(nome)
    controllo = _controlla_righello(frame, righello, depth)

    # Il passo, per ordine di forza.
    passo = None
    da_dove = ""
    if payload.get("tick") not in (None, ""):
        passo = _passo_da_tacca(righello["y_zero"], float(payload["tick"]), righello.get("pitch"))
        da_dove = "dalla tacca che hai segnato"
    elif payload.get("pitch") not in (None, ""):
        passo = float(payload["pitch"])
        da_dove = "dal passo che hai scritto"
    elif payload.get("step_mm") not in (None, "") and controllo.get("mm_per_px"):
        passo = float(payload["step_mm"]) / controllo["mm_per_px"]
        da_dove = f"da {float(payload['step_mm']):g} mm di passo e la scala del righello"
    elif controllo.get("pitch_suggested"):
        passo = controllo["pitch_suggested"]
        da_dove = "dal passo che fa tornare i conti con la depth confermata"
    elif righello.get("pitch"):
        passo = righello["pitch"]
        da_dove = "dal passo che aveva trovato il modulo"
    if not passo or passo <= 2:
        return jsonify({"error": "non c'e' un passo con cui generare le tacche"}), 400

    # Il fondo: se le etichette e la depth dicono che la barra e' tirata troppo, si accorcia.
    # Ma non quando il controllo ha appena detto che la depth stessa non torna: una proposta
    # calcolata su un numero sbagliato e' un errore travestito da conto. Su prova_4 la depth
    # confermata dice 10 mm dove il righello ne misura 113, e il "fondo suggerito" avrebbe
    # accorciato la barra a due tacche.
    if payload.get("use_depth") and controllo.get("unit_hint"):
        return jsonify({
            "error": "la depth confermata non torna col righello, quindi non si puo' usarla "
                     "per il fondo: " + controllo["unit_hint"],
            "check": controllo,
        }), 400
    if payload.get("use_depth") and controllo.get("y_far_suggested") is not None:
        righello["y_far"] = controllo["y_far_suggested"]
        da_dove += "; fondo dalla depth confermata"
    if righello.get("y_far") is None:
        return jsonify({"error": "serve il fondo del righello"}), 400

    # Il fondo e' un dato quando viene dalla depth confermata, o quando il controllo dice
    # che la barra e' gia' lunga giusta: allora la scala ci finisce sopra esatta.
    aggancia = bool(payload.get("use_depth")) or (controllo.get("span_off") or 1.0) <= 0.02
    tacche = _griglia_tacche(righello["y_zero"], righello["y_far"], passo, aggancia)
    if not tacche:
        return jsonify({"error": "con questo passo non viene fuori nessuna tacca"}), 400
    if aggancia and len(tacche) > 1:
        passo = abs(tacche[1] - tacche[0])
    righello["ticks"] = tacche
    righello["pitch"] = passo
    dopo = _controlla_righello(frame, righello, depth)

    dopo["labels_on_ticks"] = _etichette_sulle_tacche(frame, tacche)
    if payload.get("preview"):
        return jsonify({"name": nome, "ticks": tacche, "pitch": round(passo, 2),
                        "y_far": righello["y_far"], "from": da_dove,
                        "check": dopo, "check_before": controllo, "saved": False})

    def mutate(_p: Project, value: Dict) -> Dict:
        tutte = dict(value.get("corrections") or {})
        voce = dict(tutte.get(nome) or {})
        voce.update({"x": righello.get("x"), "y_zero": righello["y_zero"],
                     "y_far": righello["y_far"], "zero_end": righello["zero_end"],
                     "ticks": tacche, "pitch": round(passo, 2),
                     "ts": datetime.now().isoformat(timespec="seconds")})
        voce.pop("ticks_add", None)
        voce.pop("ticks_del", None)
        tutte[nome] = voce
        value["corrections"] = tutte
        return value

    _write_step(project_id, "scale_study", mutate, status="corrected", source="user")
    _write_scale_corrections(_project(project_id))
    return jsonify({"name": nome, "ticks": tacche, "pitch": round(passo, 2),
                    "y_far": righello["y_far"], "from": da_dove,
                    "check": dopo, "check_before": controllo, "saved": True})


@app.post("/api/projects/<project_id>/scale/study/approve")
def api_scale_study_approve(project_id: str):
    """«Va bene cosi'»: il dubbio del modulo si chiude, e l'etichetta «da rivedere» va via.

    Non tutti i "da rivedere" sono errori: spesso il modulo ha trovato il righello giusto e
    si e' solo tenuto basso col punteggio. Segnare a mano che va bene non e' una correzione
    — non c'e' niente da correggere — ma nemmeno una cosa da lasciare li': un dubbio gia'
    sciolto, se resta scritto, nasconde quelli veri.

    Si approva un fotogramma, un elenco, oppure tutti quelli che stanno in un certo stato.
    """
    project = _project(project_id)
    payload = _payload()
    dati = _dati_studio(project)
    if dati is None:
        return jsonify({"error": "lo studio della scala non e' ancora stato fatto"}), 404
    valore_ora = project.step_value("scale_study")
    correzioni = valore_ora.get("corrections") or {}
    # Con dentro le approvazioni gia' date: cosi' `statuses: ["approved"]` serve a
    # ripensarci in blocco, che e' l'altra meta' del gesto.
    approvati_ora = set(valore_ora.get("approved") or [])
    per_stato: Dict[str, List[str]] = {}
    for frame in dati.get("frames") or []:
        nome_frame = str(frame.get("name"))
        stato = _stato_effettivo(frame, correzioni.get(nome_frame),
                                 nome_frame in approvati_ora)
        per_stato.setdefault(stato, []).append(nome_frame)

    nomi = [str(n) for n in (payload.get("names") or []) if str(n).strip()]
    uno = str(payload.get("name") or "").strip()
    if uno:
        nomi.append(uno)
    for stato in (payload.get("statuses") or []):
        nomi.extend(per_stato.get(str(stato)) or [])
    nomi = sorted(set(nomi))
    if not nomi:
        return jsonify({"error": "nessun fotogramma da approvare"}), 400

    togli = bool(payload.get("reset"))

    def mutate(_p: Project, value: Dict) -> Dict:
        approvati = set(value.get("approved") or [])
        if togli:
            approvati -= set(nomi)
        else:
            approvati |= set(nomi)
        value["approved"] = sorted(approvati)
        return value

    valore = _write_step(project_id, "scale_study", mutate, status="corrected", source="user")
    return jsonify({"approved": len(nomi), "names": nomi, "removed": togli,
                    "total": len(valore.get("approved") or [])})


@app.post("/api/projects/<project_id>/scale/study/propagate")
def api_scale_study_propagate(project_id: str):
    """Da un fotogramma sistemato a tutti gli altri.

    Quello che si porta in giro **non e' il passo in pixel**: quello cambia da immagine a
    immagine, perche' cambia la depth. Sono di cartella la colonna, lo zero, il verso e il
    passo in **millimetri** - la macchina mette le tacche ogni tot millimetri e non cambia
    idea a meta' cartella. Il passo in pixel si ricalcola per ogni fotogramma dalla sua
    depth confermata: `passo_px = passo_mm * (fondo - zero) / depth`.

    Chi ha gia' una correzione a mano non si tocca: e' verita', non un'ipotesi da rifare.
    """
    project = _project(project_id)
    payload = _payload()
    modello_nome = str(payload.get("from") or "").strip()
    dati = _dati_studio(project)
    if dati is None:
        return jsonify({"error": "lo studio della scala non e' ancora stato fatto"}), 404
    frames = dati.get("frames") or []
    correzioni = project.step_value("scale_study").get("corrections") or {}
    modello_frame = next((f for f in frames if f.get("name") == modello_nome), None)
    if modello_frame is None:
        return jsonify({"error": f"fotogramma non nello studio: {modello_nome or '(vuoto)'}"}), 400

    confermate = _depth_confermate(project)
    modello = _righello_effettivo(modello_frame, correzioni.get(modello_nome))
    controllo_modello = _controlla_righello(modello_frame, modello,
                                            confermate.get(modello_nome))
    if controllo_modello.get("unit_hint") and not payload.get("force"):
        return jsonify({"error": "il fotogramma di riferimento non torna con la sua depth "
                                 "confermata: " + controllo_modello["unit_hint"],
                        "check": controllo_modello}), 400
    passo_mm = (float(payload["step_mm"]) if payload.get("step_mm") not in (None, "")
                else controllo_modello.get("step_mm") or modello.get("step_mm"))
    if not passo_mm:
        return jsonify({"error": "non si sa di quanti millimetri e' il passo: "
                                 "sistemalo prima sul fotogramma di riferimento"}), 400
    if modello.get("x") is None or modello.get("y_zero") is None:
        return jsonify({"error": "il fotogramma di riferimento non ha colonna e zero"}), 400

    sovrascrivi = bool(payload.get("overwrite"))
    esiti: List[Dict] = []
    nuove: Dict[str, Dict] = {}
    for frame in frames:
        nome = str(frame.get("name") or "")
        if nome == modello_nome:
            continue
        gia = correzioni.get(nome) or {}
        if gia and not gia.get("from_propagation") and not sovrascrivi:
            esiti.append({"name": nome, "done": False, "why": "corretto a mano, lasciato stare"})
            continue
        righello = _righello_effettivo(frame, None)
        righello["x"] = modello["x"]
        righello["y_zero"] = modello["y_zero"]
        righello["zero_end"] = modello["zero_end"]
        verso = 1.0 if modello["zero_end"] == "top" else -1.0
        depth = confermate.get(nome)
        controllo = _controlla_righello(frame, righello, depth)
        # Una depth che non torna col righello non e' un dato: e' un problema da sistemare
        # prima. Meglio fermarsi su quel fotogramma che propagarci sopra.
        if controllo.get("unit_hint"):
            esiti.append({"name": nome, "done": False, "why": controllo["unit_hint"]})
            continue
        if not depth:
            esiti.append({"name": nome, "done": False,
                          "why": "manca la depth confermata: confermala nella sezione prima"})
            continue

        # La barra e' di cartella. La macchina la disegna alla stessa altezza sempre: a
        # cambiare con la depth non e' la sua lunghezza, e' la scala. Il fondo suo lo si
        # tiene solo se regge il confronto con la sua depth; se no vale quello del modello,
        # che e' stato guardato.
        span_modello = abs(float(modello["y_far"]) - float(modello["y_zero"]))
        suo = righello.get("y_far")
        span = span_modello
        da_dove_fondo = "barra di cartella"
        if suo is not None and (controllo.get("span_off") is not None
                                and controllo["span_off"] <= 0.03):
            span = abs(float(suo) - float(modello["y_zero"]))
            da_dove_fondo = "barra sua, che torna con la sua depth"
        y_far = modello["y_zero"] + verso * span
        altezza = float(frame.get("h") or 0)
        if altezza and not (0 <= y_far <= altezza):
            esiti.append({"name": nome, "done": False,
                          "why": f"il fondo cadrebbe fuori dall'immagine ({y_far:.0f} px)"})
            continue
        # E qui sta il punto: il passo in pixel non si copia, si ricalcola. Stesso passo in
        # millimetri, stessa barra, depth diversa - quindi passo in pixel diverso.
        passo_px = passo_mm * span / depth
        if not passo_px or passo_px <= 2:
            esiti.append({"name": nome, "done": False,
                          "why": f"il passo verrebbe {passo_px:.1f} px, troppo fitto"})
            continue
        tacche = _griglia_tacche(modello["y_zero"], y_far, passo_px, True)
        if len(tacche) > 1:
            passo_px = abs(tacche[1] - tacche[0])
        if not tacche:
            esiti.append({"name": nome, "done": False, "why": "nessuna tacca con questo passo"})
            continue
        nuove[nome] = {
            "x": modello["x"], "y_zero": modello["y_zero"], "y_far": round(y_far, 1),
            "zero_end": modello["zero_end"], "ticks": tacche, "pitch": round(passo_px, 2),
            "from_propagation": modello_nome,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        esiti.append({"name": nome, "done": True, "ticks": len(tacche),
                      "pitch": round(passo_px, 2), "y_far": round(y_far, 1),
                      "depth_mm": depth, "mm_per_px": round(depth / span, 5),
                      "labels": _etichette_sulle_tacche(frame, tacche),
                      "why": f"{da_dove_fondo}; passo {passo_mm:g} mm su {depth:g} mm di depth"})

    if payload.get("preview"):
        return jsonify({"from": modello_nome, "step_mm": passo_mm, "preview": True,
                        "would_change": len(nuove), "results": esiti})

    def mutate(_p: Project, value: Dict) -> Dict:
        tutte = dict(value.get("corrections") or {})
        for nome, voce in nuove.items():
            fusa = dict(tutte.get(nome) or {})
            fusa.update(voce)
            fusa.pop("ticks_add", None)
            fusa.pop("ticks_del", None)
            tutte[nome] = fusa
        value["corrections"] = tutte
        return value

    _write_step(project_id, "scale_study", mutate, status="corrected", source="user")
    _write_scale_corrections(_project(project_id))
    return jsonify({"from": modello_nome, "step_mm": passo_mm,
                    "changed": len(nuove), "results": esiti})


@app.post("/api/projects/<project_id>/scale/study/accept")
def api_scale_study_accept(project_id: str):
    """Accetta i righelli proposti: su un fotogramma, o su tutti quelli falliti in un colpo.

    Otto fotogrammi da sistemare uno per uno sono otto volte le stesse quattro conferme. Se la
    proposta e' buona — e lo si vede dal righello disegnato sopra — vale la pena accettarla in
    blocco e poi ritoccare i pochi che stonano.
    """
    project = _project(project_id)
    payload = _payload()
    solo = str(payload.get("name") or "").strip()

    stadi = ((project.data.get("analysis") or {}).get("stages") or {})
    percorso = str((stadi.get("scale_study") or {}).get("data_json") or "")
    if not percorso or not Path(percorso).is_file():
        return jsonify({"error": "lo studio della scala non e' ancora stato fatto"}), 404
    dati = json.loads(Path(percorso).read_text(encoding="utf-8"))
    base = project.working_dir() or Path("")
    for frame in dati.get("frames") or []:
        percorso_frame = Path(str(frame.get("path") or ""))
        try:
            frame["name"] = str(percorso_frame.relative_to(base))
        except ValueError:
            frame["name"] = percorso_frame.name
    correzioni_ora = project.step_value("scale_study").get("corrections") or {}
    proposte = _suggerisci_righelli(dati.get("frames") or [], correzioni_ora,
                                    _depth_confermate(project))
    if solo:
        proposte = {k: v for k, v in proposte.items() if k == solo}
    if not proposte:
        return jsonify({"accepted": 0, "nothing": True})

    def mutate(_p: Project, value: Dict) -> Dict:
        correzioni = dict(value.get("corrections") or {})
        for nome, proposta in proposte.items():
            voce = dict(correzioni.get(nome) or {})
            voce.update({
                "x": proposta["x"], "y_zero": proposta["y_zero"],
                "y_far": proposta["y_far"], "zero_end": proposta["zero_end"],
                "ticks_add": proposta.get("ticks") or [],
                "from_suggestion": True,
                "ts": datetime.now().isoformat(timespec="seconds"),
            })
            correzioni[nome] = voce
        value["corrections"] = correzioni
        return value

    valore = _write_step(project_id, "scale_study", mutate, status="corrected", source="user")
    _write_scale_corrections(_project(project_id))
    return jsonify({"accepted": len(proposte), "names": sorted(proposte),
                    "corrections": len(valore.get("corrections") or {})})


@app.post("/api/projects/<project_id>/scale/study/run")
def api_scale_study_run(project_id: str):
    """Rifai lo studio del righello, con dentro le correzioni fatte finora."""
    _project(project_id)
    quante = int(_payload().get("max_images") or 0)
    return jsonify({"job_id": _start_job(_run_scale_study_only, project_id, quante)})


def _run_scale_study_only(job_id: str, project_id: str, max_images: int) -> None:
    try:
        project = _project(project_id)
        stadi = ((project.data.get("analysis") or {}).get("stages") or {})
        contesto = str((stadi.get("scale") or {}).get("output_dir") or "")
        if not contesto or not (Path(contesto) / "pipeline_context.json").is_file():
            raise ValueError("serve prima lo stadio della scala: lancia «Calcola la scala»")
        correzioni = _write_scale_corrections(project)
        # Lo stato di prima, per poter dire dopo quali fotogrammi sono cambiati grazie alle
        # correzioni: senza questo confronto "rifai lo studio" e' un salto nel buio.
        prima = _stati_studio(project)
        correzioni_prima = sorted((project.step_value("scale_study").get("corrections") or {}))
        # Stessi fotogrammi del giro precedente, altrimenti il confronto non vuol dire niente:
        # con un insieme piu' grande i nuovi arrivati sembrerebbero comparsi dal nulla.
        quante = max_images or max(len(prima), 14)
        _job_update(job_id, stage=f"studio del righello su {quante} fotogrammi")
        esito = stages_mod.run_scale_study(
            context_dir=Path(contesto), python_bin=sys.executable,
            vendor=((project.data.get("analysis") or {}).get("vendor") or {}).get("vendor") or "",
            max_images=quante, corrections=correzioni,
        )
        if esito.get("status") != "ok":
            raise ValueError(esito.get("error") or "lo studio non e' andato a buon fine")
        project = _project(project_id)
        analisi = dict(project.data.get("analysis") or {})
        stadi = dict(analisi.get("stages") or {})
        stadi["scale_study"] = {k: esito.get(k) for k in
                                ("status", "frames", "by_status", "vendor", "zone",
                                 "output_dir", "data_json", "corrected_frames")}
        analisi["stages"] = stadi
        project.data["analysis"] = analisi
        project.save()

        dopo = _stati_studio(_project(project_id))
        confronto = _esito_run_studio(prima, dopo, correzioni_prima)

        def registra(_p: Project, value: Dict) -> Dict:
            value["last_run"] = confronto
            return value

        _write_step(project_id, "scale_study", registra,
                    status="corrected" if correzioni_prima else "proposed",
                    source="user" if correzioni_prima else "model")
        _job_update(job_id, status="done", stage="fatto",
                    result={**stadi["scale_study"], "last_run": confronto})
    except Exception as error:  # noqa: BLE001
        _job_update(job_id, status="error", stage="errore", error=str(error))


def _stati_studio(project: Project) -> Dict[str, str]:
    """Lo stato di ogni fotogramma nello studio attuale, per poterlo confrontare dopo."""
    stadi = ((project.data.get("analysis") or {}).get("stages") or {})
    percorso = str((stadi.get("scale_study") or {}).get("data_json") or "")
    if not percorso or not Path(percorso).is_file():
        return {}
    try:
        dati = json.loads(Path(percorso).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    base = project.working_dir() or Path("")
    correzioni = project.step_value("scale_study").get("corrections") or {}
    fuori: Dict[str, str] = {}
    for frame in dati.get("frames") or []:
        percorso_frame = Path(str(frame.get("path") or ""))
        try:
            nome = str(percorso_frame.relative_to(base))
        except ValueError:
            nome = percorso_frame.name
        # Senza le approvazioni: qui si confronta cosa fa il **modulo** prima e dopo le
        # correzioni, e un'approvazione non e' un suo miglioramento.
        fuori[nome] = _stato_effettivo(frame, correzioni.get(nome))
    return fuori


ORDINE_STATO = {"reject": 0, "review": 1, "accepted": 2, "approved": 3, "corrected": 3}


def _stato_effettivo(frame: Dict, correzione: Optional[Dict],
                     approvato: bool = False) -> str:
    """Lo stato che conta per chi guarda: il modulo non lo ricalcola dopo una correzione.

    `study_scale_folder` applica le correzioni *dopo* la detection e lascia `status` com'era —
    scelta giusta la' dentro, perche' cosi' si continua a vedere cosa avrebbe detto da solo.
    Ma se il righello glielo hai dato tu, quel fotogramma un righello ce l'ha, e chiamarlo
    ancora `reject` fa credere che la correzione non sia servita a niente.

    `approvato` e' la terza voce: il modulo non era sicuro, lei ha guardato e va bene cosi'.
    Non e' una correzione — non c'e' niente da correggere — ma nemmeno un dubbio del modulo
    che resti aperto. Un "da rivedere" gia' rivisto e' rumore in mezzo a quelli veri.
    """
    if correzione and all(correzione.get(k) is not None
                          for k in ("x", "y_zero", "y_far")):
        return "corrected"
    if approvato:
        return "approved"
    return str(frame.get("status") or "")


def _esito_run_studio(prima: Dict[str, str], dopo: Dict[str, str],
                      usate: Sequence[str]) -> Dict:
    """Cosa e' cambiato fra due studi: e' questo che dice se una correzione e' servita."""
    migliorati, peggiorati = [], []
    for nome, stato in dopo.items():
        vecchio = prima.get(nome)
        if vecchio is None or vecchio == stato:
            continue
        (migliorati if ORDINE_STATO.get(stato, -1) > ORDINE_STATO.get(vecchio, -1)
         else peggiorati).append({"name": nome, "from": vecchio, "to": stato})
    return {
        "at": datetime.now().isoformat(timespec="seconds"),
        "used": sorted(usate),
        "improved": migliorati,
        "worsened": peggiorati,
        "accepted_total": sum(1 for v in dopo.values() if v in ("accepted", "corrected")),
        "frames": len(dopo),
    }


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
    # Solo i fotogrammi di questo progetto: lo studio puo' essere di prima dello sdoppiamento.
    mie = _sue_immagini(project)
    if mie:
        dati["frames"] = [f for f in (dati.get("frames") or []) if f.get("name") in mie]
    valore = project.step_value("scale_study")
    correzioni = valore.get("corrections") or {}
    dati["corrections"] = correzioni
    dati["note"] = valore.get("note") or ""
    ultima = valore.get("last_run") or {}
    dati["last_run"] = ultima
    # Una correzione fatta dopo l'ultima run non e' ancora entrata nel calcolo: dirlo e' la
    # differenza fra "salvata" e "usata", e sono due cose diverse.
    quando = str(ultima.get("at") or "")
    dati["pending"] = sorted(
        nome for nome, voce in correzioni.items()
        if not quando or str(voce.get("ts") or "") > quando
    )
    confermate = _depth_confermate(project)
    approvati = set(valore.get("approved") or [])
    dati["approved"] = sorted(approvati)
    per_nome = {}
    for frame in dati.get("frames") or []:
        # Lo stato del modulo resta visibile in `detector_status`: serve a vedere se la
        # detection da sola migliora, che e' l'altra domanda.
        frame["detector_status"] = str(frame.get("status") or "")
        frame["approved"] = frame["name"] in approvati
        frame["status"] = _stato_effettivo(frame, correzioni.get(frame["name"]),
                                           frame["approved"])
        # Il confronto con la depth confermata, fotogramma per fotogramma: e' il dato
        # esterno che dice se il righello e' al posto giusto, e va visto accanto a lui.
        righello = _righello_effettivo(frame, correzioni.get(frame["name"]))
        controllo = _controlla_righello(frame, righello, confermate.get(frame["name"]))
        controllo["labels_on_ticks"] = _etichette_sulle_tacche(frame, righello["ticks"])
        frame["depth_check"] = controllo
        frame["ruler"] = righello
        per_nome[frame["name"]] = frame
    for cambio in ultima.get("improved") or []:
        frame = per_nome.get(cambio["name"])
        if frame is not None:
            frame["improved_from"] = cambio["from"]
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
    # Le immagini che nessuno ha ancora guardato: fuori dal campione del modulo e senza
    # riquadro che le abbia coperte. Comparivano solo come "12 su 24" nella riga della
    # copertura, e non c'era modo di arrivarci per indicare la depth a mano. Ci sono, con
    # lo stato che dice cosa sono: da qui si filtrano, si aprono e si scrivono.
    presenti = {r["name"] for r in righe}
    for nome in project.dedup_names():
        if nome in presenti:
            continue
        righe.append({
            "name": nome, "status": "missing", "mode": "", "depth_mm": None, "score": None,
            "box": None, "ocr_text": "", "candidates": None,
            "reason": "il modulo non l'ha esaminata: la depth si puo' indicare a mano",
            "direct": {}, "scale": {}, "in_run": False,
        })
    # Solo le immagini di questo progetto: la run della depth puo' essere di prima dello
    # sdoppiamento, e conterrebbe ancora quelle dell'altro piano.
    mie = _sue_immagini(project)
    if mie:
        righe = [r for r in righe if r["name"] in mie]
    righe.sort(key=lambda r: r["name"])
    correzioni = valore_step.get("depth_corrections") or {}
    for nome, fix in correzioni.items():
        for r in righe:
            if r["name"] == nome:
                r.update({"depth_mm": fix.get("depth_mm"), "corrected": True,
                          "status": "corrected", "note": fix.get("note", "")})

    # Guardata e va bene cosi'. Non e' una correzione - non c'e' niente da correggere - ma
    # nemmeno un numero che nessuno ha ancora verificato: e' la differenza fra «il modulo
    # dice 46» e «46 e' giusto». Vale per il valore di allora: se poi cambia, la conferma
    # non vale piu' e l'immagine torna da guardare.
    conferme = valore_step.get("depth_reviewed") or {}
    for r in righe:
        segno = conferme.get(r["name"])
        r["reviewed"] = bool(
            segno and r["depth_mm"] is not None
            and abs(float(segno.get("depth_mm") or 0) - float(r["depth_mm"])) < 1e-6
        )

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
            "box_tightened": valore_step.get("depth_box_tightened") or None,
            "box_modes": list(DEPTH_BOX_MODES),
            "modes": DEPTH_MODES,
            "by_mode": {m: sum(1 for r in righe if r["mode"] == m) for m in DEPTH_MODES},
            "by_status": {
                s: sum(1 for r in righe if r["status"] == s)
                for s in ("accepted", "review", "reject", "corrected", "box", "missing")
            },
            "values_mm": valori,
            "reviewed_count": sum(1 for r in righe if r.get("reviewed")),
            "images_total": len(project.dedup_images()),
            "coverage": _copertura_depth(project, righe),
            "coherence": _coerenza_depth(righe),
            "confirmed": str((project.steps.get("depth_scale") or {}).get("status") or "")
                         in ("confirmed", "corrected"),
            "stage_dir": str(stage),
        }
    )


_FORMATO_OCR = re.compile(r"(\d+)(?:[.,](\d+))?\s*(cm|mm)?", re.I)


def _forma_lettura(testo: str) -> str:
    """La *forma* di un'etichetta: quante cifre, se c'e' la virgola, quale unita'.

    In una cartella la macchina scrive la depth sempre allo stesso modo, e chi devia di solito
    non e' una depth diversa: e' la stessa letta male. `3.5 cm` e `35cm` sono lo stesso numero,
    ma il secondo, preso alla lettera, vale dieci volte tanto.
    """
    trovato = _FORMATO_OCR.search(str(testo or "").lower())
    if not trovato:
        return ""
    decimali = f".{len(trovato.group(2))}" if trovato.group(2) else ""
    return f"{len(trovato.group(1))}{decimali}{trovato.group(3) or ''}"


def _coerenza_depth(righe: Sequence[Dict]) -> Dict:
    """Le letture che stonano rispetto al resto della cartella, e cosa proporre al loro posto.

    Due segnali, tenuti distinti perche' uno permette una proposta e l'altro no.

    *Valore fuori scala*: se dividendolo per dieci cade su un valore che la cartella ha gia',
    e' un punto decimale perso — `30cm` dove le altre dicono `3.0 cm`. Si propone il valore
    diviso. Attenzione che non basta la forma: in questa stessa cartella `10 cm` ha la stessa
    forma di `30cm` ma vale davvero 100 mm, ed e' il fondo scala della sonda.

    *Forma diversa dal resto*: `5 cm` dove tutte le altre sono `d.d cm` ha perso una cifra, ma
    quale non si sa — 50 mm e' un valore plausibile e nessuna proposta sarebbe onesta. Si
    segnala e basta.
    """
    forme = [_forma_lettura(r.get("ocr_text")) for r in righe if r.get("ocr_text")]
    forme = [f for f in forme if f]
    dominante = max(set(forme), key=forme.count) if forme else ""

    valori = [float(r["depth_mm"]) for r in righe if r.get("depth_mm")]
    if not valori:
        return {"format": dominante, "suspects": {}, "values": []}
    mediana = statistics.median(valori)
    normali = sorted({v for v in valori if v <= mediana * 3})

    def vicino(valore: float) -> Optional[float]:
        for buono in normali:
            if abs(buono - valore) <= max(1.0, 0.05 * buono):
                return buono
        return None

    def cifre(forma: str) -> int:
        """Quante cifre ha l'etichetta, punto escluso: `3.5` e `10` ne hanno due."""
        numeri = re.findall(r"\d+", forma.split("c")[0].split("m")[0] or "0")
        return sum(int(n) for n in numeri) if numeri else 0

    def cifra_scambiata(valore: float) -> Optional[float]:
        """Il valore buono che differisce da questo per **una sola cifra**.

        `411` in una cartella che ha `111` non e' un punto decimale perso: e' un 1 letto 4.
        Diviso dieci fa 41.1, che non e' un valore di questa cartella - la scala e' fatta di
        18, 28, 37... Se invece esiste un valore noto, uno solo, che differisce per una cifra
        sola, quella e' l'ipotesi giusta e si puo' proporre.
        """
        letto = f"{valore:g}".replace(".", "")
        trovati = []
        for buono in normali:
            testo = f"{buono:g}".replace(".", "")
            if len(testo) != len(letto):
                continue
            if sum(1 for a, b in zip(testo, letto) if a != b) == 1:
                trovati.append(buono)
        return trovati[0] if len(trovati) == 1 else None

    basso, alto = min(normali), max(normali)
    cifre_attese = cifre(dominante)

    sospetti: Dict[str, Dict] = {}
    for riga in righe:
        valore = riga.get("depth_mm")
        nome = str(riga.get("name") or "")
        if not valore or not nome:
            continue
        if valore > mediana * 3:
            diviso = valore / 10.0
            # Basta che il valore diviso stia nella scala della cartella: un `3.0 cm` letto
            # male su *tutti* i fotogrammi non lascia nessun 30 giusto da cui riconoscerlo.
            proposto = vicino(diviso)
            perche = "punto decimale perso"
            if proposto is None:
                # Prima di dare la colpa al punto decimale: una cifra letta male spiega
                # meglio un valore che, diviso dieci, non cade su niente di questa cartella.
                scambio = cifra_scambiata(valore)
                if scambio is not None:
                    proposto, perche = scambio, "una cifra letta male"
            if proposto is None and basso * 0.5 <= diviso <= alto:
                proposto = round(diviso, 1)
            if proposto is not None:
                fatto = (f"diviso dieci fa {proposto:g} mm"
                         if perche == "punto decimale perso"
                         else f"la cartella ha {proposto:g} mm, che differisce per una cifra sola")
                sospetti[nome] = {
                    "suggested_mm": proposto,
                    "reason": (f"{valore:g} mm e' fuori scala per questa cartella "
                               f"({basso:g}-{alto:g} mm); {fatto}. "
                               f"Letto «{riga.get('ocr_text')}»: {perche}"),
                }
            else:
                sospetti[nome] = {
                    "suggested_mm": None,
                    "reason": f"{valore:g} mm e' fuori scala rispetto al resto della cartella",
                }
            continue
        # La forma da sola non basta: in questa cartella `10 cm` non ha la virgola come le
        # altre, ma ha lo stesso numero di cifre ed e' il fondo scala della sonda. Conta
        # quindi se le cifre sono *meno* di quelle attese, cioe' se ne e' persa una.
        forma = _forma_lettura(riga.get("ocr_text"))
        if dominante and forma and cifre_attese and cifre(forma) < cifre_attese:
            sospetti[nome] = {
                "suggested_mm": None,
                "reason": (f"letto «{riga.get('ocr_text')}»: {cifre(forma)} cifra dove il resto "
                           f"della cartella ne ha {cifre_attese} — ne manca una, ma quale non "
                           f"si puo' indovinare"),
            }
    return {"format": dominante, "suspects": sospetti, "values": normali,
            "range": [basso, alto]}


def _copertura_depth(project: Project, righe: Sequence[Dict]) -> Dict:
    """Quante immagini hanno una depth, e **perche'** le altre no.

    Il modulo gira su un campione perche' generare i candidati costa minuti. Dove la depth e'
    scritta nell'interfaccia il campione basta: da li' si propaga il riquadro e si copre tutta
    la cartella in mezzo secondo per immagine. Dove invece la depth viene letta *dalla scala*,
    il numero cambia posto ad ogni fotogramma e un riquadro non si puo' propagare: l'unico modo
    di coprire tutte le immagini e' far girare il modulo su tutte.

    Senza questa spiegazione la sezione mostra dodici immagini su trentasei e basta.
    """
    totale = len(project.dedup_images())
    con_depth = sum(1 for r in righe if r.get("depth_mm") is not None)
    modi = [r.get("mode") for r in righe if r.get("mode")]
    prevalente = max(set(modi), key=modi.count) if modi else ""
    propagabile = prevalente in DEPTH_BOX_MODES
    if con_depth >= totale:
        motivo = ""
    elif propagabile:
        motivo = ("il modulo gira su un campione; il riquadro si propaga al resto della "
                  "cartella in mezzo secondo per immagine")
    else:
        motivo = ("qui la depth viene letta dalla scala, non da un'etichetta: il numero cambia "
                  "posto ad ogni fotogramma, quindi un riquadro non si puo' propagare. Per "
                  "coprire tutte le immagini bisogna far girare il modulo su tutte")
    return {"with_depth": con_depth, "total": totale, "mode": prevalente,
            "can_propagate": propagabile, "reason": motivo}


@app.post("/api/projects/<project_id>/depth/confirm")
def api_depth_confirm(project_id: str):
    """Conferma la depth cosi' com'e', senza toccarne il valore.

    `POST /steps/<id>` sostituirebbe il valore dello step, e li' dentro vivono anche la
    rilettura su tutta la cartella e le correzioni: confermare non deve cancellare niente.
    Da qui in poi queste depth valgono come confermate, e lo studio della scala le usa.
    """
    _project(project_id)
    annulla = bool(_payload().get("reset"))
    valore = _write_step(project_id, "depth_scale", lambda _p, v: v,
                         status="proposed" if annulla else "confirmed",
                         source="model" if annulla else "user")
    return jsonify({"confirmed": not annulla, "values": len(valore.get("depths") or [])})


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
    stretto = {
        "left": int(box["left"] + da),
        "right": int(box["left"] + a + 1),
        "top": int(box["top"] + righe[0]),
        "bottom": int(box["top"] + righe[-1] + 1),
    }
    # Una cifra non e' mai alta 2 px: quando l'inchiostro trovato e' questo, quello che si e'
    # letto non era un numero, e il riquadro da tenere e' quello di partenza - almeno si
    # vede cosa e' stato guardato.
    if stretto["right"] - stretto["left"] < 4 or stretto["bottom"] - stretto["top"] < 5:
        return box
    return stretto


def _inchiostro_fra(image_path: Path, box: Dict, x_da: float, x_a: float) -> Optional[Dict]:
    """Il riquadro dell'inchiostro che sta fra due ascisse, dentro a `box`.

    Come `_stringi_sui_pixel`, ma invece di contare i glifi da sinistra si guarda **dove**
    stanno: dei gruppi di colonne accese si tengono quelli il cui centro cade nella
    finestra. Serve alle etichette che hanno le lettere *prima* del numero — l'Hitachi
    scrive `R:3.00`, e contando da sinistra si tiene la R e si perde l'ultima cifra.
    """
    from PIL import Image  # noqa: PLC0415

    try:
        with Image.open(image_path) as immagine:
            ritaglio = immagine.convert("L").crop(
                (int(box["left"]), int(box["top"]), int(box["right"]), int(box["bottom"]))
            )
        larghezza, altezza = ritaglio.size
        if larghezza < 2 or altezza < 2:
            return None
        px = ritaglio.load()
        valori = [px[x, y] for y in range(altezza) for x in range(larghezza)]
        chiaro = sum(1 for v in valori if v >= 140) > len(valori) / 2
        if chiaro:
            soglia = min(140, max(40, min(valori) + 45))
            acceso = lambda v: v <= soglia  # noqa: E731
        else:
            soglia = max(70, min(155, max(valori) - 35))
            acceso = lambda v: v >= soglia  # noqa: E731
        colonne = [x for x in range(larghezza) if any(acceso(px[x, y]) for y in range(altezza))]
    except Exception:  # noqa: BLE001
        return None
    if not colonne:
        return None

    gruppi: List[Tuple[int, int]] = []
    inizio = precedente = colonne[0]
    for colonna in colonne[1:]:
        if colonna - precedente > 1:
            gruppi.append((inizio, precedente))
            inizio = colonna
        precedente = colonna
    gruppi.append((inizio, precedente))

    dentro = [(a, b) for a, b in gruppi
              if x_da <= box["left"] + (a + b) / 2.0 <= x_a]
    if not dentro:
        return None
    # La finestra e' una stima in proporzione, e si sbaglia quando l'OCR aggiunge un
    # carattere di troppo: su `R:12.0.` il punto finale inventato accorcia il passo e
    # l'ultimo `0` cade appena fuori. Si allunga la selezione ai gruppi **attaccati**, con
    # la distanza fra le cifre gia' selezionate a fare da metro: quello che sta lontano come
    # una spaziatura di parola (le lettere, l'unita') resta fuori.
    interni = [dentro[i + 1][0] - dentro[i][1] for i in range(len(dentro) - 1)]
    vicino = max(3, (max(interni) if interni else 0) + 1)

    def righe_di(da: int, a: int) -> Tuple[int, int]:
        y = [yy for yy in range(altezza) if any(acceso(px[x, yy]) for x in range(da, a + 1))]
        return (y[0], y[-1]) if y else (0, -1)

    # Le cifre stanno tutte sulla stessa riga: e' questo a distinguerle da cio' che le sta
    # accanto. Su prova_4 dopo `R:15.0` c'e' un triangolino di marcatura alto la meta',
    # e su prova_3 un riquadro finito sul righello prendeva la riga bianca al posto del
    # numero. Un gruppo che non divide la riga con le cifre non e' una cifra.
    base_su, base_giu = righe_di(dentro[0][0], dentro[-1][1])
    base_alta = max(1, base_giu - base_su + 1)

    def sulla_stessa_riga(da: int, a: int) -> bool:
        su, giu = righe_di(da, a)
        if giu < su:
            return False
        comune = min(base_giu, giu) - max(base_su, su) + 1
        return comune >= 0.6 * base_alta
    # ...ma non oltre: la finestra dice quanto e' lungo il numero, e crescere molto di piu'
    # vuol dire aver preso qualcos'altro. Su prova_4 due fotogrammi hanno un triangolino
    # grigio subito dopo l'etichetta, e senza tetto se lo portavano dentro (55 px invece
    # dei 36 delle altre).
    tetto = (x_a - x_da) * 1.25
    posizione = gruppi.index(dentro[-1])
    while (posizione + 1 < len(gruppi)
           and gruppi[posizione + 1][0] - dentro[-1][1] <= vicino
           and gruppi[posizione + 1][1] - dentro[0][0] <= tetto
           and sulla_stessa_riga(*gruppi[posizione + 1])):
        posizione += 1
        dentro.append(gruppi[posizione])
    da, a_fine = dentro[0][0], dentro[-1][1]
    righe = [y for y in range(altezza) if any(acceso(px[x, y]) for x in range(da, a_fine + 1))]
    if not righe:
        return None
    return {"left": int(box["left"] + da), "right": int(box["left"] + a_fine + 1),
            "top": int(box["top"] + righe[0]), "bottom": int(box["top"] + righe[-1] + 1)}


def _vicino_al_riquadro(partenza: Dict, arrivo: Dict) -> bool:
    """Il riquadro stretto deve restare dov'era il numero.

    La seconda passata cerca l'etichetta in una banda piu' larga, e su un fotogramma di
    prova_3 si e' agganciata al righello: 252 px di riga bianca invece di 84 di etichetta.
    Un riquadro largo il doppio, o spostato di piu' di mezza larghezza, non e' lo stesso
    numero visto meglio: e' un altro pezzo di schermo.
    """
    larghezza = max(1, partenza["right"] - partenza["left"])
    if arrivo["right"] - arrivo["left"] > 1.8 * larghezza:
        return False
    # A destra si e' piu' larghi che a sinistra, e non per simmetria: il riquadro di
    # partenza spesso *taglia* l'ultima cifra (l'OCR conta i glifi sbagliati quando c'e' un
    # prefisso), e il riquadro giusto finisce piu' in la' di quello vecchio. A sinistra
    # invece non c'e' niente da recuperare: li' c'e' solo il prefisso da togliere.
    return (arrivo["left"] >= partenza["left"] - 0.6 * larghezza
            and arrivo["right"] <= partenza["right"] + larghezza)


def _riquadro_sulle_cifre(image_path: Path, box: Dict, cifre: str,
                          timeout: float = 8.0) -> Optional[Dict]:
    """Il riquadro sulle sole cifre, trovando prima **tutta** l'etichetta.

    L'OCR con la lista di soli numeri consegna un riquadro di parola che comprende quel che
    sta attaccato: `R:3.00` torna come «3.00» ma largo da `R` a fine numero. Rileggendo
    senza restrizioni si ottiene invece l'etichetta intera, e a quel punto si sa *dove* sono
    le cifre dentro alla stringa: la loro posizione, in proporzione, da' la finestra in cui
    cercare l'inchiostro.
    """
    if not cifre:
        return None
    parole_di = _tesseract_words()
    larghezza = max(1.0, float(box["right"] - box["left"]))
    banda = (box["left"] - larghezza, box["top"] - 6, box["right"] + larghezza, box["bottom"] + 6)
    migliori: List[Tuple[int, float, Dict]] = []
    for variante in ("base", "invert"):
        for psm in ("7", "11"):
            try:
                parole, _ = parole_di(image_path, timeout=timeout, max_side=1800,
                                      crop_box=banda, psm=psm, preprocess=variante)
            except Exception:  # noqa: BLE001
                continue
            for parola in parole:
                testo = str(getattr(parola, "text", "") or "").strip().replace(",", ".")
                indice = testo.find(cifre)
                lunghezza = len(cifre)
                grado = 0
                if indice < 0:
                    # Le due letture non coincidono sempre: con la lista di soli numeri
                    # l'Hitachi da' «7.006», senza restrizioni «R:7.00-». Il numero e'
                    # quello, e quel che serve qui e' *dove* sta, non come si scrive.
                    corsa = re.search(r"\d+(?:\.\d+)?", testo)
                    if not corsa:
                        continue
                    indice, lunghezza, grado = corsa.start(), len(corsa.group(0)), 1
                # Solo se c'e' qualcosa **prima** del numero. Quando l'etichetta comincia
                # con le cifre (`3.5 cm`, `43 *mm`) il riquadro della prima passata parte
                # gia' dalla cifra ed e' piu' stretto: questa passata lo peggiorerebbe.
                if indice == 0:
                    continue
                per_carattere = (float(parola.right) - float(parola.left)) / max(1, len(testo))
                migliori.append((
                    grado,
                    abs((float(parola.left) + float(parola.right)) / 2.0
                        - (box["left"] + box["right"]) / 2.0),
                    {"da": float(parola.left) + indice * per_carattere,
                     "a": float(parola.left) + (indice + lunghezza) * per_carattere,
                     "left": float(parola.left), "right": float(parola.right)},
                ))
    if not migliori:
        return None
    migliori.sort(key=lambda voce: (voce[0], voce[1]))
    scelto = migliori[0][2]
    # La scatola di lavoro copre tutta l'etichetta: il numero puo' finire oltre il riquadro
    # di partenza, che e' proprio il caso in cui contando da sinistra si perdeva una cifra.
    largo = {
        "left": int(min(box["left"], scelto["left"])),
        "right": int(max(box["right"], scelto["right"])),
        "top": box["top"], "bottom": box["bottom"],
    }
    return _inchiostro_fra(image_path, largo, scelto["da"], scelto["a"])


def _rileggi_nel_riquadro(image_path: Path, box: Dict, timeout: float = 8.0,
                          accettabile=None) -> Optional[Dict]:  # noqa: ANN001
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
    # Chi chiama puo' dire cosa si aspetta - una profondita' di lavoro, per esempio. Fra le
    # sei passate dell'OCR ce n'e' spesso una che legge `120` e un'altra che ci attacca
    # l'asterisco di `*mm` e fa `1200`: senza questo filtro vince la piu' vicina al bordo,
    # che puo' essere la seconda, e l'immagine resta senza numero.
    if accettabile is not None:
        # Solo fra quelli che stanno **nello stesso posto**: piu' in la' c'e' sempre un altro
        # numero (il `13` di `TLC3-13`, a ottantotto pixel), e ripescare quello sarebbe
        # peggio che restare senza.
        vicino = candidati[0]["distanza"] + max(10.0, 1.5 * altezza)
        buoni = [c for c in candidati
                 if c["distanza"] <= vicino and accettabile(c["value"])]
        if buoni:
            candidati = buoni
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


def _inviluppo_riquadri(riquadri: Sequence[Dict]) -> Optional[Dict]:
    """Il riquadro che li contiene tutti, se sono abbastanza e stanno davvero insieme.

    Ha senso solo se l'etichetta e' ferma: se i riquadri stretti sono sparsi per lo schermo
    non c'e' nessun "posto del numero" da prestare, e l'inviluppo sarebbe mezza interfaccia.
    """
    utili = [r for r in riquadri if r]
    if len(utili) < 3:
        return None
    fuori = {
        "left": min(r["left"] for r in utili), "top": min(r["top"] for r in utili),
        "right": max(r["right"] for r in utili), "bottom": max(r["bottom"] for r in utili),
    }
    largo_medio = statistics.median(r["right"] - r["left"] for r in utili)
    alto_medio = statistics.median(r["bottom"] - r["top"] for r in utili)
    if (fuori["right"] - fuori["left"] > 3 * largo_medio
            or fuori["bottom"] - fuori["top"] > 3 * alto_medio):
        return None
    return fuori


def _riquadri_slittati(riquadri: Dict[str, Dict], minimo: int = 6) -> List[str]:
    """I riquadri spostati rispetto alla mediana della cartella, non solo piu' o meno lunghi.

    Il discrimine e' la **direzione**: se entrambi i bordi si spostano dalla stessa parte il
    riquadro e' slittato, e allora e' su un'altra cosa. Se si sposta un bordo solo, il
    numero ha semplicemente una cifra in piu' o in meno, ed e' giusto cosi' - senza questa
    distinzione ogni `102` in mezzo ai `18` verrebbe scartato.
    """
    if len(riquadri) < minimo:
        return []
    sinistre = statistics.median(b["left"] for b in riquadri.values())
    destre = statistics.median(b["right"] for b in riquadri.values())
    larghezza = statistics.median(b["right"] - b["left"] for b in riquadri.values())
    soglia = max(6.0, 0.2 * larghezza)
    fuori = []
    for nome, b in riquadri.items():
        dx_sinistra = b["left"] - sinistre
        dx_destra = b["right"] - destre
        if (abs(dx_sinistra) > soglia and abs(dx_destra) > soglia
                and dx_sinistra * dx_destra > 0):
            fuori.append(nome)
    # Se "fuori" e' la maggioranza, la mediana non era il riferimento buono: meglio non
    # toccare niente che riscrivere mezza cartella su un sospetto.
    return fuori if len(fuori) * 2 < len(riquadri) else []


def _contiene(grande: Dict, piccolo: Dict) -> bool:
    """Il riquadro di cartella sta dentro a quello largo di questa immagine?

    Se non ci sta, qui l'etichetta e' altrove: prestarglielo lo porterebbe su un pezzo di
    schermo che non c'entra, ed e' meglio lasciare il riquadro largo com'e'.
    """
    return (piccolo["left"] >= grande["left"] - 2 and piccolo["top"] >= grande["top"] - 2
            and piccolo["right"] <= grande["right"] + 2
            and piccolo["bottom"] <= grande["bottom"] + 2)


def _cifre_lette(testo: str) -> str:
    """Il numero dentro a un'etichetta, senza unita' ne' lettere: `3.5 cm` -> `3.5`."""
    trovato = re.search(r"\d+(?:[.,]\d+)?", str(testo or ""))
    return trovato.group(0).replace(",", ".") if trovato else ""


def _riquadro_stretto(image_path: Path, box: Dict, atteso_testo: str,
                      atteso_mm: Optional[float], fattore: float) -> Dict:
    """Il riquadro portato sul **solo numero**, o il motivo per cui si e' lasciato stare.

    Il riquadro che il modulo consegna e' quello di parola dell'OCR, e la parola comprende
    quello che sta attaccato al numero: l'unita' (`43 *mm`), la lettera di una sigla, la
    tacca del righello se il numero le sta accanto. Va bene per leggere, non per andare a
    match su un'altra immagine: quel contesto cambia da un fotogramma all'altro, e il
    riquadro va stretto sulle cifre.

    Il valore e' il guardiano. Se rileggendo esce un numero diverso da quello che c'era,
    non e' un riquadro piu' stretto: e' un riquadro finito su un altro numero, e si lascia
    tutto com'era. Restringere non deve poter cambiare una depth gia' giusta.
    """
    letto = _rileggi_nel_riquadro(image_path, box)
    if letto is None:
        return {"ok": False, "reason": "in quel riquadro non si legge nessun numero"}
    stretto = letto["box"]
    # Seconda passata, che vince quando riesce: rilegge l'etichetta intera e ritaglia sulle
    # sole cifre, dove che siano dentro alla stringa. Serve dove le lettere stanno *prima*
    # del numero, e li' la prima passata non poteva farcela.
    sulle_cifre = _riquadro_sulle_cifre(image_path, box, _cifre_lette(letto["text"]))
    if sulle_cifre is not None and _vicino_al_riquadro(box, sulle_cifre):
        stretto = sulle_cifre
    valore = round(float(letto["value"]) * fattore, 2)
    # Il confronto si fa sulle **cifre**, non sui millimetri: e' l'unita' che stiamo
    # buttando fuori dal riquadro, e farla entrare nel guardiano vorrebbe dire rifiutare
    # proprio le etichette in cm. `3.5 cm` e `3.5` sono lo stesso numero letto due volte.
    atteso_cifre = _cifre_lette(atteso_testo)
    lette = _cifre_lette(letto["text"])
    # Due modi di dire la stessa cosa, e ne basta uno. Le stesse cifre: il riquadro sta
    # sullo stesso numero. Gli stessi millimetri: ci sta pure quando le cifre *migliorano* —
    # dove stava scritto «30cm» il riquadro stretto legge «3.0», che e' lo stesso 30 mm con
    # il punto ritrovato, e sarebbe assurdo rifiutarlo.
    d_accordo = (atteso_mm is not None and abs(valore - float(atteso_mm)) <= 0.01)
    if not d_accordo and atteso_cifre and lette != atteso_cifre:
        return {"ok": False, "value_mm": valore, "text": letto["text"],
                "reason": f"rileggendo esce «{letto['text']}» invece di «{atteso_testo}»: "
                          "il riquadro non si stringe"}
    if not d_accordo and not atteso_cifre and atteso_mm is not None:
        return {"ok": False, "value_mm": valore, "text": letto["text"],
                "reason": f"rileggendo esce {valore:g} mm invece di {float(atteso_mm):g}: "
                          "il riquadro non si stringe"}
    # Un riquadro piu' piccolo di un glifo non e' un numero stretto: e' una scheggia.
    # Su prova_3 l'etichetta «5 cm» (che di suo e' un «3.5» letto male) si stringeva a
    # 3x3 px, e un riquadro cosi' va a match con qualunque cosa.
    alto = stretto["bottom"] - stretto["top"]
    largo = stretto["right"] - stretto["left"]
    minimo_alto = max(6, int(0.4 * (box["bottom"] - box["top"])))
    minimo_largo = max(4, 4 * len(lette.replace(".", "")))
    if alto < minimo_alto or largo < minimo_largo:
        return {"ok": False, "value_mm": valore, "text": letto["text"],
                "reason": f"il riquadro stretto verrebbe {largo}x{alto} px, troppo poco per "
                          f"«{letto['text']}»: non e' il numero"}
    prima = (box["right"] - box["left"]) * (box["bottom"] - box["top"])
    dopo = (stretto["right"] - stretto["left"]) * (stretto["bottom"] - stretto["top"])
    return {"ok": True, "box": stretto, "value_mm": valore, "text": letto["text"],
            "before": box, "shrink": round(1.0 - (dopo / prima), 3) if prima else 0.0}


@app.get("/api/projects/<project_id>/depth/tighten")
def api_depth_tighten_one(project_id: str):
    """Il riquadro stretto sul numero per una sola immagine, senza salvare niente.

    Serve a farlo vedere prima: si guarda nello zoom, e solo dopo si applica.
    """
    project = _project(project_id)
    nome = (request.args.get("name") or "").strip()
    base = project.dedup_link_dir() or Path(project.source.get("folder") or "")
    # Torna None, non solleva: senza questo controllo un nome fuori cartella arrivava a
    # PIL come percorso vuoto e usciva come «non si legge nessun numero», che e' un'altra
    # cosa e nasconde l'errore vero.
    percorso = _immagine_nella_cartella(base, nome)
    if percorso is None:
        return jsonify({"error": f"immagine non nella cartella: {nome or '(vuoto)'}"}), 400
    try:
        box = {k: int(round(float(request.args.get(k)))) for k in ("top", "left", "bottom", "right")}
    except (TypeError, ValueError):
        return jsonify({"error": "serve un riquadro completo (top/left/bottom/right)"}), 400
    atteso = request.args.get("depth_mm")
    modello = project.step_value("depth_scale").get("depth_box_template") or {}
    esito = _riquadro_stretto(percorso, box, request.args.get("text") or "",
                              float(atteso) if atteso else None,
                              float(modello.get("unit_factor") or 1.0))
    return jsonify(esito)


@app.post("/api/projects/<project_id>/depth/tighten")
def api_depth_tighten(project_id: str):
    """Stringi sul numero i riquadri gia' trovati, su tutta la cartella."""
    project = _project(project_id)
    payload = _payload()
    scope = str(payload.get("scope") or "all")
    elenco = [str(n) for n in (payload.get("names") or [])]
    righe, stage = _depth_module_rows(project)
    if stage is None:
        return jsonify({"error": "la depth non e' ancora stata calcolata"}), 404
    return jsonify({"job_id": _start_job(_run_depth_tighten, project_id, scope, elenco)})


def _run_depth_tighten(job_id: str, project_id: str, scope: str,
                       elenco: Sequence[str] = ()) -> None:
    """Il restringimento sul numero, immagine per immagine, in parallelo.

    Ogni riquadro e' il suo: non c'e' un riquadro di cartella da propagare, si tratta di
    rifilare quello che ogni immagine ha gia'. Per questo il valore atteso e' quello di
    quella riga, e chi non lo conferma resta com'e'.
    """
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    try:
        project = _project(project_id)
        base = project.dedup_link_dir() or Path(project.source.get("folder") or "")
        valore_step = project.step_value("depth_scale")
        letture_note = dict(valore_step.get("depth_box_reads") or {})
        # Le correzioni a mano sono il dato piu' forte che c'e': se lei ha scritto 30 mm,
        # il riquadro che rilegge 30 mm sta sul numero giusto, comunque fosse scritto prima.
        correzioni = valore_step.get("depth_corrections") or {}
        modello = valore_step.get("depth_box_template") or {}
        fattore = float(modello.get("unit_factor") or 1.0)

        righe, _stage = _depth_module_rows(project)
        per_nome = {r["name"]: r for r in righe}
        for nome, lettura in letture_note.items():
            riga = per_nome.setdefault(nome, {"name": nome, "mode": "direct_label"})
            riga = {**riga, "box": lettura.get("box"), "depth_mm": lettura.get("depth_mm"),
                    "ocr_text": lettura.get("ocr_text", "")}
            per_nome[nome] = riga

        # Anche le immagini che il modulo non ha mai guardato: esistono solo come segnaposto
        # nell'elenco, ma se lei ci ha scritto una depth sono a tutti gli effetti immagini
        # della cartella, e un riquadro attorno al numero lo devono avere.
        for nome in project.dedup_names():
            if nome in per_nome:
                continue
            corretta = (correzioni.get(nome) or {}).get("depth_mm")
            if corretta is None:
                continue
            per_nome[nome] = {"name": nome, "mode": "", "box": None, "ocr_text": "",
                              "depth_mm": corretta}

        if scope == "names":
            scelti = [per_nome[n] for n in elenco if n in per_nome]
        elif scope == "run":
            scelti = [r for r in righe if r.get("box")]
        else:
            scelti = list(per_nome.values())
        # Qui non si guarda il metodo. Propagare un riquadro ha senso solo dove la label
        # non si sposta; stringerlo no: si lavora sul riquadro che quell'immagine ha gia',
        # dove si trova. Su prova_3 la depth viene "dalla scala" e i riquadri contengono
        # comunque `3.5 cm` - 92x25 px di cui il numero e' meno di un terzo.
        lavoro = [r for r in scelti if r.get("box")]
        # Le immagini che un riquadro non ce l'hanno proprio: il modulo non le ha esaminate,
        # ma la depth gliel'ha data lei a mano. Se la cartella un posto del numero ce l'ha,
        # tocca anche a loro - restavano grigie, senza riquadro e senza zoom, in mezzo alle
        # altre marcate come corrette.
        senza_riquadro = [r for r in scelti
                          if not r.get("box") and r.get("depth_mm") is not None]
        if not lavoro:
            raise ValueError("nessun riquadro da stringere: la depth non e' ancora stata trovata")

        _job_update(job_id, stage=f"stringo il riquadro su {len(lavoro)} immagini",
                    total=len(lavoro), done=0)

        fatte = 0

        def stringi(riga: Dict) -> Tuple[str, Dict]:
            # L'unita' si ricava dalla riga stessa: il numero scritto e i millimetri che
            # l'app gli ha attribuito. Un `4.3` che vale 43 mm dice cm senza scriverlo.
            scritto = re.search(r"\d+(?:[.,]\d+)?", str(riga.get("ocr_text") or ""))
            numero = float(scritto.group(0).replace(",", ".")) if scritto else None
            fattore_riga = _fattore_unita(riga.get("ocr_text"), riga.get("depth_mm"), numero)
            if numero is None and not str(riga.get("ocr_text") or ""):
                fattore_riga = fattore
            corretta = (correzioni.get(riga["name"]) or {}).get("depth_mm")
            atteso = corretta if corretta is not None else riga.get("depth_mm")
            return riga["name"], _riquadro_stretto(
                base / riga["name"], riga["box"], str(riga.get("ocr_text") or ""),
                atteso, fattore_riga)

        strette: Dict[str, Dict] = {}
        invariate: List[Dict] = []
        originali = {r["name"]: r.get("box") for r in lavoro}
        with ThreadPoolExecutor(max_workers=6) as pool:
            for nome, esito in pool.map(stringi, lavoro):
                fatte += 1
                if fatte % 10 == 0 or fatte == len(lavoro):
                    _job_update(job_id, done=fatte)
                if not esito.get("ok"):
                    # Cosa ha letto il riquadro stretto si dice sempre: spesso e' *meglio*
                    # di quello che c'era. Su prova_3 dove stava scritto «30cm» il riquadro
                    # stretto legge «3.0», cioe' ritrova il punto decimale che la parola
                    # larga aveva perso. Non si applica da soli - si mette sotto gli occhi.
                    invariate.append({"name": nome, "reason": esito.get("reason", ""),
                                      "text": esito.get("text", ""),
                                      "value_mm": esito.get("value_mm")})
                    continue
                # La depth resta quella di prima. Il numero e' lo stesso - lo abbiamo
                # appena verificato cifra per cifra - e riscriverlo con l'unita' ricavata
                # qui potrebbe cambiare un valore gia' giusto. Si stringe il riquadro.
                riga = per_nome.get(nome) or {}
                strette[nome] = {"box": esito["box"],
                                 "depth_mm": riga.get("depth_mm"),
                                 "ocr_text": riga.get("ocr_text") or esito.get("text", "")}

        # Dove non si rilegge, il riquadro non si lascia largo: si mette quello di cartella.
        # L'etichetta non si sposta dentro una cartella - e' la stessa ipotesi su cui la
        # propagazione gia' si regge - quindi l'inviluppo dei riquadri stretti e' il posto
        # dove il numero sta in tutte le altre immagini. Nessuna rilettura, nessun valore
        # toccato: solo il riquadro, portato dove il numero e' senza le lettere intorno.
        # Un riquadro che si e' *spostato* rispetto ai suoi non e' il numero visto meglio.
        # Dentro una cartella l'etichetta sta ferma: i riquadri stretti possono essere piu'
        # lunghi o piu' corti - le cifre non sono sempre le stesse - ma non slittano. Su
        # prova_4 sei fotogrammi si erano agganciati un carattere piu' in la', perdendo il
        # `1` di `15.0` e prendendosi il triangolino di marcatura che segue l'etichetta.
        # Chi e' slittato torna fra quelli da sistemare col riquadro di cartella.
        slittate = _riquadri_slittati({n: v["box"] for n, v in strette.items()})
        for nome in slittate:
            strette.pop(nome, None)
            invariate.append({
                "name": nome, "text": "",
                "reason": "il riquadro stretto si era spostato rispetto agli altri della "
                          "cartella: e' finito su un altro pezzo di etichetta",
            })

        # Il riquadro prestato si calcola **dentro** al riquadro largo di ogni immagine:
        # i riquadri stretti che ci cascano dentro sono quelli della stessa etichetta. Cosi'
        # una cartella con due interfacce - su `prova` le immagini `_trans` hanno la riga
        # della depth 22 px piu' in basso - ne ha due, senza doverle raggruppare a mano; e
        # un'immagine la cui etichetta sta altrove non ne riceve nessuno.
        tutti_stretti = [v["box"] for v in strette.values()]
        globale = _inviluppo_riquadri(tutti_stretti)
        ripiegate: List[str] = []
        for voce in invariate:
            largo = originali.get(voce["name"])
            if not largo:
                continue
            vicini = [b for b in tutti_stretti if _contiene(largo, b)]
            prestato = _inviluppo_riquadri(vicini)
            if prestato is None and globale:
                prestato = globale
            # Stesso metro del restringimento: il riquadro prestato puo' sporgere a destra
            # (il riquadro largo spesso tagliava l'ultima cifra) ma non puo' finire altrove.
            if prestato is not None and not _vicino_al_riquadro(largo, prestato):
                prestato = None
            if prestato is None:
                continue
            strette[voce["name"]] = {"box": dict(prestato), "from_folder_box": True}
            ripiegate.append(voce["name"])
            voce["folder_box"] = True

        # Il riquadro di cartella sulle immagini senza: qui pero' si rilegge, perche' non
        # c'e' un riquadro vecchio a garantire che l'etichetta sia li'. Se ci si legge un
        # numero, il posto e' quello; il valore resta il suo, che e' scritto a mano.
        adottate: List[str] = []
        for riga in senza_riquadro:
            if not globale:
                invariate.append({
                    "name": riga["name"], "text": "",
                    "reason": "senza riquadro, e la cartella non ne ha uno da prestare",
                })
                continue
            # Il riquadro di cartella cosi' com'e', senza rileggere. Rileggere qui vorrebbe
            # dire allargare la banda per cercare, e cercando si trova: su NF_40 usciva un
            # «200» dalla `R:` dell'etichetta accanto, con tanto di riquadro sbagliato. Il
            # posto del numero lo sa gia' la cartella, e il valore lo ha scritto lei.
            strette[riga["name"]] = {"box": dict(globale), "from_folder_box": True}
            adottate.append(riga["name"])

        def salva(_project: Project, value: Dict) -> Dict:
            precedenti = dict(value.get("depth_box_reads") or {})
            for nome, lettura in strette.items():
                riga = per_nome.get(nome) or {}
                base_lettura = precedenti.get(nome) or {}
                fusa = {**base_lettura, **lettura}
                if fusa.get("depth_mm") is None:
                    fusa["depth_mm"] = riga.get("depth_mm")
                fusa.setdefault("ocr_text", riga.get("ocr_text") or "")
                precedenti[nome] = fusa
            value["depth_box_reads"] = precedenti
            value["depth_box_tightened"] = {
                "tightened": len(strette) - len(ripiegate), "targets": len(lavoro),
                "folder_box": len(ripiegate) + len(adottate), "box": globale,
                "adopted": len(adottate),
                "unchanged": invariate[:40], "scope": scope,
                "at": datetime.now().isoformat(timespec="seconds"),
            }
            return value

        _write_step(project_id, "depth_scale", salva, status="corrected", source="user")
        _job_update(job_id, status="done", stage="fatto",
                    result={"tightened": len(strette) - len(ripiegate) - len(adottate),
                            "targets": len(lavoro) + len(senza_riquadro),
                            "folder_box": len(ripiegate) + len(adottate),
                            "adopted": len(adottate), "unchanged": invariate})
    except Exception as error:  # noqa: BLE001
        _job_update(job_id, status="error", stage="errore", error=str(error))


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


@app.post("/api/projects/<project_id>/depth/box/save")
def api_depth_box_save(project_id: str):
    """Salva subito il riquadro di *questa* immagine, senza propagarlo.

    Le depth dedotte dalla scala possono comunque avere il numero scritto nell'interfaccia,
    ma quel numero non sta nella stessa posizione in tutti i frame. Qui il box resta quindi
    locale all'immagine; il comando esplicito `/depth/box` continua a essere quello che
    propaga una label stabile alla cartella.
    """
    project = _project(project_id)
    payload = _payload()
    nome = str(payload.get("name") or "").strip()
    if nome not in set(project.dedup_names()):
        return jsonify({"error": "immagine non nella cartella"}), 400
    grezzo = payload.get("box") or {}
    try:
        box = {k: int(round(float(grezzo[k]))) for k in ("top", "left", "bottom", "right")}
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "serve un riquadro completo (top/left/bottom/right)"}), 400
    if box["right"] - box["left"] < 3 or box["bottom"] - box["top"] < 3:
        return jsonify({"error": "il riquadro e' troppo piccolo"}), 400
    try:
        valore_inviato = payload.get("depth_mm")
        depth_inviata = float(valore_inviato) if valore_inviato is not None else None
    except (TypeError, ValueError):
        return jsonify({"error": "la depth deve essere un numero"}), 400
    righe, _stage = _depth_module_rows(project)
    riga = next((row for row in righe if row.get("name") == nome), {})
    valore_corrente = riga.get("depth_mm")

    def salva(_project: Project, value: Dict) -> Dict:
        letture = dict(value.get("depth_box_reads") or {})
        prima = dict(letture.get(nome) or {})
        letture[nome] = {
            **prima,
            "box": box,
            # Il riquadro non deve inventare un valore nuovo: conserva quello dedotto dalla
            # scala finche' l'utente non lo corregge nel campo numerico.
            "depth_mm": depth_inviata if depth_inviata is not None else (
                prima.get("depth_mm") if prima.get("depth_mm") is not None else valore_corrente),
            "ocr_text": prima.get("ocr_text") or riga.get("ocr_text") or "",
            "source": "user_box",
            "at": datetime.now().isoformat(timespec="seconds"),
        }
        value["depth_box_reads"] = letture
        return value

    value = _write_step(project_id, "depth_scale", salva, status="corrected", source="user")
    return jsonify({"saved": True, "name": nome,
                    "box": (value.get("depth_box_reads") or {}).get(nome, {}).get("box")})


def _leggi_riquadro_su(base: Path, box: Dict, fattore: float, nomi: Sequence[str],
                       progress=None) -> Tuple[Dict[str, Dict], List[str]]:  # noqa: ANN001
    """La rilettura nel riquadro su un elenco di immagini, in parallelo. 0.4 s l'una."""
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    letture: Dict[str, Dict] = {}
    falliti: List[str] = []
    fatte = 0

    def leggi(n: str) -> Tuple[str, Optional[Dict]]:
        return n, _rileggi_nel_riquadro(
            base / n, box,
            accettabile=lambda v: _depth_credibile(v * fattore),
        )

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


def _depth_credibile(valore: Optional[float]) -> bool:
    """Una profondita' di lavoro: fra 1 e 40 cm.

    Niente regola sui passi tondi: l'Esaote Nine scrive `*D 46 *mm`, la profondita' si muove
    di continuo e i valori tondi sono solo i piu' frequenti, non gli unici.
    """
    try:
        misura = float(valore)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return 10.0 <= misura <= 400.0


def _campione_sparso(nomi: Sequence[str], quante: int) -> List[str]:
    """Nomi presi a distanza regolare: una cartella e' fatta di acquisizioni in fila."""
    elenco = list(nomi)
    if len(elenco) <= quante:
        return elenco
    passo = len(elenco) / float(quante)
    return [elenco[int(indice * passo)] for indice in range(quante)]


def _riferimento_che_regge(
    base: Path, righe: Sequence[Dict], nomi: Sequence[str], prove: int = 8,
) -> Optional[Tuple[str, Dict, float, float]]:
    """Quale etichetta propagare: quella che si legge davvero **anche sulle altre immagini**.

    Prendere il riquadro piu' stretto fra quelli del modulo era una scelta al buio. Su
    Esaote Nine il modulo aveva messo dieci riquadri su due bottoni dell'interfaccia
    (`*Image`), il piu' stretto era uno di quelli, e propagandolo la cartella intera leggeva
    3, 28, 293, 3293. Qui ogni candidato si prova su un pugno di immagini sparse e si tiene
    quello che ne fa uscire profondita' credibili; se non regge nessuno non si propaga
    niente, e restano le righe del modulo - poche e da correggere, ma vere.
    """
    utili = [
        r for r in righe
        if r.get("mode") in DEPTH_BOX_MODES and r.get("box")
        and (r["box"]["right"] - r["box"]["left"]) >= 8
        and (r["box"]["bottom"] - r["box"]["top"]) >= 6
    ]
    if not utili:
        return None

    def area(riga: Dict) -> int:
        b = riga["box"]
        return (b["right"] - b["left"]) * (b["bottom"] - b["top"])

    # Immagini della stessa interfaccia danno lo stesso riquadro: si prova una volta sola.
    visti: set = set()
    candidati: List[Dict] = []
    for riga in sorted(utili, key=area):
        b = riga["box"]
        chiave = (b["left"] // 8, b["top"] // 8, b["right"] // 8, b["bottom"] // 8)
        if chiave in visti:
            continue
        visti.add(chiave)
        candidati.append(riga)
        if len(candidati) >= 6:
            break

    campione = _campione_sparso(nomi, prove)
    migliore: Optional[Tuple[float, int, str, Dict, float]] = None
    for riga in candidati:
        letto = _rileggi_nel_riquadro(base / riga["name"], riga["box"])
        if letto is None:
            continue
        fattore = _fattore_unita(riga.get("ocr_text"), riga.get("depth_mm"), letto["value"])
        valori: List[float] = []
        for nome in campione:
            esito = _rileggi_nel_riquadro(base / nome, riga["box"])
            if esito is not None:
                valori.append(round(esito["value"] * fattore, 2))
        if len(valori) < 3:
            continue
        quota = sum(1 for v in valori if _depth_credibile(v)) / float(len(valori))
        if quota < 0.6:
            continue
        voto = (quota, -area(riga), riga["name"], riga["box"], fattore)
        if migliore is None or voto[:2] > migliore[:2]:
            migliore = voto
    if migliore is None:
        return None
    return migliore[2], migliore[3], migliore[4], migliore[0]


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


# Lo stesso ritaglio con due nomi. Nato per la depth, ma con la finestra esplicita e' un
# ritaglio ingrandito di un'immagine qualunque: e' quello che serve alla lente, che di depth
# non sa niente.
@app.get("/api/projects/<project_id>/crop")
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
    percorso = _immagine_nella_cartella(base, nome)
    if percorso is None:
        return jsonify({"error": "immagine non disponibile"}), 404

    def _intero_iniziale(chiave: str) -> Optional[int]:
        try:
            return int(float(request.args[chiave]))
        except (KeyError, TypeError, ValueError):
            return None

    finestra_chiesta = [_intero_iniziale(k) for k in ("x0", "y0", "x1", "y1")]
    esplicita = all(v is not None for v in finestra_chiesta)

    letture = project.step_value("depth_scale").get("depth_box_reads") or {}
    box = (letture.get(nome) or {}).get("box")
    if box is None:
        righe, _stage = _depth_module_rows(project)
        box = next((r["box"] for r in righe if r["name"] == nome), None)
    if not box and not esplicita:
        # Senza riquadro non si sa cosa ritagliare — a meno che la finestra arrivi dal client,
        # che e' il caso dello studio della scala: li' il riquadro della depth non c'entra.
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
        if esplicita:
            x0, y0, x1, y1 = finestra_chiesta
        else:
            x0, y0 = box["left"] - 60, box["top"] - 7
            x1, y1 = box["right"] + 120, box["bottom"] + 7
        x0 = max(0, min(int(x0), immagine.size[0] - 2))
        y0 = max(0, min(int(y0), immagine.size[1] - 2))
        x1 = max(x0 + 2, min(int(x1), immagine.size[0]))
        y1 = max(y0 + 2, min(int(y1), immagine.size[1]))
        ritaglio = immagine.crop((x0, y0, x1, y1))
        ritaglio = ritaglio.resize((ritaglio.width * zoom, ritaglio.height * zoom), Image.NEAREST)
        if not grezzo and box:
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


class _Cronometro:
    """Tiene vivo l'avanzamento di un lavoro che non sa dire a che punto e'.

    Il modulo depth e' un sottoprocesso solo: scrive i suoi CSV alla fine e nel frattempo non
    dice niente. Dichiarare un totale che nessuno riempie fa sembrare tutto bloccato — «0/36»
    per cinque minuti. Meglio nessun contatore e il tempo che passa, che almeno si muove.
    """

    def __init__(self, job_id: str, testo: str) -> None:
        self.job_id = job_id
        self.testo = testo
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._gira, daemon=True)

    def _gira(self) -> None:
        partenza = time.monotonic()
        while not self._stop.wait(2.0):
            passati = int(time.monotonic() - partenza)
            quanto = f"{passati // 60}m {passati % 60:02d}s" if passati >= 60 else f"{passati}s"
            _job_update(self.job_id, stage=f"{self.testo} — {quanto}", total=0, done=0)

    def __enter__(self) -> "_Cronometro":
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:  # noqa: ANN002
        self._stop.set()


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
        testo = (f"il modulo cerca la depth su {quante} immagini "
                 f"(non riporta avanzamento, solo il tempo)")
        _job_update(job_id, stage=testo, total=0, done=0)
        with _Cronometro(job_id, testo):
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


@app.post("/api/projects/<project_id>/depth/reviewed")
def api_depth_reviewed(project_id: str):
    """«Questo numero e' giusto»: una conferma per immagine, non una correzione.

    Da non confondere con `/depth/confirm`, che conferma la depth di tutta la cartella e
    manda avanti lo step. Qui si segna una immagine per volta, scorrendole.

    Scorrendo la cartella la maggior parte delle immagini e' gia' giusta, e passare oltre
    non lasciava traccia: al giro dopo non si sapeva piu' quali erano state guardate.
    Segnarle come «corrette» sarebbe una bugia - non e' stato corretto niente, e chi legge
    il progetto crederebbe che il modulo avesse sbagliato quattrocento volte.

    La conferma porta con se' il valore di allora: se la depth cambia, decade.
    """
    progetto = _project(project_id)
    payload = _payload()
    nomi = [str(n).strip() for n in (payload.get("names") or []) if str(n).strip()]
    if not nomi:
        uno = (payload.get("name") or "").strip()
        nomi = [uno] if uno else []
    if not nomi:
        return jsonify({"error": "manca l'immagine"}), 400
    # Lo stato dello step resta quello di adesso: una conferma per immagine non lo promuove
    # ne' lo retrocede, e riscriverlo cancellerebbe il «confermata» della cartella.
    stato_step = progetto.steps.get("depth_scale") or {}

    def mutate(_p: Project, value: Dict) -> Dict:
        conferme = dict(value.get("depth_reviewed") or {})
        if payload.get("reset"):
            for nome in nomi:
                conferme.pop(nome, None)
        else:
            try:
                misura = float(payload.get("depth_mm"))
            except (TypeError, ValueError):
                raise ValueError("serve la depth confermata")
            adesso = datetime.now().isoformat(timespec="seconds")
            for nome in nomi:
                conferme[nome] = {"depth_mm": misura, "ts": adesso}
        value["depth_reviewed"] = conferme
        return value

    try:
        value = _write_step(project_id, "depth_scale", mutate,
                            status=str(stato_step.get("status") or "proposed"),
                            source=str(stato_step.get("source") or "model"))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    return jsonify({"saved": True, "applied": len(nomi),
                    "reviewed": len(value.get("depth_reviewed") or {})})


@app.post("/api/projects/<project_id>/depth/correct")
def api_depth_correct(project_id: str):
    """Correggi a mano la depth di un'immagine."""
    project = _project(project_id)
    payload = _payload()
    # Una o molte: `names` serve per le immagini che il modulo non ha esaminato, dove la
    # depth e' la stessa e riscriverla una per una sarebbe solo fatica.
    nomi = [str(n).strip() for n in (payload.get("names") or []) if str(n).strip()]
    if not nomi:
        uno = (payload.get("name") or "").strip()
        nomi = [uno] if uno else []
    if not nomi:
        return jsonify({"error": "manca l'immagine"}), 400

    def mutate(project: Project, value: Dict) -> Dict:
        correzioni = dict(value.get("depth_corrections") or {})
        if payload.get("reset"):
            for name in nomi:
                correzioni.pop(name, None)
        else:
            try:
                valore = float(payload.get("depth_mm"))
            except (TypeError, ValueError):
                raise ValueError("serve la depth in millimetri")
            adesso = datetime.now().isoformat(timespec="seconds")
            for name in nomi:
                correzioni[name] = {
                    "depth_mm": valore,
                    "note": payload.get("note", ""),
                    "source": "user",
                    "ts": adesso,
                }
        value["depth_corrections"] = correzioni
        return value

    try:
        value = _write_step(project_id, "depth_scale", mutate, status="corrected", source="user")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    return jsonify({"saved": True, "applied": len(nomi),
                    "corrections": len(value.get("depth_corrections") or {})})


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


def _immagine_nella_cartella(folder: Path, name: str) -> Optional[Path]:
    """Il file dentro alla cartella, o None se il nome prova a uscirne.

    Il controllo va fatto sul **nome**, non sul percorso risolto: lo specchio di lavoro e' fatto
    di symlink alle immagini originali, e `resolve()` porta fuori dallo specchio per definizione.
    Confrontando i percorsi risolti ogni anteprima di una cartella non ruotata rispondeva 403.
    """
    if not name or Path(name).is_absolute():
        return None
    parti = Path(name).parts
    if any(pezzo in ("..", "") for pezzo in parti):
        return None
    candidato = folder / name
    return candidato if candidato.is_file() else None


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
    path = _immagine_nella_cartella(folder, name)
    ruota_qui = 0
    if path is None:
        # Un'immagine che questo progetto non ha piu' nello specchio - dopo lo sdoppiamento
        # sono quelle dell'altro piano - si prende comunque dalla cartella d'origine: serve
        # a **guardare** la divisione, e per guardarla bisogna vederle tutte e due le file.
        # Li' pero' sono ancora storte, quindi si raddrizza qui.
        originale = Path(project.source.get("folder") or "")
        path = _immagine_nella_cartella(originale, name) if originale.is_dir() else None
        ruota_qui = project.rotation()
    if path is None:
        return jsonify({"error": "percorso non consentito"}), 403
    width = int(request.args.get("w") or 320)
    box = request.args.get("box")  # "top,left,bottom,right" in original coordinates
    box2 = request.args.get("box2")  # second box, drawn in a different colour (#13 template)
    try:
        from PIL import Image, ImageDraw

        with Image.open(path) as raw:
            image = raw.convert("RGB")
            if ruota_qui:
                # `angle` e' la rotazione oraria che raddrizza: PIL ruota antiorario.
                image = image.rotate(-ruota_qui, expand=True)
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
