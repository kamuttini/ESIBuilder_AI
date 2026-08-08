"""Single source of truth for the `.fss` generation pipeline monitor.

`collect_metrics.py` normalises heterogeneous run artifacts into flat records
tagged with a `step_id` declared here; `build_dashboard.py` renders one dedicated
space per step out of the same registry.

Editing this file is the supported way to retune the monitor:
  * `PIPELINE_STEPS`   - the steps, their status, primary KPI and target
  * `METRICS`          - how each metric is formatted and which direction is good
  * `REVIEW_REASON_TO_STEP` - maps pipeline review reasons onto the step that caused them

Step order and code references follow
`docs/albero_decisionale_pipeline_fss_head_2026-07-08.md`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Metric catalogue
# --------------------------------------------------------------------------
# fmt values:
#   pct01  - stored in 0..1, displayed as a percentage
#   pct100 - stored already as a percentage
#   ratio  - plain number in 0..1, displayed with 4 decimals
#   num    - plain number
#   px     - pixels
#   count  - integer count (never a KPI, only context)

METRICS: Dict[str, Dict[str, Any]] = {
    "accuracy": {"label": "Accuracy", "fmt": "pct01", "higher_better": True},
    "macro_f1": {"label": "Macro-F1", "fmt": "pct01", "higher_better": True},
    "balanced_accuracy": {"label": "Balanced accuracy", "fmt": "pct01", "higher_better": True},
    "f1": {"label": "F1", "fmt": "pct01", "higher_better": True},
    "precision": {"label": "Precision", "fmt": "pct01", "higher_better": True},
    "recall": {"label": "Recall", "fmt": "pct01", "higher_better": True},
    "mean_iou": {"label": "IoU medio", "fmt": "ratio", "higher_better": True},
    "best_val_iou": {"label": "IoU val (best)", "fmt": "ratio", "higher_better": True},
    "group_accuracy": {"label": "Accuracy gruppo", "fmt": "pct01", "higher_better": True},
    "box_agree_rate": {"label": "Accordo box", "fmt": "pct01", "higher_better": True},
    "envelope_iou_median": {"label": "IoU envelope (mediana)", "fmt": "ratio", "higher_better": True},
    "marker_in_gt_rate": {"label": "Marker dentro GT", "fmt": "pct01", "higher_better": True},
    "marker_found_rate": {"label": "Marker trovato", "fmt": "pct01", "higher_better": True},
    "groups_complete_rate": {"label": "Cartelle con tutti e 4 i gruppi", "fmt": "pct01", "higher_better": True},
    "review_rate": {"label": "Review rate", "fmt": "pct01", "higher_better": False},
    "step_ok_rate": {"label": "Cartelle senza review su questo step", "fmt": "pct01", "higher_better": True},
    "folder_ok_rate": {"label": "Cartelle ok (pipeline intera)", "fmt": "pct01", "higher_better": True},
    "accepted_pct": {"label": "Accepted", "fmt": "pct100", "higher_better": True},
    "reject_pct": {"label": "Reject", "fmt": "pct100", "higher_better": False},
    "strict_ok_pct": {"label": "Strict ok", "fmt": "pct100", "higher_better": True},
    "calib_ok_pct": {"label": "Calibrazione ok", "fmt": "pct100", "higher_better": True},
    "direction_ok_pct": {"label": "Direzione ok", "fmt": "pct100", "higher_better": True},
    "direction_acc": {"label": "Accuracy direzione", "fmt": "pct01", "higher_better": True},
    "x_within_tol_pct": {"label": "X entro tolleranza", "fmt": "pct100", "higher_better": True},
    "y_within_tol_pct": {"label": "Y entro tolleranza", "fmt": "pct100", "higher_better": True},
    "calib_within_tol_pct": {"label": "mm/px entro tolleranza", "fmt": "pct100", "higher_better": True},
    "handoff_score": {"label": "Handoff score", "fmt": "num", "higher_better": True},
    "err_x_px_median": {"label": "Errore X (mediana)", "fmt": "px", "higher_better": False},
    "err_y_zero_px_median": {"label": "Errore Y zero (mediana)", "fmt": "px", "higher_better": False},
    "rel_err_mm_per_px_median": {"label": "Errore rel. mm/px (mediana)", "fmt": "ratio", "higher_better": False},
    "acceptance_ratio": {"label": "Acceptance ratio", "fmt": "pct01", "higher_better": True},
    "support": {"label": "Campioni", "fmt": "count", "higher_better": None},
    "folders": {"label": "Cartelle", "fmt": "count", "higher_better": None},
    "samples": {"label": "Campioni test", "fmt": "count", "higher_better": None},
}

# Status of a block, in the sense used by CLAUDE.md / ROADMAP.md.
STATUS_LEVELS: Dict[str, Dict[str, str]] = {
    "produzione": {"label": "Produzione", "tone": "good", "icon": "●"},
    "operativo": {"label": "Operativo con review", "tone": "warning", "icon": "◐"},
    "in_corso": {"label": "In consolidamento", "tone": "serious", "icon": "◑"},
    "da_fare": {"label": "Da fare", "tone": "critical", "icon": "○"},
    "deterministico": {"label": "Deterministico", "tone": "good", "icon": "■"},
}

VENDOR_ALL = "__ALL__"  # pseudo-vendor used for the aggregate value of a run

# --------------------------------------------------------------------------
# Pipeline steps
# --------------------------------------------------------------------------
# `primary_metric` drives the KPI card, the trend chart and the vendor matrix.
# `target` is the value we consider "production ready" for that metric; set to
# None when no target has been agreed yet.

PIPELINE_STEPS: List[Dict[str, Any]] = [
    {
        "id": "00_end_to_end",
        "order": 0,
        "title": "Pipeline end-to-end",
        "module": "pipeline",
        "fss_lines": [],
        "status": "in_corso",
        "code_ref": "tools/ultrasound/predict_fss_head_from_acquisitions.py",
        "summary": "Salute complessiva di una run `fss_head`: quota di cartelle che arrivano in fondo senza nessuna review reason. E la metrica di uscita della Fase 2 della roadmap (review rate sotto il 3%).",
        "primary_metric": "folder_ok_rate",
        "target": 0.97,
        "review_reasons": [],
    },
    {
        "id": "01_scan",
        "order": 1,
        "title": "Scansione frame",
        "module": "pipeline",
        "fss_lines": [],
        "status": "deterministico",
        "code_ref": "tools/ultrasound/predict_fss_head_from_acquisitions.py:431",
        "summary": "Enumerazione delle sottocartelle e riconoscimento dei frame di acquisizione dal pattern `*_vga|hdmi_WxH`; fallback su tutte le immagini se nessun file matcha.",
        "primary_metric": "step_ok_rate",
        "target": None,
        "review_reasons": [],
    },
    {
        "id": "02_dedup",
        "order": 2,
        "title": "Deduplicazione esatta",
        "module": "pipeline",
        "fss_lines": [],
        "status": "deterministico",
        "code_ref": "tools/ultrasound/predict_fss_head_from_acquisitions.py:460",
        "summary": "Rimozione dei duplicati esatti (dimensione file + SHA-1). Se la dedup svuota la cartella la cartella viene saltata con warning.",
        "primary_metric": "step_ok_rate",
        "target": None,
        "review_reasons": [],
    },
    {
        "id": "03_rotation",
        "order": 3,
        "title": "Rotazione OSD + validazione OCR",
        "module": "orientation",
        "fss_lines": [],
        "status": "operativo",
        "code_ref": "tools/ultrasound/predict_fss_head_from_acquisitions.py:574",
        "summary": "Voto OSD Tesseract su un campione di frame (angolo vincente con >=2 voti e ratio >=0.60), poi validazione OCR sui 4 angoli. Le immagini vengono ruotate in-memory per tutte le inferenze a valle.",
        "primary_metric": "step_ok_rate",
        "target": 0.97,
        "review_reasons": ["rotation_not_reliable"],
    },
    {
        "id": "04_video_meta",
        "order": 4,
        "title": "Metadati video #06-#10",
        "module": "pipeline",
        "fss_lines": ["06", "07", "08", "09", "10"],
        "status": "deterministico",
        "code_ref": "tools/ultrasound/predict_fss_head_from_acquisitions.py:304",
        "summary": "Input (hdmi/vga) e risoluzione ricavati dal nome file, decisi per maggioranza sui frame della cartella. Fallback sulle dimensioni reali dell'immagine.",
        "primary_metric": "step_ok_rate",
        "target": 0.97,
        "review_reasons": ["missing_video_input", "missing_video_size"],
    },
    {
        "id": "05_vendor",
        "order": 5,
        "title": "Vendor (CNN + fallback OCR)",
        "module": "vendor",
        "fss_lines": [],
        "status": "produzione",
        "code_ref": "tools/ultrasound/predict_fss_head_from_acquisitions.py:6948",
        "summary": "CNN sui frame campionati, media delle softmax, top-1 con soglia 0.50. Sotto soglia scatta il fallback OCR sugli alias vendor, poi la policy di bassa confidenza.",
        "primary_metric": "macro_f1",
        "target": 0.95,
        "review_reasons": ["low_vendor_conf", "low_vendor_margin", "missing_vendor", "unseen_vendor"],
    },
    {
        "id": "06_line13",
        "order": 6,
        "title": "Riga #13 RECT_NAME_ECHO",
        "module": "rect",
        "fss_lines": ["13"],
        "status": "operativo",
        "code_ref": "tools/ultrasound/predict_fss_head_from_acquisitions.py:7100",
        "summary": "Rete rect vendor-specifica (mappa `vendor_line13_template_map`) con post-processing in cascata: consenso IoU fra immagini, gate template-matching NCC, dark-border trim. Senza rete si passa al resolver storico dai `.fss` legacy; vendor mai visto = riga vuota.",
        "primary_metric": "mean_iou",
        "target": 0.90,
        "review_reasons": [
            "missing_rect_name_echo",
            "missing_vendor_line13_model",
            "line13_vendor_model_no_prediction",
            "unseen_vendor_line13",
            "invalid_rect_name_echo",
            "low_rect_name_echo_support",
        ],
    },
    {
        "id": "07_probe",
        "order": 7,
        "title": "Probe #03",
        "module": "probe",
        "fss_lines": ["03"],
        "status": "operativo",
        "code_ref": "tools/ultrasound/predict_fss_head_from_acquisitions.py:7386",
        "summary": "CNN con la stessa aggregazione media-softmax del vendor, soglia 0.40. Nessun OCR refine dentro la pipeline (esiste solo come script separato `refine_ultrasound_probe_ocr.py`).",
        "primary_metric": "accuracy",
        "target": 0.90,
        "review_reasons": ["missing_probe_id", "low_probe_conf", "unseen_probe"],
    },
    {
        "id": "08_probetype",
        "order": 8,
        "title": "Riga #04 PROBETYPE (router)",
        "module": "probe",
        "fss_lines": ["04"],
        "status": "operativo",
        "code_ref": "tools/ultrasound/predict_fss_head_from_acquisitions.py:7455",
        "summary": "Router da anagrafica `probe_id -> PROBETYPE`. Il caso biplano `3-4` viene sciolto dalla maggioranza L/T sui crop rect (L->3, T->4), altrimenti resta ambiguo e va in review.",
        "primary_metric": "step_ok_rate",
        "target": 0.97,
        "review_reasons": ["missing_probe_type_line4", "probe_type_line4_unresolved_3_4"],
    },
    {
        "id": "09_line14",
        "order": 9,
        "title": "Riga #14 RECT_NAME_PROBE",
        "module": "resolver",
        "fss_lines": ["14"],
        "status": "operativo",
        "code_ref": "tools/ultrasound/predict_fss_head_from_acquisitions.py:5405",
        "summary": "Resolver storico su 8 granularita (dalla piu specifica: vendor+probe+video fino a global). Probe mai vista: riga vuota, nessun fallback cieco.",
        "primary_metric": "step_ok_rate",
        "target": 0.97,
        "review_reasons": [
            "missing_rect_name_probe",
            "unseen_probe_line14",
            "invalid_rect_name_probe",
            "low_rect_name_probe_support",
        ],
    },
    {
        "id": "10_line11",
        "order": 10,
        "title": "Riga #11 RECT_ECHO",
        "module": "rect",
        "fss_lines": ["11"],
        "status": "produzione",
        "code_ref": "tools/ultrasound/predict_fss_head_from_acquisitions.py:7529",
        "summary": "Rete rect globale o checkpoint vendor-specializzato (solo se in mappa e `vendor_conf >= 0.70`). Box di cartella = mediana componente per componente; possibile override dalla pipeline red-rect.",
        "primary_metric": "mean_iou",
        "target": 0.90,
        "review_reasons": ["missing_rect_echo"],
    },
    {
        "id": "11_sugiu_lr_lt",
        "order": 11,
        "title": "SU/GIU + LR marker + piano L/T",
        "module": "orientation",
        "fss_lines": [],
        "status": "in_corso",
        "code_ref": "tools/ultrasound/predict_fss_head_from_acquisitions.py:1218",
        "summary": "Stadi per-frame sui crop rect: classificatore SU/GIU, template matching LR marker (bundle, multi-scala, template pinnati dalla review) e classificatore L/T con soglia 0.55.",
        "primary_metric": "macro_f1",
        "target": 0.90,
        "review_reasons": [
            "missing_su_giu_predictions",
            "low_sugiu_conf",
            "missing_lr_marker_predictions",
            "missing_lt_predictions",
            "low_lt_conf",
        ],
    },
    {
        "id": "12_line16",
        "order": 12,
        "title": "Riga #16 RECT_ORIENTATION",
        "module": "orientation",
        "fss_lines": ["16"],
        "status": "in_corso",
        "code_ref": "tools/ultrasound/predict_fss_head_from_acquisitions.py:4100",
        "summary": "Envelope (min/max dei box) per ciascun gruppo NF/LR/UD/LRUD dalle detection LR-marker. La riga viene emessa solo se tutti e 4 i gruppi sono presenti e stabili.",
        "primary_metric": "review_rate",
        "target": 0.05,
        "review_reasons": [
            "missing_line16_rect_orientation",
            "lr_marker_envelope_missing_groups",
            "lr_marker_envelope_unstable_groups",
        ],
    },
    {
        "id": "13_rect_depth",
        "order": 13,
        "title": "RECT_DEPTH autonomo + scala",
        "module": "depth",
        "fss_lines": [],
        "status": "in_corso",
        "code_ref": "tools/ultrasound/predict_fss_head_from_acquisitions.py:8345",
        "summary": "Modulo autonomo lanciato come subprocess con un `pipeline_context.json` per cartella. La cartella e ok se `acceptance_ratio >= 0.80`. Comprende la catena scala (ladder classica, modello immagine, heatmap).",
        "primary_metric": "calib_ok_pct",
        "target": 80.0,
        "review_reasons": ["missing_rect_depth_predictions", "rect_depth_autonomous_review"],
    },
    {
        "id": "14_residual",
        "order": 14,
        "title": "Righe residue #01 / #02 / #12",
        "module": "resolver",
        "fss_lines": ["01", "02", "12"],
        "status": "operativo",
        "code_ref": "tools/ultrasound/predict_fss_head_from_acquisitions.py:5175",
        "summary": "#01 costante `4.0`; #02 ID_ECHO dal resolver storico (vendor+probe+video -> vendor); #12 forzata a 4 (symbol).",
        "primary_metric": "step_ok_rate",
        "target": 0.97,
        "review_reasons": ["missing_id_echo", "missing_group_orientation"],
    },
    {
        "id": "15_writer",
        "order": 15,
        "title": "Writer `.fss` + quality gate",
        "module": "fss",
        "fss_lines": ["file"],
        "status": "da_fare",
        "code_ref": "tools/fss/compare_fss.py",
        "summary": "Modulo che assembla il file `.fss` vero dai valori predetti (oggi la pipeline produce solo CSV/preview) e lo valida contro il legacy con `compare_fss.py`. Fase 2 della roadmap.",
        "primary_metric": None,
        "target": None,
        "review_reasons": [],
    },
]

STEP_BY_ID: Dict[str, Dict[str, Any]] = {s["id"]: s for s in PIPELINE_STEPS}

# Review reason -> step id, derived from the declarations above plus a few
# aliases emitted by older runs.
REVIEW_REASON_TO_STEP: Dict[str, str] = {}
for _step in PIPELINE_STEPS:
    for _reason in _step["review_reasons"]:
        REVIEW_REASON_TO_STEP[_reason] = _step["id"]

REVIEW_REASON_TO_STEP.update(
    {
        "osd_unavailable": "03_rotation",
        "osd_no_votes": "03_rotation",
        "osd_low_support": "03_rotation",
        "tesseract_missing": "03_rotation",
        "manual_user_input_on_low_confidence": "05_vendor",
        "lr_marker_review_file": "12_line16",
        "missing_marker_side": "11_sugiu_lr_lt",
        "opposite_sugiu_roi_low_conf": "11_sugiu_lr_lt",
        "invalid_rect": "10_line11",
        "invalid_crop_box": "10_line11",
    }
)

# Canonical vendor spelling; collect_metrics normalises whatever the artifacts use.
VENDOR_CANONICAL: Dict[str, str] = {
    "alpinion": "Alpinion",
    "bk": "BK",
    "bk-hard": "BK",
    "bkhard": "BK",
    "biopsee": "Biopsee",
    "canon": "Canon",
    "esaote": "Esaote",
    "exactvu": "ExactVu",
    "ge": "GE",
    "hitachi": "Hitachi",
    "koelis": "Koelis",
    "mindray": "Mindray",
    "philips": "Philips",
    "siemens": "Siemens",
    "terason": "Terason",
    "toshiba": "Toshiba",
    "vinno": "Vinno",
}


def canonical_vendor(raw: Optional[str]) -> Optional[str]:
    """Normalise a vendor label found in an artifact; None when not a vendor."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text == VENDOR_ALL:
        return None
    key = text.lower().replace(" ", "").replace("_", "-")
    if key in VENDOR_CANONICAL:
        return VENDOR_CANONICAL[key]
    key2 = key.split("-")[0]
    if key2 in VENDOR_CANONICAL:
        return VENDOR_CANONICAL[key2]
    return text


def metric_meta(name: str) -> Dict[str, Any]:
    return METRICS.get(name, {"label": name, "fmt": "num", "higher_better": True})
