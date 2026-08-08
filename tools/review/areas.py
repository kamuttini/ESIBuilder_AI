#!/usr/bin/env python3
"""Taxonomy of the study areas: the single source of truth of the review tool.

Every comment and every correction produced by the UI is tied to one *area* (a pipeline
module) and, when it is not free text, to one *tag* from that area's catalog. The tag is what
makes a comment machine-usable later: free text alone cannot be counted, grouped, or turned
into a training queue, while ``depth.box_includes_extra`` can.

Each area declares:
- ``label`` / ``lines``: how it shows in the UI and which ``.fss`` lines it owns.
- ``module``: the file that would have to change to fix a finding here (so a comment lands on
  a real owner instead of a generic backlog).
- ``scopes``: whether the finding makes sense per folder, per image, or both.
- ``suggestions``: the ready-made comments, each with a tag, a severity default, and whether
  it *asks for* a correction (``needs_correction``) - a wrong value without the right value is
  half a report.
- ``correction``: the fields of a manual correction and their kind, which the UI renders as
  the right widget (``select``, ``number``, ``box`` = drag on the image, ``point`` = click).
- ``export``: which existing queue a correction of this area feeds (see feedback_cli export).

The UI never hardcodes any of this: it reads ``/api/catalog``. Adding a suggestion is a
one-line change here and it appears in the tool.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Severity vocabulary, ordered. ``blocking`` means "do not ship the module like this".
SEVERITIES = ["info", "minor", "major", "blocking"]

# Verdicts: the operator's judgement on the prediction, independent of the comment text.
# ``ok`` entries matter as much as the wrong ones: they are the positive labels of a
# regression set, which is what keeps a fix from silently breaking what already worked.
VERDICTS = ["ok", "wrong", "partial", "uncertain"]

STATUSES = ["open", "ack", "done", "wontfix"]


def _s(
    tag: str,
    text: str,
    *,
    verdict: str = "wrong",
    severity: str = "major",
    needs_correction: bool = False,
    hint: str = "",
) -> Dict[str, object]:
    return {
        "tag": tag,
        "text": text,
        "verdict": verdict,
        "severity": severity,
        "needs_correction": needs_correction,
        "hint": hint,
    }


def _ok(tag: str, text: str = "Corretto") -> Dict[str, object]:
    return _s(tag, text, verdict="ok", severity="info")


AREAS: List[Dict[str, object]] = [
    {
        "id": "dedup",
        "label": "Deduplicazione",
        "lines": [],
        "module": "tools/ultrasound/predict_fss_head_from_acquisitions.py:_deduplicate_exact_images",
        "scopes": ["folder"],
        "informative": True,
        "suggestions": [
            _ok("dedup.ok", "Il conteggio torna"),
            _s("dedup.too_aggressive", "Ha rimosso immagini che non erano duplicate",
               severity="major"),
            _s("dedup.left_duplicates", "Restano duplicati (near-duplicate, non byte-identici)",
               severity="minor"),
            _s("dedup.unexpected_ratio", "Percentuale di rimossi sospetta per questa cartella",
               verdict="uncertain", severity="minor"),
        ],
        "correction": [],
        "export": "",
    },
    {
        "id": "rotazione",
        "label": "Rotazione",
        "lines": [],
        "module": "tools/ultrasound/predict_fss_head_from_acquisitions.py:_estimate_folder_rotation_with_osd",
        "scopes": ["folder", "image"],
        "suggestions": [
            _ok("rotazione.ok"),
            _s("rotazione.wrong_angle", "Angolo sbagliato", needs_correction=True),
            _s("rotazione.not_needed", "Ha ruotato una cartella che era già dritta"),
            _s("rotazione.mixed_folder", "La cartella ha immagini con rotazioni diverse",
               severity="minor"),
        ],
        "correction": [
            {"field": "rotation_deg_clockwise", "kind": "select", "label": "Rotazione oraria",
             "options": ["0", "90", "180", "270"]},
        ],
        "export": "",
    },
    {
        "id": "vendor",
        "label": "Vendor",
        "lines": ["#02"],
        "module": "tools/ultrasound/train_ultrasound_vendor_classifier.py",
        "scopes": ["folder"],
        "suggestions": [
            _ok("vendor.ok"),
            _s("vendor.wrong_class", "Vendor sbagliato", needs_correction=True,
               severity="blocking"),
            _s("vendor.right_low_conf", "Vendor giusto ma confidenza troppo bassa",
               verdict="partial", severity="minor"),
            _s("vendor.ocr_should_resolve", "Il nome è leggibile a schermo: l'OCR doveva risolverlo",
               verdict="partial"),
            _s("vendor.mixed_folder", "La cartella contiene più ecografi diversi",
               verdict="uncertain"),
            _s("vendor.missing_class", "Questo vendor non esiste fra le classi del modello",
               severity="blocking"),
        ],
        "correction": [
            {"field": "vendor", "kind": "select", "label": "Vendor corretto",
             "options_source": "vendor_classes", "allow_custom": True},
        ],
        "export": "vendor_review_csv",
    },
    {
        "id": "probe",
        "label": "Sonda",
        "lines": ["#03", "#04"],
        "module": "tools/ultrasound/train_ultrasound_probe_classifier.py",
        "scopes": ["folder"],
        "suggestions": [
            _ok("probe.ok"),
            _s("probe.wrong_id", "Sonda sbagliata", needs_correction=True, severity="blocking"),
            _s("probe.right_low_conf", "Sonda giusta ma confidenza bassa", verdict="partial",
               severity="minor"),
            _s("probe.unseen_class", "Sonda assente dal training (classe mancante)",
               severity="major"),
            _s("probe.type_wrong", "ID sonda giusto, PROBETYPE #04 sbagliato",
               verdict="partial", needs_correction=True),
            _s("probe.confusable", "Confusa con una sonda molto simile", verdict="uncertain"),
            _s("probe.ocr_refine_needed", "Il modello sonda va rifinito con l'OCR della sigla",
               verdict="partial", severity="minor"),
        ],
        "correction": [
            {"field": "probe_id", "kind": "select", "label": "ID sonda corretto",
             "options_source": "probe_classes", "allow_custom": True},
            {"field": "probe_type", "kind": "text", "label": "PROBETYPE #04", "optional": True},
        ],
        "export": "probe_review_csv",
    },
    {
        "id": "rect",
        "label": "Rettangolo ecografico",
        "lines": ["#11"],
        "module": "tools/ultrasound/train_ultrasound_rect_net.py + tools/ultrasound/rect_red_pipeline.py",
        "scopes": ["folder", "image"],
        "suggestions": [
            _ok("rect.ok"),
            _s("rect.includes_ui", "Il rect ingloba interfaccia o testo", needs_correction=True),
            _s("rect.cuts_sector", "Il rect taglia il settore ecografico", needs_correction=True,
               severity="blocking"),
            _s("rect.too_wide", "Troppo largo", needs_correction=True),
            _s("rect.too_narrow", "Troppo stretto", needs_correction=True),
            _s("rect.red_override_worse", "L'override rosso ha peggiorato la mediana",
               severity="major",
               hint="method=segment_top_red_rect: annota qui se la mediana era migliore"),
            _s("rect.per_image_unstable", "Rect per-immagine instabile nella cartella",
               verdict="partial", severity="minor"),
            _s("rect.fusion_multi_panel", "Immagine fusion/multipannello: rect ambiguo",
               verdict="uncertain"),
        ],
        "correction": [
            {"field": "box", "kind": "box", "label": "Rettangolo corretto (trascina)"},
        ],
        "export": "rect_review_csv",
    },
    {
        "id": "line13",
        "label": "Nome vendor a schermo",
        "lines": ["#13"],
        "module": "tools/ultrasound/predict_fss_head_from_acquisitions.py:_postprocess_line13_box_with_template",
        "scopes": ["folder", "image"],
        "suggestions": [
            _ok("line13.ok"),
            _s("line13.box_offset", "Box spostato rispetto alla scritta", needs_correction=True),
            _s("line13.box_too_big", "Box troppo grande (prende altra interfaccia)",
               needs_correction=True),
            _s("line13.wrong_text_region", "Ha preso una scritta che non è il nome vendor"),
            _s("line13.resolver_fallback_wrong", "Il valore storico del resolver è sbagliato"),
        ],
        "correction": [
            {"field": "box", "kind": "box", "label": "Box corretto (trascina)"},
        ],
        "export": "line13_review_csv",
    },
    {
        "id": "su_giu",
        "label": "Verso su/giù",
        "lines": ["#16"],
        "module": "tools/ultrasound/predict_fss_head_from_acquisitions.py:_predict_su_giu_on_rect_crops",
        "scopes": ["folder", "image"],
        "suggestions": [
            _ok("su_giu.ok"),
            _s("su_giu.inverted", "Verso invertito", needs_correction=True, severity="major"),
            _s("su_giu.mixed_folder_expected", "La cartella ha entrambi i versi: è corretto così",
               verdict="ok", severity="info"),
            _s("su_giu.low_conf_but_right", "Giusto ma con confidenza bassa", verdict="partial",
               severity="minor"),
        ],
        "correction": [
            {"field": "su_giu", "kind": "select", "label": "Verso corretto",
             "options": ["su", "giu"]},
        ],
        "export": "sugiu_review_csv",
    },
    {
        "id": "lr_marker",
        "label": "Marker L/R",
        "lines": ["#16"],
        "module": "tools/orientation/predict_marker_envelopes_batch.py",
        "scopes": ["folder", "image"],
        "suggestions": [
            _ok("lr_marker.ok"),
            _s("lr_marker.wrong_template", "Template sbagliato (look-alike statico dell'interfaccia)",
               severity="major",
               hint="Diventa un candidato per --pinned-templates di questa cartella"),
            _s("lr_marker.marker_absent", "In questa immagine il marker non c'è",
               verdict="uncertain"),
            _s("lr_marker.box_offset", "Box del marker spostato", needs_correction=True),
            _s("lr_marker.side_inverted", "Lato L/R invertito", needs_correction=True,
               severity="major"),
            _s("lr_marker.legacy_single_orientation",
               "Progetto legacy a orientamento singolo (#16 con 4 copie dello stesso box)",
               verdict="uncertain", severity="minor"),
            _s("lr_marker.template_size_changed",
               "Il template cambia dimensione fra immagini della stessa cartella",
               verdict="partial", severity="minor"),
        ],
        "correction": [
            {"field": "lr_label", "kind": "select", "label": "Etichetta corretta",
             "options": ["lr_flipped", "not_lr_flipped"], "optional": True},
            {"field": "box", "kind": "box", "label": "Box del marker vero (trascina)",
             "optional": True},
        ],
        "export": "lr_marker_manual_seeds",
    },
    {
        "id": "lt",
        "label": "Piano L/T",
        "lines": ["#04"],
        "module": "tools/ultrasound/train_ultrasound_lt_classifier.py",
        "scopes": ["folder", "image"],
        "suggestions": [
            _ok("lt.ok"),
            _s("lt.wrong_plane", "Piano sbagliato", needs_correction=True, severity="major"),
            _s("lt.biplane", "Immagine biplana: L/T non applicabile", verdict="uncertain"),
            _s("lt.low_conf_but_right", "Giusto ma con confidenza bassa", verdict="partial",
               severity="minor"),
        ],
        "correction": [
            {"field": "lt", "kind": "select", "label": "Piano corretto", "options": ["L", "T"]},
        ],
        "export": "lt_review_csv",
    },
    {
        "id": "depth",
        "label": "Depth",
        "lines": ["#17"],
        "module": "tools/depth/predict_rect_depth_autonomous.py",
        "scopes": ["folder", "image"],
        "suggestions": [
            _ok("depth.ok"),
            _s("depth.wrong_value", "Valore depth sbagliato", needs_correction=True,
               severity="blocking"),
            _s("depth.box_includes_extra", "Il box ingloba caratteri extra (es. la B di BG)",
               needs_correction=True,
               hint="È il difetto storico del box depth: annotalo sempre con il box giusto"),
            _s("depth.wrong_lane", "Ha letto un numero che non è la depth (Hz, FR, dB, MI)",
               severity="major"),
            _s("depth.scale_instead_of_label", "Ha usato la scala invece dell'etichetta diretta"),
            _s("depth.label_instead_of_scale", "Ha usato l'etichetta diretta invece della scala"),
            _s("depth.missing_but_visible", "Depth visibile ma non riconosciuta",
               needs_correction=True, severity="major"),
            _s("depth.groups_wrong", "Il raggruppamento delle depth della cartella è sbagliato",
               severity="major"),
            _s("depth.ocr_unreadable", "L'immagine è troppo compressa perché l'OCR la legga",
               verdict="uncertain", severity="minor"),
        ],
        "correction": [
            {"field": "depth_mm", "kind": "number", "label": "Depth corretta (mm)", "step": 1},
            {"field": "box", "kind": "box", "label": "Box del valore (trascina)",
             "optional": True},
        ],
        "export": "depth_review_csv",
    },
    {
        "id": "scala",
        "label": "Scala",
        "lines": ["#18", "#19", "#20", "#21"],
        "module": "tools/scale/detect_scale_ladder.py + tools/scale/consolidate_scale_setup.py",
        "scopes": ["folder", "image"],
        "suggestions": [
            _ok("scala.ok"),
            _s("scala.wrong_column", "Colonna del righello sbagliata (righello di un altro pannello)",
               needs_correction=True, severity="major",
               hint="È l'errore genuino noto: fallback a tutto schermo quando la banda è vuota"),
            _s("scala.no_ruler", "In questa immagine non c'è righello", verdict="uncertain"),
            _s("scala.direction_inverted", "Verso della scala invertito", needs_correction=True),
            _s("scala.zero_wrong", "Zero nel punto sbagliato", needs_correction=True),
            _s("scala.far_end_short", "Estremo lontano corto (manca l'ultima tacca)",
               needs_correction=True, severity="minor"),
            _s("scala.calibration_off", "mm_per_px sbagliato", needs_correction=True,
               severity="blocking"),
            _s("scala.ocr_misread", "L'OCR ha letto un numero che non c'è", severity="major"),
            _s("scala.weak_evidence_accepted",
               "Accettata con evidenza debole: non doveva essere accepted", severity="major",
               hint="4 volte su 4 il guadagno è venuto dalla policy di accettazione"),
            _s("scala.fusion_panel", "Immagine fusion/multipannello", verdict="uncertain",
               severity="minor"),
        ],
        "correction": [
            {"field": "x", "kind": "point_x", "label": "Colonna del righello (clic)",
             "optional": True},
            {"field": "y_zero", "kind": "point_y", "label": "Zero (clic)", "optional": True},
            {"field": "y_far", "kind": "point_y", "label": "Estremo lontano (clic)",
             "optional": True},
            {"field": "depth_mm", "kind": "number", "label": "Valore all'estremo (mm)",
             "optional": True,
             "hint": "Con zero + estremo + valore il tool ricava mm_per_px da solo"},
            {"field": "direction", "kind": "select", "label": "Verso", "options": ["su", "giu"],
             "optional": True},
        ],
        "export": "scale_corrections_csv",
    },
    {
        "id": "fss",
        "label": "Testa .fss",
        "lines": ["#01-#21"],
        "module": "tools/fss/compare_fss.py",
        "scopes": ["folder"],
        "suggestions": [
            _ok("fss.ready", "Questa testa la firmerei"),
            _s("fss.line_wrong", "Una riga è sbagliata", needs_correction=True),
            _s("fss.not_compatible", "Non compatibile con il legacy", severity="blocking"),
            _s("fss.missing_line", "Manca una riga che il legacy ha"),
        ],
        "correction": [
            {"field": "line", "kind": "text", "label": "Riga (es. #11)"},
            {"field": "value", "kind": "text", "label": "Valore corretto"},
        ],
        "export": "",
    },
    {
        "id": "orchestratore",
        "label": "Collaborazione fra reti",
        "lines": [],
        "module": "tools/review/orchestrator.py",
        "scopes": ["folder", "image", "run"],
        "suggestions": [
            _s("orch.depth_scala_incoherent", "Depth e scala non si accordano", severity="major"),
            _s("orch.depth_box_outside_rect", "Il box della depth cade fuori dal rect"),
            _s("orch.sugiu_scala_disagree", "Verso su/giù e verso della scala discordano"),
            _s("orch.vendor_probe_incompatible", "Vendor e sonda sono incompatibili"),
            _s("orch.use_vendor_history",
               "Per questo vendor lo storico dice altro: usalo come prior", verdict="uncertain",
               severity="minor"),
            _s("orch.stage_order_problem", "Uno stadio dovrebbe girare prima/dopo un altro"),
            _s("orch.share_evidence",
               "Due aree stanno ricavando la stessa cosa separatamente: condividere l'evidenza",
               verdict="uncertain", severity="minor"),
        ],
        "correction": [],
        "export": "",
    },
    {
        "id": "ui",
        "label": "Questo tool",
        "lines": [],
        "module": "tools/review/",
        "scopes": ["run", "folder", "image"],
        "suggestions": [
            _s("ui.confusing", "Non si capisce cosa sto guardando", verdict="uncertain",
               severity="minor"),
            _s("ui.missing_info", "Qui mi serve un'informazione che non c'è", verdict="partial",
               severity="minor"),
            _s("ui.too_slow", "Troppo lento", severity="minor"),
            _s("ui.bug", "Non funziona", severity="major"),
            _s("ui.idea", "Idea per migliorarlo", verdict="uncertain", severity="info"),
        ],
        "correction": [],
        "export": "",
    },
]

AREA_BY_ID: Dict[str, Dict[str, object]] = {str(a["id"]): a for a in AREAS}

# Areas shown as per-image chips in the grid, in display order. Depth and scala first because
# they are the two the operator reads on *every* image; the rest are the context that makes a
# depth or scala verdict explainable.
IMAGE_AREAS = ["depth", "scala", "su_giu", "lr_marker", "lt", "rect"]

# Areas shown once per folder, in display order.
FOLDER_AREAS = ["dedup", "rotazione", "vendor", "probe", "rect", "line13", "su_giu",
                "lr_marker", "lt", "depth", "scala", "fss", "orchestratore"]


def suggestion(area_id: str, tag: str) -> Optional[Dict[str, object]]:
    area = AREA_BY_ID.get(area_id)
    if not area:
        return None
    for item in area.get("suggestions", []):  # type: ignore[union-attr]
        if str(item.get("tag")) == tag:
            return dict(item)
    return None


def catalog() -> Dict[str, object]:
    """The payload the UI reads once at boot."""
    return {
        "areas": AREAS,
        "area_ids": [str(a["id"]) for a in AREAS],
        "image_areas": IMAGE_AREAS,
        "folder_areas": FOLDER_AREAS,
        "severities": SEVERITIES,
        "verdicts": VERDICTS,
        "statuses": STATUSES,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(catalog(), ensure_ascii=False, indent=2))
