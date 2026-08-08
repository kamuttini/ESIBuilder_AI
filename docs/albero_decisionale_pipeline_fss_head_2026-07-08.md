# Albero decisionale pipeline fss_head — dalla cartella grezza al `.fss`

Aggiornato: 2026-07-08. Ricavato dal codice di `tools/ultrasound/predict_fss_head_from_acquisitions.py` (9594 righe), wrapper `run_pipeline_single_folder_safe.py`. Ogni step riporta il riferimento `file:riga` per saltare al codice.

## Flusso a colpo d'occhio

```
Cartella acquisizioni grezze
  │ 1. Scansione frame (pattern *_vga|hdmi_WxH)
  │ 2. Dedup esatta (size + SHA-1)
  │ 3. Rotazione OSD 0/90/180/270 + validazione OCR ──► review se inaffidabile
  │ 4. Metadati video da filename (#06-#10, majority vote)
  │ 5. Vendor CNN (media softmax) ──► OCR fallback ──► policy
  │ 6. Riga #13 RECT_NAME_ECHO (rete vendor-specifica + post-proc + storico)
  │ 7. Probe CNN #03 ──► policy         8. Router #04 PROBETYPE (+split L/T se 3-4)
  │ 9. Riga #14 RECT_NAME_PROBE (resolver storico, no fallback se probe nuova)
  │ 10. Riga #11 RECT_ECHO (routing vendor, mediana box, red-rect override)
  │ 11. SU/GIU + LR marker + L/T sui crop rect
  │ 12. Riga #16 da envelope LR-marker (4 gruppi NF/LR/UD/LRUD)
  │ 13. RECT_DEPTH autonomo (subprocess, già agganciato)
  │ 14. Scala #18-#21 (subprocess, per depth) ──► review se non accettata
  │ 15. Riga #02 ID_ECHO (resolver storico)   #12 forzata a 4   #01 = 4.0
  ▼
  CSV + preview txt + summary.json  (⚠ NESSUN file .fss viene scritto)
```

Policy trasversale bassa confidenza (`--low-confidence-policy`, default `ask_user`): `ask_user` = prompt interattivo top-k (default 3) + opzione "0) Altro" manuale (registrata su encoding struct); `error` = interrompe al primo campo dubbio; `review` = tiene la predizione e marca la cartella. Status finale cartella: `ok` se nessuna review reason, altrimenti `review` (riga 8513).

---

## 1. Scansione della cartella (`_collect_acquisition_images`, :431)

- Enumerazione: sottocartelle dirette di `--dataset-root`, ordinate; filtro `--exclude-folder-regex`, limite `--max-folders`.
- Dentro la cartella: `rglob` ricorsivo, estensioni immagine (`.png .jpg .jpeg .bmp .tif .tiff .webp .gif`).
- **Decisione**: sono frame di acquisizione i file il cui nome matcha `^.+_(vga|hdmi)_(WxH)...$` (:123). Se **nessuno** matcha → fallback: prende **tutte** le immagini della cartella.
- Nota: `image_samples/` NON è esclusa in questa fase (esclusa solo nella ricerca template LR-marker, :1542).

## 2. Deduplicazione (`_deduplicate_exact_images`, :460)

- Duplicati **esatti**: bucket per dimensione file, poi SHA-1 completo solo sulle collisioni. Chiave = (size, hash).
- Se la dedup svuota la cartella → warning e skip cartella. Metriche: `images_total_raw`, `images_total`, `images_duplicates_removed`.

## 3. Rotazione (`_estimate_folder_rotation_with_osd` :574, validazione OCR :955)

- Campione uniforme di max `--rotation-max-samples` (24) frame → `tesseract --psm 0` (OSD) su ciascuno → voto angolo {0,90,180,270}.
- **Decisione** (:655): angolo vincente se `voti >= 2` (`--rotation-min-votes`) **e** `ratio >= 0.60` (`--rotation-min-ratio`). Tie-break: preferisci 0°, poi angolo minore.
- Esiti inaffidabili: `osd_unavailable` (tesseract assente), `osd_no_votes`, `osd_low_support` → rotazione 0 e review reason `rotation_not_reliable` (:8476).
- **Validazione OCR** (solo se candidato ≠ 0°): punteggio OCR sui 4 angoli; override solo se `parole >= 2` e `delta punteggio >= 12.0` (:1063). Sorgenti: `_ocr_override/_ocr_confirm/_ocr_weak_delta/...`
- Applicazione: le immagini vengono ruotate in-memory per **tutte** le inferenze a valle; i box predetti vengono riportati alle coordinate originali (`_rotate_box_back_to_original_coords`, :1094, applicato in :4578).

## 4. Metadati video #06–#10 (`_parse_capture_metadata_from_filename` :304, majority :334)

- Da ogni filename: token `hdmi|vga` + token successivo `WxH` (fallback: `WxH` ovunque nel nome).
- **Decisione**: histogram su tutti i frame → vince la maggioranza `(input, W, H)`; pareggio → preferisci `hdmi`, poi area maggiore.
- Mapping (:8535): #06 = input (codice: **hdmi→0, vga→1**), #07/#08 = #09/#10 = WxH. Se mancante → fallback dimensioni reali immagine; review reasons `missing_video_input`/`missing_video_size`.

## 5. Vendor (:6948; aggregazione `_predict_mean_probs` :4490; OCR `_resolve_vendor_with_ocr` :808)

- Campione `--sample-per-folder` (80) frame → CNN → **media delle softmax** per-immagine → top-1.
- `vendor_conf` = prob media top-1. Il margine top1−top2 è registrato ma **non** è un gate decisionale.
- **Fail-fast**: probabilità vuote → `RuntimeError`, la run si ferma (:6956).
- **Decisione** (soglia `--vendor-min-confidence` = 0.50):
  - conf ≥ 0.50 → `cnn_classifier`, ok.
  - conf < 0.50 → **OCR fallback** (max 6 frame, alias vendor pesati): accettato se `score >= 2.0` e `delta >= 0.7` e `hits >= 2` (:922) → `ocr_fallback`, conf portata a soglia+0.02.
  - OCR non conclusivo → policy (`ask_user`/`error`/`review`). Review reason: `low_vendor_conf`.

## 6. Riga #13 RECT_NAME_ECHO (:7100 — calcolata PRIMA della probe)

- Percorso di default: **rete vendor-specifica** via `--line13-vendor-map`. Se manca il checkpoint del vendor → nessuna predizione, source `vendor_model_missing`, review forzata (:7185).
- Post-processing del box in cascata (ognuno disattivabile):
  1. consenso IoU tra immagini (thr 0.35, min-keep 3, :4640)
  2. gate template-matching NCC (min-score 0.60, min-std 60, :4822)
  3. dark-border trim (thr 70, max trim 35%, :5055)
- Senza rete (`--disable-line13-vendor-models`): **resolver storico** (`RectNameEchoResolver` :5298) dai `.fss` legacy, chiavi più specifiche prima: `vendor+video → vendor → video → global`. **Vendor mai visto → riga vuota, NESSUN fallback cieco** (:7161).
- La coda dopo le coordinate è sempre forzata disabilitata: `top|left|bottom|right|0|0:...|0|` (:400): ci si fida solo del rettangolo.
- **Decisione finale**: se vuota o `support < --line13-min-support` (default 0.00) → policy. Review reasons: `unseen_vendor_line13`, `missing_vendor_line13_model`, `line13_vendor_model_no_prediction`, ecc.

## 7. Probe #03 (:7386)

- Stessa aggregazione media-softmax sugli stessi frame campionati → top-1.
- **Decisione** (soglia `--probe-min-confidence` = 0.40): sotto soglia → policy. **Niente OCR refine qui** (esiste solo come script separato `refine_ultrasound_probe_ocr.py`). Probs vuote → probe lasciata vuota (nessun fail-fast, a differenza del vendor). Review: `missing_probe_id`, `low_probe_conf`.

## 8. Riga #04 PROBETYPE (router :7455)

- `ProbeTypeRouter.resolve(probe_id)` da anagrafica: 1 (lineare) e 2 (convex) univoci.
- **Decisione biplano**: se il router risponde `"3-4"` → la maggioranza L/T sui crop rect decide: L→3, T→4; altrimenti resta `3-4` + review reason `probe_type_line4_unresolved_3_4` (:8189).

## 9. Riga #14 RECT_NAME_PROBE (`RectNameProbeResolver` :5405, uso :7463)

- Resolver storico su 8 granularità, più specifica prima: `vendor+probe+video → vendor+probe → probe+video → probe → vendor+video → vendor → video → global`.
- **Probe mai vista → riga vuota, nessun fallback** (:7471). Sotto `--line14-min-support` (0.00) → policy. Review: `missing_rect_name_probe`, `unseen_probe_line14`, `invalid_rect_name_probe`, `low_rect_name_probe_support`.

## 10. Riga #11 RECT_ECHO (:7529 + red-rect :8204)

- **Routing vendor** (:7533): checkpoint specializzato usato solo se esiste nella mappa (default: solo BK) **e** `vendor_conf >= 0.70` (`--rect-vendor-min-confidence`); altrimenti modello globale. `line_11_source`: `vendor_specialized` / `global` / `global_low_vendor_conf`.
- Inferenza su **tutte** le immagini uniche (il flag `--rect-sample-per-folder` è ignorato, :6472). Box cartella = **mediana componente per componente** dei box per-immagine (`_median_box` :4590).
- **Override red-rect** (:8271): la pipeline `compute_rect_red_pipeline` (segmenti + orientamenti su/giù) può sostituire la mediana → `line_11_method="segment_top_red_rect"`; altrimenti `median_rect_fallback`. Nota: non esiste una regola Mindray in questo script.

## 11. Stadi per-frame ausiliari

- **SU/GIU** (:1218): classificatore sui crop rect, maggioranza + prob medie.
- **LR marker** (:7659): metodo default `bundle` (template matching, `--lr-marker-min-match-score` 0.62, soglie di espansione ricerca, vendor esclusi: Biopsee).
- **L/T** (:4224): classificatore sui crop rect, soglia `--lt-min-confidence` 0.55 → review reason `low_lt_conf`.

## 12. Riga #16 RECT_ORIENTATION (`_build_line16_rect_orientation_from_lr_marker_rows` :4100)

- Envelope (min/max dei box) per ciascun gruppo orientamento {NF, LR, UD, LRUD} dalle detection LR-marker.
- **Decisione**: la riga viene emessa solo se **tutti e 4 i gruppi** sono presenti e stabili, con coda "pending" da confermare; altrimenti vuota con source `lr_marker_envelope_missing_groups` / `_unstable_groups`.

## 13. RECT_DEPTH autonomo (:1359, chiamata :8345)

- **Già agganciato** come subprocess: la pipeline scrive un `pipeline_context.json` (vendor, probe, rect #11, dimensioni video, rotazione) e lancia `tools/depth/predict_rect_depth_autonomous.py` per cartella.
- **Decisione**: cartella ok se `acceptance_ratio >= 0.80` (`--rect-depth-min-accepted-ratio`), altrimenti review reason `rect_depth_autonomous_review`. Disattivabile con `--disable-rect-depth-autonomous`.

## 14. Scala #18–#21 (aggiunta 2026-08-08)

- **Subprocess** verso `tools/scale/predict_scale_from_pipeline.py`, sul modello di rect_depth:
  `pipeline_context.json` con le evidenze già calcolate, disattivabile con `--disable-scale-stage`.
- Sta **dopo** la depth perché `#21` è per depth: `#17` definisce i gruppi e `#18`–`#22` portano
  una voce ciascuna, nello stesso ordine.
- **Non ricalcola niente**: verso dal `su_giu_pred` delle righe LR-marker (già il verdetto del
  marker sopra il prior della rete), rect per immagine da `echo_rect_*_abs`, depth dallo stadio
  RECT_DEPTH, vendor dal classificatore.
- **Decisione**: cartella ok se le depth `accepted` sono ≥ `--scale-min-accepted-ratio` (0.80);
  review reasons `scale_rotation_not_supported`, `scale_no_depth_groups`,
  `scale_line21_incomplete`, `scale_depth_without_answer`, `missing_scale_predictions`, `scale_review`.
- Dettagli, accortezze conservate e ritrovamenti da misurare:
  `docs/scala_integrazione_pipeline_2026-08-08.md`.

## 15. Righe residue

- **#01 VERSION**: costante `--fss-version` = "4.0".
- **#02 ID_ECHO** (`EchoIdResolver` :5175, uso :8387): resolver storico dal manifest di riferimento, chiavi `vendor+probe+video → vendor+probe → vendor+video → vendor`; support = quota del vincitore. Irrisolto → policy → review `missing_id_echo`.
- **#12 GROUP_ORIENTATION**: **forzata a 4 (= symbol)**, source `forced_constant` (:8427). Il `GroupOrientationResolver` (:5578) è definito ma **mai usato** (dead code); il check di review su support #12 non può mai scattare.
- **#18–#21: gestite dallo stadio scala** (sezione 14).
- **#05, #15, #17, #22–#26: NON gestite** da questo script.

## 16. Output (:8661, :9260, :9571)

- `folder_fss_head_predictions.csv` — 1 riga per cartella, ~135 colonne (`line_XX` + `_source` + `_support`, confidenze, top-3 JSON, metriche rotazione/dedup, `status`, `review_reasons`).
- `folder_fss_head_preview.txt` — blocco leggibile #01…#16 per cartella.
- `summary.json` — soglie effettive, contatori per stadio, status counts, warnings.
- CSV per-immagine: su_giu, lr_marker, lt, rect_depth, `scale_per_image_predictions.csv`; `rect_red_pipeline_by_folder.json`; encoding struct updates.
- Per cartella, lo stadio scala lascia `scale/NNNN_<cartella>_<hash>/` con `pipeline_context.json`,
  `scale_per_image.csv`, `scale_per_depth.csv`, `scale_lines.json`, `summary.json`: è la cartella
  da passare a `study_scale_folder.py --from-pipeline` per rivedere e correggere.
- **⚠ Nessun file `.fss` viene scritto**: la pipeline predice i valori delle righe in forma tabellare; l'assemblaggio del file `.fss` è ancora uno step a valle non implementato. Anche il wrapper produce solo review HTML/JSON.

## Wrapper `run_pipeline_single_folder_safe.py` (:3512)

- Prende una singola cartella, crea `input_ref/` con symlink (eventualmente filtrando immagini escluse), lancia la pipeline come subprocess con `--max-folders 1`, poi genera `step_checks_review.html` + JSON/TXT di verifica.
- Attenzione: qui `--probe-min-confidence` default **0.50** (vs 0.40 del core) e `--vendor-min-confidence` 0.50.

---

## Punti aperti emersi dalla revisione

Aggiornamento 2026-07-09 — risolti nei commit `8c21bc3` (pipeline) e `275d823` (workbench):

1. ~~Margine vendor non usato come gate~~ → **fatto**: nuovo `--vendor-min-margin` (default 0.0 = comportamento storico), stessa policy low-confidence + review reason `low_vendor_margin`.
2. ~~`--rect-sample-per-folder` ignorato~~ → **fatto**: implementato campionamento uniforme (0 = tutti i frame, default invariato).
3. ~~Soglie incoerenti core/wrapper~~ → **fatto**: wrapper allineato a probe 0.40.
4. ~~`GroupOrientationResolver` dead code~~ → **fatto**: rimosso (recuperabile da tag `v0-checkpoint-pre-claude`); #12 resta forzata a 4.
5. **Nuovo**: diff prima/dopo nel rerun selettivo del workbench (`GET /api/runs/<id>/recompute_diff` + pannello UI, caricato automaticamente dopo ogni ricalcolo).

Ancora aperti:

1. **Manca il writer `.fss`**: il gap principale per chiudere il loop end-to-end (Fase 2 roadmap) — dal CSV al file `.fss` validabile con `compare_fss.py`.
2. **rect_depth è già integrato nel codice** (subprocess hook), contrariamente ai report di stato: da validare su batch reale, non da integrare.
3. Probe senza OCR refine nella pipeline (esiste come script separato): possibile integrazione.
