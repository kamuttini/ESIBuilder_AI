# Presentazione stato lavori ESIBuilder AI

Stato aggiornato al **10 aprile 2026**.

## 1) Executive summary 

In queste settimane abbiamo portato il progetto da fase esplorativa a una pipeline operativa con output utilizzabile per la generazione `.fss`.

Risultati chiave:
- pipeline vendor a livello cartella pronta per uso operativo su **426 cartelle**;
- classificatore vendor robusto con performance alte (test **accuracy 0.9824**, **macro-F1 0.9738**);
- mapping finale conservativo già pronto per `.fss` (`folder_vendor_for_fss_recommended_ocr.csv`);
- blocchi probe/rect integrati in smoke test end-to-end;
- avviato hardening su casi critici (Mindray/GE) con miglioramenti quantitativi misurati.

## 2) Obiettivo di business coperto

Obiettivo: ridurre il lavoro manuale di configurazione e rendere la generazione `.fss` ripetibile, tracciabile e scalabile.

Copertura raggiunta oggi:
- riconoscimento automatico vendor/probe/rect su cartelle immagini;
- policy di confidenza con gestione `ok/review`;
- output CSV pronti da agganciare alla generazione `.fss`.

## 3) Lavoro fatto sul dato (fondamenta)

Pulizia e qualità dataset:
- rimosse **5775** immagini `negative` dal dataset;
- split rigenerato e stabilizzato su **426 cartelle** / **28537 immagini**;
- split train/val/test immagini: **18456 / 4299 / 5782**;
- split train/val/test cartelle: **236 / 86 / 104**.

Evidenze:
- `docs/vendor_recognition_report_2026-03-07.md`
- `artifacts/10_active_pipeline/pipeline_fss_head/references/rect_dataset_no_negative_v2/split_summary.json`

## 4) Blocco vendor: risultato principale

Training vendor (run scelto):
- `best_val_macro_f1`: **0.9001**
- test `accuracy`: **0.9824**
- test `macro_f1`: **0.9738**

Output operativo folder-level (1 vendor per cartella):
- cartelle totali: **426**
- `ok`: **414**
- `review`: **12**

Refine OCR sui soli review:
- review con almeno un hit OCR: **7/12**
- override applicati in modo conservativo: **6**
- file finale raccomandato per `.fss`:
  - `artifacts/40_outputs_eval/vendor_folder_predictions_no_negative_v1/folder_vendor_for_fss_recommended_ocr.csv`

Evidenze:
- `artifacts/10_active_pipeline/pipeline_fss_head/models/vendor_training_no_negative_v2_power/test_metrics_from_best.json`
- `artifacts/40_outputs_eval/vendor_folder_predictions_no_negative_v1/summary.json`
- `artifacts/40_outputs_eval/vendor_folder_predictions_no_negative_v1/summary_ocr_refine.json`

## 5) Blocco probe: completato e operativo con review assistita

Training probe:
- `best_val_macro_f1`: **0.8318**
- test `accuracy`: **0.8477**
- test `macro_f1`: **0.7036**

Output folder-level probe:
- cartelle totali: **426**
- `ok`: **404**
- `review`: **22**

OCR refine su review probe:
- review processate: **22/22**
- review con hit OCR: **11**
- override raccomandati: **3**

Evidenze:
- `artifacts/10_active_pipeline/pipeline_fss_head/models/probe_training_no_negative_v1/metrics.json`
- `artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/summary.json`
- `artifacts/40_outputs_eval/probe_folder_predictions_no_negative_v1/summary_probe_ocr_refine.json`

## 6) Blocco rect (riquadro ecografico): pronto per produzione

Modello rect globale:
- `best_val_iou`: **0.9331**
- test `mean_iou`: **0.9174** su **5782** campioni

Evidenza:
- `artifacts/10_active_pipeline/pipeline_fss_head/models/rect_training_e40_run2/metrics.json`

Nota tecnica importante:
- routing vendor->rect è già predisposto; nel test con soglia alta (`vendor_conf_threshold=0.9`) è andato quasi tutto in fallback globale, quindi il prossimo guadagno atteso è dalla calibrazione routing/vendor-specialized.

Evidenza:
- `artifacts/40_outputs_eval/rect_inference_vendor_routing_test/summary_test.json`

## 7) Pipeline scala: da baseline fragile a workflow controllato

Qualità dataset scala:
- dataset leak-free validato su **5573** righe / **403** gruppi
- leak check: **0** cross-split (path/setup/hash)

Baseline classica (metodo storico):
- copertura test: **6.34%** (53 su 836)
- alta percentuale `line_not_found`

Nuovo percorso image-model + coda fix:
- run medium su subset curato: **1200** righe
- coda priorità per correzioni GT: **P0=34**, **P1=49**, **P2=141**, **P3=976**

Evidenze:
- `artifacts/24_scale_pretrain_review/scale_pretrain_review_summary.json`
- `artifacts/31_scale_classic_baseline/classic_scale_summary.json`
- `artifacts/32_scale_image_model_medium_cpu/summary.json`
- `artifacts/33_scale_gt_fix_queue/summary.json`

## 8) Hardening mirato su criticità reali

Mindray post-processing (analisi 2026-04-02):
- regola raccomandata aumenta IoU medio pesato da **0.3573** a **0.5295**
- miglioramento: **+0.1722**

GE support split:
- diagnosi su modelli critici e proposta support-aware
- spostate **172** immagini test->train mantenendo holdout di test

Evidenze:
- `artifacts/40_outputs_eval/mindray_postproc_analysis_20260402_v3/recommended_rule_summary.json`
- `artifacts/40_outputs_eval/ge_training_diagnosis_20260331/ge_split_proposal_summary.json`
- `artifacts/40_outputs_eval/ge_support_split_manifest_20260331/support_split_summary.json`

## 9) Integrazione end-to-end (`fss_head`) e smoke test

Pipeline integrata (vendor + probe + rect) già eseguibile su cartelle reali con policy configurabile (`review` o `ask_user`).

Esempio smoke:
- cartelle scansionate: **5**
- cartelle predette: **5**
- dedup immagini attivo (in test: **52** duplicate rimosse su **712** immagini raw)
- output: preview + CSV strutturati per aggiornamento campi `.fss`

Evidenze:
- `artifacts/50_smoke_tests/fss_head_from_acquisitions_smoke/summary.json`
- `artifacts/50_smoke_tests/fss_head_smoke_policy_review/summary.json`
- `artifacts/50_smoke_tests/fss_head_smoke_policy_ask_custom/summary.json`

## 10) Efficienza/ordine del progetto

Miglioramenti di gestione e manutenzione:
- cleanup repository/artifact: **~1.04 GB** recuperati;
- analisi duplicati script: nessun duplicato byte-identico, refactor candidate mappati;
- toolkit QA `.fss` pronto (`compare_fss.py`, `audit_setup_outputs.py`).

Evidenze:
- `docs/cleanup_report_2026-03-07.md`
- `tools/00_analysis/DUPLICATI_SIMILI.md`
- `tools/README.md`

## 11) Rischi aperti (trasparenti)

- probe: alcune classi rare/unseen hanno recall basso o support nullo in test;
- orientation symbolic: qualità non ancora a livello produzione (IoU medio basso in eval routed);
- scale line: forte dipendenza dalla qualità GT, pipeline pronta ma backlog correzioni ancora da smaltire.

## 12) Piano prossimo step (proposta)

1. Chiudere il loop operativo `.fss` con validazione compatibilità automatica (`compare_fss.py`) su batch reale.
2. Ridurre review rate vendor/probe sotto soglia target (es. <3%) con tuning soglie e retraining mirato classi deboli.
3. Portare il blocco orientation a baseline accettabile (definizione target IoU + training/review dedicato).
4. Smaltire coda P0/P1 della scala e rilanciare valutazione su set completo leak-free.

## 13) Richieste alla capa (decisioni)

- conferma priorità: chiusura end-to-end `.fss` prima di estendere nuove feature;
- disponibilità di finestra review esperta per i casi `review` (vendor/probe/scale);
- allineamento su KPI di go-live (review rate, compatibilità `.fss`, copertura vendor/probe).

---

## Talk track rapido (2 minuti)

"Abbiamo già trasformato il progetto in una pipeline concreta: oggi riconosciamo vendor/probe/rect a livello cartella con output pronto per `.fss`, e sul vendor siamo già su metriche molto alte (98.2% accuracy test). Dove il sistema non è sicuro non forza mai: marca `review`, e abbiamo aggiunto un refine conservativo OCR+hint per ridurre errori. Abbiamo anche lavorato sulle criticità reali (Mindray/GE) con miglioramenti misurati, e abbiamo fatto pulizia/quality tooling per rendere il flusso mantenibile. Il prossimo passo è chiudere il ciclo end-to-end con validazione automatica `.fss` su batch reale e consolidare orientation/scale, che sono le ultime aree ancora non al livello del vendor." 
