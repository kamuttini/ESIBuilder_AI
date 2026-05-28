# Analisi Duplicati/Simili Tools

- Script analizzati: **29**
- Duplicati esatti (hash uguale): **0 gruppi**

## Duplicati Esatti
- Nessun duplicato byte-identico trovato.

## Script Simili (score >= 0.60)
- `tools/ultrasound/train_ultrasound_probe_classifier.py` <-> `tools/ultrasound/train_ultrasound_vendor_classifier.py` | score=0.856 (seq=0.866, jaccard=0.831)
- `tools/line16/train_orientation_template_detector.py` <-> `tools/ultrasound/train_ultrasound_rect_net.py` | score=0.697 (seq=0.696, jaccard=0.699)
- `tools/ultrasound/predict_ultrasound_probe_by_folder.py` <-> `tools/ultrasound/predict_ultrasound_vendor_by_folder.py` | score=0.636 (seq=0.653, jaccard=0.596)

## Script Simili (0.45 <= score < 0.60)
- `tools/ultrasound/train_ultrasound_rect_net.py` <-> `tools/ultrasound/train_ultrasound_vendor_classifier.py` | score=0.491 (seq=0.437, jaccard=0.615)

## Simili Deboli (0.35 <= score < 0.45, top 8)
- `tools/ultrasound/train_ultrasound_probe_classifier.py` <-> `tools/ultrasound/train_ultrasound_rect_net.py` | score=0.406 (seq=0.346, jaccard=0.547)
- `tools/line16/train_orientation_template_detector.py` <-> `tools/ultrasound/train_ultrasound_vendor_classifier.py` | score=0.402 (seq=0.359, jaccard=0.502)
- `tools/line16/train_orientation_template_detector.py` <-> `tools/ultrasound/train_ultrasound_probe_classifier.py` | score=0.385 (seq=0.336, jaccard=0.498)

## Raccomandazioni
- Tenere separati ma allineare utility condivise:
  - `train_ultrasound_probe_classifier.py` vs `train_ultrasound_vendor_classifier.py`
  - `predict_ultrasound_probe_by_folder.py` vs `predict_ultrasound_vendor_by_folder.py`
- Famiglia già gestita come legacy:
  - `tools/old/ultrasound/infer_ultrasound_rect_net.py`
- Candidate a refactor in modulo comune (non merge totale): dataset loaders, metriche, CSV writers.
