# Processo blocco Orientation — marker + banca template per vendor

Aggiornato: 2026-07-09 (sessione con Camilla). Questo documento è il riferimento
descrittivo del processo; il prototipo è il bundle
`artifacts/41_orientation_marker_detector_bundle/` + adapter
`tools/orientation/apply_bundle_marker_logic_to_pipeline_sugiu.py`; il benchmark è
`tools/orientation/eval_marker_bundle_vs_legacy_line16.py` (output in
`artifacts/42_orientation_eval_vs_legacy_line16/`).

## Principio

Le immagini ecografiche hanno 4 orientamenti (NF, LR, UD, LRUD). NF/UD si distinguono
dal contenuto (rete su/giù, esistente e affidabile); il ribaltamento sinistra/destra è
invisibile dal contenuto (immagini speculari), quindi si usa il **marker di orientamento**
che ogni ecografo disegna a schermo. Il quadrante del rettangolo ecografico in cui cade
il marker determina l'orientamento — **convenzione valida per tutti i vendor**:

| Quadrante | Orientamento |
|---|---|
| alto-sinistra | NF |
| alto-destra | LR |
| basso-sinistra | UD |
| basso-destra | LRUD |

Gli assi che dividono i quadranti sono gli assi mediani del rettangolo ecografico
(riga #11). Il marker può occupare qualsiasi posizione all'interno del suo quadrante
e può **cambiare dimensione** tra immagini della stessa cartella (matching multi-scala).

## Pipeline per cartella

1. **Rettangolo ecografico**: rete rect esistente + refine post-processing (riga #11).
2. **Rete su/giù** sul crop del rettangolo → verticale predetto (su = NF/LR, giù = UD/LRUD).
3. **Localizzazione template vendor (riga #13, nome ecografo; anche #14 nome sonda)** e
   **esclusione di quelle zone dalla ricerca marker**. Motivo: i loghi vendor possono
   essere quasi identici al marker (es. la "m" del logo Mindray) e generare falsi match.
   In produzione i box arrivano dal resolver/rete #13-#14 della pipeline; nel benchmark
   dai `.fss` legacy. Nota: la zona esclusa va **espansa con un margine** (default 0.75×
   la dimensione del box) perché il logo grafico spesso sta appena fuori dal box del
   testo (verificato su Mindray: la "m" è sopra il box #13 di "DC-70").
4. **Ricerca marker** (template matching NCC contro la banca template storica del vendor)
   nella **metà predetta** dalla rete su/giù.
5. Se lo score non è soddisfacente → **ricerca nell'altra metà**; se lì lo score è molto
   più alto, si corregge il verticale e la cartella viene **segnata per il retraining**
   della rete su/giù.
6. Se ancora niente → **espansione progressiva del margine** di ricerca oltre il
   rettangolo (il marker esterno al rect è raro ma possibile). Le soglie di tutti questi
   passaggi vanno tarate col benchmark vs riga 16 legacy.
7. **Divisione dei marker nei 4 quadranti** → per ogni quadrante un box envelope che
   contenga precisamente tutti i marker di quella zona → **riga #16** del `.fss`.
8. **Output finale**: riga #16 + **crop del template** di orientamento (ne basta uno
   per cartella) per la banca storica / il `.fss`.

## Banca template per vendor

- Storico dei template per vendor, senza duplicati troppo simili (dedupe per similarità).
- Fonte: `DB_echo/setup_*/orientation_*.png` delle cartelle elaborate col vecchio
  software (volume `SSD_esi1_n1`); attuale banca in
  `artifacts/41_orientation_marker_detector_bundle/orientation_marker_detector/templates/`.
- Vendor mancanti noti: Sonostar.
- Sviluppo futuro (quando il software sarà pronto): aggiornamento automatico della banca
  quando nelle run su cartelle nuove compare un template abbastanza diverso.

## Casi limite legacy (da gestire, vedi ROADMAP → Rischi aperti)

- **Progetti mono-orientamento**: in passato alcuni template ribaltati/speculari
  rompevano il match del vecchio software → quei progetti furono configurati per un solo
  orientamento (spesso UD, es. `BK Profocus Prato "SOLO X"`). La loro riga #16 contiene
  4 copie dello stesso box (che può puntare a un elemento UI, es. il LED "S/D") e **non**
  rappresenta NF/LR/UD/LRUD: non usarla come ground truth per-orientamento.
- **GT dai filename legacy**: i token `flip_lr`/`no_flip` nei nomi file
  `image_depth_find_*` indicano il ramo di calibrazione, **non** lo stato reale dello
  schermo (verificato visivamente). GT affidabile solo da nomi cartella
  ("SOLO X", `NoFlip/LR/UD/LRUD`) o dalla posizione del marker rispetto ai box riga #16.

## Criteri di verifica (benchmark vs legacy)

- Accuracy gruppo orientamento sulle cartelle con GT da nome cartella.
- Concordanza `gt_by_box`: gruppo predetto vs box riga 16 legacy contenente il marker.
- Quota marker dentro il box legacy del gruppo predetto.
- Review rate (policy confidenza: mai forzare predizioni incerte).
- Target di produzione da definire a valle della taratura soglie sul run completo.

---

## Aggiornamento 2026-07-22 — detector migliorato dalle correzioni umane + integrazione pipeline

Sessione con Camilla: run completo su tutte le 272 cartelle di `SSD_esi1_n3`
(runner `predict_marker_envelopes_batch.py` chained sugli output official-stages),
review umana nella gallery, e conseguente miglioramento del detector con criterio
**zero regressioni** (confronto per-cartella vecchio/nuovo — mai peggiorare ciò che
già funzionava).

### Miglioramenti al bundle detector (`detector.py`, condiviso da runner e pipeline)

1. **Gate di plausibilità patch** (`_patch_is_plausible` in `_match_template_exact`):
   l'NCC normalizzato può dare picchi spuri su zone nere/piatte (sfondo, aree di
   esclusione azzerate). Un match è valido solo se la zona ha `std >= 5` e `max >= 40`;
   altrimenti si passa al picco di correlazione successivo (fino a 6 tentativi).
   Risolve i casi "template sempre nero" (Esaote 114/115/237/238, GE 13).
2. **Matching multi-scala** (`_rescale_template`, param `scales` in `detect_marker`,
   `select_best_template`, `analyze_images`): il glifo del marker cambia dimensione tra
   ecografi/frame. Selezione a due passate (tutti i template a scala 1.0 → top-K anche
   alle altre scale) + per-immagine sulla scala vincente. Scale di default
   `0.75,1.0,1.3,1.7,2.2`. Risolve i GE dove il marker reale era ~36-63px contro
   template da ~28px.
3. **Risoluzione ambiguità** (in `detect_marker`): quando entrambe le metà falliscono
   ma il match finale è forte (`>= expanded_threshold`), il quadrante si risolve dalla
   posizione del marker invece di finire in review. Recupera i marker trovati fuori dal
   rettangolo (cartelle 224, 124, 165).

### Banca template arricchita dalla review

- Nuovo tool `tools/orientation/harvest_marker_templates_from_review.py`: dai box
  ridisegnati a mano nella gallery (CSV `review_marker_correzioni.csv`) ritaglia le
  regioni verificate e le aggiunge alla banca del vendor, deduplicando (correlazione
  >= 0.90) contro banca e harvest stesso. Aggiorna `review_decisions.json`
  (lista `accepted` + `maintenance_log`), quindi `load_vendor_templates` le usa subito.
- Sessione 2026-07-15: aggiunti **51 template** (19 Esaote, 22 GE, 4 BK, 3 Hitachi,
  3 Philips) dai 112 box corretti dall'utente.
- Attenzione: la dir GE su disco si chiama `Ge` — il matching path è case-insensitive.

### Template pinnati dalla review (`--pinned-templates`)

- Mappa `cartella -> [vendor/marker_NNN.png, ...]`
  (`artifacts/44_orientation_rerun_after_corrections/pinned_templates.json`): i template
  raccolti da UNA cartella vengono provati per-immagine su quella cartella accanto a
  quello auto-selezionato. Vincono **solo con doppia soglia**: score assoluto `>= 0.90`
  E superiore al match automatico di `+0.03` di margine. Così non possono mai peggiorare
  un frame, e le cartelle senza pin si comportano esattamente come prima.
- Distinguere il vendor: se la CNN vendor ha confidenza bassa (`< 0.35`) non sovrascrive
  più il nome cartella quando questo risolve a un vendor con banca (fix 165.Mindray, che
  la CNN dava "ExactVu" @0.14).

### Risultato validato (31 cartelle annotate)

- Box umani agganciati: da **71/112 a 97/112**, **zero regressioni** per-cartella.
- Review rate medio sulle annotate: 36% → 3%.

### Integrazione nella pipeline ufficiale (2026-07-22)

Lo stage marker era **già** nella pipeline dopo lo SU/GIU
(`_bundle_predict_lr_marker_on_su_giu_rows` in
`tools/ultrasound/predict_fss_head_from_acquisitions.py`, importa lo stesso bundle).
Gate anti-nero, risoluzione ambiguità e i 51 template raccolti erano già ereditati.
Aggiunti i due miglioramenti mancanti tramite `analyze_images` (esteso con `scales` +
`pinned_templates`, backward-compatible), esposti da due nuovi argomenti CLI:

```bash
--lr-marker-scales "0.75,1.0,1.3,1.7,2.2" \
--lr-marker-pinned-templates artifacts/44_orientation_rerun_after_corrections/pinned_templates.json
```

Senza flag → comportamento storico invariato. Smoke test pipeline completa su cartella
233: 63→93 immagini `ok`, i template pinnati 114/116/117/118 attivati, riga #16 più
accurata. Nota: la mappa pin è specifica del volume n3 (chiavi = nomi cartella di quel
dataset); su cartelle nuove i pin restano inerti — il beneficio dei pin è retroattivo,
multi-scala e template raccolti sono invece migliorie generali.
