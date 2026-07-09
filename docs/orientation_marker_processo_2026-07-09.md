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
