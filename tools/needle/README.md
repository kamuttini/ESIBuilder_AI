# Blocco aghi

Due cose diverse, che condividono i dati:

1. **trovare il materiale di calibrazione** in una acquisizione (usato nello step *Import e
   analisi* dell'app);
2. **misurare l'ago** nei fotogrammi trovati, per proporre le righe `#22` e `#23` del `.fss`
   (usato nello step *Linee guida*).

## Com'e' fatta la catena

| file | cosa fa |
|---|---|
| `prepare_needle_dataset.py` | manifest dalle cartelle grezze, split leak-free per acquisizione |
| `extract_needle_crops.py` | ritaglia al RECT_ECHO con i modelli ufficiali vendor+rect, e mette in cache |
| `train_needle_classifier.py` | classificatore "e' materiale di calibrazione?" |
| `eval_needle_classifier.py` | valutazione di un checkpoint senza riaddestrare |
| `predict_needle_on_folder.py` | lo stesso classificatore su cartelle grezze |
| `needle_frames.py` | scelta dei fotogrammi: cartella di calibrazione **e** classificatore **e** sonda giusta |
| `detect_needle_line.py` | rilevatore Hough: soglie multiple, test della cresta |
| `detect_needle_radon.py` | filtro adattato, usato solo come ripiego dove Hough non trova niente |
| `detect_needle.py` | i due insieme: Hough decide, il filtro adattato copre il vuoto |
| `refine_needles.py` | le regole di dominio: unione dei pezzi, max due aghi, paralleli, inclinazione, partenza |
| `guides_geometry.py` | geometria legacy: `.fss` + `.ndg` -> la linea, e la linea -> i numeri |
| `propose_guide_lines.py` | da una cartella a `#22` e `#23` proposti |
| `pair_configs_acquisitions.py` | accoppia le configurazioni legacy (n1) alle acquisizioni (n3) |
| `cache_legacy_setups.py` | copia in locale i `.fss`/`.ndg`/`.grid`, perche' il disco si smonta |
| `measure_needle_features.py` | misura se una caratteristica proposta separa davvero, prima di programmarla |
| `build_detector_review_html.py` | galleria di annotazione: voti per singolo ago, tracciamento, esclusioni |
| `build_needle_review_html.py` | report del classificatore |

## Dove sono i dati

- etichette umane: `artifacts/92_guides/etichette/` — 181 aghi tracciati e 355 giudizi
- copia locale dei legacy: `artifacts/92_guides/legacy_cache/` (390 configurazioni, 7.3 MB)
- accoppiamento: `artifacts/92_guides/pairing_v1/pairs.csv`
- modello del classificatore: `artifacts/91_needle_models/resnet18_256_cpu/`

## Stato

**Geometria: verificata.** Andata e ritorno a 0.03° di mediana su 12.860 combinazioni, e gli
aghi tracciati a mano cadono a 0.94° dall'angolo legacy quando l'accoppiamento e' giusto.

**Rilevatore: parziale.** 30% entro 1°, 47% entro 3° su 66 fotogrammi di 32 acquisizioni mai
usate per tarare. Basta per proporre, non per decidere.

**Biplane: non coperto.** Le sonde biplana calibrano su immagini di linee di biopsia con i
pallini, non su aghi in acqua, e il classificatore le rifiuta. E' il prossimo blocco.

## Le regole che funzionano, e perche'

Sono tutte misurate su `measure_needle_features.py`, e quelle che hanno retto vengono dalle
osservazioni di Camilla guardando gli errori, non da caratteristiche progettate a tavolino:

- **inclinazione** almeno 8° dall'orizzontale — 0 aghi su 104 sotto soglia, 27 errori su 74
- **partenza** nella meta' alta del rettangolo — 3 aghi su 104, 28 errori su 74
- unione dei pezzi con tolleranze prese dalle coppie che lei ha marcato come stesso ago

Non hanno funzionato, misurate: riverbero, frastagliatura, luminosita' della punta. E
l'estensione verticale separa benissimo come numero ma peggiora come filtro, perche' taglia via
i pezzi corti di aghi veri.
