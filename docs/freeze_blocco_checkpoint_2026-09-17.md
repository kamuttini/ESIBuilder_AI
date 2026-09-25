# Blocco FREEZE / proibite — checkpoint 2026-09-17

Obiettivo dichiarato: **ottenere lo stesso output del vecchio ESIBuilder** per le immagini
proibite, a partire dalle acquisizioni.

## Contratto di output

Per ogni setup il vecchio software scriveva tre cose. Le dobbiamo riprodurre tutte:

```
DB_setup/setup_<ID>.freeze                       INI QSettings, un [Freeze_i] per proibita
image_samples/image_proibite_setup_<i>.png       copia dello screenshot scelto
DB_echo/setup_<ID>/proibited_screen_<i>_0.png    ritaglio di rectTemplate
```

Campi di `[Freeze_i]`, e quali sono davvero da predire:

| Campo | Da predire? | Nota |
|---|---|---|
| `strFileName` | sì (selezione) | quale immagine dell'acquisizione è la proibita |
| `rectTemplate` | sì (localizzazione) | il box del marker distintivo |
| `rectFind` | no, derivato | `rectTemplate` + 3 px per lato, clampato ai bordi immagine |
| `bValueFind` | no, costante | `true` in 700/700 entry del GT |
| `bThisHasDone` | no, costante | `true` in 700/700 |
| `isScreenSaver` | da decidere | `true` in 202/700 (29%), significato da chiarire (domanda F3) |
| `nIndexOfListSetup` | no, derivato | posizione nella lista |

Ordine dei gruppi: QSettings scrive in ordine **lessicografico**, quindi con più di dieci
entry l'ordine è `Freeze_1, Freeze_10, Freeze_11, Freeze_2, ...`. Rispettarlo serve per
l'identità byte a byte.

## Ground truth disponibile

Su `SSD_esi1_n1`: 401 file `setup_N.freeze`, **700 entry** su 334 workspace, 899 ritagli
`proibited_screen_*.png`, 862 `image_proibite_setup_*.png`.

Il vecchio software rinominava le immagini in `image_proibite_setup_<i>.png` perdendo lo
stato rappresentato. La classe è stata recuperata confrontando i byte del campione con i
file originali delle acquisizioni (`ACQUISITION ELABORATION`, dove il nome dice lo stato):
188 entry recuperate per contenuto, 430 avevano ancora il nome legacy, 82 restano senza
riscontro.

Classi ricostruite sulle 700 entry:

| classe | entry | | classe | entry |
|---|---:|---|---|---:|
| unknown | 408 | | zoom | 22 |
| freeze | 103 | | split | 13 |
| cfm | 68 | | elasto | 11 |
| pw | 38 | | biopsy | 10 |
| mmode | 23 | | contrast | 4 |

Le 408 `unknown` sono entry il cui nome (legacy o recuperato) non dice lo stato
(`PROIBITA.png`, `P.png`, ...): il box c'è, manca l'etichetta. Si recuperano con la review
umana oppure raggruppando i ritagli per somiglianza dentro lo stesso vendor.

Dimensione mediana del `rectTemplate`: **38 × 20 px** su schermi 1920×1080 (per le sole
freeze: 60 × 23). È il vincolo tecnico principale: a 384 px di input il marker sparisce.

## Cosa esiste adesso in `tools/freeze/`

- `freeze_io.py` — parser e writer del formato `.freeze`, più la regola di `rectFind`.
  **Validato: round-trip byte-identical su 401/401 file legacy.**
- `compare_freeze.py` — quality gate, stessa forma di `compare_fss.py`
  (`legacy.freeze` vs `nuovo.freeze`, exit 0 = compatibile, IoU minima configurabile).
- `build_freeze_dataset.py` — costruisce il manifest unendo GT legacy e nomi originali,
  split leak-free per workspace stratificato per vendor.
- `build_freeze_review_html.py` — galleria statica di review (contesto + ritaglio ingrandito,
  verdetti solo in localStorage, esportazione CSV esplicita: non scrive mai sul dataset).

Dataset: `artifacts/20_datasets/freeze_dataset_v1/`
(700 entry, 334 workspace, split 506/102/92 su 234/50/50 workspace).

## Metrica di accettazione proposta

Replay sui workspace legacy: per ogni setup si rigenera il `.freeze` dalle acquisizioni e si
confronta con quello vero via `compare_freeze.py`. Due numeri distinti:

1. **selezione** — % di setup in cui l'insieme di immagini proibite proposto coincide
2. **localizzazione** — IoU media di `rectTemplate` e % di entry sopra soglia

Coerente con il criterio zero-regressioni: nessuna modifica accettata se peggiora il
confronto per-workspace.

## Aperto

- Output atteso dalla nuova app: vedi `docs/domande_per_francesca.md` (F2)
- `isScreenSaver`: nessun default provvisorio finché non è chiarito (F3)
- Perimetro classi: si parte da freeze, con le altre proibite come negativi difficili
