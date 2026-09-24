# App ESIBuilder AI — versione base

Wizard locale che porta da una cartella di acquisizioni al file `.fss`. Struttura e flusso:
`docs/specifiche_app_esibuilder_ai_2026-08-26.md`.

## Avvio

```bash
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/app/server.py --port 8800
```

Poi apri `http://127.0.0.1:8800/`. I progetti finiscono in `artifacts/80_app_projects/<progetto>/`
(`--projects-root` per cambiare cartella).

## Cosa c'e' dentro

| File | Ruolo |
|---|---|
| `constants.py` | costanti legacy (probe type, video input, gruppi orientamento, limite rect ESI) e default di match |
| `fss_writer.py` | serializzatore delle 23/26 righe `.fss` + validazione |
| `project.py` | `project.json`, registro degli 11 step, grafo delle dipendenze e invalidazione |
| `importer.py` | scansione cartella, dedup esatta, metadati video dai nomi file, proposta di resize |
| `rotation.py` | rotazione OSD con tesseract (>=2 voti, >=60% di supporto), 4 processi in parallelo |
| `timestamp_detection.py` | proposta OCR dell'area data/ora su 8 immagini, con supporto spaziale e confidenza |
| `anagrafica.py` | lettura/scrittura di `encoding_struct*.xlsx`: ID da modelli, candidati per sonda, nuova riga con backup |
| `sonde_per_vendor.py` | la lista sonde → vendor in `sonde_per_vendor.csv` (vendor = chi costruisce l'ecografo), ricavata dall'anagrafica e riallineata a ogni versione nuova (`--riallinea`) — la usa la decisione sulla sonda |
| `probe_ocr.py` | il nome della sonda letto sullo schermo (box #14 e schermo col ventaglio oscurato), abbinato ai nomi dell'anagrafica; casi di controllo in `selftest_probe_ocr.py` |
| `probe_decision.py` | rete sonda + nome sullo schermo: accettata se concordano, revisione se no (`analysis.probe_decision`, scheda «Quale sonda», `POST /api/projects/<id>/probe/choose`) |
| `inference.py` | vendor, sonda, rettangolo (routing per vendor), template #13 e #14, piano L/T e su/giu dai checkpoint della pipeline attiva |
| `stages.py` | marker di orientamento, depth e scala come sottoprocessi con `pipeline_context.json`, piu' i parser dalle righe del modulo al valore dello step |
| `probe_shape.py` | misura geometrica lineare/convex — **non agganciata**, i numeri misurati sono nel docstring |
| `server.py` | API Flask + servizio della UI |
| — | il materiale per le linee guida si cerca dall'import: `POST /api/projects/<id>/import/needle` |
| `static/` | wizard (vanilla JS, nessuna build) |
| `static/box_editor.js` | editor dei box: trascinamento, maniglie, slider, schermo intero con scorrimento delle immagini |
| `static/orientation_view.js` | visualizzatore orientamento: i 4 envelope, il marker per immagine, il ritaglio, filtri per gruppo, correzione con click |
| `marker_refine.py` | dal click approssimativo alla posizione precisa (match del bundle in una finestra) + verifica del gruppo + ricalcolo degli envelope |
| `orientation_marker.py` | il marker della cartella: innesco dalla banca, taglio del ritaglio, scelta per copertura, validazione dentro gli envelope |
| `selftest_roundtrip.py` | controllo: rigenera un `.fss` legacy e verifica che sia identico |
| `thresholds.py` | le soglie `TH` dei template (#13 #14 #15 #16 #17) misurate sui fotogrammi come la pagina Thresholds legacy, i ritagli `DB_echo`, la costruzione della riga #17 |

## Materiale per le linee guida (dallo step Import e analisi)

Nel pannello dell'import, la piega **Materiale per le linee guida** cerca nelle sottocartelle
del progetto i fotogrammi che servono allo step `guides` (`PAGE_CALIBRATION`, righe #22-#23).

Nel vecchio ESIBuilder quella sessione (`WdgPageCalibration`) chiede **una immagine per ogni
angolo** del kit — si rifiuta di proseguire se il numero non torna — e la richiede da capo per
ogni profondita' e per ogni flip. L'operatore ricalca l'ago che vede nel fotogramma, e da quel
tratto escono angolo e distanza dal centro. Per questo le acquisizioni sono fatte in acqua:
serve vedere l'ago nitido. Qui si risponde solo alla domanda *dove sono*, non *quale usare per
quale angolo*: quella e' la scelta dello step delle linee guida.

La risposta e' **per fotogramma**: il materiale puo' essere una manciata di immagini dentro una
cartella di centinaia, e una media di cartella le annegherebbe. Le sottocartelle servono solo a
raggruppare il risultato quando ci sono; in una cartella piatta si mostrano direttamente i
fotogrammi proposti. Sotto le 400 immagini si guardano tutte, sopra si campiona per sottocartella.

Esempio su un progetto di 146 fotogrammi in tre sottocartelle:

```
146 fotogrammi esaminati su 146 · 12 proposti
  VERIFICA AGHI    12 su 12 esaminati    punteggio migliore 0.994
  DEPTH             0 su 124
  PROIBITE          0 su 10
```

Le soglie vengono da `folder_policy.json` accanto al checkpoint (proposto >= 0.60, scartato
< 0.25, in mezzo «da verificare») e non dalla soglia del `metrics.json`, che mira al 98% di
precision e lascerebbe quasi tutto in dubbio.

Il ritaglio usa `_needle_tensor`, non `_crop_tensor`: quest'ultimo schiaccia il ritaglio in un
quadrato, e schiacciare cambia l'inclinazione degli aghi, che e' il segno da riconoscere.

Il risultato si salva in `needle_scan` dentro lo step `import` **senza invalidare** nulla a
valle: e' una informazione in piu' sulla cartella, non un dato geometrico.

Il checkpoint sta in `artifacts/91_needle_models/`, fuori dalla pipeline attiva: se manca, la
piega mostra l'errore e il resto dell'app funziona come prima. Il blocco che lo addestra e'
`tools/needle/` (preparazione dataset, ritagli, training, valutazione, report HTML).

**Limite noto:** il modello copre la sessione *a linee* (ago in acqua) e manca quella *a
griglia* (`WdgPageGridCalibration`, cartelle `GUIDA AGHI` / `GUIDA BIOPSIA`), che classifica
come negativa.

## Soglie e riga #17 (step `thresholds`)

Ogni blocco con un template porta dentro `CH:TH:P1:P2:...` la soglia `TH` sotto cui ESI
accetta il match (`TM_SQDIFF`, quindi piu' basso e' meglio). Zero vuol dire «mai». Il bottone
*Calcola le soglie* misura ogni template sui fotogrammi del progetto come faceva l'operatore
nella pagina Thresholds del vecchio ESIBuilder, ma senza liste a mano:

| riga | template | positivi | negativi |
|---|---|---|---|
| #13, #14 | il box, ritagliato con la convenzione DB_echo (+3, -5) | tutti i fotogrammi | fotogrammi di altri progetti sul disco e schermate proibite; chi mostra ancora il template e' scartato; senza niente, il template su sfondo nero |
| #15 | cio' che la schermata proibita ha e i fotogrammi normali no, fuori dal rettangolo | quella schermata | i fotogrammi normali |
| #16 | il marker consegnato, uno per gruppo | i fotogrammi del gruppo | i fotogrammi degli altri gruppi |
| #17 | il numero della depth, dal riquadro stretto dell'OCR allargato di 3 px | i fotogrammi letti a quella depth (per flip) | quelli delle altre depth |

`TH` sta a meta' fra il peggior positivo e il miglior negativo; se si sovrappongono la riga va in
review. La #17 nasce qui: un gruppo per depth nell'ordine di #18, quattro blocchi (NF, LR, UD,
LRUD) con il box di ricerca = riquadro stretto + 7 px, come i 28x26 del legacy intorno a un 19x17.
La sezione mostra anche la **prova di ESI**: ogni fotogramma con orientamento e depth noti, quanti
sono riconosciuti dal proprio blocco e solo da quello.

I ritagli finiscono in `templates/db_echo/` del progetto e, alla generazione, in
`out/DB_echo/setup_<id>/` accanto al `.fss`: senza quei PNG il file non si carica.

Servono prima rettangolo, orientamento e depth (con il riquadro dell'etichetta propagato).
Studio del formato e dei numeri dell'archivio: `docs/rect_depth_riga17_logica_2026-09-18.md`.

## Linee guida: #22 e #23 proposti dagli aghi (step `guides`)

Nello step **Linee guida** il bottone *Misura gli aghi* fa l'intera catena: prende i fotogrammi
di calibrazione che la ricerca dell'import ha gia' trovato, misura l'ago in ognuno, e converte
la misura nella coppia legacy — angolo (`#23`) e distanza dal top del RECT_ECHO al punto in cui
la prima linea incrocia la verticale centrale (`#22`).

Serve che siano gia' fatti: il **rettangolo**, i **pixel ratio** dello step depth e scala, e la
**ricerca del materiale** nello step Import e analisi. Se manca qualcosa il messaggio lo dice.

Le misure vengono raggruppate per angolo: ogni gruppo e' una famiglia di linee guida, cioe' una
voce di `#23`. *Porta la proposta nel valore* precompila l'editor nella forma che il writer si
aspetta.

### Quanto fidarsi

Misurato su 181 aghi tracciati a mano da Camilla, su fotogrammi di 40 acquisizioni diverse:

| | entro 1° | entro 3° |
|---|---|---|
| prima scelta del rilevatore | 23% | 36% |
| con le regole geometriche | **30%** | **47%** |

Quindi **e' una proposta da confermare, non un risultato**. Circa un terzo delle volte l'angolo
e' gia' giusto entro la tolleranza di 1° chiesta da ESI; in circa metà dei casi ci si arriva
correggendo poco. Le famiglie viste in un solo fotogramma stanno separate, sotto *incerte*:
con questa precisione una misura sola e' tanto probabile che sia un errore quanto un ago.

La colonna delle depth va **controllata una per una**: senza sapere a che profondita' e' stato
preso ogni fotogramma, la stessa distanza viene ripetuta su tutte. E' un punto di partenza.

### Cosa e' verificato e cosa no

La geometria e' verificata due volte: andata e ritorno sul formato legacy (12.860 combinazioni,
errore mediano 0.03°) e contro gli aghi tracciati a mano, che cadono a **0.94° di mediana**
dall'angolo legacy quando l'accoppiamento configurazione-acquisizione e' corretto. Il budget di
1° e' quindi raggiungibile: quello che manca e' la precisione del rilevatore, non la matematica.

Il blocco che misura vive in `tools/needle/`; `propose_guide_lines.py` fa la stessa cosa da riga
di comando su una cartella qualunque.

## Controllo del writer

```bash
python3 tools/app/selftest_roundtrip.py OldSoftwareEsiBuilder/templates/DB_setup/*.fss
```

Deve stampare `tutti i N file legacy rigenerati identici`: e' la prova che separatori e formati
numerici combaciano con il dialetto legacy.

## Stato

Reali in questa versione: passo zero (dedup, rotazione, vendor, sonda, rettangolo, piano L/T) come
job unico con avanzamento; pagina codici precompilata con la provenienza di ogni campo e selettore
del modello macchina quando l'anagrafica e' ambigua; rettangolo proposto con anteprima e box
disegnato; stato per step con invalidazione mirata; writer `.fss` con validazione; quality gate via
`tools/fss/compare_fss.py`.

Dopo l'import la catena prosegue da sola: piano L/T, template #14, orientamento, depth e scala.
Se la cartella contiene **L e T** e non e' stata sdoppiata, non si ferma: lavora sulle sole
immagini L (con sopra le correzioni a mano) e mette da parte le T, che restano visibili e
spostabili nella sezione Import. Spostare un'immagine da un piano all'altro fa ripartire i moduli
da li'; per configurare anche la T si sdoppia, e lo sdoppiamento ritrova tutta la cartella
(`plane_focus.json` nel progetto). Il server si ferma solo su una biplana il cui piano non e' mai
stato riconosciuto.
Durante l'import data e ora vengono cercate automaticamente dopo la rotazione: una proposta
affidabile entra subito nella deduplicazione, ma il riquadro rosa resta visibile e correggibile.
La ricerca puo' essere rilanciata o disattivata esplicitamente per il singolo progetto.

`#14 RECT_NAME_PROBE` lo propone una rete dedicata (`models/probe_template_line14/`, una sola
per tutti i vendor: separare per marchio non guadagna nulla, misurato su 48 cartelle). E' un
detector a heatmap: il picco vale come confidenza, e sotto `min_score` la riga resta vuota invece
di contenere un rettangolo a caso. Su 61 cartelle di test mai viste propone sul 57% con il 91% di
box corretti; il resto arriva in revisione. Il resolver storico che leggeva i `.fss` legacy da
`references/.../manifest_rect_echo.csv` resta inutilizzabile (quei percorsi puntano a
`ESIBuilder_AI/Dataset/`, cartella assente: 0 su 426 risolve) e comunque su macchine nuove
sbagliava il rettangolo nell'88% dei casi in cui rispondeva.

Non ancora agganciati: soglie e linee guida — quegli step accettano il valore in JSON, nella
forma che il writer si aspetta. Orientamento, depth e scala girano dai rispettivi moduli
(un bottone per modulo: "Calcola l'orientamento", "Calcola la depth", "Calcola la scala"), con gli artefatti in `<progetto>/stages/`. Il rettangolo `#11` e il template `#13` si correggono trascinando sull'immagine (anche a schermo
intero, scorrendo la cartella); il tracciamento a mano da zero degli altri box non c'e' ancora.

I checkpoint vengono da `artifacts/10_active_pipeline/pipeline_fss_head/models` (e
`artifacts/30_models` per il piano L/T). In un worktree git quella cartella sta nel checkout
principale: il server la trova da solo, oppure si passa `--models-root`.
