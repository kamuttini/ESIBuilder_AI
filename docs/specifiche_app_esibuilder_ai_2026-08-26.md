# Specifiche app ESIBuilder AI — bozza v0.1

Data: 2026-08-26. Stato: **bozza in discussione con Camilla**, non ancora approvata.
Fonti: `preparazione al progetto/requisiti_collega.md`, `preparazione al progetto/spiegazione_file_fss.md`,
`preparazione al progetto/brainstorming.md`, `docs/albero_decisionale_pipeline_fss_head_2026-07-08.md`,
codice legacy `OldSoftwareEsiBuilder/` (pagine `wdgpage*`, `wdgcreateproject`, `qfilefss`), `ROADMAP.md` Fase 4.

## 1. Obiettivo

Un programma unico, usato da colleghi interni su Windows, che porta **da una cartella di immagini di
acquisizione a un file `.fss` valido** (più `.ndg` quando serve), in cui l'AI propone ogni valore e
l'utente conferma o corregge. Sostituisce l'uso del vecchio ESIBuilder per la fase di configurazione.

## 2. Perimetro v1 (deciso)

L'app copre **tutti** gli step della configurazione, quindi tutte le righe `#01`–`#26`.
Dove l'AI non è ancora pronta (linee guida, schermate proibite, transizioni biplana) si **replica il
comportamento manuale del vecchio ESIBuilder dentro la nuova app**, semplificato nella UX: l'utente non
deve mai tornare al software legacy per completare una configurazione.

Fuori dal perimetro: l'**acquisizione** delle immagini (resta dove è oggi; l'app parte da una cartella
già acquisita) e la modalità di adattamento sul campo su ESI/Ultra Service (fase 2 dei requisiti).

## 3. Principi

- **P1 — L'AI propone, l'utente conferma.** Il tracciamento manuale è l'eccezione, non la regola.
- **P2 — Nessuno step senza via manuale.** Ogni step ha sempre il fallback legacy: se il modello tace
  o sbaglia, l'utente traccia a mano e va avanti. Nessuno step può bloccare la configurazione.
- **P3 — Un solo import.** Il legacy chiede "File acquisizioni campione" in *ogni* pagina; qui la
  cartella si importa una volta e ogni step pesca da sola le immagini che gli servono.
- **P4 — I codici si cambiano in qualsiasi momento**, senza ricominciare (requisito esplicito del collega).
- **P5 — Ogni correzione è un dato.** Va in `feedback/inbox.jsonl` e nella banca template: la volta
  dopo lo stesso caso è automatico.
- **P6 — Il file è il contratto.** Un progetto è chiuso quando `compare_fss.py` contro un `.fss` legacy
  di riferimento non segnala differenze semantiche impreviste.
- **P7 — La confidenza è visibile.** Ogni valore mostra `accepted` / `review` / `reject`: il sistema non
  spaccia mai per certo un valore incerto.

## 4. Architettura

- **Backend**: Python + web server locale. La scelta era FastAPI; nella versione base e' **Flask**,
  perche' FastAPI e uvicorn non sono installati nel venv del progetto mentre Flask 3.1.3 c'e' gia' ed
  e' quello che usano `tools/review/app.py` e il workbench. La logica di dominio (progetto, writer,
  import) sta in moduli indipendenti dal framework: passare a FastAPI toccherebbe solo il routing. Riusa i moduli esistenti senza riscriverli (`tools/ultrasound`,
  `tools/depth`, `tools/scale`, `tools/orientation`, `tools/line16`, `tools/fss`), con lo stesso schema
  già in uso: `pipeline_context.json` in ingresso, artefatti JSON/CSV in uscita.
- **Frontend**: browser locale. Un solo componente canvas riusabile (immagine + box/linee, zoom, pan,
  nudge da tastiera) condiviso da tutti gli step di tracciamento.
- **Progetto = cartella su disco**: `project.json` + `input/` (o riferimenti) + `artifacts/` per step +
  `out/setup_NN.fss`. Nessun database.
- **Windows**: inferenza CPU via ONNX; Tesseract e dipendenze a versione fissata; packaging come
  cartella portabile che apre il browser. Il retraining resta su Mac/MPS.

## 5. Modello dati del progetto

`project.json` contiene:

1. **Codici** (equivalente di `Project_struct_t` del legacy `wdgcreateproject`): nome progetto, ID setup,
   ID echo, ID probe, ID file NDG, probe type, tipo video input, size video input, size image samples,
   group orientation, orientamenti disponibili (NF/LR/UD/LRUD), metodo test biplana, ID fss next probe.
2. **Sorgente immagini**: cartella, esiti dedup, rotazione applicata, risoluzione rilevata.
3. **Stato per step**: `{status: proposed|confirmed|corrected|manual|blocked, source, value,
   confidence, user_edited, ts}`.
4. **Righe `.fss`** calcolate, ognuna con la provenienza (modello, resolver storico, costante, utente).

Regola di **invalidazione**: cambiare un codice o un valore a monte marca `stale` solo gli step che ne
dipendono, mai tutto il progetto. È il meccanismo che realizza il principio P4.

### Grafo delle dipendenze

```
codici(ID echo, ID probe, probe type, NDG)
   ├─► step 0  ──► #02 #03 #04 #05 #25 #26
   └─► step 3 (biplana)
import/rotazione/risoluzione
   ├─► step 1  ──► #06–#10
   └─► TUTTI gli step con coordinate (cambiare risoluzione o resize invalida ogni box)
step 2 (rect echo #11)
   ├─► step 5 (orientamento, lavora sui crop del rect)
   ├─► step 6 (scala: il righello sta dentro il rect)
   └─► step 7 (depth)
step 6 (depth #18) ──► step 7 (#17 è per depth) ──► step 9 (#22 è per depth per angolo)
step 2/5/7 (template) ──► step 8 (soglie) ──► step 10 (generazione)
```

Conseguenze operative: cambiare il **probe type** dopo lo step 3 invalida biplana e linee guida ma non
il rettangolo; cambiare la **risoluzione** invalida tutte le coordinate; cambiare un **ID** non invalida
nulla di geometrico. Ogni step dichiara le proprie dipendenze, l'app ricalcola solo il sottoalbero e
mostra cosa è tornato in review.

### Schema `project.json` (bozza)

```json
{
  "schema_version": 1,
  "codes": {
    "project_name": "", "id_setup": 0, "id_echo": 0, "id_probe": 0, "id_ndg": 0,
    "probe_type": 1, "video_type": 0, "video_input_size": [1920, 1080],
    "image_sample_size": [1920, 1080], "group_orientation": 4,
    "orientation_available": {"NF": true, "LR": true, "UD": true, "LRUD": true},
    "biplana_test_method": null, "id_fss_next_probe": null
  },
  "source": {
    "folder": "", "images_total": 0, "duplicates_removed": 0,
    "rotation_applied": 0, "rotation_source": "", "resize_factor": 1.0
  },
  "steps": {
    "<step_id>": {
      "status": "proposed|confirmed|corrected|manual|blocked|stale",
      "source": "model|resolver|constant|user",
      "value": {}, "confidence": null, "user_edited": false,
      "depends_on": [], "ts": ""
    }
  },
  "fss_lines": {"11": {"value": "", "source": "", "step": "rect_echo"}},
  "runs": [{"run_id": "", "ts": "", "stage": ""}]
}
```

## 6. Il wizard

Ordine ripreso dal legacy (`WDG_ID_PAGE_*`), così il collega riconosce il flusso. Ogni step si apre
**già compilato** dalla proposta AI: la conferma è un tasto solo, gli step senza eccezioni si
attraversano in sequenza.

| # | Step | Legacy | L'AI propone | L'utente | Righe |
|---|---|---|---|---|---|
| 0 | **Import e analisi** | — (era sparso) | dedup, rotazione OSD, vendor, sonda, rettangolo, piano L/T: tutto in un passaggio, prima che l'utente veda qualcosa | indica la cartella e aspetta | #06–#10 |
| 1 | Progetto e codici | `WdgCreateProject` | si apre **già compilata** con quanto ricavato al passo 0 e dall'anagrafica | conferma/edita, sceglie il modello macchina, sempre riapribile | #02 #03 #04 #05 #25 #26 |
| 2 | Ecografo e sonda | `PAGE_US` | vendor, modello, sonda, rettangolo ecografico, template eco e sonda | conferma o ritraccia; opzione "template non presente" | #11 #13 #14 |
| 3 | Biplana | `PAGE_BIPLANA` | piano L/T e coppia di setup | conferma, sceglie il metodo di riconoscimento | #04 #24 #25 #26 |
| 4 | Schermate proibite | `PAGE_PROIBITED` | propone le candidate freeze trovate | indica freeze presente/assente; se assente si genera il negativo fittizio | #15 |
| 5 | Orientamento | `PAGE_ORIENTATION` | marker per i 4 gruppi NF/LR/UD/LRUD + envelope | conferma per gruppo; gestisce il caso legacy a 4 box identici | #12 #16 |
| 6 | Depth e scala | `PAGE_DEPTH_VALUE` | legge il righello: depth in mm, pixel ratio, scale line per depth | conferma; corregge una scala e la propaga alle altre depth | #18 #19 #20 #21 |
| 7 | Ricerca depth | `PAGE_DEPTH_FIND` | box template depth per depth, stati di transizione | conferma o ritraccia | #17 |
| 8 | Soglie | `PAGE_THRESHOLDS`, `PAGE_FSS_THRESHOLDS` | calcola `TH` dai match su positivi/negativi, con `MM=6` e `P1=20`/`P2=120` fissi | verifica sulle immagini rinominate; tocca un numero solo se il margine è insufficiente | i campi `TH` di #13 #14 #15 #16 #17 #24 |
| 9 | Linee guida | `PAGE_CALIBRATION`, `PAGE_GRID_CALIBRATION` | traccia le linee dove riesce; distanze del kit dal `.ndg` | conferma o traccia (comportamento legacy) | #22 #23 |
| 10 | Generazione e verifica | — | scrive `.fss`, esegue il quality gate, passata finale sulle immagini rinominate `T_LRUD_50` | verifica piano/orientamento/depth/scala e chiude | tutte |

### L'editor dei box (#11 e #13)

I due box si correggono direttamente sull'immagine: trascinamento per spostare, 8 maniglie per
ridimensionare, e in alternativa slider più campo numerico per ciascun lato, con la misura in px
e l'avviso quando `#11` supera il limite ESI.

Per il solo rettangolo ci sono due slider in più, **margine sx/dx** e **margine alto/basso**,
espressi in **percentuale** del lato del rettangolo. Il margine **si aggiunge** al rettangolo
ecografico, dandogli più spazio sui due lati: a zero ciò che si salva è esattamente il rettangolo,
sopra zero si salva un box un po' più grande. La percentuale invece dei pixel serve a far reggere
la stessa impostazione a un cambio di risoluzione.

Chi decide cosa: il **rettangolo ecografico** è la base (proposta dalla rete, correggibile a mano),
il **margine** è additivo, e in `#11` finisce la somma dei due. Nel progetto restano memorizzati
separatamente (`rect_echo` e `margin_percent`), così la provenienza non si perde e il margine si
può cambiare senza rifare il rettangolo.

Tre box, sempre tutti visibili:

| box | colore | modificabile |
|---|---|---|
| originario, proposta della rete | grigio tratteggiato | no, è il riferimento |
| rettangolo ecografico | arancione | sì, trascinamento / maniglie / slider |
| con margine, quello che finisce in `#11` | verde tratteggiato | no, è calcolato; compare solo se un margine è > 0 |

E le tre misure sotto l'immagine:

```
originario (proposta rete):  1227 x 857 px (top 146 left 296 bottom 1003 right 1523)
rettangolo ecografico:       1227 x 857 px (top 146 left 296 bottom 1003 right 1523)
verrà salvato in #11 con margine 5% / 3%: 1349 x 909 px (top 120 left 235 bottom 1029 right 1584)
```

Il controllo del limite ESI 1450×820 si applica al box **con margine**, perché è quello che ESI
disegna. Se il margine sbatte contro il bordo dell'immagine viene tagliato, e la nota lo dice.

Lo schermo intero si apre col pulsante o con un **doppio clic sull'immagine**.

**Schermo intero**: la stessa immagine su tutta la finestra, con `‹ precedente` / `successiva ›`,
il contatore, le frecce della tastiera e `Esc`. Serve a verificare su più immagini che il
rettangolo vada bene per tutte.

**I box sono di cartella, non di immagine**: qualunque modifica, fatta in piccolo o a schermo
intero, si salva una volta per tutte le immagini.

**Modifiche reversibili**, tre livelli:

| azione | cosa fa |
|---|---|
| badge "modifiche non salvate" | appare appena tocchi un box, così non si perde una correzione cambiando step |
| Annulla le modifiche | torna all'ultimo salvataggio |
| Valori iniziali della rete | torna alla proposta dell'analisi, conservata in `analysis.rect.rect_echo` e `analysis.line13.box` |

### Semplificazioni UX rispetto al legacy

1. **Un import invece di dieci**: nessun file dialog per pagina.
2. **Un solo canvas** per tutti i tracciamenti, con zoom, pan, nudge da tastiera e snap ai bordi.
3. **"Applica a tutte le depth"** accanto all'"Applica a questa depth" del legacy.
4. **Barra di stato del progetto** con semaforo per step e "vai al primo problema".
5. **Immagini come griglia di miniature con stato**, non liste di path.
6. **Ogni box tracciato a mano entra nella banca template** e nel feedback: la configurazione successiva
   dello stesso ecografo parte già risolta.

## 7. Contratto di step

Ogni step espone: `proposta` (valore + confidenza + provenienza), `azioni` (conferma, correggi, traccia
a mano, salta con motivo), `dipendenze` (cosa invalida a valle), `esito` (`ok` / `review` / `blocked`).
Uno step `blocked` non impedisce di procedere: marca il progetto come non generabile e lo elenca nella
pagina finale.

## 8. Soglie e metodo di match — **deciso**

I requisiti chiedono di eliminare la fase manuale delle thresholds. Va distinto: le soglie **non
spariscono dal file** — sono il campo `TH` dentro `CH:TH:P1..P5` di ogni riga con template. Sparisce
l'inserimento manuale.

### Parametri di default (tutte le righe con template: #13 #14 #15 #16 #17 #24)

| Campo | Valore | Nota |
|---|---|---|
| `CH` | `7` | match su tutti i canali |
| `MM` | `6` — `FE_TH_METHOD`, euclidean distance after threshold | |
| `P1` | `2.000000e+01` (20) | sotto 20 la dinamica si azzera |
| `P2` | `1.200000e+02` (120) | sopra 120 la dinamica si azzera; fra 20 e 120 è massimizzata |
| `P3` `P4` `P5` | `0.000000e+00` | non usati |
| `TH` | calcolato | vedi sotto |

Questi valori sono la convenzione di casa: su 10 `.fss` legacy in `templates/DB_setup` la coppia
`P1=20, P2=120` compare in **621 blocchi su ~640**, con `CH=7` dominante. Metodi diversi (`MM=0`
`CV_TM_SQDIFF`, `MM=5`, `MM=7`) restano possibili solo come eccezione manuale su una singola riga.

### Calcolo automatico di `TH`

`TH` non si digita mai nel flusso normale: si deriva dagli score di match che gli step precedenti già
producono, separando le immagini in cui il template **deve** essere trovato da quelle in cui **non deve**
esserlo (è la stessa logica delle liste "Immagini Positive / Negative" della pagina legacy
`wdgpagefssthresholds`, tolto il lavoro a mano).

- Con i metodi a distanza euclidea (`MM = 0, 6, 7`) il match è **sotto** soglia: `TH` va fra il massimo
  score sui positivi e il minimo sui negativi. Con `MM = 5` (correlazione) il verso è invertito.
- Il **margine** fra le due popolazioni è la misura di affidabilità della soglia e viene registrato.
- Margine nullo o negativo (positivi e negativi si sovrappongono) ⇒ lo step va in `review`: l'app chiede
  un'immagine in più o un template più piccolo, e solo lì è possibile forzare il valore a mano.
- L'ordine di grandezza atteso è quello dei legacy (`TH` ~ `1e6`–`1e8` per SQDIFF su template piccoli):
  un `TH` fuori scala rispetto ai `.fss` dello stesso vendor è un segnale di errore, non un valore valido.

La verifica finale delle soglie avviene nella passata sulle immagini rinominate dello step 10: se un
template viene trovato dove non deve, la soglia si ricalcola con quell'immagine aggiunta ai negativi.

## 8-bis. L'output non è solo il `.fss`: sono anche i ritagli

Una configurazione legacy non è un file, è un file **più una cartella di template**. In
`OldSoftwareEsiBuilder/templates/DB_echo/setup_43/` ci sono:

| file | riga | cos'è |
|---|---|---|
| `orientation_0..3.png` | #16 | il glifo di orientamento, uno per gruppo (30×31 px in setup_43) |
| `echo_name.png` | #13 | il logo dell'ecografo |
| `probe_name.png` | #14 | la sigla della sonda |
| `depth_0..N.png` | #17 | l'etichetta di ogni depth |
| `proibited_screen_*.png` | #15 | le schermate proibite |

Le righe del `.fss` dicono **dove** cercare e con quale soglia; questi PNG sono **cosa**
cercare. Da qui viene, fra l'altro, la banca marker del progetto: è stata costruita raccogliendo
gli `orientation_*.png` delle configurazioni storiche (per `marker_027.png` il manifest indica
`SSD_esi1_n1/Esaote_MyLab 9eXP_F080101_TLC3-13_L_Rev.2/DB_echo/setup_26/orientation_0.png`).

Conseguenza per l'app: ogni stadio che propone un box deve anche saper **ritagliare** il glifo
dall'acquisizione, e la generazione deve scrivere la cartella dei template accanto al `.fss`.
Oggi il ritaglio dell'orientamento è visibile nel visualizzatore (endpoint
`/orientation/crop`), ma **la scrittura della cartella dei template non è ancora implementata**.

Punto aperto — **da discutere con Francesca** (vedi sezione 15): con quale dimensione si
ritaglia. Il box del detector su questa cartella è 17×17 px, il template della banca che ha
fatto match è 18×18, e il legacy `setup_43` usa 30×31. Va deciso se si salva il box esatto, il
box con un margine fisso, o una dimensione per setup.

## 8-ter. Orientamento: il box porta l'informazione, la soglia conferma

Misurato sui dieci setup legacy e sulla cartella reale, e cambia il disegno dello stadio.

**La tolleranza di ESI e' larghissima.** Convertendo i `TH` delle righe `#16` in differenza media
per pixel (`TH` e' una somma di differenze quadratiche sull'area del template): da **41 a 92
livelli di grigio** su 255. In `setup_53` il primo blocco usa perfino `CV_TM_CCOEFF_NORMED` con
soglia **0.368**. Ogni gruppo ha il suo `TH`, tranne nei due setup a orientamento singolo dove
sono identici.

Con una soglia cosi' lenta, quello che rende affidabile il match legacy **non e' la soglia, e' il
box**: si cerca solo dove il marker sta davvero, e la soglia conferma che qualcosa di simile c'e'.

**Verifica sulla cartella reale.** Il nostro ritaglio 13x14, cercato su tutto il rettangolo,
cadeva su rumore a 0.64-0.74 nei 29 casi del lato destro. Cercato **dentro l'envelope del
gruppo** trova il marker in **23 su 29**, con punteggi 0.956-0.999 (mediana 0.959).

### Il disegno che ne segue

Due fasi con ruoli diversi, e due fonti diverse:

| fase | scopo | cosa si usa | cosa conta |
|---|---|---|---|
| **scoperta** | costruire i quattro envelope di `#16` | banca storica **e** ritaglio della cartella, unione delle detection sul rettangolo | il richiamo: se il box non contiene il marker, ESI non lo trovera' mai |
| **validazione** | prevedere cosa fara' ESI | solo il ritaglio da consegnare, cercato **dentro ogni envelope** | la precisione: e' il numero che predice il comportamento a runtime |

La soglia `TH` di `#16` si deriva dalla distribuzione della fase di validazione, per gruppo —
come faceva il legacy, che ha un `TH` diverso per ognuno dei quattro box.

Il ritaglio da consegnare si scegle per **copertura sui quattro gruppi** misurata in fase di
validazione, non per punteggio singolo: sui candidati di questa cartella la copertura andava dal
**40%** (ritaglio 21x20, il migliore per punteggio singolo) al **91%** (13x14), a parita' di
innesco.

### Il bug che ha falsato le prime conclusioni (trovato il 2026-09-03)

`_load_gray_cached` del bundle restituisce **`(width, height, gray)`**; nei moduli dell'app era
scompattato come `height, width, gray` — in **cinque punti**. Su un frame 1920x1200 la larghezza
diventava 1200, quindi **tutto cio' che stava a destra di x=1199 veniva tagliato via** da ogni
area di ricerca.

Conseguenze, tutte spiegate da questa sola riga:

* la correzione col click **non funzionava sulla meta' destra** dell'immagine: la finestra
  attorno al punto usciva "fuori dall'immagine" e l'utente vedeva un errore mentre stava
  cliccando esattamente sul marker;
* la copertura di LR e LRUD risultava al 75% e 84% invece del 100%;
* l'envelope del nostro ritaglio si fermava a **x=1199** contro i 1487 del batch — il numero
  tondo che avrebbe dovuto insospettirmi subito;
* da qui la conclusione sbagliata che "un solo ritaglio non copre tutta la cartella" e che sul
  lato destro avesse ragione la banca.

Dopo il fix, il solo ritaglio della cartella: **378/378 immagini agganciate**, mediana 0.988,
10° percentile 0.955, ed envelope a **3 px** dal batch su NF, UD e LRUD e a 28 px su LR
(destra 1485 contro 1487).

Lezione: la conversione delle coordinate del click e' stata spostata **sul server**, che apre
il file e conosce la dimensione vera; il client manda solo dove ha cliccato e quanto e' grande
l'immagine sullo schermo. Un'assunzione in meno da sbagliare.

### Esito dell'applicazione (2026-09-02)

Lo stadio orientamento e' stato riscritto su questo schema. Sulla cartella reale:

| | prima (banca su tutte le immagini) | adesso (scoperta + validazione) |
|---|---|---|
| ritaglio consegnato | nessuno | 13x14 px, innesco `marker_043` a scala 0.75 |
| copertura complessiva | non misurata | **100%** (soglia 0.55) |
| copertura per gruppo | non misurata | NF 100% · LR 100% · UD 100% · LRUD 100% |
| mediana dei punteggi | punteggi sparsi, 24 in review | 0.957 - 0.999 |
| envelope | dalla sola banca | identici, ma da 228-310 marker invece di 78-106 |

La soglia di validazione e' **0.55**, non 0.85: la tolleranza legacy misurata e' molto piu'
lenta, e con 0.85 quattro immagini su venti del lato destro risultavano scoperte pur avendo il
marker al posto giusto.

La pagina mostra ora il **PNG che verra' consegnato** e la tabella della copertura per gruppo,
colorata: sotto il 95% quel gruppo e' a rischio su ESI. E' il cancello di qualita' che prima
non esisteva ne' nell'app ne' nel vecchio flusso.

Resta da fare: derivare il `TH` di `#16` per gruppo da questa distribuzione (il legacy ha un
`TH` diverso per ciascuno dei quattro box) e scrivere i quattro PNG identici in
`DB_echo/setup_NN/`.

## 8-quater. Perche' il suggerimento umano vince, e cosa se ne impara

Misurato il 2026-09-03, sei ritagli presi da immagini sparse nella stessa cartella:

| ritaglio | dimensione | mediana sul campione | vittorie |
|---|---|---|---|
| cand 0-2 | 21x20 | 0.828 · 0.828 · 0.827 | 9 · 6 · 2 |
| cand 3-5 | 13x14 | 0.995 · 0.996 · 0.997 | 11 · 14 · 18 |

Le due famiglie sono **coerenti al loro interno**: non sono due rese diverse del glifo, e' la
**inquadratura del ritaglio**. Un crop di 21x20 attorno a un glifo di ~13 px porta dentro sfondo
che cambia da immagine a immagine, e lo sfondo abbassa la correlazione; un crop stretto contiene
quasi solo il glifo e correla a 0.99 quasi sempre.

Da qui la risposta alla domanda "perche' il mio suggerimento fa un punteggio molto piu' alto":
il click dell'utente da' la posizione esatta, e il ritaglio tagliato li' e' **stretto sul glifo**.
Non e' che il modulo sia cieco: e' che la sua inquadratura, ereditata dalla dimensione del
template della banca che ha fatto l'innesco, puo' essere piu' larga del glifo.

**Il margine di miglioramento automatico e' quindi l'inquadratura, e si impara dai dati**: fissata
la posizione, si provano alcune rifilature del riquadro (da 0 a 4 px per lato) e si tiene quella
che massimizza la mediana sulla cartella. La misura dice che vale da **0.83 a 0.99**, cioe' un
margine molto piu' ampio sotto la soglia `TH` che dovremo scrivere in `#16`.

Nota: dentro la famiglia stretta nessun ritaglio domina (11/14/18 vittorie su 60) e il guadagno
del migliore sulla mediana dei candidati e' 0.086: i ritagli ben inquadrati sono quasi
equivalenti, quindi la scelta fra loro conta poco. Conta l'inquadratura.

Componente nuovo, il pezzo che mancava del tutto (la pipeline si ferma a CSV e preview).
Vive in `tools/app/fss_writer.py`. Regole dei separatori, misurate sul corpus legacy:

- un blocco template e' `TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|`, con la coda `BM|TH2|`
  **solo** su `#17`;
- un gruppo di blocchi e' unito da `;` e **chiuso** da un `;` finale (`#16`, `#24`);
- le righe multi-gruppo (`#15`, `#17`) chiudono **anche l'ultimo gruppo** con `,`: da qui il `;,`
  finale che si vede nei file legacy;
- i vettori (`#18`, `#19`, `#20`, `#23`) sono uniti da `|` e chiusi da `|`;
- coordinate come interi, parametri di match in notazione scientifica a sei decimali
  (`4.748910e+07`), tutto il resto `%g`.

Verifica: `tools/app/selftest_roundtrip.py` rilegge un `.fss` legacy, lo rigenera con il writer e
confronta riga per riga. **Tutti e 10 i file di `templates/DB_setup` tornano identici**, biplani
compresi. E' la prova che il dialetto e' quello giusto.

Sulla validazione: nel corpus legacy **nessun file ha righe vuote**, quindi ogni riga mancante e'
segnalata come problema bloccante prima della generazione, oltre ai controlli semantici (coerenza
fra numero di depth e `#19`/`#20`/`#21`/`#17`/`#22`, rettangolo dentro l'immagine campione e sotto
il limite ESI, 4 box di orientamento, `#25` obbligatorio sulle biplane).

## 8-quinquies. Correggere non deve far aspettare (03-09-2026)

Questo programma esiste per fare le configurazioni **piu' in fretta** del vecchio software.
Il primo giro di correzioni faceva l'opposto: un click apriva un lavoro a cinque stadi che
riprovava i ritagli su tutte le immagini della cartella, e fino alla fine non si poteva
correggere la successiva. Ora la correzione e' in **due tempi**.

**Tempo uno — il click, sincrono, 0.14-0.45 s misurati.** `refine_from_click` cerca la
posizione esatta in una finestra di 140 px, si salva la correzione, si ricalcolano gruppo
dell'immagine e i quattro envelope (minimi e massimi su dei box: microsecondi). La riga
diventa "corretta" con i valori nuovi e si puo' passare subito alla prossima.
Endpoint: `POST /orientation/quick_fix`.

**Tempo due — la cartella, in background, e coalescente.** Un solo lavoratore per progetto
riprova tutti i ritagli su tutte le immagini, rivalida, e decide quale ritaglio consegnare.
Se arrivano altre correzioni mentre lavora **non si accodano N passaggi**: si alza una
bandiera e alla fine ne riparte uno con tutte le correzioni dentro. Misurato: tre click
consecutivi hanno prodotto 2 passaggi invece di 4, e il secondo ha girato in 24 s su 378
immagini per 6 ritagli. Endpoint: `GET /orientation/consolidation`,
`POST /orientation/reprocess`.

La scrittura dello step passa da un lucchetto unico (`_write_orientation`): la
rielaborazione parte da una fotografia dello step, e senza serializzare la
lettura-modifica-scrittura riscriverebbe sopra le correzioni salvate mentre lavorava.

**Correzione in serie**: un interruttore, **spento per default**, tiene l'arma inserita e
salta alla prossima immagine dopo ogni salvataggio. Si chiude da sola dopo 12 correzioni o
45 s di pausa. Il default spento non e' prudenza astratta: con la serie accesa, dei click
involontari sull'immagine hanno salvato 14 correzioni in fila durante una prova automatica.
Un click di troppo deve poter salvare **una** correzione sbagliata (annullabile), non una
catena.

## 8-sexies. Perche' le immagini in revisione non guarivano mai da sole (03-09-2026)

Trovato lavorando sulla velocita', ed e' la ragione strutturale per cui ogni immagine in
revisione andava corretta a mano: `validate_in_envelopes` saltava le immagini **senza
gruppo**, e senza gruppo sono esattamente quelle che il modulo ha messo in revisione
(`_rebuild_orientation` assegna il gruppo solo alle righe con stato `ok`). Nessun ritaglio,
nessun suggerimento umano poteva raggiungerle: restavano fuori dalla validazione per
costruzione.

La correzione e' anche la modellazione giusta: se il gruppo non si sa, si cerca in **tutti
e quattro gli envelope** e vince il migliore — l'envelope vincente *e'* la risposta. E'
esattamente cio' che fara' ESI a runtime.

Misura sulla cartella Esaote MyLabSigma TLC3-13 (378 immagini con marker, 2 correzioni):

| | prima | dopo |
|---|---|---|
| immagini validate | 359 | **378** |
| sotto soglia 0.55 | 19 | **0** |
| gruppo invariato sulle altre | — | **359/359** |
| copertura per gruppo | 1.0 su tutti | 1.0 su tutti |

Le 19 recuperate passano da 0.40-0.53 a **0.986-0.999**, e in **19 casi su 19** l'envelope
vincente coincide col gruppo che il batch aveva predetto: la ricerca sui quattro box non
inventa gruppi, li conferma. Zero regressioni: il ramo delle immagini con gruppo e' lo
stesso codice di prima, cerca nello stesso envelope.

Di conseguenza "da rivedere" nell'interfaccia vuol dire una cosa sola: **la configurazione
che stiamo per consegnare non aggancia questa immagine** (punteggio della validazione sotto
soglia), non il dubbio che il batch aveva prima. Su questa cartella il filtro segna 0.

## 8-septies. Quando il marker scelto e' quello sbagliato (03-09-2026)

Prima cartella GE provata (`16.GE_LogiqS7_ML6-15`, 146 immagini): lo stadio orientamento
finiva in errore per un bug (`orientation_value` letta prima di essere assegnata in
`_run_advanced_stages`), e una volta corretto quel bug il risultato era **sbagliato in modo
istruttivo**: un solo gruppo NF per tutte le 136 immagini, punteggio **identico 0.6106** su
tutte, marker a `9|400|25|427` — sopra il rettangolo ecografico (`top` 117). Si era
agganciato alla scritta "RGP" dell'intestazione GE, che non si muove mai.

Il marker vero, trovato col diff fra due orientamenti, e' il blocco di testo **"LS7 Exp"**
dentro l'area ecografica, e le sue quattro posizioni danno i gruppi giusti:

| cartella | posizione | lato+verticale | gruppo |
|---|---|---|---|
| `NoFlip` | 361,277 | sinistra+su | NF |
| `FlipLR` | 1159,309 | destra+su | LR |
| `FlipUD` | 361,446 | sinistra+giu | UD |
| `FlipLR_UD` | 1159,446 | destra+giu | LRUD |

Da qui tre cose, tutte nel prodotto.

**1. Diagnosi automatica (`_marker_warning`).** Un marker di orientamento *deve* muoversi.
Se il candidato cade sempre nello stesso punto, o ha punteggio identico su tutte le
immagini, o assegna un solo gruppo a tutta la cartella, quello che ha trovato e' un
elemento fisso dell'interfaccia. L'interfaccia lo dice in una barra gialla, col pulsante
"Indica il marker giusto".

**2. Marker indicato a mano (`POST /orientation/marker_override`).** L'utente trascina un
rettangolo attorno al glifo vero e il modulo riparte **senza passare dalla banca** — se la
banca ha scelto il glifo sbagliato, ripescarla riporterebbe lo stesso errore. Da quel
ritaglio si rifanno posizioni, gruppi, envelope e validazione. L'override si salva nel
progetto (`marker_override`), quindi **sopravvive al "Ricalcola i tre moduli"**, e le
correzioni per immagine fatte prima vengono scartate dicendolo: indicavano dov'era un glifo
che non stiamo piu' cercando. Il ritaglio della cartella entra anche fra i template di
`refine_from_click`, cosi' le correzioni successive cercano il marker giusto.

**3. Due passaggi, e i falsi match fuori dal rect.** La ricerca libera (rect + 60 px)
prendeva, su un'immagine, un falso match sotto il rettangolo che vinceva 0.649 contro 0.63
del marker vero. Due regole lo risolvono: la **riassegnazione** dei gruppi cercando dentro
i quattro envelope (quello che fara' ESI), e lo **scarto dei match fuori dal rect quando
sono l'eccezione** (soglia: se >= 80% cade dentro, quelli fuori sono falsi; se la
maggioranza sta fuori non si butta niente, come nei setup storici 80 e 81).

Esito sulla cartella GE, verificato contro i nomi delle sottocartelle, che sono la verita':

| | prima | dopo |
|---|---|---|
| gruppi trovati | NF | **NF LR UD LRUD** |
| immagini nel gruppo giusto | 1 su 124 (tutte NF) | **124 su 124** |
| copertura per gruppo | 0.2721 | **1.0 su tutti e quattro** |
| ritaglio consegnato | 15x10 dal banner | **33x19 sul glifo** |

Il rilancio costa 9 secondi su 136 immagini. Sulla cartella Esaote nessuna regressione:
copertura 1.0, zero immagini da rivedere, quattro gruppi.

## 8-octies. Chi fissa i bordi degli envelope (03-09-2026)

Un envelope e' l'unione dei box del gruppo: per ogni lato **una** immagine lo ha spinto fin
la'. Saperlo serve a due cose diverse: capire perche' un box e' grande (informazione utile,
il marker si muove davvero), e accorgersi quando invece e' una detection sbagliata a
gonfiarlo — sulla cartella GE un falso match sotto il rect allargava l'envelope LRUD di
200 px e si portava dietro il gruppo assegnato a un'altra immagine.

`GET /orientation/limits` restituisce, per gruppo e per lato: la coordinata, **il secondo
valore** (dove finirebbe il lato senza chi lo fissa), quindi di quanti pixel si
stringerebbe, quante immagini condividono quell'estremo, la distanza dal centro mediano del
gruppo, e i due punteggi — quello della **scoperta**, che ha deciso la posizione, e quello
della **validazione**, che e' il numero mostrato accanto all'immagine nel resto della
pagina. Sono diversi per costruzione: la scoperta cerca in tutto il rect con match
ridimensionato, la validazione cerca dentro l'envelope a piena risoluzione.

Nell'interfaccia: una scheda per gruppo con la tabella dei quattro lati, il nome
dell'immagine cliccabile (porta subito a guardarla), e un interruttore nella legenda che
**disegna sull'immagine i quattro marker che fissano i bordi** del gruppo corrente,
etichettati col lato e la coordinata.

**Quando segnalare.** Non basta "unica e distante": il marker si muove con la depth, e un
estremo trovato a 1.0 e' una posizione vera che sta al bordo. La firma del falso match e'
un estremo **isolato, che pesa (>= 15 px) e scoperto male (< 0.75)**. Verificato:

- GE `prova`: un solo avviso su sedici lati — LRUD alto, 42 px, confidenza 0.662, immagine
  singola. Ed e' l'unico caso che vale un occhio.
- Esaote: nessun avviso. I quattro estremi (35 px ciascuno) sono tutti immagini
  `depthmin_aperturamin`, scoperte fra 0.81 e 0.85 e validate fra 0.94 e 0.99: posizioni
  vere di una depth estrema, che l'envelope deve coprire.

## 8-nonies. Correggere un vertice deve stringere il box (03-09-2026)

Difetto trovato da Camilla: corretto un marker che faceva da vertice in un envelope Esaote,
il box non si stringeva. La causa: la correzione veniva **aggiunta** alle detection del
modulo invece di **sostituire** quella della sua immagine, quindi la posizione vecchia
restava nell'unione e continuava a tenere aperto il lato.

Regola, ora in tutte le strade che ricalcolano gli envelope (click veloce, rielaborazione in
background, correzione manuale del box, ricalcolo dello stadio, annullamento): **una riga per
immagine, e la correzione vince**. Ne segue gratis la seconda richiesta: poiche' quella riga
porta con se' anche il gruppo, una correzione che cambia raggruppamento sposta il marker fra
i gruppi — quello che lo perde si stringe, quello che lo prende si allarga.

Verificato sulla correzione vera di Camilla (`DEPTHSX_ZOOM1_208`, spostata da LR a NF):

| | prima | dopo |
|---|---|---|
| LR | `166\|966\|569\|1487`, 103 marker | **`166\|966\|542\|1487`, 102 marker** (lato basso −27 px) |
| NF | `172\|334\|542\|855`, 98 marker | `172\|334\|542\|855`, **99 marker** (il box corretto era gia' dentro) |

I 27 px sono esattamente il "secondo" che la scheda dei bordi indicava per quel lato.

**Zero regressioni, misurate.** Passando da due box per immagine a uno (nel ricalcolo dello
stadio il ritaglio della cartella aggiungeva un secondo box anche per le immagini che la
banca aveva gia' trovato) gli envelope della cartella Esaote sono **identici in tutte e
quattro le coordinate** — cambia solo il conteggio dei box, che si dimezza (NF 202 -> 102,
LR 210 -> 110, UD 155 -> 80, LRUD 165 -> 86) — e la validazione resta a copertura 1.0 su
tutti i gruppi. Il ritaglio continua a riempire i buchi dove la banca non arriva (24 immagini
su questa cartella), solo non raddoppia le altre.

Corretto insieme un difetto vicino: l'annullamento di una correzione ricostruiva gli envelope
dalle righe del batch anche su un progetto col **marker indicato a mano**, cioe' riportava
dentro le posizioni del glifo sbagliato.

## 8-decies. La sezione orientamento, impaginata compatta (03-09-2026)

Stesse funzioni e stesse informazioni di prima, in meno spazio e con l'immagine subito
raggiungibile. Prima la pagina impilava: un paragrafo di istruzioni, filtri, ordine,
navigazione, legenda, un secondo paragrafo, quattro barre, l'immagine, il dettaglio, tre
schede, l'elenco delle confidenze, **un'altra copia** di filtri e ordine, altre tre schede.

Adesso:

- **una riga di stato** con i conteggi e le istruzioni dentro un «come si corregge»
  richiudibile;
- **una barra sola** con i filtri come pastiglie (col conto, che si rifa' dopo ogni
  correzione: prima era fermo al primo disegno), l'ordine come menu' a tendina invece di tre
  pulsanti, le frecce di navigazione e il conto delle immagini nel filtro;
- **una barra delle azioni**: correggi, in serie, marker sbagliato, vertici;
- **due colonne**: a sinistra l'immagine con, sotto, didascalia, dettaglio, legenda e il
  ritaglio del marker corrente; a destra i **bordi degli envelope** in vista compatta e
  l'**elenco delle confidenze**, ognuno col suo scorrimento e insieme alti esattamente come
  l'immagine (l'altezza la calcola il JS sull'immagine caricata). Si passa da un'immagine
  all'altra senza muovere la pagina, e la riga corrente si porta in vista da sola;
- **in fondo, cinque pieghe** (`<details>`) con l'etichetta che porta il numero utile: bordi
  («1 da guardare»), correzioni («3»), ultima rielaborazione («10 migliorate»), ritaglio
  consegnato e copertura, confronto dei ritagli.

La vista compatta dei bordi tiene una riga per lato — lato, coordinata, quanto si
stringerebbe, confidenza della scoperta, immagine — cliccabile per andare a quell'immagine,
evidenziata quando e' quella mostrata, in giallo quando e' un estremo sospetto. Il dettaglio
completo (secondo valore, confidenza della validazione, distanza dalla mediana, quante
immagini condividono l'estremo) sta nel tooltip di ogni riga e nella tabella della piega.

## 8-undecies. Il ritaglio mostrato e quello vero (03-09-2026)

Camilla: «cos'e' "ritaglio corrente", non mi sembra un template aggiornato». Aveva ragione:
il rettangolo bianco e il ritaglio venivano dal box della **scoperta**, mentre il punteggio
accanto era quello della **validazione** — due passaggi diversi, che sulle immagini agganciate
da un suggerimento cadono in punti diversi (misurato: scoperta `286|1144`, validazione
`204|1143`). Ora c'e' una sola funzione (`_current_marker_box`) che decide qual e' il box
corrente, con la stessa priorita' per il rettangolo disegnato, il ritaglio mostrato e quello
che va in banca: **correzione dell'utente, poi validazione, poi marker indicato a mano, poi
scoperta del batch**. Le etichette del confronto dicono cosa e' cosa: il template della banca
e' «il riferimento che ha fatto match, non il glifo vero», il ritaglio dell'immagine e' «il
glifo vero, tagliato dove lo vedi riquadrato», e quello consegnato e' «questo finisce in
DB_echo».

## 8-duodecies. Versare il ritaglio nella banca del vendor (03-09-2026)

La banca `lr_marker_vendor_template_library/<Vendor>/marker_NNN.png` e' costruita dagli
`orientation_*.png` delle configurazioni storiche. Un ritaglio nato da questa app e' la
stessa cosa — il glifo vero di un'acquisizione — quindi puo' entrarci, ed e' il modo in cui
una configurazione fatta a mano insegna qualcosa alla prossima: sulla cartella GE il modulo
ha sbagliato **perche' quel glifo non era in banca**.

`GET /orientation/bank` dice cosa c'e' (vendor, quanti template, che nome avrebbe il
prossimo, se il ritaglio e' gia' presente); `POST /orientation/bank` lo versa. Dettagli che
contano:

- il nome della cartella del vendor si risolve **senza badare a maiuscole e spazi**: la rete
  dice `GE`, la banca ha `Ge`;
- si scrive con la convenzione della banca: `marker_NNN.png` col numero successivo, piu' la
  voce in `<Vendor>/manifest.json` e la riga in `manifest.csv` (stesse colonne, stesso
  ordine), con in aggiunta `added_by`, `project_id`, `source_box` e `added_at` — cosi' si
  distingue un template nato qui da uno storico;
- **niente doppioni**: il digest SHA-1 del PNG si confronta con quelli in cartella e un
  secondo versamento dello stesso ritaglio viene rifiutato (409) dicendo sotto quale nome
  c'e' gia'.

Primo versamento vero: il glifo GE «LS7» di `prova` e' entrato come `Ge/marker_032.png`
(33x19 px, da `NoFlip_depth_10.png`). La prossima cartella GE LogiqS7 lo trovera' da sola.

## 8-terdecies. Tre schermate al posto di «Ecografo e sonda» (03-09-2026)

Il legacy aveva una pagina sola (`PAGE_US`) e qui erano tre cose impilate. Si confermano con
criteri diversi, quindi ora sono tre step:

| step | riga | cosa si conferma |
|---|---|---|
| **Ecografo e vendor** | `#13` | il vendor riconosciuto e il riquadro col nome della macchina a schermo |
| **Sonda** | `#14` | ID sonda dalla rete, modello e tipo dall'anagrafica, riquadro con la sigla |
| **Rettangolo ecografico** | `#11` | un solo rettangolo per tutta la cartella, coi margini percentuali |

Dipendenze rifatte di conseguenza: biplana dipende da **sonda**; schermate proibite,
orientamento e depth/scala dipendono dal **rettangolo**, che e' la geometria da cui discende
tutto il resto; le soglie da tutti e tre.

**Migrazione automatica**: un progetto scritto quando lo step era uno solo si divide al primo
caricamento (`_split_echo_probe`) e la divisione si scrive subito su disco. I quattro progetti
esistenti sono passati senza perdere niente: `#11`, `#13` e `#14` rigenerati identici, e il
round-trip dei `.fss` legacy resta byte-identico.

Nell'editor dei box compare anche `rect_name_probe` (#14), che prima non era disegnabile: la
sonda ha ora il suo riquadro trascinabile sull'immagine, con un pulsante che lo crea quando
non esiste ancora (il resolver storico avrebbe bisogno del corpus legacy `.fss`, oggi non
disponibile).

### Il rettangolo: cosa c'e' oggi e cosa manca

Oggi il rettangolo viene dalla rete (`rect_training_e40_run2`, con routing per vendor sopra
confidenza 0.70): mediana dei box su un campione di immagini, piu' l'accordo (IoU) fra le
immagini come misura di fiducia. Uno solo per la cartella, applicato a tutte.

Cercato nel repository l'algoritmo con la **verifica di specularita' sui quattro orientamenti
usando le diagonali**: non c'e'. Cercato in `tools/`, `docs/`, `preparazione al progetto/`,
nel vecchio ESIBuilder C++ e nella storia git (`git log -S` su specular/simmetr/diagonal/
ribaltat). Quello che gli somiglia di piu' e' `tools/ultrasound/rect_red_reusable.py` +
`rect_red_pipeline.py`: un rettangolo unico di cartella ottenuto come mediana dei rect per
gruppo, con l'ampiezza orizzontale fissata dal **segmento superiore piu' largo** trovato in
tutta la cartella piu' un margine percentuale — ma **senza** nessun controllo di
specularita'. Da chiarire con Camilla dove vive quell'algoritmo prima di riscriverlo.

## 8-quaterdecies. La catena del rettangolo ecografico (03-09-2026)

I moduli si tengono per mano, e sembra un cerchio: l'orientamento si studia **dentro** il
rettangolo, ma il rettangolo si rifinisce **con** l'orientamento; il piano L/T si riconosce
dentro il rettangolo e serve al rettangolo per cercare il segmento piu' lungo nel piano
giusto. Non e' un cerchio vizioso se lo si percorre **a giri**: un rettangolo provvisorio
basta a far girare orientamento e piano, e con quelli il rettangolo si stringe. Ogni giro
**propone**, mostra i numeri, e applica solo se l'utente conferma.

| giro | serve | cosa fa | stato |
|---|---|---|---|
| 1 · rete | — | mediana dei box su un campione | fatto |
| 2 · specularita' | orientamento | un rettangolo per gruppo, confronto fra i quattro, consenso | **implementato** |
| 3 · segmento piu' lungo | orientamento + piano L/T | la corda piu' larga del ventaglio nel piano giusto fissa l'ampiezza | da scrivere |
| 4 · incrocio con la depth | + depth e scala | chiude il rettangolo in verticale | da scrivere |

`GET /rect/chain` dice cosa e' pronto e cosa aspetta cosa; `POST /rect/refine` lancia un
giro (come job, con avanzamento); `POST /rect/apply` applica la proposta e registra lo
storico. Applicare fa tornare in review gli step che dipendono dalla geometria, come deve.

### Cosa ha detto la misura, e perche' cambia la domanda

Il giro 2 non assume quale sia l'invariante: misura **entrambe** le ipotesi — i quattro
rettangoli identici fra loro, oppure speculari rispetto al centro dell'immagine — e riporta
quale tiene.

| cartella | scarto se identici | scarto se speculari | conclusione |
|---|---|---|---|
| GE LogiqS7 (1280x1024) | 13-19 px | 89-109 px | **identici** |
| Esaote MyLabSigma (1920x1200) | 1-3 px | 48-52 px | **identici** |

Quindi il rettangolo **non si ribalta col contenuto**: e' un riquadro fisso di schermo, lo
stesso per tutti e quattro gli orientamenti. La macchina ribalta l'immagine dentro un'area
che resta dov'e'.

La specularita' che conta e' allora un'altra, ed e' quella del requisito sulla scala: **se
si ribalta la scala, ricade dentro lo stesso riquadro?** Solo se il rettangolo e' centrato
rispetto all'immagine. Misurato:

| | scostamento dal centro | spostamento se lo ribalti |
|---|---|---|
| GE LogiqS7 | dx 88 px · dy -77 px | orizzontale 176 px · verticale 154 px |
| Esaote | dx -48 px · dy -23.5 px | orizzontale 96 px · verticale 47 px |

E sulle **dieci configurazioni storiche** di `templates/DB_setup`, la riga `#11` non e'
centrata in nessuna: orizzontalmente lo scostamento va da 0 a 64 px (quattro setup su dieci
sono centrati entro 2.5 px), verticalmente da 20 a 151 px, sempre. Il ventaglio e' centrato
in orizzontale perche' la sonda sta in mezzo, ma in verticale parte sotto l'intestazione.

**Conseguenza operativa**: il ribaltamento sinistra-destra attorno al centro dell'immagine e'
legittimo (quattro storiche su dieci a scarto zero, sette entro 15 px), quello sopra-sotto
**non lo e' mai**: va fatto attorno al centro del *rettangolo*, non dell'immagine. E' il
controllo che il requisito del collega chiedeva («verificare se, anche da ribaltata, la scala
rimane esattamente sovrapposta»), ed e' ora una misura che l'app mostra invece di
un'assunzione.

### Giro 3: il segmento piu' lungo, e lo studio disegnato sul frame (03-09-2026)

Confermato da Camilla: il confronto di specularita' era fra i **segmenti**, non fra i
rettangoli. Il giro 3 cerca, gruppo per gruppo, la corda orizzontale piu' larga del
ventaglio — riusando `estimate_horizontal_segment_from_path` di `rect_red_pipeline.py`, non
riscrivendolo — dentro il **piano dominante** del gruppo, riconosciuto immagine per immagine
col modello L/T (per questo il giro non aspetta il piano: se lo calcola da solo).

Anche qui si misurano entrambe le ipotesi, e la risposta e' istruttiva:

| coppia | asse | stessa posizione | specchiate | tiene |
|---|---|---|---|---|
| NF-LR (GE) | x | **0.0 px** | 245.0 px | identita' |
| UD-LRUD (GE) | x | **1.2 px** | 216.2 px | identita' |
| NF-UD (GE) | y | 423.9 px | **216.3 px** | specularita' |

Sull'asse **x** le corde di NF e LR cadono nello stesso posto: il ventaglio e' simmetrico
attorno al proprio asse, quindi la sua corda piu' larga non si sposta quando il contenuto si
ribalta. Sull'asse **y** invece il ribaltamento si vede (216 px contro 424). E' la stessa
conclusione del giro 2, vista da un'altra parte: il ribaltamento sinistra-destra e'
innocuo, quello sopra-sotto no.

L'ampiezza proposta viene dall'**unione** delle corde piu' un margine del 2%. Sulla cartella
GE il rettangolo della rete tagliava a destra: proposta 296-1228 contro 285-1158, cioe' 70 px
in piu'. Su Esaote lo scarto e' 15 px.

**Conflitto di piano.** Su Esaote i gruppi cadono su piani diversi (NF/LR sul piano L,
UD/LRUD sul T) e le corde non sono confrontabili: 637 px contro 1016 px. La cartella contiene
**due configurazioni**, una per piano. Il giro lo dichiara (`plane_conflict`) e blocca il
pulsante di applicazione invece di proporre un numero senza senso.

**Lo studio si vede.** Nella sezione del rettangolo c'e' la scheda «lo studio
sull'immagine»: si sceglie il gruppo con le pastiglie e sul frame di quel gruppo compaiono il
rettangolo attuale (bianco), quello del gruppo (colore del gruppo), la proposta (verde
tratteggiata), la corda trovata (linea spessa) e la corda del gruppo speculare (tratteggiata,
col suo colore) — cosi' la specularita' si guarda invece di leggerla, e accanto restano i
numeri della coppia.

### Assi, intersezione e candidati: la scelta si fa guardando (03-09-2026)

`GET /rect/axes` misura l'asse di ribaltamento da **due feature indipendenti** e non lo
assume mai:

- dalle **corde**: punto medio della corda piu' larga per la x, punto medio fra la posizione
  «su» e «giu» della stessa corda per la y;
- dai **marker** di orientamento: punto medio fra i centri mediani delle coppie NF/LR e
  UD/LRUD (decine di immagini per gruppo).

L'accordo fra le due e' la fiducia nell'asse. Misurato: **7-14 px sulla x** (asse verificato)
e **42-49 px sulla y** (asse non affidabile). Per confronto, il centro dell'immagine dista
122 px dall'asse x vero su GE e 48 px su Esaote: non e' un riferimento.

Cinque candidati per `#11`, ognuno con la sua misura, lo scostamento dall'attuale e il
controllo del limite ESI: `attuale`, `consenso` (giro 2), `ampiezza` (giro 3),
`simmetrico_corde` e `simmetrico_marker` — questi ultimi allargano il lato corto finche' il
rettangolo e' simmetrico attorno all'asse, cosi' **ribaltarlo lo lascia dov'e', per
costruzione**. Sulla cartella GE costa 69 px in larghezza e 56 in altezza; su Esaote la
larghezza era gia' simmetrica entro 1 px su 1227 — la rete aveva trovato l'asse del ventaglio
da sola.

Nella scheda «lo studio sull'immagine» ogni livello si accende e si spegne (rettangolo
attuale, rettangolo del gruppo, corde, corda speculare, assi dalle corde, assi dai marker,
assi dell'immagine, centri dei marker) e ogni candidato si puo' **vedere sul frame** prima di
applicarlo: assi come linee con la loro etichetta, intersezione come cerchio, candidato come
riquadro tratteggiato.

**Conferma in due tempi.** `confirm()` non protegge: in questo ambiente un clic vagante lo
attraversa e ha applicato due volte un rettangolo che nessuno aveva scelto. Le azioni che
cambiano `#11` ora chiedono un secondo clic su un pulsante che prima non esisteva.

### Le corde si correggono a mano, e i conti si rifanno (04-09-2026)

La corda e' il dato di partenza del giro 3: se il detector la prende storta, tutto il resto
segue l'errore. Si corregge trascinandola sul frame — due maniglie agli estremi cambiano la
larghezza, quella centrale la sposta in alto o in basso — e `POST /rect/segment` la salva e
**rifa i conti che ne dipendono**: coppie speculari, ampiezza, proposta, assi e candidati.
Il ricalcolo non richiama nessuna rete (sono minimi, massimi e differenze), quindi risponde
subito; il pezzo comune sta in `_segments_analysis`, che il giro 3 e la correzione usano
uguale.

Provato allungando di 60 px la corda di NF sulla cartella GE:

| | prima | dopo | dopo l'annullamento |
|---|---|---|---|
| corda NF | 880 px | 940 px | 880 px |
| coppia NF-LR | 0.0 px | **60.0 px** | 0.0 px |
| asse x dalle corde | 761.9 | 791.9 | 761.9 |
| accordo coi marker | 7.4 px | **37.4 px** | 7.4 px |
| candidato «ampiezza» | 296-1228 | 296-1279 | 296-1228 |

Si vede anche il valore della verifica incrociata: la correzione sbagliata peggiora subito
l'accordo con la feature indipendente (7 px -> 37 px), e il numero lo dice prima che venga
applicato qualcosa. L'annullamento riporta esattamente allo stato di partenza — ripristina la
geometria conservando il resto della riga (immagine, piano), cosa che nella prima stesura non
faceva.

### Pulizia della sezione rettangolo (04-09-2026)

Camilla: «ci sono troppe proposte ed elaborazioni sulle immagini, non capisco a cosa si
riferiscano». Aveva ragione: sul frame potevano finire dieci cose insieme (tre rettangoli,
due corde, quattro assi, sei croci, tre maniglie) e i candidati si chiamavano `consenso`,
`ampiezza`, `simmetrico_corde` — nomi che dicono come sono stati calcolati, non cosa sono.

- **Sull'immagine, di default, due cose sole**: il rettangolo di adesso e la proposta che
  stai guardando. Tutto il resto e' materiale di indagine e si accende quando serve.
- **Un interruttore mostra una cosa sola.** Il livello «corda del ventaglio» mostrava anche
  un rettangolo (la proposta del giro 3), che pero' e' gia' fra le proposte: due strade per
  la stessa cosa. Tolto.
- **Nomi in chiaro, col perche' sotto**: «Quello di adesso», «Media dei quattro
  orientamenti», «Allargato fino alla corda piu' larga», «Simmetrico sull'asse del
  ventaglio», «Simmetrico sull'asse dei marker». Ogni riga porta le coordinate, la misura,
  di quanti pixel cambia, un `vedi` che lo disegna e un `applica` in due tempi.
- **Avvisi al posto giusto**: un candidato che supera il limite ESI o che poggia su un asse
  non verificato lo dice sulla sua riga (`l'asse dei marker non concorda con quello delle
  corde: 42.8 px sulla verticale`), non in una nota generale.
- **Il «come» sta in due pieghe chiuse**: «assi di ribaltamento misurati» (con l'accordo fra
  le due feature nell'etichetta) e «come vengono calcolate le proposte» (i quattro giri, le
  misure, i dati del modello). In prima pagina resta la scelta, non il procedimento.

Corretti guardando davvero la pagina, cosa che prima non avevo potuto fare: l'immagine dello
studio non caricava (`name=undefined`, residuo del difetto dell'annullamento) e il riquadro
collassava a 0x0; gli overlay erano calcolati sulla larghezza dell'immagine **appena
caricata** (920 px) invece che su quella finale (384 px), quindi cadevano fuori posto di 2.4
volte — ora un `ResizeObserver` li ridisegna quando l'immagine cambia dimensione, qui e nel
visualizzatore dell'orientamento, che aveva lo stesso difetto latente. E le maniglie della
corda si accumulavano a ogni ridisegno.

### Vertici della corda e assi: cosa e' cosa (04-09-2026)

Camilla: «non sto capendo bene gli assi a quale elemento si riferiscono, alla corda?».
Risposta: l'asse non e' della corda, e' **dell'immagine** — la linea attorno a cui il
contenuto si ribalta fra un orientamento e l'altro. La corda serve a **misurarlo**: la
verticale passa per il centro della corda, l'orizzontale sta a meta' fra dove la corda cade
in «su» e dove cade in «giu». Reso esplicito in tre punti:

- le etichette sulle linee dicono cosa ribaltano e da dove vengono: `ribalta sx↔dx · dal
  ventaglio · 762`, `ribalta su↔giu · dal ventaglio · 404`;
- quando l'asse dal ventaglio e' acceso, il **centro della corda** e' marcato con un punto
  etichettato: si vede che l'asse verticale ci passa per costruzione;
- sotto l'immagine compare una riga per ogni asse acceso, che spiega da cosa e' ricavato e
  cosa significa se due assi non coincidono. Il centro dell'immagine e' dichiarato per
  quello che e': un riferimento per il confronto, non un asse di ribaltamento.

I nomi dei livelli seguono: «asse di ribaltamento (dal ventaglio)», «asse di ribaltamento
(dai marker)», «centro dell'immagine, per confronto».

**Vertici della corda, posizionabili al pixel.** Sull'ecografia in scala di grigi una
maniglia scura sparisce, e un pixel dell'immagine vale meno di mezzo pixel sullo schermo:
trascinare non basta. Ora le maniglie hanno nucleo bianco, anello colorato, una **tacca
verticale** che prolunga il punto sopra e sotto, e la **coordinata scritta accanto**. Si
selezionano con un clic e le **frecce della tastiera** le muovono di un pixel (shift: dieci).
Verificato: cinque frecce a destra spostano l'estremo da 1177 a 1182, e la riga sotto
l'immagine segue in tempo reale.

### Scorrere tutte le immagini sotto le elaborazioni (04-09-2026)

Le elaborazioni sono di cartella (un rettangolo, una corda per orientamento, un asse), le
immagini sono tante: adesso si scorrono **tutte** sotto gli stessi disegni. Sopra
l'immagine una barra con `‹` `›`, il conto («immagine 7 di 37 del gruppo NF»), il nome del
file e un pulsante «quella della misura» che riporta all'immagine su cui il modulo ha
misurato la corda — segnalata con un'etichetta quando la stai guardando.

Serve a controllare che corda, rettangolo e assi tengano su **ogni** immagine e non solo su
quella scelta dal modulo; e se correggi la corda mentre ne guardi un'altra, e' quella che
viene registrata come riferimento della misura (`image` nella correzione).

L'ordine della scheda ora e' deciso in un posto solo — titolo, gruppo, immagine, disegno,
legenda, numeri, correzione, spiegazione degli assi, livelli, proposte — perche' comporlo a
pezzi sparsi lo aveva gia' fatto uscire mescolato una volta.

### Il rettangolo costruito sugli assi (04-09-2026)

Costruzione decisa da Camilla, e ora implementata come proposta a se':

- **asse verticale** = il centro delle corde (mediana dei centri). Il rettangolo ci sta
  simmetrico attorno, quindi il margine a destra e quello a sinistra sono uguali **per
  costruzione**, non per fortuna;
- **asse orizzontale** = a meta' fra la corda **piu' alta** dei gruppi «su» (NF, LR) e la
  **piu' bassa** dei gruppi «giu» (UD, LRUD). Con margine zero quelle due toccano
  esattamente i bordi e tutte le altre stanno dentro;
- **mezze estensioni** = distanza dall'asse all'estremo piu' lontano, su entrambi gli assi;
- **due slider** per il margine, uno orizzontale e uno verticale, che si aggiunge uguale sui
  due lati. Il disegno segue lo slider senza tornare al server, perche' la costruzione
  arriva senza margine e il margine e' un'addizione.

Per trovare gli estremi servono tutte le corde, non la sola rappresentativa: il giro 3 ora
**conserva ogni corda misurata** (`all` per gruppo, 56 su questa cartella con 14 immagini per
gruppo) e la costruzione le usa tutte.

Sulla cartella GE: asse x 748.1 con mezza larghezza 459.4 (289-1208), asse y 383.2 con mezza
altezza 248.8 (134-632), 919x498 px a margine zero. La corda piu' larga e' di LR
(`FlipLR_depth_28`), la piu' alta di LR (`FlipLR_depth_40`, y 134.4), la piu' bassa di LRUD
(`FlipLR_UD_depth_70_trans`, y 632).

**Correggere una corda rifa il rettangolo.** La correzione entra anche nell'elenco delle
corde del gruppo, non solo nella rappresentativa, quindi gli estremi si ricalcolano.
Verificato alzando di 40 px la corda di LR: diventa lei la piu' alta (134.4 -> 109.4), l'asse
orizzontale si sposta (383.2 -> 370.7) e la mezza altezza cresce (248.8 -> 261.3); annullando
si torna esattamente al punto di partenza.

**Un numero da guardare**: i centri delle corde non coincidono fra loro — su questa cartella
ballano di 48 px. La simmetria costa quindi un po' di slack da un lato (a margine zero il
bordo sinistro sta 33 px piu' in la' della corda piu' a sinistra). Il numero e' scritto
accanto al conto delle corde, perche' se ballasse troppo vorrebbe dire che l'asse non e'
fidato.

## 10. Apprendimento progressivo

Ogni conferma e ogni correzione produce una riga in `feedback/inbox.jsonl` con contesto completo
(step, immagine, soglie, confidenze). `feedback_cli.py export` alimenta le code dei moduli. Le conferme
`ok` sono il set di regressione: il criterio di accettazione del progetto resta **zero regressioni**.

## 11. Requisiti non funzionali

- Windows, inferenza CPU, nessuna GPU richiesta.
- Rettangolo ecografico entro **1450×820** (limite schermo ESI). **Deciso**: risoluzione nativa per
  default, un solo sistema di coordinate; se il rettangolo proposto sfora, l'app calcola il fattore di
  scala, lo mostra e chiede conferma. Solo in quel caso l'immagine campione è ridimensionata e `#09/#10`
  differiscono da `#07/#08`; il progetto registra il fattore applicato e tutte le coordinate dei template
  vivono nel sistema dell'immagine campione.
- Configurazione robusta al cambio di risoluzione: l'app registra la risoluzione di configurazione e le
  posizioni template in modo che ESI possa riadattarle a una risoluzione diversa.
- Progetto riapribile e ripristinabile in qualunque momento; nessuna perdita di lavoro sui cambi di codici.
- Determinismo: stessa cartella, stessi modelli, stesso `.fss`.

## 12. Fuori perimetro v1

Acquisizione video dall'ecografo; modalità di adattamento sul campo su ESI (Ultra Service);
training dei modelli dentro l'app (resta un processo separato su Mac).

## 13. Punti aperti

Punti da decidere. Restano **aperti e vuoti**: nessun default provvisorio, si compilano quando la
decisione arriva. Nessuno di questi blocca la parte già decisa della specifica.

1. ~~Soglie di match~~ — **deciso**, vedi sezione 8: calcolo automatico, `CH=7`, `MM=6`, `P1=20`,
   `P2=120`, override solo su margine insufficiente.
2. **Versione in `#01`** — i `.fss` legacy in `templates/DB_setup` hanno 3.1/3.2/3.3 e il valore arriva
   dalla versione dell'app (`QFileFss::writeFile(version, ...)`), non da una costante; la pipeline oggi
   scrive `4.0`. Serve sapere quale versione l'ESI in campo accetta.
   Decisione: _da definire_
3. **Blocco linee guida e `.ndg`** — riuso di un `.ndg` esistente rinominato con tracciamento manuale
   legacy, creazione completa dentro l'app, oppure riuso più riconoscimento automatico delle linee.
   Decisione: _da definire_
4. **Verifica della scala ribaltata** — serve un controllo automatico che la scala speculare resti
   sovrapposta a quella ecografica, come chiedono i requisiti?
   Decisione: _da definire_
5. **Riferimento per il quality gate** — quale `.fss` legacy si usa come riferimento per un ecografo
   nuovo, mai configurato prima?
   Decisione: _da definire_
6. **Biplana** — una sessione genera i due `.fss` (L e T) insieme con `#25` incrociato, o si configurano
   come due progetti collegati?
   Decisione: _da definire_
7. **Template assente in `#13`/`#14`** — il requisito chiede di poter dire "template non presente", ma
   nei 10 `.fss` legacy nessuna riga e' mai vuota. Serve sapere cosa accetta ESI: una riga vuota, oppure
   un box segnaposto con `B` invertito.
   Decisione: _da definire_

## 12-bis. Il passo zero: una sola attesa, poi tutto è già proposto

L'utente indica la cartella e aspetta una volta. Dentro quel passaggio, in quest'ordine:

| stadio | come | esito misurato sulla cartella di prova |
|---|---|---|
| scansione + dedup esatta | size + SHA-1, in parità con la pipeline | 3128 → 666 immagini |
| (esito della dedup) | la lista delle uniche finisce in `dedup_images.json` ed è **l'unico insieme di lavoro** da qui in poi | 666 nomi |
| rotazione | `tesseract --psm 0` su un campione, ≥2 voti e ≥60% di supporto | 0°, 12/12 voti, 0.9 s con 4 processi |
| area data/ora + seconda dedup | OCR su 8 immagini gia' dritte; box applicato solo se ricompare nella stessa posizione su almeno 3 immagini e con supporto ≥35% | proposta visibile e correggibile nell'Import |
| ecografo | CNN vendor `vendor_training_no_negative_v2_power`, media delle softmax | Esaote 0.65 |
| sonda (#03) | CNN probe `probe_training_no_negative_v1` — **le classi sono gli ID sonda**, quindi il risultato è già `#03` | ID 12, conf 0.56 |
| rettangolo (#11) | rete rect `rect_training_e40_run2` (val IoU 0.933), specializzata BK con vendor ≥ 0.70; box cartella = mediana per componente | accordo fra immagini IoU 0.9925, dispersione 3–41 px |
| template ecografo (#13) | rete per-vendor dalla mappa `vendor_line13_template_map.json` (14 vendor, tutti presenti; Esaote val IoU 0.837) + consenso IoU 0.35 come la pipeline | box sul logo in alto a sinistra, accordo IoU 0.9877, 12/12 immagini tenute |
| piano L/T | `lt_training_transrectal_rect_only_v2_trecall` sui crop del rect, solo se la sonda è biplana | L 12/12, confidenza media 1.0 |

Nessuna riscrittura dei modelli: sono i checkpoint della pipeline attiva, con lo stesso
preprocessing del training. Il rettangolo arriva **proposto** allo step 2, con l'immagine e il
box disegnato sopra: l'utente conferma o corregge i quattro numeri.

Perché il passo zero deve venire prima dei codici: senza il rect non si possono ritagliare i
crop, e senza i crop non si può decidere il piano L/T — che è ciò che distingue `#04` = 3 da 4.

### Orientamento, depth e scala: i tre moduli agganciati

Girano come **sottoprocessi**, esattamente come li lancia la pipeline: `pipeline_context.json`
in ingresso, artefatti su disco in uscita (`<progetto>/stages/...`). Non è pigrizia: il
detector della scala vuole cv2 e Tesseract, e una cartella il cui OCR si impianta deve costare
uno step in `review`, non l'app intera; inoltre nessuna logica viene duplicata, quindi app e
pipeline non possono divergere.

| stadio | come | esito sulla cartella di prova (12 immagini Esaote TLC3-13) |
|---|---|---|
| marker di orientamento | `tools/orientation/predict_marker_envelopes_batch.py` con il bundle `41_orientation_marker_detector_bundle` | envelope per NF (4 marker) e LR (8); UD e LRUD assenti perché la cartella ha solo immagini "su" |
| su/giu | classificatore sui crop del rect (`model_su_giu_rect`, val balanced acc 0.991) | 12/12 "su", confidenza media 1.0 |
| depth | `tools/depth/predict_rect_depth_autonomous.py` | 12/12 accettate, depth da 18 a 120 mm, lette dall'etichetta dell'interfaccia (`D120 mm`) |
| scala | `tools/scale/predict_scale_from_pipeline.py` con i frame tessuti da su/giu + rect + depth | `#18` prodotta, `#19`/`#20`/`#21` vuote: 0 depth accettate su 9 |

Il risultato della scala **non è un difetto dell'aggancio**: il righello è stato individuato
(zona x=719, trovata in 10 frame su 12) ma le tacche per depth no. È il collo di bottiglia noto
del blocco scala — nella `ROADMAP` "50 righe su 91 Esaote sono setup interi senza alcun
agganciamento". L'app lo mostra con i motivi di review del modulo
(`scale_line21_incomplete`, `scale_depth_without_answer`) invece di far finta di niente.

**Un gruppo non trovato non è un errore.** Significa che quell'orientamento non esiste in questa
configurazione, ed è un'informazione che appartiene alla pagina codici — è il
`vbOrientationAvailable` del legacy. Quindi:

1. gli **orientamenti disponibili** nella pagina codici si aggiornano da soli: le spunte dei
   gruppi mai visti si tolgono, con la provenienza scritta accanto;
2. `#16` non resta monca: il posto del gruppo assente porta la **copia di uno esistente**, la
   convenzione che il formato documenta per le righe a quattro gruppi e che i progetti legacy a
   un solo orientamento hanno davvero in `#16`;
3. lo step resta `proposed`, non `blocked`, e la pagina dice cosa è stato copiato e da dove.

Se invece quelle acquisizioni dovrebbero esserci, l'utente rimette la spunta dalla pagina codici.

Verificato su una cartella con un solo orientamento (24 immagini): trovati NF e LR, assenti UD e
LRUD, spunte aggiornate a `NF/LR presenti`, `#16` con quattro blocchi di cui il terzo e il quarto
copie del primo (`166|338|185|742`), step `proposed`.

### Tutto lavora solo sulle immagini deduplicate

La dedup non serve solo a contare: le uniche sono l'insieme di lavoro di **ogni** stadio.

- Le reti (vendor, sonda, rect, `#13`, L/T, su/giu) campionano dalla lista salvata in
  `dedup_images.json`, non dalla scansione grezza.
- I moduli in sottoprocesso (marker, depth, scala) **scandiscono una cartella da soli**: non si
  può passargli una lista. Ricevono quindi `input_dedup/`, un mirror di sole *symlink* alle
  immagini uniche che conserva i percorsi relativi — lo stesso espediente di `input_ref/` in
  `run_pipeline_single_folder_safe.py`. I nomi relativi restano identici, quindi le detection
  del marker continuano a combaciare con l'elenco del visualizzatore (verificato: 16/16).
- Il visualizzatore scorre le 666 uniche, non le 3128 grezze.

Il mirror si ricostruisce solo se cambia la cartella o il numero di uniche.

**Effetto misurato**: con il campione preso dalle sole uniche, la depth passa da 16/16 accettate
a 1/16. Non è una regressione, è una misura più onesta: prima le prime 16 immagini della
scansione grezza venivano tutte dalla stessa sottocartella omogenea, ora il campione è
distribuito su tutta la cartella e incontra anche i casi difficili.

### Il visualizzatore dell'orientamento

Nello step Orientamento, sopra le schede dei moduli:

- i **quattro envelope** disegnati sull'immagine, un colore per gruppo, con il numero di marker
  che li ha generati nella legenda;
- il **marker trovato nell'immagine corrente**, riquadrato in bianco, con gruppo, punteggio di
  match, nome del template e ambito di ricerca;
- il **ritaglio dell'immagine corrente** alle coordinate del match, ingrandito accanto
  all'immagine: è il glifo che la configurazione deve portarsi dietro, non un riferimento;
- in fondo, il **confronto con la banca**: il template storico che ha fatto match accanto al
  ritaglio corrente, per vedere a occhio quanto differiscono;
- lo scorrimento di **tutte** le immagini uniche con `‹ precedente` / `successiva ›` e le frecce
  della tastiera, più `prossima con marker`;
- i **filtri per gruppo**, che partizionano l'elenco e su cui lo scorrimento continua a
  funzionare: `tutte` · `NF` · `LR` · `UD` · `LRUD` · `da rivedere` · `senza orientamento` ·
  `escluse dal modulo`. Ogni filtro porta il suo conteggio e si spegne se vuoto;
- la **confidenza** dell'immagine corrente accanto al contatore, colorata per soglia (verde da
  0.80, ambra da 0.62, rossa sotto), e un **elenco** sotto l'immagine con punteggio, gruppo e
  nome per ogni immagine del filtro attivo: si clicca una riga e si va lì;
- l'**ordinamento**: `cartella` · `confidenza crescente (peggiori prima)` · `confidenza
  decrescente`. Vale **dentro il filtro**, quindi si scegle il gruppo e poi si guardano i suoi
  casi peggiori — sul gruppo LR della cartella reale i primi sono 0.404, 0.426, 0.535, mentre la
  mediana del gruppo sta sopra 0.83. Le immagini senza punteggio restano in fondo in ogni caso.

### Segnalare e correggere un marker sbagliato

`Segnala e correggi` mette il visualizzatore in modalità scelta (mirino sull'immagine). Si clicca
**approssimativamente** dove sta il marker e il resto lo fa il programma:

1. **La confidenza si ricalcola subito attorno al punto, scegliendo la sovrapposizione
   migliore.** Nella finestra di ±70 px attorno al click si prova **tutta la banca del vendor a
   più scale** (0.75 · 0.9 · 1.0 · 1.15 · 1.3 · 1.7 · 2.2), con lo stesso `TM_CCOEFF_NORMED` del
   detector, e vince il punteggio più alto. Due passate come fa il bundle: tutti i template alla
   dimensione nativa, poi le scale extra solo sui sei migliori — 118 sovrapposizioni in 0.1-0.6 s.
   La scansione delle scale serve perché **il glifo cambia dimensione fra ecografi**: sul caso
   peggiore del batch (0.3882) il raffinamento arriva a 0.666 scegliendo un altro template a
   scala 1.15, dove la stessa banca a scala nativa si fermava a 0.6471.

   La pagina mostra il confronto, non solo il risultato: `nuova confidenza 0.9461 (prima 0.8487,
   migliorata)`, la sovrapposizione scelta con template e scala, e `migliore di 118
   sovrapposizioni provate · seconda: 0.8751`. Così si vede che è stata una scelta fra
   alternative, non un tiro singolo.
2. **Si verifica il gruppo**, con la regola del bundle (`_group_from_side_vertical`): lato e metà
   verticale della posizione dentro il rettangolo ecografico danno
   `sinistra+su → NF`, `destra+su → LR`, `sinistra+giù → UD`, `destra+giù → LRUD`. La pagina dice
   se il gruppo resta quello o se la correzione sposta l'immagine, prima di salvare.
3. **Ogni click è un suggerimento che ricalcola tutto**, come job con avanzamento a stadi
   (`1/5 raffino la posizione` → `2/5 salvo la correzione` → `3/5 ritaglio il marker corretto e
   lo provo sulle altre immagini` → `4/5 ricalcolo i quattro envelope` → `5/5 rivalido il
   ritaglio su tutta la cartella`). Senza i messaggi l'operazione sembrava bloccata.

   Il ritaglio nato dalla correzione **entra in gara** con quello attualmente consegnato: si
   misura la copertura di entrambi sullo stesso campione e vince chi copre più immagini, con la
   mediana come spareggio a pari copertura. Serve perché un ritaglio che copre uguale ma
   aderisce peggio (mediana 0.70 contro 0.99) non deve rimpiazzarne uno migliore.

   **La rielaborazione si vede sull'immagine**, non in una riga di testo altrove: l'immagine si
   scurisce, compare uno **spillo azzurro nel punto cliccato**, uno spinner, il nome della tappa
   corrente e una **barra a cinque segmenti** che avanza; la riga di quell'immagine nell'elenco
   delle confidenze si marca "in rielaborazione". A lavoro finito il velo sparisce e il nuovo box
   lampeggia in verde.

   **Il suggerimento vale per tutte le immagini, non solo per quella corretta.** Il ritaglio
   nato dal click entra in un insieme di template che vengono provati su ogni immagine dentro
   il suo envelope, e per ciascuna vince il punteggio migliore. Prima il suggerimento entrava
   in gioco *solo* se vinceva la gara per diventare il ritaglio consegnato: se non vinceva, per
   le altre immagini non cambiava nulla — da cui l'impressione che non venisse mai riusato.
   Misurato: il primo suggerimento ha migliorato **102 immagini** (0.9323 -> 0.9938).

   Chi diventa il **ritaglio consegnato** si decide dopo la validazione e con la stessa misura:
   vince chi aggancia piu' immagini dentro gli envelope. Sulla cartella di prova il conteggio
   e' `consegnato 195 · hint_03 105 · hint_04 57`, quindi nessuna promozione. Verificato che il
   ritaglio consegnato **da solo** copre comunque il 100%: i suggerimenti aggiungono margine,
   non copertura.

   Le correzioni **sopravvivono al ricalcolo** dello stadio e contribuiscono agli envelope: prima
   "Ricalcola i tre moduli" le cancellava.

   Alla fine la pagina mostra tre sezioni: **questa immagine** (posizione, confidenza prima e
   dopo, gruppo), **il ritaglio da consegnare** (copertura del candidato contro l'attuale, e se
   è stato adottato con il motivo), **la cartella dopo la correzione** (copertura per gruppo
   prima→dopo e quali envelope si sono spostati).

4. **Il salvataggio è automatico**: il click è la conferma, la correzione entra nel progetto e gli
   **envelope si ricalcolano** includendola. L'annullamento resta a un tasto (`Annulla questa
   correzione`), perché un salvataggio automatico senza ritorno indietro sarebbe una trappola.
   **Un click, una correzione**: dopo il salvataggio la modalità mirino si spegne da sola. Senza
   questo, restando in modalità, ogni click successivo sull'immagine salvava un'altra correzione
   senza che l'utente lo volesse — è successo per davvero durante i test automatici, con 13
   correzioni scritte per sbaglio.

Le correzioni hanno una **sezione propria** in testa allo step, che appare solo quando ce n'è
almeno una: una riga per immagine con il **ritaglio salvato**, gruppo (e quello precedente se è
cambiato), box, punteggio, template e data, più `vai` per tornare su quell'immagine e `rimuovi`
per annullarla. Il ritaglio mostrato segue la correzione, non più la posizione del batch.

Misurato: con un click 11 px a destra e 8 px sopra il marker vero, il raffinamento trova
`166|722|185|742` con punteggio **0.9607** — più alto dello 0.8369 del batch, perché nella
finestra prova tutta la banca e vince un template diverso (`marker_021.png`, 21×20 invece di
18×18). Lo scostamento dal punto cliccato viene mostrato, così si vede quanto ha corretto.

Il ricalcolo degli envelope usa **la stessa popolazione del batch** — solo le detection accettate,
le `review` restano fuori — verificato riproducendo esattamente le quattro caselle del batch a
correzioni vuote. Serve a questo: dopo una correzione si muove **solo** il gruppo interessato
(`NF` da `172|334|542|855` a `166|334|542|855`), non tutti e quattro. I conteggi differiscono di
poco da quelli del batch (97 contro 101) perché il batch somma le righe di due run e l'app conta
le immagini uniche.

I conteggi tornano e dicono qualcosa: sulla cartella reale `NF 107 · LR 105 · UD 79 · LRUD 87`
fanno 378, cioè tutte le analizzate, `senza orientamento 0` (il marker non ha mai fallito) e
`escluse dal modulo 288`, che con le 378 fanno le 666 uniche. `da rivedere 24` è trasversale ai
gruppi: il marker c'è ma il modulo non se ne fida, ed è la coda su cui vale la pena guardare.

**Il marker gira su tutte le immagini uniche, non su un campione.** L'envelope di un gruppo vale
quanto i marker che ha visto, e il visualizzatore deve poter mostrare un box su qualsiasi
immagine si scorra. Verificato sulla cartella reale: 378 immagini analizzate, marker trovato nel
**100%**, review rate 0.066, envelope con 101 marker per NF, 106 LR, 78 UD, 83 LRUD — contro i
3-5 marker per gruppo del campione da 16. Gli envelope che ne risultano sono molto più larghi, ed
è esattamente ciò che chiede il brainstorming: "voglio trovare il rettangolo che racchiuda tutte
le posizioni per ciascun orientamento".

Le 288 immagini restanti non sono un buco: sono le **schermate proibite** (`ZOOM-1_PROIBITE`),
che il modulo esclude per nome insieme a `Thumbs.db`, `Software Release` e `System Info` — lì un
marker di orientamento non c'è. Il visualizzatore lo dice esplicitamente quando ci si passa sopra,
invece di lasciar credere a una detection mancata.

Due cose imparate agganciando:

1. **Il marker runner ricavava il vendor dal nome della cartella** (convenzione dell'SSD:
   `1.Esaote_Nine_TLC3-13_...`). Su una cartella con un nome qualsiasi finiva in
   `error:no_templates`. Gli ho aggiunto un `--vendor` per passare quello riconosciuto dalla
   rete: l'app non dipende più da come è nominata la cartella.
2. **I moduli devono vedere il rettangolo con il margine**, non quello base: è il box che ESI
   disegnerà, ed è quello dentro cui vanno cercati righello e depth.
3. **Il box del template ecografo #13 e' escluso dalla ricerca del marker.** L'app ora usa
   la stessa regola della pipeline ufficiale: allarga quel box del 75% e oscura la zona in
   ogni ricerca automatica. Il nome o logo del vendor e' fisso sullo schermo e non deve mai
   poter essere scelto come marker di orientamento.

### Cosa manca ancora al passo zero, e perché

`#14 RECT_NAME_PROBE` **non è calcolabile oggi**. Non serve una rete: nella pipeline viene dal
`RectNameProbeResolver`, un resolver storico che legge i `.fss` legacy elencati nel manifest di
riferimento (`references/rect_dataset_no_negative_v2/manifest_rect_echo.csv`, 426 cartelle).
Quei file puntano a `ESIBuilder_AI/Dataset/<cartella>/DB_setup/setup_NN.fss`, e **la cartella
`Dataset/` non esiste più nel repo**: 0 dei 426 percorsi risolve. Lo stesso corpus serve al
resolver storico di `#02` e al fallback di `#13`.

Finché il corpus non è raggiungibile (cartella ripristinata o volume esterno montato), `#14`
resta vuoto e va compilato a mano. Non è un problema dell'app: è un dato di ingresso mancante.

Il post-processing di `#13` è parziale: c'è il consenso IoU fra immagini, mancano il gate di
template-matching NCC e il trim del bordo scuro che la pipeline può applicare dopo.

## 13-bis. La pagina codici si compila da sola — `encoding_struct`

Requisito: l'utente non deve digitare gli ID. Il programma li ricava dall'anagrafica
`encoding_struct*.xlsx` a partire da **modello ecografo** e **modello sonda**; se la
corrispondenza non c'e', **crea la riga** nel file.

Struttura dell'anagrafica (letta dal file del 2026-02-23):

| Foglio | Righe | Cosa contiene |
|---|---|---|
| `ECO` | 190 | `Modello ecografo` -> `ID ecografo` (#02) |
| `PROBE` | 77 | `Modello sonda` -> `ID Sonda` (#03) |
| `PROBE TYPE` | 5 usate | 1 lineare, 2 convex, 3 transrettale L, 4 transrettale T |
| `NDG` | 100 | `ID Needle Guide`, codice KIT, angoli |
| `FSS` | 410 | **la tabella maestra**: eco + sonda + ndg -> `ID ECHO`, `ID SONDA`, `ID NDG`, `Tipo sonda`, `FSS` |

Ordine di risoluzione: prima si cerca la **combinazione eco+sonda** nel foglio `FSS`, che da'
in un colpo `#02`, `#03`, `#05`, `#04` e l'ID setup; se la combinazione non esiste si ricade sui
fogli `ECO` e `PROBE` per i singoli ID. Il confronto e' su token normalizzati con punteggio, e
la pagina mostra sempre i candidati con il loro punteggio: propone, non decide.

Quando manca, l'app crea la riga con il **primo ID libero** (max + 1) e fa un **backup
timestampato** del file prima di scrivere. `encoding_struct` e' versionato in git, quindi la
nuova riga e' una modifica tracciata che resta da rivedere e committare.

Verifica: per "Esaote, MyLabSigma" + "TLC3-13" la risoluzione propone `ID_ECHO 17`,
`ID_PROBE 12`, `ID_NDG 13`, tipo `3`, setup `41` — che sono **esattamente** le righe #02-#05 di
`setup_41.fss` legacy. Le combinazioni vicine mostrano anche come l'anagrafica registra le
biplane: `... F080104 - L + TLC3-13` con tipo 3 e fss 41, `... - T + TLC3-13` con tipo 4 e fss 42.

### Tipo sonda (#04): risolto, ma per un'altra strada

Il requisito era ricavare lineare/convex dalla forma dell'immagine. La misura geometrica
**non funziona**, provata tre volte su cartelle reali con verità dall'anagrafica:

| misura | esito |
|---|---|
| ampiezza dell'area luminosa sul frame intero | 3/5 cartelle |
| stessa misura sulla banda contigua più lunga | 1/6 |
| maschera di speckle (varianza locale) | 1/6 |
| **ampiezza dentro il crop del rettangolo** | 0/6 |

Sul frame intero la geometria è dominata dalla grafica dell'ecografo; dentro il crop il
problema è opposto e più sottile: il `RECT_ECHO` è la finestra che ESI mostra e **taglia
l'apice del ventaglio**, quindi fra bordo alto e bordo basso la larghezza cresce poco e anche
un convex sembra lineare. Le misure restano in `tools/app/probe_shape.py`, che non è agganciato.

La strada che funziona non ha bisogno di nuove reti: **la CNN sonda predice direttamente l'ID
sonda**, e l'anagrafica dall'ID dà il tipo — `ID 12 → Esaote TLC3-13 → tipi [3, 4]`,
`ID 2 → GE C1-6 → [2]`, `ID 3 → GE ML6-15 → [1]`. Quando i tipi sono `[3, 4]` la sonda è
transrettale biplana e il **piano L/T** scegle fra 3 e 4. È la stessa logica del
`probe_type_router` della pipeline.

Resta scoperto un solo caso: una sonda **mai vista**, non in anagrafica. Lì servirebbe un
classificatore lineare/convex addestrato sui crop del rect — con le etichette gratis
dall'anagrafica (ID sonda → tipo) — non un'euristica geometrica.

### ID ecografo (#02): si scegle, non si indovina

La rete vendor riconosce la **marca**, non il modello di macchina. In anagrafica la stessa
sonda TLC3-13 compare su **68 righe con 30 `ID ECHO` distinti**, e la colonna risoluzione è
compilata solo in 16 righe su 68: marca + sonda non identificano la macchina.

Quindi `#02`, `#05` e l'ID setup si compilano automaticamente **solo se la riga è univoca**.
Se non lo è, la pagina codici mostra i setup registrati per quella sonda e quel piano — 33
voci per la sonda 12 su piano L, ognuna col nome della macchina — e l'utente scegle la sua.
Lo stesso vale per il fallback sul foglio `ECO`: un vincitore senza margine sul secondo non
è una risposta.

## 14. Stato dell'implementazione (26-08-2026)

Codice in `tools/app/` (vedi `tools/app/README.md`). Avvio:
`OldSoftwareEsiBuilder/.venv-mps/bin/python tools/app/server.py --port 8800`.

Reale e verificato:

- creazione e riapertura progetto, `project.json` con codici, sorgente e stato per step;
- pagina codici sempre modificabile, con invalidazione **mirata**: cambiare un ID non tocca niente,
  cambiare il tipo sonda tocca biplana e linee guida, cambiare la dimensione dell'immagine campione
  riporta in review tutti gli step con coordinate;
- import di una cartella reale: 3128 file -> 666 dopo la dedup esatta, 2462 duplicati rimossi,
  input video e risoluzione dedotti, griglia di miniature, avviso quando i nomi file non parlano;
- writer `.fss` con validazione e anteprima riga per riga, generazione del file, quality gate con
  `tools/fss/compare_fss.py`;
- wizard a 11 step con semaforo, pagina legacy e righe `.fss` di competenza, "vai al primo problema";
  lo step Biplana si spegne da solo sulle sonde non biplane.

- risoluzione dei codici da `encoding_struct` (sezione 13-bis) con creazione della riga mancante,
  backup automatico e ricarica dell'anagrafica;
- **passo zero completo** (sezione 12-bis): dedup, rotazione OSD, vendor, sonda, rettangolo e
  piano L/T come unico job in background con avanzamento per stadio; il rettangolo arriva allo
  step 2 come proposta con anteprima e box disegnato, e i codici si aprono già compilati con la
  provenienza di ogni campo.

Non ancora agganciato: gli stadi di inferenza (vendor, probe, rect, orientamento, depth, scala) e il
canvas di tracciamento. Gli step relativi accettano il valore in JSON, nella forma che il writer
si aspetta.

Limite noto: su cartelle grandi su disco esterno l'import richiede minuti (la dedup legge ogni file
per lo SHA-1, in parita' con la pipeline). La UI mostra un contatore di avanzamento; una barra di
progresso reale e' un miglioramento da fare.

## 15. Da discutere con Francesca

Domande che non si risolvono guardando il codice o i dati: dipendono da come ESI si comporta
davvero e da come sono state fatte le configurazioni storiche.

1. **Dimensione del ritaglio dei template** (sezione 8-bis). Il detector propone un box di
   17×17 px, il template storico che ha fatto match è 18×18, il legacy `setup_43` salva 30×31.
   Cosa si aspetta ESI: il box esatto, un margine fisso attorno, o una dimensione fissa per
   setup? Un ritaglio troppo stretto fa fallire il match, uno troppo largo prende contesto che
   cambia fra acquisizioni.
2. **Riga vuota in `#13`/`#14` quando il template non c'è** (punto aperto 7). Nei 10 `.fss`
   legacy nessuna riga è mai vuota: ESI accetta la riga vuota o vuole un box segnaposto con `B`
   invertito?
3. **Versione da scrivere in `#01`** (punto aperto 2). I legacy in `DB_setup` hanno 3.1/3.2/3.3
   e il valore arriva dalla versione del builder. Quale versione accetta l'ESI installato oggi?
4. **Convenzione per l'orientamento assente** (sezione 12-bis). Quando un gruppo non compare,
   riempio il suo posto in `#16` con la copia di uno esistente, come fanno i file legacy a un
   solo orientamento. È la convenzione giusta anche per ESI, o preferisce un box nullo?
5. **Cartella con entrambi i piani L e T** (punto aperto 6). `1.Esaote_Nine_TLC3-13...` contiene
   `TLC3-13 L` e `TLC3-13 T`, e in anagrafica sono due setup distinti (41 e 42). Una cartella
   così va trattata come due configurazioni separate?

## 8-quindecies. La depth: perché sbagliava sull'Esaote Nine

Sulle 16 immagini di `1.Esaote_Nine_TLC3-13_Firenze` il modulo dava `30/80/60/20/90 mm`: valori
che non stanno da nessuna parte sullo schermo. Il modulo era quello giusto e più aggiornato
(`tools/depth/predict_rect_depth_autonomous.py`, checkpoint 02-07-2026), ma non girava nella sua
forma completa e sbagliava per un motivo preciso.

**Il ranker non veniva caricato.** Nel `summary.json` della run compariva `"ranker_model": ""`,
mentre tutte le run storiche buone hanno il percorso pieno. Due cause indipendenti: il default si
calcola da `REPO_ROOT`, e in un worktree `artifacts/` non esiste; e il venv dell'app non aveva
`joblib`/`scikit-learn`. Rimesso il ranker, però, le 16 predizioni restano identiche: era un
difetto vero, non la causa.

**La causa.** Sul Nine la depth sta nella riga alta dell'interfaccia, `*TEI *D 162 *mm`.
Tesseract in modalità riga la legge tutta attaccata — `13-TELD162mm/M0FF/2` — e il `D` isolato e
il `mm` si perdono. Il candidato `162 mm` c'era, con `image_support=16` e `unique_values=16`, ma
valeva 0.20: ha vinto un `'3'` a 0.24, preso dallo **slider di frequenza "3 … 13"** della sonda
TLC3-13. Da lì `30 mm`. La passata OCR a token sparsi (psm 11/13), quella aggiunta per l'Hitachi,
era abilitata solo per le bande nel 15% *basso* dell'immagine: l'Hitachi ha la label in basso, il
Nine ce l'ha in alto. Con la passata giusta Tesseract restituisce `D` `162` `mm` separati.

**Cosa è cambiato**

- `tools/depth/rect_depth_hybrid.py`: la passata a token sparsi vale anche per le bande che
  toccano il bordo alto, e la striscia superiore viene letta anche a tessere compatte
  sovrapposte, così la label conserva un box stretto e non ne nasce uno fuso largo.
- `tools/depth/predict_rect_depth_autonomous.py`: `_load_ranker` dice a `stderr` perché ripiega
  sulle sole regole (joblib assente, modello non trovato, pickle illeggibile) invece di degradare
  in silenzio.
- `tools/app/stages.py`: il ranker viene risolto anche dal checkout principale quando si lavora in
  un worktree; `--probe` riceve il **modello** (`TLC3-13`) e non l'id anagrafica (`12`), che per le
  regole di profilo non vuol dire niente; `ranker_model` e `folder_strategy` tornano nel risultato.

**Misura** (stesse 16 immagini, prima → dopo)

| | prima | dopo |
|---|---|---|
| modo | 8 `scale` + 8 `direct_label` | 16 `direct_label` |
| strategia di cartella | `scale_from_variable_scale` | `direct_label_unit_stable` |
| esito | 1 accepted, 8 review, 7 reject | 8 accepted, 8 review, 0 reject |
| valori | 20/30/60/70/80/90 mm | 15/24/33/43/46/52/61/70/79/107/120/126/145/162 mm |

Zero regressioni sulle cartelle di riferimento: Hitachi Arietta V60 L441 28/28 predizioni
identiche al checkpoint; Esaote MyLab X8 TLC3-13 nessun peggioramento e un miglioramento
(`LR_111` da `review` ad `accepted`).

**Ambiente.** Il venv dell'app ha ora `joblib 1.4.2` e `scikit-learn 1.6.1`: la versione conta,
il ranker è stato addestrato con la 1.6 e con la 1.9 il pickle non si apre. `numpy` e `torch`
non sono stati toccati. Su Windows va fissata la stessa coppia di versioni.

## 8-sexdecies. Il riquadro della depth si stringe una volta e vale per la cartella

Quando la depth è **scritta a schermo**, la label non si sposta: sull'Esaote Nine sta sempre a
`left 582, top 87`, e cambia solo di larghezza perché le cifre crescono — 15 px per `43`, 23 px
per `162`. Perché il match tenga su ogni immagine il riquadro deve stare stretto sul **solo
numero**: se prende anche il `*mm` che segue, o la riga di interfaccia sotto, si porta dietro
contesto che cambia da un'acquisizione all'altra.

Quindi nella sezione depth il riquadro si trascina e si ridimensiona (maniglie sull'immagine,
oppure i quattro lati a numero per la precisione al pixel), e un solo comando lo applica a
**tutte** le immagini in cui la depth è stata letta dall'interfaccia — comprese quelle dove il
modulo l'aveva già trovata da solo. Su ognuna il valore viene **riletto lì dentro**, non copiato.

Sulla depth **dedotta dalla scala** il comando non c'è e la sezione lo dice: lì il numero cambia
posto ad ogni immagine, propagare un riquadro sarebbe falso.

**Come rilegge** (`POST /depth/box`)

1. La banda di ricerca è il riquadro allargato lateralmente (±120% della sua larghezza): serve
   spazio alle cifre in più o in meno.
2. Tre passate OCR (psm 7, 11, 6) con alfabeto ristretto alle sole cifre.
3. Fra i numeri letti vince quello che comincia più vicino al bordo sinistro del riquadro: è lì
   che l'utente ha messo l'ancora.
4. Fra le letture uguali si tiene il riquadro più stretto.
5. Il riquadro finale si porta sui pixel accesi, tenendo **tanti gruppi di colonne quanti sono i
   caratteri del numero**. Lo spazio non basterebbe a separare il numero dall'unità: fra `162` e
   `*mm` ci sono 5 px, fra le cifre 3-4. Il conteggio dei caratteri sì.
6. In verticale il riquadro si taglia sulla fascia disegnata dall'utente (con un quarto di gioco):
   il riquadro dell'OCR sconfina spesso sulla riga di interfaccia sottostante.
7. mm o cm si decidono dall'unità scritta accanto; se non c'è, dal rapporto fra il valore che il
   modulo aveva dato e il numero letto.

**Misura** — riquadro stretto a mano su una sola immagine (`580,84 → 620,104`) e propagato:

- applicato a 16 immagini su 16, nessun fallimento;
- valori: 15/24/33/43/46/52/61/70/79/107/120/126/145/162 mm, tutti giusti;
- riquadri: **13 px di altezza su tutte**, 14-16 px di larghezza per due cifre, 23 px per tre,
  ancorati a `left 582-583`. Prima erano 165 px (con `D` e `mm` dentro) o 307 px (riga fusa).

`Torna ai riquadri del modulo` rimette tutto com'era.

**L'ambito è la cartella, non il campione.** Il modulo depth gira su un campione (12-24 immagini)
perché generare i candidati costa minuti. Ma la label sta nell'interfaccia di *ogni* fotogramma, e
rileggere un numero dentro a un riquadro già noto costa 0,37 s per immagine. Quindi la
propagazione lavora di default su **tutte** le immagini deduplicate della cartella, in parallelo su
sei thread, come job con avanzamento; le immagini fuori dal campione entrano nell'elenco con stato
`dal riquadro`. Il menu accanto al comando permette di restringersi al solo campione.

Misura sulla cartella intera di `1.Esaote_Nine_TLC3-13_Firenze`: **666 immagini su 666 in 48
secondi, zero fallimenti**. Tutti i riquadri alti 13 px, larghi 13-23 px, ancorati a `left 582-583,
top 87`. I 25 valori distinti tornano con i nomi delle cartelle, che è la verifica più severa
disponibile:

| cartella | valori letti |
|---|---|
| `depthminimo_apertura minima` (L) | 18 mm su 12 immagini |
| `depthmax_aperturamax` (L) | 120 mm su 12 immagini |
| `DEPTHMIN_APERTURAMAX` (T) | 15 mm su 13 immagini |
| `DEPTHMAX_APERTURAMAX` (T) | 162 mm su 12 immagini |
| `DEPTH SX_DX_ZOOM0` (L) | 18/28/37/46/55/65/74/83/92/102/111/120 — dodici gradini, due frame ciascuno |
| `DEPTH_DX_ZOOM0` (T) | 15/24/33/43/52/61/70/79/88/107/126/145/162 — tredici gradini |
| tutte le `PROIBITE` e le prove di zoom | 46 mm, depth fissa |

La scala L coincide con quella letta a suo tempo sull'Esaote MyLab X8 con la stessa sonda. Le due
scale diverse fra L e T confermano che la cartella contiene due setup distinti (punto aperto 5 per
Francesca).

**Il riepilogo per valore.** A grandezza naturale il numero della depth è alto 13 px: guardando
l'immagine intera non si giudica se il riquadro si è fermato sulle cifre. La sezione depth ha
quindi due viste, scambiabili con una coppia di chip:

- *una per una* — l'immagine intera, il riquadro modificabile, le correzioni;
- *riepilogo per valore* — una scheda per ogni depth trovata, con la striscia di interfaccia
  **ingrandita 3 volte** e il riquadro verde disegnato sopra, la sua dimensione e il suo ancoraggio,
  quante immagini e quante cartelle danno quel valore.

I ritagli arrivano da `GET /depth/crop?name=…&zoom=N`, che li compone lato server dal riquadro
memorizzato (`image-rendering: pixelated` lato client, altrimenti il browser risfoca proprio ciò
che si vuole giudicare). `vedi tutte` apre la scheda a tutta riga e mostra i ritagli di ogni
immagine con quel valore, cliccabili per saltare alla vista singola.

Su 666 immagini il riepilogo sta in 25 schede: si vede in un colpo d'occhio che il riquadro è
sempre sul numero, e non su una tacca della scala o sullo slider di frequenza.

**L'area ingrandita è la superficie di modifica.** Il riquadro si ritagliava sull'immagine intera,
dove il numero è alto 13 px e il riquadro stesso finiva per coprirlo: non si vedeva più che cosa
si stava ritagliando. Nella vista *una per una* c'è ora, sopra all'immagine, la stessa striscia
**ingrandita 6 volte**, ed è lì che il riquadro si trascina e si ridimensiona. Sull'immagine intera
resta solo un riferimento tratteggiato, non modificabile, che dice dove si sta guardando.

- La finestra del ritaglio è il riquadro ± 34 px in orizzontale (dove le cifre crescono) e ± 8 px
  in verticale (dove non c'è niente da vedere: le righe di interfaccia distano una ventina di
  pixel). Resta **ferma** mentre si trascina, e si rifà solo se il riquadro ne esce.
- Il ritaglio arriva da `GET /depth/crop?…&raw=1&x0=&y0=&x1=&y1=`: pulito, perché il riquadro qui
  è un overlay HTML con le maniglie, non un rettangolo disegnato dal server.
- Le frecce `‹ ›` stanno accanto all'area ingrandita, e rispondono anche i tasti **← →** della
  tastiera (non mentre si scrive in un campo): si scorre la cartella controllando a ogni passo
  che il riquadro tenga.
- I quattro lati restano modificabili a numero, per il pixel esatto.

**Quattro difetti trovati provandolo su `prova`** (1280×1024, depth a 1251-1275, cioè a cinque
pixel dal bordo destro):

1. *Il riquadro si schiacciava invece di fermarsi.* Trascinandolo contro un bordo, il clamp
   fermava un lato e lasciava correre l'altro: da 24 px di larghezza diventava 6. Spostare non è
   ridimensionare — ora la traslazione viene limitata tutta intera.
2. *Le scale non coincidevano.* La finestra del ritaglio la calcolava il client e la ritagliava il
   server: contro il bordo destro il server serviva 64 px mentre il client ne assumeva 93, e il
   trascinamento andava a 64/93 della velocità del cursore. Ora è il client a ritagliarla, con la
   stessa regola, e spostando la finestra invece di stringerla.
3. *Più il riquadro era stretto, più la rilettura falliva in silenzio.* La banda di ricerca eredita
   il minimo del modulo OCR, 40×28 px: con un riquadro alto 14 px la banda ne misurava 22 e veniva
   rifiutata, senza errore. Ora la banda ha un pavimento.
4. *Le righe di interfaccia selezionate.* Nelle 60 immagini `_trans` la riga `D 1.5` è evidenziata,
   testo scuro su barra chiara: ogni passata tarata sul testo luminoso la perdeva, e il valore
   usciva `6460 mm`. Aggiunta una passata `invert` all'OCR e la polarità al ritaglio sui pixel.

Verifica contro i nomi dei file, che portano la depth vera: **124 giusti, 0 sbagliati** (prima 60
valori su 136 erano `460` o `6460 mm`). 143 immagini su 146; le tre che restano fuori sono
`PROIBITE/CF`, `CF_1` e `PW`, schermate colour-flow e pulsed-wave dove quella riga non c'è —
segnalate come non lette, non indovinate. Nessuna regressione sull'Esaote Nine: su un campione di
14 immagini le letture e i riquadri sono identici a prima.

**Un secondo riquadro per una parte della cartella.** La sezione ha ora un filtro per nome file e
l'ambito «solo le immagini elencate ora»: se una cartella ha due interfacce diverse si filtra il
sottoinsieme, si mette il riquadro al posto giusto su una di quelle immagini e si applica solo a
loro. Le letture sono per immagine, quindi la seconda applicazione aggiorna le sue e lascia stare
le altre; `depth_box_applications` tiene lo storico.

**Da fare**: questi riquadri sono esattamente ciò che serve alla riga `#17` (RECT_DEPTH), che oggi
si compila dallo step `depth_find`. Il collegamento non è ancora fatto perché `#17` è organizzata
**per valore di depth** (un gruppo per ogni voce di `#18`, con due immagini per blocco) e va deciso
quali due immagini per depth e con quale ordine.

## 8-septendecies. La cartella si sceglie sfogliando

Il percorso della cartella di acquisizioni si incollava a mano. Ora accanto al campo c'è
`Sfoglia…`, che apre un selettore.

Il browser non può dare il percorso vero di una cartella scelta con `<input type=file
webkitdirectory>` — restituisce nomi relativi, e all'importer serve il percorso sul disco. Ma il
server gira sulla stessa macchina dell'utente, quindi è lui a sfogliare: `GET /api/browse?path=…`
restituisce le sottocartelle di un percorso, e senza `path` i punti di partenza.

- **Punti di partenza**: home e Desktop/Documents/Downloads, i volumi montati (`/Volumes` su
  macOS, le lettere di unità su Windows, `/media` e `/mnt` altrove) e le cartelle già usate negli
  altri progetti, ciascuna insieme alla sua cartella madre — da lì si prende l'acquisizione
  successiva, che di solito è quella accanto.
- **Accanto a ogni cartella**: quante immagini contiene e quante sottocartelle ha. È quello che
  serve per capire, senza aprirla, se è l'acquisizione giusta. Il conteggio è solo del livello
  diretto: su un disco esterno con migliaia di file una conta ricorsiva costerebbe secondi per
  ogni riga.
- Il campo del percorso resta scrivibile e accetta Invio, per chi il percorso ce l'ha già.

Niente di specifico per macOS: la stessa finestra funziona sul Windows dei colleghi, che è il
motivo per cui non si è usato un selettore nativo di sistema.

## 8-octodecies. La rotazione va applicata, non solo misurata

Su `204. BK Specto - E14CL4b` l'analisi diceva «vanno ruotate di 90 gradi» e poi faceva girare
tutti i moduli sulle immagini dritte com'erano. Due difetti distinti.

**La rotazione non arrivava nemmeno a `source`.** Veniva scritta in memoria in
`project.source["rotation_applied"]`, ma poche righe dopo il progetto veniva **riletto da disco**
e quelle modifiche sparivano: sul disco restava `rotation_applied: 0`, `rotation_source:
"not_run"`, mentre `steps.import.value` diceva `90` con fonte `osd` e `reliable: true`. I moduli
leggono da `source`, e ricevevano zero.

**E anche con il numero giusto non sarebbe bastato.** L'angolo veniva soltanto *passato*: alle reti
(vendor, sonda, rettangolo, piano L/T) non arriva affatto, e il modulo della scala si rifiuta di
lavorare su un contesto ruotato — torna `rotation_not_supported`. Un numero nel contesto non
raddrizza nessun pixel.

La correzione ruota **i pixel, una volta sola**. Lo specchio di lavoro (`input_dedup`), che finora
conteneva symlink alle immagini originali, quando c'è una rotazione contiene **copie già
raddrizzate**; `Project.working_dir()` è da dove leggono tutti — reti, moduli, il visualizzatore,
la conversione dei click. Il timbro `.built_from` include l'angolo, così cambiando rotazione lo
specchio si rifà. Al contesto dei moduli si passa `rotation_deg_clockwise: 0`, che è la verità
sulle immagini che ricevono: dirgli l'angolo vero significherebbe farglielo applicare due volte.

I lati del fotogramma si scambiano — `native_size`, `image_sample_size` e `video_input_size` da
1280×1024 a 1024×1280 — perché è il fotogramma raddrizzato quello che ESI vedrà, ed è in quelle
coordinate che vanno scritte tutte le righe del `.fss`.

**Verifica** sulla stessa cartella BK, import e stadi rifatti da capo:

| | prima | dopo |
|---|---|---|
| `source.rotation_applied` | `0`, fonte `not_run` | `90`, fonte `osd` |
| dimensioni | 1280×1024 | 1024×1280 |
| immagini di lavoro | symlink agli originali | copie raddrizzate |
| vendor | — | BK, 0.631 |
| rettangolo | — | `138\|200\|1042\|887`, che racchiude il settore e lascia fuori interfaccia e righello |
| depth | — | 12/12 accettate, 20-70 mm |
| scala | `rotation_not_supported` | gira: profilo BK, `#18 = 25\|50\|60\|` |

Nel passaggio sono venuti fuori cinque punti che leggevano ancora dalla cartella originale mentre
i moduli misuravano sullo specchio: l'anteprima del rettangolo (che infatti falliva con *is not in
the subpath of*), il ritaglio del marker, il refine da click e la conversione dei click in
coordinate immagine. Tutti spostati su `working_dir()`.

I progetti senza rotazione non cambiano: lo specchio resta di symlink, e viene rifatto una volta
sola perché il timbro ora include l'angolo.

### Il template dell'ecografo non sopravvive a un cambio di fotogramma

Dopo aver rifatto l'import della cartella BK, nella sezione ecografo il template `#13` non si
vedeva più. Il box c'era, ma valeva `left 1058 → right 1114` in un fotogramma largo 1024: era
quello misurato sul fotogramma **storto**, prima della rotazione, e cadeva fuori dall'immagine.

La causa è una riga scritta per proteggere il lavoro dell'utente — `if line13.get("box") and not
vendor_value.get("rect_name_echo")` — che però proteggeva anche una proposta della rete, e quindi
nessuna rianalisi la rifaceva mai. Ora il box viene rifatto sempre, tranne quando l'utente l'ha
corretto a mano; e anche in quel caso, se cade fuori dal fotogramma corrente viene sostituito e la
sostituzione viene dichiarata in `rect_name_echo_replaced`, perché un box misurato su una
geometria diversa non è recuperabile.

**Da chiedere a Francesca** (punto aperto): sulla BK Specto la rete propone per `#13` il blocco
`MEDKONSULT DEMO / E14CL4b (T) – Brachytherapy/Prostate…`, cioè l'intestazione con studio e sonda,
mentre il nome della macchina — `bkSpecto` — è il logo in alto a sinistra. L'accordo fra le 24
immagini è 0.9757, quindi la rete è coerente con sé stessa: la domanda è quale dei due testi ESI
si aspetta di trovare in `#13`.

## 8-novodecies. I moduli girano da soli, l'utente corregge

> Aggiornamento 9 settembre 2026: questo comportamento e' stato sostituito dal flusso in
> due tempi. L'import non lancia piu' i tre moduli: prima si controllano i piani L/T e si
> sdoppia la cartella quando li contiene entrambi; solo dopo orientamento, depth e scala
> partono separatamente per progetto. L'endpoint dei moduli rifiuta una cartella biplana
> non ancora classificata o ancora mista.

L'import faceva dedup, rotazione e le reti — ecografo, sonda, rettangolo, piano L/T — e si
fermava lì. Orientamento, depth e scala restavano da lanciare a mano, con un «Calcola con i
moduli» nella loro sezione. Ma un wizard che chiede di premere calcola ad ogni tappa non sta
proponendo niente: sta facendo compilare il modulo all'utente, un pezzo per volta.

Ora l'import prosegue **da solo** nei tre moduli, dentro lo stesso job e con la stessa barra di
avanzamento. La precondizione è il rettangolo ecografico, che l'import ha appena trovato: se
manca, la catena si ferma lì e lo dichiara (`stages_skipped`) invece di fallire.

Il comando nella sezione dei moduli resta, ma cambia mestiere: non è più «calcola», è **«rifai i
tre moduli»** — quello che si usa dopo aver corretto qualcosa da cui dipendono, prima fra tutte il
rettangolo. È il giro di raffinamento, non il primo calcolo. Chi vuole il vecchio comportamento
passa `stages: false` all'import.

Misura sulla cartella BK, progetto azzerato e rifatto con **un solo comando**:

```
   5s  riconoscimento ecografo
  10s  orientamento: marker su tutte le 56 immagini
  20s  orientamento: ritaglio del marker della cartella
  70s  depth: lettura dalla scala ecografica
 175s  scala: righello e righe #18-#21
 180s  fatto
```

Alla fine ogni sezione ha la sua proposta: ecografo, rettangolo e orientamento `proposed`, i tre
moduli `ok`, depth 12 accettate su 12, marker con tre gruppi su quattro. La scala risulta
`blocked` perché le sue righe `#19-#21` non sono complete — che è un esito onesto da correggere,
non un passo da lanciare.

## 8-vicies. La scala: il primo blocco era la depth, non il righello

Lo stadio della scala girava già, ma su tutte e tre le cartelle di prova finiva a
`depths_accepted: 0`. Guardando `scale_per_depth.csv` la causa era la stessa ovunque:

```
25.0 mm  reject   frames_total=1 usable=0 agree=0  setup_too_small_for_consensus
50.0 mm  review   frames_total=1 usable=1 agree=1  setup_too_small_for_consensus
60.0 mm  review   frames_total=1 usable=1 agree=1  setup_too_small_for_consensus
```

Il modulo misura il righello **per valore di depth**, e per accettarne uno vuole più
fotogrammi che concordino. L'app gliene passava dodici presi a campione piatto sulla cartella:
ogni depth ne aveva uno. E `#18` elencava solo le depth capitate nel campione — `25|50|60|` su
una cartella che ne ha tredici.

La correzione non tocca il righello: cambia **cosa gli si dà da guardare**.

1. Dopo il modulo depth, il riquadro più stretto fra quelli accettati viene propagato a tutta
   la cartella con la rilettura da 0,4 s per immagine — la stessa della sezione depth. Ora ogni
   fotogramma ha un valore, non solo i dodici del campione.
2. I fotogrammi per la scala si scelgono **per valore di depth**: fino a sei per ciascuno,
   distribuiti nel gruppo per non pescarli tutti dalla stessa acquisizione.
3. Ogni fotogramma porta il suo `orientation_group`, che era sempre vuoto benché il marker
   fosse già girato: è il dato che dice alla scala da che parte guardare.

**Misura sulla cartella BK**, stessi moduli, stesse immagini:

| | prima | dopo |
|---|---|---|
| depth con un valore | 12 (il campione) | 54 su 56 |
| depth viste dalla scala | 3 | 13 |
| depth accettate | 0 | **7** |
| `#18` | `25\|50\|60\|` | `10\|20\|25\|30\|35\|40\|50\|55\|60\|65\|70\|75\|80\|` |
| motivi di review | `line21_incomplete`, `depth_without_answer` | solo `line21_incomplete` |

`scale_depth_without_answer` è sparito: ogni depth ha una risposta.

**Cosa resta.** Le sei depth non accettate sono tutte quelle basse (10, 20, 25, 30, 35, 40): o
`no_detection_filled_from_neighbours`, o `rejected_as_incoherent_with_setup`. Il `mm_per_px` che
il modulo trova va da 0,070 a 0,145 fra 10 e 80 mm, cioè cresce di 2× mentre la depth cresce di
8×: sulle depth basse la misura non è credibile, ed è lì che va guardato il righello. E `#19-#21`
restano vuote finché `#21` non è completa, che richiede una risposta buona per ogni depth.

Il passo successivo è la revisione umana della scala dentro l'app — la stessa cosa che oggi fa
`tools/scale/build_scale_correction_tool.py` in una pagina HTML separata.

## 8-unvicies. Lo studio del righello entra nell'app

Il modulo che studia la scala è un'altra cosa dallo stadio che consolida `#18-#21`:
`detect_scale_ladder` più i controlli incrociati di `study_scale_folder` trovano **la colonna
del righello**, ne staccano **le tacche**, distinguono **lo zero dal fondo**, ricavano **il
passo** e leggono **i numeri con l'OCR**. Non gli serve la depth — due etichette lette a due
altezze danno già `mm_per_px` e lo zero, che è quello che `#19`, `#20` e `#21` codificano.

Integrato con lo stesso patto degli altri: sottoprocesso, nessuna logica duplicata. Gira sul
**contesto che lo stadio della scala ha già scritto** (`--from-pipeline`), quindi i due guardano
gli stessi fotogrammi e i loro esiti sono confrontabili tacca per tacca.

Una sola aggiunta al modulo: `--data-json`, che scrive i dati dello studio **senza le immagini
in base64**. Sono quelle a far pesare megabyte la pagina HTML, e l'app le immagini ce le ha già —
21 KB invece di alcuni MB. `GET /scale/study` le serve, con i nomi ricondotti allo specchio di
lavoro.

**Primo esito sulla cartella BK** — 14 fotogrammi, 5 accepted, 1 review, 8 reject:

```
needle last hole.png   accepted  x=866.8  zero=784.4  7 tacche  passo 91.2 px  0.1087 mm/px  passo 10 mm
AGHI_T.png             accepted  x=959.5  zero=728.5  7 tacche  passo 89.3 px  0.1115 mm/px  passo 10 mm
T_LRUD_80.png          accepted  x=959.5  zero=741.5  9 tacche  passo 69.0 px  0.1449 mm/px  passo 10 mm
T_UD_40.png            accepted  x=959.5  zero=702.5  5 tacche  passo 128 px   0.0781 mm/px  passo 10 mm
L_UD_30.png            review    ...  geometry_step_over_pitch_single_label
L_LRUD_20/25/30, T_LRUD_20/25, L_UD_20/25, T_UD_25   reject   no_ladder
```

Due cose si leggono subito. Il passo torna **10 mm su tutti i fotogrammi accettati**, e il
`mm_per_px` cresce proporzionalmente alla depth (0,078 a 40 mm, 0,145 a 80): dove il righello lo
trova, lo misura bene. E **tutti gli otto scarti sono `no_ladder` sulle depth basse** — 20, 25,
30 — cioè esattamente le stesse che lo stadio non riusciva ad accettare. Non sono due problemi:
è uno solo, e sta nella ricerca della scala a depth bassa, dove le tacche visibili sono poche.

Il passo successivo è la sezione dell'app che disegna il righello sul fotogramma e lascia
correggere colonna, zero, fondo, passo e numeri — le correzioni sono già previste dal modulo
(`--corrections`), che le rimette nel calcolo.

## 8-duovicies. La sezione «Scala: righello e tacche»

Sezione sua, subito dopo la depth. Non possiede righe del `.fss` — `#19-#21` continua a
consolidarle lo stadio dentro `depth_scale` — perché il suo mestiere è un altro: far vedere e
correggere quello che il modulo ha trovato sul righello, un fotogramma per volta.

**Cosa si vede.** Sul fotogramma: la colonna del righello (tratteggiata), ogni tacca, lo zero in
verde e il fondo in ambra, e accanto alla colonna i numeri letti dall'OCR con il loro valore in
cm. Sotto: cosa ha trovato il modulo — x della colonna, y dello zero e del fondo, quante tacche,
il passo in pixel, il `mm_per_px`, i numeri e il loro passo in mm, da che parte sta lo zero, e il
confronto con la depth dell'interfaccia.

**Cosa si corregge**, e sono le cinque cose che il modulo sa riprendere:

| | come |
|---|---|
| colonna | si trascina in orizzontale |
| zero e fondo | si trascinano in verticale |
| tacche | modo «aggiungi/togli»: clic sull'immagine per aggiungerne una, clic su una tacca per toglierla |
| numeri | modo «correggi i numeri»: clic sul numero e si riscrive il valore in cm |
| verso | due chip: lo zero è in alto / in basso |

**Come impara.** Le correzioni finiscono nello step `scale_study` e vengono scritte in
`scale_study_corrections.json` nel formato che il modulo già si aspetta — `{cartella: {note,
frames: {nome: {...}}}}`. Ogni run successiva le riceve con `--corrections`, e il modulo le
applica **dopo** la detection: così la pagina continua a mostrare anche cosa avrebbe detto da
solo, e si vede la differenza. Una colonna corretta su un fotogramma diventa l'ancora di tutta la
cartella (`_corr_anchor_x`), a patto che le colonne corrette non si contraddicano fra loro.

Verifica del giro completo: correggendo colonna e verso su `L_LRUD_20.png` e rifacendo lo studio,
il fotogramma torna con `corr_applied: ['verso', 'colonna']` e la colonna a 959.5. Resta `reject`
perché lì la scala non ha proprio tacche (`no_ladder`): la colonna da sola non basta, e infatti è
per quello che le tacche si possono aggiungere a mano.

Round-trip dei 10 `.fss` legacy ancora identico dopo l'aggiunta dello step.

### La sezione non compariva: mancava la riga nel registro dei pannelli

Lo step c'era, la vista era scritta, l'endpoint rispondeva — e sulla pagina non si vedeva niente.
`PANELS` è il registro che associa uno step alla funzione che lo disegna, e `scale_study` non ci
era stato messo: la pagina cadeva sul pannello generico, che mostra il JSON dello step e nessuna
immagine. Aggiunta la riga, la sezione disegna colonna, tacche, numeri ed estremi.

Nell'occasione: un progetto analizzato prima che lo studio esistesse non ha i suoi dati, e la
pagina restava vuota anche a ragione. Ora in quel caso mostra un comando — *Studia il righello di
questa cartella* — che lo fa partire da lì, senza dover rifare tutti e tre i moduli.

### E l'immagine non si vedeva: `index.html` in cache

Aggiungere una sezione vuol dire aggiungere uno script a `index.html`. Il browser però teneva la
pagina in cache, e con lei la vecchia lista degli script: `scale_view.js` non veniva mai scaricato,
`createScaleViewer` non esisteva, e il pannello si fermava sulla `ReferenceError` **senza dire
niente** — restava l'intestazione dello step e nessuna immagine.

Due correzioni. Il server manda `Cache-Control: no-store` per `/` e per `/static/`: qui la pagina
si ricarica di continuo mentre si lavora, la cache non fa guadagnare nulla e nasconde le
modifiche. E il pannello, se la funzione della sezione non c'è, lo dice e suggerisce di
ricaricare, invece di fermarsi muto.

## 8-tervicies. Dove il righello non si trova: indicarlo, e usarlo come consiglio

Su `prova_2` otto fotogrammi su quattordici escono `no_ladder`: la scala non viene trovata, e
non c'è niente da correggere perché non c'è niente. Due aggiunte.

**Indicarlo a mano.** Modo *«indica tu il righello»*: un clic dove sta lo zero — che fissa anche
la colonna — e un secondo clic sul fondo. Da quei due punti si ricavano il verso e le tacche, e
il tutto diventa una correzione come le altre.

**Usarlo come consiglio sugli altri.** `GET /scale/study/suggestions` prende i fotogrammi in cui
il righello si sa — accettati dal modulo o indicati da te — e propone un righello su quelli
falliti. Sul fotogramma la proposta si disegna in **ambra punteggiata**, per non confonderla con
una misura, e un comando la accetta trasformandola in correzione.

**Su cosa si basa la proposta, e perché non sulla depth.** Le grandezze buone sono geometriche e
di cartella: la macchina disegna il righello sempre nello stesso posto, sempre lungo lo stesso, e
con lo stesso passo. Quindi colonna e zero dalla mediana dei noti, lunghezza della barra dalla
mediana dei noti, passo dalla mediana dei noti — e `mm_per_px = passo_mm / passo_px`.

Il primo modello che avevo scritto ricavava la lunghezza dalla depth, ed era sbagliato. I dati lo
dicono chiaramente:

| fotogramma | depth dell'interfaccia | barra misurata | mm/px × barra |
|---|---|---|---|
| `T_LRUD_80` | 20 mm | 550 px | **79,8 mm** |
| `T_UD_40` | 20 mm | 511 px | **39,9 mm** |
| `AGHI_T` | 20 mm | 537 px | **59,9 mm** |

`E_depth_interface` dice 20 mm su tutti e tre, mentre il righello letto dal modulo li dà a 80, 40
e 60 — e i nomi dei file confermano il righello. La lettura della depth su quella cartella è
sbagliata, quella del righello è giusta: un modello appoggiato alla depth avrebbe propagato
l'errore su otto fotogrammi.

La proposta dichiara sempre su quanti fotogrammi si basa e con quale scarto — «barra lunga 549 px
(scarto 39 px); passo 91 px = 10 mm» — e avverte quando le colonne note non concordano fra loro.

### La depth confermata sì, quella letta no

La proposta del righello non usa la depth *letta* dal modulo, per la ragione misurata sopra. Ma
una depth **confermata dall'utente** è un'altra cosa, e il modulo della scala lo sa già: nel suo
codice la depth dell'operatore è commentata come «the strongest kind of evidence there is». Quindi
si usa, e attraverso il canale che il modulo ha già.

Conta come confermata: un valore riscritto a mano nella sezione depth; le letture nate da un
riquadro che l'utente ha applicato lui (`scope` diverso da `auto`); o tutte, se lo step della
depth risulta confermato o corretto dall'utente. Le automatiche della catena no.

Finiscono in `scale_study_corrections.json` come `depth_mm` per fotogramma, così le usa tutto lo
studio e non solo la proposta. E nella proposta danno la lunghezza della barra: `depth / (passo_mm
/ passo_px)`.

Verifica su `L_LRUD_20.png`, che il modulo scarta con `no_ladder`:

| | proposta |
|---|---|
| senza depth confermata | fondo a 192,6 · 7 tacche · «barra lunga 549 px (scarto 39 px)» |
| con 20 mm confermati | fondo a 559,2 · 3 tacche · «lunghezza dalla depth che hai confermato (20 mm) col passo di 10 mm ogni 91 px» |

Venti millimetri a 10 mm ogni 91 px fanno 182 px di barra, non 549: la depth confermata cambia la
proposta, e il perché è scritto accanto.

## 8-quatervicies. Correggere il righello con un gesto, non con un modo

La prima versione della sezione aveva quattro modi da scambiare con dei chip — guarda, disegna,
tacca, numero — e per riscrivere un numero apriva la finestrella di sistema. Correggere costava
più clic del dovuto, e ogni clic andava speso a dire al programma *cosa* si stava per fare.

Rifatta senza modi: ogni gesto vale da sé.

| gesto | cosa fa |
|---|---|
| trascina la colonna, lo zero o il fondo | li sposta |
| **alt** + trascina | sposta **tutto il righello** insieme, tacche comprese |
| **shift** + clic sull'immagine | aggiunge una tacca |
| clic su una tacca | la toglie |
| **doppio clic** su un numero | lo riscrive lì dove sta, con Invio o Esc |
| due clic, dove il righello non c'è | lo indicano: prima lo zero e la colonna, poi il fondo |
| **shift+↑↓** / **alt+↑↓** | muovono lo zero / il fondo di un pixel |
| **← →** | cambiano fotogramma |
| **cmd+Z** | annulla l'ultimo gesto |

**Le due finestre ingrandite.** Sopra il fotogramma, zero e fondo a **4×** con una mira: a
grandezza naturale una tacca è alta due pixel e il pixel giusto non si vede. Trascinando dentro la
finestra si muove l'estremo, e il ritaglio segue. Arriva da `GET /depth/crop`, che ora accetta una
finestra esplicita anche senza riquadro della depth — prima la pretendeva, e nello studio della
scala non c'entra: era il motivo per cui le due finestre restavano vuote.

**Accettare in blocco.** Otto fotogrammi da sistemare uno per uno sono otto volte le stesse
quattro conferme. `POST /scale/study/accept` accetta i righelli proposti su un fotogramma o su
tutti, e la sezione ha *«Accetta tutti gli N proposti»*: si guarda che la proposta sia buona — è
disegnata sopra — si accetta in blocco, e si ritoccano i pochi che stonano. Nell'elenco laterale
ogni fotogramma dice se è `proposto`, `corretto` o se ha solo una `proposta` in attesa.

Ogni salvataggio lo dice — un «salvato» accanto alla didascalia — così non serve chiederselo.

## 8-quinvicies. Far vedere che le correzioni sono state accettate e usate

Correggere e non sapere se è servito è lavorare al buio. Mancavano tre risposte: la correzione è
salvata? il modulo l'ha riletta? e cos'è cambiato per gli altri fotogrammi?

**Salvata ≠ usata.** Ogni correzione porta il suo `ts`, ogni giro dello studio il suo. Le
correzioni più recenti dell'ultimo giro sono *in attesa*: una fascia ambra le conta e le nomina —
«3 correzioni ancora da usare — sono salvate, ma il modulo non le ha ancora rilette» — e nell'elenco
laterale quei fotogrammi portano `in attesa`. Dopo il giro la fascia diventa verde e dice cosa è
successo: quante correzioni ha riletto, quanti fotogrammi hanno un righello, quali sono migliorati
e quali peggiorati. Su ciascun fotogramma si legge cosa il modulo ha applicato davvero
(`corr_applied`): «il modulo ha applicato: verso, colonna».

**Lo stato diceva il falso.** `study_scale_folder` applica le correzioni *dopo* la detection e
lascia `status` com'era — scelta giusta là dentro, perché così si continua a vedere cosa avrebbe
detto da solo. Ma in una pagina di revisione un fotogramma di cui hai disegnato tu il righello
restava `reject`, e sembrava che la correzione non fosse servita a niente. L'app calcola ora uno
stato effettivo: `corrected` quando la correzione fornisce colonna, zero e fondo. Quello del
modulo resta visibile accanto — «stato corrected (il modulo da solo: reject)» — perché la domanda
«la detection sta migliorando?» è un'altra e va tenuta.

I filtri seguono: *trovati* dal modulo, *dati da te*, *da rivedere*, *senza righello*.

**Giro completo misurato** sulla cartella BK, 29 fotogrammi:

| | trovati | dati da te | da rivedere | senza righello | in attesa |
|---|---|---|---|---|---|
| partenza | 14 | 0 | 1 | 14 | 0 |
| accettate le 9 proposte | 14 | **9** | 1 | 5 | **9** |
| dopo il giro | 14 | 9 | 1 | 5 | **0** |

«ultimo giro: il modulo ha riletto 9 correzioni · 23/29 fotogrammi con righello». Da 14 a 23 con
un comando e un giro.

Un dettaglio che rendeva il confronto senza senso: la run rifaceva lo studio su
`len(correzioni) + 14` fotogrammi, quindi ogni giro ne guardava di più e i nuovi arrivati
sembravano comparsi dal nulla. Ora il numero di fotogrammi resta quello del giro precedente.

### Le finestre ingrandite non corrispondevano all'immagine intera

Il ritaglio arriva dal server ingrandito 4×, ma il CSS lo porta poi alla larghezza della scatola.
La mira invece veniva posizionata assumendo 4 px per pixel nativo: su una scatola larga 361 px per
150 px di finestra la scala vera è **2,407**, quindi la mira finiva al 66% oltre il punto giusto —
e lo stesso errore rallentava il trascinamento dentro la finestra. Guardando le due viste non si
poteva sapere se la correzione fosse a posto, che è l'unica ragione per cui le finestre esistono.

Corretto: la scala si misura, `larghezza mostrata / larghezza della finestra`, e si ricalcola
quando la scatola cambia dimensione. La finestra viene inoltre ritagliata sui bordi dell'immagine
**spostandola** e non stringendola, con la stessa regola del server: altrimenti vicino a un bordo
i due parlerebbero di ritagli diversi — lo stesso difetto già trovato nella sezione depth.

E per rendere la corrispondenza verificabile a occhio, dentro la finestra si disegnano ora **le
stesse cose** dell'immagine intera: la colonna tratteggiata, le tacche che ci cadono dentro, e la
mira dell'estremo. Il titolo porta le coordinate esatte — «zero: y=784 · colonna x=867».

Verifica: ricostruendo le coordinate dai pixel disegnati si torna a `y=784` e `x=867`, cioè
esattamente i valori del titolo e dell'overlay.

### La depth tornava alle dodici del campione: lo stadio della scala la cancellava

Sul progetto BK la depth risultava trovata su 12 immagini invece che su tutte, benché lo stadio
registrasse `box_reads: 54`. La propagazione era avvenuta davvero: veniva cancellata subito dopo.

Alla fine del giro lo stadio della scala scriveva lo step `depth_scale` **sostituendo** il valore
per intero — `project.set_step("depth_scale", {**parsed, ...})` — e nello stesso step vivono anche
`depth_box_reads`, `depth_box_template` e le correzioni dell'utente. Sparivano tutte, e la sezione
depth tornava alle dodici immagini del campione.

Ora quella scrittura aggiorna il valore invece di sostituirlo. Rifacendo i tre moduli sulla stessa
cartella: **da 12 righe a 54 su 56**, riquadro di cartella applicato a 54, 13 valori distinti.

È lo stesso difetto del template `#13`: uno step ha più padroni, e chi scrive per ultimo cancella
il lavoro degli altri se non si limita alla propria parte.

### Le anteprime sparite: `resolve()` esce dallo specchio per definizione

Nella pagina dell'orientamento le anteprime non caricavano più: **403 «percorso non consentito»**.
La causa è la difesa contro il path traversal, scritta così:

```python
path = (folder / name).resolve()
if not path.is_file() or folder.resolve() not in path.parents:
    return 403
```

Finché `folder` era la cartella originale funzionava. Da quando le immagini si servono dallo
specchio di lavoro, no: **lo specchio è fatto di symlink**, e `resolve()` li segue fino al file
vero sull'SSD — che sta fuori dallo specchio per costruzione. Quindi ogni anteprima di una cartella
**non ruotata** veniva rifiutata, mentre quelle ruotate (dove lo specchio contiene copie vere)
passavano: è per questo che `prova_2` mostrava le immagini e gli altri no.

Il controllo va fatto sul **nome**, non sul percorso risolto: niente percorso assoluto, nessun
segmento `..`, e poi il file deve esistere. Raccolto in `_immagine_nella_cartella()` e usato nei
tre punti che avevano lo stesso schema — anteprima, ritaglio del marker, ritaglio della depth.

Verifica: 12 anteprime su 12 per ciascuno dei tre progetti (378, 136 e 56 fotogrammi), e il
ritaglio del marker risponde 200.

## 8-sexvicies. Dodici immagini su trentasei, e perché

Sul Philips Affiniti 70 (`prova 3`, 36 immagini) la sezione depth ne mostrava dodici e basta.
Non è un difetto: è una conseguenza, ma non era detta da nessuna parte.

Il modulo gira su un campione perché generare i candidati costa minuti. Dove la depth è **scritta
nell'interfaccia**, il campione basta: da lì si propaga il riquadro e la cartella si copre in mezzo
secondo per immagine. Dove invece la depth è **letta dalla scala** — ed è il caso di questo
Philips, tutte e dodici le righe sono `mode: scale` — il numero cambia posto ad ogni fotogramma e
un riquadro non si può propagare. È la stessa regola che vale per la propagazione: giusta, ma
lasciava la sezione a dodici immagini senza spiegazione.

Ora la sezione lo dice in testa — «12 immagini su 36 hanno una depth — qui la depth viene letta
dalla scala…» — e nel caso non propagabile offre l'unica strada che resta: *Cerca la depth su tutte
le 36*, cioè il modulo su ogni immagine.

**E la conferma.** `POST /depth/confirm` marca lo step confermato **senza toccarne il valore**:
`POST /steps/<id>` sostituirebbe il valore, e lì dentro vivono la rilettura su tutta la cartella e
le correzioni — confermare non deve cancellare niente. Da quel momento quelle depth valgono come
confermate e lo studio della scala le usa come dato certo, che è il collegamento fra le due
sezioni.

**Attenzione ai valori, però.** Su questa cartella tre depth su dodici non tornano col nome del
file, e il motivo è l'OCR che perde il punto decimale:

| file | atteso | letto | OCR |
|---|---|---|---|
| `LR_35` | 35 mm | **350 mm** | `35cm` |
| `LR_30` | 30 mm | **300 mm** | `30cm` |
| `LRUD_30` | 30 mm | **300 mm** | `30cm` |
| `LRUD_35` | 35 mm | 35 mm | `3.5 cm` |

Lo stesso valore viene letto `3.5 cm` su un fotogramma e `35cm` su un altro. Trenta centimetri su
una L12-3 non esistono: vanno corretti prima di confermare.

### Le tacche si spostano, e i gesti si vedono

Due cose rendevano il tool poco intuitivo.

**Un clic su una tacca la cancellava.** È l'ultima cosa che ci si aspetta da un oggetto che
sembra afferrabile: il gesto naturale su una tacca fuori posto è trascinarla. Ora la tacca **si
trascina**, ha una presa quadrata visibile sulla colonna, e per toglierla c'è una **×** che
compare passandoci sopra. Per il modulo spostare una tacca è toglierla da dov'era e rimetterla
dove sta ora, quindi la correzione viaggia come `ticks_del` + `ticks_add` — il contratto che ha
già.

**I gesti erano in un paragrafo.** Nessuno legge un paragrafo mentre corregge. Al suo posto una
legenda a chip, sempre in vista: *trascina · alt+trascina · doppio clic · × sulla tacca · doppio
clic sul numero · shift+↑↓ · ← → · cmd+Z*.

Aggiunte, per lo stesso motivo:

- **doppio clic** sull'immagine aggiunge una tacca — più facile da scoprire di shift+clic, che
  resta perché è più rapido quando se ne aggiungono parecchie;
- **«Tacche a passo costante»**: le tacche di un righello sono equidistanti, e quando ce ne sono
  di storte rifarle tutte dal passo costa un comando invece di dieci trascinamenti;
- l'elenco laterale porta ora segnali stretti (`corr`, `prop`, `·`, `~`) invece di parole intere:
  i nomi dei file tornano leggibili per intero — erano diventati «ne…», «A…».

### «0/36» per cinque minuti: un contatore che nessuno riempiva

Lanciando la ricerca della depth su tutte le immagini la barra restava a `0/36` e sembrava
bloccata. Non lo era — il sottoprocesso girava, 25% di CPU — ma il lavoro dichiarava un totale
che nessuno aggiornava mai: il modulo depth è un sottoprocesso solo, scrive i suoi CSV alla fine
e nel frattempo non dice niente.

Un contatore fermo è peggio di nessun contatore: fa credere a un blocco e invita a premere di
nuovo, che è esattamente quello che è successo (due run sovrapposte sulla stessa cartella). Ora
il lavoro non dichiara nessun totale e mostra il tempo che passa — «il modulo cerca la depth su 36
immagini (non riporta avanzamento, solo il tempo) — 2m 14s» — che almeno si muove.

## 8-septvicies. L'unità di misura è di cartella, non di fotogramma

Sul Philips le depth uscivano incoerenti: `3.5 cm` diventava 35 mm, `3.0 cm` letto `30cm`
diventava 300. Il convertitore cm→mm faceva il suo dovere; era **il testo** a essere sbagliato,
perché l'OCR perde il punto decimale. Ma l'incoerenza si vede solo guardando la cartella intera,
ed è lì che va cercata: la macchina scrive la depth sempre allo stesso modo, e chi devia di solito
non è una depth diversa — è la stessa letta male.

`GET /depth` restituisce ora una lettura di coerenza con due segnali, tenuti distinti perché uno
permette una proposta e l'altro no.

**Valore fuori scala.** Se diviso dieci cade nella scala della cartella, è un punto decimale
perso: `30cm` dove le altre dicono `3.0 cm`. Si propone il valore diviso, e la sezione lo applica
con un comando solo.

**Forma con una cifra in meno.** `5 cm` dove il resto della cartella scrive `d.d cm` ha perso una
cifra, ma *quale* non si sa: 50 mm è un valore plausibile e nessuna proposta sarebbe onesta. Si
segnala e si lascia decidere.

Il discrimine fra i due non è la forma dell'etichetta ma **il numero di cifre**: in questa stessa
cartella `10 cm` non ha la virgola come le altre, eppure vale davvero 100 mm ed è il fondo scala
della sonda. Ha però lo stesso numero di cifre di `3.5`, mentre `5 cm` ne ha una in meno. La prima
versione della regola guardava la forma e bocciava tutti i `10 cm`.

**Misura** su `prova 3`, 36 immagini, confronto con i nomi dei file:

| | tornano | no |
|---|---|---|
| prima | 29 | 7 |
| dopo il comando «Correggi i 5 valori fuori scala» | **34** | 2 |

I due che restano sono i `5 cm`, segnalati e lasciati a mano — ed è giusto così: la cifra persa è
un `3` che nessuna regola può indovinare. La scala della cartella diventa
`30·35·40·45·50·60·70·80·100 mm`, che è una ladder Philips pulita.
