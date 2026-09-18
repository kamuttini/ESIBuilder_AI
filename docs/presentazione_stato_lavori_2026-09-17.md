# Presentazione stato lavori ESIBuilder AI

Stato aggiornato al **17 settembre 2026**. Precedente: `presentazione_stato_lavori_2026-04-10.md`.

## 1) Executive summary

Ad aprile avevamo tre blocchi di riconoscimento (vendor, sonda, rettangolo) e una pipeline che produceva CSV.
Oggi abbiamo **un'app locale che accompagna una cartella di acquisizioni lungo tutti gli 11 step della
configurazione**, con l'AI che propone e l'utente che conferma o corregge, e un writer `.fss` che rigenera
i file legacy identici byte per byte.

Cosa è cambiato da luglio:
- **scala** (righe #18-#21): da blocco fragile a stadio della pipeline con consenso di setup e criterio zero-regressioni misurato;
- **orientamento**: detector del marker integrato in pipeline, migliorato dalle correzioni umane (71 → 97 box su 112, nessuna regressione);
- **linee guida** (righe #22-#23) e file `.ndg`: nuovo blocco, geometria verificata sull'archivio storico, rilevatore aghi come proposta da confermare;
- **template sonda** (riga #14): dataset da 93k fotogrammi e rete che si astiene quando non è sicura, al posto di un resolver che su macchine nuove sbaglia nell'88% dei casi;
- **app** (`tools/app/`): wizard, progetto su disco con invalidazione mirata, import con dedup, divisione L/T, editor di box, lente per il secondo schermo;
- **strumenti di lavoro**: tool di revisione snella con feedback strutturato, monitor dello stato per step.

Volume di lavoro: 269 commit, di cui 186 a settembre; circa 59.000 righe aggiunte rispetto al master di luglio.

**Prima prova end-to-end su una configurazione reale (18/09, `docs/prova_end_to_end_setup347_2026-09-18.md`)**:
cartella 247 di ACQUISITION ELABORATION → app → `setup_347.fss` → confronto con il legacy. Senza intervento
umano 7 righe su 10 di geometria e riconoscimento coincidono con il legacy entro pochi pixel (vendor, sonda,
rettangolo, template #13 e #14, i 4 box di orientamento, risoluzione). La depth ha letto l'etichetta sbagliata
(guadagno invece di «18 mm»); con **una** correzione nell'app le 14 depth tornano identiche al legacy e 6
calibrazioni di scala su 14 sono entro lo 0,4%. Nella notte è stato scritto lo step **Soglie** che mancava:
le soglie di match ora si misurano sui fotogrammi come nel vecchio tool e arrivano nell'ordine di grandezza del
legacy (#13 1,1·10⁸ contro 1,4·10⁸), e da lì nascono anche #15 e #17 (riconoscimento simulato 60/60). Restano
vuote o da confermare: verso e 8 depth della scala, linee guida. Tempo totale sotto i 10 minuti.

## 2) Stato per blocco (righe `.fss`)

| Blocco | Righe | Stato | Numero che conta |
|---|---|---|---|
| Vendor | #02 | produzione | acc test 0.982, 414/426 cartelle automatiche |
| Sonda | #03 | operativo con review | acc test 0.848, 404/426 automatiche |
| Rettangolo ecografico | #11 | produzione | IoU test 0.917 |
| Template ecografo | #13 | reti per vendor + resolver storico | val IoU 0.65-0.89 sui vendor grandi, coda lunga scoperta |
| Soglie di match | TH in #13-#17 | **nuovo (18/09)**: misurate sui fotogrammi, positivi contro negativi | sul 347 stesso ordine del legacy, 60/60 riconosciuti |
| Schermate proibite | #15 | **nuovo (18/09)**: template da ciò che distingue la schermata | 2 schermate sul 347 |
| Box depth | #17 | **nuovo (18/09)**: dal riquadro dell'etichetta letto per frame | box entro 1-2 px dal legacy |
| Template sonda | #14 | **nuovo (17/09)**, rete generica pronta, soglia da decidere | IoU mediana 0.69; con astensione 91% corrette su 57% delle config nuove |
| Orientamento | #16 | integrato in pipeline, template matching | 97/112 box, zero regressioni, 272 cartelle SSD_n3 |
| Depth | #17-#18 | modulo autonomo integrato | validazione batch reale ancora da fare |
| Scala | #19-#21 | stadio pipeline, consenso di setup | none 31% → 0% con rete; calib_ok su accepted 90.9% senza rete |
| Linee guida | #22-#23 | **nuovo**, proposta da confermare | 30% entro 1°, 47% entro 3° su acquisizioni mai viste |
| Kit `.ndg` | file esterno | lettura/scrittura verificata | 390/390 file riscritti identici |
| Biplane a griglia | `.grid` | **non coperto** | 2/35 acquisizioni con pallini colorati rilevabili |
| Writer `.fss` | tutte | pronto | 10/10 legacy rigenerati identici (verifica del 9/9) |
| App wizard | — | usabile, 13 progetti di prova + il 347 | primo `.fss` reale generato con DB_echo il 18/09 (setup 347) |
| Windows / deploy | — | **non iniziato** | — |

## 3) Cosa mostrare (demo, in ordine)

1. **App** (`tools/app/server.py --port 8800`), progetto `prova_2` (BK Specto, 56 immagini, 28 L + 28 T):
   import con dedup → codici precompilati da `encoding_struct` → divisione L/T → rettangolo proposto e
   correggibile → orientamento con i 4 envelope → depth e scala con righello disegnato.
   Messaggio: *un solo import, ogni step già proposto, la correzione è un gesto sull'immagine.*
2. **Anteprima `.fss` riga per riga** nello step "genera", anche se non completo: fa vedere che il formato è quello legacy.
3. **Monitor** (`artifacts/71_monitor/index.html`, doppio click): uno spazio per step con metrica, obiettivo e andamento.
   Messaggio: *ogni blocco ha un numero e un criterio di uscita.*
4. **Review template sonda** (`artifacts/40_outputs_eval/probe_template_line14_review_20260917/index.html`):
   rosso il box legacy, verde quello ricostruito. Fa vedere che la GT si rivede, non si prende per oro colato.
5. **Galleria aghi** (`artifacts/92_guides/annotazione_varieta.html`): il rilevatore propone, l'utente conferma.

Regola per la demo: **mai cliccare su un progetto vero** dal pannello browser (un POST di prova ha già cancellato ore
di lavoro). Usare una copia in scratch o un server su un'altra porta.

## 4) Il metodo, che vale quanto i risultati

- **Il sistema non forza mai una predizione incerta**: `accepted` / `review` / `reject`. Sul template sonda i
  fallimenti arrivano con confidenza bassa: la rete tace invece di scrivere un numero sbagliato.
- **Zero regressioni**: ogni modifica agli algoritmi è accettata solo con confronto per cartella senza peggioramenti.
  Sulla scala questo ha fatto revertire tre patch che miglioravano una metrica e ne rompevano un'altra.
- **La GT si audita**: 5.290 righe di scala, 98,5% coerenti. Il backlog P0/P1 di aprile era un falso problema, il ritardo stava nel modello.
- **Split leak-free per layout**, non per cartella: le coppie L/T della stessa macchina stanno nello stesso split, altrimenti i numeri sono gonfiati.
- **Ogni correzione umana è un dato** con area, tag, verdetto, predizione e correzione: rientra nei moduli e nel set di regressione.
- **Il formato legacy è il contratto**: writer verificato per andata e ritorno, `.ndg` riscritti byte per byte, quality gate `compare_fss.py`.

## 5) Cosa ha funzionato e cosa no (da dire a voce, senza nasconderlo)

Funzionato:
- consenso di setup sulla scala: errore massimo su `mm_per_px` da 519% a 19,8%, medio da 40,9% a 1,79%;
- rete heatmap come ripiego, non come vincolo: copre le 5 cartelle perse intere (133 righe) al costo di 2,7 punti di purezza;
- le due regole che hanno migliorato il rilevatore aghi vengono da Camilla che guardava gli errori, non da feature progettate a tavolino;
- rete unica generica per il template sonda: 13 vittorie contro 12 rispetto alle reti per vendor, delta mediano nullo, 16 modelli in meno da mantenere.

Non funzionato (misurato, non stimato):
- regressori scalari sulla scala: abbandonati, metà del target era rumore di etichettatura;
- bande di ricerca più larghe per il righello: peggiorano tutto, revertite;
- classificatore per riordinare i candidati aghi: AUC 1.00 in validazione, identico al grezzo su configurazioni mai viste (troppe poche scene);
- indicatore di reticolo scritto a mano per le griglie biplane: non separa. È l'argomento per la rete.

## 6) Rischi aperti

- **End-to-end dimostrato su una sola configurazione** (setup 347): il quality gate `compare_fss.py` non passa perché confronta numero per numero, e tre blocchi (depth su UI Esaote 80xPro, verso della scala, linee guida) hanno sbagliato in modo circoscritto e riproducibile. Servono altre 3-5 coppie configurazione/acquisizione prima di parlare di percentuali.
- **Consolidamento del codice**: il master è fermo al 10 luglio, il lavoro vive in 9 worktree e nel checkout principale ci sono modifiche di agosto non committate. Va fatto un merge e un tag prima di allargare il team.
- **Biplane a griglia**: calibrazione su file `.grid` separato, rilevatore della matrice di pallini da scrivere. Chi generasse solo il `.fss` avrebbe una configurazione a metà.
- **Depth per fotogramma nelle linee guida**: oggi la stessa distanza viene proposta per tutte le depth.
- **Famiglie UI scoperte** sul template sonda (Hitachi Aloka/Arietta, Koelis, Biopsee, Mindray DC70, Siemens, Toshiba): servono configurazioni dal volume `n3` o si accetta la review.
- **Windows**: inferenza CPU/ONNX, versione Tesseract fissa, packaging. Non iniziato; il training resta su Mac.
- **Volumi esterni instabili**: durante le eval il disco è caduto più volte, va previsto un mirror locale dei dataset.
- **Vecchi rischi ancora validi**: classi sonda rare, orientamenti speculari legacy con riga #16 a 4 copie.

## 7) Decisioni chieste alla capa

1. **Priorità**: chiudere prima il ciclo end-to-end su 3-5 configurazioni reali (linear/convex), o coprire subito le biplane a griglia?
2. **Soglia di astensione** per il template sonda: 0,2 dà 91% di correttezza su 57% delle configurazioni nuove; più bassa copre di più e sbaglia di più. È una scelta di policy, non tecnica.
3. **KPI di go-live** da fissare insieme: % di righe accettate automaticamente, % di `.fss` compatibili con i legacy, tempo per configurazione rispetto al vecchio ESIBuilder.
4. **Finestra con Francesca** per le domande di dominio aperte: dimensione del ritaglio template, righe #13/#14 vuote, versione in #01, convenzione per orientamento assente, cartelle con entrambi i piani, testo atteso in #13 sul BK Specto.
5. **Pilota**: quale collega e quali configurazioni per il primo uso reale dell'app.
6. **Quando partire con Windows**: prima del pilota (deploy vero) o dopo (pilota su Mac).

## 8) Prossimi passi proposti (4-6 settimane)

1. Ripetere la prova end-to-end su altre coppie della colonna «Numero Aquisition elaboration» (FSS 365, 251, 374) e correggere le tre cause trovate sulla 347: prior dell'etichetta «mm» nella depth, verso della scala dal gruppo NF, riga #17 dal box della depth.
2. Merge dei rami in master, tag, aggiornamento di `ROADMAP.md` (ferma al 8 luglio) e `CLAUDE.md`.
3. Agganciare la rete del template sonda allo stadio di raffinamento già esistente per la #13 (template matching + dark trim) e fissare la soglia.
4. Depth per fotogramma nelle linee guida, che è l'approssimazione più grossa della proposta attuale.
5. Etichette per le biplane con la galleria `build_biopsy_dataset_html.py`, poi rete e rilevatore del reticolo.
6. Far alimentare `feedback/inbox.jsonl` dall'app, così le correzioni fatte nel wizard rientrano nei moduli.

---

## Talk track (2 minuti)

"Ad aprile riconoscevamo vendor, sonda e rettangolo e producevamo CSV. Oggi c'è un'app che prende una cartella di
acquisizioni e la porta lungo tutti gli step della configurazione: l'AI propone ogni valore, l'utente conferma o
corregge con un gesto sull'immagine, e il file `.fss` esce nel formato legacy, verificato byte per byte sui file
storici. Da luglio abbiamo chiuso la scala, integrato l'orientamento, aperto il blocco delle linee guida e del kit
`.ndg`, e costruito una rete per il template della sonda che si astiene quando non è sicura invece di sbagliare
con sicurezza. Il metodo è sempre lo stesso: nessuna predizione forzata, nessuna modifica senza confronto
per cartella, ground truth auditata prima di dare la colpa al modello. Quello che manca è la prova end-to-end:
nessuna configurazione reale è ancora arrivata fino al file generato e validato. È il prossimo lavoro, e ho bisogno
di tre decisioni: la priorità fra chiusura del ciclo e biplane, la soglia di astensione, e una finestra con Francesca
per le domande di dominio."
