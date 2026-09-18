# Prova end-to-end: setup 347 dalla cartella 247 di ACQUISITION ELABORATION

Data: 2026-09-18. Prima configurazione reale portata dall'app (`tools/app/`) fino al file `.fss`
generato e confrontato con il legacy. Materiale in `docs/prova_setup347/`.

## Come è stata scelta

Il foglio `FSS` di `encoding_struct 2026 08 31.xlsx` ha la colonna «Numero Aquisition elaboration»:
per 114 configurazioni dice da quale cartella di `SSD_esi1_n3/ACQUISITION ELABORATION` è nata.
Fra queste ho preso la più semplice e recente con materiale completo: sonda lineare, 1920×1080
nativo, vendor ben coperto.

| | |
|---|---|
| configurazione | FSS 347, Esaote MyLab 80xPro F090201 (Rev02), L4-15, Bari oncologico, «ultima giusta» |
| legacy di riferimento | `SSD_esi1_n1/Esaote_MyLab_80xPro_F090201(Rev02)-L4-15-Oncologico_Bari/DB_setup/setup_347.fss` (copia in `artifacts/92_guides/legacy_cache/`) |
| acquisizione | `SSD_esi1_n3/ACQUISITION ELABORATION/247. Esaote MyLab 80xPro F090201 (Rev02), L4-15 (Oncologico Bari)` |
| immagini usate | le 62 della radice: 14 depth × 4 orientamenti, 4 aghi (`1_75` … `4_60`), `FREEZE`, `_NEGATIVA`. Esclusa la sottocartella «Non usate» (67 frame grezzi scartati dall'operatore) e `NF/` (copia della radice) |

I nomi dei file (`LRUD_102.png`, `NF_18.png`) dicono orientamento e depth, ma l'app non li legge:
tutto quello che segue viene dai pixel.

## Come è stata fatta

Server dell'app su una porta separata (8811) con i progetti in una cartella di scratch, mai sui
progetti veri. Tutto via API, nell'ordine dell'utente: creazione progetto → import (dedup,
rotazione, vendor, sonda, rettangolo, #13, #14) → pipeline automatica (orientamento, depth,
scala) → codici → ricerca materiale aghi → linee guida → anteprima → genera → `compare_fss.py`.

Due passaggi:

- **A. tutto automatico**, nessun intervento umano (`setup_347_A_automatico.fss`);
- **B. una correzione**, quella che l'utente farebbe nell'app: il riquadro trascinato
  sull'etichetta «18 mm» di un frame, propagato a tutta la cartella; poi scala e linee guida
  ricalcolate (`setup_347_B_con_correzione_depth.fss`).

I codici (#01 versione, #02 ID ecografo, #05 kit) sono stati impostati a mano ai valori
dell'anagrafica: l'ID ecografo per Esaote è ambiguo fra più modelli (l'app propone 4 candidati, non
sceglie), il kit e la versione non si leggono dalle immagini.

## Risultato riga per riga

| riga | legacy | A automatico | B con correzione | esito |
|---|---|---|---|---|
| #02 ID_ECHO | 155 | a mano (4 candidati Esaote) | = | atteso: si sceglie |
| #03 ID_PROBE | 19 | **19** (conf 0.98) | = | ✅ |
| #04 PROBE_TYPE | 1 | **1** (da anagrafica) | = | ✅ |
| #06-#10 video | 0, 1920×1080, 1920×1080 | **uguali** | = | ✅ |
| #11 RECT_ECHO | 126\|338\|914\|1495 | 124\|340\|913\|1498 | = | ✅ entro 3 px |
| #12 GROUP_ORIENTATION | 4 | **4** | = | ✅ |
| #13 RECT_NAME_ECHO | 4\|98\|48\|255 | 3\|102\|49\|247 | = | ✅ entro 8 px |
| #14 RECT_NAME_PROBE | 107\|357\|128\|408 | 107\|358\|128\|407 | = | ✅ entro 1 px (rete nuova del 17/09) |
| #15 PROIBITED | 1 box (911\|601\|948\|655) | vuota | vuota | ❌ step senza automazione |
| #16 RECT_ORIENTATION | 4 box, es. NF 144\|389\|183\|813 | 4 box, NF 153\|397\|175\|804 | = | ✅ stessi 4 gruppi, box dentro il legacy di 8-9 px |
| #17 RECT_DEPTH | 14 gruppi, box 72\|613\|98\|641 | vuota | vuota | ❌ step senza automazione (il box giusto però ora è noto: è quello della correzione) |
| #18 VECT_DEPTH | 14 valori 18…166 | **150** (1 valore) | **14 valori identici** | ❌ → ✅ |
| #19/#20 PIXEL_RATIO | 14 valori | 1 valore sbagliato | vuote (riga incompleta) | ⚠ vedi sotto: 6 su 14 esatte |
| #21 SCALE_LINE | 14 segmenti, zero in alto | 1 segmento | vuota | ⚠ verso sbagliato |
| #22/#23 linee guida | 4 angoli 17.6/25.4/29.4/33.7 | proposte non promosse | proposte non promosse | ❌ angoli −15/13.7/21.0/23.5 |

`compare_fss.py` dà esito 1 (non compatibile) in entrambi i casi: confronta numero per numero,
non tollera i 2-3 px. Il file però ha 23 righe come il legacy, si genera e si rilegge.

### Cosa è andato bene

Sette righe su dieci di geometria e riconoscimento sono uguali al legacy entro pochi pixel, senza
alcun intervento: vendor, sonda, rettangolo, i due template (#13 e #14), i quattro box di
orientamento (tutti e quattro i gruppi trovati, copertura 1.0 del marker su 62/62 immagini),
rotazione e risoluzione. I box legacy sono sistematicamente 8-9 px più larghi dei nostri su #16:
è il margine che l'operatore lasciava, da mettere nel writer come margine fisso e da chiedere a
Francesca.

### Cosa è andato male, e perché

**Depth (il buco che ha fatto crollare il resto in A).** Il modulo ha scelto l'etichetta sbagliata:
«15» della scala TEI (guadagno) invece di «18 mm», e con l'unità dedotta ×10 ha scritto 150 mm su
tutte e 62 le immagini. Vedi `etichette_depth_LR_18.png`: verde il box legacy #17 sull'«18 mm»,
giallo quello scelto dal modulo. Con una sola depth la scala si è ridotta a una riga e le linee
guida a un'approssimazione. **Una correzione** (riquadro sull'etichetta, 10 secondi nell'app) ha
riletto 62/62 immagini e restituito **esattamente le 14 depth del legacy**.

**Scala.** Sulle 6 depth in cui il righello è stato agganciato, `mm_per_px` è entro lo 0,4% del
legacy (46, 65, 74, 102, 129, 166 mm). Le altre 8 sono interpolate dai vicini: bene in mezzo
(148 mm: +0,0%), male alle depth basse (18 mm: +80%, 28: +36%, 37: +15%), dove il righello ha
poche tacche. Due problemi strutturali:

- **il verso è deciso a maggioranza sulla cartella** (voto 6/10 «zero in basso»), ma la cartella
  contiene per costruzione i gruppi UD/LRUD ribaltati. Il legacy misura sul gruppo NF, zero in alto
  (y=140). Il verso va deciso sul gruppo NF, che l'orientamento ha già assegnato frame per frame;
- la **colonna x** è più a sinistra del legacy di 34-260 px, con scarto decrescente con la depth:
  l'operatore cliccava un'altra colonna del righello (probabilmente quella dei numeri). Da chiarire
  con Francesca quale colonna aspetta ESI.

La riga #21 non viene emessa perché incompleta (8 depth in review): è la policy voluta, il file non
mente. Nell'app le 8 si correggono una per una nello studio del righello.

**Linee guida.** I 4 frame aghi sono stati trovati dalla rete (4/4, punteggio > 0,99) ma la misura
sbaglia: angoli −15,0 / 13,7 / 21,0 / 23,5 contro 17,6 / 25,4 / 29,4 / 33,7. Il primo ha il segno
sbagliato. Coerente con quanto misurato il 16/09 (30% entro 1°): è una proposta da confermare.
Le distanze #22 dipendono dai pixel ratio, quindi ereditano gli errori della scala.

**Righe senza automazione (#15, #17).** Lo step delle schermate proibite e quello del box della
depth (#17) non hanno ancora un modulo: nell'app restano da tracciare a mano. Il caso #17 è
paradossale: il box è esattamente quello della correzione della depth, basta riusarlo.

## Passaggio C: le soglie e le righe #15 e #17 (stessa notte)

Le soglie `TH` erano a zero in tutti i blocchi, cioè «mai»: lo step Soglie non era mai stato
scritto. Scritto (`tools/app/thresholds.py`, logica della pagina Thresholds legacy: TH a metà fra
peggior positivo e miglior negativo, positivi e negativi presi dal progetto) e rilanciato sul 347
dopo aver marcato come proibite `FREEZE.png` e `_NEGATIVA.png` (un gesto nell'app):

| riga | legacy | C (soglie) | esito |
|---|---|---|---|
| #13 TH | 1,40·10⁸ | 1,13·10⁸ | ✅ stesso ordine; negativo = la `_NEGATIVA` (logo coperto di rosso), `FREEZE` scartata perché mostra ancora il logo |
| #14 TH | 3,00·10⁷ | 2,39·10⁷ | ✅ |
| #15 | 1 schermata, box 911\|601\|948\|655 (scarabocchio rosso in basso) | 2 schermate: `FREEZE` (parola «Frame»), `_NEGATIVA` (scarabocchio rosso sul logo) | ✅ automatico dalla differenza con i frame normali fuori dal rect |
| #16 TH | 1,16–1,21·10⁷ | 0,91–1,32·10⁷ | ✅; ESI simulato: 60/60 fotogrammi al gruppo giusto e solo a quello |
| #17 | 14 gruppi × 4, box 72\|613\|98\|641, template 19×17, TH 0,8–5,2·10⁶ | 14 × 4, box 73\|614\|99\|643, template 23×19, TH 0,7–4,6·10⁶ | ✅; ESI simulato: 60/60 alla depth giusta e solo a quella |
| DB_echo | 21 PNG | 21 PNG in `out/DB_echo/setup_347/` | ✅ |

File: `docs/prova_setup347/setup_347_C_con_soglie.fss`. Restano fuori, come prima: #19–#21 (scala:
8 depth in review e verso), #22–#23 (linee guida).

## Cosa se ne ricava

1. **Il ciclo end-to-end funziona**: cartella → app → `.fss` con 23 righe nel formato legacy → quality
   gate. Era il punto mancante del progetto.
2. **Con una correzione umana e due schermate marcate** il file ha giuste #02-#18 comprese le soglie,
   e 6 depth di scala su 14. Le righe ancora vuote o sbagliate sono note e circoscritte: #19-#21
   (verso e 8 depth), #22-#23.
3. **Tre correzioni da fare nel codice**, misurabili su questa stessa cartella:
   - depth: il box del legacy #17 (etichetta «mm») come prior contro le etichette del guadagno;
   - scala: verso dal gruppo NF, non dalla maggioranza;
   - ~~#17: costruire i gruppi dal box della depth propagato~~ — fatto nel passaggio C.
4. **Domande per Francesca**: margine dei box #16 (8-9 px), colonna del righello in #21.
5. Il metodo è ripetibile: la colonna «Numero Aquisition elaboration» dà altre 113 coppie. Le
   prossime: FSS 365 (Omega, convex, cartella 258), FSS 251 (stessa macchina, versione 2023,
   cartella 191), FSS 374 (Omega eXP LX3-15, cartella 263).

## Riproduzione

```bash
cd .claude/worktrees/fss-pipeline-monitoring-tool-ee56d1
OldSoftwareEsiBuilder/.venv-mps/bin/python tools/app/server.py --port 8811 \
  --projects-root <cartella di scratch> --encoding-struct "encoding_struct 2026 08 31.xlsx"
```

Poi: `POST /api/projects`, `POST /import` con la cartella copiata senza «Non usate», attesa del
job automatico, `POST /codes`, `POST /depth/box` con `{"name":"LR_18.png","box":{"top":72,"left":613,
"bottom":98,"right":641}}`, `POST /analyze_stages {"stages":["scala"]}`, `POST /guides/propose`,
`POST /generate {"force":true}`, `POST /compare {"reference": <legacy>}`.

Tempi sulla macchina di sviluppo (MPS): import 20 s, orientamento 2,5 min, depth 3,5 min, scala
30 s, correzione depth 10 s. Totale sotto gli 8 minuti.
