# Biplane: come strutturare lo studio delle linee guida

Punto della situazione e piano, scritto mentre si prepara la rete per le immagini di biopsia.

## Perché le biplane non passano dalla pipeline degli aghi

Per lineari e convex la calibrazione si fa con l'ago in acqua: si ricalca l'ago, ne escono
`#23` (angolo) e `#22` (distanza dal centro), e il `.ndg` del kit moltiplica la misura su tutte
le linee di quell'angolo. Per le biplane no: si usano **le linee di biopsia che disegna
l'ecografo**, e la calibrazione è una cosa diversa nel vecchio ESIBuilder — pagina
`WdgPageGridCalibration`, abilitata solo se il `.ndg` dichiara la griglia **e** il tipo sonda
non è `PROBE_TYPE_TRANS_L`. Cioè: **la griglia si calibra sul piano T**.

E il risultato non finisce nel `.fss`: finisce in `setup_<ID>.grid`, scritto da `QFileFssGrid`.
Chi generasse solo il `.fss` avrebbe una configurazione a metà. Dettagli del formato in
[ndg_studio_2026-09-16.md](ndg_studio_2026-09-16.md).

## Le due forme, che non sono la stessa cosa

| forma | dove | che cosa si misura |
|---|---|---|
| **linea di pallini** | anche lineari/convex (`UA1250`, `UA1341`) | una retta: angolo e distanza, come un ago |
| **matrice di pallini** | biplane a griglia (piano T) | un reticolo: il rettangolo della griglia e quali righe/colonne si vedono |

La prima è già coperta: `detect_biopsy_dots.detect` la trova quando è **colorata**. La seconda no.

## Che cosa serve misurare, per la matrice

Da `QFileFssGrid` e da `qgridobj.cpp`, per ogni depth e per ogni flip:

- `CircleRect` — il rettangolo del **cerchio sonda**, in pixel e in mm
- `GridRect` — il rettangolo della **griglia**, in pixel e in mm
- `bVisibleMatrix` — prima e ultima riga, prima e ultima colonna visibili nell'immagine
- i pixel ratio e la depth in mm, che vengono dallo step scala

La matrice del kit (quante righe e colonne, con che passo, quali fori esistono) **non si
misura**: sta nel `kit_needle_guide_<ID>.grid`, ed è un ingresso. Nell'archivio sono 53 file,
quasi tutti 13×13 a 5 mm.

## Stato del riconoscimento: misurato oggi, non stimato

Sulle 831 immagini delle 56 cartelle «biopsia» dell'archivio:

| | |
|---|---|
| guida a pallini **colorata** trovata | 90 immagini, **2 acquisizioni su 35** |
| guida a pallini **chiara** trovata (rilevatore nuovo) | 9 immagini |
| niente | 732 |

Quindi quasi ovunque i pallini sono bianchi o grigi, e la maschera a saturazione non li vede.
Ho aggiunto `bright_dot_mask` (top-hat: massimi locali più stretti dell'elemento strutturante)
e `detect_bright`, che trovano i pallini ma non bastano a decidere: su una matrice vera la
maschera dà 400 componenti buone e il RANSAC ne allinea 60-80, ma **una matrice non è una
retta** e i controlli su span e regolarità la scartano.

Provato anche un indicatore di reticolo — la dispersione del passo fra pallini vicini — e
**non separa**: 0.71 e 1.00 su due matrici vere contro 0.57-0.79 su ecografie qualsiasi. Il
rumore del tessuto produce gli stessi puntini piccoli e chiari.

**Conclusione: la regola scritta a mano non ci arriva, e questo è l'argomento per la rete.**

## La rete: come strutturarla

Stessa forma di quella degli aghi, che funziona: ResNet18 su ritaglio `RECT_ECHO` in
letterbox, split leave-one-acquisition-out, policy a due soglie (accetta / rivedi / scarta).

Quello che cambia è **da dove vengono le etichette**. Il nome della cartella non basta: in
`ACQUISITION ELABORATION` le cartelle «biopsia» contengono schermate di Windows, pannelli
dell'interfaccia ed ecografie senza pallini. Addestrare su quel nome insegnerebbe alla rete che
cosa c'era scritto sulla cartella.

Perciò: `tools/needle/build_biopsy_dataset_html.py` costruisce una galleria di 146 immagini (6
per acquisizione, 35 acquisizioni) dove si segna a clic se i pallini ci sono. Esporta un JSON
con `file` e `pallini: true|false`, che è l'ingresso del trainer.

    python3 tools/needle/build_biopsy_dataset_html.py \
        --root "/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION" \
        --output artifacts/92_guides/etichette_biopsia.html

Con le etichette in mano l'ordine è: prepara i ritagli → addestra → misura su acquisizioni mai
viste → aggancia allo step import come già fa il modello aghi, con la sua banda di confidenza.

## Poi, per la calibrazione vera

1. **Distinguere linea da matrice** sull'immagine riconosciuta: sono due strade diverse e la
   sceglie il `.ndg` (intestazione `(-2, 2)`), non l'immagine.
2. **Linea**: riusare la misura degli aghi — angolo e distanza dal centro, le stesse `#22`/`#23`.
3. **Matrice**: trovare il reticolo e ricavarne `GridRect` e `bVisibleMatrix`, più il cerchio
   sonda. È l'unico pezzo davvero nuovo, e va scritto contro `setup_<ID>.grid` dell'archivio:
   29 file da cui si possono prendere i valori giusti e confrontarli, come si è fatto con i
   `.fss` per gli aghi.
