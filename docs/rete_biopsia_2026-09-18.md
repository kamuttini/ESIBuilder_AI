# La rete che riconosce le immagini di biopsia — stato al 18/09

## Le etichette

Due giri di etichettatura di Camilla, che non sono la stessa cosa e non vanno trattati uguale.

Il primo: 146 immagini dentro cartelle che si chiamano «biopsia». Lì l'etichetta si propaga a
tutta la cartella, perché quelle cartelle contengono quella scena a profondità diverse — e
infatti **nessuna delle 48 cartelle è tornata mista**. 146 decisioni ne etichettano 823.

Il secondo: 77 file con «biops» nel nome ma dentro cartelle qualsiasi. Lì **non si propaga**:
accanto ci sono fotogrammi di tutt'altro, e in quattro acquisizioni le due risposte sono
davvero diverse. Valgono una per una, ma portano **32 acquisizioni nuove**.

Dataset: 5427 immagini, 20 acquisizioni positive in addestramento contro le 9 del primo giro.

## Perché il secondo giro è servito

Con nove macchine la media delle precisioni ballava di trenta punti fra un'epoca e l'altra —
0.31, 0.37, 0.49, 0.32, 0.71 — perché la rete imparava l'aspetto delle macchine e a ogni epoca
cambiava quali acquisizioni di validazione le somigliavano. Con venti l'oscillazione si
dimezza: da 0.31-0.71 a 0.37-0.59.

## Il riferimento, misurato

Prima rete, sul test mai visto (8 acquisizioni positive, 3 negative):

| | |
|---|---|
| acquisizioni positive trovate | **8 su 8** (basta un fotogramma buono per calibrare) |
| precisione per immagine | 0.39 |
| richiamo | 0.72 |
| falsi concentrati | 76 su 111 in **una sola** acquisizione negativa |
| negativi difficili presi per buoni | **38 su 46** |

Trova tutte le acquisizioni, ma ha imparato in parte «questa è una schermata di biopsia»
invece di «ci sono i pallini».

## Il motivo, e non è la quantità di dati

Un pallino misura **6 pixel** nell'immagine originale (misurato su sei positivi: 3, 4, 5, 6, 8,
8 px). La catena di preparazione lo riduce due volte: il ritaglio è limitato a 512 px di lato
(da 1920) e la rete lavora a 224. Quel pallino arriva alla rete come **0.7 pixel**.

Cioè: la prova che la rete dovrebbe cercare viene distrutta prima che la veda. Questo spiega
perché il plateau resta intorno a 0.45 comunque, e perché la rete si aggrappa all'aspetto
generale della schermata — che è l'unica cosa che sopravvive alla riduzione.

## Che cosa fare, in ordine di costo

1. **Togliere il tetto di 512 al ritaglio e alzare l'ingresso della rete.** A 1024/512 il
   pallino resta 3 px. Costa memoria, e su questa macchina la memoria è il vincolo.
2. **Dare alla rete la mappa top-hat invece dell'immagine** (o come canale in più). Il top-hat
   concentra l'energia del pallino e la lascia visibile anche dopo la riduzione, mentre
   nell'immagine grezza il pallino si perde contro il tessuto chiaro.
3. **Addestrare su tasselli a risoluzione nativa** invece che sull'immagine intera. È il modo
   più fedele al problema — un pallino è un dettaglio locale — e il più costoso da mettere in
   piedi.

Nessuna di queste è stata provata: la misura del pallino è arrivata mentre la corsa di
confronto era in mezzo, e cambiare la catena mentre si misura avrebbe reso il confronto
inutile.
