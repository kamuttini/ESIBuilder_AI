# Il budget di 1° è raggiungibile: misurato su 61 aghi tracciati a mano

Data: 2026-09-15.

## Il risultato

Camilla ha tracciato a mano 61 aghi su 29 fotogrammi. Confrontando l'angolo di ogni ago con
l'angolo legacy `#23` più vicino della sua configurazione (ammessi i quattro flip, perché `#23`
è in convenzione No Flip):

| | n | mediana | entro 1° | entro 2° |
|---|---|---|---|---|
| prima del filtro per sonda | 43 | 1.78° | 28% | 53% |
| **dopo il filtro per sonda** | **61** | **0.94°** | **52%** | **80%** |

La mediana è **sotto il grado**. Dentro quel numero ci sono l'arrotondamento ai pixel interi del
codice legacy (fino a 0.238° da solo), la variabilità del tracciamento a mano, e il fatto che
l'ago inserito non coincide perfettamente con la linea guida nominale.

Quindi: la geometria portata è corretta, `#23` è recuperabile dall'immagine entro la tolleranza
chiesta, e il collo di bottiglia è **solo** il trovare l'ago.

## Cos'era il filtro per sonda

Un'acquisizione copre spesso più sonde in sottocartelle — `0. LA332`, `1. CA541`, `2. LA523`,
`3. TRT33` — e l'accoppiamento configurazione-acquisizione le trattava come un blocco unico. Una
configurazione per una sonda veniva quindi misurata sugli aghi di un'altra, che ha angoli guida
diversi. Fino a 18 configurazioni rivendicavano la stessa acquisizione.

Il filtro estrae il codice sonda dal nome della configurazione (LA523, CA541, SI2C41…) e tiene
solo i fotogrammi la cui sottocartella lo nomina; se nessuna lo nomina, l'acquisizione è a sonda
singola e non filtra niente.

Non è un dettaglio di pulizia: **ha dimezzato l'errore mediano**, da 1.78° a 0.94°.

## Il rilevatore resta il problema

Sui giudizi di Camilla per la pipeline raffinata: 39 segmenti giusti su 104, il **38%**.
Il raffinamento riduce l'uscita del 61% ma non alza la percentuale di correttezza — i filtri
scartano anche aghi buoni, perché per tenerne due bisogna saper scegliere quali.

Materiale disponibile per addestrare: **61 aghi con coordinate** e **355 giudizi per segmento**
(251 sulla versione grezza, 104 su quella raffinata).
