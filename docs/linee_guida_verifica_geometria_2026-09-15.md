# Linee guida (#22/#23): la geometria è verificata, e il budget di 1° è raggiungibile

Data: 2026-09-15. Blocco: opzione A, riempimento delle righe `#22` e `#23` del `.fss`.

## Il punto

La geometria del vecchio software è stata portata in Python
(`tools/needle/guides_geometry.py`) e **verificata contro le immagini**, non solo contro sé
stessa. La verifica dice anche che la tolleranza di 1° chiesta da Camilla è alla portata.

## Come è stata verificata

L'andata e ritorno (numeri → linea → numeri) chiude a 0.03° di mediana, ma non dimostra niente
sull'interpretazione: se avessi letto male il segno della distanza o l'origine delle coordinate,
chiuderebbe lo stesso. Serviva un riscontro esterno.

Camilla ha giudicato a mano 35 rilevazioni dell'ago su fotogrammi veri, dicendo per ognuna se il
segmento trovato stava sull'ago o no. Incrociando il suo giudizio con gli angoli legacy delle
rispettive configurazioni — ammettendo i quattro flip, perché `#23` è memorizzata in convenzione
No Flip e il fotogramma può essere ribaltato:

| giudizio umano | n | scarto dall'angolo legacy più vicino |
|---|---|---|
| sull'ago | 19 | **mediana 0.88°** — 11 entro 1°, 12 entro 3°, 1 oltre 10° |
| sbagliata | 12 | **mediana 13.03°** — nessuna entro 1°, 7 oltre 10° |

Le due popolazioni sono nettamente separate. Se la lettura del formato fosse sbagliata, le
rilevazioni corrette **non** cadrebbero sugli angoli legacy: cadono, quindi il formato è letto
bene — origine delle coordinate, segno della distanza e convenzione dell'angolo compresi.

## Conseguenze

**Il budget di 1° è raggiungibile.** La mediana delle rilevazioni corrette è già 0.88°, e dentro
quel numero ci sono l'errore del rilevatore, l'arrotondamento ai pixel interi del codice legacy
(fino a 0.238° da solo) e il fatto che l'ago tracciato non coincide perfettamente con la linea.

**Attenzione a come si usa il confronto con gli angoli legacy.** Filtrando le rilevazioni a 1°
dall'angolo legacy la precisione va al 100% (11 su 11), ma quel filtro usa la risposta: non è
utilizzabile per calibrare una macchina nuova, dove `#23` è proprio ciò che si deve produrre.
Resta utilizzabile per la validazione, e come *prior* quando esiste già una configurazione con la
stessa coppia (sonda, kit): fra configurazioni della stessa coppia gli angoli distano 1.58° in
mediana, quindi il kit restringe il campo e la misura sull'immagine rifinisce.

## Cosa non va ancora

- **Precisione del rilevatore: 19 su 35 (54%).** Va alzata prima di poterci misurare sopra.
- **I convex vanno peggio dei lineari**: 1 su 6 contro 18 su 29.
- Gli errori hanno una firma riconoscibile: angolo quasi orizzontale (0° oppure fra -5° e -13°)
  e contrasto basso (mediana 25 contro 39 delle corrette). Sono bande di tessuto e linee della
  pelle, non aghi.
- **La selezione dei fotogrammi usa i nomi delle cartelle** e fa entrare immagini che non sono di
  calibrazione (Camilla ne ha segnalata una senza sonda in acqua). Va sostituita con il
  classificatore già addestrato, che è esattamente il suo mestiere.
