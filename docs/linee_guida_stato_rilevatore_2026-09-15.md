# Rilevatore dell'ago: cosa dicono le etichette di Camilla

Data: 2026-09-15. Materiale in `artifacts/92_guides/etichette/`.

## Cosa abbiamo ora

- **251 giudizi per singolo ago** su 82 fotogrammi: 131 sull'ago, 120 sbagliati.
- **43 aghi tracciati a mano** con le coordinate, su 19 fotogrammi distinti.
- Quattro regole di dominio: uno o due aghi per immagine, paralleli se due, i pezzi che si
  toccano sono lo stesso ago, e gli aghi hanno le creste mentre i riflessi in acqua no.

## Il rilevatore classico

Precisione complessiva **52%** (131 su 251). Ma il dato non è uniforme, e il perché conta:

| | segmenti giusti | sbagliati | precisione |
|---|---|---|---|
| fotogrammi dove Camilla ha tracciato l'ago | 22 | 52 | **30%** |
| fotogrammi dove non l'ha tracciato | 109 | 68 | **62%** |

Camilla ha tracciato l'ago **dove il rilevatore sbagliava**, che è il comportamento sensato e
rende quei 19 fotogrammi il sottoinsieme difficile, non un campione.

Su quel sottoinsieme, l'errore vero contro l'ago tracciato:

| | mediana | entro 1° | entro 3° |
|---|---|---|---|
| prima scelta | 46.4° | 5/19 | 6/19 |
| migliore della rosa | 46.1° | 6/19 | 6/19 |
| dopo le regole di dominio | 31.9° | 7/19 | 7/19 |

**Quando sbaglia, sbaglia di tanto**, non di poco: la distribuzione è bimodale, o azzecca entro
il grado o finisce su tutt'altro. È una buona notizia per una misura di confidenza — i due casi
sono separabili — ed è una cattiva notizia per l'idea di raffinare una scelta approssimativa.

## Le regole di dominio

Unione dei pezzi e vincolo di parallelismo tagliano i segmenti da 273 a 113 (meno della metà)
tenendo la precisione al 41%, e migliorano l'errore sul sottoinsieme difficile da 46° a 32°.
Il criterio delle creste invece **peggiora** (41% → 32%): l'implementazione attuale misura la
cosa sbagliata, non la regola. Camilla le riconosce a occhio, quindi il segnale c'è.

## Un difetto dell'accoppiamento configurazione-acquisizione

Confrontando gli aghi tracciati con gli angoli legacy, lo scarto mediano è 1.78° ma con una
coda pesante (media 5.03°). Le peggiori sono tutte su **acquisizioni contese**: una sola
acquisizione viene rivendicata da fino a 18 configurazioni diverse, e i loro angoli legacy sono
diversi, quindi al massimo una ha ragione.

Il criterio attuale — nome simile più risoluzione dichiarata presente fra le immagini — non
distingue fra configurazioni della stessa famiglia. Va aggiunto qualcosa che le separi, ad
esempio l'ID sonda `#03` o il rettangolo `#11` confrontato con quello misurato sulle immagini.

## Prossimi passi

1. Copiare in locale i `.fss` e `.ndg` legacy: `SSD_esi1_n1` si smonta da solo di continuo e
   blocca ogni analisi. Sono pochi MB.
2. Sciogliere le acquisizioni contese, che oggi inquinano ogni misura contro il legacy.
3. Addestrare il rilevatore. Il materiale c'è: 43 aghi con coordinate e 251 giudizi, concentrati
   proprio sui casi che il metodo classico sbaglia.
