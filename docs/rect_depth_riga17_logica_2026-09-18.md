# La riga #17 `RECT_DEPTH`: che cos'è, come la costruiva il vecchio ESIBuilder, come generarla

Data: 2026-09-18. Fonti: `OldSoftwareEsiBuilder/{qfilefss.cpp,h, wdgpagedepthvalue.cpp, wdgpagedepthfind.cpp,
qfileprojectfss.cpp, wdgmanagetamplatesthresholds.cpp, cvmatchtemplate.cpp, general_define.h}` e i 390 `.fss`
in `artifacts/92_guides/legacy_cache/` più i `DB_echo` su `SSD_esi1_n1` (384 configurazioni con template).

## In una frase

La #17 è **il riconoscitore della depth corrente**: per ogni valore di depth della #18 c'è un piccolo
template con il numero scritto nell'interfaccia (`DB_echo/setup_N/depth_K.png`) e, per ognuno dei quattro
stati di flip, un rettangolo di ricerca in cui ESI lo cerca con template matching. La depth riconosciuta è
quella il cui template supera la soglia. Non misura niente: legge l'etichetta.

## Struttura

```
#17 = gruppo_depth_0 , gruppo_depth_1 , … , gruppo_depth_{N-1} ,        (N = numero di valori in #18)
gruppo = blocco_NF ; blocco_LR ; blocco_UD ; blocco_LRUD ;               (sempre 4, ordine fisso)
blocco = TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|BM|CH2:TH2:…|
```

| campo | significato | valori nei 390 legacy |
|---|---|---|
| TOP LEFT BOTTOM RIGHT | **rettangolo di ricerca** (non il template): dove ESI cerca il numero | mediana 35×24 px; dentro il rect #11 nel 42% dei casi, sopra 32%, a lato 14%, sotto 12% |
| B | check attivo | 1 in tutti i 21.160 blocchi |
| CH | canale | 7 = tutti i canali, in tutti |
| TH | soglia del match | dipende dal metodo; mediana 3,0·10⁶ (SQDIFF: più basso è meglio, accettato se min < TH) |
| P1, P2 | ginocchi della normalizzazione del contrasto (`FE_TH_METHOD`) | 20 e 120 in tutti |
| MM | metodo di match | 6 = `FE_TH_METHOD` 86%, 0 = `TM_SQDIFF` puro 13%, 5 = `TM_CCOEFF_NORMED` su Canny 1% |
| BM | esiste una seconda immagine (stato di transizione) | 1 nel 19% dei blocchi, 65 configurazioni, tutte Esaote |
| TH2 | soglia per la seconda immagine, stessa sintassi | presente sempre, usata solo se BM = 1 |

Il lettore legacy (`QFileFss::readFile`) **rifiuta il file** se la riga è vuota, se un gruppo non ha
esattamente 4 blocchi, se un blocco non ha 9 separatori, se il rettangolo esce dal video o se manca
`depth_K.png`. La #17 è quindi obbligatoria per ESI: un `.fss` con #17 vuota non si carica.

Vincoli verificati sull'archivio: gruppi = valori di #18 in 384/390 (i 6 restanti hanno un solo gruppo,
file di versione vecchia); 4 blocchi per gruppo in 5.290 gruppi su 5.296.

## Come nasceva nel vecchio ESIBuilder (tre pagine)

1. **Depth Value** (`WdgPageDepthValue`). L'operatore carica **una immagine per depth** (le `NF_18.png …
   NF_166.png` della cartella 247), scrive il valore in mm di ciascuna e disegna **un rettangolo sul numero
   della depth** (`rectTemplate`, magenta), di solito una volta e «applica a tutte». Alla generazione, il
   crop di quel rettangolo su ogni immagine diventa `depth_K.png`: è il template. Vedi
   `riga17/template_depth_setup347.png`: quattordici ritagli «18, 28, 37 … 166», 19×17 px per le due
   cifre, 27×17 per le tre.
2. **Depth Find** (`WdgPageDepthFind`). Per ognuno dei quattro orientamenti l'operatore carica le stesse
   depth (stesso numero di file, controllato) e disegna il **rettangolo di ricerca** (`rectFind`) intorno al
   numero; la pagina mostra il template centrato dentro. «Applica a tutte» lo copia su tutte le depth. Questo
   è il box che finisce nella #17. Se un orientamento non è disponibile, il suo blocco è la copia di quello di
   riferimento (stessa regola della #16).
3. **Thresholds** (`WdgManageTamplatesThresholds`). Per ogni template e ogni flip l'operatore indica immagini
   **positive** (quella depth) e **negative** (le altre depth). Il tool esegue il match nel rettangolo di
   ricerca su tutte, prende il **peggior positivo** e il **miglior negativo** e propone la soglia a una
   percentuale fra i due; se il peggior positivo è peggio del miglior negativo la soglia è −1 e va cambiato
   metodo o box. Per questo la TH è diversa per ogni depth anche a box uguale.

Il match (`CvMatchTemplate::getMinValTh`): ritaglia il rettangolo di ricerca, riporta i grigi da [P1, P2]
a [0, 255] (contrasto forzato, taglia il rumore scuro), `matchTemplate` con `TM_SQDIFF`, `minMaxLoc`.

## Che cosa dicono i numeri dell'archivio

- **Il box di ricerca è il template più un margine**: mediana 6 px in larghezza e 5 in altezza (minimo 0,
  i valori tipici 3, 5, 7, 9). L'operatore disegnava il rettangolo di ricerca poco più largo del numero.
- **Il flip LR non sposta il numero**: blocco LR = blocco NF in 381/384. Il **flip UD lo sposta** in 89/384
  (23%) e LRUD in 77/384: su quelle macchine il ribaltamento verticale porta l'etichetta altrove. È il
  motivo per cui i quattro blocchi esistono.
- **Il box cambia con le cifre**: nel setup 347 è 72|613|98|641 per le depth a due cifre e 72|613|98|651
  per 102–166. Nel 66% dei file il box è costante o ha ≤ 3 varianti; nel resto l'operatore lo ha ridisegnato
  depth per depth.
- **Stato di transizione** (BM = 1): sugli Esaote, mentre si gira la manopola, il numero della depth si
  accende in giallo. Il secondo template `depth_t_K.png` è la stessa etichetta in giallo
  (`riga17/template_depth_transizione_esaote_classC.png`), stesso rettangolo di ricerca, soglia TH2 propria.
  Serve a ESI per riconoscere la depth anche durante il cambio. Il setup 347 non lo usa (BM = 0).

## Come viene generata nella nuova app (fatto il 18/09, `tools/app/thresholds.py`)

Lo step **Soglie** dell'app la costruisce insieme alle soglie di #13, #14, #15 e #16, con questi
ingredienti, tutti già presenti dopo lo stadio depth corretto (vedi la prova sulla cartella 247):

| ingrediente | da dove |
|---|---|
| valori di depth e frame per depth | `depth_box_reads`: 62 frame letti nello stesso riquadro, raggruppati per valore |
| rettangolo dell'etichetta | il riquadro propagato (`depth_box_template.box`), oggi già identico al legacy sul 347 |
| template `depth_K.png` | il riquadro stretto dell'OCR sul frame **NF** di quella depth, allargato di 3 px per lato (legacy 19×17 attorno a cifre alte 11 px) |
| box di ricerca per flip | l'unione dei riquadri stretti dei frame di quella depth e di quel gruppo, allargata di 7 px per lato (legacy 28×26); così se il flip UD sposta il numero il box lo segue |
| TH per depth e flip | il calcolo legacy: `TM_SQDIFF` dopo normalizzazione [20,120] del template sui frame della sua depth (positivi) e sulle altre (negativi), soglia a metà fra peggior positivo e miglior negativo; stessa funzione per #13, #14, #15 e #16 |
| MM, P1, P2, CH, B | costanti: 6, 20, 120, 7, 1 |
| BM / TH2 | 0 salvo Esaote con frame di transizione disponibili: in tal caso il crop giallo diventa `depth_t_K.png` |

Esito sul setup 347: box di ricerca 73|614|99|643 contro il legacy 72|613|98|641, template 23×19
contro 19×17, soglie fra 0,7 e 4,6·10⁶ contro 0,8–5,2·10⁶, e nella simulazione di ESI 60 fotogrammi
su 60 riconosciuti dalla propria depth e solo da quella. Nessun intervento umano oltre al riquadro
dell'etichetta.

Criterio di verifica su larga scala: sui 384 legacy con template, rigenerare box e template dalla loro stessa acquisizione
e misurare (a) IoU fra il box generato e il legacy, (b) se il template generato **riconosce la depth giusta
e solo quella** sui frame dell'acquisizione con la soglia calcolata. Il punto (b) è quello che conta per
ESI, il punto (a) è quello che conta per `compare_fss.py`.

## Due cose da chiedere a Francesca

1. Il margine del box di ricerca: 3 px per lato basta, o ESI ha bisogno di più gioco per le acquisizioni
   dal vivo (piccoli spostamenti del video)?
2. Per gli Esaote senza frame di transizione in acquisizione: si può omettere BM (come nel 347, che è in
   produzione) o ESI si comporta peggio durante il cambio depth?
