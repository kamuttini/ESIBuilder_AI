# Il file `.ndg` nel vecchio ESIBuilder

Studio del formato e dell'uso, letto dal codice legacy (`qfilendg.cpp/h`, `qlineguide.cpp`,
`wdgpagecalibration.cpp`, `qgridobj.cpp`, `mainwindow.cpp`) e verificato sui 390 file
dell'archivio in `artifacts/92_guides/legacy_cache`.

## Che cosa contiene

Il `.ndg` descrive **il kit fisico di guida aghi**, non la configurazione dell'ecografo: per
ogni angolo del kit, la distanza dei fori successivi dal primo, in millimetri.

    kit_needle_guide_<ID>.ndg
      [ -2, 2 ]                    solo per le guide a griglia
      numero di angoli             N
      per ogni angolo:
        numero di distanze         K        -> le linee di quell'angolo sono K+1
        K valori                   in mm, dal primo foro

La prima linea non ha un valore: e' l'origine, ed e' quella che la calibrazione misura (riga
`#22`). Le altre K si ricavano da qui.

Il commento in testa a `qfilendg.h` dice che il file e' testuale. **Non lo e'**: e' un
`QDataStream` binario big endian. I contatori sono interi a 32 bit; i valori sono `float` nel
C++ ma nel file occupano 8 byte, perche' QDataStream scrive in doppia precisione se non gli si
dice altro. Chi si fida del commento non apre il file.

`tools/needle/guides_geometry.py` lo legge (`read_ndg`) e lo scrive (`write_ndg`). Riscritti
tutti i 390 file dell'archivio a partire da quello che se ne legge: **390 identici byte per
byte**, nessuno rifiutato.

## Le regole che il legacy impone

`QFileNdg::writeFile` e `QFileNdg::loadFile` applicano gli stessi due controlli, in scrittura
e in lettura:

- la prima distanza deve essere **maggiore di zero**
- la serie deve essere **strettamente crescente**

Un file che non li rispetta viene letto e poi scartato (`m_vvflDistances.clear()`), e il
progetto si blocca con «Il file NDG indicato non esiste». Scrivere un file non valido equivale
a non scriverne nessuno, quindi `write_ndg` rifiuta prima di scrivere.

## Come entra nella geometria

In `QLineGuide::upDateLine` la distanza dal centro della linea `m` di un angolo e':

    centre_distance = #22[depth][angolo]
    se m > 0:  centre_distance += ndg[angolo][m-1] / cos(angolo)      (angolo != 90°)
               centre_distance += ndg[angolo][m-1]                    (angolo == 90°)

Il `/cos` dice che cosa sono davvero quei millimetri: la distanza **perpendicolare fra le
linee**, cioe' la spaziatura fisica dei fori. Dividerla per il coseno la converte nello
scostamento lungo la verticale, che e' la grandezza in cui `#22` e' scritto.

Quindi il `.ndg` e' il moltiplicatore della calibrazione: l'operatore ricalca **un ago solo**
per ogni coppia (angolo, depth) e da quella misura escono tutte le linee di quell'angolo.
`WdgPageCalibration::updateechoimage` disegna tutte le `getNumOfLines(angolo)` linee, ma nel
`.fss` finisce un valore solo per angolo.

## Vincoli verificati sull'archivio

Su 384 configurazioni con `.fss` e `.ndg` leggibili:

| regola | esito |
|---|---|
| numero di angoli nel `.ndg` == angoli della riga `#23` | 384 su 384 |
| `#05` del `.fss` == numero nel nome del file `.ndg` | 384 su 384 |

Due invarianti da rispettare quando si genera una configurazione nuova.

## Lo stesso kit non ha sempre lo stesso file

65 kit distinti nell'archivio; **58 hanno contenuto identico ovunque, 7 no**. Le differenze
sono di tre nature diverse e vanno distinte:

- **varianti a griglia**: kit 13, 28, 50 hanno una minoranza di file con un solo angolo e due
  fori a 40, 60 o 70 mm. Sono le configurazioni «Template GRID»: li' il `.ndg` e' un
  segnaposto e il contenuto vero sta nel `.grid`.
- **kit fisicamente diversi con lo stesso ID**: kit 31 (GE) compare a 4 angoli e a 6 angoli;
  kit 74 (Mindray) a 8 e a 10 fori; kit 40 (Hitachi) a 9 fori da 5.1 e a 7 da 5.5.
- **errori di battitura**: kit 62 (Koelis) ha due file identici tranne l'ultimo foro, 48 mm
  contro 49 mm.

Conseguenza pratica: **l'ID del kit e' un ottimo punto di partenza e non una garanzia**.
Riusare il `.ndg` di un'altra configurazione con lo stesso `#05` e' quasi sempre giusto, ma va
mostrato e confermato, non applicato in silenzio.

## Le guide a griglia (biplane)

Se il file comincia con gli interi `(-2, 2)`, il kit e' una griglia e il legacy carica anche
`kit_needle_guide_<ID>.grid`, un INI con la matrice «battaglia navale»:

    [Matrix]   dim_N, dim_M, Orientation, row_0..row_{N-1}   (1 = foro presente)
    [LabelX]   label   es. "A, ,B, ,C, ,D, ,E, ,F, ,G,"
    [SpaceX]   space   es. "5.00,5.00,..."
    [LabelY]   label   es. "1, ,2, ,3, ,4, ,5, ,6, ,7,"
    [SpaceY]   space

Nell'archivio: 53 file di questo tipo, quasi tutti **13x13 a 5 mm** (41 su 53), piu' qualche
9x13, 15x13, 29x25 a 2.5 mm e un 7x7. Sette matrici hanno fori assenti (380 zeri: sono i
Biopsee, dove la griglia e' parziale).

**L'autorita' e' l'intestazione del `.ndg`, non la presenza del `.grid`.** Quattro
configurazioni hanno un `.grid` accanto ma un `.ndg` senza intestazione a griglia: sono guide
a linea con un file avanzato da un'altra configurazione, e il legacy lo ignora.

La pagina di calibrazione a griglia si accende solo se `isGridNdg()` **e** il tipo sonda non e'
`PROBE_TYPE_TRANS_L` (3): la griglia si calibra sul piano **T**.

## Attenzione: `.grid` sono due cose diverse

| file | intestazione | che cos'e' |
|---|---|---|
| `kit_needle_guide_<ID>.grid` | `[Matrix]` | **ingresso**: la griglia del kit fisico |
| `setup_<ID>.grid` | `[CircleRect]` | **uscita**: il risultato della calibrazione a griglia |

Il secondo e' scritto da `QFileFssGrid` e contiene, per depth e per flip, il rettangolo del
cerchio sonda e quello della griglia (in pixel e in mm), i pixel ratio, e `bVisibleMatrix` =
prima e ultima riga, prima e ultima colonna visibili nell'immagine. I rettangoli in mm sono
`QVariant` binari dentro l'INI (`@Variant(...)`, QRectF big endian), quelli in pixel sono
`@Rect(x y w h)`.

Per le biplane a griglia, quindi, **la calibrazione non finisce nel `.fss`**: finisce in un
file a parte. Chi generasse solo il `.fss` avrebbe una configurazione a meta'.

## Che cosa serve a noi

1. **Leggere e scrivere il `.ndg`**: fatto e verificato byte per byte.
2. **Sapere quale kit**: viene dall'anagrafica (`#05`), non dalle immagini. Non e' una cosa che
   si misura: e' una cosa che si sceglie, e il file del kit o esiste gia' o va creato a mano
   dalle misure della guida fisica, come faceva `QDlgCreateNdg` («Inserisci spaziatura tra
   fori»).
3. **Proporre il `.ndg` di un kit gia' visto**: l'archivio ha 65 kit; per una configurazione
   nuova con un kit noto il file si copia. Con le tre eccezioni qui sopra da mostrare.
4. **Per le biplane a griglia**: serve anche `setup_<ID>.grid`, e prima ancora il rilevamento
   della matrice di pallini, che oggi non e' coperto — il rilevatore in
   `tools/needle/detect_biopsy_dots.py` trova le guide **a linea** colorate, non le matrici.
