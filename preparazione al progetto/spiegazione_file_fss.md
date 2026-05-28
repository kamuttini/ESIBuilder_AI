# File .fss

```jsx
//#01 Versione File FSS.  ✅
//#02 ID_ECHO  (int)ID univoco dell'ecografo ✅
//#03 ID_PROBE (int)ID univoco della probe ✅
//#04 PROBETYPE (int) ✅
//#05 KITNEEDLEGUIDE    (int) ID del file ".ndg" a cui è associato un kit needle guide ✅
//#06 VIDO_INPUT (int) 0=HDMI, 1=VGA ✅
//#07 VIDEO_INPUT_SIZE_X (int) dimensione del frame video in uscita dall’ecografo ✅ (in ingresso dall eco)
//#08 VIDEO_INPUT_SIZE_Y (int) dimensione del frame video in uscita dall’ecografo ✅
//#09 VIDEO_X_SIZE (int) larghezza in pixel dell’immagine campione.  ✅. (immagini ruotate resizate diamo a esi info sulle trasformazioni che dovrà fare) 
//#10 VIDEO_Y_SIZE (int) altezza in pixel dell’immagine campione.    ✅ 
//#11 RECT_ECHO (TOP|LEFT|BOTTOM|RIGHT|)(int) ✅
//#12 GROUP_ORIENTATION (int) ID del gruppo che si occuperà dell'orientation ✅
//#13 RECT_NAME_ECHO (TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|)	✅//TOP|LEFT|BOTTOM|RIGHT| rect dove cercare il template, B = 1 se check positivo (se match method con distanza euclidea sotto soglia TH, se match metodo di correlazione sopra soglia TH); CH:TH:P1:P2:P3:P4:P5 dove CH è il canale sul quale eseguire il match method (1 se Red, 2 se Green, 4 se Blue, 7 se su tutti); P1-P5 sono i parametri eventuali usati nei match method mentre TH è la soglia determinata da esi_builder sotto/sopra la quale c'è match/mismatch (in base al tipo di match method); MM Match Method: euclidean distance (CV_TM_SQDIFF(0)), correlation estimation after edge recognition (CV_TM_CCOEFF_NORMED(5)), euclidean distance after threshold method (FE_TH_METHOD(6), P1 e P2 rappresentano le soglie sotto P1 e sopra P2 si azzera la dinamica ed entro cui la stessa è massimizzata), euclidean distance after Histogram Equalization (FE_EQ_METHOD(7)).
//#14 RECT_NAME_PROBE   (TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|) ✅
//#15 PROIBITED_SCREEN ✅ TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|;......;TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|;,.....,TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|;......;TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|;,
//#16 RECT_ORIENTATION  ✅TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|;TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|;TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|;TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|;
//#17 RECT_DEPTH        ✅ TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|BM|TH2|;TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|BM|TH2|;TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|BM|TH2|;TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|BM|TH2|;,......,TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|BM|TH2|;TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|BM|TH2|;TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|BM|TH2|;TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|BM|TH2|;, TH2 è un insieme di parametri siffatto CH:TH:P1:P2:P3:P4:P5 (analogo a quelli per l'orientation, le schermate proibite, il logo ecografico e quello della probe utilizzata) ed usato secondo il MM scelto e il criterio di buon match BM se e' presente, per la depth considerata, lo stato di transizione.
//#18 VECT_DEPTH        (D1|D2|D3....|DN|)(float) valori in "mm" dei depth disponibili
//#19 PIXEL_RATIO_X     (P1|P2|P3....|PN|)(float) valori in "mm/pixel" del pixel ratio lungo X
//#20 PIXEL_RATIO_Y     (P1|P2|P3....|PN|)(float) valori in "mm/pixel" del pixel ratio lungo X
//#21 SCALE_LINE (X01|X02|Y01|Y02|L0|D0|S0|;X11|X12|Y11|Y12|L1|D1|S1|;.........;XN1|XN2|YN1|YN2|LN|DN|SN|;)
 "Dove posizionare la scala e di quanti pixel farla per ogni valore di depth: Xx1,Xx2,Yx1,Yx2 coordinate della linea, Lx lunghezza in mm della stessa, Dx delta tick in mm sulla scala, Sx lato della scala dove si vuole la label (-1 sx , +1 dx)"
//#22 CENTRE_DISTANCE   (C1|C2|C3....|CM|;C1|C2|C3....|CM|;.........C1|C2|C3....|CM|;)(float) distanze in millimetri dell'intercetta tra la prima linea e la linea verticale centrale e il top del rectEcho per ogni depth per ogni angolo, nello stesso ordine del vettore depth ed in cui sono archiviati nel file ".ndg" per ogni depth
//#23 ANGLE       (C1|C2|C3....|CM|)(float) angoli in gradi (-180 : +180) delle linee rispetto all'asse orizzontale X (non necessariamente equivale a quello nominale della guida), angolo usato per disegnare la linea
//////D’ora in poi vi sono le sezioni per le sonde biplane
//#24 RECT_TRANS        TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|;TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|;TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|;TOP|LEFT|BOTTOM|RIGHT|B|CH:TH:P1:P2:P3:P4:P5|MM|;Schermate di transizione: se non ci sono tutte e 4 le combinazioni quelle non esistenti saranno riempite con una di quelle esistenti
//#25 ID NEXT PROBE     ID del file ".fss" dell’altro file associato alla stessa sonda transrettale (si riferisce ad altro file .fss con "file.ndg" e tutti i parametri in commune fatto salvo delle distanze delle linee. "|" se non transrettale.
//#26 BIPLANA RECOGNITION MODE   Modalità di riconoscimento del tipo di sonda (lineare o trasversa)

    NOTE
    -TOP, LEFT, BOTTOM, RIGHT (int), rappresentano i punti di riferimento di un rettangolo all'interno del rettangolo video (0, 0, VIDEO_Y_SIZE-1, VIDEO_X_SIZE-1)
    -B, se = 0 è un false cioè il check è positivo
    -TH è un double ed è la soglia relativa al tipo di match metodo utilizzato
	-CH (int) identifica i canali su cui fare il match (1 se Red, 2 se Green, 4 se Blue, 7 se su tutti)
	-P1:P2:P3:P4:P5, ulteriori parametri che servono per il metodo di match. Attualmente l'unico metodo che li sfrutta è il FE_TH_METHOD(6) dove P1 e P2 rappresentano le soglie: sotto P1 e sopra P2 si azzera la dinamica ed entro le medesima la stessa viene massimizzata (vedasi le note sui etodi di match).
    -MM (int), indica il MATCH_METHOD utilizzato (CV_TM_SQDIFF = 0, FE_CORR_METHOD=5, FE_TH_METHOD=6, FE_EQ_METHOD=7)
    -BM se 1 = true quindi per quel check esiste un’altra immagine da controllare
    -TH2 (double), soglia relativa al tipo di match metodo utilizzato per l'immagine2. Se BM = 0 questo valore non viene utilizzato

```