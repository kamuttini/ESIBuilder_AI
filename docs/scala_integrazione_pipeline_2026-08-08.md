# Scala dentro la pipeline ufficiale — integrazione del 2026-08-08

Lo studio della scala (`tools/scale/study_scale_folder.py`) è diventato uno **stadio della
pipeline ufficiale**, con le stesse accortezze dello studio singolo ma senza ricalcolare
niente di quello che la pipeline già sa.

## Dove sta, e perché lì

Nuovo stadio 14 in `tools/ultrasound/predict_fss_head_from_acquisitions.py`, subito **dopo**
RECT_DEPTH e **prima** del blocco `review_reasons`.

Non è una scelta di comodità: la riga `#21 SCALE_LINE` è **per depth**. La `#17` definisce i
gruppi di depth e le righe `#18`–`#22` portano una voce ciascuna, nello stesso ordine. Prima
della depth non esiste l'unità di misura della riga; dopo, esiste.

```
... 13. RECT_DEPTH autonomo  ──► depth per immagine (valore + box + modo)
  │ 14. SCALA #18-#21
  │      intreccia: vendor, rect #11, rect per immagine, su/giù del marker, depth
  │      calcola:   colonna del righello, tacche, passo, numeri (OCR)
  │      raggruppa: frame -> depth -> depth_index -> consenso di setup
  ▼
  #18 VECT_DEPTH   #19/#20 PIXEL_RATIO_X/Y   #21 SCALE_LINE
```

## Cosa NON ricalcola (l'intreccio)

Lo studio singolo, per arrivare ai suoi sei stadi, si carica il classificatore vendor, il
regressore rect, la rete su/giù, il bundle del marker e lancia il modulo depth come
subprocess. Dentro la pipeline tutto questo esiste già, calcolato una volta sugli **stessi**
frame:

| serve allo studio | da dove arriva nella pipeline |
|---|---|
| verso (su/giù) | `su_giu_pred` delle righe LR-marker, che è **già** il verdetto del marker sopra il prior della rete (`bundle_vertical_correction` dice quando l'ha corretto) |
| rect per immagine | `echo_rect_*_abs` delle stesse righe; fallback: crop del classificatore su/giù; poi riga `#11` |
| depth d'interfaccia | stadio RECT_DEPTH, con il suo `mode`/`status`/box, così l'evidenza viaggia con il valore |
| vendor | classificatore vendor (media softmax sulla cartella) |

Conseguenza pratica: lo stadio costa solo il suo lavoro (detection + OCR), e scala e resto
della pipeline non possono più dare due risposte diverse sullo stesso frame.

## Accortezze conservate dallo studio singolo

- **Il verso decide quale capo porta lo zero**, le tacche dicono *dove* sta quel capo.
- **Il marker ha l'ultima parola** sul verso, la rete è solo il suo prior (a monte).
- **La colonna del righello è un fatto di cartella**: mediana delle detection, poi secondo
  passaggio con `prior_x` sui frame che da soli non l'avevano trovata.
- **Controprova dello zero con l'OCR**: il numero più piccolo deve stare dallo zero alla
  distanza che il suo stesso valore implica (e "ambiguo" quando sta a mezza scala).
- **Controprova della depth**: solo una depth *scritta nell'interfaccia* (`mode=direct_label`)
  è evidenza indipendente; una dedotta dalla scala si confronterebbe con se stessa.
- **Coerenza dei numeri attraverso il passo costante** (`check_numbers`), che è ciò che smaschera
  una cifra letta male senza avere una GT.
- **Consenso di setup** (`consolidate_scale_setup`): Theil-Sen su `log(mm_per_px)` + isotonica
  lungo il depth_index, mediana locale su `x`/`y_zero`, buchi riempiti e marcati `interpolated`.
- **Evidenza debole non ancora il trend** (`weak_anchor`): calibrazione geometrica a
  un'etichetta, righello fuori dalla banda del pannello, e — nuovo qui — frame della stessa
  depth che non si accordano fra loro.
- **Nessun parsing dei nomi file**: su materiale grezzo il nome non dice niente.

## Cosa aggiunge lo stadio

1. **Ponte frame → depth**: i valori depth vengono arrotondati al millimetro (arrivano
   dall'OCR), raggruppati, e ordinati **in senso crescente** → `depth_index`. L'ordine non è
   estetico: il consenso fitta `mm_per_px` con una regressione isotonica lungo quell'indice,
   quindi un ordine sbagliato combatterebbe l'unica invariante su cui è costruito.
   Verificato sui `.fss` legacy: `#18` `20|25|...|100` crescente, con `#19`/`#20` crescenti a fianco.
2. **Aggregazione per depth deliberatamente conservativa**: più frame della stessa depth sono
   lo stesso stato macchina, quindi dovrebbero concordare; quando non concordano, quel
   disaccordo serve solo a **declassare** la depth a evidenza debole, mai a inventare un valore
   nuovo facendo una media. Inventare una regola di accettazione senza misurarla è esattamente
   ciò che ha fatto regredire questo blocco tre volte.
3. **Rotazione rifiutata, non indovinata**: se la cartella è stata normalizzata a 90/180/270,
   il righello non è dove (né come) sembra, e lo stadio torna `scale_rotation_not_supported`.
4. **`tick` = 0.5 cm**, la costante legacy (128/128 gruppi dei dieci `.fss` di esempio la usano);
   il passo effettivamente misurato viaggia a fianco come `tick_step_mm_detected` invece di
   essere scritto nella riga, perché il detector non distingue le tacche da 0.5 cm da "una su due".
5. **Riga emessa solo se completa**: una `#21` con un buco in mezzo sposterebbe silenziosamente
   l'accoppiamento con `#17`/`#18` nel reader legacy.

## Anello di correzione

```
pipeline ──► <run>/scale/NNNN_cartella/{pipeline_context.json, scale_per_image.csv,
                                        scale_per_depth.csv, scale_lines.json, summary.json}
   │
   ├─► tools/scale/study_scale_folder.py --from-pipeline <quella cartella>
   │      la pagina a sei stadi mostra le predizioni DELLA PIPELINE (non le ricalcola)
   │      e le rende correggibili una per una
   │
   └─◄ correzioni esportate ──► --scale-corrections nella pipeline
          da lì viene usata **solo la colonna** del righello, che è un fatto di cartella;
          le correzioni per singolo frame restano nella pagina, dove si vedono
```

Applicare mezze correzioni in silenzio farebbe divergere pagina e stadio sulle stesse
correzioni: per questo la colonna è esplicita (`corrections_applied: ["colonna"]` nel summary)
e il resto no.

## Verifica fatta

Gli SSD non erano montati, quindi la verifica gira su **righelli sintetici la cui verità è
esatta**: `tools/scale/selftest_predict_scale_from_pipeline.py` (nessuna rete, nessun `.fss`,
solo cv2 + Tesseract).

| scenario | esito |
|---|---|
| normale (4 depth × 3 frame, zero in alto) | `mm_per_px` esatto su tutte e 4, zero e estremo al pixel |
| ribaltato (zero in basso, cifre diritte) | verso `-1`, `y1=670` / `y2=70` come i legacy con UD flip |
| buco (una depth senza righello) | riempita dal trend, `source=interpolated`, le altre intatte |
| innesto nella pipeline | il verso arriva dal **marker** e non dalla rete, il rect dalle righe del marker, la depth scartata non viaggia, solo `direct_label` conta come evidenza indipendente |

Nota sul secondo scenario: ribaltare l'immagine finita **specchia anche le cifre**, cosa che
nessun ecografo fa, e l'OCR legge numeri che non esistono. Il fixture disegna il righello con
lo zero in basso e le cifre diritte, che è come appare davvero un UD flip.

## Da misurare quando gli SSD tornano

Tre ritrovamenti emersi leggendo il codice, **non toccati** perché cambiare la detection senza
la chain eval violerebbe il criterio zero-regressioni.

1. **La riga `#11` è `TOP|LEFT|BOTTOM|RIGHT`**, non `x1|y1|x2|y2`. Verificato sui dieci `.fss`
   di `OldSoftwareEsiBuilder/templates`: il quarto token supera `VIDEO_Y_SIZE` su 8 di essi,
   quindi può solo essere una X. `scale_common.parse_rect_echo` la legge come `x1|y1|x2|y2`.
2. **Conseguenza sull'audit**: da quel mis-parsing viene la conclusione "su BK la scala sta
   *fuori* dal rettangolo, nel margine nero a destra". Con la lettura corretta, sui dieci setup
   locali il righello sta **dentro** il rect in 8 casi su 10 (da 29 a 268 px dentro il bordo
   destro). La nota d'intake del 2026-04-16 ("scala dentro il rect") potrebbe avere ragione, e
   le statistiche `side_vs_rect` per vendor sono sistematicamente sbagliate.
3. **Il detector non ha una convenzione sola per il rect**: `_bands_for` legge l'indice 2 come
   bordo destro (vuole `x1,y1,x2,y2`), `_out_of_band` legge `rect[0]+rect[2]` come bordo destro
   (vuole `x,y,w,h`), lo studio passa `(left, top, width, height)` e l'eval harness passa una
   terza cosa ancora (le colonne GT `rect_x1..rect_y2`, che per il punto 1 contengono
   `top,left,bottom,right`). Nell'eval la banda destra è quindi costruita attorno a **bottom**,
   una Y usata come X.

   Misura sui dieci setup locali, banda del profilo default `(-60, +420)`: il righello vero cade
   dentro la banda **4/10** con la convenzione dell'eval e **7/10** con quella corretta — e i 3
   che mancano con quella corretta sono i righelli *dentro* il rect, che la tolleranza di 60 px
   verso l'interno non raggiunge. Nessuna delle due vince: il numero da rimettere in discussione
   è la tolleranza verso l'interno dei profili, non solo la convenzione.

   Lo stadio riproduce **esattamente** quello che passa lo studio singolo (`rect_for_detector`,
   un solo punto con il nome che dice cosa fa), perché è la forma con cui sono stati prodotti
   tutti i numeri già guardati. Sistemare la convenzione è un cambio di predizioni: va misurato
   con la chain eval, non deciso qui.

Restano fuori dallo stadio, come prima: `#17 RECT_DEPTH` come riga (esiste il valore, non la
riga) e `#22 CENTRE_DISTANCE`. E la pipeline continua a non scrivere nessun file `.fss`.
