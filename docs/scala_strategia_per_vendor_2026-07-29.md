# Blocco scala — ripartenza e strategia per vendor (2026-07-29)

Documento di riferimento per il blocco scala (riga 21 `.fss`, `SCALE_LINE`), che
sostituisce la strategia del 2026-04-16 (`artifacts/34_scale_gt_fix_focus_20260416/scale_detection_strategy_from_intake_2026-04-16.md`),
in più punti smentita dai dati.

Artifact prodotti in questa sessione:

- `artifacts/37_scale_gt_audit_20260729/` — audit completo della GT su `SSD_esi1_n1`
- `artifacts/38_scale_ladder_eval_20260729/` — prima eval del detector nuovo + gallery
- `tools/scale/` — modulo nuovo: `scale_common.py`, `audit_scale_gt.py`,
  `detect_scale_ladder.py`, `eval_scale_detector.py`

---

## 1. Cosa è davvero la riga 21

Formato: `x1|x2|y1|y2|length_mm|tick|side|` ripetuto con `;` per ogni indice di depth.

Nel vecchio Qt questa linea è etichettata **"Linea calcolo pixel ratio, punto zero e
posizione scala"** (`wdgpagedepthvalue.cpp`). L'operatore trascinava un segmento
verticale dalla tacca `0` a un'altra tacca etichettata e dichiarava la lunghezza fisica.
Quindi la riga porta tre cose, e **solo due sono oggettive**:

| Contenuto | Oggettivo? | Note |
|---|---|---|
| `mm_per_px = length_mm / length_px` | **sì** | coincide con `PIXEL_RATIO_Y` (riga 20) entro lo 0.6% su tutte e 5290 le righe (p99 0.26%, mediana 8e-7); `PIXEL_RATIO_X == PIXEL_RATIO_Y` sempre |
| `x`, `y1` (punto zero) | **quasi** | è il *click dell'operatore*, non la tacca stampata: su `Hitachi Arietta 65 - C41L47RP - T` d14 lo zero stampato è a y≈716 e `y1` vale 733, **17 px di scarto**, oltre il doppio della tolleranza di produzione. `y1` porta un rumore di etichettatura di ordine 10 px |
| `y2` (estremo lontano) | **no** | è dove l'operatore ha smesso di trascinare |

`length_mm` è in **mm**, `tick` è in **cm** (incoerenza legacy: `qscaledline.cpp`
stampa le etichette come `i * tick`, che corrisponde all'overlay "0 cm / 1 / 2.0"
disegnato dall'ecografo).

### Perché i regressori precedenti non potevano funzionare

1. **`y2` è una convenzione, non un fatto.** Le run `32_`/`34_`/`36_` regredivano
   `y_top` e `y_bottom` come target indipendenti: metà del target era rumore di
   etichettatura umana. Il MAE di ~33 px su `y` era in buona parte irriducibile.
2. **`y1` è lo zero, non il punto più in alto.** 1146 righe su 5290 (**21.66%**) hanno
   `y1 > y2`: sulle acquisizioni ribaltate UD l'ecografo stampa `0.0 cm` **in basso**.
   `train_scale_line_image_model.py:179-180` fa letteralmente `y_top = min(y1,y2)` /
   `y_bottom = max(y1,y2)`, quindi metteva lo zero all'estremo sbagliato su un quinto
   del dataset. Per vendor (su righe totali): BK 544/1286, Hitachi 364/933,
   Esaote 92/1733, Canon 68/228, GE 0/557.
3. **La grandezza che conta non era mai misurata.** L'unico numero che finisce
   nel `.fss` in modo verificabile è `mm_per_px`, e nessuna run lo riportava.

   Attenzione a non spingere troppo questo confronto: i numeri storici *non* sono
   commensurabili con quelli nuovi. `34_scale_bbox_model_bk_20260421` misura MAE sugli
   spigoli di una **bbox** (test da (24.1, 16.9) a (100.4, 64.0) — due run migliori del
   valore spesso citato); `36_.../bbox_ticks_ocr_depth_refine_*` misura
   `0.5*(|Δy_top|+|Δy_bottom|)` sulla coppia **collassata** (test mae_x 33.4 / mae_y
   32.9 su 206 righe di split fisso); l'eval nuova misura un **solo** estremo (lo zero)
   più la calibrazione, su un campione senza split. Il difetto strutturale del vecchio
   target è reale e dimostrato dal codice; la superiorità numerica del detector sulla
   *localizzazione* non lo è ancora (vedi §5).

## 2. Cosa dice l'audit della GT

`python3 tools/scale/audit_scale_gt.py --root /Volumes/SSD_esi1_n1 --output-dir artifacts/37_scale_gt_audit_20260729`

5290 righe su 384 cartelle, **5210 clean (98.5%)**, 80 in quarantena
(57 `non_integer_intervals`, 23 `length_above_vect_depth`).

**La GT è buona.** Il backlog "P0=34 / P1=49" della roadmap descriveva un problema che
non è il collo di bottiglia: l'identità di calibrazione `length_px * PIXEL_RATIO_Y ==
length_mm` regge entro lo **0.6%** su **tutte e 5290 le righe, quarantena inclusa**
(p99 0.26%). Nessuna riga è in quarantena per calibrazione. Il ritardo del blocco era
nel *modello*, non nelle etichette.

I due soli criteri che scattano sono **aritmetici sull'etichetta**, non sulla difficoltà
visiva: `non_integer_intervals` (57, `length_mm` non multiplo di 5 mm) e
`length_above_vect_depth` (23). Il filtro però **non è vendor-neutro**: GE 42/557
(7.5%), Vinno 4/22 (18%), **ExactVu 4/4 (100%: il vendor sparisce dalla GT di eval, ed è
proprio una cartella interamente UD-ribaltata)**. Da tenere presente prima di leggere le
metriche per vendor come rappresentative.

### Assunzioni del 2026-04-16 da correggere

| Assunzione vecchia | Realtà misurata |
|---|---|
| "scala dentro il rect" | quasi sempre **fuori**, nel margine nero a destra; offset mediano dal bordo 165 px (BK) a 680 px (Hitachi) |
| "priorità a destra, sinistra rara" | confermato: destra dominante in 13 vendor su 15; Alpinion è a **sinistra** |
| "zero in alto" | **falso nel 21.7% dei casi** (ribaltamento UD) |
| "passo tacche variabile" | il campo `tick` vale **0.5 cm su tutte e 5290 le righe**, così come `side = -1` e `x1 == x2`. Nota: il `tick_pitch_px` dei profili è **derivato** (`5 mm / mm_per_px`), non misurato sull'immagine — sulla riga Hitachi citata sopra vale 15.2 px mentre le tacche stampate distano 30.5 px (passo visibile 1 cm) |
| "unità spesso assente" | l'unità `cm` c'è spesso, ma **non sempre accanto allo zero**: su BK Profocus l'unica etichetta è "2.0 cm" all'estremo *lontano*, e su BK FlexFocus Template grid non c'è alcuna etichetta numerica |

### Note operative emerse

- `side` in riga 21 vale **-1 su tutte le righe**: indica che le etichette numeriche
  stanno a *sinistra* delle tacche, non da che lato dell'immagine sta la scala. Non è
  un segnale di posizione.
- Il `RECT_ECHO` (riga 11) **non è un ancoraggio affidabile**: su layout multi-pannello
  e su Siemens è disegnato in modo da inglobare la scala o è semplicemente sbagliato.
  Per questo i controlli rect-dipendenti dell'audit sono flag, non motivi di quarantena,
  e il detector cerca in una banda con tolleranza generosa da entrambi i lati del bordo.
- Le immagini in `image_samples` sono screenshot **full-screen multi-pannello**: su
  fusion/biplana ci sono due righelli e il detector può agganciare quello sbagliato.
- Il vecchio loader (`tools/fss/train_scale_line_image_model.py`) come ultima risorsa
  accoppiava una riga di GT con *qualunque* immagine della cartella. `find_depth_image`
  in `scale_common.py` non lo fa: preferisce restituire `None`, perché un accoppiamento
  immagine↔depth sbagliato avvelena la GT in silenzio.

## 3. Strategia: detector deterministico, non regressione

La scala è un **righello**: colonna di tacche corte equispaziate, con numeri ogni N
tacche e spesso l'unità accanto allo zero. È un oggetto fortemente strutturato, quindi
lo si rileva, non lo si regredisce.

```
stage 1  tacche candidate   componenti connesse a forma di trattino corto e chiaro
stage 2  raggruppamento     trattini che condividono una x, con y su y0 + k*pitch
stage 3  OCR etichette      numeri accanto al righello, una chiamata per striscia
stage 4  calibrazione       fit lineare  value_mm = slope * (y - y_zero)
stage 5  confidenza         consenso del fit -> accepted / review / reject
```

**Lo stage 4 è il punto di tutto il progetto.** Due etichette lette correttamente a due
`y` note danno `mm_per_px` e `y_zero` direttamente — cioè esattamente ciò che le righe
19/20/21 del `.fss` codificano. Non serve la depth, non serve un prior appreso.

Tre conseguenze che semplificano molto:

1. **Il verso si ricava dalle etichette.** Se i numeri crescono verso l'alto, lo zero è
   in basso. Il blocco orientation non è un prerequisito: la scala si autodetermina.
2. **Il pitch delle tacche corrobora le etichette.** Con solo due numeri il fit è esatto
   e quindi non verificabile; però `mm_per_px * pitch` deve cadere su un passo che un
   ecografo stampa davvero (1, 2, 2.5, 5, 10, 20 mm). È un controllo indipendente e
   permette di accettare i casi a due etichette.
3. **`y2` si genera per convenzione**, non si predice: `y2 = y_zero + direction *
   length_mm / mm_per_px`, con `length_mm` la tacca in centimetri interi più profonda
   osservata. Qualunque scelta coerente produce un `.fss` semanticamente valido.

### Specializzazione per vendor

`ScaleProfile` in `detect_scale_ladder.py` raccoglie i prior per vendor (banda di
ricerca, lato preferito, passo minimo, forma del trattino, soglie di accettazione). I
valori attuali derivano da `scale_gt_profiles.json`. La struttura dell'algoritmo resta
una sola: si specializzano i parametri, non il codice, così un vendor nuovo si aggiunge
con una riga di profilo invece che con un modello nuovo.

Priorità di consolidamento, per volume di dati:
Esaote (1733 righe / 134 cartelle) → BK (1286 / 95) → Hitachi (933 / 63) →
GE (557 / 34) → Mindray, Canon → coda lunga (Philips, Siemens, Terason, Alpinion,
Koelis, Vinno, ExactVu, Sonostar: 22 righe o meno ciascuno, da trattare a mano o con
il profilo default).

## 4. Metriche e criterio di uscita

`tools/scale/eval_scale_detector.py` misura, per vendor:

| Metrica | Tolleranza | Perché |
|---|---|---|
| `rel_err_mm_per_px` | ≤ 2% | è la calibrazione che finisce nelle righe 19/20 |
| `err_x_px` | ≤ 6 px | colonna del righello giusta |
| `err_y_zero_px` | ≤ 8 px | punto zero giusto |
| `direction_ok` | esatto | zero all'estremo giusto |
| `strict_ok` | tutte e quattro | criterio di produzione |

Riporta anche `strict_ok` e `calib_ok` **ristretti agli `accepted`**: è il numero che
conta per la policy "l'AI propone, l'utente conferma". Un blocco con pochi `accepted`
ma tutti giusti è utilizzabile; un blocco con molti `accepted` sbagliati non lo è.

**Criterio di uscita fase proposto** (da confermare): per ogni vendor con ≥ 200 righe,
`strict_ok` sugli `accepted` ≥ 95% e copertura `accepted` ≥ 70%. Il resto va in review
assistita, coerentemente con la policy degli altri blocchi.

## 5. Stato del prototipo e backlog

`artifacts/38_scale_ladder_eval_20260729/bk/` — BK, 90 righe, ≤ 2 per cartella:

| | valore |
|---|---|
| accepted / review / reject | 27 / 35 / 28 |
| verso corretto sugli accepted | **27 / 27** |
| calibrazione corretta (≤ 2%) sugli accepted | 18 / 27 (67%) |
| `strict_ok` sugli accepted | 9 / 27 (33%) |
| errore relativo su `mm_per_px` (accepted) | mediana **0.34%**, **media 40.9%** |

**La media va letta prima della mediana.** L'errore relativo sui 27 accepted, ordinato
(%), è: `0×7, 0.01, 0.04, 0.09, 0.18, 0.20, 0.31, 0.34, 0.37, 0.37, 0.40, 0.47, 29.7,
29.9, 30.6, 31.2, 31.2, 31.2, 99.1, 299.3, 519.3`. Diciotto righe sono corrette allo
0.5% e nove sbagliano dal 30% al 519%. Quindi **la policy di confidenza non funziona
ancora**: accetta casi grossolanamente sbagliati, ed è il difetto più grave da chiudere
prima di qualunque altra cosa. Un blocco così non è utilizzabile nemmeno con conferma
umana, perché un errore del 30% sulla calibrazione non si vede a occhio in una gallery.

`artifacts/38_scale_ladder_eval_20260729/multivendor/` — 90 righe, 1 per cartella,
profili `[BK, Canon, Esaote, GE, Hitachi, Mindray, default]`. **Le mediane di `err x` e
`err y0` sono calcolate solo sulle righe con una predizione**, cioè escludono i reject:
la colonna `n con pred` dice su quante righe:

| Vendor | Righe | n con pred | err x (med) | err y0 (med) | calib OK | reject principale |
|---|---|---|---|---|---|---|
| Esaote | 33 | 11 | 5.0 px | 1.1 px | 12% | `no_ladder` 22/33 |
| BK | 26 | 17 | 8.5 px | 203 px | 12% | |
| Hitachi | 13 | 8 | 69 px | 618 px | 8% | |
| GE | 9 | 5 | 135 px | 143 px | 0% | |
| Mindray | 4 | 2 | 5.3 px | 36 px | 25% | |

Aggregato multivendor: 17 accepted su 90 (19%), verso corretto 14/17, calibrazione 9/17,
`strict_ok` 5/17.

Tre letture oneste:

1. **Il verso si risolve, ed è il risultato solido della sessione.** 27 su 27 sugli
   accepted BK (14/17 sul multivendor): il problema del 21.7% di scale invertite, che
   avvelenava il vecchio target, si chiude leggendo le etichette.
2. **La fase 1-2 non generalizza ancora.** `no_ladder` è il motivo dominante: 22 righe su
   33 su Esaote, 42 su 90 sul multivendor. L'affermazione "il righello si trova sempre,
   manca solo l'OCR" è smentita dai dati. Dove il righello *viene* trovato la
   localizzazione è ottima (Esaote: 1 px mediano sullo zero su 11 righe), ma è un
   sottoinsieme selezionato.
3. **Sulla localizzazione il detector non batte ancora il baseline.** BK
   `err_y_zero_px` su tutte le 62 righe con predizione: mediana 156.5 px, media 237 px,
   contro 32.9 px di MAE della run `36_` su 206 righe (con le cautele di §1 sulla non
   commensurabilità). Solo sui 27 accepted scende a 28 px di mediana. Il guadagno
   rivendicabile oggi riguarda **la calibrazione** — che le run vecchie non misuravano —
   e **il verso**, non la precisione geometrica.

Nota sulla tolleranza: `err_y_zero ≤ 8 px` è **più stretta del rumore della GT stessa**
(17 px misurati sul caso Hitachi). Va rivista, altrimenti `strict_ok` misura in parte
l'imprecisione del click dell'operatore legacy.

### Due trappole già trovate e risolte, da non reintrodurre

- **Il fit del passo preferiva un sottomultiplo.** Una griglia 3 volte più fine
  interpola le stesse tacche con residuo *minore*, quindi vinceva: su 33 righe
  correttamente localizzate il rapporto `pitch_GT / pitch_predetto` era un 2 o un 3
  esatto in 29 casi. Risolto pesando la **copertura** delle celle della griglia, non
  solo il residuo (`_fit_uniform_grid`).
- **La striscia di OCR non deve dipendere dal passo.** Legando l'altezza della striscia
  al pitch, la soglia Otsu — e quindi quali cifre tesseract vede — cambiava al variare
  di un parametro non correlato: correggere il pitch faceva *perdere* etichette. Ora il
  margine è fisso.

## 6. Consenso a livello di setup — il bloccante è chiuso

`tools/scale/consolidate_scale_setup.py`, aggiunto dopo la verifica. Il detector guarda
una immagine alla volta, e su una singola immagine una calibrazione sbagliata è
invisibile: niente nel frame la contraddice. Una **cartella** invece la contraddice
eccome.

### I vincoli, misurati sulla GT (setup con ≥ 4 depth e verso omogeneo, n = 371)

| Grandezza | Comportamento dentro un setup |
|---|---|
| verso (zero in alto/basso) | omogeneo in **371 / 376** setup → voto di maggioranza |
| `mm_per_px` fra depth **adiacenti** | rapporto in [0.85, 1.35] nel **99%** dei casi, e non decresce quasi mai (p01 = 0.998) |
| `mm_per_px` vs indice di depth | residuo da Theil-Sen su `log`: mediana **3.9%**, p90 12.1%, p99 30.7% |
| `y_zero` | residuo da Theil-Sen: mediana **1.0 px**, p90 20 px, p99 91 px |
| `x` | residuo da Theil-Sen: mediana **1.2 px**, p90 18 px, p99 59 px |

Quello che **non** funziona, ed è stato provato: confrontare con la mediana del setup.
`mm_per_px` varia legittimamente di un fattore **4.6** (mediano) dentro un setup, perché
la depth va da 20 a 80 mm. L'invariante è il *trend*, non il livello. Come dicevi tu: la
scala non è uguale su ogni immagine della cartella, si accorcia e lo zero si sposta — solo
che lo fa in modo regolare, 1-2 px per volta.

Allo stesso modo non funziona la mediana locale su `y_zero`: un valore sbagliato sta
*dentro* la finestra dei suoi vicini e ne trascina la mediana, quindi un guasto condanna
le righe accanto invece di sé stesso. Theil-Sen guarda tutte le coppie e lo mette in
minoranza.

### Effetto misurato

BK, 6 setup / 95 righe (`artifacts/38_scale_ladder_eval_20260729/bk_consensus/`):

| | per immagine | con consenso |
|---|---|---|
| accepted | 27 / 90 (30%) | **45 / 95 (47%)** |
| verso corretto sugli accepted | 27/27 | **45/45** |
| calibrazione corretta sugli accepted | 18/27 (67%) | **38/45 (84%)** |
| `strict_ok` sugli accepted | 9/27 (33%) | **28/45 (62%)** |
| errore su `mm_per_px`: mediana / media / **max** | 0.34% / 40.9% / **519%** | 0.17% / 1.79% / **19.8%** |
| `err_y_zero` mediano sugli accepted | 28 px | **0.5 px** |

Esaote, 8 setup / 91 righe: accepted 28 (31%), verso 28/28, calibrazione 22/28 (79%),
errore su `mm_per_px` mediana 0.22% / media 2.0% / max 14.8%.

**La coda catastrofica è sparita**, ed era il difetto che rendeva il blocco inutilizzabile
anche con conferma umana. Il consenso non solo scarta gli incoerenti: li **riporta sulla
curva**, quindi in molti casi la riga diventa corretta invece che scartata.

Cautele oneste: il campione è piccolo (6 e 8 setup) e la copertura resta sotto il target
(47% e 31% di accepted), perché i reject sono dominati da setup interi in cui il righello
non viene trovato affatto — problema di detection, non di consenso.

### Backlog in ordine di valore atteso

0. ~~Riparare la policy di confidenza~~ — **fatto** con il consenso a setup (§6).
   Residuo: 4 righe su 45 accepted BK restano oltre il 10% di errore.
1. **Copertura del righello (`no_ladder`).** Ora è il collo di bottiglia unico: nel run
   con consenso 35 righe su 95 (BK) e 50 su 91 (Esaote) hanno `source = none`, cioè
   setup interi senza nessun agganciamento da cui interpolare. Era il motivo dominante
   già per immagine (42/90 sul multivendor, 22/33 su Esaote), non l'OCR.
   Serve una risposta morfologica (top-hat
   orizzontale) invece della sola soglia, e `min_ticks` per vendor: su BK FlexFocus
   Template grid le tacche visibili sono **3**, sotto la soglia attuale di 5.
2. **Ancora sull'unità `cm`.** L'OCR usa una whitelist di sole cifre e scarta il token
   `cm`. Aiuta a fissare lo zero e a risolvere gli "off-by-N-tacche", **ma non salva
   BK**: su BK Profocus l'unica etichetta è "2.0 cm" all'estremo lontano e su BK
   FlexFocus Template grid non ce n'è nessuna. Per quelle famiglie serve una via senza
   etichette: passo delle tacche noto per vendor + zero all'estremo del righello.
3. **Layout multi-pannello.** Su fusion/biplana ci sono due righelli; va scelto quello
   del pannello che contiene il rect, con un vincolo di appartenenza al pannello.
4. **Selezione del pitch.** Trattini spuri vicini fanno agganciare un sottomultiplo del
   passo vero; il termine di copertura in `_fit_uniform_grid` lo corregge, ma è meglio
   filtrare i trattini duplicati in stage 1.
5. **Offset di `x` per vendor.** Misurato sugli accepted del run con consenso:
   su **BK non c'è bias** (mediana di `pred_x - gt_x` = 0.0), ma la distribuzione è
   trimodale — un gruppo a ±3 px e due cluster a **±30 px**, cioè colonna sbagliata
   occasionale, non offset da correggere. Su **Esaote c'è un bias reale di +16 px**
   (p25 +6, p75 +25): l'operatore cliccava a sinistra dei trattini. Vale un campo
   `x_offset_px` nel profilo, ma con n=28 e quella dispersione conviene prima raccogliere
   più dati che applicarlo alla cieca.
6. **Rivedere le tolleranze** alla luce del rumore della GT (17 px misurati fra `y1` e la
   tacca zero stampata): `err_y_zero ≤ 8 px` è più stretta del rumore dell'etichetta.
7. **Integrazione**: writer `.fss` della riga 21 dal `ScalePrediction` (`fss_segment()`
   è già pronto) e collegamento nella pipeline `fss_head`.
8. **Test su `SSD_esi1_n3`** (`ACQUISITION ELABORATION`, 272+ cartelle) come set di
   verifica indipendente, senza GT, con gallery di review.

### Cosa NON rifare

- Non riprendere i regressori `32_`/`34_`/`36_` sul target `(y_top, y_bottom)`: il
  target è in parte rumore. Se in futuro serve una rete, il target giusto è
  `(x, y_zero, log mm_per_px)` con il verso come classificazione a due classi.
- Non spendere giorni sul backlog GT P0/P1 prima di misurare: la GT è al 98.5% coerente.
