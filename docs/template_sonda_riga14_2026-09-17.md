# Template sonda (riga #14 `RECT_NAME_PROBE`) — analisi, dataset, training

Data: 2026-09-17. Volume analizzato: `/Volumes/SSD_esi1_n1` (401 cartelle, 390 configurazioni valide).
Domanda di partenza: per riconoscere il template del nome sonda conviene **una rete generica o una rete per vendor?**

## Risposta breve

**Rete unica generica.** Il vendor, quando la rete può guardare l'immagine, aggiunge quasi nulla
(+0,03 di IoU in un test non parametrico), mentre spezzare il dataset per vendor lascia 12 marchi su 16
con meno di 15 configurazioni — e sulla riga #13, che è lo stesso tipo di problema, le reti per-vendor
addestrate in passato sono rimaste a IoU 0,00–0,17 proprio su quei marchi. Il vendor resta utile come
*input* di condizionamento o come routing di un eventuale fine-tuning, non come criterio per separare i dati.

---

## 1. Cosa c'è sul volume

| vendor | config | layout UI distinti | immagini |
|---|---:|---:|---:|
| Esaote | 140 | 84 | 169.116 |
| BK | 95 | 53 | 72.825 |
| Hitachi | 63 | 35 | 53.020 |
| GE | 37 | 27 | 40.048 |
| Mindray | 15 | 7 | 10.054 |
| Canon | 8 | 5 | 10.674 |
| Koelis | 7 | 5 | 310 |
| Philips | 6 | 6 | 2.361 |
| Biopsee / Terason | 4 + 4 | 4 + 1 | 3.184 |
| Siemens / Toshiba | 3 + 3 | 3 + 2 | 15.317 |
| Alpinion / ExactVu / Sonostar / Vinno | 2+1+1+1 | 5 | 2.708 |
| **totale** | **390** | **237** | **379.617** |

Undici cartelle sono state scartate perché prive di `DB_setup` (tra queste alcune marcate `SBAGLIATO`/
`NON USARE` nel nome).

Il box della riga #14 è minuscolo: mediana **74×25 px** su frame per lo più 1920×1080, cioè lo **0,09%**
dell'area. Nel 94% dei casi cade fuori dal rettangolo ecografico (#11), cioè nella cornice della UI.

## 2. La ground truth si ricostruisce esattamente dal crop

Il crop salvato in `DB_echo/<setup>/probe_name.png` e la riga #14 sono legati da un bordo costante,
misurato su tutto il volume:

```
crop top-left = (riga14_left + 3, riga14_top + 3)
crop size     = (larghezza_riga14 - 5, altezza_riga14 - 5)
```

Cercando il crop nei frame con template matching (`TM_CCOEFF_NORMED`) il ritrovamento è al **98%** e
l'offset è costante `(+3, +3)` in 280 casi su 293 verificati a campione. Questo rende la GT **automatica
e pixel-esatta su ogni frame**, non una sola scatola copiata su tutta la cartella.

## 3. Il vendor non è la variabile giusta

Dentro lo stesso **layout UI** (vendor + risoluzione + box riga #13) il box non si muove: range mediano
**0 px**, p90 ≤ 2 px. Ma un box costante *per vendor* (leave-one-out) fallisce:

| predittore | IoU mediana | ≥0,5 | IoU = 0 |
|---|---:|---:|---:|
| box costante globale | 0,000 | 0% | 89% |
| box costante per vendor | 0,000 | 17% | 79% |
| box costante per vendor+risoluzione | 0,000 | 25% | 66% |
| box costante per layout UI | 1,000 | 62% | 32% |

Su Esaote il costante per-vendor sbaglia completamente nel 100% dei casi: 140 configurazioni sono
**84 layout diversi**. Separare per vendor non riduce la multimodalità, la sposta di un livello.

## 4. Quanto serve davvero il vendor: test non parametrico

1-NN sull'**aspetto** del frame (immagine ridotta a 64×48), escludendo sempre la stessa famiglia UI:

| ricerca del vicino | IoU mediana | ≥0,5 | IoU = 0 |
|---|---:|---:|---:|
| globale, senza sapere il vendor | 0,577 | 54% | 33% |
| ristretta allo stesso vendor | 0,604 | 58% | 28% |

Il guadagno è **+0,03**. Motivo: l'84% dei vicini per aspetto è già dello stesso vendor — il vendor è
già scritto nell'immagine, l'etichetta è ridondante. Guardare l'immagine vale molto di più che conoscere
il marchio (0,577 contro 0,000 del prior per vendor).

## 5. Il precedente della riga #13

La riga #13 (`RECT_NAME_ECHO`, template del nome ecografo) è lo stesso identico problema ed è già stata
affrontata con **14 reti per-vendor** (`vendor_line13_template_map.json`). Val IoU raggiunte:

| vendor | val IoU | immagini train |
|---|---:|---:|
| Biopsee | 0,89 | 381 |
| Esaote | 0,84 | 25.764 |
| Siemens | 0,75 | 188 |
| Canon | 0,74 | 487 |
| BK | 0,65 | 10.831 |
| Terason | 0,61 | 189 |
| Mindray | 0,26–0,44 | 562–617 |
| GE | 0,36–0,40 | 1.909–6.411 |
| Hitachi | 0,27–0,38 | 2.906–11.533 |
| Koelis | 0,17–0,26 | 14–103 |
| Toshiba | 0,16 | 170 |
| Philips | 0,08 | 188 |
| Alpinion | 0,00 | 354 |

La coda lunga non ha imparato nulla, e infatti la pipeline continua ad appoggiarsi al resolver storico.

## 6. Il vero collo di bottiglia è la risoluzione, non la partizione

Con un regressore bbox a pooling globale su input 320–384 px, un box di 74×25 diventa ~15×5 px: la
precisione spaziale è persa prima della testa di regressione. Per questo il detector qui è
**CenterNet-style** (ResNet18 + decoder FPN a stride 4, teste heatmap/offset/size): mantiene risoluzione
e fornisce una confidenza vera (il picco della heatmap). A valle resta la catena di raffinamento già
esistente per la #13 (template matching in un intorno + dark trim), che porta il box all'estensione
esatta del testo.

## 7. Dataset prodotto

`artifacts/20_datasets/probe_template_line14_20260917/` — costruito con
`tools/ultrasound/prepare_probe_template_dataset.py` (250 frame per cartella):

- **93.243** frame analizzati, **91.456** con template trovato (98,1%), 1.787 negativi
- **390** configurazioni, **237** layout UI, **205** gruppi leak-free
- stato GT: `ok` 354, `fss_mismatch` 33, `low_hit_rate` 3
- split: train 265 cartelle / 63.665 frame, val 60 / 13.481, test 65 / 14.310
- nessun gruppo, layout o nome macchina attraversa gli split (verificato)

Gli split sono a livello di **layout**, non di cartella: le 80 coppie L/T della stessa macchina
finiscono nello stesso split. Splittando per cartella i numeri sarebbero gonfiati.

I negativi non sono un artefatto: 1.454 su 1.787 sono i frame che il software legacy stesso aveva
raccolto come `*_negative_*` (schermate di altre macchine), gli altri 333 vengono da poche cartelle in
cui il nome sonda compare solo in una parte delle acquisizioni. Insegnano alla rete ad astenersi.

## 8. Ground truth da rivedere (36 cartelle)

Pagina di review: `artifacts/40_outputs_eval/probe_template_line14_review_20260917/index.html`
(rosso = box del `.fss` legacy, verde = box ricostruito dal crop).

Due famiglie di disaccordo:

- **il box legacy è su un'area diversa** (IoU 0): es. `BK FlexFocus400_8848_4.106.14283.37 - L/T`, dove
  il crop inquadra `8848` e la riga #14 punta a un rettangolo vuoto sotto la data. Qui il `.fss` legacy
  sembra semplicemente sbagliato.
- **il box legacy è più generoso**: es. `Esaote MyLab 80XPro F100200`, dove il crop sta stretto su
  `TLC 3-13` e il box legacy include anche parte di `Urologia`. Qui è una differenza di convenzione di
  chi ha disegnato il template.

Le tre `low_hit_rate` non sono errori: il nome sonda compare solo in una frazione dei frame (Canon 48%,
due Esaote 7%), il resto diventano negativi corretti.

## 9. Esperimento: generica vs per-vendor

Cinque training a parità di condizioni (`--image-size 512 --epochs 10 --max-images-per-folder 40`,
stessi split, stesso seed): una rete generica su tutti i vendor e quattro reti dedicate ai soli vendor
con abbastanza dati. Valutazione sulle **stesse cartelle di test**, che per costruzione sono **layout mai
visti in training**.

| vendor | cartelle test | generica: IoU mediana | ≥0,5 | per-vendor: IoU mediana | ≥0,5 |
|---|---:|---:|---:|---:|---:|
| Esaote | 18 | 0,782 | 100% | **0,812** | 89% |
| BK | 15 | **0,693** | 60% | 0,682 | 80% |
| Hitachi | 10 | 0,000 | 30% | 0,000 | 30% |
| GE | 5 | **0,875** | 100% | 0,862 | 100% |

Confronto appaiato cartella per cartella (48 cartelle): la rete per-vendor è **migliore su 13, peggiore
su 12, pari su 23**. Delta IoU **mediano +0,000**, medio +0,053. È esattamente l'ordine di grandezza
previsto dall'analisi non parametrica (+0,03): **nessun vantaggio sistematico**.

Il budget di training non è il fattore limitante: tutte e cinque le reti vanno in plateau prima
dell'epoca 10 (generica dall'epoca 3, Esaote dall'epoca 4, BK dall'epoca 1, GE dall'epoca 2), quindi il
fatto che la generica veda più campioni per epoca non falsa il confronto.

### La rete generica sull'intero test (61 cartelle, 12 vendor)

| metrica | valore |
|---|---:|
| IoU mediana per cartella | 0,693 |
| IoU ≥ 0,5 | 64% |
| IoU ≥ 0,75 | 41% |
| errore massimo per coordinata, mediana | 6,4 px |
| errore ≤ 5 px | 36% |
| errore ≤ 10 px | 59% |

Confronto con il metodo in produzione oggi, sulle stesse cartelle:

| metodo | copertura | IoU mediana | IoU ≥ 0,5 | riga identica |
|---|---:|---:|---:|---:|
| resolver storico (riga #14 attuale) | 82% | 0,000 | 12% | 0/53 |
| rete generica | 100% | 0,693 | 64% | — |

Su macchine già configurate in passato il resolver resta esatto (trova la riga identica in archivio);
su macchine **nuove**, che è il caso d'uso della rete, sbaglia il rettangolo nell'88% dei casi in cui
pure risponde.

### La confidenza funziona come meccanismo di astensione

Il picco della heatmap separa bene i casi buoni dai fallimenti:

| soglia score | accettate | corrette (IoU ≥ 0,5) | errore ≤ 5 px | buone mandate in review |
|---:|---:|---:|---:|---:|
| 0,0 | 61/61 | 64% | 36% | 0 |
| 0,1 | 43/61 | 77% | 49% | 6 |
| **0,2** | **35/61** | **91%** | 57% | 7 |
| 0,5 | 33/61 | 91% | 58% | 9 |
| 0,8 | 24/61 | 88% | 54% | 18 |

I fallimenti arrivano quasi tutti con score basso (mediana 0,00–0,15): la rete **non forza** una
predizione, si astiene. Restano **3 casi su 61 di errore sicuro di sé** (score ≥ 0,5 e IoU 0):
`Biopsee_3_3(6)_LA-CLA_1536N1011_NO_FUSION_L/T` e `Hitachi_Aloka_AriettaS60_6.1.1_L441`.

### Dove fallisce

Il discriminante è se la **famiglia di UI** è rappresentata nel training:

| caso | cartelle riuscite |
|---|---:|
| vendor + risoluzione già visti in train | 37/54 (69%) |
| combinazione mai vista | 2/7 (29%) |

I 22 fallimenti sono concentrati su Hitachi 7 (le famiglie `Aloka Noblus`, `Arietta 750VE`, `Arietta S60`
non hanno gemelli in training), BK 6, e la coda Koelis 3, Biopsee 2, Mindray 2, Siemens 1, Toshiba 1 —
vendor che nel test hanno 1–3 cartelle e in training pochissimi layout. Non è un problema di *quale*
rete: la rete Hitachi dedicata fallisce esattamente sulle stesse cartelle (IoU mediana 0,000, identica
alla generica).

## 10. Conclusione

**Rete unica generica.** L'esperimento conferma l'analisi: il vendor come partizione dei dati non porta
guadagno misurabile (13 vittorie contro 12, delta mediano nullo), mentre costa 16 modelli da mantenere e
lascia scoperta tutta la coda. Il vendor semmai entra come input di condizionamento nella stessa rete.

Con l'astensione attiva la rete è già utilizzabile nel loop "AI propone → l'utente conferma": su macchine
mai viste propone un rettangolo su ~57% delle configurazioni con il 91% di correttezza, e segnala le
altre invece di scrivere un numero sbagliato. Oggi al suo posto c'è un resolver che su quelle stesse
macchine risponde con sicurezza e sbaglia nell'88% dei casi.

## 11. Cosa manca (decisioni aperte)

- **Soglia di accettazione**: la tabella sopra dà il trade-off, la scelta è da fare.
- **Le 36 cartelle in review**: in particolare i casi in cui la riga #14 legacy punta a un'area vuota —
  se sono errori storici vanno corretti, e cambiano la GT di quelle configurazioni.
- **Stadio di raffinamento**: l'errore mediano di 6,4 px è quello grezzo della rete. Per la riga #13
  esiste già la catena template matching + dark trim (`_postprocess_line13_box_with_template`,
  `_postprocess_line13_box_with_dark_trim`): riusandola l'errore dovrebbe scendere all'estensione esatta
  del testo. Non è stata ancora agganciata qui.
- **Famiglie UI scoperte**: Hitachi Aloka Noblus / Arietta 750VE / S60, Koelis, Biopsee, Mindray DC70,
  Siemens, Toshiba. Servono più configurazioni (anche dal volume `n3`) oppure si accetta che finiscano
  in review.
- **Risoluzione e durata**: questo run è a 512 px, 10 epoche, 40 frame per cartella — un confronto, non
  un modello ottimizzato. A 768 px e con tutti i frame c'è margine, soprattutto sull'errore in pixel.

Artefatti: dataset `artifacts/20_datasets/probe_template_line14_20260917/`, modelli
`artifacts/30_models/probe_template_line14_20260917/` (`generic`, `vendor_*`, `comparison.json`),
review `artifacts/40_outputs_eval/probe_template_line14_review_20260917/`.
