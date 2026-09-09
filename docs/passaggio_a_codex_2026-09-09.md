# Passaggio di consegne — app ESIBuilder AI (9 settembre 2026)

Documento per chi riprende il lavoro sull'app (`tools/app/`) da un altro assistente.
Contiene: come far girare le cose, le regole di lavoro che sono costate errori,
lo stato attuale e cosa resta aperto.

## 1. Cos'e'

Un'app locale (Flask + JavaScript senza build) che porta una cartella di acquisizioni
ecografiche fino a un file `.fss` valido: l'AI propone, l'utente conferma o corregge.
Utente finale: colleghi interni su **Windows**. Sviluppo su macOS.

- server: `tools/app/server.py` (~7000 righe, un file solo)
- interfaccia: `tools/app/static/{app.js, box_editor.js, orientation_view.js,
  depth_view.js, scale_view.js, lente.js, style.css, index.html}`
- stato di un progetto: `artifacts/80_app_projects/<id>/project.json`

## 2. Ambiente

```bash
# il server che usa Camilla, sulla porta 8800
/Users/camilla/Documents/Develop/ESIBuilder_AI/OldSoftwareEsiBuilder/.venv-mps/bin/python \
  tools/app/server.py --port 8800 \
  --projects-root /Users/camilla/Documents/Develop/ESIBuilder_AI/artifacts/80_app_projects
```

Il Python **non** e' quello di sistema: e' il venv `OldSoftwareEsiBuilder/.venv-mps/bin/python`
(torch con device `mps`, PIL, joblib, scikit-learn 1.6.1 — la 1.9 non riesce a
deserializzare il ranker della depth).

I file statici si rileggono a ogni richiesta; il **Python no**: dopo una modifica al server
va riavviato, se no gli endpoint nuovi rispondono 404. Dopo una modifica ai `.js` basta
ricaricare, ma spesso serve **cmd+shift+R** perche' il browser tiene la versione vecchia.

## 3. Quality gate — prima di ogni commit

```bash
/Users/camilla/Documents/Develop/ESIBuilder_AI/OldSoftwareEsiBuilder/.venv-mps/bin/python \
  tools/app/selftest_roundtrip.py \
  /Users/camilla/Documents/Develop/ESIBuilder_AI/OldSoftwareEsiBuilder/templates/DB_setup/*.fss
```

Deve stampare «tutti i 10 file legacy rigenerati identici». E' il controllo che nessuna
modifica ha cambiato il formato del `.fss`.

Per il JavaScript non c'e' node: si controlla la sintassi con JavaScriptCore.

```bash
/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc \
  -e "try{new Function(readFile('tools/app/static/app.js'));print('ok')}catch(e){print('FAIL '+e)}"
```

## 4. Regole di lavoro (imparate sbagliando)

1. **Mai scrivere sui progetti veri per provare.** Un `POST` di prova ha cancellato ore di
   lavoro. Si copia il progetto in una cartella di scratch e si fa girare un secondo server
   su un'altra porta (8811) puntato li'.
2. **Verificare in lettura o su copia.** I `GET` sui progetti veri vanno bene.
3. **Gli script di patch devono scrivere subito.** Uno script Python con cinque
   sostituzioni e un `write_text` finale, se la quinta fallisce, **perde le prime quattro** e
   lascia il file a meta' — con riferimenti a variabili che non esistono. E' successo tre
   volte. Applicare una sostituzione per volta, ognuna col suo `write_text`.
4. **Il pannello browser blocca i popup.** La lente (`window.open`) non si puo' collaudare
   di la': si sostituisce `window.open` con un iframe e si fa girare il codice vero dentro.
5. **`rectStudyCard` ricompone la scheda con `replaceChildren` in fondo**: gli `append`
   sparsi prima non contano. Un pulsante aggiunto e mai comparso viene da li'.
6. **Le sezioni ricostruiscono il DOM**: quello che si aggancia al `body` (lente,
   anteprime) va rimosso quando il pannello cambia, se no resta appeso.
7. Documenti e commenti di dominio in italiano; il codice puo' restare in inglese dove gia'
   lo e'. I messaggi di commit spiegano **perche'**, non cosa.

## 5. Ordine della pipeline (deciso con Camilla, 9 settembre)

1. deduplicazione bit a bit
2. rotazione (OSD)
3. deduplicazione a meno del timestamp (l'area dell'orologio si indica a mano)
4. riconoscimento vendor — decide se usare le reti specializzate
5. sonda
6. abbozzo di rettangolo ecografico (mediana sui campioni) — serve solo a far girare
   piano L/T e orientamento
7. divisione L/T, e se ci sono entrambi due progetti distinti
8. orientamento
9. depth
10. scala

I punti 1-7 sono implementati in `_run_import_analysis`. **Il punto 8 no**: l'orientamento
gira ancora dentro l'analisi iniziale, cioe' *prima* della divisione. E' il prossimo lavoro
(vedi 7).

## 6. Cosa e' stato fatto negli ultimi giorni

- **Dedup con area dell'ora**: `deduplicate(paths, timestamp_box)` confronta i **pixel** con
  un rettangolo annerito. Scarto diviso in `identiche` e `solo_timestamp`; entrambe fuori
  dallo specchio di lavoro, elencabili nell'interfaccia.
- **Divisione L/T**: `predict_lt_each` da' il piano immagine per immagine;
  `POST /planes` classifica, `POST /planes/correct` corregge (una o piu'), `POST /split`
  crea o **aggiorna** il progetto della T. Il T eredita codici, vendor, sonda, rettangolo e
  il rapporto pixel/mm della scala — copia, non legame (`derived_from` / `split_into`).
  Nell'interfaccia: due strisce di immagini, anteprima al passaggio del mouse, selezione
  multipla con shift+clic, vista a tutto schermo con frecce e tasti `L`/`T`.
- **Lente** (`lente.js`): finestra separata per il secondo schermo. Disegna riquadri, assi e
  corde; `focus` decide *cosa inquadrare* separatamente da cosa disegnare; bersagli
  cliccabili; spostamento della vista; lati afferrabili lungo tutto il bordo; frecce per
  muovere di un pixel; appartiene alla sezione che la usa (`source`) e si sospende quando la
  pagina cambia sezione.
- **Layout**: ogni sezione con un'immagine ha la stessa forma — immagine grande a sinistra
  (griglia `minmax(0,1fr)`, appiccicata in alto), strumenti in una colonna fissa di 380 px a
  destra, e una barra sopra l'immagine con frecce e lente sempre nello stesso posto.
- **Eliminazione progetto**: `DELETE /api/projects/<id>`, con conferma scrivendo l'id.
  Cancella lo studio, mai la cartella delle immagini; scioglie il legame dell'altra meta'
  di uno sdoppiamento.

## 7. Cosa resta aperto

1. **Spezzare la catena dei moduli in due tempi.** Oggi `_run_advanced_stages` fa
   orientamento + depth + scala dentro l'analisi iniziale. Dovrebbe fermarsi al punto 6
   dell'ordine, e orientamento/depth/scala partire **dopo** la divisione L/T, per progetto.
2. **Il rettangolo grossolano si calcola su un campione che mescola L e T.** Per l'abbozzo
   va bene; dopo la divisione andrebbe rifatto per progetto, e oggi non succede da solo.
3. `feedback/inbox.jsonl` non viene ancora alimentato dall'app (richiesta vecchia, mai
   fatta).
4. Righe `#19-#21` ancora vuote su alcune cartelle (`scale_line21_incomplete`), e
   `no_ladder` sulle depth basse di BK.
5. Domande aperte per Francesca (riferimento di dominio su ESI): dimensione del crop, `#13`
   e `#14` vuoti, versione in `#01`, convenzione per l'orientamento mancante, cartelle con
   entrambi i piani, e quale testo ESI si aspetta in `#13` sul BK Specto.
6. Su `prova_4` le depth sono memorizzate dieci volte piu' piccole del vero (3.0 invece di
   30 mm): l'etichetta e' in centimetri e la conversione non e' stata applicata. Da chiarire
   con Camilla prima di generare quel `.fss`.

## 8. Progetti di prova utili

| progetto | cosa serve a provare |
|---|---|
| `prova_2` | BK Specto, 56 immagini, **due piani** 28 L e 28 T — la divisione |
| `prova_3` | Philips, depth dalla scala, righello leggibile |
| `prova_4` | Hitachi, etichetta `R:3.00` — lo stringimento del riquadro col prefisso |
| `prova_gruppo_mancante` | riquadri depth gia' stretti |
| `esaote_..._calenzano` | 666 immagini — le prestazioni |

Le immagini stanno su volumi esterni (`/Volumes/SSD_esi1_n1`, `SSD_esi1_n3`, `NO NAME`):
se un volume non e' montato le sezioni si vedono vuote e le anteprime danno 404.
