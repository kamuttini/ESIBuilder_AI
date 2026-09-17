# Linee guida: dove siamo, la mattina del 16 settembre

## Cosa puoi usare da subito

**Nell'app, step «Linee guida».** Il bottone *Misura gli aghi* fa l'intera catena e propone
`#22` e `#23`. Serve che siano gia' fatti il rettangolo, i pixel ratio (step depth e scala) e la
ricerca del materiale (step Import e analisi). *Porta la proposta nel valore* precompila l'editor
nella forma che il writer si aspetta.

**Nell'app, step «Import e analisi».** Il bottone `aghi` su ogni cartella dice dove sta il
materiale di calibrazione, con i nuovi comandi per escludere un fotogramma e per segnalare due
segmenti che sono lo stesso ago.

**Da riga di comando**, su una cartella qualunque:

```bash
python3 tools/needle/propose_guide_lines.py \
  --folder "/percorso/della/cartella/Aghi" \
  --rect 364,159,1468,882 --pixel-ratio-y 0.05443 --size 1920x1080
```

Per far ripartire l'app:

```bash
cd /Users/camilla/Documents/Develop/ESIBuilder_AI/.claude/worktrees/fss-pipeline-monitoring-tool-ee56d1
/Users/camilla/Documents/Develop/ESIBuilder_AI/OldSoftwareEsiBuilder/.venv-mps/bin/python \
  tools/app/server.py --port 8800 \
  --projects-root /Users/camilla/Documents/Develop/ESIBuilder_AI/artifacts/80_app_projects
```

## Quanto vale, in numeri

Misurato su 66 fotogrammi di 32 acquisizioni **mai usate per tarare niente**:

| | entro 1° | entro 3° |
|---|---|---|
| rilevatore grezzo | 23% | 36% |
| con le regole geometriche | **30%** | **47%** |

Quindi: **proposta da confermare, non risultato**. Un terzo delle volte l'angolo e' gia' dentro
la tolleranza di 1°; circa metà delle volte ci si arriva correggendo poco.

Su una configurazione reale (Esaote MyLab 80xPro) la catena ha prodotto 16.29° contro 15.93°
legacy, e 9.89 mm contro 9.85.

## Cosa e' dimostrato

**La geometria e' corretta e il budget di 1° e' raggiungibile.** Andata e ritorno sul formato
legacy: 12.860 combinazioni, errore mediano 0.03°. E i 181 aghi tracciati a mano cadono a
**0.94° di mediana** dall'angolo legacy quando l'accoppiamento e' giusto, 80% entro 2°.

Il collo di bottiglia e' solo il rilevatore, non la matematica ne' il formato.

## Cosa non e' coperto

**Le biplane.** Si calibrano sulle immagini di linee di biopsia con i pallini, e sono due casi
diversi:

- **guida a linea** (`UA1250`, `UA1341`): il rilevatore nuovo funziona — verificato su BK3000,
  16 pallini, retta a 0.016 di regolarita' — ma ha pochi casi su cui misurarsi;
- **griglia** (`GUIDA AGHI - Abilitate` del BK5000): non e' una linea ma una **matrice** di
  pallini, la battaglia navale del file `.grid`. Va trattata come reticolo, non come retta, e
  non e' coperta.

**La depth per fotogramma.** Senza sapere a che profondita' e' stato preso ogni fotogramma, la
stessa distanza viene proposta per tutte le depth. Va corretta una per una.

## Cosa ho provato e non ha funzionato

Vale la pena saperlo, per non rifarlo:

- **filtro adattato** al posto di Hough: trova piu' spesso qualcosa (48/48 contro 36/48) ma
  sceglie molto peggio (23% contro 72%). Ora fa solo da ripiego.
- **classificatore addestrato** per riordinare i candidati: AUC 1.00 in validazione e da 27.8° a
  2.5° di errore — tutto perdita di dati. Sui fotogrammi di configurazioni mai viste fa identico
  al rilevatore grezzo. 61 aghi su 30 fotogrammi e 4 configurazioni sono troppe poche scene.
- **pseudo-etichette dal legacy**: quelle che produce sono ottime (0.71° di mediana contro gli
  aghi umani) ma sono 39 su 657 fotogrammi. Meno di quante Camilla ne ha fatte a mano in una
  sera, quindi non e' la strada per crescere il dataset.
- **estensione verticale** come filtro: la caratteristica che separa meglio di tutte in assoluto,
  e come filtro peggiora (3.48° -> 9.74°), perche' taglia i pezzi corti di aghi veri.
- **riverbero, frastagliatura, luminosita' della punta**: misurate, non separano.

Le due cose che hanno funzionato — inclinazione minima e partenza nella meta' alta — vengono
entrambe da Camilla che guardava gli errori, non da caratteristiche progettate a tavolino.

## Se vuoi andare avanti

1. **Piu' fotogrammi annotati da acquisizioni diverse.** E' l'unica leva rimasta per il
   rilevatore: la galleria `annotazione_varieta.html` e' fatta apposta, 116 fotogrammi da 40
   acquisizioni, e con altri ~100 il classificatore ha una possibilita' vera.
2. **Il reticolo della griglia** per le biplane, che e' un blocco a se'.
3. **La depth per fotogramma**, che oggi e' l'approssimazione piu' grossa della proposta.

## Che cosa succede con più immagini che angoli (misurato il 17/09)

Domanda di Camilla: se le immagini con la sonda in acqua sono più degli angoli del kit, che
fine fanno? Provato ricostruendo le misure dai 380 setup legacy leggibili — una misura per
ogni casella angolo × depth, più un'immagine in più — e contando le famiglie che escono.

| caso | famiglie rispetto agli angoli |
|---|---|
| una immagine per casella (angolo × depth) | esatte, 380 su 380 |
| una immagine per angolo, a una sola depth | esatte, 380 su 380 |
| una in più dello stesso angolo **a un'altra depth** | esatte, 380 su 380 |
| una in più dello stesso angolo **alla stessa depth** | **+1, in tutti e 380** |

Con rumore del rilevatore (sigma 1.2°, la sua precisione misurata) i risultati non cambiano.

Quindi le immagini in più non danno fastidio, e anzi servono, **finché stanno a depth diverse**:
entrano nella famiglia del loro angolo e la colonna di #22 passa dall'essere estrapolata con la
pendenza tipica (0.34 mm di errore) all'essere una retta stimata sui dati (0.056 mm).

Due immagini dello stesso angolo **alla stessa depth** invece diventano sempre due famiglie. È
la regola «mai due misure della stessa depth nella stessa famiglia», che esiste perché senza di
lei una correzione a mano può far cadere una riga dentro la tolleranza di quella accanto e far
sparire un angolo. La si paga qui.

Non è aggirabile guardando i numeri: nell'archivio due linee guida *diverse* possono stare a
0.01° l'una dall'altra (14% delle coppie adiacenti sta sotto i 3°) e distinguersi solo per #22,
di 4.11 mm mediani ma con un minimo di 0.30 mm — sotto la precisione con cui il rilevatore
misura la distanza. Una regola «stessa depth, stesso angolo *e* stessa distanza» non separa i
due casi.

Provata comunque una regola alternativa che raggruppa su angolo **e** distanza riportata a una
depth comune con la pendenza tipica: ricostruisce le famiglie giuste nel 67% dei setup contro
il 100% della regola attuale. Peggiora, e resta quella di adesso.

Quello che si fa invece: lo step dice quante famiglie sono uscite rispetto agli angoli che il
kit dichiara («4 su 4 — tutte», oppure «5 su 4 — 1 di troppo»), e il fotogramma in eccesso si
toglie dallo studio con il pulsante sulla sua miniatura nella galleria dell'import.
