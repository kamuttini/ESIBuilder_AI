# Obiettivi e requisiti

### **Acquisizione delle immagini**

Si ritiene opportuno continuare ad acquisire le immagini tramite ESI Builder. Spostare la fase di configurazione direttamente su ESI renderebbe il processo dipendente dall’operatore sul campo, che spesso non è sufficientemente affidabile per questo tipo di attività. Inoltre, mantenere la configurazione su ESI Builder garantisce maggiore controllo, ripetibilità e qualità delle configurazioni.

### **Suddivisione del processo in due fasi**

Si propone di suddividere il lavoro in due fasi distinte:

- una prima fase principale su **ESI Builder**, in cui viene creata e addestrata la configurazione;
- una seconda fase opzionale su **ESI**, accessibile solo in una modalità protetta (ad esempio Ultra Service), dedicata all’adattamento della configurazione sul campo.

La fase su ESI servirebbe esclusivamente per adattare configurazioni già esistenti nel caso in cui, ad esempio, una nuova versione software dell’ecografo abbia spostato leggermente alcuni template o modificato il layout grafico. In questa modalità, con l’ecografo collegato, l’utente può scorrere le immagini e indicare manualmente piano, orientamento e depth nei casi in cui il riconoscimento automatico fallisca, fornendo al sistema ulteriori esempi utili.

### **Pagina iniziale della configurazione**

La configurazione inizia sempre con una pagina in cui vengono inseriti i codici della nuova configurazione (codice config, NDG, ecografo, sonda, ecc.). Sarebbe utile poter modificare questi parametri anche successivamente, durante la fase di configurazione, nel caso ci si accorga di un errore o cambi qualcosa. Il sistema dovrebbe adattarsi dinamicamente senza richiedere di ricominciare l’intera procedura. Per come funziona adesso, una volta confermati questi dati se vuoi cambiarli dopo non si può e devi rifare la configurazione da capo.

In questa fase potrebbe essere utile specificare in modo più strutturato il nome dell’ecografo, il tipo e il modello, così che il sistema possa collegarsi a immagini e configurazioni già esistenti per lo stesso vendor, continuando l’allenamento progressivo.

### **Gestione generale dei template**

Per tutti gli step successivi, l’idea è che il sistema proponga automaticamente la posizione dei template (se li riconosce), chiedendo all’utente solo una conferma o una correzione. Il tracciamento completamente manuale dovrebbe essere l’eccezione e non la regola.

Le immagini duplicate devono essere automaticamente scartate.

### **Risoluzione**

La risoluzione dell’immagine deve essere riconosciuta automaticamente. L’utente deve comunque avere la possibilità di modificarla manualmente durante la configurazione, ad esempio per adattare meglio l’immagine ecografica o testare dimensioni diverse.

Una volta creata la configurazione, questa deve essere robusta ai cambi di risoluzione: se cambia lo schermo dell’ecografo o l’output video, il sistema deve adattarsi automaticamente senza andare in errore, anche se le posizioni e le dimensioni dei template risultano diverse. Deve si riconoscere che la risoluzione magari che sta leggendo è diversa da quella con cui è stata fatta la configurazione, però basandosi su questa consapevolezza dovrà riadattare la posizione dei template in base a che tipo di risoluzione legge ora.

### **Riquadro ecografico**

Idealmente il riquadro ecografico dovrebbe essere riconosciuto automaticamente. L’eventuale tracciamento manuale dovrebbe servire solo come supporto iniziale, senza vincolare rigidamente la configurazione. L’obiettivo è che il sistema normalizzi l’immagine internamente, mantenendo simmetria e corretto orientamento indipendentemente dalle dimensioni.

 questo punto potrebbe essere più complicato da fare automaticamente, considerando che sarà proprio il riquadro che verrà visualizzato su ESI e che quindi ha dei limiti di grandezza (che dipendono proprio dalle dimensioni dello schermo di ESI), quindi se difficile da fare automatico lasciare questa fare manuale.

quando dovrà riconoscere la scala dell’ecografo e ridisegnare la scala di ESI sopra l’immagine (deve essere identica a quella dell’ecografo senno sballa tutte le dimensioni), per come viene fatto ora si caricano tutte le immagini con tutte le profondità in un orientamento specifico (es: UD), e poi disegno a mano le scale una ad una per ogni profondità. ora che lo farà automaticamente, puo sempre farlo partendo da un singolo orientamento, quindi prende tutte le profondità di UD e riconosce le depth e disegna le scale. poi puo ribaltare in maniera speculare, sopra sotto, e destra sinistra come fa ora, la scala, dato che il riquadro è simmetrico rispetto al centro dell’immagine, pero dovrebbe fare un controllo e verificare se effettivamente viene ribaltata la scala correttamente e se, anche da ribaltata rimane esattamente sovrapposta alla scala ecografica.

### **Riconoscimento dell’ecografo**

Il riconoscimento dell’ecografo non dovrebbe richiedere il tracciamento manuale di un template. Il sistema dovrebbe identificarlo automaticamente sulla base del layout grafico complessivo, degli overlay e delle caratteristiche dell’immagine.

Ogni tanto non c’è un template per l’ecografo nell’immagine, in questo caso dare la possibilità di skippare questo punto e passare a quello dopo (quindi possibilità di mettere tipo: template non presente)

### **Riconoscimento della sonda**

Il riconoscimento della sonda dovrebbe essere automatico ove possibile. In alternativa, l’utente può indicare manualmente il template la prima volta; successivamente il sistema deve essere in grado di riconoscerlo anche se si sposta leggermente o cambia dimensione.

### **Riconoscimento del piano**

Il riconoscimento del piano (T o L) dovrebbe essere completamente automatico. Il sistema può basarsi su template grafici, differenze di depth o layout specifici dell’immagine. Nei casi più complessi, in cui non ci siano indicatori evidenti, il sistema deve comunque fornire una stima e chiedere conferma all’utente se il livello di confidenza è basso.

Questa è la fase più importante che sarebbe da automatizzare, perché spesso capita che non ci sono info nell’immagine ecografica che ti permettono di capire se sei in L o in T.

### **Orientamento**

L’obiettivo è che il riconoscimento dell’orientamento sia robusto rispetto a cambi di risoluzione, scaling e leggere variazioni dei template. Nel caso di vendor come Esaote, il sistema dovrebbe già disporre di un database di template provenienti da altre configurazioni, anche di versioni software diverse. Se il template cambia leggermente a seguito di un aggiornamento dell’ecografo, il sistema dovrebbe comunque riconoscerlo; solo nel caso di un template completamente nuovo deve segnalare un errore e richiedere una revisione della configurazione.

 in tutte queste fasi della configurazione, il software ti consiglia sempre dove si trova il template e tu devi confermare o modificare se sbagliato

### **Depth e scala**

La depth dovrebbe essere riconosciuta sempre a partire dalla scala ecografica. L’utente può indicare una volta l’area complessiva della scala (da 0 a N), valida per tutte le depth e orientamenti. Il sistema deve quindi identificare automaticamente l’inizio e la fine della scala e leggere tutti i numeri presenti.

 se sono necessarie più info manuali richiederle la prima volta, tipo indicare inizio e fine della scala per alcune o tutte le depth, capire se si può non dover fare manualmente o se va fatto la prima volta per forza manualmente.

Questo approccio rende il riconoscimento più robusto, anche nel caso in cui un singolo numero non sia leggibile o si sovrapponga a un’area iperecogena. In questo modo viene eliminato anche il problema delle depth di transizione, poiché la depth non dipende più da un singolo template.

### **Eliminazione delle thresholds**

La fase di impostazione manuale delle thresholds può essere eliminata. Al suo posto, il sistema può scorrere automaticamente le immagini e chiedere all’utente di confermare o correggere ciò che è stato riconosciuto. In alternativa, nella fase finale, tutte le immagini possono essere scorse manualmente una per una, già rinominate automaticamente (ad esempio T_LRUD_50), e l’utente deve solo verificare che piano, orientamento, depth e scala siano corretti.

### **Linee guida**

Per le linee guida, l’utente carica le immagini in base al numero di angoli disponibili. Il sistema traccia automaticamente le linee e l’utente conferma o corregge. Se possibile, sarebbe utile caricare anche i dati dei set guida (distanze reali tra i fori), in modo che il sistema possa gestire automaticamente bracket con configurazioni diverse e nomenclature differenti.