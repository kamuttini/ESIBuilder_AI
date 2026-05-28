# Configurazione

Non è una sonda biplana → no riconoscimento L/T

nome della configurazione:

nome eco  + modello + v software + codice sonda +  stato + nome città/ospedale  (L/T)

ID CONF

ID ECHO

ID PROBE

tipo probe → lineare 

nell’80% dei casi NDG è già associato alla sonda (è possibile fare un training anche in questo senso, ma è più complesso 

ndg + bracket alle volte già esistono con le stesse configurazioni, ma il codice del kit guida (e quindi la label su esi è diversa) quindi posso riusare un vecchio file ndg basta che l orinomino. 

se è biplana 

uso guida lineare e quindi nelle imgs vedo linee di biopsia → guida lineare (ce n’è una sola per sonda )

se c’è la griglia → devo acquisire ogni depth 

per sonda lineare o convex 

svariati ndg che dipendono dal tipo di bracket (acquistati) che si aggancia alla sonda 

i bracket hanno alcune posizioni prefissate a diverse angoalzioni

i kit guida 

size image e size image sample

può succedere che il modello ecografo non sia indicato 

il rettangolo non può essere più grande di 1450 x 820

per le proibite in genere mette immagine freeze, se non ce l’ho crea un’immagine negativa fittizia (un’immagine che non apparirà mai) 

per esibuildere mettere un’opzione se c’è o meno l’immagine freeze 

orientamento → parto dal tempalte il più piccolo possibile

voglio trovare il rettangolo che racchiuda tutte le posizioni per ciascun orientamento 

Alle volte vengono tolte delle depth che potrebbero dare errore su esi

se le depth o il template ddell’orientamento sono allinterno dell’immagine ecografica allora è difficile da distinguere quindi in quel caso franci chiede di usare immagini con più contrasto 

di un solo orientamento (gli altri vengono generati automaticamente ribaltando l’immagine) selezione tutte le depth per cui ho fatto ocnfigurazione 

riconoscere:

1. tolgo le immagini duplicate
2. HDMI O VGA da nome file 
3. risoluzione da proprietà 
4. cerco testo qualsiasi e capisco se è da ruotare → ruoto
5. cerco rettangolo ecografico → se è maggiore di 1450 x 820  o molto piccolo [DEFINIAMO PICCOLO] → resize sotto 1450 x 820 mantenendo proporzioni (in realtà posso cercare di avvicinarmi sempre)
6. ricerca ocr su tutta l’immagine 
7. marca ecografo
8. modello eco
9. modello sonda (da excel) → trovato ricavo se biplana, convex o lineare
    
    → non trovato -> papabile nome sonda fra il testo → riconosco rettangolo ecografico  → dentro il rettangolo cerco se ci sono L-T (molto diverse) → si ? → biplano
    
    → no? riconosci se lineare o convex 
    

Se mancassero nel file excel una di queste informazione (marco eco - modello eco o modello sonda ) gestire l’inserimento nei rispettivi fogli excel

se non compaiono (quindi skippate e skippabili) messaggio perchè venga gestito manualmente 

1. se biplana riconosco riconosco e L/ e rinomino L_ o T_ ( se non è biplana partirò dall’orientamento)
2. trovo orientamento (template e rettangoli per ogni orientamento) → rinomino con NF/LR/UD/LRUD
3. cerco la scala → cerco tutti il numero più grande in prossimità della scala sono sempre in cm nel file fss sono in sempre in mm 
    
    attualmente esibuilder può partire solo da 0 
    
    la scala non può stare stare fuori dal rettangolo ecografico, mentre il template il template si 
    

EUCLIDIAN DISTANCE AFTER TRHESHOLD