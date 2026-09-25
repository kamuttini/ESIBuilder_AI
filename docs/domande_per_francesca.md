# Domande per Francesca

Lista viva delle domande di dominio ESI da porre a Francesca. Ogni voce: contesto minimo,
domanda, e perche' serve (cosa blocca o cosa cambia nel lavoro).

Aggiornata: 2026-09-17.

---

## Blocco FREEZE / immagini proibite

Contesto ricavato dal codice legacy (`OldSoftwareEsiBuilder/qfilefreeze.*`,
`wdgpageproibited.cpp`) e dal ground truth su `SSD_esi1_n1` (401 file `setup_N.freeze`,
700 entry, 899 crop `proibited_screen_*.png`).

### F1 — Cosa fa ESI a runtime quando trova il template di una proibita
**Domanda:** quando ESI riconosce il template di una proibita (freeze, CFM, PW, zoom...)
dentro `rectFind` sul frame live, cosa succede esattamente? Blocca la procedura, mostra
solo un avviso, sospende il tracking dell'ago?
**Perche' serve:** determina quanto e' costoso un falso positivo rispetto a un falso
negativo, e quindi la policy di soglia della rete (quanto conservativa deve essere).

### F2 — Output atteso dalla rete
**Domanda:** nella nuova app, cosa vogliamo che l'AI proponga per le proibite?
(a) solo "in questa cartella c'e'/non c'e' l'immagine freeze";
(b) quale immagine e' la freeze;
(c) anche il rettangolo template gia' disegnato, da confermare/correggere.
**Perche' serve:** cambia l'architettura (classificatore vs detector) e la metrica di
accettazione. — *in attesa, Camilla la chiede a Francesca*

### F3 — Significato di `isScreenSaver`
**Domanda:** nel file `.freeze` il flag `isScreenSaver` e' true in 202 entry su 700 (29%).
Cosa identifica: il salvaschermo del monitor dell'ecografo? Un template che va trattato
diversamente a runtime?
**Perche' serve:** decide se e' una classe a se' nel dataset o un attributo da ignorare.

### F4 — Quali proibite sono obbligatorie
**Domanda:** per un setup, quali stati proibiti vanno sempre configurati e quali sono
facoltativi? Nel GT la media e' 2,1 proibite per setup (205 setup su 334 ne hanno una sola,
il massimo e' 14).
**Perche' serve:** dice se l'assenza di un'immagine freeze in una cartella e' un dato reale
o una lacuna di acquisizione (cioe' se i "negativi per cartella" sono affidabili).

### F5 — Criterio con cui sceglieva il rettangolo template
**Domanda:** quando disegnava il rettangolo, seguiva una regola (il piu' piccolo possibile?
solo l'icona? icona + scritta? un elemento che non si sovrappone mai all'immagine
ecografica?) o era a sensibilita'?
**Perche' serve:** e' il target che la rete deve imparare; se la regola e' esplicita si puo'
misurare l'accordo ed evitare di imparare rumore.

### F6 — Immagine negativa fittizia
**Domanda:** dal brainstorming: "per le proibite in genere mette immagine freeze, se non ce
l'ho crea un'immagine negativa fittizia (un'immagine che non apparira' mai)". Come viene
costruita questa immagine fittizia e in quali casi?
**Perche' serve:** sono esempi da escludere dal dataset (sono sintetici, non stati reali
dell'ecografo) e vanno riconosciuti automaticamente.
