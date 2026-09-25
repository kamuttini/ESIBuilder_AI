/* Lo studio del righello, fotogramma per fotogramma, con le correzioni sotto le dita.

   Il modulo trova la colonna della scala, ne stacca le tacche, distingue lo zero dal fondo,
   ricava il passo e legge i numeri con l'OCR. Qui si guarda quello che ha trovato e si
   corregge — e la correzione deve costare un gesto, non un cambio di modo.

   Quindi: si trascina direttamente (colonna, zero, fondo, o tutto il righello insieme),
   shift+clic aggiunge una tacca, clic su una tacca la toglie, doppio clic su un numero lo
   riscrive lì dove sta. Trascinare una tacca non sposta un segno: dice il passo, e le altre
   tacche si rimettono in fila da sole — un righello ha le tacche equidistanti, e correggerne
   una per una a mano sarebbe rifare a mano un conto che si sa fare. Le due finestre ingrandite su zero e fondo servono perché a
   grandezza naturale una tacca è alta due pixel e il pixel giusto non si vede.

   Le correzioni rientrano nel modulo alla run successiva (`--corrections`), applicate *dopo*
   la detection: la pagina continua a mostrare anche cosa avrebbe detto da solo. */

const COLORI_SCALA = { accepted: '#3fb950', corrected: '#40d0ff', approved: '#7ee787',
  review: '#d29922',
                       reject: '#f85149' };

/* I gesti, scritti dove servono. Una riga di prosa lunga non la legge nessuno mentre corregge. */
function legenda() {
  const voci = [
    ['trascina', 'colonna, zero o fondo'],
    ['trascina una tacca', 'le altre si rimettono a quel passo'],
    ['alt + trascina', 'la tacca da sola · dalla colonna, tutto il righello'],
    ['doppio clic', 'aggiunge una tacca'],
    ['× sulla tacca', 'la toglie, col suo numero'],
    ['clic su «scrivi cm»', 'inserisci il valore accanto a qualunque tacca; vuoto lo toglie'],
    ['nelle finestre ingrandite', 'clic mette zero o fondo, shift+clic una tacca'],
    ['nella lente', 'scegli in alto cosa indichi, poi clicca'],
    ['shift+↑↓ · alt+↑↓', 'zero · fondo, un pixel per volta'],
    ['← →', 'cambia fotogramma'],
    ['cmd+Z', 'annulla'],
  ];
  return el('div', { class: 'scala-legenda' },
    ...voci.map(([gesto, cosa]) => el('span', { class: 'scala-legenda-voce' },
      el('kbd', {}, gesto), el('span', {}, cosa))));
}
const INGRANDIMENTO_SCALA = 4;

async function createScaleViewer(projectId) {
  let dati = await api(`/projects/${projectId}/scale/study`);
  const root = el('div', {});
  let frames = dati.frames || [];
  if (!frames.length) {
    root.append(el('p', { class: 'hint' }, 'nessun fotogramma studiato su questa cartella'));
    return root;
  }

  let indice = 0;
  let filtro = '';
  let suggerimenti = {};
  let attesaSecondoClic = null;   // il righello che si sta indicando a mano
  let righelloVisibile = true;
  let codaSalvataggi = Promise.resolve();
  let salvataggiInCorso = 0;
  const storia = [];             // per annullare l'ultimo gesto sul fotogramma
  const corrente = () => frames[indice] || {};
  const passa = (f) => !filtro || f.status === filtro;
  const visibili = () => frames.filter(passa);
  const correzioneDi = (nome) => (dati.corrections || {})[nome] || {};
  const correzione = () => correzioneDi(corrente().name);
  const proposta = () => suggerimenti[corrente().name];

  /* --- il fotogramma intero, con il righello sopra ------------------------------------ */
  const stage = el('div', { class: 'editor-stage scala-stage' });
  const image = el('img', { alt: '' });
  const strati = el('div', { class: 'scala-strati' });
  const editorTacche = el('div', { class: 'scala-editor-tacche' });
  let renderTacche = () => {};
  stage.append(image, strati);
  const scala = () => (image.clientWidth || 1) / (corrente().w || image.naturalWidth || 1);

  const disegna = () => {
    const f = corrente();
    strati.innerHTML = '';
    strati.style.display = righelloVisibile ? '' : 'none';
    renderTacche();
    if (!righelloVisibile) return;
    const s = scala();
    if (!s || !f.w) return;
    const alto = (f.h || 0) * s;

    const prop = proposta();
    if (prop && f.x == null) {
      const colonna = el('div', { class: 'scala-colonna proposta' });
      colonna.style.left = `${prop.x * s}px`;
      colonna.style.height = `${alto}px`;
      strati.append(colonna);
      for (const t of prop.ticks || []) {
        const tacca = el('div', { class: 'scala-tacca proposta' });
        tacca.style.top = `${t * s}px`;
        tacca.style.left = `${(prop.x - 20) * s}px`;
        tacca.style.width = `${40 * s}px`;
        strati.append(tacca);
      }
      for (const [chiave, classe, testo] of [['y_zero', 'zero', '0 proposto'],
                                             ['y_far', 'fondo', 'fondo proposto']]) {
        const riga = el('div', { class: `scala-estremo proposta scala-${classe}` },
          el('span', {}, testo));
        riga.style.top = `${prop[chiave] * s}px`;
        strati.append(riga);
      }
    }

    if (attesaSecondoClic) {
      const colonna = el('div', { class: 'scala-colonna disegno' });
      colonna.style.left = `${attesaSecondoClic.x * s}px`;
      colonna.style.height = `${alto}px`;
      const zero = el('div', { class: 'scala-estremo disegno scala-zero' }, el('span', {}, '0'));
      zero.style.top = `${attesaSecondoClic.y_zero * s}px`;
      strati.append(colonna, zero);
    }

    if (f.x != null) {
      const colonna = el('div', { class: 'scala-colonna', title: 'trascina: sposta la colonna' });
      colonna.style.left = `${f.x * s}px`;
      colonna.style.height = `${alto}px`;
      colonna.addEventListener('pointerdown', (e) => trascina(e, e.altKey ? 'tutto' : 'x', scala));
      strati.append(colonna);
    }
    for (const t of f.ticks || []) {
      const tacca = el('div', { class: 'scala-tacca',
        title: 'trascina: da qui il passo, e le altre tacche seguono (alt: solo questa)' },
        el('span', { class: 'scala-tacca-presa' }),
        el('button', { class: 'scala-tacca-via', title: 'togli questa tacca' }, '×'));
      tacca.style.top = `${t * s}px`;
      tacca.style.left = `${((f.x ?? 0) - 30) * s}px`;
      tacca.style.width = `${60 * s}px`;
      tacca.addEventListener('pointerenter', () => preparaZoomTacca(t));
      tacca.addEventListener('pointerdown', (e) => {
        if (e.target.closest('.scala-tacca-via')) return;
        e.stopPropagation();
        trascinaTacca(e, t);
      });
      tacca.querySelector('.scala-tacca-via').addEventListener('click', (e) => {
        e.stopPropagation(); togliTacca(t);
      });
      strati.append(tacca);
    }
    /* I numeri: quelli letti dall'OCR e quelli che mancano.

       Il riconoscimento dei numeri sbaglia spesso — sono due cifre alte otto pixel su uno
       sfondo che cambia — e finora si poteva solo correggere quello che l'OCR *aveva*
       letto: dove non aveva letto niente non c'era niente su cui fare doppio clic, e quello
       e' proprio il caso in cui serve. Adesso ogni tacca porta il suo numero, letto o
       calcolato dal passo, e si scrive su tutte allo stesso modo.

       Scriverne uno non e' correggere un'etichetta: e' calibrare. Da quel numero e dalla
       distanza dallo zero escono i millimetri per pixel, e tutti gli altri numeri seguono. */
    const lette = new Map((f.labels || []).map(([y, v]) => [Math.round(y), v]));
    const passoMm = (f.ruler || {}).step_mm || f.D_step_mm || null;
    const usate = new Set();
    for (const t of f.ticks || []) {
      const vicina = [...lette.keys()].find((y) => Math.abs(y - t) <= 6);
      const cm = vicina !== undefined ? lette.get(vicina) : null;
      if (vicina !== undefined) usate.add(vicina);
      const passi = passoMm && f.y_zero != null && (f.ruler || {}).pitch
        ? Math.round(Math.abs(t - f.y_zero) / f.ruler.pitch) : null;
      const calcolato = (cm == null && passoMm != null && passi != null)
        ? (passi * passoMm) / 10 : null;
      const mostrato = cm != null ? cm : calcolato;
      const numero = el('div', {
        class: 'scala-numero' + (mostrato == null ? ' vuoto' : (cm == null ? ' calcolato' : '')),
        title: mostrato == null
          ? 'clic per scrivere il valore di questa tacca'
          : cm == null
          ? 'calcolato dal passo: clic per scriverlo tu'
          : 'letto dall\'immagine: clic per correggerlo',
      }, mostrato == null ? 'scrivi cm' : `${Number(mostrato.toFixed(2))} cm`);
      numero.style.top = `${t * s}px`;
      numero.style.left = `${((f.x ?? 0) + 34) * s}px`;
      const apri = (e) => { e.stopPropagation(); cambiaNumero(numero, t, mostrato ?? ''); };
      numero.addEventListener('click', apri);
      numero.addEventListener('dblclick', apri);
      strati.append(numero);
    }
    // Un numero letto lontano da ogni tacca resta dov'e': e' un disaccordo fra OCR e tacche,
    // e nasconderlo vorrebbe dire nascondere proprio la cosa da guardare.
    for (const [y, valore] of f.labels || []) {
      if (usate.has(Math.round(y))) continue;
      const numero = el('div', { class: 'scala-numero fuori',
        title: 'letto qui, ma non su una tacca: clic per correggerlo' }, `${valore} cm`);
      numero.style.top = `${y * s}px`;
      numero.style.left = `${((f.x ?? 0) + 34) * s}px`;
      numero.addEventListener('click', (e) => { e.stopPropagation(); cambiaNumero(numero, y, valore); });
      strati.append(numero);
    }
    for (const [chiave, classe, testo] of [['y_zero', 'zero', '0'], ['y_far', 'fondo', 'fondo']]) {
      if (f[chiave] == null) continue;
      const riga = el('div', { class: `scala-estremo scala-${classe}` }, el('span', {}, testo));
      riga.style.top = `${f[chiave] * s}px`;
      riga.addEventListener('pointerdown', (e) => trascina(e, e.altKey ? 'tutto' : chiave, scala));
      strati.append(riga);
    }
  };

  /* --- le due finestre ingrandite: e' li' che si prende il pixel giusto --------------

     La finestra deve corrispondere **esattamente** all'overlay sull'immagine intera, o
     guardandola non si sa se la correzione e' giusta. Due cose la rompevano: il ritaglio
     arriva ingrandito 4x ma poi il CSS lo porta alla larghezza della scatola, quindi la
     scala vera non e' 4 ma `larghezza / finestra`; e la finestra va ritagliata sui bordi
     dell'immagine con la stessa regola del server — spostandola, non stringendola —
     altrimenti client e server parlano di due ritagli diversi. */
  /* Quanta immagine sta nella finestra: meno ne sta, piu' e' ingrandita. Quarantaquattro
     pixel di altezza vanno bene per lo zero, ma quando le tacche sono fitte servono meno,
     e quando il righello e' da ritrovare ne servono di piu'. */
  let VICINANZA = 1;
  const LARGA_ZOOM = () => Math.round(150 * VICINANZA);
  const ALTA_ZOOM = () => Math.round(44 * VICINANZA);
  const zoom = {};
  let modoZoom = 'estremo';
  let taccaInModifica = null;
  let taccaOrigine = null;
  let trascinandoTacca = false;
  const centroZoom = (chiave, f) => chiave === 'y_tacca' ? taccaInModifica : f[chiave];

  const finestraZoom = (f, y) => {
    const larghezza = Math.min(LARGA_ZOOM(), f.w || LARGA_ZOOM());
    const altezza = Math.min(ALTA_ZOOM(), f.h || ALTA_ZOOM());
    let x0 = Math.round(f.x - larghezza / 3);
    let y0 = Math.round(y - altezza / 2);
    x0 = Math.max(0, Math.min(x0, (f.w || larghezza) - larghezza));
    y0 = Math.max(0, Math.min(y0, (f.h || altezza) - altezza));
    return [x0, y0, x0 + larghezza, y0 + altezza];
  };
  const scalaZoom = (chiave) => {
    const z = zoom[chiave];
    if (!z || !z.finestra) return 1;
    return (z.img.clientWidth || 1) / (z.finestra[2] - z.finestra[0]);
  };

  /* Le due finestre ingrandite non sono un'anteprima: sono il posto dove si corregge.

     A grandezza naturale una tacca e' alta due pixel e lo zero e' un segno fra due segni:
     si trascinava a fatica sull'immagine intera e si sbagliava di tre pixel. Qui c'e' il
     quadruplo, quindi qui dentro si fa tutto — si indica lo zero e il fondo con un clic, si
     aggiunge e si toglie una tacca, si scrive e si cancella il numero. L'immagine intera
     resta per guardare il righello per intero, non per lavorarci di precisione. */
  const creaZoom = (chiave, etichetta, colore) => {
    const scatola = el('div', { class: `scala-zoom${chiave === 'y_tacca' ? ' tacca-live' : ''}` });
    const img = el('img', { alt: '' });
    const titolo = el('div', { class: 'scala-zoom-titolo' }, etichetta);
    const piano = el('div', { class: 'scala-zoom-piano' });
    scatola.append(titolo, img, piano);
    piano.addEventListener('pointerdown', (e) => {
      // Lo zoom della tacca e' soprattutto uno specchio in tempo reale. La tacca disegnata
      // dentro resta comunque afferrabile; il vuoto non sposta zero o fondo per errore.
      if (chiave === 'y_tacca' && !e.target.closest('.scala-zoom-tacca')) return;
      // Sopra una tacca o un numero comandano loro: qui si prende solo il vuoto.
      if (e.target.closest('.scala-zoom-tacca, .scala-zoom-numero')) return;
      const partenza = { x: e.clientX, y: e.clientY };
      let mosso = false;
      const muovi = (ev) => {
        if (Math.abs(ev.clientY - partenza.y) > 3 || Math.abs(ev.clientX - partenza.x) > 3) {
          mosso = true;
          piano.removeEventListener('pointermove', muovi);
          piano.removeEventListener('pointerup', molla);
          // Trascinare resta trascinare: la mira segue il dito come prima.
          trascina(e, chiave, () => ({ x: scalaZoom(chiave), y: scalaZoom(chiave) }));
        }
      };
      const molla = (ev) => {
        piano.removeEventListener('pointermove', muovi);
        piano.removeEventListener('pointerup', molla);
        if (mosso) return;
        // Un clic secco dice «sta qui»: e' il gesto piu' preciso che c'e', perche' si mira
        // invece di inseguire.
        const y = puntoZoom(chiave, ev);
        if (y == null) return;
        if (ev.shiftKey || modoZoom === 'aggiungi') { aggiungiTacca(y); return; }
        if (modoZoom === 'elimina') {
          const vicina = taccaVicina(y, Math.max(8, Math.round(10 * VICINANZA)));
          if (vicina == null) { stato('nessuna tacca vicino al punto indicato'); return; }
          togliTacca(vicina); return;
        }
        const f = corrente();
        ricorda();
        f[chiave] = y;
        if (chiave === 'y_zero') pulisciTaccheFuoriDalloZero(f);
        salva(chiave === 'y_zero' ? campiZero(f) : { [chiave]: y });
        stato(`${etichetta} a y=${y}`);
        disegna(); aggiornaZoom(); renderDati(); renderPasso();
      };
      piano.addEventListener('pointermove', muovi);
      piano.addEventListener('pointerup', molla);
    });
    zoom[chiave] = { scatola, img, piano, titolo, colore, etichetta, finestra: null };
    piano.addEventListener('pointermove', (e) => {
      const z = zoom[chiave];
      const y = puntoZoom(chiave, e);
      if (y == null || !z.cursore) return;
      z.cursore.style.display = '';
      z.cursore.style.top = `${(y - z.finestra[1]) * scalaZoom(chiave)}px`;
      if (chiave === 'y_tacca' && taccaInModifica != null) {
        const delta = taccaOrigine == null ? 0 : Math.round(taccaInModifica - taccaOrigine);
        z.dettaglio.textContent = `tacca y=${Math.round(taccaInModifica)} · Δ ${delta > 0 ? '+' : ''}${delta} px`
          + (trascinandoTacca ? ' · trascinamento in corso' : ' · pronta da trascinare');
      } else {
        z.dettaglio.textContent = `y=${y} · ${modoZoom === 'estremo'
          ? `posiziona ${z.etichetta}` : modoZoom === 'aggiungi' ? 'aggiungi tacca' : 'elimina tacca'}`;
      }
    });
    piano.addEventListener('pointerleave', () => {
      const z = zoom[chiave];
      if (z.cursore) z.cursore.style.display = 'none';
    });
    if (window.ResizeObserver) new ResizeObserver(() => aggiornaZoom()).observe(img);
    img.addEventListener('load', () => aggiornaZoom());
    return scatola;
  };
  /* La y nativa sotto al puntatore dentro a una delle due finestre. */
  const puntoZoom = (chiave, ev) => {
    const z = zoom[chiave];
    if (!z || !z.finestra) return null;
    const r = z.img.getBoundingClientRect();
    const s = scalaZoom(chiave);
    return Math.round(z.finestra[1] + (ev.clientY - r.top) / (s || 1));
  };
  const zoomZero = creaZoom('y_zero', 'zero', '#3fb950');
  const zoomFondo = creaZoom('y_far', 'fondo', '#d29922');
  const zoomTacca = creaZoom('y_tacca', 'tacca in modifica', '#ff4fd8');
  const preparaZoomTacca = (y) => {
    if (trascinandoTacca || y == null) return;
    const cambiata = taccaInModifica == null || Math.round(taccaInModifica) !== Math.round(y);
    taccaInModifica = y;
    taccaOrigine = y;
    if (cambiata && zoom.y_tacca) zoom.y_tacca.finestra = null;
    aggiornaZoom();
  };
  const vicinanzaSel = el('select', { style: 'width:108px' },
    ...[['0.25', 'dettaglio massimo'], ['0.5', 'molto vicino'], ['1', 'vicino'],
         ['2', 'largo'], ['4', 'tutta la barra']]
      .map(([v, etichetta]) => el('option',
        { value: v, ...(v === '1' ? { selected: '' } : {}) }, etichetta)));
  vicinanzaSel.addEventListener('change', () => {
    VICINANZA = parseFloat(vicinanzaSel.value) || 1;
    for (const chiave of ['y_zero', 'y_far', 'y_tacca']) zoom[chiave].finestra = null;
    aggiornaZoom();
  });
  const modiZoom = el('div', { class: 'row scala-zoom-modi' });
  const renderModiZoom = () => {
    modiZoom.innerHTML = '';
    for (const [id, testo] of [['estremo', 'Sposta zero/fondo'], ['aggiungi', '+ tacca'],
                               ['elimina', '− tacca']]) {
      const b = el('button', { class: 'chip' + (modoZoom === id ? ' on' : '') }, testo);
      b.addEventListener('click', () => { modoZoom = id; renderModiZoom(); });
      modiZoom.append(b);
    }
  };
  renderModiZoom();

  const aggiornaZoom = () => {
    const f = corrente();
    for (const chiave of ['y_zero', 'y_far', 'y_tacca']) {
      const z = zoom[chiave];
      const y = centroZoom(chiave, f);
      const taccaLive = chiave === 'y_tacca';
      z.scatola.style.display = (!taccaLive && (y == null || f.x == null)) ? 'none' : '';
      z.scatola.classList.toggle('vuota', taccaLive && (y == null || f.x == null));
      if (y == null || f.x == null) {
        if (taccaLive) {
          z.titolo.textContent = 'tacca: passaci sopra o afferrala';
          z.img.style.visibility = 'hidden';
          z.piano.innerHTML = '';
        }
        continue;
      }
      z.img.style.visibility = '';
      const nuova = finestraZoom(f, y);
      // Durante il trascinamento la lente della tacca resta ferma: si deve vedere la linea
      // muoversi rispetto ai pixel originali, non tenere la linea ferma spostando l'immagine.
      const cambiaFinestra = !z.finestra
        || (chiave !== 'y_tacca' && z.finestra.join() !== nuova.join());
      if (cambiaFinestra) {
        z.finestra = nuova;
        z.img.src = `/api/projects/${projectId}/depth/crop`
          + `?name=${encodeURIComponent(f.name)}&zoom=${INGRANDIMENTO_SCALA}&raw=1`
          + `&x0=${nuova[0]}&y0=${nuova[1]}&x1=${nuova[2]}&y1=${nuova[3]}`;
      }
      const s = scalaZoom(chiave);
      z.titolo.textContent = `${z.etichetta}: y=${Math.round(y)} · colonna x=${Math.round(f.x)}`;
      // Gli stessi disegni dell'immagine intera, alla scala vera della finestra: e' il
      // confronto fra i due che dice se la correzione e' a posto.
      z.piano.innerHTML = '';
      z.piano.style.display = righelloVisibile ? '' : 'none';
      const colonna = el('div', { class: 'scala-zoom-colonna' });
      colonna.style.left = `${(f.x - z.finestra[0]) * s}px`;
      z.piano.append(colonna);
      // Le tacche dentro alla finestra, con tutto quello che ci si fa sopra: trascinare
      // (e le altre seguono il passo), togliere, scrivere il numero. E' il posto giusto per
      // farlo: qui una tacca e' alta otto pixel invece di due.
      const lette = new Map((f.labels || []).map(([yy, v]) => [Math.round(yy), v]));
      const passoMm = (f.ruler || {}).step_mm || f.D_step_mm || null;
      for (const t of f.ticks || []) {
        if (t < z.finestra[1] || t > z.finestra[3]) continue;
        const attiva = taccaInModifica != null
          && Math.round(t) === Math.round(taccaInModifica);
        const tacca = el('div', { class: `scala-zoom-tacca${attiva ? ' attiva' : ''}`,
          title: 'trascina: da qui il passo · alt+trascina: solo questa' },
          el('span', { class: 'scala-zoom-y' }, `y ${Math.round(t)}`),
          el('button', { class: 'scala-zoom-via', title: 'togli la tacca e il suo numero' }, '×'));
        tacca.style.top = `${(t - z.finestra[1]) * s}px`;
        tacca.addEventListener('pointerenter', () => preparaZoomTacca(t));
        tacca.addEventListener('pointerdown', (e) => {
          if (e.target.closest('.scala-zoom-via')) return;
          if (modoZoom === 'elimina') {
            e.preventDefault(); e.stopPropagation(); togliTacca(t); return;
          }
          e.stopPropagation();
          trascinaTacca(e, t, s);
        });
        tacca.querySelector('.scala-zoom-via').addEventListener('click', (e) => {
          e.stopPropagation(); togliTacca(t);
        });
        z.piano.append(tacca);

        const vicina = [...lette.keys()].find((yy) => Math.abs(yy - t) <= 6);
        const cm = vicina !== undefined ? lette.get(vicina) : null;
        const passi = passoMm && f.y_zero != null && (f.ruler || {}).pitch
          ? Math.round(Math.abs(t - f.y_zero) / f.ruler.pitch) : null;
        const calcolato = (cm == null && passoMm != null && passi != null)
          ? (passi * passoMm) / 10 : null;
        const mostrato = cm != null ? cm : calcolato;
        const numero = el('div', {
          class: 'scala-zoom-numero' + (mostrato == null ? ' vuoto' : (cm == null ? ' calcolato' : '')),
          title: mostrato == null ? 'clic per scrivere il valore di questa tacca'
                 : cm == null ? 'calcolato dal passo: clic per scriverlo tu'
                              : 'letto dall\'immagine: clic per correggerlo, vuoto per toglierlo',
        }, mostrato == null ? 'scrivi cm' : `${Number(mostrato.toFixed(2))} cm`);
        numero.style.top = `${(t - z.finestra[1]) * s}px`;
        numero.addEventListener('pointerdown', (e) => e.stopPropagation());
        numero.addEventListener('click', (e) => {
          e.stopPropagation(); cambiaNumero(numero, t, mostrato ?? '');
        });
        z.piano.append(numero);
      }
      const mira = el('div', { class: 'scala-mira', style: `border-top-color:${z.colore}` });
      mira.style.top = `${(y - z.finestra[1]) * s}px`;
      const cursore = el('div', { class: 'scala-zoom-cursore', style: 'display:none' });
      const dettaglio = el('div', { class: 'scala-zoom-dettaglio' });
      if (chiave === 'y_tacca' && taccaInModifica != null) {
        const delta = taccaOrigine == null ? 0 : Math.round(taccaInModifica - taccaOrigine);
        dettaglio.textContent = `tacca y=${Math.round(taccaInModifica)} · Δ ${delta > 0 ? '+' : ''}${delta} px`
          + (trascinandoTacca ? ' · trascinamento in corso' : ' · pronta da trascinare');
      }
      z.cursore = cursore; z.dettaglio = dettaglio;
      z.piano.append(mira, cursore, dettaglio);
    }
  };

  /* --- trascinare: un lato, o tutto il righello insieme ------------------------------- */
  const dallaParteDelFondo = (y, zero, fondo) => {
    if (zero == null) return true;
    if (fondo == null) return y >= zero;
    return fondo >= zero ? y >= zero && y <= fondo : y <= zero && y >= fondo;
  };
  const pulisciTaccheFuoriDalloZero = (f, tacche, etichette) => {
    f.ticks = (tacche || f.ticks || [])
      .filter((y) => dallaParteDelFondo(y, f.y_zero, f.y_far));
    f.labels = (etichette || f.labels || [])
      .filter(([y]) => dallaParteDelFondo(y, f.y_zero, f.y_far));
  };
  const campiZero = (f) => ({
    y_zero: f.y_zero,
    ticks: [...(f.ticks || [])],
    nums: (correzione().nums || [])
      .filter(([y]) => dallaParteDelFondo(y, f.y_zero, f.y_far)),
  });

  function trascina(event, quale, scalaFn) {
    event.preventDefault();
    event.stopPropagation();
    const f = corrente();
    const s = scalaFn();
    const fx = typeof s === 'object' ? s.x : s;
    const fy = typeof s === 'object' ? s.y : s;
    const da = { cx: event.clientX, cy: event.clientY,
                 x: f.x, y_zero: f.y_zero, y_far: f.y_far,
                 ticks: [...(f.ticks || [])], labels: (f.labels || []).map((v) => [...v]) };
    ricorda();
    const muovi = (e) => {
      const dx = Math.round((e.clientX - da.cx) / (fx || 1));
      const dy = Math.round((e.clientY - da.cy) / (fy || 1));
      if (quale === 'tutto') {
        f.x = limita(da.x + dx, f.w);
        f.y_zero = limita(da.y_zero + dy, f.h);
        f.y_far = limita(da.y_far + dy, f.h);
        f.ticks = da.ticks.map((t) => limita(t + dy, f.h));
      } else if (quale === 'x') {
        f.x = limita(da.x + dx, f.w);
      } else {
        f[quale] = limita(da[quale] + dy, f.h);
        if (quale === 'y_zero') pulisciTaccheFuoriDalloZero(f, da.ticks, da.labels);
      }
      disegna(); aggiornaZoom(); renderDati();
    };
    const molla = () => {
      window.removeEventListener('pointermove', muovi);
      window.removeEventListener('pointerup', molla);
      const campi = quale === 'tutto'
        ? { x: f.x, y_zero: f.y_zero, y_far: f.y_far, ticks: [...f.ticks] }
        : (quale === 'y_zero' ? campiZero(f) : { [quale]: f[quale] });
      salva(campi);
    };
    window.addEventListener('pointermove', muovi);
    window.addEventListener('pointerup', molla);
  }
  const limita = (v, massimo) => Math.max(0, Math.min(Math.round(v), Math.round(massimo || 0)));

  /* --- gesti sull'immagine: shift+clic aggiunge, due clic indicano il righello -------- */
  stage.addEventListener('click', (event) => {
    if (event.target.closest('.scala-tacca, .scala-numero')) return;
    const s = scala();
    const riquadro = stage.getBoundingClientRect();
    const x = Math.round((event.clientX - riquadro.left) / (s || 1));
    const y = Math.round((event.clientY - riquadro.top) / (s || 1));
    const f = corrente();

    if (f.x == null) {
      // Righello mai trovato: due clic lo indicano. Primo lo zero e la colonna, poi il fondo.
      if (!attesaSecondoClic) {
        attesaSecondoClic = { x, y_zero: y };
        stato('ora clicca il fondo del righello');
      } else {
        indicaRighello(attesaSecondoClic, y);
        attesaSecondoClic = null;
      }
      disegna();
      Lente.aggiornaSeAttiva(contestoLente());
      return;
    }
    if (event.shiftKey) generaDaTacca(y);
  });
  stage.addEventListener('dblclick', (event) => {
    // Doppio clic su una tacca: da li' viene fuori tutta la scala. Con alt si aggiunge la
    // singola tacca e basta, che serve quando la scala non e' regolare - ma e' il caso raro.
    if (event.target.closest('.scala-tacca, .scala-numero') || corrente().x == null) return;
    const s = scala();
    const y = Math.round((event.clientY - stage.getBoundingClientRect().top) / (s || 1));
    if (event.altKey) aggiungiTacca(y); else generaDaTacca(y);
  });
  const aggiungiTacca = (y) => {
    const f = corrente();
    ricorda();
    f.ticks = [...(f.ticks || []), y].sort((a, b) => a - b);
    taccaInModifica = y;
    if (zoom.y_tacca) zoom.y_tacca.finestra = null;
    salva({ ticks: [...f.ticks] });
    disegna(); aggiornaZoom();
    Lente.aggiornaSeAttiva(contestoLente());
  };

  /* Una tacca sola, e le altre vengono da se'.

     Dentro a un fotogramma il passo non cambia mai: e' l'unica promessa che un righello fa
     sempre. Quindi segnarne una vuol dire averle segnate tutte, e chiederle una per una
     sarebbe far ripetere a mano un conto che si sa fare. Il conto lo fa il server, che ha
     anche la depth confermata con cui verificarlo. */
  const generaDaTacca = async (y) => {
    const f = corrente();
    if (f.y_zero == null) { stato('prima lo zero'); return; }
    ricorda();
    try {
      const esito = await api(`/projects/${projectId}/scale/study/ticks`,
        { body: { name: f.name, tick: y } });
      f.ticks = esito.ticks;
      f.pitch = esito.pitch;
      f.y_far = esito.y_far;
      f.depth_check = esito.check;
      dati.corrections = dati.corrections || {};
      dati.corrections[f.name] = { ...(dati.corrections[f.name] || {}),
        ticks: esito.ticks, pitch: esito.pitch, y_far: esito.y_far };
      stato(`${esito.ticks.length} tacche a passo ${esito.pitch} px — ${esito.from}`);
      disegna(); aggiornaZoom(); renderPasso(); renderDati(); renderLista();
      Lente.aggiornaSeAttiva(contestoLente());
    } catch (errore) { stato(errore.message); toast(errore.message, true); }
  };

  const indicaRighello = (inizio, yFondo) => {
    const f = corrente();
    ricorda();
    const passoMm = (proposta() || {}).step_mm || 10;
    const depth = (proposta() || {}).depth_confirmed_mm || f.E_depth_interface;
    f.x = inizio.x; f.y_zero = inizio.y_zero; f.y_far = yFondo;
    f.B_zero_end = yFondo < inizio.y_zero ? 'bottom' : 'top';
    const tacche = [];
    if (depth) {
      const passoPx = (passoMm / depth) * Math.abs(yFondo - inizio.y_zero);
      const direzione = yFondo > inizio.y_zero ? 1 : -1;
      for (let k = 0; k * passoPx <= Math.abs(yFondo - inizio.y_zero) + 0.5; k += 1) {
        tacche.push(Math.round(inizio.y_zero + direzione * k * passoPx));
      }
    }
    f.ticks = tacche;
    salva({ x: f.x, y_zero: f.y_zero, y_far: f.y_far,
            zero_end: f.B_zero_end, ticks: tacche });
    stato('righello indicato');
  };

  /* Il passo di adesso: quello salvato, o quello che dicono le tacche che ci sono. */
  const passoDiAdesso = (f) => {
    if (f.pitch && f.pitch > 2) return f.pitch;
    const t = (f.ticks || []).slice().sort((a, b) => a - b);
    if (t.length < 2) return null;
    return (t[t.length - 1] - t[0]) / (t.length - 1);
  };
  /* Quanti passi c'e' fra lo zero e questa tacca: e' il numero per cui dividere.

     Si arrotonda col passo di adesso, che e' quello sbagliato che si sta correggendo - ma
     per sbagliare il conto dovrebbe essere sbagliato di piu' di mezzo passo, e in quel caso
     il righello non lo si aggiusta trascinando, lo si rifa'. Il numero viene scritto mentre
     si trascina, cosi' se e' quello sbagliato si vede. */
  const passiDalloZero = (f, y) => {
    const passo = passoDiAdesso(f);
    const distanza = Math.abs(y - f.y_zero);
    if (!passo || distanza <= 0) return 0;
    return Math.max(0, Math.round(distanza / passo));
  };
  /* Le tacche a passo costante dallo zero al fondo: la stessa griglia che fa il server
     (`_griglia_tacche`), rifatta qui solo per far vedere il righello mentre si trascina. */
  const grigliaLocale = (f, passo) => {
    if (!passo || passo <= 2) return f.ticks || [];
    const fondo = f.y_far != null ? f.y_far : (f.ticks || []).slice(-1)[0];
    if (fondo == null) return f.ticks || [];
    const verso = fondo >= f.y_zero ? 1 : -1;
    const lunghezza = Math.abs(fondo - f.y_zero);
    const quante = Math.round(lunghezza / passo);
    if (quante < 1 || quante > 200) return f.ticks || [];
    const nuove = [];
    for (let k = 0; k <= quante; k += 1) nuove.push(Math.round(f.y_zero + verso * k * passo));
    return nuove;
  };

  /* Trascinare una tacca vuol dire dire il passo, non spostare un segno.

     Le tacche di un righello sono equidistanti: se la terza dallo zero va due pixel piu'
     giu', ci vanno anche la quarta, la quinta e tutte le altre - di quattro, di sei, di
     dieci. Muoverne una sola lasciava il righello storto e costava un trascinamento per
     tacca. Adesso quella che si tiene in mano definisce il passo, e le altre si rimettono
     in fila da se': piu' lontana e' la tacca, piu' fine e' la regolazione, perche' il passo
     e' la distanza divisa per il numero di passi.

     Con alt si muove solo quella: serve quando il righello *non* e' regolare, che e' il
     caso raro ma esiste. */
  function trascinaTacca(event, y, scalaPropria) {
    event.preventDefault();
    const f = corrente();
    // La scala e' quella di dove si sta trascinando: sull'immagine intera un pixel di
    // schermo e' un pixel e mezzo di immagine, dentro alla finestra ingrandita e' un quarto.
    const s = scalaPropria || scala();
    const partenza = event.clientY;
    const originale = y;
    const passi = event.altKey || f.y_zero == null ? 0 : passiDalloZero(f, y);
    const insieme = passi >= 1;
    const primaTicks = [...(f.ticks || [])];
    const primaPitch = f.pitch;
    let attuale = y;
    let mossa = false;
    const nuovaTacca = taccaInModifica == null || Math.round(taccaInModifica) !== Math.round(y);
    taccaInModifica = y;
    taccaOrigine = y;
    trascinandoTacca = true;
    if (nuovaTacca && zoom.y_tacca) zoom.y_tacca.finestra = null;
    aggiornaZoom();
    ricorda();
    const muovi = (e) => {
      const dy = (e.clientY - partenza) / (s || 1);
      if (Math.abs(dy) < 0.5 && !mossa) return;
      mossa = true;
      attuale = limita(originale + dy, f.h);
      taccaInModifica = attuale;
      if (insieme) {
        const passo = Math.abs(attuale - f.y_zero) / passi;
        f.pitch = Math.round(passo * 10) / 10;
        f.ticks = grigliaLocale(f, passo);
        stato(`${passi}ª tacca dallo zero · passo ${f.pitch} px · ${f.ticks.length} tacche`);
      } else {
        f.ticks = (f.ticks || [])
          .map((t) => (Math.round(t) === Math.round(originale) ? attuale : t))
          .sort((a, b) => a - b);
      }
      disegna(); aggiornaZoom(); renderDati();
    };
    const molla = async () => {
      window.removeEventListener('pointermove', muovi);
      window.removeEventListener('pointerup', molla);
      trascinandoTacca = false;
      aggiornaZoom();
      if (!mossa) { storia.pop(); return; }
      if (!insieme) {
        // Spostare una tacca sola, per il modulo, e' toglierla da dov'era e rimetterla qui.
        salva({ ticks: [...f.ticks] });
        return;
      }
      // Il passo lo ricalcola il server, con la stessa regola: cosi' quello salvato e'
      // uno solo, e torna indietro anche il controllo con la depth confermata.
      try {
        const esito = await api(`/projects/${projectId}/scale/study/ticks`,
          { body: { name: f.name, tick: Math.round(attuale) } });
        f.ticks = esito.ticks;
        f.pitch = esito.pitch;
        f.y_far = esito.y_far;
        f.depth_check = esito.check;
        dati.corrections = dati.corrections || {};
        dati.corrections[f.name] = { ...(dati.corrections[f.name] || {}),
          ticks: esito.ticks, pitch: esito.pitch, y_far: esito.y_far };
        stato(`${esito.ticks.length} tacche a passo ${esito.pitch} px — ${esito.from}`);
        disegna(); aggiornaZoom(); renderPasso(); renderDati(); renderLista();
      } catch (errore) {
        // Il righello torna com'era: lasciarlo con le tacche nuove e la correzione vecchia
        // farebbe credere salvato quello che non lo e'.
        f.ticks = primaTicks; f.pitch = primaPitch;
        disegna(); aggiornaZoom(); renderDati();
        stato(errore.message); toast(errore.message, true);
      }
    };
    window.addEventListener('pointermove', muovi);
    window.addEventListener('pointerup', molla);
  }

  /* Le tacche di un righello sono equidistanti: quando ce ne sono di storte, rifarle tutte
     dal passo costa un comando invece di dieci trascinamenti. */
  const ridistribuisci = () => {
    const f = corrente();
    if (f.y_zero == null || f.y_far == null) { stato('servono zero e fondo'); return; }
    const passoMm = (f.D_step_mm || (proposta() || {}).step_mm || 10);
    const quante = (f.ticks || []).length;
    const lunghezza = Math.abs(f.y_far - f.y_zero);
    const passoPx = quante > 1 ? lunghezza / (quante - 1) : (f.pitch || lunghezza);
    if (!passoPx) { stato('non so con che passo'); return; }
    ricorda();
    const direzione = f.y_far > f.y_zero ? 1 : -1;
    const nuove = [];
    for (let k = 0; k * passoPx <= lunghezza + 0.5; k += 1) {
      nuove.push(Math.round(f.y_zero + direzione * k * passoPx));
    }
    f.ticks = nuove;
    salva({ ticks: nuove });
    stato(`${nuove.length} tacche a passo costante (${passoMm} mm)`);
    disegna(); aggiornaZoom();
  };

  /* Togliere una tacca porta via anche il suo numero.

     Il numero non e' un'etichetta appesa all'immagine: e' il valore *di quella tacca*.
     Toglievo la tacca e il numero restava li' a mezz'aria, a raccontare una scala che non
     c'era piu' - e siccome dai numeri esce la calibrazione, quel resto continuava anche a
     contare. */
  const togliTacca = (y) => {
    const f = corrente();
    ricorda();
    const vicino = (yy) => Math.abs(yy - y) <= 6;
    f.ticks = (f.ticks || []).filter((t) => Math.round(t) !== Math.round(y));
    if (taccaInModifica != null && Math.round(taccaInModifica) === Math.round(y)) {
      taccaInModifica = null;
      if (zoom.y_tacca) zoom.y_tacca.finestra = null;
    }
    const aveva = (f.labels || []).some(([yy]) => vicino(yy));
    f.labels = (f.labels || []).filter(([yy]) => !vicino(yy));
    const campi = { ticks: [...f.ticks] };
    if (aveva || (correzione().nums || []).some(([yy]) => vicino(yy))) {
      campi.nums = (correzione().nums || []).filter(([yy]) => !vicino(yy));
      // La lista vuota e' intenzionale: significa che i numeri sono stati tolti tutti.
    }
    salva(campi);
    disegna(); aggiornaZoom(); renderDati();
    Lente.aggiornaSeAttiva(contestoLente());
    stato(aveva ? 'tacca e numero tolti' : 'tacca tolta');
  };
  const taccaVicina = (y, quanto = 12) => {
    const f = corrente();
    let migliore = null;
    for (const t of f.ticks || []) {
      if (Math.abs(t - y) <= quanto && (migliore === null || Math.abs(t - y) < Math.abs(migliore - y))) {
        migliore = t;
      }
    }
    return migliore;
  };

  /* --- i numeri si riscrivono dove stanno, senza finestrelle di sistema -------------- */
  const salvaValore = (f, y, scritto) => {
    const testo = String(scritto ?? '').trim();
    const vicino = (yy) => Math.abs(yy - y) <= 6;
    if (!testo) {
      f.labels = (f.labels || []).filter(([yy]) => !vicino(yy));
      salva({ nums: (correzioneDi(f.name).nums || []).filter(([yy]) => !vicino(yy)) }, f.name);
      stato('valore della tacca tolto');
      return true;
    }
    const cm = parseFloat(testo.replace(',', '.'));
    if (!isFinite(cm)) { stato('scrivi un numero valido'); return false; }
    const c_era = (f.labels || []).some(([yy]) => Math.round(yy) === Math.round(y));
    f.labels = c_era
      ? (f.labels || []).map(([yy, vv]) => (Math.round(yy) === Math.round(y) ? [yy, cm] : [yy, vv]))
      : [...(f.labels || []), [y, cm]].sort((a, b) => a[0] - b[0]);
    salva({ nums: [...(correzioneDi(f.name).nums || [])
      .filter(([yy]) => Math.round(yy) !== Math.round(y)), [Math.round(y), cm * 10]] }, f.name);
    stato(`valore della tacca: ${cm} cm`);
    return true;
  };

  const cambiaNumero = (nodo, y, valore) => {
    const f = corrente();
    const campo = el('input', { class: 'scala-numero-campo', type: 'text',
      inputmode: 'decimal', placeholder: 'cm', value: String(valore ?? '') });
    nodo.replaceChildren(campo);
    campo.focus(); campo.select();
    let chiuso = false;
    const chiudi = (salvare) => {
      if (chiuso) return;
      chiuso = true;
      const scritto = campo.value.trim();
      if (salvare) {
        ricorda();
        salvaValore(f, y, scritto);
      }
      disegna(); aggiornaZoom(); renderDati(); renderPasso();
    };
    campo.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); chiudi(true); }
      if (e.key === 'Escape') { e.preventDefault(); chiudi(false); }
      // Canc su un campo gia' vuoto: togli il numero e passa oltre, senza dover confermare
      // il vuoto con Invio.
      if ((e.key === 'Delete' || e.key === 'Backspace') && !campo.value.trim()) {
        e.preventDefault(); chiudi(true);
      }
    });
    campo.addEventListener('blur', () => chiudi(true));
  };

  /* Una tabella semplice resta accanto all'immagine: non serve centrare un'etichetta alta
     pochi pixel per correggere un valore o trovare la croce della tacca da eliminare. */
  renderTacche = () => {
    const f = corrente();
    editorTacche.innerHTML = '';
    editorTacche.append(el('div', { class: 'scala-editor-titolo' },
      `Tacche e valori (${(f.ticks || []).length})`));
    if (!(f.ticks || []).length) {
      editorTacche.append(el('div', { class: 'hint' }, 'nessuna tacca'));
      return;
    }
    const lette = new Map((f.labels || []).map(([y, v]) => [Math.round(y), v]));
    const passoMm = (f.ruler || {}).step_mm || f.D_step_mm || null;
    for (const [posizione, t] of [...(f.ticks || [])].sort((a, b) => a - b).entries()) {
      const vicina = [...lette.keys()].find((y) => Math.abs(y - t) <= 6);
      const letto = vicina === undefined ? null : lette.get(vicina);
      const passi = passoMm && f.y_zero != null && (f.ruler || {}).pitch
        ? Math.round(Math.abs(t - f.y_zero) / f.ruler.pitch) : null;
      const calcolato = letto == null && passoMm != null && passi != null
        ? Number(((passi * passoMm) / 10).toFixed(2)) : null;
      const input = el('input', { class: 'scala-valore-input', type: 'text', inputmode: 'decimal',
        value: letto == null ? '' : String(letto),
        placeholder: calcolato == null ? 'cm' : String(calcolato),
        'aria-label': `valore della tacca ${posizione + 1} in centimetri` });
      const iniziale = input.value;
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
        if (e.key === 'Escape') { e.preventDefault(); input.value = iniziale; input.blur(); }
      });
      input.addEventListener('blur', () => {
        if (input.value.trim() === iniziale.trim()) return;
        ricorda();
        if (salvaValore(f, t, input.value)) {
          disegna(); aggiornaZoom(); renderDati(); renderPasso();
        }
      });
      const via = el('button', { class: 'scala-editor-via', title: 'elimina questa tacca' }, '×');
      via.addEventListener('click', () => togliTacca(t));
      editorTacche.append(el('div', { class: 'scala-editor-riga' },
        el('span', { class: 'scala-editor-y' }, `${posizione + 1} · y ${Math.round(t)}`),
        input, el('span', { class: 'hint' }, 'cm'), via));
    }
  };

  /* --- annullare l'ultimo gesto ------------------------------------------------------- */
  const ricorda = () => {
    const f = corrente();
    storia.push({ name: f.name, x: f.x, y_zero: f.y_zero, y_far: f.y_far,
                  ticks: [...(f.ticks || [])], labels: (f.labels || []).map((l) => [...l]),
                  zero_end: f.B_zero_end, correzione: { ...correzione() } });
    if (storia.length > 40) storia.shift();
  };
  const annulla = async () => {
    const prima = storia.pop();
    if (!prima) { stato('niente da annullare'); return; }
    const f = frames.find((x) => x.name === prima.name);
    if (!f) return;
    Object.assign(f, { x: prima.x, y_zero: prima.y_zero, y_far: prima.y_far,
                       ticks: prima.ticks, labels: prima.labels, B_zero_end: prima.zero_end });
    indice = frames.indexOf(f);
    try {
      await api(`/projects/${projectId}/scale/study/correct`,
        { body: { name: f.name, reset: true } });
      const rimasto = Object.fromEntries(Object.entries(prima.correzione)
        .filter(([k]) => k !== 'ts'));
      if (Object.keys(rimasto).length) {
        await api(`/projects/${projectId}/scale/study/correct`,
          { body: { name: f.name, ...rimasto } });
        dati.corrections[f.name] = prima.correzione;
      } else {
        delete dati.corrections[f.name];
      }
      await caricaSuggerimenti();
      stato('annullato');
      mostra();
    } catch (errore) { toast(errore.message, true); }
  };

  /* --- salvataggio, con un segnale che si vede -------------------------------------- */
  const statoBox = el('span', { class: 'hint scala-stato' });
  let orologio = null;
  const stato = (testo, persistente = false) => {
    statoBox.textContent = testo;
    clearTimeout(orologio);
    if (!persistente) orologio = setTimeout(() => { statoBox.textContent = ''; }, 2200);
  };
  const salva = (campi, nomeForzato = null) => {
    // Il nome va catturato prima della richiesta: se nel frattempo si passa al fotogramma
    // successivo, la risposta non deve finire nelle correzioni di quello nuovo. Inoltre le
    // richieste partono in fila: due trascinamenti rapidi non possono piu' arrivare al
    // server nell'ordine inverso e cancellarsi a vicenda.
    const nome = nomeForzato || corrente().name;
    dati.corrections = dati.corrections || {};
    const voce = { ...(dati.corrections[nome] || {}) };
    for (const [chiave, valore] of Object.entries(campi)) {
      if (valore === null || valore === '') delete voce[chiave];
      else voce[chiave] = valore;
    }
    voce.ts = new Date().toISOString();
    dati.corrections[nome] = voce;
    dati.pending = [...new Set([...(dati.pending || []), nome])];
    salvataggiInCorso += 1;
    stato(`salvataggio in corso${salvataggiInCorso > 1 ? ` (${salvataggiInCorso})` : ''}…`);
    renderGiro(); renderDati(); renderLista(); renderAzioni();

    const invia = async () => {
      try {
        return await api(`/projects/${projectId}/scale/study/correct`,
          { body: { name: nome, ...campi } });
      } catch (prima) {
        // Una sola ripetizione copre una perdita momentanea della richiesta senza poter
        // duplicare effetti: ogni payload descrive lo stato esatto, non un incremento.
        await new Promise((resolve) => setTimeout(resolve, 350));
        return api(`/projects/${projectId}/scale/study/correct`,
          { body: { name: nome, ...campi } });
      }
    };
    const operazione = codaSalvataggi.then(invia);
    codaSalvataggi = operazione.catch(() => {});
    return operazione.then(async () => {
      salvataggiInCorso -= 1;
      if (!salvataggiInCorso) stato('salvato — non ancora riletto dal modulo');
      await caricaSuggerimenti();
      renderGiro(); renderDati(); renderLista(); renderAzioni();
    }).catch((errore) => {
      salvataggiInCorso -= 1;
      stato(`NON salvato: ${errore.message}`, true);
      toast(`correzione del righello non salvata: ${errore.message}`, true);
    });
  };

  const caricaSuggerimenti = async () => {
    try {
      const r = await api(`/projects/${projectId}/scale/study/suggestions`);
      suggerimenti = r.suggestions || {};
      const quanti = Object.keys(suggerimenti).length;
      notiBox.innerHTML = '';
      if (!quanti) return;
      notiBox.append(el('span', { class: 'hint' },
        `righello proposto su ${quanti} fotogrammi, dai ${r.known} in cui si sa dov'e'`
        + `${r.depth_confirmed ? ` · ${r.depth_confirmed} depth confermate da te` : ''}`));
      const tutti = el('button', { class: 'ghost' }, `Accetta tutti i ${quanti} proposti`);
      tutti.addEventListener('click', async () => {
        try {
          const esito = await api(`/projects/${projectId}/scale/study/accept`, { body: {} });
          toast(`accettati ${esito.accepted} righelli proposti`);
          dati = await api(`/projects/${projectId}/scale/study`);
          frames = dati.frames || [];
          await caricaSuggerimenti();
          mostra();
        } catch (errore) { toast(errore.message, true); }
      });
      notiBox.append(tutti);
    } catch (errore) { suggerimenti = {}; notiBox.innerHTML = ''; }
  };
  const notiBox = el('div', { class: 'row' });

  /* --- la fascia che dice a che punto e' il giro ------------------------------------- */
  const giroBox = el('div', { class: 'scala-giro' });
  const renderGiro = () => {
    giroBox.innerHTML = '';
    const inAttesa = dati.pending || [];
    const ultima = dati.last_run || {};
    if (inAttesa.length) {
      giroBox.className = 'scala-giro in-attesa';
      giroBox.append(
        el('strong', {}, `${inAttesa.length} correzion${inAttesa.length === 1 ? 'e' : 'i'} `
          + 'ancora da usare'),
        el('span', { class: 'hint' },
          ' — sono salvate, ma il modulo non le ha ancora rilette. Rifai lo studio per '
          + 'applicarle anche ai fotogrammi ancora da sistemare.'),
      );
      const nomi = inAttesa.slice(0, 6).map((n) => n.split('/').pop()).join(', ');
      giroBox.append(el('div', { class: 'hint' },
        nomi + (inAttesa.length > 6 ? ` … e altre ${inAttesa.length - 6}` : '')));
      return;
    }
    if (!ultima.at) { giroBox.className = 'scala-giro vuota'; return; }
    giroBox.className = 'scala-giro fatto';
    const migliorati = ultima.improved || [];
    const peggiorati = ultima.worsened || [];
    giroBox.append(el('strong', {},
      ultima.used && ultima.used.length
        ? `ultimo giro: il modulo ha riletto ${ultima.used.length} correzion${ultima.used.length === 1 ? 'e' : 'i'}`
        : 'ultimo giro: nessuna correzione da rileggere'),
      el('span', { class: 'hint' },
        ` · ${ultima.accepted_total}/${ultima.frames} fotogrammi con righello`));
    if (migliorati.length) {
      giroBox.append(el('div', { class: 'scala-migliorati' },
        `migliorati ${migliorati.length}: `
        + migliorati.slice(0, 6).map((m) => `${m.name.split('/').pop()} ${m.from}→${m.to}`).join(' · ')
        + (migliorati.length > 6 ? ` … e altri ${migliorati.length - 6}` : '')));
    }
    if (peggiorati.length) {
      giroBox.append(el('div', { class: 'scala-peggiorati' },
        `peggiorati ${peggiorati.length}: `
        + peggiorati.map((m) => `${m.name.split('/').pop()} ${m.from}→${m.to}`).join(' · ')));
    }
    if (!migliorati.length && !peggiorati.length) {
      giroBox.append(el('div', { class: 'hint' }, 'nessun fotogramma ha cambiato stato'));
    }
  };

  /* --- i dati: cosa ha trovato, e cosa hai corretto tu ------------------------------- */
  const datiBox = el('div', { class: 'hint' });
  const renderDati = () => {
    const f = corrente();
    const c = correzione();
    const corretti = Object.keys(c).filter((k) => !['ts', 'from_suggestion'].includes(k));
    datiBox.innerHTML = '';
    datiBox.append(
      el('div', {},
        `stato ${f.status}`
        + (f.status === 'corrected' ? ` (il modulo da solo: ${f.detector_status})` : '')
        + (f.reason ? ` — ${f.reason}` : '')),
      el('div', {},
        `colonna x=${f.x != null ? Math.round(f.x) : '—'} · zero y=${f.y_zero != null ? Math.round(f.y_zero) : '—'}`
        + ` · fondo y=${f.y_far != null ? Math.round(f.y_far) : '—'} · ${(f.ticks || []).length} tacche`
        + ` · passo ${f.pitch ? Math.round(f.pitch) + ' px' : '—'}`
        + ` · ${f.mm_per_px ? f.mm_per_px.toFixed(4) + ' mm/px' : '—'}`),
      el('div', {},
        `numeri: ${(f.labels || []).map(([, v]) => `${v}`).join(' · ') || 'nessuno'}`
        + `${f.D_step_mm ? ` · passo ${f.D_step_mm} mm` : ''}`
        + `${f.D_coherent === false ? ' · numeri incoerenti fra loro' : ''}`),
      el('div', {},
        `zero ${f.B_zero_end === 'top' ? 'in alto' : f.B_zero_end === 'bottom' ? 'in basso' : '—'}`
        + `${f.E_depth_interface ? ` · depth letta ${f.E_depth_interface} mm` : ''}`
        + `${(proposta() || {}).depth_confirmed_mm ? ` · depth confermata ${proposta().depth_confirmed_mm} mm` : ''}`
        + `${f.E_verdict ? ` · ${f.E_verdict}` : ''}`),
    );
    if (corretti.length) {
      const usate = f.corr_applied || [];
      const inAttesa = (dati.pending || []).includes(f.name);
      datiBox.append(el('div', { style: 'margin-top:4px' },
        el('span', { class: 'score-fixed' },
          `${c.from_suggestion ? 'dalla proposta' : 'corretto da te'}: ${corretti.join(', ')}`),
        el('span', {
          class: inAttesa ? 'scala-attesa' : 'scala-usata',
          style: 'margin-left:6px',
        }, inAttesa
          ? 'non ancora riletta dal modulo'
          : (usate.length ? `il modulo ha applicato: ${usate.join(', ')}` : 'riletta dal modulo'))));
    }
    if (f.improved_from) {
      datiBox.append(el('div', { class: 'scala-migliorati' },
        `nell'ultimo giro e' passato da ${f.improved_from} a ${f.status}`));
    }
    // Le verifiche che vengono da fuori: la depth confermata e le corde del rettangolo.
    // Non sono opinioni del rilevatore, sono cose gia' decise - percio' stanno qui in
    // chiaro e non fra i dettagli.
    const ck = f.checks || {};
    if (ck.why_labels) {
      datiBox.append(el('div', { class: 'scala-fuori' }, ck.why_labels
        + ': ' + (ck.labels_over_depth || []).map((n) => `${n.mm} mm a y=${n.y}`).join(', ')));
    }
    if (ck.why_bar) datiBox.append(el('div', { class: 'scala-fuori' }, ck.why_bar));
    if (ck.why_zero) datiBox.append(el('div', { class: 'scala-fuori' }, ck.why_zero));
    if (ck.why_end) datiBox.append(el('div', { class: 'scala-fuori' }, ck.why_end));
  };

  /* Il passo, la verifica con la depth confermata, e la propagazione.

     Sono la stessa cosa vista da tre lati. Il passo in millimetri e' di cartella: la
     macchina mette le tacche ogni tot e non cambia idea a meta' cartella. Il passo in
     **pixel** invece e' di questo fotogramma soltanto, perche' dipende dalla sua depth. E
     la depth confermata e' il metro esterno: dice quanto deve valere la barra, e da li' si
     vede se il righello e' al posto giusto o no. */
  const passoBox = el('div', { class: 'scala-passo' });
  const VERDETTI = {
    torna: ['#3fb950', 'torna con la depth confermata'],
    'da correggere': ['#d29922', 'quasi: c\'e' + ' un pelo da sistemare'],
    'non torna': ['#f85149', 'non torna con la depth confermata'],
  };

  const chiamaTacche = async (corpo, messaggio) => {
    const f = corrente();
    try {
      const esito = await api(`/projects/${projectId}/scale/study/ticks`,
        { body: { name: f.name, ...corpo } });
      f.ticks = esito.ticks; f.pitch = esito.pitch; f.y_far = esito.y_far;
      f.depth_check = esito.check;
      stato(messaggio ? `${messaggio} — ${esito.from}` : esito.from);
      disegna(); aggiornaZoom(); renderPasso(); renderDati(); renderLista();
    } catch (errore) { stato(errore.message); toast(errore.message, true); }
  };

  const renderPasso = () => {
    passoBox.innerHTML = '';
    const f = corrente();
    const check = f.depth_check || {};
    const ru = f.ruler || {};
    if (f.x == null) { passoBox.style.display = 'none'; return; }
    passoBox.style.display = '';

    const [colore, testo] = VERDETTI[check.verdict] || ['var(--muted)', check.verdict || ''];
    const riga = el('div', { class: 'row' });
    riga.append(el('span', { class: 'scala-verdetto', style: `color:${colore};border-color:${colore}` },
      testo || 'senza depth confermata'));
    if (check.depth_mm) {
      riga.append(el('span', { class: 'hint' },
        `depth ${check.depth_mm} mm · barra ${check.span_px || '—'} px`
        + (check.span_expected_px && Math.abs((check.span_off || 0)) > 0.02
          ? ` (ne servirebbero ${check.span_expected_px})` : '')
        + (check.step_mm ? ` · passo ${check.step_mm} mm = ${ru.pitch ? Math.round(ru.pitch * 10) / 10 : '—'} px` : '')));
    }
    const et = check.labels_on_ticks || {};
    if (et.total) {
      const buone = et.ok === et.total;
      riga.append(el('span', {
        class: 'hint',
        style: `color:${buone ? '#3fb950' : 'var(--warn)'}`,
        title: (et.off || []).map((o) => `${o.cm} cm letto a y=${o.y}, tacca a ${o.nearest}`).join('\n'),
      }, ` · ${et.ok} numeri su ${et.total} cadono sulle tacche`));
    }
    passoBox.append(riga);

    if (check.unit_hint) {
      passoBox.append(el('div', { class: 'scala-avviso' }, check.unit_hint));
    }

    const comandi = el('div', { class: 'row' });
    const rigenera = el('button', { class: 'ghost' }, 'Rigenera le tacche a passo costante');
    rigenera.addEventListener('click', () => chiamaTacche({}, 'tacche rigenerate'));
    comandi.append(rigenera);

    if (check.y_far_suggested != null && !check.unit_hint) {
      const fondo = el('button', { class: 'ghost' },
        `Porta il fondo a ${check.y_far_suggested} px (dalla depth)`);
      fondo.addEventListener('click', () => chiamaTacche({ use_depth: true }, 'fondo dalla depth'));
      comandi.append(fondo);
    }

    const passoMm = el('select', {});
    for (const v of [1, 2, 2.5, 5, 10, 20, 25, 50]) {
      passoMm.append(el('option', { value: String(v),
        ...(check.step_mm === v ? { selected: 'selected' } : {}) }, `${v} mm`));
    }
    passoMm.addEventListener('change', () => chiamaTacche(
      { step_mm: parseFloat(passoMm.value) }, `passo ${passoMm.value} mm`));
    comandi.append(el('span', { class: 'hint' }, 'passo'), passoMm);
    passoBox.append(comandi);

    // La propagazione: da questo fotogramma a tutti gli altri.
    const propaga = el('div', { class: 'row' });
    const dice = el('span', { class: 'hint' });
    const guarda = el('button', { class: 'ghost' }, 'Propaga a tutte le immagini');
    guarda.addEventListener('click', async () => {
      guarda.disabled = true;
      try {
        const prima = await api(`/projects/${projectId}/scale/study/propagate`,
          { body: { from: f.name, preview: true } });
        const fatti = (prima.results || []).filter((r) => r.done);
        const fermi = (prima.results || []).filter((r) => !r.done);
        dice.innerHTML = '';
        dice.append(el('div', {},
          `${fatti.length} fotogrammi prenderebbero questa colonna, questo zero e il passo `
          + `di ${prima.step_mm} mm, ciascuno col suo passo in pixel ricavato dalla sua depth`));
        for (const r of fermi.slice(0, 6)) {
          dice.append(el('div', { style: 'color:var(--warn)' }, `${r.name.split('/').pop()}: ${r.why}`));
        }
        if (!fatti.length) return;
        dice.append(confermaInDueTempi(`Applica a ${fatti.length} fotogrammi`,
          'ognuno prende la barra di cartella e il passo che gli tocca. '
          + 'Quelli corretti a mano restano come sono.',
          async () => {
            try {
              const esito = await api(`/projects/${projectId}/scale/study/propagate`,
                { body: { from: f.name } });
              dati = await api(`/projects/${projectId}/scale/study`);
              frames = dati.frames || [];
              await caricaSuggerimenti();
              toast(`propagato a ${esito.changed} fotogrammi`);
              mostra();
            } catch (errore) { toast(errore.message, true); }
          }));
      } catch (errore) { dice.textContent = errore.message; toast(errore.message, true); }
      finally { guarda.disabled = false; }
    });
    propaga.append(guarda, dice);
    passoBox.append(propaga);
    passoBox.append(el('div', { class: 'hint' },
      'doppio clic su una tacca: da quella vengono fuori tutte le altre, perche\' dentro a '
      + 'un\'immagine il passo non cambia. Con alt si aggiunge la singola tacca.'));
  };

  const azioni = el('div', { class: 'row' });
  const renderAzioni = () => {
    azioni.innerHTML = '';
    const f = corrente();
    for (const [dove, etichetta] of [['top', 'zero in alto'], ['bottom', 'zero in basso']]) {
      const b = el('button', { class: 'chip' + (f.B_zero_end === dove ? ' on' : '') }, etichetta);
      b.addEventListener('click', () => {
        ricorda(); f.B_zero_end = dove; salva({ zero_end: dove }); renderAzioni();
      });
      azioni.append(b);
    }
    const prop = proposta();
    if (prop && f.x == null) {
      const usa = el('button', {}, 'Usa il righello proposto');
      usa.addEventListener('click', async () => {
        try {
          await api(`/projects/${projectId}/scale/study/accept`, { body: { name: f.name } });
          dati = await api(`/projects/${projectId}/scale/study`);
          frames = dati.frames || [];
          await caricaSuggerimenti();
          stato('proposta accettata');
          mostra();
        } catch (errore) { toast(errore.message, true); }
      });
      azioni.append(usa);
    }
    if (f.x != null && (f.ticks || []).length > 1) {
      const rifai = el('button', { class: 'ghost' }, 'Tacche a passo costante');
      rifai.addEventListener('click', ridistribuisci);
      azioni.append(rifai);
    }
    /* «Va bene cosi'». Non tutti i «da rivedere» sono errori: spesso il righello e' quello
       giusto e il modulo si e' solo tenuto basso col punteggio. Dirlo non e' una correzione
       - non c'e' niente da correggere - ma lasciare il dubbio scritto nasconde quelli veri. */
    if (['review', 'reject', 'accepted'].includes(f.status)) {
      const bene = el('button', {}, f.status === 'accepted' ? 'Confermo, va bene' : 'Va bene cosi\', togli «da rivedere»');
      bene.addEventListener('click', async () => {
        try {
          await api(`/projects/${projectId}/scale/study/approve`, { body: { name: f.name } });
          f.status = 'approved'; f.approved = true;
          stato('confermato: non e\' piu\' da rivedere');
          renderAzioni(); renderLista(); rifaiChips(); mostra();
        } catch (errore) { toast(errore.message, true); }
      });
      azioni.append(bene);
    } else if (f.status === 'approved') {
      const ripensa = el('button', { class: 'ghost' }, 'Rimettilo fra quelli da rivedere');
      ripensa.addEventListener('click', async () => {
        try {
          await api(`/projects/${projectId}/scale/study/approve`,
            { body: { name: f.name, reset: true } });
          dati = await api(`/projects/${projectId}/scale/study`);
          frames = dati.frames || [];
          stato('torna da rivedere');
          rifaiChips(); mostra();
        } catch (errore) { toast(errore.message, true); }
      });
      azioni.append(ripensa);
    }
    const indietro = el('button', { class: 'ghost' }, 'Annulla l\'ultimo gesto');
    indietro.addEventListener('click', annulla);
    azioni.append(indietro);
    if (Object.keys(correzione()).filter((k) => k !== 'ts').length) {
      azioni.append(confermaInDueTempi('Togli le mie correzioni qui',
        'torna a quello che ha trovato il modulo.',
        async () => {
          await api(`/projects/${projectId}/scale/study/correct`,
            { body: { name: corrente().name, reset: true } });
          delete dati.corrections[corrente().name];
          await caricaSuggerimenti();
          stato('correzioni togliate');
          renderAzioni(); renderDati(); renderLista();
        }));
    }
    if (prop) azioni.append(el('span', { class: 'hint' }, prop.reason || ''));
  };

  const listBox = el('div', { class: 'score-list' });
  const renderLista = () => {
    listBox.innerHTML = '';
    for (const f of visibili()) {
      const c = (dati.corrections || {})[f.name];
      const riga = el('div', { class: 'score-row scala-riga' + (f === corrente() ? ' current' : '') },
        el('span', { class: 'score-value', style: `color:${COLORI_SCALA[f.status] || 'var(--muted)'}` },
          { accepted: 'ok', corrected: 'tuo', review: 'rev' }[f.status] || 'no'),
        el('span', { class: 'score-group' }, `${(f.ticks || []).length}t`),
        el('span', { class: 'score-name', title: f.name }, f.name.split('/').pop()),
        c ? el('span', { class: 'score-fixed', title: c.from_suggestion ? 'dalla proposta' : 'corretto da te' },
          c.from_suggestion ? 'prop' : 'corr') : null,
        (!c && suggerimenti[f.name]) ? el('span', { class: 'hint', title: 'righello proposto' }, '~') : null,
        f.improved_from ? el('span', { class: 'scala-migliorati' }, '↑') : null,
        (c && (dati.pending || []).includes(f.name))
          ? el('span', { class: 'scala-attesa', title: 'non ancora riletta dal modulo' }, '·') : null,
      );
      riga.addEventListener('click', () => { indice = frames.indexOf(f); mostra(); });
      listBox.append(riga);
      if (f === corrente()) requestAnimationFrame(() => tieniInVista(listBox, riga));
    }
  };

  /* La lente segue il puntatore anche qui: sul righello serve piu' che altrove, perche'
     quello che si cerca - una tacca, lo zero - e' alto due pixel. */
  const seguiConLaLente = (event) => {
    const source = `scala:${projectId}`;
    if (!Lente.attiva(source) || !Lente.segueOra()) return;
    const r = image.getBoundingClientRect();
    if (!r.width || !r.height) return;
    const f = corrente();
    const sx = (f.w || image.naturalWidth || 0) / r.width;
    const sy = (f.h || image.naturalHeight || 0) / r.height;
    if (!sx || !sy) return;
    Lente.segui((event.clientX - r.left) * sx, (event.clientY - r.top) * sy, source);
  };
  stage.addEventListener('pointermove', seguiConLaLente);

  const didascalia = el('div', { class: 'hint' });
  const mostra = () => {
    const elenco = visibili();
    if (!elenco.length) { didascalia.textContent = 'nessun fotogramma con questo filtro'; return; }
    if (!elenco.includes(corrente())) indice = frames.indexOf(elenco[0]);
    const f = corrente();
    taccaInModifica = null;
    taccaOrigine = null;
    trascinandoTacca = false;
    if (zoom.y_tacca) zoom.y_tacca.finestra = null;
    attesaSecondoClic = null;
    image.src = `/api/projects/${projectId}/image?name=${encodeURIComponent(f.name)}&w=980`;
    didascalia.innerHTML = '';
    didascalia.append(
      `${elenco.indexOf(f) + 1} di ${elenco.length} — ${f.name.split('/').pop()} · `,
      el('strong', { style: `color:${COLORI_SCALA[f.status] || 'var(--text)'}` }, f.status));
    if (f.x == null) {
      didascalia.append(el('span', {}, ' · righello non trovato: due clic per indicarlo'));
    }
    disegna(); aggiornaZoom(); renderDati(); renderPasso(); renderAzioni(); renderLista();
    if (root.isConnected) Lente.aggiornaSeAttiva(contestoLente());
  };
  const passo = (delta) => {
    const elenco = visibili();
    if (!elenco.length) return;
    const dove = elenco.indexOf(corrente());
    indice = frames.indexOf(elenco[((dove < 0 ? 0 : dove) + delta + elenco.length) % elenco.length]);
    mostra();
  };

  const chips = el('div', { class: 'ov-chips' });
  const perStato = (s) => frames.filter((f) => f.status === s).length;
  /* I chip si rifanno a ogni cambio di stato: approvare un fotogramma sposta un numero da
     «da rivedere» a «confermati da te», ed e' proprio quel numero che si vuole vedere
     scendere. */
  const rifaiChips = () => {
    chips.innerHTML = '';
    for (const [chiave, etichetta] of [['', `tutti ${frames.length}`],
                                       ['accepted', `trovati ${perStato('accepted')}`],
                                       ['corrected', `dati da te ${perStato('corrected')}`],
                                       ['approved', `confermati da te ${perStato('approved')}`],
                                       ['review', `da rivedere ${perStato('review')}`],
                                       ['reject', `senza righello ${perStato('reject')}`]]) {
      const b = el('button', { class: 'chip' + (chiave === filtro ? ' on' : '') }, etichetta);
      if (chiave) b.style.borderColor = COLORI_SCALA[chiave];
      b.addEventListener('click', () => {
        filtro = chiave;
        for (const altro of chips.children) altro.classList.remove('on');
        b.classList.add('on');
        mostra();
      });
      chips.append(b);
    }
    // In blocco: quando si e' scorso tutto e i dubbi rimasti sono buoni, chiuderli uno per
    // uno e' otto volte lo stesso gesto.
    const dubbi = perStato('review') + perStato('reject');
    if (dubbi) {
      chips.append(confermaInDueTempi(
        dubbi === 1 ? 'Confermo l\'unico da rivedere' : `Confermo i ${dubbi} da rivedere`,
        'restano com\'e' + ' il modulo li ha trovati, ma senza piu\' il cartellino: '
        + 'guardali prima, che dopo non si distinguono dagli altri.',
        async () => {
          try {
            const esito = await api(`/projects/${projectId}/scale/study/approve`,
              { body: { statuses: ['review', 'reject'] } });
            dati = await api(`/projects/${projectId}/scale/study`);
            frames = dati.frames || [];
            toast(`${esito.approved} fotogrammi confermati`);
            rifaiChips(); mostra();
          } catch (errore) { toast(errore.message, true); }
        }));
    }
  };
  rifaiChips();

  const avanzamento = el('span', { class: 'hint' });
  const rifai = confermaInDueTempi('Rifai lo studio con le mie correzioni',
    'il modulo rigira su tutti i fotogrammi, con dentro quello che hai corretto.',
    async () => {
      try {
        await codaSalvataggi;
        const avvio = await api(`/projects/${projectId}/scale/study/run`, { body: {} });
        await pollJob(avvio.job_id, avanzamento);
        dati = await api(`/projects/${projectId}/scale/study`);
        frames = dati.frames || [];
        indice = Math.min(indice, frames.length - 1);
        avanzamento.textContent = '';
        await caricaSuggerimenti();
        const esito = dati.last_run || {};
        const quanti = (esito.improved || []).length;
        toast(quanti
          ? `studio rifatto: ${quanti} fotogrammi migliorati`
          : 'studio rifatto: nessun cambio di stato');
        renderGiro();
        mostra();
      } catch (errore) { avanzamento.textContent = ''; toast(errore.message, true); }
    });

  const daTastiera = (event) => {
    if (!root.isConnected) return;
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes((event.target || {}).tagName)) return;
    const f = corrente();
    if (event.key === 'z' && (event.metaKey || event.ctrlKey)) { event.preventDefault(); annulla(); return; }
    const su_giu = { ArrowUp: -1, ArrowDown: 1 }[event.key];
    if (su_giu && (event.shiftKey || event.altKey) && f.x != null) {
      // Il pixel esatto con la tastiera: shift muove lo zero, alt il fondo.
      event.preventDefault();
      const chiave = event.altKey ? 'y_far' : 'y_zero';
      ricorda();
      f[chiave] = limita((f[chiave] || 0) + su_giu, f.h);
      if (chiave === 'y_zero') pulisciTaccheFuoriDalloZero(f);
      disegna(); aggiornaZoom(); renderDati();
      salva(chiave === 'y_zero' ? campiZero(f) : { [chiave]: f[chiave] });
      return;
    }
    if (event.key === 'ArrowLeft') { event.preventDefault(); passo(-1); }
    if (event.key === 'ArrowRight') { event.preventDefault(); passo(1); }
  };
  window.addEventListener('keydown', daTastiera);

  /* Cosa vede la lente qui: la colonna del righello, con le tacche e i due estremi. Non
     c'e' un riquadro da trascinare - il righello e' una riga di segni - quindi si guarda e
     basta, ma il tasto sta dove sta in tutte le altre sezioni. */
  /* La lente sul righello: non una vetrina, uno strumento.

     E' la finestra dove il righello si vede davvero — dieci volte piu' grande, su un
     secondo schermo se c'e'. Quindi e' li' che si deve poter lavorare: si sceglie in alto
     cosa si sta indicando (zero, fondo, tacca, passo) e un clic lo mette. Alt+clic toglie
     la tacca vicina, col suo numero.

     Prima da qui si poteva solo guardare, e per spostare una tacca di due pixel si tornava
     sull'immagine piccola — cioe' proprio dove non si vedeva. */
  let modoLente = 'tacca';
  const MODI_LENTE = {
    zero: 'clic: metti lo zero',
    fondo: 'clic: metti il fondo',
    tacca: 'clic: aggiungi una tacca · alt+clic: togli quella vicina',
    passo: 'clic su una tacca giusta: le altre si rimettono a quel passo',
  };
  const applicaPuntoLente = (modo, y, alt) => {
    const f = corrente();
    if (alt || modo === 'togli') {
      const vicina = taccaVicina(y, 14);
      if (vicina == null) { stato('nessuna tacca li\' vicino'); return; }
      togliTacca(vicina);
    } else if (modo === 'zero' || modo === 'fondo') {
      const chiave = modo === 'zero' ? 'y_zero' : 'y_far';
      ricorda();
      f[chiave] = y;
      if (chiave === 'y_zero') pulisciTaccheFuoriDalloZero(f);
      salva(chiave === 'y_zero' ? campiZero(f) : { [chiave]: y });
      stato(`${modo} a y=${y}`);
      disegna(); aggiornaZoom(); renderDati(); renderPasso();
    } else if (modo === 'passo') {
      generaDaTacca(y);
      return;
    } else {
      aggiungiTacca(y);
      stato(`tacca a y=${y}`);
    }
    Lente.aggiornaSeAttiva(contestoLente());
  };
  const contestoLente = () => {
    const f = corrente();
    if (f.x == null) {
      const primo = attesaSecondoClic;
      const cx = primo ? primo.x : Math.round((f.w || 0) / 2);
      const cy = primo ? primo.y_zero : Math.round((f.h || 0) / 2);
      return {
        source: `scala:${projectId}`, projectId, name: f.name, size: [f.w || 0, f.h || 0],
        boxes: [],
        lines: primo ? [
          { x: primo.x, color: '#40d0ff', label: 'colonna' },
          { y: primo.y_zero, color: '#3fb950', label: 'zero' },
        ] : [],
        focus: (f.w && f.h) ? {
          left: Math.max(0, cx - 70), right: Math.min(f.w, cx + 70),
          top: Math.max(0, cy - 45), bottom: Math.min(f.h, cy + 45),
        } : null,
        caption: primo ? 'zero indicato: ora clicca il fondo'
          : 'righello non trovato: clicca prima lo zero, poi il fondo',
        onPunto: ({ x, y }) => {
          if (!attesaSecondoClic) {
            attesaSecondoClic = { x, y_zero: y };
            stato('ora clicca il fondo del righello');
          } else {
            indicaRighello(attesaSecondoClic, y);
            attesaSecondoClic = null;
          }
          disegna(); aggiornaZoom(); renderDati(); renderPasso(); renderAzioni();
          Lente.aggiornaSeAttiva(contestoLente());
        },
        puntoLabel: primo ? 'indica il fondo del righello' : 'indica lo zero e la colonna',
      };
    }
    const x = Number(f.x);
    const ys = [f.y_zero, f.y_far].filter((v) => v != null).map(Number);
    const su = ys.length ? Math.min(...ys) : 0;
    const giu = ys.length ? Math.max(...ys) : (f.h || 0);
    const attorno = (y, meta) => ({ left: x - 60, right: x + 60,
                                    top: Math.round(y - meta), bottom: Math.round(y + meta) });
    return {
      source: `scala:${projectId}`,
      projectId, name: f.name, size: [f.w || 0, f.h || 0],
      boxes: righelloVisibile ? [{ box: { left: x - 26, right: x + 26, top: su, bottom: giu },
                color: '#40d0ff', label: 'righello' }] : [],
      segments: righelloVisibile ? (f.ticks || []).map((y) => ({
        x1: x - 22, x2: x + 22, y: Number(y), color: '#3fb950' })) : [],
      lines: righelloVisibile ? [
        ...(f.y_zero != null ? [{ y: Number(f.y_zero), color: '#3fb950', label: 'zero' }] : []),
        ...(f.y_far != null ? [{ y: Number(f.y_far), color: '#d29922', label: 'fondo' }] : []),
      ] : [],
      focus: { left: x - 70, right: x + 70, top: su - 20, bottom: giu + 20 },
      caption: `${(f.ticks || []).length} tacche`,
      // I bersagli fanno due mestieri insieme: portano la vista dove serve e dicono cosa
      // fara' il prossimo clic. Sono la stessa cosa, in fondo: si va dove si vuole agire.
      targets: [
        ...(f.y_zero != null ? [{ id: 'zero', label: 'zero', box: attorno(f.y_zero, 40) }] : []),
        ...(f.y_far != null ? [{ id: 'fondo', label: 'fondo', box: attorno(f.y_far, 40) }] : []),
        { id: 'tacca', label: 'tacche',
          box: { left: x - 60, right: x + 60, top: su, bottom: giu } },
        { id: 'passo', label: 'passo',
          box: { left: x - 60, right: x + 60, top: su, bottom: giu } },
      ],
      onTarget: (voce) => {
        modoLente = voce.id;
        // Il cartellino in alto dice cosa fara' il prossimo clic: se non si rinfresca,
        // resta quello di prima e si clicca credendo di fare un'altra cosa.
        Lente.aggiornaSeAttiva(contestoLente());
      },
      onPunto: ({ y, alt, target }) => applicaPuntoLente(target || modoLente, y, alt),
      puntoLabel: MODI_LENTE[modoLente] || MODI_LENTE.tacca,
    };
  };

  // Frecce e lente sopra l'immagine, come in tutte le altre sezioni.
  const visibilita = el('button', { class: 'ghost' }, 'Nascondi righello');
  visibilita.addEventListener('click', () => {
    righelloVisibile = !righelloVisibile;
    visibilita.textContent = righelloVisibile ? 'Nascondi righello' : 'Mostra righello';
    disegna(); aggiornaZoom();
    Lente.aggiornaSeAttiva(contestoLente());
  });
  const barraImmagine = el('div', { class: 'barra-immagine' },
    el('div', { class: 'ov-nav' },
      el('button', { class: 'ghost sq', onclick: () => passo(-1) }, '‹'),
      el('button', { class: 'ghost sq', onclick: () => passo(1) }, '›'),
      didascalia, statoBox),
    visibilita,
    Lente.bottone(contestoLente));

  /* Quello che la cartella sa gia' prima che il righello venga guardato.

     Lo studio della scala viene dopo l'orientamento e dopo la depth, e da quei due si sanno
     tre cose che il rilevatore da solo non ha: su quale orientamento si misura (uno, quello
     che ha tutte le depth), da che parte sta lo zero (in alto per NF e LR, in basso per UD
     e LRUD) e dove comincia l'immagine ecografica — sopra la corda piu' alta lo zero non ci
     puo' stare, perche' li' ecografia non ce n'e'. */
  const testaCartella = el('div', { class: 'scala-testa' });
  const renderTesta = () => {
    testaCartella.innerHTML = '';
    const rg = dati.ruler_group || {};
    const zf = dati.zero_folder;
    const za = dati.zero_expected;
    testaCartella.append(el('p', { class: 'hint' },
      `il righello di questa cartella: colonna a x=${Math.round((dati.zone || {}).x || 0)}, `
      + `trovata su ${(dati.zone || {}).found || 0} fotogrammi su ${(dati.zone || {}).total || 0}.`
      + (rg.group ? ` Misurato su ${rg.group}, zero ${rg.zero_end === 'top' ? 'in alto' : 'in basso'}`
        + ` — gli altri orientamenti si ottengono ribaltandolo.` : '')));
    if (rg.group && !rg.complete) {
      testaCartella.append(el('p', { class: 'avviso' },
        `${rg.group} non ha tutte le depth: manca ${(rg.missing_depths || []).join(', ')} mm. `
        + 'Quelle depth resterebbero senza righello da ribaltare.'));
    }
    if (za) {
      const riga = el('div', { class: 'row' },
        el('span', { class: 'hint' },
          `il ventaglio comincia a y=${za.y} (${za.from})`
          + (zf ? ` · i righelli si accordano su y=${zf.y}: ${zf.from}` : '')));
      if (zf && (zf.off || []).length) {
        const b = el('button', { class: 'ghost' },
          `Metti lo zero a ${zf.y} sui ${zf.off.length} fuori posto`);
        b.title = zf.off.map((n) => n.split('/').pop()).join(', ');
        b.addEventListener('click', async () => {
          b.disabled = true;
          try {
            const esito = await api(`/projects/${projectId}/scale/study/zero`, { body: {} });
            toast(`zero messo su ${esito.changed.length} fotogrammi`);
            dati = await api(`/projects/${projectId}/scale/study`);
            frames = dati.frames || [];
            renderTesta(); mostra(); rifaiChips();
          } catch (errore) { toast(errore.message, true); }
          finally { b.disabled = false; }
        });
        riga.append(b);
      }
      testaCartella.append(riga);
    }
  };
  renderTesta();

  root.append(
    testaCartella,
    legenda(),
    chips, notiBox, giroBox,
    // Immagine a sinistra, strumenti a destra: la stessa forma di tutte le sezioni.
    el('div', { class: 'ov-body' },
      el('div', { class: 'ov-main' },
        barraImmagine,
        el('div', { class: 'row', style: 'margin:0 0 4px' },
          el('span', { class: 'hint' }, 'finestre ingrandite:'), vicinanzaSel,
          modiZoom,
          el('span', { class: 'hint' }, 'scegli l\'azione, poi clicca sul pixel preciso')),
        el('div', { class: 'scala-zoom-coppia' }, zoomZero, zoomFondo, zoomTacca),
        stage),
      el('div', { class: 'ov-side' }, editorTacche, passoBox, datiBox, azioni,
        el('div', { class: 'row' }, rifai, avanzamento), listBox)),
  );

  image.addEventListener('load', () => { disegna(); aggiornaZoom(); });
  window.addEventListener('resize', disegna);
  if (window.ResizeObserver) new ResizeObserver(() => { disegna(); }).observe(image);
  await caricaSuggerimenti();
  renderGiro();
  mostra();
  return root;
}
