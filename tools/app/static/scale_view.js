/* Lo studio del righello, fotogramma per fotogramma, con le correzioni sotto le dita.

   Il modulo trova la colonna della scala, ne stacca le tacche, distingue lo zero dal fondo,
   ricava il passo e legge i numeri con l'OCR. Qui si guarda quello che ha trovato e si
   corregge — e la correzione deve costare un gesto, non un cambio di modo.

   Quindi: si trascina direttamente (colonna, zero, fondo, o tutto il righello insieme),
   shift+clic aggiunge una tacca, clic su una tacca la toglie, doppio clic su un numero lo
   riscrive lì dove sta. Le due finestre ingrandite su zero e fondo servono perché a
   grandezza naturale una tacca è alta due pixel e il pixel giusto non si vede.

   Le correzioni rientrano nel modulo alla run successiva (`--corrections`), applicate *dopo*
   la detection: la pagina continua a mostrare anche cosa avrebbe detto da solo. */

const COLORI_SCALA = { accepted: '#3fb950', corrected: '#40d0ff', approved: '#7ee787',
  review: '#d29922',
                       reject: '#f85149' };

/* I gesti, scritti dove servono. Una riga di prosa lunga non la legge nessuno mentre corregge. */
function legenda() {
  const voci = [
    ['trascina', 'colonna, zero, fondo o una tacca'],
    ['alt + trascina', 'tutto il righello insieme'],
    ['doppio clic', 'aggiunge una tacca'],
    ['× sulla tacca', 'la toglie'],
    ['doppio clic sul numero', 'lo riscrive'],
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
  const storia = [];             // per annullare l'ultimo gesto sul fotogramma
  const corrente = () => frames[indice] || {};
  const passa = (f) => !filtro || f.status === filtro;
  const visibili = () => frames.filter(passa);
  const correzione = () => (dati.corrections || {})[corrente().name] || {};
  const proposta = () => suggerimenti[corrente().name];

  /* --- il fotogramma intero, con il righello sopra ------------------------------------ */
  const stage = el('div', { class: 'editor-stage scala-stage' });
  const image = el('img', { alt: '' });
  const strati = el('div', { class: 'scala-strati' });
  stage.append(image, strati);
  const scala = () => (image.clientWidth || 1) / (corrente().w || image.naturalWidth || 1);

  const disegna = () => {
    const f = corrente();
    strati.innerHTML = '';
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
      const tacca = el('div', { class: 'scala-tacca', title: 'trascina per spostarla' },
        el('span', { class: 'scala-tacca-presa' }),
        el('button', { class: 'scala-tacca-via', title: 'togli questa tacca' }, '×'));
      tacca.style.top = `${t * s}px`;
      tacca.style.left = `${((f.x ?? 0) - 30) * s}px`;
      tacca.style.width = `${60 * s}px`;
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
    for (const [y, valore] of f.labels || []) {
      const numero = el('div', { class: 'scala-numero', title: 'doppio clic: correggi' },
        `${valore} cm`);
      numero.style.top = `${y * s}px`;
      numero.style.left = `${((f.x ?? 0) + 34) * s}px`;
      numero.addEventListener('dblclick', (e) => { e.stopPropagation(); cambiaNumero(numero, y, valore); });
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
  const LARGA_ZOOM = 150;
  const ALTA_ZOOM = 44;
  const zoom = {};

  const finestraZoom = (f, y) => {
    const larghezza = Math.min(LARGA_ZOOM, f.w || LARGA_ZOOM);
    const altezza = Math.min(ALTA_ZOOM, f.h || ALTA_ZOOM);
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

  const creaZoom = (chiave, etichetta, colore) => {
    const scatola = el('div', { class: 'scala-zoom' });
    const img = el('img', { alt: '' });
    const titolo = el('div', { class: 'scala-zoom-titolo' }, etichetta);
    const piano = el('div', { class: 'scala-zoom-piano' });
    scatola.append(titolo, img, piano);
    piano.addEventListener('pointerdown', (e) => trascina(e, chiave, () => ({
      x: scalaZoom(chiave), y: scalaZoom(chiave),
    })));
    zoom[chiave] = { scatola, img, piano, titolo, colore, etichetta, finestra: null };
    if (window.ResizeObserver) new ResizeObserver(() => aggiornaZoom()).observe(img);
    img.addEventListener('load', () => aggiornaZoom());
    return scatola;
  };
  const zoomZero = creaZoom('y_zero', 'zero', '#3fb950');
  const zoomFondo = creaZoom('y_far', 'fondo', '#d29922');

  const aggiornaZoom = () => {
    const f = corrente();
    for (const chiave of ['y_zero', 'y_far']) {
      const z = zoom[chiave];
      const y = f[chiave];
      z.scatola.style.display = (y == null || f.x == null) ? 'none' : '';
      if (y == null || f.x == null) continue;
      const nuova = finestraZoom(f, y);
      if (!z.finestra || z.finestra.join() !== nuova.join()) {
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
      const colonna = el('div', { class: 'scala-zoom-colonna' });
      colonna.style.left = `${(f.x - z.finestra[0]) * s}px`;
      z.piano.append(colonna);
      for (const t of f.ticks || []) {
        if (t < z.finestra[1] || t > z.finestra[3]) continue;
        const tacca = el('div', { class: 'scala-zoom-tacca' });
        tacca.style.top = `${(t - z.finestra[1]) * s}px`;
        z.piano.append(tacca);
      }
      const mira = el('div', { class: 'scala-mira', style: `border-top-color:${z.colore}` });
      mira.style.top = `${(y - z.finestra[1]) * s}px`;
      z.piano.append(mira);
    }
  };

  /* --- trascinare: un lato, o tutto il righello insieme ------------------------------- */
  function trascina(event, quale, scalaFn) {
    event.preventDefault();
    event.stopPropagation();
    const f = corrente();
    const s = scalaFn();
    const fx = typeof s === 'object' ? s.x : s;
    const fy = typeof s === 'object' ? s.y : s;
    const da = { cx: event.clientX, cy: event.clientY,
                 x: f.x, y_zero: f.y_zero, y_far: f.y_far,
                 ticks: [...(f.ticks || [])] };
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
      }
      disegna(); aggiornaZoom(); renderDati();
    };
    const molla = () => {
      window.removeEventListener('pointermove', muovi);
      window.removeEventListener('pointerup', molla);
      const campi = quale === 'tutto'
        ? { x: f.x, y_zero: f.y_zero, y_far: f.y_far, ticks_add: f.ticks }
        : { [quale]: f[quale] };
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
      disegna(); return;
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
    salva({ ticks_add: [...(correzione().ticks_add || []), y] });
    disegna(); aggiornaZoom();
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
            zero_end: f.B_zero_end, ticks_add: tacche });
    stato('righello indicato');
  };

  function trascinaTacca(event, y) {
    event.preventDefault();
    const f = corrente();
    const s = scala();
    const partenza = event.clientY;
    const originale = y;
    let attuale = y;
    let mossa = false;
    ricorda();
    const muovi = (e) => {
      const dy = (e.clientY - partenza) / (s || 1);
      if (Math.abs(dy) < 0.5 && !mossa) return;
      mossa = true;
      attuale = limita(originale + dy, f.h);
      f.ticks = (f.ticks || [])
        .map((t) => (Math.round(t) === Math.round(originale) ? attuale : t))
        .sort((a, b) => a - b);
      disegna(); aggiornaZoom(); renderDati();
    };
    const molla = () => {
      window.removeEventListener('pointermove', muovi);
      window.removeEventListener('pointerup', molla);
      if (!mossa) { storia.pop(); return; }
      // Spostare una tacca, per il modulo, e' toglierla da dov'era e metterla dove sta ora.
      salva({
        ticks_del: [...(correzione().ticks_del || []), Math.round(originale)],
        ticks_add: [...(correzione().ticks_add || [])
          .filter((v) => Math.round(v) !== Math.round(originale)), Math.round(attuale)],
      });
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
    const vecchie = (f.ticks || []).map((t) => Math.round(t));
    f.ticks = nuove;
    salva({ ticks_del: [...(correzione().ticks_del || []), ...vecchie], ticks_add: nuove });
    stato(`${nuove.length} tacche a passo costante (${passoMm} mm)`);
    disegna(); aggiornaZoom();
  };

  const togliTacca = (y) => {
    const f = corrente();
    ricorda();
    f.ticks = (f.ticks || []).filter((t) => Math.round(t) !== Math.round(y));
    salva({ ticks_del: [...(correzione().ticks_del || []), Math.round(y)] });
    disegna();
  };

  /* --- i numeri si riscrivono dove stanno, senza finestrelle di sistema -------------- */
  const cambiaNumero = (nodo, y, valore) => {
    const campo = el('input', { class: 'scala-numero-campo', type: 'text', value: String(valore) });
    nodo.replaceChildren(campo);
    campo.focus(); campo.select();
    const chiudi = (salvare) => {
      const cm = parseFloat(campo.value.replace(',', '.'));
      if (salvare && isFinite(cm)) {
        const f = corrente();
        ricorda();
        f.labels = (f.labels || []).map(([yy, vv]) =>
          (Math.round(yy) === Math.round(y) ? [yy, cm] : [yy, vv]));
        salva({ nums: [...(correzione().nums || []).filter(([yy]) => Math.round(yy) !== Math.round(y)),
                       [Math.round(y), cm * 10]] });
      }
      disegna();
    };
    campo.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); chiudi(true); }
      if (e.key === 'Escape') { e.preventDefault(); chiudi(false); }
    });
    campo.addEventListener('blur', () => chiudi(true));
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
  const stato = (testo) => {
    statoBox.textContent = testo;
    clearTimeout(orologio);
    orologio = setTimeout(() => { statoBox.textContent = ''; }, 2200);
  };
  const salva = async (campi) => {
    try {
      await api(`/projects/${projectId}/scale/study/correct`,
        { body: { name: corrente().name, ...campi } });
      dati.corrections = dati.corrections || {};
      dati.corrections[corrente().name] = { ...correzione(), ...campi };
      stato('salvato — non ancora riletta dal modulo');
      dati.pending = [...new Set([...(dati.pending || []), corrente().name])];
      await caricaSuggerimenti();
      renderGiro(); renderDati(); renderLista(); renderAzioni();
    } catch (errore) { toast(errore.message, true); }
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
      if (f === corrente()) requestAnimationFrame(() => riga.scrollIntoView({ block: 'nearest' }));
    }
  };

  const didascalia = el('div', { class: 'hint' });
  const mostra = () => {
    const elenco = visibili();
    if (!elenco.length) { didascalia.textContent = 'nessun fotogramma con questo filtro'; return; }
    if (!elenco.includes(corrente())) indice = frames.indexOf(elenco[0]);
    const f = corrente();
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
      disegna(); aggiornaZoom(); renderDati();
      salva({ [chiave]: f[chiave] });
      return;
    }
    if (event.key === 'ArrowLeft') { event.preventDefault(); passo(-1); }
    if (event.key === 'ArrowRight') { event.preventDefault(); passo(1); }
  };
  window.addEventListener('keydown', daTastiera);

  root.append(
    el('p', { class: 'hint' },
      `il righello di questa cartella: colonna a x=${Math.round((dati.zone || {}).x || 0)}, `
      + `trovata su ${(dati.zone || {}).found || 0} fotogrammi su ${(dati.zone || {}).total || 0}.`),
    legenda(),
    chips, notiBox, giroBox,
    el('div', { class: 'ov-bar' },
      el('div', { class: 'ov-nav' },
        el('button', { class: 'ghost sq', onclick: () => passo(-1) }, '‹'),
        el('button', { class: 'ghost sq', onclick: () => passo(1) }, '›'),
        didascalia, statoBox)),
    el('div', { class: 'ov-body' },
      el('div', { class: 'ov-main' },
        el('div', { class: 'scala-zoom-coppia' }, zoomZero, zoomFondo),
        stage,
        el('div', { class: 'ov-under' },
          el('div', { class: 'ov-under-text' }, passoBox, datiBox, azioni,
            el('div', { class: 'row' }, rifai, avanzamento)))),
      el('div', { class: 'ov-side' }, listBox)),
  );

  image.addEventListener('load', () => { disegna(); aggiornaZoom(); });
  window.addEventListener('resize', disegna);
  if (window.ResizeObserver) new ResizeObserver(() => { disegna(); }).observe(image);
  await caricaSuggerimenti();
  renderGiro();
  mostra();
  return root;
}
