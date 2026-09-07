/* Lo studio del righello, fotogramma per fotogramma.

   Il modulo trova la colonna della scala, ne stacca le tacche, distingue lo zero dal fondo,
   ricava il passo e legge i numeri con l'OCR. Qui si guarda quello che ha trovato disegnato
   sull'immagine e lo si corregge: la colonna e lo zero si trascinano, le tacche si aggiungono
   e si tolgono con un clic, i numeri si riscrivono.

   Le correzioni non restano qui: rientrano nel modulo alla run successiva (`--corrections`),
   che le applica *dopo* la detection — cosi' la pagina continua a mostrare anche cosa avrebbe
   detto da solo — e una colonna corretta fa da ancora per tutta la cartella. */

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
  let modo = 'guarda';          // guarda · tacca · numero
  const corrente = () => frames[indice] || {};
  const passa = (f) => !filtro || f.status === filtro;
  const visibili = () => frames.filter(passa);

  const stage = el('div', { class: 'editor-stage scala-stage' });
  const image = el('img', { alt: '' });
  const strati = el('div', { class: 'scala-strati' });
  stage.append(image, strati);

  const scala = () => {
    const f = corrente();
    return (image.clientWidth || 1) / (f.w || image.naturalWidth || 1);
  };

  /* --- il disegno: colonna, zero, fondo, tacche, numeri ------------------------------- */
  const disegna = () => {
    const f = corrente();
    strati.innerHTML = '';
    const s = scala();
    if (!s || !f.w) return;
    const alto = (f.h || 0) * s;
    if (f.x != null) {
      const colonna = el('div', { class: 'scala-colonna', title: `colonna x=${Math.round(f.x)}` });
      colonna.style.left = `${f.x * s}px`;
      colonna.style.height = `${alto}px`;
      colonna.addEventListener('pointerdown', (e) => trascina(e, 'x'));
      strati.append(colonna);
    }
    for (const t of f.ticks || []) {
      const tacca = el('div', { class: 'scala-tacca', title: `tacca y=${Math.round(t)}` });
      tacca.style.top = `${t * s}px`;
      tacca.style.left = `${((f.x ?? 0) - 26) * s}px`;
      tacca.style.width = `${52 * s}px`;
      if (modo === 'tacca') {
        tacca.classList.add('togliibile');
        tacca.addEventListener('click', (e) => { e.stopPropagation(); togliTacca(t); });
      }
      strati.append(tacca);
    }
    for (const [y, valore] of f.labels || []) {
      const numero = el('div', { class: 'scala-numero' }, `${valore} cm`);
      numero.style.top = `${y * s}px`;
      numero.style.left = `${((f.x ?? 0) + 34) * s}px`;
      if (modo === 'numero') {
        numero.classList.add('modificabile');
        numero.addEventListener('click', (e) => { e.stopPropagation(); cambiaNumero(y, valore); });
      }
      strati.append(numero);
    }
    for (const [chiave, classe, etichetta] of [['y_zero', 'zero', '0'], ['y_far', 'fondo', 'fondo']]) {
      if (f[chiave] == null) continue;
      const riga = el('div', { class: `scala-estremo scala-${classe}` },
        el('span', {}, etichetta));
      riga.style.top = `${f[chiave] * s}px`;
      riga.addEventListener('pointerdown', (e) => trascina(e, chiave));
      strati.append(riga);
    }
  };

  /* --- trascinare: la colonna in orizzontale, zero e fondo in verticale --------------- */
  function trascina(event, quale) {
    event.preventDefault();
    event.stopPropagation();
    const f = corrente();
    const s = scala();
    const da = { x: event.clientX, y: event.clientY, valore: f[quale] };
    const muovi = (e) => {
      const delta = quale === 'x' ? (e.clientX - da.x) : (e.clientY - da.y);
      const massimo = quale === 'x' ? f.w : f.h;
      f[quale] = Math.max(0, Math.min(massimo, Math.round(da.valore + delta / (s || 1))));
      disegna();
      renderDati();
    };
    const molla = () => {
      window.removeEventListener('pointermove', muovi);
      window.removeEventListener('pointerup', molla);
      salva({ [quale]: f[quale] });
    };
    window.addEventListener('pointermove', muovi);
    window.addEventListener('pointerup', molla);
  }

  /* --- tacche: si aggiungono cliccando sull'immagine, si tolgono cliccandole ---------- */
  stage.addEventListener('click', (event) => {
    if (modo !== 'tacca' || event.target.closest('.scala-tacca')) return;
    const s = scala();
    const y = Math.round((event.clientY - stage.getBoundingClientRect().top) / (s || 1));
    const f = corrente();
    f.ticks = [...(f.ticks || []), y].sort((a, b) => a - b);
    const aggiunte = [...(correzione().ticks_add || []), y];
    salva({ ticks_add: aggiunte });
    disegna();
    renderDati();
  });
  const togliTacca = (y) => {
    const f = corrente();
    f.ticks = (f.ticks || []).filter((t) => Math.round(t) !== Math.round(y));
    const tolte = [...(correzione().ticks_del || []), Math.round(y)];
    salva({ ticks_del: tolte });
    disegna();
    renderDati();
  };
  const cambiaNumero = (y, valore) => {
    const scritto = window.prompt(`valore della tacca a y=${Math.round(y)} (in cm)`, String(valore));
    if (scritto === null) return;
    const cm = parseFloat(scritto.replace(',', '.'));
    if (!isFinite(cm)) { toast('non e\' un numero', true); return; }
    const f = corrente();
    f.labels = (f.labels || []).map(([yy, vv]) => (Math.round(yy) === Math.round(y) ? [yy, cm] : [yy, vv]));
    const numeri = [...(correzione().nums || []).filter(([yy]) => Math.round(yy) !== Math.round(y)),
                    [Math.round(y), cm * 10]];
    salva({ nums: numeri });
    disegna();
    renderDati();
  };

  const correzione = () => (dati.corrections || {})[corrente().name] || {};
  const salva = async (campi) => {
    try {
      await api(`/projects/${projectId}/scale/study/correct`,
        { body: { name: corrente().name, ...campi } });
      dati.corrections = dati.corrections || {};
      dati.corrections[corrente().name] = { ...correzione(), ...campi };
      renderDati();
    } catch (errore) { toast(errore.message, true); }
  };

  /* --- il pannello dei dati: cosa ha trovato, e cosa hai corretto tu ------------------ */
  const datiBox = el('div', { class: 'hint' });
  const renderDati = () => {
    const f = corrente();
    const c = correzione();
    const corretti = Object.keys(c).filter((k) => k !== 'ts');
    datiBox.innerHTML = '';
    datiBox.append(
      el('div', {},
        `stato ${f.status}${f.reason ? ` — ${f.reason}` : ''}`),
      el('div', {},
        `colonna x=${f.x != null ? Math.round(f.x) : '—'} · zero y=${f.y_zero != null ? Math.round(f.y_zero) : '—'}`
        + ` · fondo y=${f.y_far != null ? Math.round(f.y_far) : '—'} · ${(f.ticks || []).length} tacche`
        + ` · passo ${f.pitch ? Math.round(f.pitch) + ' px' : '—'}`
        + ` · ${f.mm_per_px ? f.mm_per_px.toFixed(4) + ' mm/px' : '—'}`),
      el('div', {},
        `numeri letti: ${(f.labels || []).map(([, v]) => `${v}`).join(' · ') || 'nessuno'}`
        + `${f.D_step_mm ? ` · passo ${f.D_step_mm} mm` : ''}`
        + `${f.D_coherent === false ? ' · numeri incoerenti fra loro' : ''}`),
      el('div', {},
        `verso: lo zero e' ${f.B_zero_end === 'top' ? 'in alto' : f.B_zero_end === 'bottom' ? 'in basso' : '—'}`
        + `${f.B_zero_moved ? ' (spostato rispetto alla detection)' : ''}`
        + `${f.E_depth_interface ? ` · depth dall'interfaccia ${f.E_depth_interface} mm` : ''}`
        + `${f.E_verdict ? ` · ${f.E_verdict}` : ''}`),
    );
    if (corretti.length) {
      datiBox.append(el('div', { class: 'score-fixed', style: 'display:inline-block;margin-top:4px' },
        `corretto da te: ${corretti.join(', ')}`));
    }
  };

  /* --- il verso e il ripristino ------------------------------------------------------- */
  const azioni = el('div', { class: 'row' });
  const renderAzioni = () => {
    azioni.innerHTML = '';
    const f = corrente();
    for (const [dove, etichetta] of [['top', 'lo zero e\' in alto'], ['bottom', 'lo zero e\' in basso']]) {
      const b = el('button', { class: 'chip' + (f.B_zero_end === dove ? ' on' : '') }, etichetta);
      b.addEventListener('click', () => { f.B_zero_end = dove; salva({ zero_end: dove }); renderAzioni(); });
      azioni.append(b);
    }
    if (Object.keys(correzione()).filter((k) => k !== 'ts').length) {
      azioni.append(confermaInDueTempi('Togli le mie correzioni su questo fotogramma',
        'torna a quello che ha trovato il modulo.',
        async () => {
          await api(`/projects/${projectId}/scale/study/correct`,
            { body: { name: corrente().name, reset: true } });
          toast('correzioni tolte: rifai lo studio per rivederlo');
          delete dati.corrections[corrente().name];
          renderAzioni(); renderDati();
        }));
    }
  };

  /* --- elenco laterale ---------------------------------------------------------------- */
  const listBox = el('div', { class: 'score-list' });
  const renderLista = () => {
    listBox.innerHTML = '';
    for (const f of visibili()) {
      const c = (dati.corrections || {})[f.name];
      const riga = el('div', {
        class: 'score-row' + (f === corrente() ? ' current' : ''),
      },
        el('span', { class: 'score-value', style: `color:${COLORI_SCALA[f.status] || 'var(--muted)'}` },
          f.status === 'accepted' ? 'ok' : f.status === 'review' ? 'rev' : 'no'),
        el('span', { class: 'score-group' }, `${(f.ticks || []).length}t`),
        el('span', { class: 'score-name', title: f.name }, f.name.split('/').pop()),
        c ? el('span', { class: 'score-fixed' }, 'corretto') : null,
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
    image.src = `/api/projects/${projectId}/image?name=${encodeURIComponent(f.name)}&w=980`;
    didascalia.innerHTML = '';
    didascalia.append(
      `fotogramma ${elenco.indexOf(f) + 1} di ${elenco.length} — ${f.name.split('/').pop()} · `,
      el('strong', { style: `color:${COLORI_SCALA[f.status] || 'var(--text)'}` }, f.status));
    disegna(); renderDati(); renderAzioni(); renderLista();
  };
  const passo = (delta) => {
    const elenco = visibili();
    if (!elenco.length) return;
    const dove = elenco.indexOf(corrente());
    indice = frames.indexOf(elenco[((dove < 0 ? 0 : dove) + delta + elenco.length) % elenco.length]);
    mostra();
  };

  /* --- barra: filtri, modo, rifai ------------------------------------------------------ */
  const chips = el('div', { class: 'ov-chips' });
  const perStato = (s) => frames.filter((f) => f.status === s).length;
  for (const [chiave, etichetta] of [['', `tutti ${frames.length}`],
                                     ['accepted', `accettati ${perStato('accepted')}`],
                                     ['review', `da rivedere ${perStato('review')}`],
                                     ['reject', `scartati ${perStato('reject')}`]]) {
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

  const modi = el('div', { class: 'ov-chips' });
  for (const [chiave, etichetta] of [['guarda', 'guarda e trascina'],
                                     ['tacca', 'aggiungi/togli tacche'],
                                     ['numero', 'correggi i numeri']]) {
    const b = el('button', { class: 'chip' + (chiave === modo ? ' on' : '') }, etichetta);
    b.addEventListener('click', () => {
      modo = chiave;
      for (const altro of modi.children) altro.classList.remove('on');
      b.classList.add('on');
      disegna();
    });
    modi.append(b);
  }

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
        toast('studio rifatto');
        mostra();
      } catch (errore) { avanzamento.textContent = ''; toast(errore.message, true); }
    });

  const daTastiera = (event) => {
    if (!root.isConnected) return;
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes((event.target || {}).tagName)) return;
    if (event.key === 'ArrowLeft') { event.preventDefault(); passo(-1); }
    if (event.key === 'ArrowRight') { event.preventDefault(); passo(1); }
  };
  window.addEventListener('keydown', daTastiera);

  root.append(
    el('p', { class: 'hint' },
      `il righello di questa cartella: colonna a x=${Math.round((dati.zone || {}).x || 0)}, `
      + `trovata su ${(dati.zone || {}).found || 0} fotogrammi su ${(dati.zone || {}).total || 0}. `
      + 'Trascina la colonna, lo zero e il fondo; con i modi qui sotto aggiungi o togli tacche '
      + 'e correggi i numeri. Le correzioni rientrano nel modulo quando rifai lo studio.'),
    chips, modi,
    el('div', { class: 'ov-bar' },
      el('div', { class: 'ov-nav' },
        el('button', { class: 'ghost sq', onclick: () => passo(-1) }, '‹'),
        el('button', { class: 'ghost sq', onclick: () => passo(1) }, '›'),
        didascalia)),
    el('div', { class: 'ov-body' },
      el('div', { class: 'ov-main' }, stage,
        el('div', { class: 'ov-under' },
          el('div', { class: 'ov-under-text' }, datiBox, azioni,
            el('div', { class: 'row' }, rifai, avanzamento)))),
      el('div', { class: 'ov-side' }, listBox)),
  );

  image.addEventListener('load', disegna);
  window.addEventListener('resize', disegna);
  if (window.ResizeObserver) new ResizeObserver(() => disegna()).observe(image);
  mostra();
  return root;
}

const COLORI_SCALA = { accepted: '#3fb950', review: '#d29922', reject: '#f85149' };
