/* Visualizzatore dell'orientamento: i quattro envelope sull'immagine, il marker trovato
   nell'immagine corrente, il template scelto, e le frecce per scorrere tutta la cartella. */

const GROUP_COLORS = {
  NF: '#ff6040',
  LR: '#40d0ff',
  UD: '#d29922',
  LRUD: '#3fb950',
};

async function createOrientationViewer(projectId, sampleSize) {
  const data = await api(`/projects/${projectId}/orientation`);
  const list = await api(`/projects/${projectId}/images`);
  const names = list.names || [];
  const size = sampleSize || list.sample_size || [0, 0];

  const byName = new Map();
  for (const row of data.per_image || []) byName.set(row.name, row);
  const excluded = data.excluded_pattern ? new RegExp(data.excluded_pattern, 'i') : null;
  const analysed = (data.per_image || []).length;
  const skipped = excluded ? names.filter((name) => excluded.test(name)).length : 0;

  /* I filtri: un gruppo per orientamento, uno per le analizzate senza esito, uno per le
     escluse dal modulo. Sono partizioni dell'elenco, non viste separate: lo scorrimento
     lavora sempre sulla lista filtrata. */
  const groupOf = (name) => {
    const row = byName.get(name);
    if (row && row.box && row.group) return row.group;
    if (row) return 'NESSUNO';
    return excluded && excluded.test(name) ? 'ESCLUSE' : 'NESSUNO';
  };

  /* "da rivedere" vuol dire una cosa sola: la configurazione che stiamo per consegnare
     non aggancia questa immagine. E' il punteggio della validazione a dirlo, non il dubbio
     che il batch aveva prima — un'immagine che il batch segnava "review" e che il ritaglio
     consegnato trova a 0.99 e' sistemata, non da rivedere. */
  const threshold = Number((data.validation || {}).min_score) || 0.55;
  const inReview = (name) => {
    const row = byName.get(name);
    if (!row) return false;
    return row.score == null || Number(row.score) < threshold;
  };
  const FILTERS = [
    { key: '', label: 'tutte' },
    ...Object.keys(GROUP_COLORS).map((group) => ({ key: group, label: group })),
    { key: 'REVIEW', label: 'da rivedere' },
    { key: 'NESSUNO', label: 'senza orientamento' },
    { key: 'ESCLUSE', label: 'escluse dal modulo' },
  ];
  let filter = '';
  const matches = (name) =>
    !filter || (filter === 'REVIEW' ? inReview(name) : groupOf(name) === filter);

  /* L'ordinamento per confidenza vale dentro il filtro: prima si scegle il gruppo, poi si
     guardano i suoi casi peggiori. Le immagini senza punteggio restano in fondo in ogni caso:
     non hanno una confidenza da confrontare. */
  const scoreOf = (name) => {
    const row = byName.get(name);
    return row && row.score != null ? Number(row.score) : null;
  };
  let sortMode = 'folder';
  const visible = () => {
    const list = names.filter(matches);
    if (sortMode === 'folder') return list;
    const withScore = list.filter((n) => scoreOf(n) != null);
    const without = list.filter((n) => scoreOf(n) == null);
    withScore.sort((a, b) => (sortMode === 'asc'
      ? scoreOf(a) - scoreOf(b)
      : scoreOf(b) - scoreOf(a)));
    return [...withScore, ...without];
  };

  const first = (data.per_image || []).find((row) => row.box);
  let index = Math.max(0, names.indexOf(first ? first.name : names[0]));

  const root = el('div', {});
  const stage = el('div', { class: 'editor-stage' });
  const image = el('img', { alt: '' });
  stage.append(image);
  const overlays = {};
  for (const group of Object.keys(GROUP_COLORS)) {
    const node = el('div', { class: 'editor-box readonly', style: `--box-color:${GROUP_COLORS[group]}` });
    node.style.display = 'none';
    overlays[group] = node;
    stage.append(node);
  }
  const markerNode = el('div', { class: 'editor-box marker-box' });
  markerNode.style.display = 'none';
  /* Le maniglie per stringere il marker. Restano nascoste finche' non si entra in modifica:
     il rettangolo che si vede normalmente e' un risultato, non un comando. */
  const LATI_MANIGLIA = ['nw', 'n', 'ne', 'w', 'e', 'sw', 's', 'se'];
  const maniglieMarker = LATI_MANIGLIA.map((lato) => {
    const h = el('div', { class: `handle handle-${lato}` });
    h.style.display = 'none';
    h.addEventListener('pointerdown', (e) => trascinaMarker(e, lato));
    markerNode.append(h);
    return h;
  });
  markerNode.addEventListener('pointerdown', (e) => {
    if (!modificaMarker || e.target !== markerNode) return;
    trascinaMarker(e, 'move');
  });
  stage.append(markerNode);

  /* La rielaborazione si vede sull'immagine, non in una riga di testo altrove: velo,
     spillo nel punto cliccato, spinner, tappa corrente e barra a cinque segmenti. */
  const clickPin = el('div', { class: 'click-pin' });
  clickPin.style.display = 'none';
  stage.append(clickPin);

  // Il velo copre solo il raffinamento di questa immagine, che dura un attimo: la barra a
  // cinque segmenti serviva quando il click aspettava tutta la cartella. Ora quel lavoro
  // sta nella striscia del background, che non blocca niente.
  const busyLabel = el('div', { class: 'busy-label' });
  const busyOverlay = el('div', { class: 'busy-overlay' },
    el('div', { class: 'busy-spinner' }), busyLabel);
  busyOverlay.style.display = 'none';
  stage.append(busyOverlay);

  const setBusy = (on, text) => {
    busyOverlay.style.display = on ? 'flex' : 'none';
    stage.classList.toggle('busy', on);
    if (text !== undefined) busyLabel.textContent = text;
    if (!on) clickPin.style.display = 'none';
  };

  const caption = el('div', { class: 'hint' });
  const detail = el('div', { class: 'hint' });
  // Il progresso deve restare visibile mentre si guarda l'immagine: nella didascalia
  // sotto la tavola era facile perderlo, soprattutto sulle immagini alte.
  const imageCounter = el('strong', {
    class: 'ov-image-counter',
    title: 'immagine corrente sul totale',
    'aria-live': 'polite',
    'aria-atomic': 'true',
  }, '0 / 0');

  const scaleX = () => image.clientWidth / ((size[0] || image.naturalWidth) || 1);
  const scaleY = () => image.clientHeight / ((size[1] || image.naturalHeight) || 1);

  const place = (node, box) => {
    if (!box) { node.style.display = 'none'; return; }
    node.style.display = 'block';
    node.style.left = `${box.left * scaleX()}px`;
    node.style.top = `${box.top * scaleY()}px`;
    node.style.width = `${(box.right - box.left) * scaleX()}px`;
    node.style.height = `${(box.bottom - box.top) * scaleY()}px`;
  };

  /* Cosa vede la lente: il marker di questa immagine e, dietro, l'envelope del suo
     gruppo — e' il confronto che conta, perche' il marker deve starci dentro. */
  const contestoLente = () => {
    const row = byName.get(names[index]);
    const gruppo = row && row.group;
    const voci = [];
    if (modificaMarker && bozzaMarker) {
      voci.push({ box: bozzaMarker, color: '#ff6040', label: 'marker da stringere' });
    } else if (drawing && bozzaIndicata) {
      // Quello che hai appena disegnato: resta la voce principale, cosi' nella lente si
      // prende per i lati e si aggiusta al pixel prima di confermarlo.
      voci.push({ box: bozzaIndicata, color: '#58a6ff', label: 'marker indicato' });
    } else if (row && row.box) {
      voci.push({ box: row.box, color: GROUP_COLORS[gruppo] || '#40d0ff',
                  label: `marker ${gruppo || ''}`.trim() });
    }
    const busta = gruppo && (data.envelopes || {})[gruppo];
    if (busta) voci.push({ box: busta, color: GROUP_COLORS[gruppo] || '#8b949e',
                           label: `envelope ${gruppo}` });
    /* Cosa inquadrare: il **marker**, sempre. Senza dirlo la lente inquadrava il primo
       riquadro che trovava, e sulle immagini dove il marker non c'e' quello era
       l'envelope: mezzo schermo, cioe' nessun ingrandimento proprio dove serviva di piu'.
       Senza marker si guarda l'envelope, che e' l'unico indizio di dove cercarlo. */
    const mira = (modificaMarker && bozzaMarker) || (drawing && bozzaIndicata)
      || (row && row.box) || busta || null;
    return {
      source: 'orientamento',
      projectId, name: names[index], size: [size[0] || 0, size[1] || 0], boxes: voci,
      focus: mira,
      caption: modificaMarker ? 'stringilo sul glifo'
        : (drawing ? 'tira il rettangolo attorno al marker vero'
          : (row ? `score ${row.score}` : 'nessun marker')),
      // Indicare il marker dalla lente: e' quindici pixel, e sulla pagina principale
      // significa tirare un rettangolo di sette pixel e sbagliarlo.
      onDraw: (modificaMarker || drawing) ? (nuovo) => {
        if (modificaMarker) {
          bozzaMarker = limitaBox(nuovo);
          place(markerNode, bozzaMarker);
          mostraManiglie();
          renderModifica();
        } else {
          indicaMarker(nuovo);
        }
        if (Lente.viva()) Lente.aggiorna(contestoLente());
      } : null,
      // Stringere un marker vuol dire guardarlo da vicino: e' il caso in cui la lente
      // serve di piu', quindi qui si trascina anche da li'.
      onChange: modificaMarker ? (nuovo) => {
        bozzaMarker = limitaBox(nuovo);
        place(markerNode, bozzaMarker);
        renderModifica();
      } : (drawing && bozzaIndicata ? (nuovo) => indicaMarker(nuovo) : null),
    };
  };

  const paint = () => {
    for (const [group, node] of Object.entries(overlays)) place(node, (data.envelopes || {})[group]);
    if (typeof paintLimits === 'function') paintLimits();
    if (typeof markCurrentLimits === 'function') markCurrentLimits();
    // l'elenco a destra e' alto come l'immagine: si scorre lui, non la pagina
    if (typeof listBox !== 'undefined' && stage.clientHeight > 120) {
      limitsSide.style.maxHeight =
        `${Math.max(150, Math.round(stage.clientHeight * 0.46))}px`;
      listBox.style.maxHeight =
        `${Math.max(160, stage.clientHeight - limitsSide.offsetHeight - 8)}px`;
    }
    const row = byName.get(names[index]);
    // In modifica comanda la bozza: ridisegnare il box del modulo cancellerebbe la
    // stretta appena fatta.
    place(markerNode, modificaMarker ? bozzaMarker : (row && row.box));
    mostraManiglie();
    if (Lente.viva()) Lente.aggiorna(contestoLente());
    detail.innerHTML = '';
    if (row) {
      detail.append(el('span', {},
        `marker in questa immagine: gruppo ${row.group} · score ${row.score} · ` +
        `template ${row.template} · ricerca ${row.scope}${row.status && row.status !== 'ok' ? ' · ' + row.status : ''}` +
        (row.corrected ? ' · CORRETTA A MANO' : '')));
    } else if (excluded && excluded.test(names[index] || '')) {
      detail.append(el('span', {},
        'schermata proibita: il modulo non ci cerca il marker (esclusa per nome), ' +
        'si vedono solo i quattro envelope'));
    } else {
      detail.append(el('span', {},
        'nessuna detection per questa immagine: si vedono solo i quattro envelope'));
    }
  };

  let compareRefresh = null;
  const show = () => {
    const list = visible();
    if (!list.length) {
      imageCounter.textContent = '0 / 0';
      imageCounter.title = 'nessuna immagine nel gruppo selezionato';
      caption.textContent = 'nessuna immagine in questo gruppo';
      return;
    }
    const position = Math.max(0, list.indexOf(names[index]));
    imageCounter.textContent = `${position + 1} / ${list.length}`;
    imageCounter.title = filter
      ? `immagine ${position + 1} di ${list.length} nel filtro selezionato (${names.length} totali)`
      : `immagine ${position + 1} di ${list.length}`;
    const score = scoreOf(names[index]);
    caption.innerHTML = '';
    caption.append(
      `immagine ${position + 1} di ${list.length}` +
      (filter
        ? ` nel gruppo ${({ NESSUNO: 'senza orientamento', ESCLUSE: 'escluse', REVIEW: 'da rivedere' })[filter] || filter}`
        : '') +
      ` — ${names[index] || ''} · `,
      el('strong', { style: `color:${score == null ? 'var(--muted)' : score >= 0.8 ? 'var(--ok)' : score >= 0.62 ? 'var(--warn)' : 'var(--err)'}` },
        score == null ? 'nessuna confidenza' : `confidenza ${score.toFixed(4)}`),
    );
    image.src = `/api/projects/${projectId}/image?name=${encodeURIComponent(names[index])}&w=980`;
    paint();
    paintCrop();
    if (compareRefresh) compareRefresh();
  };
  const step = (delta) => {
    const list = visible();
    if (!list.length) return;
    const current = list.indexOf(names[index]);
    const next = ((current < 0 ? 0 : current) + delta + list.length) % list.length;
    index = names.indexOf(list[next]);
    show();
  };
  const applyFilter = (key) => {
    filter = key;
    const list = visible();
    if (list.length && !list.includes(names[index])) index = names.indexOf(list[0]);
    for (const [button, buttonKey] of filterButtons) {
      button.className = 'chip' + (buttonKey === filter ? ' on' : '');
    }
    show();
    renderList();
  };
  const filterButtons = [];
  const stepFound = (delta) => {
    if (!names.length) return;
    for (let offset = 1; offset <= names.length; offset += 1) {
      const candidate = (index + delta * offset + names.length * offset) % names.length;
      const row = byName.get(names[candidate]);
      if (row && row.box) { index = candidate; show(); return; }
    }
    toast('nessun altra immagine con marker trovato', true);
  };

  /* --- correzione: si segnala l'immagine sbagliata e si clicca sul marker vero ---

     Due tempi. Il click risponde subito (posizione precisa, gruppo, envelope) e si resta
     in correzione, cosi' le immagini in revisione si sistemano una dopo l'altra senza
     attese. Il riprovare i ritagli su tutta la cartella corre in background, e si vede
     nella striscia sotto la barra: se correggi ancora, non si accodano N rielaborazioni,
     ne riparte una con tutte le correzioni dentro. */
  let correcting = false;
  let pending = null;
  const fixNode = el('div', { class: 'editor-box fix-box' });
  fixNode.style.display = 'none';
  stage.append(fixNode);
  const fixPanel = el('div', { class: 'card', style: 'display:none' });

  /* Il click si manda in pixel dello schermo, con la dimensione con cui l'immagine e'
     mostrata: la conversione la fa il server, che conosce la dimensione vera del file.
     Cosi' non c'e' nessuna assunzione da sbagliare qui. */
  const toClickPayload = (event) => {
    const rect = image.getBoundingClientRect();
    return {
      cx: Math.round(event.clientX - rect.left),
      cy: Math.round(event.clientY - rect.top),
      display_width: Math.round(rect.width),
      display_height: Math.round(rect.height),
    };
  };

  /* La lente segue il puntatore. Con due schermi si guarda l'immagine grande e la lente
     sta di lato: passando sopra a un marker lo si vuole vedere ingrandito li', subito,
     senza prima dire alla lente dove andare. */
  const seguiConLaLente = (event) => {
    if (!Lente.viva() || !Lente.segueOra()) return;
    const r = image.getBoundingClientRect();
    if (!r.width || !r.height) return;
    const sx = (size[0] || image.naturalWidth || 0) / r.width;
    const sy = (size[1] || image.naturalHeight || 0) / r.height;
    if (!sx || !sy) return;
    Lente.segui((event.clientX - r.left) * sx, (event.clientY - r.top) * sy);
  };
  stage.addEventListener('pointermove', seguiConLaLente);

  const setCorrecting = (on) => {
    correcting = on;
    stage.classList.toggle('picking', on);
    fixButton.textContent = on ? 'Annulla (non correggere)' : 'Correggi il marker';
    fixButton.className = on ? 'ghost' : '';
    pickHint.textContent = on
      ? '← clicca sul marker vero nell\'immagine qui sotto'
      : '';
    if (!on) {
      fixNode.style.display = 'none';
      pending = null;
    }
  };

  const pct = (v) => (v == null ? '—' : `${Math.round(v * 100)}%`);
  const colorFor = (v) => (v == null ? 'var(--muted)'
    : v >= 0.95 ? 'var(--ok)' : v >= 0.80 ? 'var(--warn)' : 'var(--err)');

  /* --- la striscia della rielaborazione in background --- */
  const bgLabel = el('span', { class: 'hint' }, 'nessuna rielaborazione in corso');
  const bgFill = el('div', { class: 'bg-fill' });
  const bgTrack = el('div', { class: 'bg-track' }, bgFill);
  const bgDot = el('span', { class: 'bg-dot' });
  const bgReport = el('span', { class: 'hint' });
  const bgBar = el('div', { class: 'bg-bar' }, bgDot, bgLabel, bgTrack, bgReport);

  let bgTimer = null;
  let bgRuns = null;
  const paintBackground = (state) => {
    const running = state.state === 'running';
    bgBar.classList.toggle('running', running);
    bgDot.className = 'bg-dot' + (running ? ' spin' : '');
    const queued = state.queued || 0;
    bgLabel.textContent = running
      ? (state.stage || 'rielaborazione della cartella') +
        (queued ? ` · ${queued} correzion${queued === 1 ? 'e' : 'i'} in coda per il giro dopo` : '')
      : 'nessuna rielaborazione in corso';
    const total = state.total || 0;
    bgTrack.style.display = running && total ? 'block' : 'none';
    bgFill.style.width = total ? `${Math.round(100 * (state.done || 0) / total)}%` : '0%';
    if (running && total) bgLabel.textContent += ` — ${state.done}/${total}`;
    const last = state.last;
    bgReport.textContent = state.error
      ? `ultimo passaggio in errore: ${state.error}`
      : last
        ? `ultimo passaggio: ${last.recovered_count} migliorate` +
          (last.crossed_count ? `, ${last.crossed_count} sopra soglia` : '') +
          ` · ${last.review_after} da rivedere · ${last.hints} suggerimenti in uso`
        : '';
    bgReport.style.color = state.error ? 'var(--err)' : 'var(--muted)';
  };

  const watchBackground = () => {
    if (bgTimer) return;
    const tick = async () => {
      if (!document.body.contains(root)) { clearInterval(bgTimer); bgTimer = null; return; }
      let state;
      try {
        state = await api(`/projects/${projectId}/orientation/consolidation`);
      } catch (error) { return; }
      paintBackground(state);
      // il contatore dei passaggi dice quando c'e' qualcosa di nuovo da rileggere
      if (bgRuns != null && (state.runs || 0) > bgRuns) {
        await refreshData();
        if (state.last) renderConsolidation(state.last);
        if (state.last && state.last.recovered_count) {
          toast(`rielaborazione finita: ${state.last.recovered_count} immagini migliorate` +
            (state.last.crossed_count ? `, ${state.last.crossed_count} risalite sopra soglia` : ''));
        }
      }
      bgRuns = state.runs || 0;
      if (state.state !== 'running') { clearInterval(bgTimer); bgTimer = null; }
    };
    bgRuns = bgRuns == null ? 0 : bgRuns;
    bgTimer = setInterval(tick, 1200);
    tick();
  };

  /* Rilettura in posto: i punteggi, i box e gli envelope si aggiornano senza ricostruire
     la pagina, cosi' non si perde ne' il filtro ne' l'immagine su cui si sta lavorando. */
  const refreshData = async () => {
    const fresh = await api(`/projects/${projectId}/orientation`);
    data.envelopes = fresh.envelopes;
    data.corrections_detail = fresh.corrections_detail;
    data.validation = fresh.validation;
    data.folder_template = fresh.folder_template;
    data.marker_warning = fresh.marker_warning;
    data.marker_override = fresh.marker_override;
    byName.clear();
    for (const row of fresh.per_image || []) byName.set(row.name, row);
    show();
    renderList();
    renderLegend();
    paintWarning();
    renderConsegnato();
    loadLimits();
    renderCorrections(fresh.corrections_detail || []);
    renderPosizioni();
  };

  /* Cosa e' cambiato per QUESTA immagine: si legge subito dopo il click. */
  const renderQuick = (result) => {
    fixPanel.innerHTML = '';
    fixPanel.style.display = 'block';
    const r = result.refined;
    place(fixNode, r.box);
    fixPanel.append(el('h3', { style: 'margin-top:0' },
      `${names[index] ? names[index].slice(-46) : 'immagine'} — corretta`));
    const before = r.score_before;
    for (const [k, v] of [
      ['posizione salvata', `${r.box.top}|${r.box.left}|${r.box.bottom}|${r.box.right}`],
      ['confidenza', `${r.score}` + (before != null
        ? ` (prima ${before}${r.score > before ? ', migliorata' : ''})` : '')],
      ['sovrapposizione scelta', `${r.template}` +
        (r.template_size ? ` ${r.template_size[0]}x${r.template_size[1]}` : '') +
        (r.scale != null ? ` a scala ${r.scale}` : '') +
        (r.overlaps_tried ? ` · migliore di ${r.overlaps_tried} provate` : '')],
      ['scostamento dal click', `dx ${r.offset_from_click.dx} · dy ${r.offset_from_click.dy} px`],
      ['gruppo', r.group_changes ? `${r.group_before} -> ${r.group}` : `${r.group} (invariato)`],
      ['correzioni nel progetto', `${result.corrections}`],
    ]) fixPanel.append(el('div', { class: 'kv' }, el('span', {}, k), el('span', {}, String(v))));
    fixPanel.append(el('p', { class: 'hint' },
      'il ritaglio di questa correzione viene provato su tutte le immagini dalla ' +
      'rielaborazione in background: puoi continuare a correggere nel frattempo.'));
    fixPanel.append(el('div', { class: 'row' },
      el('button', {
        class: 'ghost',
        onclick: async () => {
          try {
            await api(`/projects/${projectId}/orientation/correction/delete`,
              { body: { name: result.name } });
            toast('correzione annullata');
            await refreshData();
            await api(`/projects/${projectId}/orientation/reprocess`, { body: {} });
            watchBackground();
          } catch (error) { toast(error.message, true); }
        },
      }, 'Annulla questa correzione')));
  };

  /* Cosa e' cambiato per la CARTELLA: arriva quando il passaggio in background finisce. */
  const renderConsolidation = (last) => {
    const card = el('div', { class: 'card' });
    card.append(el('h3', { style: 'margin-top:0' },
      `rielaborazione finita — ${last.corrections} correzioni, ${last.hints} suggerimenti in uso`));
    for (const [k, v] of [
      ['in revisione prima / dopo',
        `${last.review_before} -> ${last.review_after} (soglia ${last.threshold})`],
      ['migliorate dai suggerimenti', last.recovered_count
        ? `${last.recovered_count} immagini` +
          (last.crossed_count ? `, di cui ${last.crossed_count} risalite sopra soglia` : '')
        : 'nessuna: le altre erano gia\' agganciate meglio'],
      ['peggiorate', last.worsened_count ? `${last.worsened_count} immagini` : 'nessuna'],
      ['copertura della cartella', pct(last.coverage)],
      ['ritaglio consegnato', last.promoted
        ? `promosso ${last.promoted} (${(last.template_size || []).join('x')} px)`
        : `resta quello di prima (${(last.template_size || []).join('x')} px)`],
    ]) card.append(el('div', { class: 'kv' }, el('span', {}, k), el('span', {}, String(v))));
    for (const item of (last.recovered || [])) {
      card.append(el('div', { class: 'hint', style: 'margin:2px 0' },
        `${item.before} -> ${item.after} · ${item.name.slice(-52)}`));
    }
    const table = el('table', { class: 'lines' });
    table.append(el('tr', {},
      el('td', { class: 'name' }, 'gruppo'),
      el('td', { class: 'name' }, 'copertura'),
      el('td', { class: 'name' }, 'mediana'),
      el('td', { class: 'name' }, 'minimo')));
    for (const g of Object.keys(GROUP_COLORS)) {
      const c = (last.coverage_by_group || {})[g];
      table.append(el('tr', {},
        el('td', { class: 'name', style: `color:${GROUP_COLORS[g]}` }, g),
        el('td', { class: 'val', style: `color:${colorFor(c && c.coverage)}` }, c ? pct(c.coverage) : '—'),
        el('td', { class: 'val' }, c && c.median != null ? String(c.median) : '—'),
        el('td', { class: 'val' }, c && c.min != null ? String(c.min) : '—')));
    }
    card.append(table);
    bgReportHost.innerHTML = '';
    bgReportHost.append(card);
    bgTag.textContent = last.recovered_count
      ? `${last.recovered_count} migliorate` : `${last.review_after} da rivedere`;
  };
  const bgReportHost = el('div', {},
    el('p', { class: 'hint' }, 'nessuna rielaborazione ancora in questa sessione'));
  const bgTag = el('span', { class: 'ov-fold-tag' });

  let saving = false;
  let busyName = null;
  stage.addEventListener('click', async (event) => {
    if (drawing) return;
    if (!correcting) {
      toast('per correggere: premi «Correggi il marker», poi clicca sul marker vero');
      return;
    }
    if (saving) return;
    saving = true;
    const target = names[index];
    const click = toClickPayload(event);
    // lo spillo resta dove hai cliccato: si vede su cosa sto lavorando
    clickPin.style.left = `${click.cx}px`;
    clickPin.style.top = `${click.cy}px`;
    clickPin.style.display = 'block';
    // la prossima da correggere si decide prima: dopo il salvataggio questa esce dal filtro
    const list = visible();
    const after = list[(list.indexOf(target) + 1 + list.length) % (list.length || 1)];
    setBusy(true, 'riposiziono il marker su questa immagine');
    busyName = target;
    renderList();
    try {
      const result = await api(`/projects/${projectId}/orientation/quick_fix`, {
        body: { name: target, ...click, window: raggio() },
      });
      busyName = null;
      // la riga diventa "corretta" con i valori nuovi, senza rileggere tutto dal server
      const row = byName.get(target) || { name: target };
      byName.set(target, {
        ...row,
        group: result.refined.group,
        box: result.refined.box,
        score: result.refined.score,
        status: 'corrected',
        corrected: true,
        template: result.refined.template,
        scope: 'click',
      });
      // sostituzione, non fusione: un gruppo che si stringe deve stringersi, e uno che
      // resta senza marker deve sparire dal disegno
      data.envelopes = result.groups || {};
      setBusy(false);
      renderQuick({ ...result, name: target });
      markerNode.classList.add('flash');
      setTimeout(() => markerNode.classList.remove('flash'), 1200);
      paintBackground(result.background || { state: 'running' });
      watchBackground();
      // in serie: si resta armati e si salta alla prossima, cosi' le immagini in revisione
      // si sistemano di seguito senza aspettare nulla
      if (serieToggle.checked) {
        serieCount += 1;
        if (serieCount >= SERIE_MAX) {
          closeSerie(`tetto di ${SERIE_MAX}`);
        } else {
          keepSerie();
          if (after && after !== target) index = names.indexOf(after);
          show();
          stage.classList.add('picking');
          pickHint.textContent = '← clicca sul marker vero nell\'immagine qui sotto';
        }
      } else {
        setCorrecting(false);
        pickHint.textContent = 'per correggerne un\'altra premi di nuovo Correggi il marker';
      }
      renderList();
      toast(`corretta nel gruppo ${result.refined.group} · confidenza ${result.refined.score}`);
    } catch (error) {
      busyName = null;
      setBusy(false);
      renderList();
      fixPanel.style.display = 'block';
      fixPanel.innerHTML = '';
      fixPanel.append(el('p', { class: 'hint', style: 'color:var(--err)' }, error.message));
      setCorrecting(false);
    } finally {
      saving = false;
    }
  });

  const fixButton = el('button', {}, 'Correggi il marker');
  const pickHint = el('span', { class: 'hint' });
  fixButton.addEventListener('click', () => {
    if (correcting) { closeSerie(''); return; }
    serieCount = 0;
    setCorrecting(true);
    if (serieToggle.checked) keepSerie();
  });
  /* --- il marker trovato e' sbagliato: si indica quello giusto e il modulo riparte ---

     Correggere immagine per immagine non serve quando il glifo cercato e' il glifo
     sbagliato: su una cartella GE il modulo si e' agganciato alla scritta fissa
     dell'intestazione, che non si muove mai, e ha messo tutte le immagini in un gruppo.
     Qui l'utente disegna il rettangolo attorno al marker vero e da quel ritaglio si
     rifanno posizioni, gruppi ed envelope, senza passare dalla banca. */
  /* --- stringere il marker ---

     Il rettangolo che il modulo ha trovato e' spesso piu' largo del glifo: ci sta dentro
     un pezzo di cornice, o l'ombra della scritta accanto. Stringerlo non e' correggere una
     posizione sbagliata - quella c'e' gia' - e' dare al modulo un ritaglio che aderisce, e
     un ritaglio che aderisce si ritrova meglio su tutte le altre immagini.

     E' lo stesso gesto del «marker sbagliato», ma partendo da quello che c'e' invece che
     dal foglio bianco: stessa strada, `marker_override`, che ritaglia, ricerca su tutte le
     immagini e rifa' gruppi ed envelope. Quindi diventa davvero **il** marker della
     cartella, anche nei blocchi #12/#16 del file. */
  let modificaMarker = false;
  let bozzaMarker = null;
  let partenzaMarker = null;
  const modificaPanel = el('div', { class: 'card', style: 'display:none' });
  const modificaButton = el('button', { class: 'ghost' }, 'Stringi il marker');

  const limitaBox = (box) => {
    const [w, h] = [size[0] || image.naturalWidth || 0, size[1] || image.naturalHeight || 0];
    const out = { ...box };
    out.left = Math.max(0, Math.round(out.left));
    out.top = Math.max(0, Math.round(out.top));
    out.right = Math.max(out.left + 4, Math.round(out.right));
    out.bottom = Math.max(out.top + 4, Math.round(out.bottom));
    if (w) out.right = Math.min(out.right, w);
    if (h) out.bottom = Math.min(out.bottom, h);
    return out;
  };

  function trascinaMarker(event, lato) {
    if (!modificaMarker || !bozzaMarker) return;
    event.preventDefault();
    event.stopPropagation();
    const partenza = { x: event.clientX, y: event.clientY };
    const box = { ...bozzaMarker };
    const sx = scaleX() || 1;
    const sy = scaleY() || 1;
    const bersaglio = event.currentTarget;
    bersaglio.setPointerCapture(event.pointerId);
    const muovi = (e) => {
      const dx = Math.round((e.clientX - partenza.x) / sx);
      const dy = Math.round((e.clientY - partenza.y) / sy);
      const n = { ...box };
      if (lato === 'move') {
        // Si sposta intero: i lati vanno limitati insieme, se no contro un bordo si
        // schiaccia invece di fermarsi.
        const largo = box.right - box.left;
        const alto = box.bottom - box.top;
        n.left = box.left + dx; n.top = box.top + dy;
        n.right = n.left + largo; n.bottom = n.top + alto;
      } else {
        if (lato.includes('w')) n.left = Math.min(box.left + dx, box.right - 4);
        if (lato.includes('e')) n.right = Math.max(box.right + dx, box.left + 4);
        if (lato.includes('n')) n.top = Math.min(box.top + dy, box.bottom - 4);
        if (lato.includes('s')) n.bottom = Math.max(box.bottom + dy, box.top + 4);
      }
      bozzaMarker = limitaBox(n);
      place(markerNode, bozzaMarker);
      mostraManiglie();
      renderModifica();
      if (Lente.viva()) Lente.aggiorna(contestoLente());
    };
    const molla = () => {
      bersaglio.removeEventListener('pointermove', muovi);
      bersaglio.removeEventListener('pointerup', molla);
      bersaglio.removeEventListener('pointercancel', molla);
    };
    bersaglio.addEventListener('pointermove', muovi);
    bersaglio.addEventListener('pointerup', molla);
    bersaglio.addEventListener('pointercancel', molla);
  }

  const mostraManiglie = () => {
    for (const h of maniglieMarker) h.style.display = modificaMarker ? 'block' : 'none';
    markerNode.classList.toggle('in-modifica', modificaMarker);
  };

  /* Entrando in modifica il software ci prova per primo: toglie il bordo di sfondo e
     lascia il glifo. Non e' una decisione presa al posto suo - il riquadro di partenza
     resta scritto, e basta un trascinamento per rimetterlo come si vuole - e' solo che
     togliere il nero attorno e' un conto, non un giudizio, e farglielo fare a mano ogni
     volta sarebbe farle ripetere quel conto. */
  let strettaAutomatica = null;
  const setModifica = async (on) => {
    const row = byName.get(names[index]);
    if (on && !(row && row.box)) { toast('qui non c\'e\' un marker da stringere', true); return; }
    modificaMarker = on;
    partenzaMarker = on ? { ...row.box } : null;
    bozzaMarker = on ? { ...row.box } : null;
    strettaAutomatica = null;
    modificaButton.textContent = on ? 'Annulla' : 'Stringi il marker';
    modificaPanel.style.display = on ? 'block' : 'none';
    if (!on) paint(); else place(markerNode, bozzaMarker);
    mostraManiglie();
    renderModifica();
    if (!on) { if (Lente.viva()) Lente.aggiorna(contestoLente()); return; }
    // La risposta puo' arrivare dopo che si e' gia' cambiata immagine o usciti: in quel
    // caso e' la risposta a una domanda che non si fa piu', e va lasciata cadere.
    const chiesta = names[index];
    try {
      const b = partenzaMarker;
      const esito = await api(`/projects/${projectId}/orientation/marker_tight`
        + `?name=${encodeURIComponent(chiesta)}`
        + `&box=${b.top}|${b.left}|${b.bottom}|${b.right}`);
      if (names[index] !== chiesta || !modificaMarker) return;
      if (esito.changed) {
        strettaAutomatica = esito;
        bozzaMarker = limitaBox(esito.box);
        place(markerNode, bozzaMarker);
        mostraManiglie();
      } else {
        strettaAutomatica = { changed: false, reason: esito.reason || '' };
      }
    } catch (errore) {
      if (names[index] !== chiesta || !modificaMarker) return;
      strettaAutomatica = { changed: false, reason: errore.message };
    }
    renderModifica();
    if (Lente.viva()) Lente.aggiorna(contestoLente());
  };
  modificaButton.addEventListener('click', () => setModifica(!modificaMarker));

  const renderModifica = () => {
    if (!modificaMarker || !bozzaMarker) { modificaPanel.innerHTML = ''; return; }
    const b = bozzaMarker;
    const largo = b.right - b.left;
    const alto = b.bottom - b.top;
    const partito = partenzaMarker || (byName.get(names[index]) || {}).box || b;
    modificaPanel.innerHTML = '';
    modificaPanel.append(el('h3', { style: 'margin-top:0' }, 'stringi il marker'));
    modificaPanel.append(el('p', { class: 'hint' },
      'trascina le maniglie sull\'immagine o dentro la lente, oppure ridisegnalo tirando '
      + 'dentro la lente. Quando va bene, questo '
      + 'ritaglio diventa il marker cercato su tutte le immagini della cartella, e quello '
      + 'che finisce nei blocchi #12/#16 del file.'));
    modificaPanel.append(el('div', { class: 'kv' }, el('span', {}, 'ora'),
      el('span', {}, `${b.top}|${b.left}|${b.bottom}|${b.right} — ${largo}x${alto} px`)));
    modificaPanel.append(el('div', { class: 'kv' }, el('span', {}, 'partiva da'),
      el('span', {}, `${partito.right - partito.left}x${partito.bottom - partito.top} px`)));
    if (strettaAutomatica && strettaAutomatica.changed) {
      const t = strettaAutomatica.trimmed || {};
      const rimetti = el('button', { class: 'ghost sq2', style: 'margin-left:8px' },
        'rimetti com\'era');
      rimetti.addEventListener('click', () => {
        bozzaMarker = { ...partito };
        place(markerNode, bozzaMarker);
        mostraManiglie();
        renderModifica();
        if (Lente.viva()) Lente.aggiorna(contestoLente());
      });
      modificaPanel.append(el('div', { class: 'depth-prestito' },
        `il bordo di sfondo l'ho tolto io: ${t.top || 0} px sopra, ${t.bottom || 0} sotto, `
        + `${t.left || 0} a sinistra, ${t.right || 0} a destra. `
        + 'Correggi con le maniglie se non ti torna, oppure conferma.', rimetti));
    } else if (strettaAutomatica && strettaAutomatica.reason) {
      modificaPanel.append(el('div', { class: 'hint' },
        `da togliere non ho trovato niente: ${strettaAutomatica.reason}`));
    }
    if (largo > 45 || alto > 45) {
      modificaPanel.append(el('p', { class: 'hint', style: 'color:var(--warn)' },
        `i ritagli orientation_*.png delle configurazioni storiche vanno da 16x18 a 30x31: `
        + `questo e' ${largo}x${alto}. Piu' stretto sul glifo aderisce meglio.`));
    }
    modificaPanel.append(el('img', {
      src: `/api/projects/${projectId}/orientation/crop?name=${encodeURIComponent(names[index])}`
        + `&scale=6&pad=1&box=${b.top}|${b.left}|${b.bottom}|${b.right}`,
      style: 'border:1px solid var(--line);image-rendering:pixelated;margin:6px 0',
    }));
    const stato = el('span', { class: 'hint' });
    const vai = el('button', {}, 'Usa questo marker su tutta la cartella');
    vai.addEventListener('click', async () => {
      vai.disabled = true;
      setBusy(true, 'cerco il marker stretto su tutte le immagini');
      try {
        const started = await api(`/projects/${projectId}/orientation/marker_override`,
          { body: { name: names[index], box: bozzaMarker, narrow: true } });
        const job = await pollJob(started.job_id,
          { set textContent(v) { busyLabel.textContent = v; stato.textContent = v; } });
        setBusy(false);
        setModifica(false);
        await refreshData();
        renderOverrideResult(job.result);
        toast(`marker stretto: ${job.result.matched} immagini su ${job.result.images}, `
          + `gruppi ${(job.result.groups_found || []).join(' ') || 'nessuno'}`);
      } catch (errore) {
        setBusy(false);
        stato.textContent = '';
        modificaPanel.append(el('p', { class: 'hint', style: 'color:var(--err)' }, errore.message));
        vai.disabled = false;
      }
    });
    modificaPanel.append(el('div', { class: 'row' }, vai,
      el('button', { class: 'ghost', onclick: () => setModifica(false) }, 'Annulla'), stato));
  };

  let drawing = false;
  let drawStart = null;
  /* Il rettangolo appena indicato resta una **bozza**: si aggiusta con le maniglie e con
     le frecce - meglio dentro la lente, dove il glifo si vede - e il pannello di conferma
     si aggiorna mentre lo muovi. Prima il disegno era un colpo solo: se veniva storto
     bisognava ricominciare da capo. */
  let bozzaIndicata = null;
  let attesaPannello = null;
  const indicaMarker = (box) => {
    bozzaIndicata = limitaBox(box);
    place(drawNode, bozzaIndicata);
    drawNode.style.display = 'block';
    // Il pannello ricarica l'anteprima del ritaglio: rifarlo a ogni pixel del
    // trascinamento sarebbe una richiesta al server per pixel.
    clearTimeout(attesaPannello);
    attesaPannello = setTimeout(() => proponiMarkerBox(bozzaIndicata), 160);
    // La lente deve vedere la misura nuova: il suo riquadro si muove da solo mentre lo
    // trascini, ma la riga in alto - «24 x 24 px · top…» - la scrive dal contesto.
    if (Lente.viva()) Lente.aggiorna(contestoLente());
  };
  const drawNode = el('div', { class: 'editor-box draw-box' });
  drawNode.style.display = 'none';
  stage.append(drawNode);
  const wrongPanel = el('div', { class: 'card', style: 'display:none' });
  const wrongButton = el('button', { class: 'ghost' }, 'Il marker trovato e\' sbagliato');
  const wrongHint = el('span', { class: 'hint' });

  const setDrawing = (on) => {
    drawing = on;
    if (on) setCorrecting(false);
    stage.classList.toggle('drawing', on);
    wrongButton.textContent = on ? 'Annulla (non indicare)' : 'Il marker trovato e\' sbagliato';
    wrongHint.textContent = on
      ? '← trascina un rettangolo attorno al marker vero, stretto sul glifo — '
        + 'meglio ancora dentro la lente, dove e\' ingrandito'
      : '';
    if (Lente.viva()) Lente.aggiorna(contestoLente());
    if (!on) {
      drawNode.style.display = 'none';
      drawStart = null;
      bozzaIndicata = null;
      clearTimeout(attesaPannello);
      wrongPanel.style.display = 'none';
    }
  };
  wrongButton.addEventListener('click', () => setDrawing(!drawing));

  const drawAt = (a, b) => {
    drawNode.style.display = 'block';
    drawNode.style.left = `${Math.min(a.x, b.x)}px`;
    drawNode.style.top = `${Math.min(a.y, b.y)}px`;
    drawNode.style.width = `${Math.abs(a.x - b.x)}px`;
    drawNode.style.height = `${Math.abs(a.y - b.y)}px`;
  };
  const local = (event) => {
    const r = image.getBoundingClientRect();
    return { x: Math.round(event.clientX - r.left), y: Math.round(event.clientY - r.top),
             w: Math.round(r.width), h: Math.round(r.height) };
  };

  stage.addEventListener('mousedown', (event) => {
    if (!drawing) return;
    event.preventDefault();
    drawStart = local(event);
    drawAt(drawStart, drawStart);
  });
  stage.addEventListener('mousemove', (event) => {
    if (!drawing || !drawStart) return;
    drawAt(drawStart, local(event));
  });
  stage.addEventListener('mouseup', (event) => {
    if (!drawing || !drawStart) return;
    const end = local(event);
    const a = drawStart;
    drawStart = null;
    if (Math.abs(a.x - end.x) < 4 || Math.abs(a.y - end.y) < 4) {
      toast('rettangolo troppo piccolo: trascina attorno al marker', true);
      drawNode.style.display = 'none';
      return;
    }
    proponiMarker(a, end);
    if (Lente.viva()) Lente.aggiorna(contestoLente());
  });

  /* Prima di rilanciare si vede cosa si sta per usare: il ritaglio ingrandito, la sua
     dimensione, e il confronto con le misure dei ritagli storici. */
  const proponiMarker = (a, b) => {
    // Dal rettangolo tirato sullo schermo a quello dell'immagine: la stessa conversione
    // che serve all'anteprima, cosi' il pannello e il server guardano lo stesso riquadro.
    const scale = (size[0] || image.naturalWidth) / (image.clientWidth || 1);
    indicaMarker({
      left: Math.round(Math.min(a.x, b.x) * scale), top: Math.round(Math.min(a.y, b.y) * scale),
      right: Math.round(Math.max(a.x, b.x) * scale), bottom: Math.round(Math.max(a.y, b.y) * scale),
    });
  };

  /* Il pannello di conferma, dato il rettangolo in pixel dell'immagine. Ci si arriva da due
     strade: tirandolo sull'immagine grande, o - meglio - dentro la lente, dove un marker di
     quindici pixel si vede per quello che e'. */
  const proponiMarkerBox = (box) => {
    const target = names[index];
    wrongPanel.innerHTML = '';
    wrongPanel.style.display = 'block';
    wrongPanel.append(el('h3', { style: 'margin-top:0' }, 'marker indicato a mano'));
    const larg = box.right - box.left;
    const alt = box.bottom - box.top;
    const anteprima = el('img', {
      src: `/api/projects/${projectId}/image?name=${encodeURIComponent(target)}` +
        `&w=980&box=${box.top}|${box.left}|${box.bottom}|${box.right}`,
      style: 'max-width:320px;border:1px solid var(--line)',
    });
    wrongPanel.append(el('div', { class: 'kv' }, el('span', {}, 'immagine'),
      el('span', {}, target.slice(-52))));
    wrongPanel.append(el('div', { class: 'kv' },
      el('span', {}, 'rettangolo (px immagine, stima)'),
      el('span', {}, `${box.top}|${box.left}|${box.bottom}|${box.right} — ${larg}x${alt}`)));
    wrongPanel.append(el('p', { class: 'hint' },
      'si aggiusta ancora: maniglie e frecce, o un altro trascinamento - anche dentro la '
      + 'lente, dove il glifo si vede ingrandito. Qui sotto vedi il ritaglio come sara\'.'));
    if (larg > 45 || alt > 45) {
      wrongPanel.append(el('p', { class: 'hint', style: 'color:var(--warn)' },
        `i ritagli orientation_*.png delle configurazioni storiche vanno da 16x18 a 30x31: ` +
        `questo e' ${larg}x${alt}. Piu' stretto sul glifo aderisce meglio.`));
    }
    wrongPanel.append(anteprima);
    const stato = el('span', { class: 'hint' });
    const vai = el('button', {}, 'Rilancia il modulo con questo marker');
    vai.addEventListener('click', async () => {
      vai.disabled = true;
      setBusy(true, 'cerco il marker che hai indicato su tutte le immagini');
      try {
        const started = await api(`/projects/${projectId}/orientation/marker_override`, {
          body: { name: target, box },
        });
        const job = await pollJob(started.job_id,
          { set textContent(value) { busyLabel.textContent = value; stato.textContent = value; } });
        setBusy(false);
        setDrawing(false);
        await refreshData();
        renderOverrideResult(job.result);
        toast(`marker rilanciato: ${job.result.matched} immagini su ${job.result.images}, ` +
          `gruppi ${(job.result.groups_found || []).join(' ') || 'nessuno'}`);
      } catch (error) {
        setBusy(false);
        stato.textContent = '';
        wrongPanel.append(el('p', { class: 'hint', style: 'color:var(--err)' }, error.message));
        vai.disabled = false;
      }
    });
    wrongPanel.append(el('div', { class: 'row' }, vai,
      el('button', { class: 'ghost', onclick: () => setDrawing(false) }, 'Annulla'), stato));
    const perse = (data.corrections_detail || []).length;
    if (perse) {
      wrongPanel.append(el('p', { class: 'hint', style: 'color:var(--warn)' },
        `attenzione: le ${perse} correzioni fatte finora indicavano dov'era il glifo vecchio ` +
        `e verranno scartate, perche' non stiamo piu' cercando quel glifo.`));
    }
  };

  const renderOverrideResult = (r) => {
    wrongPanel.innerHTML = '';
    wrongPanel.style.display = 'block';
    wrongPanel.append(el('h3', { style: 'margin-top:0' }, 'modulo rilanciato col tuo marker'));
    for (const [k, v] of [
      ['ritaglio usato', `${((r.crop || {}).size || []).join('x')} px`],
      ['immagini agganciate', `${r.matched} su ${r.images}`],
      ['gruppi trovati', (r.groups_found || []).join(' ') || 'nessuno'],
      ['gruppi mancanti', (r.missing_groups || []).join(' ') || 'nessuno'],
      ['immagini per gruppo', Object.entries(r.by_group || {})
        .map(([g, n]) => `${g}: ${n}`).join(' · ') || '—'],
      ['copertura della cartella', pct(r.coverage)],
      ['correzioni scartate', r.corrections_dropped
        ? `${r.corrections_dropped} (riferite al glifo vecchio)`
        : (r.corrections_kept
          ? `nessuna: ${r.corrections_kept} tenute e spostate col ritaglio`
          : 'nessuna')],
    ]) wrongPanel.append(el('div', { class: 'kv' }, el('span', {}, k), el('span', {}, String(v))));
    if (r.legacy_size_hint) {
      wrongPanel.append(el('p', { class: 'hint', style: 'color:var(--warn)' }, r.legacy_size_hint));
    }
    if ((r.warning || {}).static) {
      wrongPanel.append(el('p', { class: 'hint', style: 'color:var(--err)' },
        `${r.warning.message} — ${(r.warning.reasons || []).join('; ')}`));
    }
  };

  /* La diagnosi automatica: un marker che non si muove non e' un marker di orientamento. */
  const warnBar = el('div', { class: 'warn-bar' });
  warnBar.style.display = 'none';
  const paintWarning = () => {
    const w = data.marker_warning || {};
    warnBar.innerHTML = '';
    if (!w.static) { warnBar.style.display = 'none'; return; }
    warnBar.style.display = 'flex';
    warnBar.append(el('span', { class: 'warn-mark' }, '!'),
      el('div', {},
        el('div', {}, w.message),
        el('div', { class: 'hint' }, (w.reasons || []).join(' · '))));
    const vai = el('button', {}, 'Indica il marker giusto');
    vai.addEventListener('click', () => { setDrawing(true); warnBar.scrollIntoView({ block: 'center' }); });
    warnBar.append(vai);
  };

  /* --- i marker che fissano i bordi degli envelope ---

     Un envelope e' l'unione dei box del gruppo: per ogni lato c'e' un'immagine che lo ha
     spinto fin la'. Vederle serve a capire perche' un box e' grande, e ad accorgersi
     quando e' una detection sbagliata a gonfiarlo: se un lato lo fissa una sola immagine e
     quella sta lontana dal grosso del gruppo, e' quella da guardare. */
  const limitNodes = [];
  for (let i = 0; i < 4; i += 1) {
    const node = el('div', { class: 'editor-box limit-box' }, el('span', { class: 'limit-tag' }, ''));
    node.style.display = 'none';
    stage.append(node);
    limitNodes.push(node);
  }
  let limitsData = null;
  let showLimits = false;
  const limitsCard = el('div', { class: 'card' });
  const limitsTag = el('span', { class: 'ov-fold-tag' });

  const paintLimits = () => {
    for (const node of limitNodes) node.style.display = 'none';
    const row = byName.get(names[index]);
    const group = row && row.group;
    const info = (limitsData && group) ? (limitsData.groups || {})[group] : null;
    if (!showLimits || !info) return;
    Object.values(info.limits).forEach((limit, i) => {
      const node = limitNodes[i];
      if (!node) return;
      place(node, limit.box);
      node.style.setProperty('--box-color', GROUP_COLORS[group] || '#ffffff');
      node.firstChild.textContent = `${limit.side} ${limit.value}`;
      node.classList.toggle('here', limit.image === names[index]);
    });
  };

  const goTo = (name) => {
    const target = names.indexOf(name);
    if (target < 0) { toast('immagine non in elenco', true); return; }
    index = target;
    if (filter && !matches(name)) applyFilter(''); else { show(); renderList(); }
    stage.scrollIntoView({ block: 'center' });
  };

  const renderLimits = () => {
    limitsCard.innerHTML = '';
    const dubbi = Object.values((limitsData || {}).groups || {}).reduce(
      (n, info) => n + Object.values(info.limits).filter(
        (l) => l.shared === 1 && l.gain >= 15 && (l.score == null || Number(l.score) < 0.75),
      ).length, 0);
    limitsTag.textContent = !limitsData ? '' : dubbi ? `${dubbi} da guardare` : 'in ordine';
    limitsTag.className = 'ov-fold-tag' + (dubbi ? ' warn' : '');
    limitsCard.append(el('h3', { style: 'margin-top:0' }, 'chi fissa i bordi degli envelope'));
    if (!limitsData || !Object.keys(limitsData.groups || {}).length) {
      limitsCard.append(el('p', { class: 'hint' }, 'nessun envelope ancora calcolato'));
      return;
    }
    limitsCard.append(el('p', { class: 'hint' },
      'ogni lato del box di raccolta e\' stato spinto fin la\' da un\'immagine. "secondo" e\' ' +
      'dove finirebbe quel lato senza di lei, quindi "si stringe di" e\' quanto pesa quella ' +
      'sola immagine. La confidenza della **scoperta** e\' quella che ha deciso la posizione ' +
      '(e dice se il bordo e\' fidato); quella della **validazione** e\' il ritaglio ' +
      'consegnato cercato dentro l\'envelope, cioe\' il numero che si vede accanto ' +
      'all\'immagine. Un estremo isolato e scoperto male e\' probabilmente una detection ' +
      'sbagliata: correggila e l\'envelope si stringe.'));
    for (const group of (limitsData.order || Object.keys(limitsData.groups))) {
      const info = (limitsData.groups || {})[group];
      if (!info) continue;
      const e = info.envelope;
      const box = el('div', { style: 'margin:10px 0 14px' });
      box.append(el('div', { style: `font-weight:600;color:${GROUP_COLORS[group]}` },
        `${group} — ${e.top}|${e.left}|${e.bottom}|${e.right} · ` +
        `${info.size ? `${info.size[0]}x${info.size[1]} px` : '—'} da ${info.markers} marker`));
      const stretto = info.tight_size || [];
      if (info.size && stretto.length &&
          (stretto[0] !== info.size[0] || stretto[1] !== info.size[1])) {
        box.append(el('div', { class: 'hint' },
          `senza i quattro estremi sarebbe ${stretto[0]}x${stretto[1]} px`));
      }
      const table = el('table', { class: 'lines' });
      table.append(el('tr', {},
        el('td', { class: 'name' }, 'lato'),
        el('td', { class: 'name' }, 'coordinata'),
        el('td', { class: 'name' }, 'secondo'),
        el('td', { class: 'name' }, 'si stringe di'),
        el('td', { class: 'name' }, 'confidenza scoperta'),
        el('td', { class: 'name' }, 'confidenza validazione'),
        el('td', { class: 'name' }, 'dal centro del gruppo'),
        el('td', { class: 'name' }, 'immagine')));
      // Sospetta non e' "unica e distante": il marker si muove davvero, e un estremo
      // trovato a 1.0 e' una posizione vera che sta al bordo. Quello che merita un occhio
      // e' un estremo **isolato e trovato male**: e' la firma del falso match.
      const dubbio = (l) => l.shared === 1 && l.gain >= 15
        && (l.score == null || Number(l.score) < 0.75);
      for (const limit of Object.values(info.limits)) {
        const solo = limit.shared === 1;
        const sospetta = dubbio(limit);
        const link = el('button', { class: 'ghost link' }, limit.image.slice(-40));
        link.addEventListener('click', () => goTo(limit.image));
        table.append(el('tr', {},
          el('td', { class: 'name' }, limit.side),
          el('td', { class: 'val' }, String(limit.value)),
          el('td', { class: 'val' }, String(limit.second)),
          el('td', {
            class: 'val',
            style: `color:${sospetta ? 'var(--warn)' : 'var(--muted)'}`,
          }, limit.gain ? `${limit.gain} px` : '—'),
          el('td', {
            class: 'val',
            style: `color:${limit.score == null ? 'var(--muted)'
              : limit.score >= 0.8 ? 'var(--ok)' : limit.score >= 0.62 ? 'var(--warn)' : 'var(--err)'}`,
          }, limit.score == null ? '—' : Number(limit.score).toFixed(4)),
          el('td', {
            class: 'val',
            style: `color:${limit.score_validation == null ? 'var(--muted)'
              : limit.score_validation >= 0.8 ? 'var(--ok)'
              : limit.score_validation >= 0.62 ? 'var(--warn)' : 'var(--err)'}`,
          }, limit.score_validation == null ? '—' : Number(limit.score_validation).toFixed(4)),
          el('td', { class: 'val' }, `${limit.distance_from_median} px`),
          el('td', { class: 'val' },
            link,
            limit.corrected ? el('span', { class: 'score-fixed' }, 'corretta') : null,
            solo ? null : el('span', { class: 'hint' }, ` +${limit.shared - 1} pari`))));
      }
      box.append(table);
      const sospetti = Object.values(info.limits).filter(dubbio);
      if (sospetti.length) {
        box.append(el('div', { class: 'hint', style: 'color:var(--warn)' },
          `da guardare: ${sospetti.map((l) => `${l.side} (−${l.gain} px, ` +
            `confidenza ${l.score == null ? '—' : Number(l.score).toFixed(3)})`).join(', ')} ` +
          `— ${sospetti.length === 1 ? 'quel lato lo fissa una sola immagine, trovata male'
            : 'quei lati li fissano immagini singole, trovate male'}`));
      }
      limitsCard.append(box);
    }
  };

  /* La stessa informazione dei bordi, ma compatta e accanto all'immagine: una riga per
     lato, cliccabile, cosi' si passa da un vertice all'altro senza scorrere la pagina.
     Il dettaglio completo (secondo valore, confidenza della validazione, distanza dalla
     mediana, quante immagini condividono l'estremo) resta nel tooltip e nella tabella in
     fondo. */
  const limitsSide = el('div', { class: 'ov-limits' });
  const limitRows = [];
  const SIDE_SHORT = { alto: 'alto', basso: 'basso', sinistra: 'sx', destra: 'dx' };

  /* Oltre a evidenziare la riga dell'immagine corrente, la porta sotto gli occhi dentro
     la scatola. Niente scrollIntoView: quello muoverebbe anche la pagina, e qui si sta
     scorrendo fra le immagini con le frecce. */
  const markCurrentLimits = () => {
    let corrente = null;
    for (const { node, image } of limitRows) {
      const qui = image === names[index];
      node.classList.toggle('here', qui);
      if (qui && !corrente) corrente = node;
    }
    if (!corrente) return;
    const testa = limitsSide.firstChild && limitsSide.firstChild.offsetHeight || 0;
    const alto = corrente.offsetTop - testa - 22;
    const basso = corrente.offsetTop + corrente.offsetHeight - limitsSide.clientHeight + 6;
    if (limitsSide.scrollTop > alto) limitsSide.scrollTop = Math.max(0, alto);
    else if (limitsSide.scrollTop < basso) limitsSide.scrollTop = basso;
  };

  const renderLimitsSide = () => {
    limitsSide.innerHTML = '';
    limitRows.length = 0;
    const groups = (limitsData || {}).groups || {};
    if (!Object.keys(groups).length) {
      limitsSide.append(el('div', { class: 'hint', style: 'padding:6px 8px' },
        'nessun envelope calcolato'));
      return;
    }
    limitsSide.append(el('div', { class: 'ov-limits-head' },
      el('span', {}, 'bordi degli envelope'),
      el('span', { class: 'hint' }, 'clicca per andarci')));
    for (const group of (limitsData.order || Object.keys(groups))) {
      const info = groups[group];
      if (!info) continue;
      limitsSide.append(el('div', {
        class: 'ov-limits-group',
        style: `color:${GROUP_COLORS[group]}`,
        title: `${group}: envelope ${info.envelope.top}|${info.envelope.left}|`
          + `${info.envelope.bottom}|${info.envelope.right}`
          + (info.size ? ` = ${info.size[0]}x${info.size[1]} px` : '')
          + ` da ${info.markers} marker`
          + (info.tight_size ? ` · senza i quattro estremi ${info.tight_size[0]}x${info.tight_size[1]}` : ''),
      },
        el('span', {}, group),
        el('span', { class: 'hint' },
          info.size ? `${info.size[0]}x${info.size[1]}` : '—')));
      for (const limit of Object.values(info.limits)) {
        const dubbio = limit.shared === 1 && limit.gain >= 15
          && (limit.score == null || Number(limit.score) < 0.75);
        const riga = el('button', {
          class: 'lim-row' + (dubbio ? ' dubbio' : '') + (limit.corrected ? ' fixed' : ''),
          title: `${limit.side} ${limit.value} — lo fissa ${limit.image}\n`
            + `senza di lei il lato starebbe a ${limit.second} (${limit.gain} px in meno)\n`
            + `confidenza scoperta ${limit.score ?? '—'} · validazione ${limit.score_validation ?? '—'}\n`
            + `${limit.distance_from_median} px dal centro mediano del gruppo · `
            + (limit.shared === 1 ? 'estremo di questa sola immagine'
               : `estremo condiviso da ${limit.shared} immagini`)
            + (limit.corrected ? ' · posizione corretta a mano' : ''),
        },
          el('span', { class: 'lim-side' }, SIDE_SHORT[limit.side] || limit.side),
          el('span', { class: 'lim-val' }, String(limit.value)),
          el('span', { class: 'lim-gain' }, limit.gain ? `\u2212${limit.gain}` : '—'),
          el('span', {
            class: 'lim-score',
            style: `color:${limit.score == null ? 'var(--muted)'
              : limit.score >= 0.8 ? 'var(--ok)' : limit.score >= 0.62 ? 'var(--warn)' : 'var(--err)'}`,
          }, limit.score == null ? '—' : Number(limit.score).toFixed(2)),
          el('span', { class: 'lim-name' }, limit.image.split('/').pop()));
        riga.addEventListener('click', () => goTo(limit.image));
        limitRows.push({ node: riga, image: limit.image });
        limitsSide.append(riga);
      }
    }
    markCurrentLimits();
  };

  const loadLimits = async () => {
    try {
      limitsData = await api(`/projects/${projectId}/orientation/limits`);
    } catch (error) { limitsData = null; }
    renderLimits();
    renderLimitsSide();
    paintLimits();
  };

  const limitsToggle = el('input', { type: 'checkbox' });
  limitsToggle.addEventListener('change', () => {
    showLimits = limitsToggle.checked;
    paintLimits();
  });

  const serieToggle = el('input', { type: 'checkbox' });
  serieToggle.checked = false;
  const serieBadge = el('span', { class: 'hint' });
  let serieCount = 0;
  let serieIdle = null;
  const SERIE_MAX = 12;
  const SERIE_IDLE_MS = 45000;

  /* La serie si chiude da sola: per inattivita' o dopo un tetto di correzioni. Un clic
     partito per sbaglio sull'immagine puo' salvare una correzione (si annulla dalla
     sezione correzioni), ma non puo' innescarne una catena silenziosa. */
  const closeSerie = (reason) => {
    if (serieIdle) { clearTimeout(serieIdle); serieIdle = null; }
    const done = serieCount;
    serieCount = 0;
    serieBadge.textContent = '';
    setCorrecting(false);
    if (reason && done) toast(`serie chiusa (${reason}): ${done} correzioni`);
  };
  const keepSerie = () => {
    serieBadge.textContent = `serie: ${serieCount} corrett${serieCount === 1 ? 'a' : 'e'}` +
      ` · si chiude a ${SERIE_MAX} o dopo 45s di pausa`;
    if (serieIdle) clearTimeout(serieIdle);
    serieIdle = setTimeout(() => closeSerie('pausa'), SERIE_IDLE_MS);
  };
  serieToggle.addEventListener('change', () => {
    if (!serieToggle.checked) closeSerie('');
  });

  image.addEventListener('load', paint);
  window.addEventListener('resize', paint);
  // stessa ragione dello studio del rettangolo: il layout puo' restringere l'immagine dopo
  // il caricamento, e i box vanno ridisegnati sulla larghezza vera
  if (window.ResizeObserver) new ResizeObserver(() => paint()).observe(image);

  const onKey = (event) => {
    if (!document.body.contains(root)) {
      window.removeEventListener('keydown', onKey);
      return;
    }
    if (event.target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) return;
    if (event.key === 'ArrowRight') { event.preventDefault(); step(1); }
    else if (event.key === 'ArrowLeft') { event.preventDefault(); step(-1); }
  };
  window.addEventListener('keydown', onKey);

  /* Il ritaglio dell'immagine corrente: e' il glifo che la configurazione deve portarsi
     dietro (nel legacy: DB_echo/setup_NN/orientation_0..3.png). La banca serve solo a
     confronto, in fondo. */
  const cropCard = el('div', { class: 'template-card' });
  const cropImage = el('img', { alt: 'ritaglio' });
  const cropNote = el('div', { class: 'hint' });
  cropCard.append(cropImage, cropNote);

  const paintCrop = () => {
    const row = byName.get(names[index]);
    if (!row || !row.box) {
      cropImage.style.display = 'none';
      cropNote.textContent = 'nessun marker in questa immagine: niente da ritagliare';
      return;
    }
    cropImage.style.display = 'block';
    cropImage.src = `/api/projects/${projectId}/orientation/crop` +
      `?name=${encodeURIComponent(names[index])}&scale=6&pad=2&t=${Date.now()}`;
    const w = row.box.right - row.box.left;
    const h = row.box.bottom - row.box.top;
    const ship = (data.folder_template || {}).size || [];
    const uguale = ship.length === 2 && ship[0] === w + 1 && ship[1] === h + 1;
    cropNote.innerHTML = '';
    cropNote.append(
      el('strong', {}, `${w + 1}x${h + 1} px`), el('br'),
      el('span', {}, `marker di questa immagine · ${row.group}`), el('br'),
      el('span', { style: 'color:var(--muted)' },
        uguale
          ? 'stessa misura del ritaglio consegnato'
          : `in DB_echo va il ritaglio della cartella (${ship.join('x') || '—'})`),
    );
  };

  /* Il ritaglio consegnato e la copertura.

     Si rifa' a ogni rielaborazione, e non e' un dettaglio: prima si costruiva una volta
     sola all'apertura della sezione, cosi' dopo aver stretto il marker si continuava a
     vedere il ritaglio vecchio - proprio la cosa che si era appena cambiata. E l'immagine
     porta un contrassegno che cambia col ritaglio: il PNG sta sempre allo stesso indirizzo,
     e senza quello il browser mostrerebbe comunque quello di prima. */
  const consegnatoCard = el('div', { class: 'card' });
  const renderConsegnato = () => {
    const ft = data.folder_template || {};
    const val = data.validation || {};
    const copertura = val.coverage_by_group || {};
    consegnatoCard.innerHTML = '';
    if (!ft.url && !Object.keys(copertura).length) {
      consegnatoCard.style.display = 'none';
      return;
    }
    consegnatoCard.style.display = '';
    consegnatoCard.append(el('h3', { style: 'margin-top:0' }, 'ritaglio consegnato e copertura'));
    const body = el('div', { style: 'display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap' });
    if (ft.url) {
      // Il contrassegno e' il momento in cui il PNG e' stato scritto: e' l'unica cosa che
      // cambia di sicuro quando cambia il ritaglio, e sta sempre allo stesso indirizzo.
      const marca = ft.mtime || (data.marker_override || {}).ts || '';
      const shipped = el('div', { class: 'template-card' });
      shipped.append(el('img', {
        src: `${ft.url}?v=${encodeURIComponent(marca || String(Date.now()))}`,
        alt: 'ritaglio consegnato',
      }));
      shipped.append(el('div', { class: 'hint' },
        el('strong', {}, `${(ft.size || []).join('x')} px`), el('br'),
        ft.from_user_marker ? 'dal marker che hai indicato tu' : `innesco ${ft.seed_bank_template || '?'}`,
        el('br'),
        ft.from_user_marker
          ? `da ${(ft.source_image || '').split('/').pop() || '?'}`
          : `scala ${ft.seed_scale ?? '?'}`));
      body.append(shipped);
    }
    const table = el('table', { class: 'lines', style: 'flex:1;min-width:320px' });
    table.append(el('tr', {},
      el('td', { class: 'name' }, 'gruppo'),
      el('td', { class: 'name' }, 'copertura'),
      el('td', { class: 'name' }, 'mediana'),
      el('td', { class: 'name' }, 'minimo')));
    for (const group of Object.keys(GROUP_COLORS)) {
      const c = copertura[group];
      if (!c) continue;
      const pct = Math.round(c.coverage * 100);
      table.append(el('tr', {},
        el('td', { class: 'name', style: `color:${GROUP_COLORS[group]}` }, group),
        el('td', { class: 'val', style: `color:${pct >= 95 ? 'var(--ok)' : pct >= 80 ? 'var(--warn)' : 'var(--err)'}` },
          `${c.matched}/${c.images} = ${pct}%`),
        el('td', { class: 'val' }, String(c.median ?? '—')),
        el('td', { class: 'val' }, String(c.min ?? '—'))));
    }
    body.append(table);
    consegnatoCard.append(body);
    consegnatoCard.append(el('p', { class: 'hint' },
      'la copertura e\' il ritaglio consegnato cercato DENTRO l\'envelope del suo gruppo, ' +
      `soglia ${val.min_score ?? '?'}: e\' la previsione di cosa fara\' ESI, che a runtime ` +
      'usera\' proprio questo PNG. Sotto il 95% quel gruppo e\' a rischio.'));
  };
  renderConsegnato();

  /* --- i controlli, compatti: una barra sola per filtri, ordine, navigazione e azioni.
     Prima erano quattro righe piu' due copie di filtri e ordine, e l'immagine finiva
     lontana dall'inizio della pagina. --- */
  const sortSelect = el('select', { title: 'ordine delle immagini' });
  for (const [mode, label] of [
    ['folder', 'ordine: cartella'],
    ['asc', 'ordine: peggiori prima'],
    ['desc', 'ordine: migliori prima'],
  ]) sortSelect.append(el('option', { value: mode }, label));
  sortSelect.addEventListener('change', () => {
    sortMode = sortSelect.value;
    const list = visible();
    if (list.length && !list.includes(names[index])) index = names.indexOf(list[0]);
    show();
    renderList();
  });

  const listBox = el('div', { class: 'score-list' });
  const listCount = el('span', { class: 'hint' });
  const LIST_LIMIT = 300;
  const renderList = () => {
    const list = visible();
    listCount.textContent = `${list.length} immagini`;
    updateChips();
    listBox.innerHTML = '';
    for (const name of list.slice(0, LIST_LIMIT)) {
      const score = scoreOf(name);
      const row = byName.get(name);
      const item = el('div', {
        class: 'score-row' + (name === names[index] ? ' current' : '')
          + (name === busyName ? ' busy' : ''),
      },
        el('span', { class: 'score-value', style: `color:${score == null ? 'var(--muted)' : score >= 0.8 ? 'var(--ok)' : score >= 0.62 ? 'var(--warn)' : 'var(--err)'}` },
          score == null ? '—' : score.toFixed(3)),
        el('span', { class: 'score-group' }, (row && row.group) || (excluded && excluded.test(name) ? 'esclusa' : '—')),
        el('span', { class: 'score-name', title: name }, name.split('/').pop()),
        row && row.corrected ? el('span', { class: 'score-fixed' }, 'corretta') : null,
        row && row.score_batch != null && row.score != null && row.score !== row.score_batch
          ? el('span', { class: 'score-delta' }, `era ${row.score_batch}`) : null,
      );
      item.addEventListener('click', () => { index = names.indexOf(name); show(); renderList(); });
      listBox.append(item);
      if (name === names[index]) {
        // l'immagine corrente resta in vista anche quando si scorre con le frecce
        requestAnimationFrame(() => item.scrollIntoView({ block: 'nearest' }));
      }
    }
    if (list.length > LIST_LIMIT) {
      listBox.append(el('div', { class: 'hint', style: 'padding:6px 8px' },
        `prime ${LIST_LIMIT} di ${list.length}: restringi con un filtro`));
    }
  };

  /* I filtri sono pastiglie con il conto, e il conto si rifa' dopo ogni correzione: un
     numero fermo su "da rivedere" sarebbe una bugia. */
  const filterRow = el('div', { class: 'ov-chips' });
  const chipCount = (key) => (!key
    ? names.length
    : names.filter((name) => (key === 'REVIEW' ? inReview(name) : groupOf(name) === key)).length);
  const updateChips = () => {
    for (const [button, key] of filterButtons) {
      const count = chipCount(key);
      button.textContent = `${FILTERS.find((f) => f.key === key).label} ${count}`;
      button.disabled = count === 0 && key !== filter;
    }
  };
  for (const spec of FILTERS) {
    const button = el('button', { class: 'chip' + (spec.key === filter ? ' on' : '') },
      `${spec.label} ${chipCount(spec.key)}`);
    button.addEventListener('click', () => applyFilter(spec.key));
    if (spec.key && GROUP_COLORS[spec.key]) button.style.borderColor = GROUP_COLORS[spec.key];
    button.disabled = chipCount(spec.key) === 0;
    filterButtons.push([button, spec.key]);
    filterRow.append(button);
  }

  const navRow = el('div', { class: 'ov-nav' },
    el('button', { class: 'ghost sq', title: 'immagine precedente (freccia sinistra)',
      onclick: () => step(-1) }, '‹'),
    imageCounter,
    el('button', { class: 'ghost sq', title: 'immagine successiva (freccia destra)',
      onclick: () => step(1) }, '›'),
    el('button', { class: 'ghost', title: 'salta alla prossima immagine con un marker trovato',
      onclick: () => stepFound(1) }, 'prossima con marker'),
    sortSelect, listCount);

  /* Quanto lontano dal punto indicato si cerca.

     Era settanta pixel per lato: su un marker alto quindici, il vincitore poteva stare
     nove marker piu' in la', e la correzione smetteva di essere una correzione per
     diventare un secondo parere. Chi clicca il centro lo sbaglia di qualche pixel. Venti
     e' il valore giusto quasi sempre; qui si cambia quando non lo e'. */
  const raggioInput = el('input', {
    type: 'number', min: '4', max: '200', step: '2', value: '20', style: 'width:62px',
    title: 'quanto lontano dal punto che indichi puo\' stare il marker',
  });
  const raggio = () => {
    const v = parseInt(raggioInput.value, 10);
    return Number.isFinite(v) ? Math.max(4, Math.min(v, 200)) : 20;
  };

  const fixBar = el('div', { class: 'ov-actions' },
    fixButton,
    el('label', { class: 'serie', title: 'il marker viene cercato solo dentro a questo raggio dal punto che indichi' },
      el('span', {}, 'raggio'), raggioInput, el('span', {}, 'px')),
    el('label', { class: 'serie', title: 'resta armato e passa alla prossima immagine dopo ogni correzione' },
      serieToggle, el('span', {}, 'in serie')),
    wrongButton, modificaButton,
    el('label', { class: 'serie', title: 'disegna sull\'immagine i quattro marker che fissano i bordi dell\'envelope di questo gruppo' },
      limitsToggle, el('span', {}, 'vertici')),
    pickHint, serieBadge, wrongHint);

  const legend = el('div', { class: 'ov-legend' });
  const renderLegend = () => {
    legend.innerHTML = '';
    for (const [group, color] of Object.entries(GROUP_COLORS)) {
      const box = (data.envelopes || {})[group];
      legend.append(el('span', {
        class: 'editor-legend',
        title: box ? `${box.markers} marker nell'envelope ${group}` : `nessun marker ${group}`,
      },
        el('span', { class: 'swatch', style: `background:${color}` }),
        `${group}${box ? ` ${box.markers}` : ' —'}`));
    }
    legend.append(el('span', { class: 'editor-legend' },
      el('span', { class: 'swatch', style: 'background:#ffffff' }), 'marker'));
  };
  renderLegend();

  /* Le correzioni vivono in una sezione richiudibile, ma il contenitore c'e' sempre: prima
     la scheda nasceva solo se le correzioni esistevano gia', e la prima correzione della
     sessione non si vedeva fino al ricaricamento. */
  const corrHost = el('div');
  const corrTag = el('span', { class: 'ov-fold-tag' });
  const renderCorrections = (list) => {
    corrTag.textContent = list.length ? `${list.length}` : 'nessuna';
    corrHost.innerHTML = '';
    corrHost.append(list.length
      ? correctionsSection(projectId, list, (name) => {
          const target = names.indexOf(name);
          if (target >= 0) { filter = ''; index = target; applyFilter(''); }
        })
      : el('p', { class: 'hint' }, 'nessuna correzione manuale su questa cartella'));
  };
  renderCorrections(data.corrections_detail || []);

  // In fondo, il riferimento della banca: serve solo a confrontare.
  const template = data.template || {};
  const compare = el('div', { class: 'card' });
  compare.append(el('h3', { style: 'margin-top:0' }, 'confronto: banca, ritaglio consegnato, immagine corrente'));
  if (template.url) {
    const side = el('div', { style: 'display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap' });
    const bank = el('div', { class: 'template-card' });
    bank.append(el('img', { src: template.url, alt: template.name }));
    bank.append(el('div', { class: 'hint' },
      el('strong', {}, template.name), el('br'),
      `riferimento della banca ${template.vendor}`, el('br'),
      el('span', { style: 'color:var(--muted)' }, 'ha fatto match, non e\' il glifo vero')));
    const mine = el('div', { class: 'template-card' });
    const mineImage = el('img', { alt: 'ritaglio' });
    mine.append(mineImage, el('div', { class: 'hint' },
      el('strong', {}, 'immagine corrente'), el('br'),
      'il glifo vero, tagliato dove lo vedi riquadrato'));
    // Il ritaglio consegnato sta anche qui, accanto al riferimento della banca e al glifo
    // di questa immagine. Anche questo si rinfresca: e' il PNG che finisce in DB_echo, e
    // dopo averlo stretto deve essere quello nuovo a farsi vedere.
    const shippedImage = el('img', { alt: 'consegnato' });
    const shippedNote = el('div', { class: 'hint' });
    const shipped = el('div', { class: 'template-card' }, shippedImage, shippedNote);
    compareRefresh = () => {
      const row = byName.get(names[index]);
      mineImage.style.display = row && row.box ? 'block' : 'none';
      if (row && row.box) {
        mineImage.src = `/api/projects/${projectId}/orientation/crop` +
          `?name=${encodeURIComponent(names[index])}&scale=6&pad=2&t=${Date.now()}`;
      }
      const attuale = data.folder_template || {};
      shipped.style.display = attuale.url ? '' : 'none';
      if (!attuale.url) return;
      const marca = attuale.mtime || (data.marker_override || {}).ts || Date.now();
      shippedImage.src = `${attuale.url}?v=${encodeURIComponent(marca)}`;
      shippedNote.innerHTML = '';
      shippedNote.append(el('strong', {}, 'consegnato'), el('br'),
        `${(attuale.size || []).join('x')} px`, el('br'),
        'e\' questo che finisce in DB_echo');
    };
    side.append(bank, shipped, mine);
    compare.append(side);
    compare.append(el('p', { class: 'hint' },
      'la banca e\' costruita dagli orientation_*.png delle configurazioni storiche: il ' +
      'template a sinistra e\' il riferimento che ha fatto match, quello a destra e\' il ' +
      'glifo vero di questa acquisizione. Se differiscono molto, il punteggio di match lo dice.'));
  } else {
    compare.append(el('p', { class: 'hint' }, 'nessun template della banca registrato'));
  }

  /* Versare il ritaglio nella banca del vendor: e' il modo in cui questa configurazione
     insegna qualcosa alla prossima. La banca e' fatta dagli orientation_*.png delle
     configurazioni storiche, e questo ritaglio e' la stessa cosa — il glifo vero di
     un'acquisizione. */
  const bankRow = el('div', { class: 'row', style: 'margin-top:10px' });
  const bankNote = el('span', { class: 'hint' });
  const bankButton = el('button', { class: 'ghost' }, 'Aggiungi alla banca');
  let bankState = null;
  const paintBank = () => {
    bankNote.innerHTML = '';
    if (!bankState || bankState.error) {
      bankButton.disabled = true;
      bankNote.append(bankState ? bankState.error : 'banca non raggiungibile');
      return;
    }
    const s = bankState.shipped || {};
    bankButton.textContent = `Aggiungi alla banca ${bankState.vendor || '?'}`;
    bankButton.disabled = !s.available || Boolean(s.already_in_bank);
    bankNote.append(
      `banca ${bankState.vendor}: ${bankState.templates} template`
      + (s.already_in_bank
        ? ` · questo ritaglio c'e' gia' come ${s.already_in_bank}`
        : s.available
          ? ` · il ritaglio consegnato (${(s.size || []).join('x')} px) entrerebbe come ${bankState.next_name}`
          : ' · nessun ritaglio da versare'));
  };
  const loadBank = async () => {
    try {
      bankState = await api(`/projects/${projectId}/orientation/bank`);
    } catch (error) { bankState = { error: error.message }; }
    paintBank();
  };
  bankButton.addEventListener('click', async () => {
    bankButton.disabled = true;
    try {
      const res = await api(`/projects/${projectId}/orientation/bank`, { body: {} });
      toast(`aggiunto alla banca ${res.vendor} come ${res.name} (${(res.size || []).join('x')} px)`);
      await loadBank();
    } catch (error) {
      toast(error.message, true);
      await loadBank();
    }
  });
  bankRow.append(bankButton, bankNote);
  compare.append(bankRow);
  loadBank();
  /* --- impaginazione ---
     Sopra: una riga di stato e la barra dei controlli. In mezzo: l'immagine a sinistra,
     l'elenco delle confidenze a destra che scorre per conto suo, cosi' si passa da
     un'immagine all'altra senza muovere la pagina. Sotto: le analisi, richiudibili. */
  /* Dove e' stato trovato il marker, per posizione.

     Correggere un marker vale per una immagine. Ma un glifo sbagliato non sbaglia una volta
     sola: su prova 5 il ritaglio si ritrovava su sette immagini nello stesso punto del
     pannello di destra, sempre a 0.62, e quelle sette inventavano un gruppo LR che nella
     cartella non c'e'. Correggerle una per una non serviva a niente - l'envelope e' la loro
     unione, e finche' ci sono dentro il gruppo resta. Qui si vedono raggruppate per posto e
     si buttano tutte insieme. */
  const posizioniHost = el('div', {});
  const posizioniTag = el('span', { class: 'ov-fold-tag' });
  /* Il posto e' la **riga**, non il punto. Il pallino di Esaote sta al bordo dell'immagine
     ecografica e scorre in orizzontale con la larghezza del ventaglio: raggruppando per
     punto usciva una riga ogni quaranta pixel - diciassette per quarantasette immagini - e
     non si capiva piu' niente. L'altezza invece e' quella che distingue il marker vero da
     un falso aggancio: sono a righe diverse dello schermo. */
  const chiaveDiPosto = (box) => `${Math.round((box.top + box.bottom) / 2 / 40)}`;
  const gruppiDiPosto = () => {
    const mappa = new Map();
    for (const row of byName.values()) {
      if (!row.box) continue;
      const chiave = `${row.group || '?'}#${chiaveDiPosto(row.box)}`;
      if (!mappa.has(chiave)) mappa.set(chiave, []);
      mappa.get(chiave).push(row);
    }
    return [...mappa.values()].sort((a, b) => b.length - a.length);
  };
  const renderPosizioni = () => {
    posizioniHost.innerHTML = '';
    const gruppi = gruppiDiPosto();
    const rifiutate = [...byName.values()].filter((r) => r.refused).length;
    const scontente = [...byName.values()].filter((r) => r.disagrees).length;
    posizioniTag.textContent = `${gruppi.length} posizioni`
      + (rifiutate ? ` · ${rifiutate} rifiutate` : '')
      + (scontente ? ` · ${scontente} in disaccordo` : '');
    if (!gruppi.length) {
      posizioniHost.append(el('p', { class: 'hint' }, 'nessun marker trovato.'));
      return;
    }
    posizioniHost.append(el('p', { class: 'hint' },
      'ogni riga e\' un posto dove il ritaglio si e\' ritrovato. Quelle con poche immagini e '
      + 'punteggio basso, lontane dalle altre, di solito non sono il marker: buttarle '
      + 'stringe l\'envelope e toglie i gruppi inventati.'));
    const griglia = el('div', { class: 'depth-grid' });
    for (const elenco of gruppi) {
      const primo = elenco[0];
      const b = primo.box;
      const punteggi = elenco.map((r) => Number(r.score) || 0).sort((a, c) => a - c);
      const mediana = punteggi[Math.floor(punteggi.length / 2)];
      const tutteRifiutate = elenco.every((r) => r.refused);
      // Fuori dagli envelope perche' danno torto a una correzione: e' un rifiuto che hai
      // dato tu senza dirlo, e va detto.
      const tutteScontente = !tutteRifiutate && elenco.every((r) => r.disagrees);
      const scheda = el('div', {
        class: 'depth-card' + (tutteRifiutate || tutteScontente ? ' spenta' : ''),
      });
      scheda.append(
        el('div', { class: 'depth-value' }, `${primo.group || '?'} · ${elenco.length} immagini`,
          el('span', { class: 'hint' }, ` · mediana ${mediana ? mediana.toFixed(3) : '—'}`)),
        el('img', {
          class: 'depth-crop', loading: 'lazy', alt: '',
          src: `/api/projects/${projectId}/orientation/crop`
            + `?name=${encodeURIComponent(primo.name)}&scale=6&pad=6`
            + `&box=${b.top}|${b.left}|${b.bottom}|${b.right}`,
        }),
        el('div', { class: 'hint' },
          `riga y ${Math.min(...elenco.map((r) => r.box.top))}-`
          + `${Math.max(...elenco.map((r) => r.box.bottom))}`
          + ` · x da ${Math.min(...elenco.map((r) => r.box.left))} a `
          + `${Math.max(...elenco.map((r) => r.box.left))}`
          + ` · ${b.right - b.left}x${b.bottom - b.top} px`
          + (tutteRifiutate ? ' · rifiutate' : '')
          + (tutteScontente ? ' · fuori: un\'altra riga rispetto alla tua correzione' : '')),
      );
      const apri = el('button', { class: 'ghost sq2' }, 'apri la prima');
      apri.addEventListener('click', () => {
        const dove = names.indexOf(primo.name);
        if (dove >= 0) { index = dove; show(); }
      });
      const butta = el('button', { class: 'ghost sq2' },
        tutteRifiutate ? 'rimettile' : `non e' il marker (${elenco.length})`);
      butta.addEventListener('click', async () => {
        butta.disabled = true;
        try {
          await api(`/projects/${projectId}/orientation/refuse`, {
            body: { names: elenco.map((r) => r.name), reset: tutteRifiutate },
          });
          toast(tutteRifiutate
            ? `${elenco.length} rimesse in gioco`
            : `${elenco.length} detection buttate`);
          await refreshData();
        } catch (errore) { toast(errore.message, true); butta.disabled = false; }
      });
      scheda.append(el('div', { class: 'row' }, apri, butta));
      griglia.append(scheda);
    }
    posizioniHost.append(griglia);
  };

  const fold = (titolo, tag, contenuto, aperto) => {
    const nodo = el('details', { class: 'ov-fold' });
    if (aperto) nodo.setAttribute('open', 'open');
    const testa = el('summary', {}, el('span', {}, titolo));
    if (tag) testa.append(tag);
    nodo.append(testa, contenuto);
    return nodo;
  };

  root.append(el('div', { class: 'ov-status' },
    el('span', { class: 'hint' },
      `${analysed} immagini analizzate` +
      (skipped ? ` · ${skipped} proibite escluse` : '') +
      ` · ${names.length} uniche`),
    el('details', { class: 'ov-help' },
      el('summary', {}, 'come si corregge'),
      el('p', { class: 'hint' },
        'scorri con le frecce e controlla il marker riquadrato in bianco. Se e\' nel posto ' +
        'sbagliato premi «Correggi il marker» e clicca sul marker vero: posizione, gruppo ed ' +
        'envelope arrivano subito, il resto della cartella si rielabora in background — non ' +
        'serve aspettarlo. Con «in serie» resti armato e passi alla prossima. Se invece il ' +
        'glifo cercato e\' proprio quello sbagliato, usa «Il marker trovato e\' sbagliato» e ' +
        'trascina un rettangolo attorno a quello vero.'))));

  /* Il primo passo della revisione: **quale glifo**.

     Il modulo sceglie il ritaglio che si ritrova sul maggior numero di immagini, non quello
     col picco piu' alto. E' la regola giusta, ma resta una scelta fatta al buio da chi
     guarda: due ritagli con copertura 1.0 sono uguali per la macchina e diversissimi per
     chi sa cos'e' il marker — uno e' il simbolo, l'altro la sigla della sonda stampata li'
     accanto. E finche' il glifo non e' quello giusto, tutto quello che viene dopo —
     posizioni, gruppi, envelope — descrive un altro oggetto, e guardarlo e' tempo buttato.

     Percio' sta in cima, prima di tutto il resto: si vede il vincitore, si vedono i
     concorrenti con quello che hanno ottenuto, e si dice quale e'. Se e' quello che aveva
     preso lui si tira dritto; se e' un altro, il modulo rifa' lo studio con quello. */
  const poolCard = el('div', { class: 'card' });
  let pool = null;
  const scegliCandidato = async (indice) => {
    setBusy(true, 'rifaccio lo studio col marker che hai scelto');
    try {
      const esito = await api(`/projects/${projectId}/orientation/candidate`,
        { body: { index: indice } });
      if (esito.same) {
        toast('e\' lo stesso ritaglio: lo studio resta quello che vedi');
        setBusy(false);
        await caricaPool();
        return;
      }
      const job = await pollJob(esito.job_id, { set textContent(v) { busyLabel.textContent = v; } });
      setBusy(false);
      await refreshData();
      await caricaPool();
      renderOverrideResult(job.result);
      toast(`rifatto col marker scelto: ${job.result.matched} immagini su ${job.result.images}`);
    } catch (errore) {
      setBusy(false);
      toast(errore.message, true);
    }
  };
  const renderPool = () => {
    poolCard.innerHTML = '';
    if (!pool || !(pool.candidates || []).length) { poolCard.style.display = 'none'; return; }
    poolCard.style.display = '';
    const scelto = pool.candidates.find((c) => c.chosen);
    poolCard.append(el('h3', { style: 'margin-top:0' },
      pool.confirmed ? 'il marker della cartella — confermato da te' : 'il marker della cartella: e\' questo?'));
    poolCard.append(el('p', { class: 'hint' },
      pool.from_user
        ? 'stai cercando il ritaglio che hai indicato tu.'
        : 'il modulo ha scelto quello che si ritrova su piu\' immagini. Qui sotto gli altri '
          + 'che ha provato: se il marker vero e\' uno di loro, scegli quello e rifa\' lo studio.'));
    const fila = el('div', { class: 'ov-pool' });
    for (const c of pool.candidates) {
      const carta = el('div', { class: 'ov-pool-voce' + (c.chosen ? ' scelto' : '') });
      carta.append(el('img', { src: c.url, alt: 'candidato' }));
      carta.append(el('div', { class: 'hint' },
        `${(c.size || []).join('x')} px`, el('br'),
        `copertura ${c.coverage != null ? Math.round(c.coverage * 100) + '%' : '—'}`, el('br'),
        `mediana ${c.median != null ? Number(c.median).toFixed(3) : '—'}`, el('br'),
        el('span', { title: `dal fotogramma ${c.image}` }, (c.image || '').split('/').pop())));
      if (c.chosen) {
        carta.append(el('span', { class: 'ov-pool-tag' },
          pool.from_user ? 'scelto da te' : 'scelto dal modulo'));
      } else {
        const b = el('button', { class: 'ghost sq2' }, 'usa questo');
        b.addEventListener('click', () => scegliCandidato(c.index));
        carta.append(b);
      }
      fila.append(carta);
    }
    poolCard.append(fila);
    if (!pool.confirmed && scelto) {
      const ok = el('button', {}, 'Si\', il marker e\' questo');
      ok.addEventListener('click', async () => {
        ok.disabled = true;
        try {
          await api(`/projects/${projectId}/orientation/candidate`, { body: { confirm: true } });
          toast('marker confermato: da qui in poi si guarda dove cade');
          await caricaPool();
        } catch (errore) { toast(errore.message, true); ok.disabled = false; }
      });
      poolCard.append(el('div', { class: 'row' }, ok,
        el('span', { class: 'hint' },
          'confermarlo non cambia niente nei dati: dice che il resto della revisione si puo\' '
          + 'guardare, perche\' parla del glifo giusto.')));
    }
    if (pool.confirmed) {
      poolCard.append(el('p', { class: 'hint', style: 'color:var(--ok, #3fb950)' },
        `confermato ${pool.confirmed_at ? `il ${pool.confirmed_at.replace('T', ' alle ')}` : ''}`));
    }
  };
  const caricaPool = async () => {
    try {
      pool = await api(`/projects/${projectId}/orientation/candidates`);
    } catch (errore) { pool = null; }
    renderPool();
  };
  root.append(poolCard);
  caricaPool();

  root.append(el('div', { class: 'ov-bar' }, filterRow));

  root.append(fixBar);
  root.append(warnBar);
  root.append(bgBar);

  // Immagine a sinistra, strumenti a destra: la stessa forma di tutte le sezioni.
  // Frecce e lente sopra l'immagine, nello stesso posto di ogni altra sezione.
  const barraImmagine = el('div', { class: 'barra-immagine' },
    navRow, Lente.bottone(contestoLente));
  root.append(el('div', { class: 'ov-body' },
    el('div', { class: 'ov-main' }, barraImmagine, stage, caption),
    el('div', { class: 'ov-side' },
      detail, legend, cropCard, wrongPanel, modificaPanel, fixPanel,
      limitsSide, listBox)));

  root.append(el('div', { class: 'ov-folds' },
    fold('dove e\' stato trovato il marker', posizioniTag, posizioniHost),
    fold('chi fissa i bordi degli envelope', limitsTag, limitsCard),
    fold('correzioni manuali', corrTag, corrHost),
    fold('ultima rielaborazione in background', bgTag, bgReportHost),
    fold('ritaglio consegnato e copertura', null, consegnatoCard),
    fold('confronto: banca, consegnato, immagine corrente', null, compare),
  ));

  show();
  renderList();
  renderPosizioni();
  paintWarning();
  loadLimits();
  // se una rielaborazione era in corso da prima, la striscia la ritrova
  api(`/projects/${projectId}/orientation/consolidation`).then((state) => {
    paintBackground(state);
    bgRuns = state.runs || 0;
    if (state.state === 'running') watchBackground();
    if (state.last) renderConsolidation(state.last);
  }).catch(() => {});
  return root;
}


/* La sezione delle correzioni: cosa e' stato corretto, dove, e come tornare indietro. */
function correctionsSection(projectId, corrections, goToImage) {
  const box = el('div', { class: 'card' });
  box.append(el('h3', { style: 'margin-top:0' }, `correzioni manuali (${corrections.length})`));
  box.append(el('p', { class: 'hint' },
    'ogni riga e\' una singola immagine corretta a mano: il ritaglio e\' quello salvato, e ' +
    'concorre all\'envelope del suo gruppo.'));

  const list = el('div', { style: 'display:flex;flex-direction:column;gap:10px' });
  for (const fix of corrections) {
    const row = el('div', {
      style: 'display:flex;gap:12px;align-items:flex-start;padding:10px;' +
             'border:1px solid var(--line);border-radius:8px',
    });
    row.append(el('img', {
      src: `/api/projects/${projectId}/orientation/crop?name=${encodeURIComponent(fix.name)}&scale=4&pad=2`,
      alt: 'ritaglio',
      style: 'width:64px;height:64px;object-fit:contain;image-rendering:pixelated;' +
             'background:#000;border:1px solid var(--line);border-radius:4px;flex:0 0 auto',
    }));
    const info = el('div', { style: 'flex:1;min-width:0' });
    info.append(el('div', { style: 'word-break:break-all' }, fix.name));
    const meta = [
      `gruppo ${fix.group}`,
      fix.group_before && fix.group_before !== fix.group ? `(era ${fix.group_before})` : '',
      `box ${fix.box.top}|${fix.box.left}|${fix.box.bottom}|${fix.box.right}`,
      fix.score != null ? `score ${fix.score}` : '',
      fix.template ? `template ${fix.template}` : '',
      fix.ts ? `· ${fix.ts.replace('T', ' ')}` : '',
    ].filter(Boolean).join(' · ');
    info.append(el('div', { class: 'hint' }, meta));
    row.append(info);
    const actions = el('div', { style: 'display:flex;flex-direction:column;gap:6px;flex:0 0 auto' });
    actions.append(el('button', { class: 'ghost', onclick: () => goToImage(fix.name) }, 'vai'));
    actions.append(el('button', {
      class: 'ghost',
      onclick: async () => {
        try {
          await api(`/projects/${projectId}/orientation/correction/delete`, { body: { name: fix.name } });
          toast('correzione rimossa');
          await reload();
        } catch (error) { toast(error.message, true); }
      },
    }, 'rimuovi'));
    row.append(actions);
    list.append(row);
  }
  box.append(list);
  return box;
}
