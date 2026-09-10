/* ESIBuilder AI - versione base del wizard. Vanilla JS, nessuna build. */

const state = {
  meta: null, projectId: null, project: null, status: [], step: 'codes',
  advancedStages: { ready: false, blocked_reason: '' },
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === 'class') node.className = value;
    else if (key === 'html') node.innerHTML = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
};

function toast(message, isError = false) {
  const node = $('#toast');
  node.textContent = message;
  node.className = 'toast show' + (isError ? ' error' : '');
  setTimeout(() => { node.className = 'toast'; }, 3600);
}

async function api(path, options = {}) {
  const response = await fetch('/api' + path, {
    method: options.method || (options.body ? 'POST' : 'GET'),
    headers: options.body ? { 'Content-Type': 'application/json' } : {},
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || ('errore ' + response.status));
  return data;
}

/* ---------- shell ---------- */

async function boot() {
  state.meta = await api('/meta');
  await refreshProjects();
  $('#new-project').addEventListener('click', createProject);
  $('#del-project').addEventListener('click', eliminaProgetto);
  $('#project-select').addEventListener('change', (event) => openProject(event.target.value));
}

/* Buttare via un progetto: lo studio, non le immagini. La cartella delle acquisizioni non
   e' nostra e non si tocca - e va detto, se no non si sa cosa si sta cancellando. */
async function eliminaProgetto() {
  if (!state.projectId) { toast('scegli prima un progetto', true); return; }
  const nome = ((state.project || {}).codes || {}).project_name || state.projectId;
  const conferma = prompt(
    `Eliminare il progetto «${nome}»?\n\n`
    + 'Si cancella lo studio: riconoscimenti, correzioni, riquadri, artefatti dei moduli.\n'
    + 'La cartella delle immagini NON viene toccata.\n\n'
    + `Per confermare scrivi:  ${state.projectId}`);
  if (conferma === null) return;
  if (conferma.trim() !== state.projectId) { toast('nome non corrispondente: non elimino niente', true); return; }
  try {
    const esito = await api(`/projects/${state.projectId}`, { method: 'DELETE' });
    toast(`progetto eliminato${esito.unlinked && esito.unlinked.length
      ? ` · sciolto il legame con ${esito.unlinked.join(', ')}` : ''}`);
    state.projectId = null;
    state.project = null;
    $('#panel').innerHTML = '<p class="empty">Crea o seleziona un progetto per iniziare.</p>';
    $('#steps').innerHTML = '';
    await refreshProjects();
  } catch (errore) { toast(errore.message, true); }
}

async function refreshProjects() {
  const { projects } = await api('/projects');
  const select = $('#project-select');
  select.innerHTML = '';
  select.append(el('option', { value: '' }, '— progetto —'));
  for (const project of projects) {
    select.append(el('option', { value: project.project_id },
      `${project.name}  (${project.steps_done}/${project.steps_total})`));
  }
  if (state.projectId) select.value = state.projectId;
}

async function createProject() {
  const name = $('#new-project-name').value.trim();
  if (!name) return toast('serve un nome', true);
  try {
    const { project_id } = await api('/projects', { body: { name } });
    $('#new-project-name').value = '';
    await refreshProjects();
    await openProject(project_id);
  } catch (error) { toast(error.message, true); }
}

async function openProject(projectId) {
  state.projectId = projectId || null;
  if (!projectId) { $('#steps').innerHTML = ''; $('#panel').innerHTML = ''; return; }
  const data = await api('/projects/' + projectId);
  state.project = data.project;
  state.status = data.status;
  state.advancedStages = data.advanced_stages || { ready: true, blocked_reason: '' };
  state.splitPending = data.split_pending || null;
  state.fssPath = data.fss_path;
  $('#project-select').value = projectId;
  render();
}

async function reload() {
  if (state.projectId) await openProject(state.projectId);
}

function render() {
  renderSteps();
  renderHead();
  renderPanel();
}

function renderHead() {
  const relevant = state.status.filter((s) => s.relevant);
  const done = relevant.filter((s) => ['confirmed', 'corrected', 'manual'].includes(s.status));
  const stale = state.status.filter((s) => s.status === 'stale');
  const parts = [`${done.length}/${relevant.length} step completati`];
  if (stale.length) parts.push(`${stale.length} da rivedere`);
  const head = $('#head-status');
  head.innerHTML = '';
  head.append(parts.join(' · '), ' ');
  const first = relevant.find((s) => !['confirmed', 'corrected', 'manual'].includes(s.status));
  if (first) {
    head.append(el('button', { class: 'ghost', onclick: () => selectStep(first.id) },
      'vai al primo problema'));
  }
}

function renderSteps() {
  const nav = $('#steps');
  nav.innerHTML = '';
  for (const step of state.status) {
    const lines = step.fss_lines.length
      ? '#' + step.fss_lines.map((n) => String(n).padStart(2, '0')).join(' #')
      : '—';
    nav.append(el('div', {
      class: 'step' + (state.step === step.id ? ' active' : '') + (step.relevant ? '' : ' skipped'),
      onclick: () => selectStep(step.id),
    },
      el('div', { class: 'dot ' + step.status }),
      el('div', {},
        step.label,
        el('span', { class: 'meta' }, `${step.legacy_page} · ${lines}`)),
    ));
  }
}

function selectStep(stepId) { state.step = stepId; render(); }

/* ---------- pannelli ---------- */

const PANELS = {
  codes: panelCodes,
  import: panelImport,
  vendor: panelVendor,
  probe: panelProbe,
  rect: panelRect,
  orientation: panelModuleStage,
  depth_scale: panelModuleStage,
  scale_study: panelModuleStage,
  generate: panelGenerate,
};

const VALUE_HINTS = {
  biplana: `{"rect_trans": [<4 blocchi NF/LR/UD/LRUD>], "id_next_probe": 42, "recognition_mode": 0}`,
  proibited: `{"groups": [[<blocchi NF>], [<blocchi LR>], [<blocchi UD>], [<blocchi LRUD>]]}`,
  orientation: `{"group_orientation": 4, "blocks": [<blocco NF>, <LR>, <UD>, <LRUD>]}`,
  depth_scale: `{"depths": [15,22,30], "pixel_ratio_x": [...], "pixel_ratio_y": [...],
 "scale_lines": [{"x1":1532,"x2":1532,"y1":171,"y2":388,"length_mm":10,"tick_mm":0.5,"label_side":-1}]}`,
  depth_find: `{"groups": [[<4 blocchi per la depth 1>], [<4 blocchi per la depth 2>]]}`,
  thresholds: `{"note": "le soglie TH vivono dentro i blocchi degli altri step"}`,
  guides: `{"angles": [-179.952], "centre_distance": [[11.2861], [11.2322]]}`,
};

const BLOCK_HINT = `un blocco e' {"top":6,"left":23,"bottom":49,"right":166,"check":1,
 "params":{"threshold":47489100.0}}  — CH, MM, P1 e P2 arrivano dai default`;

function renderPanel() {
  const panel = $('#panel');
  // La lente appartiene alla sezione che la usa: cambiando pannello quella di prima non c'e'
  // piu', e continuare a trascinarci dentro vorrebbe dire scrivere su un pannello morto.
  if (typeof Lente !== 'undefined') Lente.sospendi();
  // L'anteprima al passaggio del mouse vive attaccata al body: cambiando sezione va tolta,
  // se no resta appesa sopra a una pagina che non la riguarda piu'.
  document.querySelectorAll('.piani-lente').forEach((n) => n.remove());
  panel.innerHTML = '';
  const step = state.status.find((s) => s.id === state.step);
  if (!step) return;
  panel.append(el('h2', {}, step.label));
  const lines = step.fss_lines.length ? ' · righe #' + step.fss_lines.join(' #') : '';
  panel.append(el('p', { class: 'hint' },
    `pagina legacy: ${step.legacy_page}${lines} · stato: ${step.status}` +
    (step.relevant ? '' : ' · non applicabile a questa sonda')));
  (PANELS[state.step] || panelGeneric)(panel, step);
}

function field(label, value, onInput, type = 'text') {
  const input = el('input', { type, value: value ?? '' });
  input.addEventListener('input', () => onInput(input.value));
  return el('div', {}, el('label', {}, label), input);
}

function selectField(label, value, options, onChange) {
  const select = el('select', {});
  for (const [key, text] of Object.entries(options)) {
    select.append(el('option', { value: key, selected: String(value) === String(key) ? '' : null }, text));
  }
  select.addEventListener('change', () => onChange(select.value));
  return el('div', {}, el('label', {}, label), select);
}

async function applyCodes(patch, message) {
  try {
    const result = await api(`/projects/${state.projectId}/codes`, { body: { codes: patch } });
    toast(result.stale.length ? `${message} · tornati in review: ${result.stale.join(', ')}` : message);
    await reload();
  } catch (error) { toast(error.message, true); }
}

/* --- step 0: codici --- */
function panelCodes(panel) {
  const codes = { ...state.project.codes };
  const meta = state.meta;
  const set = (key) => (value) => { codes[key] = value; };
  const setInt = (key) => (value) => { codes[key] = parseInt(value || '0', 10) || 0; };
  const size = (key, index) => (value) => {
    const current = (codes[key] || [0, 0]).slice();
    current[index] = parseInt(value || '0', 10) || 0;
    codes[key] = current;
  };

  panel.append(anagraficaCard(codes));

  const filled = ((state.project.analysis || {}).codes_filled) || {};
  if (Object.keys(filled).length) {
    const card = el('div', { class: 'card' });
    card.append(el('h3', { style: 'margin-top:0' }, 'compilati dall\'analisi del passo 0'));
    for (const [key, source] of Object.entries(filled)) {
      card.append(el('div', { class: 'kv' },
        el('span', {}, key), el('span', {}, `${codes[key]}  —  ${source}`)));
    }
    panel.append(card);
  }

  panel.append(el('h3', {}, 'Codici del file'));
  panel.append(el('div', { class: 'grid' },
    field('Nome progetto', codes.project_name, set('project_name')),
    field('Versione file (#01)', codes.fss_version, set('fss_version')),
    field('ID setup', codes.id_setup, setInt('id_setup'), 'number'),
    field('ID ecografo (#02)', codes.id_echo, setInt('id_echo'), 'number'),
    field('ID sonda (#03)', codes.id_probe, setInt('id_probe'), 'number'),
    field('ID file NDG (#05)', codes.id_ndg, setInt('id_ndg'), 'number'),
    selectField('Tipo sonda (#04)', codes.probe_type, meta.probe_types, (v) => { codes.probe_type = parseInt(v, 10); }),
    selectField('Input video (#06)', codes.video_type, meta.video_inputs, (v) => { codes.video_type = parseInt(v, 10); }),
    selectField('Gruppo orientamento (#12)', codes.group_orientation, meta.group_orientations,
      (v) => { codes.group_orientation = parseInt(v, 10); }),
    selectField('Metodo biplana (#26)', codes.biplana_test_method ?? 0, meta.biplana_modes,
      (v) => { codes.biplana_test_method = parseInt(v, 10); }),
    field('ID fss sonda gemella (#25)', codes.id_fss_next_probe ?? '', setInt('id_fss_next_probe'), 'number'),
    field('Video input X (#07)', (codes.video_input_size || [])[0], size('video_input_size', 0), 'number'),
    field('Video input Y (#08)', (codes.video_input_size || [])[1], size('video_input_size', 1), 'number'),
    field('Immagine campione X (#09)', (codes.image_sample_size || [])[0], size('image_sample_size', 0), 'number'),
    field('Immagine campione Y (#10)', (codes.image_sample_size || [])[1], size('image_sample_size', 1), 'number'),
  ));

  panel.append(el('h3', {}, 'Orientamenti disponibili'));
  const markerStage = ((state.project.analysis || {}).stages || {}).marker || {};
  if (markerStage.orientation_available) {
    panel.append(el('p', { class: 'hint' },
      'compilati dal marker: ' +
      Object.entries(markerStage.orientation_available)
        .map(([group, ok]) => `${group} ${ok ? 'presente' : 'assente'}`).join(' · ')));
  }
  const available = { ...(codes.orientation_available || {}) };
  const row = el('div', { class: 'row' });
  for (const key of state.meta.orientation_keys) {
    const box = el('input', { type: 'checkbox', checked: available[key] ? '' : null });
    box.addEventListener('change', () => { available[key] = box.checked; codes.orientation_available = available; });
    row.append(el('label', { style: 'display:flex;gap:6px;align-items:center;color:var(--text)' }, box, key));
  }
  panel.append(row);

  panel.append(el('div', { class: 'row' },
    el('button', {
      onclick: async () => {
        try {
          const result = await api(`/projects/${state.projectId}/codes`, { body: { codes } });
          toast(result.stale.length
            ? 'codici salvati · tornati in review: ' + result.stale.join(', ')
            : 'codici salvati');
          await reload();
        } catch (error) { toast(error.message, true); }
      },
    }, 'Salva codici'),
    el('span', { class: 'hint' }, 'i codici restano modificabili in qualsiasi momento: cambia solo cio\' che dipende da loro'),
  ));
}

/* --- step 0: import e analisi --- */
async function pollJob(jobId, status) {
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, 700));
    const job = await api('/jobs/' + jobId);
    const progress = job.total ? ` ${job.done}/${job.total}` : '';
    if (status) status.textContent = `${job.stage}${progress}...`;
    if (job.status === 'done') return job;
    if (job.status === 'error') throw new Error(job.error || 'analisi fallita');
  }
}

function panelImport(panel) {
  const value = (state.project.steps.import || {}).value || {};
  const analysis = state.project.analysis || {};

  panel.append(el('p', { class: 'hint' },
    'primo tempo: dedup, rotazione, ecografo, sonda e abbozzo del rettangolo. Poi controlla '
    + 'il piano di ogni immagine e, se la cartella contiene sia L sia T, dividila in due '
    + 'progetti. Orientamento, depth e scala si calcolano solo dopo, su un piano per volta.'));

  const input = el('input', {
    type: 'text', value: value.folder || state.project.source.folder || '',
    placeholder: '/percorso/della/cartella/acquisizioni', style: 'width:520px',
  });
  const sfoglia = el('button', { class: 'ghost' }, 'Sfoglia…');
  sfoglia.addEventListener('click', () => apriSelettoreCartella({
    start: input.value,
    onPick: (percorso) => { input.value = percorso; },
  }));
  panel.append(el('div', {}, el('label', {}, 'Cartella acquisizioni'),
    el('div', { class: 'row' }, input, sfoglia)));

  const status = el('span', { class: 'hint' }, 'la cartella grande su disco esterno richiede qualche minuto');
  const button = el('button', {});
  button.textContent = value.folder ? 'Reimporta e rianalizza' : 'Importa e analizza';
  button.addEventListener('click', async () => {
    button.disabled = true;
    try {
      const { job_id } = await api(`/projects/${state.projectId}/import`, { body: { folder: input.value } });
      await pollJob(job_id, status);
      toast('analisi iniziale completata · ora controlla il piano L/T');
      await reload();
    } catch (error) {
      toast(error.message, true);
      status.textContent = error.message;
    } finally { button.disabled = false; }
  });
  panel.append(el('div', { class: 'row' }, button, status));

  if (!value.folder) return;

  panel.append(el('h3', {}, 'cartella'));
  const card = el('div', { class: 'card' });
  const rotation = value.rotation_detail || {};
  const doppie = value.duplicates || {};
  const uguali = (doppie.identical || []).length;
  const orologio = (doppie.timestamp || []).length;
  const rilevamentoOra = value.timestamp_detection || {};
  const statoAreaOra = value.timestamp_box
    ? (rilevamentoOra.source === 'ocr' && rilevamentoOra.reliable
      ? 'riconosciuta automaticamente'
      : rilevamentoOra.source === 'user' ? 'corretta dall\'utente' : 'area salvata')
    : (value.timestamp_disabled ? 'disattivata dall\'utente'
      : rilevamentoOra.box ? 'proposta da controllare' : 'non riconosciuta');
  for (const [key, val] of [
    ['immagini trovate', value.images_total_raw],
    ['immagini tenute', value.images_total],
    ['scartate perche\' identiche', uguali],
    ['scartate perche\' cambia solo l\'ora',
      value.timestamp_box ? orologio : statoAreaOra],
    ['input video (#06)', state.meta.video_inputs[value.video_input] || value.video_input],
    ['video input (#07/#08)', (value.video_input_size || []).join(' x ')],
    ['immagine campione (#09/#10)', (value.image_sample_size || []).join(' x ')],
    ['rotazione', `${value.rotation_applied ?? 0}° — ${rotation.reason || value.rotation_source || '—'}`],
  ]) card.append(el('div', { class: 'kv' }, el('span', {}, key), el('span', {}, String(val))));
  panel.append(card);
  for (const warning of value.warnings || []) {
    panel.append(el('ul', { class: 'problems' }, el('li', {}, warning)));
  }

  /* L'area dell'orologio.

     Due fotogrammi della stessa scena presi a un secondo di distanza differiscono in ogni
     byte del file e in nessun pixel tranne l'ora: nessuna misura li trova uguali, e restano
     tutti e due a pesare due volte in ogni mediana dei moduli. Il programma propone l'area
     leggendo data e ora su piu' immagini; l'utente vede il risultato e puo' correggerlo. */
  const anteprima = rilevamentoOra.preview_image
    || ((state.project.steps.vendor || {}).value || {}).preview_image
    || (value.images || [])[0]?.name;
  if (anteprima) {
    panel.append(el('h3', {}, 'Data e ora escluse dal confronto'));
    const descrizioneOra = el('p', { class: 'hint' });
    if (value.timestamp_box) {
      descrizioneOra.textContent = rilevamentoOra.source === 'ocr' && rilevamentoOra.reliable
        ? 'area riconosciuta automaticamente e gia\' usata nella deduplicazione. Controlla '
          + 'il riquadro rosa: se comprende testo sbagliato, trascinalo attorno alla sola '
          + 'data/ora e applica la correzione.'
        : 'questa e\' l\'area scelta dall\'utente e gia\' usata nella deduplicazione. Puoi '
          + 'ancora trascinarla e applicare una nuova correzione.';
    } else if (rilevamentoOra.box && !value.timestamp_box) {
      descrizioneOra.textContent = 'il sistema ha trovato una possibile data/ora ma non ha '
        + 'abbastanza conferme per usarla da solo. Controlla il riquadro rosa, correggilo se '
        + 'serve e confermalo.';
    } else if (value.timestamp_disabled) {
      descrizioneOra.textContent = 'il riconoscimento automatico e\' disattivato per questo '
        + 'progetto. Puoi riattivarlo con «Riconosci di nuovo».';
    } else {
      descrizioneOra.textContent = 'il sistema non ha letto una data o un\'ora con sufficiente '
        + 'sicurezza. Puoi disegnare il riquadro a mano oppure riprovare il riconoscimento.';
    }
    panel.append(descrizioneOra);
    if (rilevamentoOra.reason) {
      const percentuale = rilevamentoOra.source === 'ocr'
        && Number.isFinite(Number(rilevamentoOra.confidence))
        ? ` · confidenza ${Math.round(Number(rilevamentoOra.confidence) * 100)}%` : '';
      const letture = (rilevamentoOra.texts || []).slice(0, 3).join(' · ');
      panel.append(el('p', { class: 'hint timestamp-detection' },
        `${rilevamentoOra.reason}${percentuale}${letture ? ` · letto: ${letture}` : ''}`));
    }
    // Se l'OCR non trova niente resta un piccolo box iniziale: l'utente non deve costruire
    // quattro lati da zero, ma soltanto portarlo e stringerlo sulla scritta corretta.
    const misura = value.image_sample_size || [1920, 1080];
    const boxOra = {
      timestamp: value.timestamp_box ? { ...value.timestamp_box }
        : rilevamentoOra.box ? { ...rilevamentoOra.box } : {
        left: Math.round(misura[0] * 0.02), top: Math.round(misura[1] * 0.02),
        right: Math.round(misura[0] * 0.16), bottom: Math.round(misura[1] * 0.06),
      },
    };
    const editor = createBoxEditor({
      imageSrc: `/api/projects/${state.projectId}/image`
        + `?name=${encodeURIComponent(anteprima)}&w=980`,
      boxes: boxOra,
      sampleSize: value.image_sample_size,
      projectId: state.projectId,
      imageName: anteprima,
      onChange: () => { statoOra.textContent = ''; },
    });
    panel.append(editor.root);
    const statoOra = el('span', { class: 'hint' });
    const applica = el('button', {}, value.timestamp_box
      ? 'Applica la correzione e rifai la deduplicazione'
      : 'Conferma quest\'area e rifai la deduplicazione');
    applica.addEventListener('click', async () => {
      const box = editor.boxes.timestamp;
      if (!box) { toast('disegna prima il rettangolo attorno all\'ora', true); return; }
      applica.disabled = true;
      try {
        const avvio = await api(`/projects/${state.projectId}/import/timestamp`, { body: { box } });
        const job = await pollJob(avvio.job_id, statoOra);
        const r = job.result || {};
        toast(`${r.kept} immagini tenute · ${r.identical} identiche e ${r.timestamp} `
          + 'uguali a meno dell\'ora, scartate');
        await reload();
      } catch (errore) { toast(errore.message, true); statoOra.textContent = errore.message; }
      finally { applica.disabled = false; }
    });
    const riconosci = el('button', { class: 'ghost' }, rilevamentoOra.reason
      ? 'Riconosci di nuovo' : 'Riconosci automaticamente');
    riconosci.addEventListener('click', async () => {
      riconosci.disabled = true;
      try {
        const avvio = await api(`/projects/${state.projectId}/import/timestamp/detect`, { body: {} });
        const job = await pollJob(avvio.job_id, statoOra);
        if ((job.result || {}).applied) {
          toast('data e ora riconosciute automaticamente · deduplicazione rifatta');
        } else {
          toast('trovata una proposta da controllare', true);
        }
        await reload();
      } catch (errore) { toast(errore.message, true); statoOra.textContent = errore.message; }
      finally { riconosci.disabled = false; }
    });
    const riga = el('div', { class: 'row' }, applica, riconosci,
      Lente.bottone(editor.contestoLente), statoOra);
    if (value.timestamp_box) {
      riga.append(confermaInDueTempi('Togli l\'area', 'si torna a confrontare tutta l\'immagine.',
        async () => {
          try {
            const avvio = await api(`/projects/${state.projectId}/import/timestamp`,
              { body: { reset: true } });
            await pollJob(avvio.job_id, statoOra);
            toast('area tolta');
            await reload();
          } catch (errore) { toast(errore.message, true); }
        }));
    }
    panel.append(riga);
    const elenco = [...((value.duplicates || {}).timestamp || []),
                    ...((value.duplicates || {}).identical || [])];
    if (elenco.length) {
      const dett = el('details', { class: 'ov-fold' },
        el('summary', {}, `le ${elenco.length} immagini scartate`));
      const corpo = el('div', { class: 'depth-rimaste' });
      for (const voce of elenco.slice(0, 200)) {
        corpo.append(el('div', { class: 'hint' },
          `${voce.name} — ${voce.kind === 'identiche' ? 'identica a' : 'uguale a meno dell\'ora a'} ${voce.of}`));
      }
      dett.append(corpo);
      panel.append(dett);
    }
  }

  /* I due piani.

     Una cartella puo' contenere sia L sia T: la sonda e' la stessa ma sono due studi
     diversi, e finche' stanno insieme ogni modulo li mescola - il rettangolo esce mediato
     fra due ventagli, la depth su due scale. Si riconosce dopo dedup e rotazione, perche'
     la rete guarda dentro al rettangolo. */
  panel.append(el('h3', {}, 'Piano L / T'));
  const pianiBox = el('div', {});
  panel.append(pianiBox);
  const renderPiani = () => {
    pianiBox.innerHTML = '';
    const conteggi = value.plane_counts || {};
    const quantiL = conteggi.L || 0;
    const quantiT = conteggi.T || 0;
    const senza = conteggi['?'] || 0;
    const stato = el('span', { class: 'hint' });
    if (!Object.keys(conteggi).length) {
      pianiBox.append(el('p', { class: 'hint' },
        'non ancora riconosciuto. Serve il rettangolo ecografico, perche\' la rete guarda '
        + 'li\' dentro.'));
    } else {
      pianiBox.append(el('div', { class: 'kv' }, el('span', {}, 'immagini in L'),
        el('span', {}, String(quantiL))));
      pianiBox.append(el('div', { class: 'kv' }, el('span', {}, 'immagini in T'),
        el('span', {}, String(quantiT))));
      if (senza) {
        pianiBox.append(el('div', { class: 'kv' }, el('span', {}, 'senza piano riconosciuto'),
          el('span', { style: 'color:var(--warn)' }, String(senza))));
      }
    }
    if (value.plane === 'T' || (state.project.source || {}).plane === 'T') {
      pianiBox.append(el('p', { class: 'hint' },
        'questo e\' il progetto della T, nato dallo sdoppiamento.'));
    }
    // Correggere un piano non sposta l'immagine da sola: finche' non si riconferma la
    // divisione, i moduli continuano a girare su quella di prima - e non si vedeva.
    const sospese = state.splitPending || {};
    if (sospese.to_send || sospese.to_take) {
      const quante = (sospese.to_send || 0) + (sospese.to_take || 0);
      pianiBox.append(el('p', { class: 'avviso' },
        `${quante} immagini stanno ancora nel progetto sbagliato: le tue correzioni sono `
        + 'salvate ma non applicate. Premi «Conferma la divisione» qui sotto, altrimenti i '
        + 'moduli continuano a girare anche su quelle dell\'altro piano.'
        + ((sospese.examples || []).length
          ? ` Per esempio: ${(sospese.examples || []).join(', ')}.` : '')));
    }
    const riconosci = el('button', { class: 'ghost' },
      Object.keys(conteggi).length ? 'Rifai il riconoscimento del piano' : 'Riconosci il piano di ogni immagine');
    riconosci.addEventListener('click', async () => {
      riconosci.disabled = true;
      try {
        const avvio = await api(`/projects/${state.projectId}/planes`, { body: {} });
        const job = await pollJob(avvio.job_id, stato);
        stato.textContent = '';
        const c = (job.result || {}).counts || {};
        toast(`L: ${c.L || 0} · T: ${c.T || 0}${c['?'] ? ` · senza piano: ${c['?']}` : ''}`);
        await reload();
      } catch (errore) { toast(errore.message, true); stato.textContent = errore.message; }
      finally { riconosci.disabled = false; }
    });
    const riga = el('div', { class: 'row' }, riconosci, stato);
    const gia = (state.project.source || {}).split_into;
    if (quantiL && quantiT) {
      riga.append(confermaInDueTempi(
        gia ? `Conferma la divisione: ${quantiL} in L qui, ${quantiT} in «${gia}»`
            : `Sdoppia: ${quantiL} in L qui, ${quantiT} in T in un progetto nuovo`,
        gia
          ? 'i due progetti che ci sono gia\' vengono riportati alla divisione di adesso. '
            + 'Quello che hai gia\' guardato di la\' non viene toccato: si aggiornano le '
            + 'immagini, non il lavoro.'
          : 'la L resta in questo progetto, la T ne apre uno accanto che eredita codici, '
            + 'vendor col suo template, sonda, rettangolo e il rapporto pixel/mm della '
            + 'scala. E\' una copia: da li\' in poi le due configurazioni vivono per conto loro.',
        async () => {
          try {
            const esito = await api(`/projects/${state.projectId}/split`, { body: {} });
            toast(esito.updated
              ? `${esito.created} aggiornato: ${esito.L} in L, ${esito.T} in T`
              : `creato ${esito.created}: ${esito.T} immagini in T`);
            await refreshProjects();
            await reload();
          } catch (errore) { toast(errore.message, true); }
        }));
    }
    if (gia) {
      riga.append(el('span', { class: 'hint' }, `la T sta in «${gia}»`));
    }
    pianiBox.append(riga);

    /* Le due file di immagini, per guardare se la divisione e' venuta giusta. Un numero
       («28 in L, 28 in T») non dice niente su *quali*: e sbagliarne una vuol dire portare
       un fotogramma dell'altro piano dentro a tutte le mediane dei moduli. Un clic su
       un'immagine la sposta di la'. */
    const piani = value.planes || {};
    const correzioni = value.plane_corrections || {};
    const dettoDa = (nome) => correzioni[nome] || (piani[nome] || {}).plane || '?';
    const tutte = Object.keys(piani);
    if (tutte.length) {
      const ricontaEDisegna = () => {
        const c = { L: 0, T: 0, '?': 0 };
        for (const n of tutte) c[dettoDa(n)] = (c[dettoDa(n)] || 0) + 1;
        value.plane_counts = c;
        renderPiani();
      };
      const sposta = async (nomi, verso) => {
        const elenco = Array.isArray(nomi) ? nomi : [nomi];
        if (!elenco.length) return;
        try {
          await api(`/projects/${state.projectId}/planes/correct`,
            { body: { names: elenco, plane: verso } });
          for (const n of elenco) correzioni[n] = verso;
          value.plane_corrections = correzioni;
          scelte.clear();
          // La correzione e' gia' scritta nel progetto: dirlo evita di cercare un tasto
          // «salva» che non c'e', e di non fidarsi di quello che si e' appena fatto.
          toast(elenco.length === 1
            ? `salvata: ${elenco[0].split('/').pop()} in ${verso}`
            : `salvate: ${elenco.length} immagini in ${verso}`);
          ricontaEDisegna();
        } catch (errore) { toast(errore.message, true); }
      };
      /* L'anteprima grande al passaggio del mouse. I francobolli servono a vedere quante
         sono e in che fila stanno; per giudicare *se* una e' L o T serve guardarla, e
         aprirla ogni volta sarebbe un giro lungo per una domanda di un secondo. */
      // Uno solo, riusato: `renderPiani` gira a ogni correzione, e crearne uno ogni volta
      // ne lasciava una pila attaccata al body.
      const lente = document.querySelector('.piani-lente')
        || document.body.appendChild(el('div', { class: 'piani-lente' }, el('img', { alt: '' })));
      lente.style.display = 'none';
      const mostraLente = (nome, evento) => {
        const img = lente.querySelector('img');
        const voluta = `/api/projects/${state.projectId}/image`
          + `?name=${encodeURIComponent(nome)}&w=760`;
        if (img.getAttribute('src') !== voluta) img.setAttribute('src', voluta);
        lente.style.display = 'block';
        // Si mette accanto al cursore, e si sposta dall'altra parte se sborderebbe: una
        // anteprima tagliata dal bordo dello schermo non fa vedere proprio la parte che
        // interessa.
        const largo = 420, alto = 320;
        let x = evento.clientX + 18;
        let y = evento.clientY + 18;
        if (x + largo > window.innerWidth) x = evento.clientX - largo - 18;
        if (y + alto > window.innerHeight) y = Math.max(8, window.innerHeight - alto - 8);
        lente.style.left = `${Math.max(8, x)}px`;
        lente.style.top = `${y}px`;
      };
      const nascondiLente = () => { lente.style.display = 'none'; };

      /* La selezione. Clic sceglie, shift+clic prende tutto quello che sta in mezzo: quando
         un'acquisizione intera e' finita dalla parte sbagliata sono venti immagini di
         fila, e prenderle una per una e' venti volte lo stesso gesto. Il perno e' l'ultima
         toccata, come in qualunque elenco. */
      const scelte = state.__pianiScelte || (state.__pianiScelte = new Set());
      let perno = state.__pianiPerno || null;
      const clicCella = (nome, ordineFila, evento) => {
        if (evento.shiftKey && perno && ordineFila.includes(perno)) {
          const da = ordineFila.indexOf(perno);
          const a2 = ordineFila.indexOf(nome);
          for (const n of ordineFila.slice(Math.min(da, a2), Math.max(da, a2) + 1)) scelte.add(n);
        } else if (scelte.has(nome)) {
          scelte.delete(nome);
        } else {
          scelte.add(nome);
        }
        perno = nome;
        state.__pianiPerno = nome;
        renderPiani();
      };

      const fila = (piano) => {
        const nomi = tutte.filter((n) => dettoDa(n) === piano).sort();
        const altro = piano === 'T' ? 'L' : 'T';
        const box = el('div', { class: 'piani-fila' });
        const scelteQui = nomi.filter((n) => scelte.has(n));
        const testa = el('div', { class: 'row', style: 'margin:0 0 2px' });
        testa.append(el('span', { class: 'hint' },
          `${piano === '?' ? 'senza piano' : piano} — ${nomi.length} immagini`
          + (piano === '?' ? ' (restano con la L)' : '')));
        if (nomi.length) {
          const tutta = el('button', { class: 'ghost sq2' },
            scelteQui.length === nomi.length ? 'deseleziona la fila' : 'seleziona la fila');
          tutta.addEventListener('click', () => {
            if (scelteQui.length === nomi.length) for (const n of nomi) scelte.delete(n);
            else for (const n of nomi) scelte.add(n);
            renderPiani();
          });
          testa.append(tutta);
        }
        box.append(testa);
        const strip = el('div', { class: 'piani-strip' });
        for (const nome of nomi) {
          const cella = el('div', { class: 'piani-cella'
            + (correzioni[nome] ? ' corretta' : '') + (scelte.has(nome) ? ' scelta' : '') });
          cella.append(el('img', {
            loading: 'lazy',
            src: `/api/projects/${state.projectId}/image`
              + `?name=${encodeURIComponent(nome)}&w=150`,
            alt: nome,
            title: `${nome}\nclic: scegli · shift+clic: fino a qui · doppio clic: grande`,
          }));
          // Il pulsantino sposta questa e basta, senza passare dalla selezione: per una
          // sola immagine e' il gesto piu' corto.
          const flip = el('button', { class: 'piani-flip', title: `porta in ${altro}` }, `→${altro}`);
          flip.addEventListener('click', (ev) => { ev.stopPropagation(); sposta(nome, altro); });
          cella.append(flip);
          cella.append(el('span', {}, nome.split('/').pop()));
          cella.addEventListener('click', (ev) => clicCella(nome, nomi, ev));
          cella.addEventListener('dblclick', (ev) => { ev.preventDefault(); aSchermoIntero(nome); });
          cella.addEventListener('mouseenter', (e) => mostraLente(nome, e));
          cella.addEventListener('mousemove', (e) => mostraLente(nome, e));
          cella.addEventListener('mouseleave', nascondiLente);
          strip.append(cella);
        }
        box.append(strip);
        return box;
      };
      /* A tutto schermo, per guardarle una per una. Su un francobollo di 118 px il piano si
         indovina; a schermo pieno si vede. Le frecce scorrono, L e T assegnano, Esc chiude:
         cosi' si passa in rassegna una cartella intera senza mai staccare le mani. */
      const aSchermoIntero = (partenza) => {
        const ordine = ['L', 'T', '?'].flatMap((pi) => tutte.filter((n) => dettoDa(n) === pi).sort());
        let dove = Math.max(0, ordine.indexOf(partenza));
        const velo = el('div', { class: 'piani-pieno' });
        const img = el('img', { alt: '' });
        const testa = el('div', { class: 'piani-pieno-testa' });
        const info = el('span', {});
        const inL = el('button', {}, 'porta in L  (L)');
        const inT = el('button', {}, 'porta in T  (T)');
        const chiudi = el('button', { class: 'ghost' }, 'chiudi  (Esc)');
        testa.append(
          el('button', { class: 'ghost sq' }, '‹'),
          el('button', { class: 'ghost sq' }, '›'),
          info, inL, inT, chiudi);
        velo.append(testa, el('div', { class: 'piani-pieno-scena' }, img));
        const mostra = () => {
          const nome = ordine[dove];
          if (!nome) return;
          img.src = `/api/projects/${state.projectId}/image`
            + `?name=${encodeURIComponent(nome)}&w=1600`;
          info.innerHTML = '';
          const piano = dettoDa(nome);
          info.append(
            el('strong', { style: `color:${piano === 'L' ? '#3fb950' : piano === 'T' ? '#d29922' : 'var(--muted)'}` },
              `piano ${piano}`),
            el('span', { class: 'hint' },
              ` · ${dove + 1} di ${ordine.length} · ${nome.split('/').pop()}`
              + (correzioni[nome] ? ' · corretta da te' : '')));
          inL.disabled = piano === 'L';
          inT.disabled = piano === 'T';
        };
        const vai = (delta) => { dove = (dove + delta + ordine.length) % ordine.length; mostra(); };
        const assegna = async (verso) => {
          const nome = ordine[dove];
          if (dettoDa(nome) === verso) return;
          try {
            await api(`/projects/${state.projectId}/planes/correct`,
              { body: { name: nome, plane: verso } });
            correzioni[nome] = verso;
            value.plane_corrections = correzioni;
            mostra();
          } catch (errore) { toast(errore.message, true); }
        };
        const tasti = (ev) => {
          if (ev.key === 'ArrowRight') { ev.preventDefault(); vai(1); }
          else if (ev.key === 'ArrowLeft') { ev.preventDefault(); vai(-1); }
          else if (ev.key === 'l' || ev.key === 'L') assegna('L');
          else if (ev.key === 't' || ev.key === 'T') assegna('T');
          else if (ev.key === 'Escape') via();
        };
        const via = () => {
          document.removeEventListener('keydown', tasti);
          velo.remove();
          // I conteggi e le strisce si rifanno con quello che si e' deciso qui dentro.
          const c = { L: 0, T: 0, '?': 0 };
          for (const n of tutte) c[dettoDa(n)] = (c[dettoDa(n)] || 0) + 1;
          value.plane_counts = c;
          renderPiani();
        };
        testa.children[0].addEventListener('click', () => vai(-1));
        testa.children[1].addEventListener('click', () => vai(1));
        inL.addEventListener('click', () => assegna('L'));
        inT.addEventListener('click', () => assegna('T'));
        chiudi.addEventListener('click', via);
        velo.addEventListener('click', (ev) => { if (ev.target === velo) via(); });
        document.addEventListener('keydown', tasti);
        document.body.append(velo);
        mostra();
      };

      const gallerie = el('div', { class: 'piani-galleria' });
      for (const piano of ['L', 'T', '?']) {
        if (tutte.some((n) => dettoDa(n) === piano)) gallerie.append(fila(piano));
      }
      const aperturaPiena = el('button', { class: 'ghost' }, 'Guardale a tutto schermo');
      aperturaPiena.addEventListener('click', () => aSchermoIntero(tutte[0]));
      const comandi = el('div', { class: 'row', style: 'margin-top:10px' }, aperturaPiena);
      if (scelte.size) {
        const quante = scelte.size;
        for (const verso of ['L', 'T']) {
          const b2 = el('button', {}, `Porta le ${quante} scelte in ${verso}`);
          b2.addEventListener('click', () => sposta([...scelte], verso));
          comandi.append(b2);
        }
        const via = el('button', { class: 'ghost' }, 'deseleziona');
        via.addEventListener('click', () => { scelte.clear(); renderPiani(); });
        comandi.append(via);
      }
      pianiBox.append(comandi);
      pianiBox.append(el('p', { class: 'hint', style: 'margin:4px 0 0' },
        'un clic sceglie, shift+clic prende tutto quello che sta in mezzo, «→L»/«→T» '
        + 'sposta la singola, doppio clic la apre grande. A tutto schermo: frecce per '
        + 'scorrere, L e T per assegnare, Esc per chiudere.'));
      // Due cose diverse, e confonderle fa cercare un tasto «salva» che non esiste.
      const quante = Object.keys(correzioni).length;
      pianiBox.append(el('p', { class: 'hint', style: 'margin:2px 0 0' },
        quante
          ? `le tue ${quante} correzioni sono gia' salvate nel progetto (bordo azzurro). `
          : 'ogni correzione si salva da sola nel momento in cui la fai. '));
      pianiBox.append(el('p', { class: 'hint', style: 'margin:2px 0 0' },
        gia
          ? 'Per portarle nei due progetti premi «Conferma la divisione» qui sopra: '
            + 'senza quello, i progetti restano com\'erano.'
          : 'Per creare i due progetti premi «Sdoppia» qui sopra.'));
      pianiBox.append(gallerie);
    }
  };
  renderPiani();

  if (analysis.vendor || analysis.probe) {
    panel.append(el('h3', {}, 'riconoscimento'));
    const box = el('div', { class: 'card' });
    const vendor = analysis.vendor || {};
    const probe = analysis.probe || {};
    const registry = analysis.registry || {};
    const rect = analysis.rect || {};
    const plane = analysis.plane || {};
    const rows = [
      ['ecografo', vendor.vendor ? `${vendor.vendor} (conf ${vendor.confidence}, margine ${vendor.margin ?? '—'})` : '—'],
      ['sonda (#03)', probe.probe_id != null ? `ID ${probe.probe_id} (conf ${probe.confidence})` : (probe.reason || '—')],
      ['sonda in anagrafica', registry.model ? `${registry.model} — tipi ${JSON.stringify(registry.probe_types || [])}` : '—'],
      ['piano L/T', plane.plane ? `${plane.plane} (${plane.counts ? JSON.stringify(plane.counts) : ''} conf media ${plane.mean_confidence})` : (plane.reason || 'non applicabile')],
      ['rettangolo', rect.agreement_iou != null ? `accordo IoU ${rect.agreement_iou}, modello ${rect.source}` : '—'],
      ['template ecografo (#13)', (analysis.line13 || {}).box
        ? `proposto, accordo IoU ${(analysis.line13 || {}).agreement_iou} (rete ${(analysis.line13 || {}).vendor})`
        : ((analysis.line13 || {}).reason || '—')],
      ['device', analysis.device || '—'],
    ];
    for (const [key, val] of rows) box.append(el('div', { class: 'kv' }, el('span', {}, key), el('span', {}, String(val))));
    panel.append(box);
    const filled = analysis.codes_filled || {};
    const keys = Object.keys(filled);
    if (keys.length) {
      panel.append(el('p', { class: 'hint' },
        'codici compilati automaticamente: ' + keys.map((k) => `${k} (${filled[k]})`).join(' · ')));
    }
  }

  if ((value.images || []).length) {
    panel.append(el('h3', {}, `immagini (${value.images_listed} di ${value.images_total})`));
    const thumbs = el('div', { class: 'thumbs' });
    for (const image of value.images) {
      thumbs.append(el('figure', { style: 'margin:0' },
        el('img', { src: `/api/projects/${state.projectId}/image?name=${encodeURIComponent(image.name)}&w=260`, loading: 'lazy' }),
        el('figcaption', {}, image.name)));
    }
    panel.append(thumbs);
  }
}

/* --- step 2: ecografo e sonda --- */
/* --- Ecografo, sonda e rettangolo: tre schermate ---------------------------
   Il legacy aveva una pagina sola (PAGE_US) e qui erano tre cose impilate. Si confermano
   con criteri diversi — il vendor col nome della macchina a schermo, la sonda col suo
   nome, il rettangolo con la geometria da cui dipende tutto il resto — quindi ora sono
   tre step, ognuno col suo box e la sua conferma. */

/* La riga dell'analisi: e' la stessa rete per tutti e tre, quindi lo stesso comando. */
function analysisRow(panel, label) {
  const imported = (state.project.steps.import || {}).value || {};
  const status = el('span', { class: 'hint' },
    imported.folder ? label : 'importa prima una cartella');
  const analyse = el('button', {}, 'Rilancia l\'analisi della cartella');
  analyse.disabled = !imported.folder;
  analyse.addEventListener('click', async () => {
    analyse.disabled = true;
    try {
      const { job_id } = await api(`/projects/${state.projectId}/analyze`, { body: { sample: 24 } });
      await pollJob(job_id, status);
      toast('proposta pronta');
      await reload();
    } catch (error) {
      toast(error.message, true);
      status.textContent = error.message;
    } finally { analyse.disabled = false; }
  });
  panel.append(el('div', { class: 'row' }, analyse, status));
  return status;
}

function infoCard(panel, title, rows) {
  const card = el('div', { class: 'card' });
  card.append(el('h3', { style: 'margin-top:0' }, title));
  for (const [key, val] of rows) {
    card.append(el('div', { class: 'kv' }, el('span', {}, key), el('span', {}, String(val ?? '—'))));
  }
  panel.append(card);
  return card;
}

/* Il blocco template: le coordinate si trascinano sull'immagine, qui restano i parametri
   di match e la possibilita' di dire che in questa acquisizione il template non c'e'. */
function templateBlockEditor(panel, value, key, line, note) {
  panel.append(el('h3', {}, `Parametri del template (${line})`));
  if (note) panel.append(el('p', { class: 'hint' }, note));
  const absent = el('input', { type: 'checkbox', checked: value[key] ? null : '' });
  const area = el('textarea', { style: 'min-height:110px' });
  area.value = JSON.stringify(value[key] || null, null, 1);
  area.disabled = !value[key];
  absent.addEventListener('change', () => {
    area.disabled = absent.checked;
    if (absent.checked) {
      value[key] = null;
      area.value = 'null';
    } else if (!value[key]) {
      value[key] = { top: 0, left: 0, bottom: 0, right: 0, check: 1, params: { threshold: 0 } };
      area.value = JSON.stringify(value[key], null, 1);
    }
  });
  area.addEventListener('input', () => {
    try { value[key] = JSON.parse(area.value); } catch (_) { /* validato al salvataggio */ }
  });
  panel.append(el('label', { style: 'display:flex;gap:6px;align-items:center;color:var(--text)' },
    absent, 'template non presente in questa acquisizione'));
  panel.append(area);
  panel.append(el('p', { class: 'hint' }, BLOCK_HINT));
  return area;
}

/* Un editor per un solo box del template, sull'immagine campione. */
function templateBoxPanel(panel, step, value, key, color, emptyHint) {
  const imported = (state.project.steps.import || {}).value || {};
  const sampleSize = imported.image_sample_size || state.project.codes.image_sample_size;
  const dirty = el('span', { class: 'dirty' }, 'modifiche non salvate');
  dirty.style.display = 'none';
  const readout = el('div', { class: 'hint' });
  const original = JSON.stringify(cloneBoxes(value));

  const describe = (box) => box
    ? `${box.right - box.left + 1} x ${box.bottom - box.top + 1} px ` +
      `(top ${box.top} left ${box.left} bottom ${box.bottom} right ${box.right})`
    : 'nessun box';
  const refresh = () => {
    readout.textContent = describe(value[key]);
    dirty.style.display = JSON.stringify(cloneBoxes(value)) === original ? 'none' : 'inline-block';
  };
  const save = async () => {
    try {
      await api(`/projects/${state.projectId}/steps/${step.id}`, { body: { value } });
      toast('box salvato per tutta la cartella');
      await reload();
    } catch (error) { toast(error.message, true); }
  };
  const applyBoxes = (boxes) => {
    if (boxes[key]) {
      value[key] = { ...(value[key] || { check: 1, params: { threshold: 0 } }), ...boxes[key] };
    }
    refresh();
  };

  if (!value.preview_image) {
    panel.append(el('p', { class: 'hint' }, 'nessuna anteprima: lancia prima l\'analisi.'));
    return { refresh, save };
  }
  if (!value[key]) {
    panel.append(el('p', { class: 'hint' }, emptyHint));
    const crea = el('button', { class: 'ghost' }, 'Disegna il box qui sopra');
    crea.addEventListener('click', () => {
      const w = Math.round((sampleSize?.[0] || 1920) * 0.08);
      const h = Math.round((sampleSize?.[1] || 1080) * 0.03);
      value[key] = { top: 10, left: 10, bottom: 10 + h, right: 10 + w,
                     check: 1, params: { threshold: 0 } };
      render();
    });
    panel.append(el('div', { class: 'row' }, crea));
    return { refresh, save };
  }

  const editor = createBoxEditor({
    imageSrc: `/api/projects/${state.projectId}/image` +
      `?name=${encodeURIComponent(value.preview_image)}&w=980`,
    boxes: cloneBoxes(value),
    sampleSize,
    projectId: state.projectId,
    imageName: value.preview_image,
    onChange: applyBoxes,
    onDoubleClick: () => openFullscreenEditor({
      projectId: state.projectId,
      startValue: value,
      sampleSize,
      startImage: value.preview_image,
      onSave: (boxes) => { applyBoxes(boxes); save(); },
    }),
  });
  panel.append(editor.root);
  panel.append(el('div', { class: 'row' }, Lente.bottone(editor.contestoLente),
    el('span', { class: 'hint' }, 'una finestra a parte con il riquadro ingrandito: '
      + 'si trascina sul secondo schermo e segue quello che fai qui')));
  panel.append(el('p', { class: 'hint' },
    'trascina il box o usa le maniglie · doppio clic per aprire a schermo intero e ' +
    'scorrere le immagini · il box vale per tutta la cartella'));
  panel.append(readout);
  panel.append(el('div', { class: 'row' },
    el('button', { onclick: save }, 'Salva il box'),
    dirty,
    el('button', {
      class: 'ghost',
      onclick: () => { render(); toast('modifiche annullate'); },
    }, 'Annulla le modifiche')));
  refresh();
  return { refresh, save };
}

function panelVendor(panel, step) {
  const value = JSON.parse(JSON.stringify((state.project.steps.vendor || {}).value || {}));
  const analysis = state.project.analysis || {};
  const vendor = analysis.vendor || {};
  analysisRow(panel, 'la rete propone il vendor e il suo template, tu confermi');

  infoCard(panel, 'ecografo riconosciuto', [
    ['vendor', vendor.vendor
      ? `${vendor.vendor} (confidenza ${vendor.confidence}, margine ${vendor.margin ?? '—'})` : '—'],
    ['alternative', (vendor.top || []).slice(1).map((t) => `${t.vendor} ${t.prob}`).join(' · ') || '—'],
    ['modello in anagrafica', (state.project.codes || {}).echo_model || '—'],
    ['template #13 proposto da',
      value.rect_name_echo_source
        ? `rete per vendor · accordo IoU ${value.rect_name_echo_agreement ?? '—'}`
        : (value.rect_name_echo_reason || '—')],
  ]);

  panel.append(el('h3', {}, 'Template col nome dell\'ecografo (#13)'));
  panel.append(el('p', { class: 'hint' },
    'e\' il riquadro che contiene la scritta del modello a schermo: ESI lo cerca per capire ' +
    'che e\' quella macchina.'));
  templateBoxPanel(panel, step, value, 'rect_name_echo', '#40d0ff',
    'nessun box proposto per questo vendor: disegnalo tu attorno alla scritta del modello.');
  templateBlockEditor(panel, value, 'rect_name_echo', '#13',
    'le coordinate si modificano sopra, qui restano i parametri di match');
  panel.append(saveRow(step.id, () => value));
}

function panelProbe(panel, step) {
  const value = JSON.parse(JSON.stringify((state.project.steps.probe || {}).value || {}));
  const analysis = state.project.analysis || {};
  const probe = analysis.probe || {};
  const registry = analysis.registry || {};
  const plane = analysis.plane || {};
  analysisRow(panel, 'la rete propone la sonda, l\'anagrafica il modello e il tipo');

  infoCard(panel, 'sonda riconosciuta', [
    ['ID sonda (rete)', probe.probe_id != null
      ? `${probe.probe_id} (confidenza ${probe.confidence ?? '—'})` : '—'],
    ['modello (anagrafica)', registry.model || '—'],
    ['tipi possibili', (registry.probe_types || []).join(' · ') || '—'],
    ['biplana', registry.biplane ? 'si' : 'no'],
    ['piano L/T (rete)', plane.plane
      ? `${plane.plane} (confidenza ${plane.confidence ?? '—'})` : '—'],
    ['tipo sonda nei codici', (state.project.codes || {}).probe_type ?? '—'],
  ]);

  panel.append(el('h3', {}, 'Template col nome della sonda (#14)'));
  panel.append(el('p', { class: 'hint' },
    'il riquadro con la sigla della sonda a schermo. Il resolver storico avrebbe bisogno ' +
    'del corpus legacy .fss, oggi non disponibile: qui il box si disegna a mano.'));
  templateBoxPanel(panel, step, value, 'rect_name_probe', '#d29922',
    'nessun box per la sonda: disegnalo tu attorno alla sigla a schermo.');
  templateBlockEditor(panel, value, 'rect_name_probe', '#14',
    'le coordinate si modificano sopra, qui restano i parametri di match');
  panel.append(saveRow(step.id, () => value));
}


/* --- la catena del rettangolo, come wizard -------------------------------
   I moduli si tengono per mano: l'orientamento si studia dentro il rettangolo, ma il
   rettangolo si rifinisce con l'orientamento; il piano L/T si riconosce dentro il
   rettangolo e serve al rettangolo per cercare il segmento piu' lungo nel piano giusto.
   Si percorre a giri: un rettangolo provvisorio, poi ogni conferma dell'utente ne rende
   possibile uno piu' stretto. Ogni giro propone, non applica. */
/* Conferma in due tempi. `confirm()` non protegge: in questo ambiente un clic vagante lo
   attraversa, e ha gia' applicato due volte un rettangolo che nessuno aveva scelto. Con due
   pulsanti distinti serve un secondo clic, su un elemento che prima non esisteva. */
function confermaInDueTempi(label, testo, azione) {
  const riga = el('div', { class: 'row' });
  const primo = el('button', {}, label);
  primo.addEventListener('click', () => {
    riga.innerHTML = '';
    const conferma = el('button', { style: 'background:var(--warn);color:#101010' }, 'Sì, applica');
    const annulla = el('button', { class: 'ghost' }, 'No');
    conferma.addEventListener('click', async () => {
      conferma.disabled = true;
      await azione();
    });
    annulla.addEventListener('click', () => { riga.innerHTML = ''; riga.append(primo, nota); });
    riga.append(el('span', { class: 'hint', style: 'color:var(--warn)' }, testo), conferma, annulla);
  });
  const nota = el('span', { class: 'hint' }, 'due clic: nessuna modifica per sbaglio');
  riga.append(primo, nota);
  return riga;
}

function rectChainCard(panel, ganciRect) {
  const studyHost = el('div');
  panel.append(studyHost);
  const card = el('details', { class: 'ov-fold', style: 'margin-top:12px' });
  card.append(el('summary', {},
    el('span', {}, 'lo studio delle corde'),
    el('span', { class: 'ov-fold-tag' }, 'come si misura')));
  const body = el('div');
  const detail = el('div');
  card.append(body, detail);
  panel.append(card);

  const STATE_LABEL = {
    done: ['fatto', 'var(--ok)'],
    applied: ['applicato', 'var(--ok)'],
    available: ['si puo\' fare ora', 'var(--accent)'],
    proposed: ['proposta da guardare', 'var(--warn)'],
    waiting: ['aspetta', 'var(--muted)'],
    todo: ['da scrivere', 'var(--muted)'],
  };
  const NEED_LABEL = { orientation: 'orientamento', plane: 'piano L/T', depth: 'depth e scala' };

  const px = (v) => (v == null ? '—' : `${v} px`);

  const showSegments = (r) => {
    detail.innerHTML = '';
    detail.append(el('h3', {}, 'la corda piu\' larga, per gruppo'));
    if (r.plane_conflict) {
      detail.append(el('p', { class: 'hint', style: 'color:var(--err)' }, r.plane_note));
    }
    const table = el('table', { class: 'lines' });
    table.append(el('tr', {},
      el('td', { class: 'name' }, 'gruppo'),
      el('td', { class: 'name' }, 'piano'),
      el('td', { class: 'name' }, 'corda'),
      el('td', { class: 'name' }, 'mediana del gruppo'),
      el('td', { class: 'name' }, 'posizione (x1-x2 · y)')));
    for (const [group, m] of Object.entries(r.per_group)) {
      const s = m.segment;
      table.append(el('tr', {},
        el('td', { class: 'name', style: `color:${GROUP_COLORS_RECT[group]}` }, group),
        el('td', { class: 'val' }, m.plane || '—'),
        el('td', { class: 'val' }, `${s.length_px} px`),
        el('td', { class: 'val' }, `${m.median_length_px} px`),
        el('td', { class: 'val' }, `${s.x1.toFixed(3)}-${s.x2.toFixed(3)} · ${s.y.toFixed(3)}`)));
    }
    detail.append(table);

    detail.append(el('h3', {}, 'specularita\' fra le coppie'));
    const pairs = el('table', { class: 'lines' });
    pairs.append(el('tr', {},
      el('td', { class: 'name' }, 'coppia'),
      el('td', { class: 'name' }, 'asse'),
      el('td', { class: 'name' }, 'stessa posizione'),
      el('td', { class: 'name' }, 'specchiate'),
      el('td', { class: 'name' }, 'tiene')));
    for (const [nome, c] of Object.entries(r.pairs || {})) {
      pairs.append(el('tr', {},
        el('td', { class: 'name' }, nome),
        el('td', { class: 'val' }, c.axis),
        el('td', { class: 'val' }, `${c.identity.max_px} px`),
        el('td', { class: 'val' }, `${c.mirror.max_px} px`),
        el('td', { class: 'val', style: 'color:var(--ok)' }, c.hypothesis)));
    }
    detail.append(pairs);
    detail.append(el('p', { class: 'hint' },
      'sull\'asse x le corde di NF e LR cadono nello stesso posto quando il ventaglio e\' ' +
      'simmetrico attorno al proprio asse: e\' il contenuto a ribaltarsi, non la corda. ' +
      'Sull\'asse y invece il ribaltamento si vede.'));

    detail.append(el('h3', {}, 'proposta'));
    for (const [k, v] of [
      ['unione delle corde', `${r.union.x1.toFixed(3)} - ${r.union.x2.toFixed(3)} ` +
        `(margine ${Math.round(r.margin * 100)}%)`],
      ['rettangolo attuale', `${r.current.top}|${r.current.left}|${r.current.bottom}|${r.current.right}`],
      ['proposta', `${r.proposal.top}|${r.proposal.left}|${r.proposal.bottom}|${r.proposal.right}`],
      ['differenza', `lato peggiore ${r.delta.max_side_px} px`],
    ]) detail.append(el('div', { class: 'kv' }, el('span', {}, k), el('span', {}, String(v))));
    if (r.plane_conflict) {
      detail.append(el('p', { class: 'hint', style: 'color:var(--err)' },
        'applicazione bloccata: i gruppi cadono su piani diversi'));
    } else {
      detail.append(confermaInDueTempi(
        'Applica l\'ampiezza trovata',
        'l\'ampiezza orizzontale viene dalle corde.',
        async () => {
          try {
            await api(`/projects/${state.projectId}/rect/apply`, { body: { pass: 'segmento' } });
            toast('ampiezza applicata');
            await reload();
          } catch (error) { toast(error.message, true); }
        }));
    }
  };

  const load = async () => {
    let chain;
    try {
      chain = await api(`/projects/${state.projectId}/rect/chain`);
    } catch (error) {
      body.innerHTML = '';
      body.append(el('p', { class: 'hint' }, error.message));
      return;
    }
    body.innerHTML = '';
    body.append(el('p', { class: 'hint' },
      'la corda e\' il segmento piu\' largo del ventaglio, cercata nel piano giusto. Fissa '
      + 'l\'ampiezza orizzontale, e le due della coppia devono essere speculari: quanto si '
      + 'scostano dice se il rettangolo e\' al posto giusto.'));
    // Solo il giro delle corde: e' lo studio di questa sezione. Gli altri girano lo stesso
    // - da soli, con l'analisi - ma raccontarli qui riempiva la pagina di roba che non
    // riguarda quello che si sta guardando.
    for (const pass of chain.passes.filter((p) => p.id === 'segmento')) {
      const [label, color] = STATE_LABEL[pass.state] || [pass.state, 'var(--muted)'];
      const riga = el('div', { class: 'chain-step' });
      riga.append(el('span', { class: 'chain-state', style: `color:${color};border-color:${color}` }, label));
      riga.append(el('div', {},
        el('div', { style: 'font-weight:600' }, pass.label),
        el('div', { class: 'hint' }, pass.what),
        pass.missing.length
          ? el('div', { class: 'hint', style: 'color:var(--warn)' },
              'aspetta: ' + pass.missing.map((m) => NEED_LABEL[m] || m).join(', '))
          : null,
        // Un giro mai fatto, con tutto pronto: dire che parte da solo evita di leggere
        // la sezione vuota come "il modulo non sa rispondere".
        (!pass.missing.length && !(pass.saved && pass.saved.proposal))
          ? el('div', { class: 'hint' },
              'gira da solo insieme all\'analisi; qui si rifa\' quando serve')
          : null));
      if (!pass.missing.length) {
        const run = el('button', { class: 'ghost' },
          pass.saved && pass.saved.proposal ? 'Rifai la misura delle corde' : 'Cerca le corde');
        const stato = el('span', { class: 'hint' });
        run.addEventListener('click', async () => {
          run.disabled = true;
          try {
            const started = await api(`/projects/${state.projectId}/rect/refine`,
              { body: { pass: 'segmento', per_group: 6 } });
            const job = await pollJob(started.job_id, stato);
            stato.textContent = '';
            showSegments(job.result);
            await load();
            toast('misura pronta: guarda la proposta prima di applicarla');
          } catch (error) {
            toast(error.message, true);
            stato.textContent = error.message;
          } finally { run.disabled = false; }
        });
        riga.append(el('div', { class: 'row', style: 'margin:0' }, run, stato));
      }
      body.append(riga);
    }
    if (studyHost) { studyHost.innerHTML = ''; rectStudyCard(studyHost, chain, ganciRect); }
  };
  load();
}

const GROUP_COLORS_RECT = { NF: '#ff6040', LR: '#40d0ff', UD: '#d29922', LRUD: '#3fb950' };


/* --- lo studio, disegnato sull'immagine ------------------------------------
   I numeri dei giri di raffinamento da soli non si giudicano: qui si vedono sul frame.
   Per ogni gruppo di orientamento: il rettangolo di quel gruppo, quello attuale, la
   proposta, la corda piu' larga trovata e quella del gruppo speculare — cosi' si vede se
   cadono una sull'altra. */
function rectStudyCard(panel, chainIniziale, ganci) {
  let chain = chainIniziale;
  /* Il rettangolo si modifica **qui**, sull'immagine dello studio: e' l'unica immagine
     della sezione. Prima ce n'erano due, la stessa cosa disegnata due volte a due scale
     diverse - si guardavano le corde su una e si trascinavano i bordi sull'altra. I ganci
     arrivano dal pannello, che e' chi possiede il valore da salvare. */
  const gancio = ganci || {};
  let study = (chain.study || {}).segments;
  const specularita = ((chain.passes || []).find((p) => p.id === 'specularita') || {}).saved || {};
  if (!study && !specularita.per_group) return null;

  const card = el('div', { class: 'card' });
  const titolo = el('h3', { style: 'margin-top:0' }, 'lo studio sull\'immagine');
  card.append(titolo);
  const chips = el('div', { class: 'ov-chips' });
  const stage = el('div', { class: 'editor-stage' });
  const image = el('img', { alt: '' });
  stage.append(image);
  const info = el('div', { class: 'hint', style: 'margin-top:6px' });
  const legenda = el('div', { class: 'ov-legend', style: 'margin-top:4px' });
  card.append(chips);
  panel.append(card);

  const layers = el('div', { class: 'ov-chips', style: 'margin:6px 0' });
  const candidati = el('div', { style: 'margin-top:8px' });
  card.append(layers);
  const size = study ? study.image_size : null;
  let axesData = null;
  /* Cosa si vede aprendo: il rettangolo di adesso, la corda dell'orientamento scelto e
     l'asse che ne esce. Una corda alla volta - si cambia col chip del gruppo, e cosi' si
     guarda una cosa per volta invece di due sovrapposte. La speculare resta un livello,
     per quando serve il confronto della coppia. Sono lo studio, non un livello facoltativo - questa sezione serve a guardare
     quelli. Il resto (rettangolo del gruppo, assi dai marker, centro immagine, marker)
     resta materiale di indagine e si accende quando serve: prima erano dieci disegni
     insieme e non si capiva piu' a cosa si riferissero. */
  const attivi = {
    attuale: true, gruppo: false, corde: true, speculare: false,
    assi_corde: true, assi_marker: false, assi_immagine: false, marker: false,
    candidato: '',
  };
  const gruppi = study ? Object.keys(study.per_group) : Object.keys(specularita.per_group || {});
  const PARTNER = { NF: 'LR', LR: 'NF', UD: 'LRUD', LRUD: 'UD' };
  let corrente = gruppi.includes('NF') ? 'NF' : gruppi[0];

  const scala = () => ({
    x: image.clientWidth / ((size && size[0]) || image.naturalWidth || 1),
    y: image.clientHeight / ((size && size[1]) || image.naturalHeight || 1),
  });
  const boxNode = (box, color, dashed, label) => {
    const s = scala();
    const node = el('div', {
      class: 'editor-box readonly study-box',
      style: `--box-color:${color};left:${box.left * s.x}px;top:${box.top * s.y}px;`
        + `width:${(box.right - box.left) * s.x}px;height:${(box.bottom - box.top) * s.y}px;`
        + (dashed ? 'border-style:dashed;' : ''),
    });
    if (label) node.append(el('span', { class: 'study-tag', style: `background:${color}` }, label));
    return node;
  };
  /* Cosa vede la lente: la stessa immagine e gli stessi disegni di qui sotto - rettangolo,
     corde, assi - perche' la domanda che si fa ingrandendo e' sempre la stessa: questo
     bordo dove cade rispetto alla corda?

     E soprattutto **dove** guarda. Il rettangolo ecografico e' mezzo schermo: inquadrarlo
     vorrebbe dire non ingrandire niente. La mira e' quindi il pezzo che si sta toccando -
     l'angolo sotto le dita mentre lo si trascina - e a riposo la zona delle corde, che e'
     quello che questa sezione studia. */
  let miraLente = null;   // {left, top, right, bottom} in pixel immagine, o null

  const zonaCorde = () => {
    const m = study && study.per_group[corrente];
    if (!m || !m.segment) return null;
    const px = bozza || segPx(m.segment);
    const altro = study.per_group[PARTNER[corrente]];
    const ys = [px.y];
    const xs = [px.x1, px.x2];
    if (altro && altro.segment && attivi.speculare) {
      const q = segPx(altro.segment);
      ys.push(q.y); xs.push(q.x1, q.x2);
    }
    const margine = 40;
    return {
      left: Math.max(0, Math.min(...xs) - margine),
      right: Math.min((size ? size[0] : 1e9), Math.max(...xs) + margine),
      top: Math.max(0, Math.min(...ys) - margine),
      bottom: Math.min((size ? size[1] : 1e9), Math.max(...ys) + margine),
    };
  };

  /* L'angolo che si sta trascinando, come quadratino attorno. `move` non ha un angolo: li'
     si sposta tutto, e la cosa da guardare torna a essere la corda. */
  const zonaAngolo = (box, lato) => {
    if (!lato || lato === 'move') return null;
    const raggio = 90;
    const x = lato.includes('w') ? box.left : lato.includes('e') ? box.right
      : (box.left + box.right) / 2;
    const y = lato.includes('n') ? box.top : lato.includes('s') ? box.bottom
      : (box.top + box.bottom) / 2;
    return { left: x - raggio, top: y - raggio, right: x + raggio, bottom: y + raggio };
  };

  const contestoLente = () => {
    const m = study && study.per_group[corrente];
    const rect = gancio.leggiRect ? gancio.leggiRect() : chain.rect;
    const linee = [];
    const segmenti = [];
    if (axesData && attivi.assi_corde) {
      const a = axesData.axes;
      if (a.corde_x != null) linee.push({ x: a.corde_x, color: '#3fb950', label: 'asse sx↔dx' });
      if (a.corde_y != null) linee.push({ y: a.corde_y, color: '#3fb950', label: 'asse su↔giu' });
    }
    if (m && m.segment && attivi.corde) {
      const px = bozza || segPx(m.segment);
      segmenti.push({ x1: Math.min(px.x1, px.x2), x2: Math.max(px.x1, px.x2), y: px.y,
                      color: GROUP_COLORS_RECT[corrente],
                      label: `corda ${corrente} ${Math.round(Math.abs(px.x2 - px.x1))} px` });
    }
    const altro = study && study.per_group[PARTNER[corrente]];
    if (altro && altro.segment && attivi.speculare) {
      const q = segPx(altro.segment);
      segmenti.push({ x1: Math.min(q.x1, q.x2), x2: Math.max(q.x1, q.x2), y: q.y,
                      color: GROUP_COLORS_RECT[PARTNER[corrente]], dashed: true,
                      label: `corda ${PARTNER[corrente]}` });
    }
    // I bersagli della lente: un clic e va li'. Girare intorno al rettangolo bordo per
    // bordo e' il lavoro vero di questa sezione, e farlo trascinando la vista ogni volta
    // sarebbe un giro lungo per una cosa che si sa gia' dove sta.
    const bersagli = [];
    if (rect) {
      const r = 90;
      const meta = (a, b2) => (a + b2) / 2;
      const attorno = (x, y) => ({ left: x - r, top: y - r, right: x + r, bottom: y + r });
      bersagli.push(
        { id: 'tutto', label: 'tutto il rettangolo', box: { ...rect } },
        { id: 'w', label: 'bordo sx', side: 'w', box: attorno(rect.left, meta(rect.top, rect.bottom)) },
        { id: 'e', label: 'bordo dx', side: 'e', box: attorno(rect.right, meta(rect.top, rect.bottom)) },
        { id: 'n', label: 'bordo alto', side: 'n', box: attorno(meta(rect.left, rect.right), rect.top) },
        { id: 's', label: 'bordo basso', side: 's', box: attorno(meta(rect.left, rect.right), rect.bottom) },
        { id: 'nw', label: 'angolo ↖', side: 'nw', box: attorno(rect.left, rect.top) },
        { id: 'ne', label: 'angolo ↗', side: 'ne', box: attorno(rect.right, rect.top) },
        { id: 'sw', label: 'angolo ↙', side: 'sw', box: attorno(rect.left, rect.bottom) },
        { id: 'se', label: 'angolo ↘', side: 'se', box: attorno(rect.right, rect.bottom) },
      );
    }
    const corde = zonaCorde();
    if (corde) bersagli.push({ id: 'corde', label: `corda ${corrente}`, box: corde });

    return {
      source: 'rect',
      projectId: state.projectId,
      name: image.title || '',
      size: [(size && size[0]) || 0, (size && size[1]) || 0],
      boxes: rect ? [{ box: rect, color: '#ffffff', label: 'rettangolo (#11)' }] : [],
      lines: linee,
      segments: segmenti,
      focus: miraLente || corde || rect || null,
      targets: bersagli,
      caption: miraLente ? 'stai lavorando qui' : 'zona delle corde',
      onTarget: (voce) => { miraLente = voce.box; draw(); },
      onChange: gancio.scriviRect ? (nuovo) => { gancio.scriviRect(nuovo); draw(); } : null,
    };
  };

  /* Il rettangolo di adesso, trascinabile: otto maniglie e il corpo. E' lo stesso gesto che
     stava nell'altro editor, portato sull'immagine dove ci sono anche le corde - che e' il
     motivo per cui lo si sposta: si vede subito se il bordo taglia il ventaglio. */
  const LATI_RECT = ['nw', 'n', 'ne', 'w', 'e', 'sw', 's', 'se'];
  const rettangoloModificabile = (box) => {
    const s = scala();
    const node = el('div', {
      class: 'editor-box study-box' + (gancio.scriviRect ? '' : ' readonly'),
      style: `--box-color:#ffffff;left:${box.left * s.x}px;top:${box.top * s.y}px;`
        + `width:${(box.right - box.left) * s.x}px;height:${(box.bottom - box.top) * s.y}px;`,
    });
    node.append(el('span', { class: 'study-tag', style: 'background:#ffffff;color:#0d1117' },
      `${box.right - box.left}x${box.bottom - box.top}`));
    if (!gancio.scriviRect) return node;
    node.addEventListener('pointerdown', (e) => {
      if (e.target === node) trascinaRect(e, 'move');
    });
    for (const lato of LATI_RECT) {
      const h = el('div', { class: `handle handle-${lato}` });
      h.addEventListener('pointerdown', (e) => trascinaRect(e, lato));
      node.append(h);
    }
    return node;
  };

  function trascinaRect(event, lato) {
    event.preventDefault();
    event.stopPropagation();
    const partenza = { x: event.clientX, y: event.clientY };
    const box = { ...gancio.leggiRect() };
    // La lente segue l'angolo sotto le dita: e' il senso di averla aperta.
    miraLente = zonaAngolo(box, lato);
    const s = scala();
    const [W, H] = [(size && size[0]) || 0, (size && size[1]) || 0];
    const bersaglio = event.currentTarget;
    bersaglio.setPointerCapture(event.pointerId);
    const muovi = (e) => {
      const dx = Math.round((e.clientX - partenza.x) / (s.x || 1));
      const dy = Math.round((e.clientY - partenza.y) / (s.y || 1));
      const n = { ...box };
      if (lato === 'move') {
        const largo = box.right - box.left;
        const alto = box.bottom - box.top;
        n.left = Math.max(0, Math.min(box.left + dx, (W || box.right) - largo));
        n.top = Math.max(0, Math.min(box.top + dy, (H || box.bottom) - alto));
        n.right = n.left + largo;
        n.bottom = n.top + alto;
      } else {
        if (lato.includes('w')) n.left = Math.min(box.left + dx, box.right - 4);
        if (lato.includes('e')) n.right = Math.max(box.right + dx, box.left + 4);
        if (lato.includes('n')) n.top = Math.min(box.top + dy, box.bottom - 4);
        if (lato.includes('s')) n.bottom = Math.max(box.bottom + dy, box.top + 4);
        n.left = Math.max(0, n.left); n.top = Math.max(0, n.top);
        if (W) n.right = Math.min(n.right, W);
        if (H) n.bottom = Math.min(n.bottom, H);
      }
      const finale = { left: Math.round(n.left), top: Math.round(n.top),
                       right: Math.round(n.right), bottom: Math.round(n.bottom) };
      miraLente = zonaAngolo(finale, lato) || miraLente;
      gancio.scriviRect(finale);
      draw();
    };
    const molla = () => {
      bersaglio.removeEventListener('pointermove', muovi);
      bersaglio.removeEventListener('pointerup', molla);
      bersaglio.removeEventListener('pointercancel', molla);
      draw();
    };
    bersaglio.addEventListener('pointermove', muovi);
    bersaglio.addEventListener('pointerup', molla);
    bersaglio.addEventListener('pointercancel', molla);
  }

  /* La corda si corregge trascinandola: due maniglie agli estremi cambiano x1 e x2, il
     corpo della linea la sposta in verticale. Il detector puo' sbagliarla, e questa e' la
     misura da cui dipende tutto il giro 3: correggerla deve valere piu' del suo risultato. */
  let bozza = null;      // {x1, x2, y} in pixel immagine, mentre si trascina
  let trascino = null;   // 'x1' | 'x2' | 'y'

  const segPx = (seg) => ({
    x1: seg.x1 * (size ? size[0] : 1),
    x2: seg.x2 * (size ? size[0] : 1),
    y: seg.y * (size ? size[1] : 1),
  });
  const dallaPagina = (event) => {
    const r = image.getBoundingClientRect();
    const s = scala();
    return {
      x: (event.clientX - r.left) / s.x,
      y: (event.clientY - r.top) / s.y,
    };
  };
  let selezionata = null;   // maniglia scelta: le frecce la muovono di un pixel

  const maniglia = (x, y, quale, color) => {
    const s = scala();
    const scelta = selezionata === quale;
    const node = el('div', {
      class: 'seg-handle' + (scelta ? ' scelta' : '') + (quale === 'y' ? ' centro' : ''),
      style: `left:${x * s.x}px;top:${y * s.y}px;--box-color:${color}`,
      title: quale === 'y'
        ? 'trascina per spostare la corda in alto o in basso · frecce ↑↓ per un pixel alla volta'
        : 'trascina questo estremo · frecce ←→ per un pixel alla volta (shift: dieci)',
    });
    // la tacca: prolunga la maniglia sopra e sotto, cosi' si vede esattamente dove cade
    node.append(el('span', { class: 'handle-tick' }));
    node.append(el('span', { class: 'handle-coord' },
      quale === 'y' ? `y ${Math.round(y)}` : `${Math.round(x)}`));
    node.addEventListener('mousedown', (event) => {
      event.preventDefault();
      event.stopPropagation();
      trascino = quale;
      selezionata = quale;
      // Anche la corda porta la lente con se': e' l'altra cosa che si trascina qui, e
      // guardare dove si ferma il suo estremo e' esattamente il genere di cosa da
      // ingrandire.
      miraLente = { left: x - 90, top: y - 90, right: x + 90, bottom: y + 90 };
      draw();
      renderCorrezione();
    });
    return node;
  };

  /* Le frecce muovono la maniglia scelta di un pixel: il trascinamento arriva vicino, la
     tastiera chiude. Senza questo la precisione dipende da quanti pixel dello schermo vale
     un pixel dell'immagine, che qui e' meno di mezzo. */
  const tasti = (event) => {
    if (!selezionata || !card.isConnected) return;
    const m = study && study.per_group[corrente];
    if (!m) return;
    const passo = event.shiftKey ? 10 : 1;
    const delta = { ArrowLeft: -passo, ArrowRight: passo, ArrowUp: -passo, ArrowDown: passo };
    if (!(event.key in delta)) return;
    const orizzontale = event.key === 'ArrowLeft' || event.key === 'ArrowRight';
    if (selezionata === 'y' && orizzontale) return;
    if (selezionata !== 'y' && !orizzontale) return;
    event.preventDefault();
    bozza = bozza || segPx(m.segment);
    if (selezionata === 'y') bozza.y += delta[event.key];
    else if (selezionata === 'x1') bozza.x1 += delta[event.key];
    else bozza.x2 += delta[event.key];
    draw();
    renderCorrezione();
  };
  window.addEventListener('keydown', tasti);

  const segNode = (seg, color, dashed, label) => {
    const s = scala();
    const w = ((size && size[0]) || 1) * s.x;
    const h = ((size && size[1]) || 1) * s.y;
    const node = el('div', {
      class: 'study-seg' + (dashed ? ' dashed' : ''),
      style: `left:${seg.x1 * w}px;top:${seg.y * h}px;width:${(seg.x2 - seg.x1) * w}px;`
        + `background:${color};border-color:${color}`,
    });
    if (label) node.append(el('span', { class: 'study-tag', style: `background:${color}` }, label));
    return node;
  };

  const asseNode = (valore, orientamento, color, label) => {
    const s = scala();
    const node = el('div', {
      class: `study-axis ${orientamento}`,
      style: orientamento === 'v'
        ? `left:${valore * s.x}px;background:${color}`
        : `top:${valore * s.y}px;background:${color}`,
    });
    node.append(el('span', { class: 'study-tag', style: `background:${color}` }, label));
    return node;
  };
  const puntoNode = (x, y, color, label) => {
    const s = scala();
    const node = el('div', {
      class: 'study-cross',
      style: `left:${x * s.x}px;top:${y * s.y}px;border-color:${color}`,
    });
    node.append(el('span', { class: 'study-tag', style: `background:${color}` }, label));
    return node;
  };

  const draw = () => {
    [...stage.querySelectorAll('.study-box, .study-seg, .study-axis, .study-cross, .seg-handle')]
      .forEach((n) => n.remove());
    const m = study && study.per_group[corrente];
    const rectDelGruppo = (specularita.per_group || {})[corrente];
    if (attivi.attuale) {
      const suo = gancio.leggiRect ? gancio.leggiRect() : chain.rect;
      if (suo) {
        // Quello che finisce davvero in #11: il rettangolo col margine attorno. Si disegna
        // per primo, sotto, e si aggiorna mentre si muovono i cursori - il margine e' una
        // percentuale, e a occhio non si sa quanti pixel faccia finche' non lo si vede.
        const salvato = gancio.leggiSalvato ? gancio.leggiSalvato() : null;
        if (salvato && (salvato.left !== suo.left || salvato.right !== suo.right
            || salvato.top !== suo.top || salvato.bottom !== suo.bottom)) {
          const stretto = salvato.right - salvato.left < suo.right - suo.left;
          stage.append(boxNode(salvato, '#3fb950', true,
            `salvato in #11 · ${salvato.right - salvato.left}x${salvato.bottom - salvato.top}`
            + (stretto ? ' (stretto)' : '')));
        }
        stage.append(rettangoloModificabile(suo));
      }
    }
    if (attivi.gruppo && rectDelGruppo) {
      stage.append(boxNode(rectDelGruppo, GROUP_COLORS_RECT[corrente], false, `rect ${corrente}`));
    }
    if (axesData) {
      const a = axesData.axes;
      const coppie = [
        ['assi_corde', a.corde_x, a.corde_y, '#3fb950', 'dal ventaglio'],
        ['assi_marker', a.marker_x, a.marker_y, '#ff4fd8', 'dai marker'],
        ['assi_immagine', a.immagine_x, a.immagine_y, '#9aa7b8', 'centro immagine'],
      ];
      for (const [chiave, ax, ay, color, label] of coppie) {
        if (!attivi[chiave]) continue;
        // l'etichetta dice cosa ribalta quell'asse, non solo da dove viene
        if (ax != null) {
          stage.append(asseNode(ax, 'v', color, `ribalta sx↔dx · ${label} · ${Math.round(ax)}`));
        }
        if (ay != null) {
          stage.append(asseNode(ay, 'h', color, `ribalta su↔giu · ${label} · ${Math.round(ay)}`));
        }
        if (ax != null && ay != null) stage.append(puntoNode(ax, ay, color, 'incrocio'));
      }
      if (attivi.marker) {
        for (const [g, c] of Object.entries(axesData.marker_centres || {})) {
          stage.append(puntoNode(c[0], c[1], GROUP_COLORS_RECT[g] || '#fff', g));
        }
      }
      if (attivi.candidato === '__costruito') {
        const r = rettCostruito();
        if (r) stage.append(boxNode(r, '#d29922', true, 'costruito sugli assi'));
      } else {
        const scelto = (axesData.candidates || []).find((c) => c.id === attivi.candidato);
        if (scelto) stage.append(boxNode(scelto.rect, '#d29922', true, scelto.label || scelto.id));
      }
    }
    if (m && attivi.corde) {
      const px = bozza || segPx(m.segment);
      const W = size ? size[0] : 1, H = size ? size[1] : 1;
      const vista = { x1: px.x1 / W, x2: px.x2 / W, y: px.y / H,
                      length_px: Math.round(Math.abs(px.x2 - px.x1)) };
      const colore = GROUP_COLORS_RECT[corrente];
      stage.append(segNode(vista, colore, false,
        `corda ${corrente} ${bozza ? Math.round(Math.abs(px.x2 - px.x1)) : m.segment.length_px} px`
        + (m.segment.corrected ? ' · corretta' : '') + (bozza ? ' · in modifica' : '')));
      stage.append(maniglia(Math.min(px.x1, px.x2), px.y, 'x1', colore));
      stage.append(maniglia(Math.max(px.x1, px.x2), px.y, 'x2', colore));
      stage.append(maniglia((px.x1 + px.x2) / 2, px.y, 'y', colore));
      // il centro della corda e' il punto da cui nasce l'asse verticale: marcarlo rende
      // evidente il legame fra le due cose, che altrimenti sembrano scollegate
      if (attivi.assi_corde) {
        stage.append(puntoNode((px.x1 + px.x2) / 2, px.y, '#3fb950', 'centro della corda'));
      }
      const partner = PARTNER[corrente];
      const pair = (study.pairs || {})[`${corrente}-${partner}`]
        || (study.pairs || {})[`${partner}-${corrente}`];
      const altro = study.per_group[partner];
      if (altro && attivi.speculare) {
        stage.append(segNode(altro.segment, GROUP_COLORS_RECT[partner], true,
          `corda ${partner}`));
      }
      info.innerHTML = '';
      info.append(el('div', {},
        `gruppo ${corrente} · piano ${m.plane || '?'} · corda ${m.segment.length_px} px ` +
        `(mediana del gruppo ${m.median_length_px} px) · immagine ${m.segment.image}`));
      if (pair) {
        info.append(el('div', {
          style: `color:${pair.max_px <= 8 ? 'var(--ok)' : 'var(--warn)'}`,
        }, `contro ${partner}: nella stessa posizione ${pair.identity.max_px} px, ` +
           `specchiate ${pair.mirror.max_px} px → tiene «${pair.hypothesis}»`));
      }
    } else {
      info.textContent = 'per questo gruppo non c\'e\' ancora lo studio delle corde: lancia il giro 3';
    }
    if (Lente.viva()) Lente.aggiorna(contestoLente());
  };
  // Il pannello puo' chiedere di ridisegnare: succede quando si cambia un numero sotto.
  if (ganci) ganci.ridisegna = draw;

  /* Le elaborazioni sono di cartella, le immagini sono tante: si scorrono tutte sotto gli
     stessi disegni, cosi' si controlla che corda, rettangolo e assi tengano su ognuna e non
     solo su quella scelta dal modulo. */
  let elenco = [];
  let posizione = 0;
  const navTesto = el('span', { class: 'hint' });
  const navNome = el('span', { class: 'hint' });

  const carica = (nome, rappresentativa) => {
    if (!nome) return;
    image.src = `/api/projects/${state.projectId}/image`
      + `?name=${encodeURIComponent(nome)}&w=980`;
    image.title = nome;
    navTesto.textContent = elenco.length
      ? `immagine ${posizione + 1} di ${elenco.length} del gruppo ${corrente}`
      : 'immagine del progetto';
    navNome.innerHTML = '';
    // append(null) scriverebbe la parola "null": i figli si filtrano prima
    navNome.append(...[
      el('span', {}, nome.split('/').pop()),
      rappresentativa
        ? el('span', { class: 'score-fixed', style: 'margin-left:6px' }, 'usata per la misura')
        : null,
    ].filter(Boolean));
    draw();
  };

  const vaiA = (indice) => {
    if (!elenco.length) return;
    posizione = (indice + elenco.length) % elenco.length;
    const m = study && study.per_group[corrente];
    const rif = m && m.segment && m.segment.image;
    carica(elenco[posizione], elenco[posizione] === rif);
  };

  const mostra = (group) => {
    corrente = group;
    for (const [btn, key] of pulsanti) btn.className = 'chip' + (key === group ? ' on' : '');
    const m = study && study.per_group[group];
    const rif = (m && m.segment && m.segment.image) || null;
    elenco = ((chain.images_by_group || {})[group] || []).slice();
    if (elenco.length) {
      posizione = Math.max(0, elenco.indexOf(rif));
      carica(elenco[posizione], elenco[posizione] === rif);
      return;
    }
    // ripiego: senza elenco si mostra almeno la rappresentativa o l'anteprima del progetto
    elenco = [];
    posizione = 0;
    carica(rif || chain.preview_image, Boolean(rif));
  };

  const navRow = el('div', { class: 'row', style: 'margin:6px 0 4px' },
    el('button', { class: 'ghost sq', title: 'immagine precedente del gruppo',
      onclick: () => vaiA(posizione - 1) }, '‹'),
    el('button', { class: 'ghost sq', title: 'immagine successiva del gruppo',
      onclick: () => vaiA(posizione + 1) }, '›'),
    el('button', {
      class: 'ghost',
      title: 'torna all\'immagine su cui e\' stata misurata la corda',
      onclick: () => {
        const m = study && study.per_group[corrente];
        const rif = m && m.segment && m.segment.image;
        if (rif && elenco.includes(rif)) vaiA(elenco.indexOf(rif));
      },
    }, 'quella della misura'),
    navTesto, navNome);
  const pulsanti = [];
  for (const group of ['NF', 'LR', 'UD', 'LRUD'].filter((g) => gruppi.includes(g))) {
    const btn = el('button', { class: 'chip' }, group);
    btn.style.borderColor = GROUP_COLORS_RECT[group];
    btn.addEventListener('click', () => mostra(group));
    pulsanti.push([btn, group]);
    chips.append(btn);
  }
  for (const [colore, testo] of [
    ['#ffffff', 'rettangolo di adesso'],
    ['#d29922', 'proposta che stai guardando'],
    [GROUP_COLORS_RECT.NF, 'corda dell\'orientamento scelto'],
  ]) {
    legenda.append(el('span', { class: 'editor-legend' },
      el('span', { class: 'swatch', style: `background:${colore}` }), testo));
  }
  legenda.append(el('span', { class: 'editor-legend' },
    el('span', { class: 'swatch', style: 'background:var(--muted)' }), 'tratteggiato: gruppo speculare'));
  stage.addEventListener('mousemove', (event) => {
    if (!trascino) return;
    const m = study && study.per_group[corrente];
    if (!m) return;
    const p = dallaPagina(event);
    const base = bozza || segPx(m.segment);
    bozza = { ...base };
    if (trascino === 'y') bozza.y = Math.max(0, Math.min((size ? size[1] : 1) - 1, p.y));
    else if (trascino === 'x1') bozza.x1 = Math.max(0, Math.min((size ? size[0] : 1) - 1, p.x));
    else bozza.x2 = Math.max(0, Math.min((size ? size[0] : 1) - 1, p.x));
    draw();
    renderCorrezione();
  });
  // La mira resta dov'era anche dopo aver mollato: dopo aver sistemato un bordo si vuole
  // continuare a guardarlo mentre lo si rifinisce coi numeri o con le frecce. Si torna alla
  // veduta d'insieme col tasto «inquadra le corde».
  const fineTrascino = () => { trascino = null; };
  stage.addEventListener('mouseup', fineTrascino);
  stage.addEventListener('mouseleave', fineTrascino);

  const correzione = el('div', { class: 'row', style: 'margin-top:6px' });
  const spiegaAssi = el('div', { class: 'hint', style: 'margin-top:4px' });
  const renderSpiegaAssi = () => {
    spiegaAssi.innerHTML = '';
    if (!attivi.assi_corde && !attivi.assi_marker && !attivi.assi_immagine) return;
    const righe = [];
    if (attivi.assi_corde) {
      righe.push('asse dal ventaglio (verde): la verticale passa per il centro della corda, '
        + 'l\'orizzontale sta a meta\' fra dove cade la corda in «su» e dove cade in «giu». '
        + 'E\' la linea attorno a cui il contenuto si ribalta.');
    }
    if (attivi.assi_marker) {
      righe.push('asse dai marker (rosa): la stessa cosa misurata sui marker di orientamento '
        + 'invece che sul ventaglio. Se le due linee coincidono, l\'asse e\' verificato.');
    }
    if (attivi.assi_immagine) {
      righe.push('centro dell\'immagine (grigio): non e\' un asse di ribaltamento, e\' solo '
        + 'il riferimento per vedere quanto quello vero se ne discosta.');
    }
    for (const r of righe) spiegaAssi.append(el('div', {}, r));
  };

  const renderCorrezione = () => {
    correzione.innerHTML = '';
    renderSpiegaAssi();
    const m = study && study.per_group[corrente];
    if (!m) return;
    if (bozza) {
      const misura = `x ${Math.round(Math.min(bozza.x1, bozza.x2))}-`
        + `${Math.round(Math.max(bozza.x1, bozza.x2))} · y ${Math.round(bozza.y)} · `
        + `lunga ${Math.round(Math.abs(bozza.x2 - bozza.x1))} px`;
      // anche questo salvataggio va in due tempi: e' una misura da cui dipende tutto il
      // giro 3, e un clic solo l'ha gia' cambiata una volta senza che nessuno lo chiedesse
      correzione.append(
        confermaInDueTempi('Salva la corda corretta', `la corda diventa ${misura}.`, async () => {
          try {
            const res = await api(`/projects/${state.projectId}/rect/segment`,
              { body: { group: corrente, x1: bozza.x1, x2: bozza.x2, y: bozza.y,
                        image: elenco[posizione] } });
            bozza = null;
            toast(`corda di ${corrente} corretta: ${res.segment.length_px} px · conti rifatti`);
            await ricarica();
          } catch (error) { toast(error.message, true); }
        }),
        el('button', { class: 'ghost', onclick: () => { bozza = null; draw(); renderCorrezione(); } },
          'Annulla la modifica'),
        el('span', { class: 'hint' }, misura));
      return;
    }
    const px = segPx(m.segment);
    correzione.append(el('span', { class: 'hint' },
      `corda: da x ${Math.round(px.x1)} a x ${Math.round(px.x2)} · y ${Math.round(px.y)} · `
      + `lunga ${m.segment.length_px} px`));
    correzione.append(el('span', { class: 'hint' },
      '— trascina le maniglie (agli estremi la larghezza, al centro l\'altezza), '
      + 'poi le frecce della tastiera la muovono di un pixel'));
    if (m.segment.corrected) {
      correzione.append(confermaInDueTempi(
        'Torna alla corda trovata dal modulo',
        'la tua correzione viene scartata.',
        async () => {
          try {
            await api(`/projects/${state.projectId}/rect/segment`,
              { body: { group: corrente, reset: true } });
            toast('corda ripristinata · conti rifatti');
            await ricarica();
          } catch (error) { toast(error.message, true); }
        }));
    }
  };
  card.append(correzione, spiegaAssi);

  /* Dopo una correzione si rilegge tutto: le coppie, l'ampiezza, la proposta e i candidati
     sono cambiati, e devono cambiare anche sullo schermo. */
  const ricarica = async () => {
    const fresca = await api(`/projects/${state.projectId}/rect/chain`);
    chain = fresca;
    study = (fresca.study || {}).segments;
    axesData = await api(`/projects/${state.projectId}/rect/axes`).catch(() => axesData);
    renderCandidati();
    mostra(corrente);
    renderCorrezione();
  };

  card.append(navRow, stage, legenda, info);
  // Va nell'elenco di `replaceChildren` in fondo, che e' chi decide l'ordine della scheda:
  // appenderla qui e basta non serve a niente, perche' quella composizione ricomincia da capo.
  const inquadraCorde = el('button', { class: 'ghost sq2' }, 'inquadra le corde');
  inquadraCorde.addEventListener('click', () => { miraLente = null; draw(); });

  /* Solo i disegni dello studio delle corde. Gli altri - il rettangolo visto nel gruppo,
     gli assi dai marker, il centro dell'immagine, i marker - venivano da studi che questa
     sezione non racconta piu': lasciarli era lasciare interruttori senza una stanza. */
  const LIVELLI = [
    ['attuale', 'rettangolo di adesso'],
    ['corde', 'corda del ventaglio'],
    ['speculare', 'corda dell\'orientamento speculare'],
    ['assi_corde', 'asse di ribaltamento (dal ventaglio)'],
  ];
  for (const [chiave, label] of LIVELLI) {
    const btn = el('button', { class: 'chip' + (attivi[chiave] ? ' on' : '') }, label);
    btn.addEventListener('click', () => {
      attivi[chiave] = !attivi[chiave];
      btn.className = 'chip' + (attivi[chiave] ? ' on' : '');
      draw();
      renderCorrezione();
    });
    layers.append(btn);
  }

  /* I candidati: si accendono uno alla volta sul frame e si applicano da qui, cosi' la
     scelta si fa guardando invece che leggendo delle coordinate. */
  /* Il rettangolo costruito sugli assi: centro sull'asse delle corde, mezze estensioni
     fino all'estremo piu' lontano, e due slider per il margine — uguale a destra e a
     sinistra, uguale sopra e sotto, per costruzione. Il disegno segue lo slider senza
     tornare al server: la costruzione arriva gia' senza margine. */
  let margineX = 0;
  let margineY = 0;
  /* Il verticale e' l'anello debole: la corda e' dove il detector ha trovato il tratto
     chiaro piu' lungo, che sta *dentro* l'immagine e non sul suo bordo. In orizzontale gli
     estremi della corda sono i fianchi veri del ventaglio; in verticale no. Quindi si puo'
     applicare solo l'orizzontale, ed e' il default finche' il giro 4 non porta la depth. */
  let soloOrizzontale = true;
  const rettCostruito = () => {
    const c = study && study.construction;
    if (!c || c.axis_x == null) return null;
    const W = c.image_size[0], H = c.image_size[1];
    const left = Math.max(0, Math.round(c.axis_x - c.half_x - margineX));
    const right = Math.min(W - 1, Math.round(c.axis_x + c.half_x + margineX));
    if (c.axis_y == null || soloOrizzontale) {
      return { ...(chain.rect || {}), left, right };
    }
    return {
      left,
      right,
      top: Math.max(0, Math.round(c.axis_y - c.half_y - margineY)),
      bottom: Math.min(H - 1, Math.round(c.axis_y + c.half_y + margineY)),
    };
  };

  const renderCandidati = () => {
    candidati.innerHTML = '';
    if (!axesData) return;
    const a = axesData.axes;
    const acc = axesData.agreement || {};
    const misure = el('details', { class: 'ov-fold', style: 'margin-bottom:10px' });
    const somma = acc.x_px != null && acc.y_px != null
      ? `orizzontale ${acc.x_px} px · verticale ${acc.y_px} px` : '';
    misure.append(el('summary', {},
      el('span', {}, 'assi di ribaltamento misurati'),
      el('span', {
        class: 'ov-fold-tag' + (acc.x_px != null && acc.x_px <= 15 ? '' : ' warn'),
      }, somma)));
    const dentro = el('div');
    misure.append(dentro);
    candidati.append(misure);
    // Resta il confronto con i marker, che e' la verifica dell'asse delle corde: due misure
    // indipendenti che concordano valgono piu' di una sola. Via invece il centro
    // dell'immagine e quello del rettangolo, che erano numeri da leggere e basta.
    for (const [k, v] of [
      ['asse x dalle corde', a.corde_x], ['asse x dai marker', a.marker_x],
      ['accordo sulla x', acc.x_px != null ? `${acc.x_px} px` : '—'],
      ['asse y dalle corde', a.corde_y], ['asse y dai marker', a.marker_y],
      ['accordo sulla y', acc.y_px != null ? `${acc.y_px} px` : '—'],
    ]) {
      const riga = el('div', { class: 'kv' }, el('span', {}, k), el('span', {}, String(v ?? '—')));
      if (k.startsWith('accordo')) {
        const px = k.endsWith('x') ? acc.x_px : acc.y_px;
        riga.lastChild.style.color = px == null ? 'var(--muted)'
          : px <= 15 ? 'var(--ok)' : 'var(--warn)';
      }
      dentro.append(riga);
    }
    dentro.append(el('p', { class: 'hint' },
      'l\'asse si misura da due cose indipendenti: il ventaglio e i marker. Se concordano ' +
      '(sotto i 15 px) ribaltare attorno a quell\'asse e\' verificato; se no, no.'));
    const costr = study && study.construction;
    if (costr && costr.axis_x != null) {
      const blocco = el('div', { class: 'proposta' + (attivi.candidato === '__costruito' ? ' scelta' : '') });
      const misura = el('div', { class: 'hint' });
      const mostra = el('button', { class: 'chip' + (attivi.candidato === '__costruito' ? ' on' : '') },
        attivi.candidato === '__costruito' ? 'nascondi' : 'vedi');
      mostra.addEventListener('click', () => {
        attivi.candidato = attivi.candidato === '__costruito' ? '' : '__costruito';
        renderCandidati();
        draw();
      });
      blocco.append(el('div', { class: 'proposta-testa' },
        el('span', { style: 'font-weight:600' }, 'Costruito sugli assi'),
        el('span', { class: 'hint' }, `${costr.chords} corde`
          + (costr.centre_spread_px != null
            ? ` · i loro centri ballano di ${costr.centre_spread_px} px` : '')),
        mostra));
      blocco.append(el('div', { class: 'hint' },
        `centro sull'asse delle corde (${costr.axis_x})`
        + (costr.axis_y != null
          ? `, a meta' fra la corda piu' alta (${costr.highest.group} y ${costr.highest.y}) `
            + `e la piu' bassa (${costr.lowest.group} y ${costr.lowest.y})`
          : '')
        + '. Con margine zero le corde estreme toccano esattamente i bordi; il margine si '
        + 'aggiunge uguale sui due lati.'));

      const soloBox = el('input', { type: 'checkbox' });
      soloBox.checked = soloOrizzontale;
      soloBox.addEventListener('change', () => {
        soloOrizzontale = soloBox.checked;
        renderCandidati();
        draw();
      });
      blocco.append(el('label', { class: 'serie', style: 'display:flex;gap:6px;margin:6px 0' },
        soloBox,
        el('span', {}, 'cambia solo destra e sinistra, lascia sopra e sotto come sono')));
      if (soloOrizzontale && costr.axis_y != null) {
        blocco.append(el('div', { class: 'hint' },
          'consigliato: in orizzontale gli estremi della corda sono i fianchi veri del '
          + 'ventaglio, in verticale la corda cade dentro l\'immagine e non sul suo bordo, '
          + 'quindi userebbe un\'altezza piu\' stretta del vero. Sopra e sotto li chiudera\' '
          + 'il giro con la depth.'));
      }

      const aggiorna = () => {
        const r = rettCostruito();
        if (!r) return;
        misura.innerHTML = '';
        misura.append(el('span', {},
          `${r.top}|${r.left}|${r.bottom}|${r.right} · ${r.right - r.left}x${r.bottom - r.top} px`));
        const limite = axesData.esi_limit || [1450, 820];
        if (r.right - r.left > limite[0] || r.bottom - r.top > limite[1]) {
          misura.append(el('span', { style: 'color:var(--err);margin-left:8px' },
            `oltre il limite ESI ${limite[0]}x${limite[1]}`));
        }
        if (attivi.candidato === '__costruito') draw();
      };
      for (const [etichetta, leggi, scrivi] of [
        ['margine destra e sinistra', () => margineX, (v) => { margineX = v; }],
        ['margine sopra e sotto', () => margineY, (v) => { margineY = v; }],
      ]) {
        if (etichetta.includes('sopra') && (costr.axis_y == null || soloOrizzontale)) continue;
        const valore = el('span', { class: 'hint', style: 'min-width:52px' }, `${leggi()} px`);
        const slider = el('input', {
          type: 'range', min: '0', max: '150', step: '1', value: String(leggi()),
        });
        slider.addEventListener('input', () => {
          scrivi(parseInt(slider.value, 10) || 0);
          valore.textContent = `${leggi()} px`;
          aggiorna();
        });
        blocco.append(el('div', { class: 'slider-row' },
          el('label', {}, etichetta), slider, valore));
      }
      blocco.append(misura);
      blocco.append(confermaInDueTempi('applica questo', 'costruito sugli assi delle corde.',
        async () => {
          try {
            await api(`/projects/${state.projectId}/rect/apply`,
              { body: { pass: 'costruito', rect: rettCostruito() } });
            toast('rettangolo costruito applicato');
            await reload();
          } catch (error) { toast(error.message, true); }
        }));
      candidati.append(blocco);
      aggiorna();
    }

    candidati.append(el('h3', {}, 'oppure una di queste'));
    candidati.append(el('p', { class: 'hint' },
      'premi «vedi» per disegnarlo sull\'immagine sopra, accanto a quello di adesso. ' +
      'Niente cambia finche\' non applichi.'));
    // «attuale» non e' una proposta: e' il rettangolo gia' disegnato sull'immagine e gia'
    // scritto nel riepilogo sotto, e in mezzo alle altre si legge come una quinta scelta.
    // «consenso» viene dalla specularita' e «simmetrico_marker» dai marker: due studi che
    // questa sezione non fa piu'.
    // Resta la sola proposta che nasce davvero dalla misura delle corde: allargare fino
    // alla piu' larga. Il «simmetrico sull'asse del ventaglio» non serve.
    const SOLO_CORDE = ['ampiezza'];
    for (const c of axesData.candidates.filter((v) => SOLO_CORDE.includes(v.id))) {
      const r = c.rect;
      const scelto = attivi.candidato === c.id;
      const riga = el('div', { class: 'proposta' + (scelto ? ' scelta' : '') });
      const mostraBtn = el('button', { class: 'chip' + (scelto ? ' on' : '') },
        c.id === 'attuale' ? 'in uso' : (scelto ? 'nascondi' : 'vedi'));
      if (c.id !== 'attuale') {
        mostraBtn.addEventListener('click', () => {
          attivi.candidato = scelto ? '' : c.id;
          renderCandidati();
          draw();
        });
      } else {
        mostraBtn.disabled = true;
      }
      const testa = el('div', { class: 'proposta-testa' },
        el('span', { style: 'font-weight:600' }, c.label || c.id),
        el('span', { class: 'hint' },
          `${r.top}|${r.left}|${r.bottom}|${r.right} · ${c.size[0]}x${c.size[1]} px`
          + (c.id === 'attuale' ? '' : ` · cambia di ${c.delta.max_side_px} px`)),
        mostraBtn);
      riga.append(testa, el('div', { class: 'hint' }, c.note || ''));
      const fit = c.axis_fit;
      if (fit) {
        const centrato = Math.abs(fit.off_axis_x) <= 2;
        riga.append(el('div', {
          class: 'hint',
          style: `color:${centrato ? 'var(--ok)' : 'var(--warn)'}`,
        }, centrato
          ? `il centro cade sull'asse del ventaglio (${fit.axis_x}): ribaltandolo a `
            + `sinistra-destra resta dov'e'`
          : `il centro sta a ${fit.centre_x}, l'asse del ventaglio a ${fit.axis_x}: `
            + `pende di ${Math.abs(fit.off_axis_x)} px `
            + `(${fit.half_left} px a sinistra dell'asse, ${fit.half_right} a destra)`));
      }
      for (const avviso of (c.warnings || [])) {
        riga.append(el('div', { class: 'hint', style: 'color:var(--warn)' }, '⚠ ' + avviso));
      }
      if (c.id !== 'attuale') {
        riga.append(confermaInDueTempi('applica questo', `${c.label}: ${c.note}`, async () => {
          try {
            await api(`/projects/${state.projectId}/rect/apply`,
              { body: { pass: 'scelta', rect: c.rect } });
            toast(`applicato: ${c.label}`);
            await reload();
          } catch (error) { toast(error.message, true); }
        }));
      }
      candidati.append(riga);
    }
  };
  card.append(candidati);

  api(`/projects/${state.projectId}/rect/axes`).then((data) => {
    axesData = data;
    renderCandidati();
    draw();
  }).catch(() => {});

  /* L'ordine della scheda si decide qui, per intero: prima si sceglie il gruppo e
     l'immagine, poi si guarda, poi si legge, poi si sceglie. Comporlo a pezzi sparsi lo
     aveva gia' fatto uscire mescolato. */
  /* Due colonne: l'immagine a sinistra, tutto il resto a destra. Prima era una colonna
     sola, e per guardare l'immagine mentre si leggeva un numero - o si spostava un
     margine - bisognava scorrere su e giu'. La colonna dell'immagine sta appiccicata in
     alto mentre la destra scorre: e' l'immagine il punto fermo del lavoro. */
  const colonnaSinistra = el('div', { class: 'rect-immagine' },
    chips,           // quale orientamento
    // Frecce e lente nella stessa riga e nello stesso posto di ogni altra sezione.
    el('div', { class: 'barra-immagine' }, navRow,
      Lente.bottone(contestoLente), inquadraCorde,
      el('span', { class: 'hint' }, 'la lente segue quello che tocchi')),
    stage,           // l'immagine con le elaborazioni
    legenda,         // che colore e' cosa
  );
  const colonnaDestra = el('div', { class: 'rect-lato' },
    info,            // i numeri di questo gruppo
    correzione,      // correggere la corda che si sta guardando
    spiegaAssi,      // cosa sono gli assi accesi
    layers,          // cosa mostrare sull'immagine
    candidati,       // quale rettangolo consegnare
  );
  card.replaceChildren(titolo, el('div', { class: 'rect-due' }, colonnaSinistra, colonnaDestra));
  // Il pannello ci mette i suoi comandi (numeri, margini, salvataggio): stanno a destra
  // anche loro, se no si torna a scorrere per raggiungerli.
  if (ganci) {
    ganci.colonnaDestra = colonnaDestra;
    for (const nodo of (ganci.inAttesa || [])) colonnaDestra.append(nodo);
    ganci.inAttesa = [];
  }

  image.addEventListener('load', draw);
  window.addEventListener('resize', draw);
  /* Il caricamento non basta: l'immagine viene disegnata alla sua larghezza naturale e solo
     dopo il layout la restringe. Senza osservare quel cambio, gli overlay restano calcolati
     sulla larghezza vecchia e cadono fuori posto — succedeva di 2.4 volte. */
  if (window.ResizeObserver) new ResizeObserver(() => draw()).observe(image);
  mostra(corrente);
  renderCorrezione();
  return card;
}

function panelRect(panel, step) {
  const stored = (state.project.steps.rect || {}).value || {};
  const value = JSON.parse(JSON.stringify(stored));
  value.rect_echo = value.rect_echo || { top: 0, left: 0, bottom: 0, right: 0 };
  const analysis = state.project.analysis || {};
  const rect = analysis.rect || {};
  const imported = (state.project.steps.import || {}).value || {};
  analysisRow(panel, 'la rete propone un rettangolo per tutta la cartella, tu confermi');
  panel.append(el('p', { class: 'hint' },
    'un solo rettangolo per tutta la cartella: e\' la geometria da cui dipendono ' +
    'orientamento, depth e scala.'));
  // I ganci per l'immagine dello studio, che e' l'unica: da li' si trascina il rettangolo,
  // e il valore vive qui. Si passano come funzioni perche' `panelRect` lo costruisce dopo.
  let editorRef = null;   // i comandi numerici, che devono seguire il trascinamento
  const ganciRect = {
    leggiRect: () => value.rect_echo,
    // Il rettangolo col margine: e' quello che finisce nel file, e va visto sull'immagine
    // mentre si muovono i cursori.
    leggiSalvato: () => savedRect(),
    scriviRect: (box) => {
      applyBoxes({ rect_echo: box });
      // I numeri sotto sono una copia: senza questo restavano fermi mentre il rettangolo
      // si muoveva, e due punti della stessa pagina dicevano cose diverse.
      if (editorRef) {
        Object.assign(editorRef.boxes.rect_echo, value.rect_echo);
        editorRef.paint();
      }
    },
    // Lo studio ci mette dentro il suo `draw`: cosi' cambiando un numero si muove anche
    // il rettangolo sull'immagine, che e' l'altro verso dello stesso legame.
    ridisegna: null,
    // I comandi del pannello vanno nella colonna destra dello studio, che pero' nasce
    // dopo (la catena si carica in asincrono): fino ad allora si mettono in coda.
    colonnaDestra: null,
    inAttesa: [],
  };
  const aDestra = (nodo) => {
    if (ganciRect.colonnaDestra) ganciRect.colonnaDestra.append(nodo);
    else ganciRect.inAttesa.push(nodo);
  };
  rectChainCard(panel, ganciRect);

  aDestra(el('h3', { style: 'margin:14px 0 4px' }, 'Il rettangolo in numeri (#11)'));
  aDestra(el('p', { class: 'hint' },
    'i bordi si trascinano sull\'immagine a sinistra, dove ci sono anche le corde. Qui ci '
    + 'sono i numeri e i margini, che si aggiungono attorno in percentuale: a zero il '
    + 'salvato coincide col rettangolo.'));

  const rectSize = el('div', { class: 'hint' });
  const dirtyBadge = el('span', { class: 'dirty' }, 'modifiche non salvate');
  dirtyBadge.style.display = 'none';
  const limit = state.meta.rect_limit;
  const sampleSize = imported.image_sample_size || state.project.codes.image_sample_size;
  const original = JSON.stringify([cloneBoxes(stored), stored.margin_percent || { x: 0, y: 0 }]);
  value.margin_percent = { x: 0, y: 0, ...(value.margin_percent || {}) };
  const margins = value.margin_percent;
  const proposal = { rect_echo: rect.rect_echo };

  const editorBoxes = (source) => {
    const boxes = cloneBoxes(source);
    if (proposal.rect_echo) boxes.rect_echo_original = { ...proposal.rect_echo };
    return boxes;
  };
  const saveBoxes = async () => {
    try {
      await api(`/projects/${state.projectId}/steps/${step.id}`, { body: { value } });
      toast('rettangolo salvato per tutta la cartella');
      await reload();
    } catch (error) { toast(error.message, true); }
  };
  const describe = (box) => box
    ? `${box.right - box.left} x ${box.bottom - box.top} px ` +
      `(top ${box.top} left ${box.left} bottom ${box.bottom} right ${box.right})`
    : '—';
  const savedRect = () => expandBox(
    value.rect_echo, margins, (sampleSize || [0, 0])[0], (sampleSize || [0, 0])[1]);
  const readout = () => {
    const saved = savedRect();
    const over = saved && (saved.right - saved.left > limit[0] || saved.bottom - saved.top > limit[1]);
    rectSize.innerHTML = '';
    if (proposal.rect_echo) {
      rectSize.append(el('div', { style: 'color:var(--muted)' },
        `originario (proposta rete): ${describe(proposal.rect_echo)}`));
    }
    rectSize.append(el('div', { style: 'color:#ff8a70' },
      `rettangolo ecografico: ${describe(value.rect_echo)}`));
    const suffix = (margins.x || margins.y)
      ? ` con margine ${margins.x}% / ${margins.y}%`
      : ' (margini a zero: identico al rettangolo)';
    const line = el('div', {}, `verra\' salvato in #11${suffix}: ${describe(saved)} — ` +
      (over ? `oltre il limite ESI ${limit[0]}x${limit[1]}` : `entro il limite ESI ${limit[0]}x${limit[1]}`));
    line.style.color = over ? 'var(--warn)' : 'var(--ok)';
    rectSize.append(line);
  };
  const applyBoxes = (boxes, marginState) => {
    if (marginState) value.margin_percent = { x: marginState.x, y: marginState.y };
    if (boxes.rect_echo) value.rect_echo = { ...value.rect_echo, ...boxes.rect_echo };
    if (ganciRect && ganciRect.ridisegna) ganciRect.ridisegna();
    readout();
    dirtyBadge.style.display =
      JSON.stringify([cloneBoxes(value), value.margin_percent]) === original
        ? 'none' : 'inline-block';
  };
  const openFullscreen = () => openFullscreenEditor({
    projectId: state.projectId,
    startValue: value,
    extraBoxes: proposal.rect_echo ? { rect_echo_original: proposal.rect_echo } : {},
    margins,
    sampleSize,
    startImage: value.preview_image,
    onSave: (boxes, marginState) => { applyBoxes(boxes, marginState); saveBoxes(); },
  });

  if (value.preview_image) {
    const editor = createBoxEditor({
      imageSrc: `/api/projects/${state.projectId}/image` +
        `?name=${encodeURIComponent(value.preview_image)}&w=980`,
      boxes: editorBoxes(value),
      sampleSize,
      margins,
      projectId: state.projectId,
      imageName: value.preview_image,
      onChange: applyBoxes,
      // Niente immagine qui: il rettangolo si trascina su quella dello studio, sopra, dove
      // ci sono anche le corde. Restano i numeri e i margini, che un'immagine non la vogliono.
      soloControlli: true,
    });
    editorRef = editor;
    aDestra(editor.root);
    // La lente sta con l'immagine, sopra: qui non c'e' piu' un'immagine a cui riferirsi,
    // e un secondo tasto che apre la stessa finestra su un contesto diverso confonderebbe.
    aDestra(el('div', { class: 'row' },
      el('button', { class: 'ghost', onclick: () => openFullscreen() }, 'Schermo intero')));
    aDestra(el('div', { class: 'row' },
      el('button', { onclick: saveBoxes }, 'Salva il rettangolo'),
      dirtyBadge,
      el('button', {
        class: 'ghost',
        onclick: () => { render(); toast('modifiche annullate, tornata all\'ultimo salvataggio'); },
      }, 'Annulla le modifiche'),
      el('button', {
        class: 'ghost',
        onclick: async () => {
          if (!proposal.rect_echo) {
            return toast('nessuna proposta salvata: lancia "Rilancia l\'analisi"', true);
          }
          if (!confirm('Tornare ai valori proposti dalla rete?\n\n' +
            'Le modifiche fatte a mano sul rettangolo vengono sostituite.')) return;
          value.rect_echo = { ...proposal.rect_echo };
          value.margin_percent = { x: 0, y: 0 };
          await saveBoxes();
          toast('ripristinati i valori della rete');
        },
      }, 'Valori iniziali della rete'),
      el('button', {
        class: 'ghost',
        onclick: async () => {
          const result = await api(`/projects/${state.projectId}/resize_check`,
            { body: { rect_echo: savedRect() } });
          if (!result.proposal) return toast('il rettangolo sta nel limite ESI: nessun resize');
          const p = result.proposal;
          if (!confirm(`${p.reason}\n\nfattore ${p.factor}\nimmagine campione -> ${p.image_sample_size.join('x')}` +
            `\nrettangolo -> ${p.rect_size_after.join('x')}\n\nApplicare?`)) return;
          value.rect_echo = p.rect_echo;
          await api(`/projects/${state.projectId}/codes`,
            { body: { codes: { image_sample_size: p.image_sample_size } } });
          toast('resize applicato');
          await reload();
        },
      }, 'Verifica limite ESI'),
    ));
  } else {
    aDestra(el('p', { class: 'hint' }, 'nessuna anteprima: lancia prima l\'analisi.'));
    const setRect = (key) => (raw) => { value.rect_echo[key] = parseInt(raw || '0', 10) || 0; readout(); };
    aDestra(el('div', { class: 'grid' },
      field('top', value.rect_echo.top, setRect('top'), 'number'),
      field('left', value.rect_echo.left, setRect('left'), 'number'),
      field('bottom', value.rect_echo.bottom, setRect('bottom'), 'number'),
      field('right', value.rect_echo.right, setRect('right'), 'number'),
    ));
  }
  aDestra(rectSize);
  readout();
  aDestra(saveRow(step.id, () => value));
}

/* --- orientamento e depth/scala: i moduli girano come sottoprocessi --- */
function panelModuleStage(panel, step) {
  const stages = (state.project.analysis || {}).stages || {};
  const rect = (state.project.steps.rect || {}).value || {};
  const stagesGate = state.advancedStages || { ready: true, blocked_reason: '' };

  // Lo studio del righello ha una schermata tutta sua: niente comandi dei moduli sopra,
  // ha il suo «rifai lo studio» che ci rimette dentro le correzioni.
  if (step.id === 'scale_study') {
    const host = el('div', {});
    panel.append(host);
    if (typeof createScaleViewer !== 'function') {
      // Succede con una `index.html` vecchia in cache, senza lo script della sezione: senza
      // questo controllo il pannello si fermava qui, muto.
      host.append(el('p', { class: 'hint' },
        'la vista della scala non e\' stata caricata: ricarica la pagina '
        + '(cmd+shift+R) e riprova.'));
      return;
    }
    createScaleViewer(state.projectId)
      .then((node) => host.append(node))
      .catch((error) => {
        // Un progetto analizzato prima che lo studio esistesse non ha ancora i suoi dati:
        // invece di lasciare la pagina vuota, la si fa partire da qui.
        host.append(el('p', { class: 'hint' }, error.message + '.'));
        const stato = el('span', { class: 'hint' }, 'qualche minuto: gira sui fotogrammi della scala');
        const avvia = el('button', {}, 'Studia il righello di questa cartella');
        avvia.addEventListener('click', async () => {
          avvia.disabled = true;
          try {
            const { job_id } = await api(`/projects/${state.projectId}/scale/study/run`, { body: {} });
            await pollJob(job_id, stato);
            toast('studio del righello fatto');
            await reload();
          } catch (errore) {
            toast(errore.message, true);
            stato.textContent = errore.message;
          } finally { avvia.disabled = false; }
        });
        host.append(el('div', { class: 'row' }, avvia, stato));
      });
    return;
  }

  // Ogni modulo si lancia per conto suo, dalla sezione che lo riguarda. Un comando solo
  // per tutti e tre voleva dire rifare la scala - minuti - per una correzione al marker.
  const moduli = step.id === 'orientation'
    ? [{ chiave: 'orientamento', titolo: 'orientamento', nome: 'l\'orientamento',
         fatto: !!stages.marker,
         attesa: 'il marker gira su tutte le immagini uniche: qualche minuto' }]
    : [{ chiave: 'depth', titolo: 'depth', nome: 'la depth', fatto: !!stages.depth,
         attesa: 'legge la scala ecografica su un campione di immagini' },
       { chiave: 'scala', titolo: 'scala', nome: 'la scala', fatto: !!stages.scale,
         attesa: 'righello e righe #18-#21: parte dalle depth gia\' lette' }];

  panel.append(el('p', { class: 'hint' },
    'girano come i moduli della pipeline, in sottoprocesso: stesso codice, stessi '
    + 'artefatti su disco. Uno alla volta: quello che non lanci resta com\'e\', '
    + 'con i suoi risultati.'));

  for (const modulo of moduli) {
    const status = el('span', { class: 'hint' },
      stagesGate.blocked_reason || (rect.rect_echo
        ? modulo.attesa
        : 'serve prima il rettangolo ecografico'));
    const run = el('button', { class: modulo.fatto ? 'ghost' : '' },
      `${modulo.fatto ? 'Rifai' : 'Calcola'} ${modulo.nome}`);
    run.disabled = !rect.rect_echo || !stagesGate.ready;
    run.addEventListener('click', async () => {
      run.disabled = true;
      try {
        const { job_id } = await api(`/projects/${state.projectId}/analyze_stages`,
          { body: { sample: 12, stages: [modulo.chiave] } });
        await pollJob(job_id, status);
        toast(`${modulo.titolo}: fatto`);
        await reload();
      } catch (error) {
        toast(error.message, true);
        status.textContent = error.message;
      } finally { run.disabled = false; }
    });
    panel.append(el('div', { class: 'row' }, run, status));
  }

  const card = (title, rows) => {
    const box = el('div', { class: 'card' });
    box.append(el('h3', { style: 'margin-top:0' }, title));
    for (const [key, value] of rows) {
      box.append(el('div', { class: 'kv' }, el('span', {}, key), el('span', {}, String(value ?? '—'))));
    }
    panel.append(box);
  };

  if (step.id === 'depth_scale' && stages.depth && stages.depth.status === 'ok') {
    const host = el('div', {});
    panel.append(host);
    const sampleSize = ((state.project.steps.import || {}).value || {}).image_sample_size;
    createDepthViewer(state.projectId, sampleSize)
      .then((node) => host.append(node))
      .catch((error) => host.append(el('p', { class: 'hint' }, error.message)));
  }

  if (step.id === 'orientation' && stages.marker && stages.marker.status === 'ok') {
    const host = el('div', {});
    panel.append(host);
    const sampleSize = ((state.project.steps.import || {}).value || {}).image_sample_size;
    createOrientationViewer(state.projectId, sampleSize)
      .then((node) => host.append(node))
      .catch((error) => host.append(el('p', { class: 'hint' }, error.message)));
  }

  if (step.id === 'orientation' && stages.marker) {
    const m = stages.marker;
    card('marker di orientamento', [
      ['esito', `${m.status}${m.error ? ' — ' + m.error : ''}`],
      ['gruppi trovati', (m.groups || []).join(' · ') || '—'],
      ['gruppi assenti', (m.missing_groups || []).join(' · ') || 'nessuno'],
      ['#16 completata copiando', (m.filled_groups || []).length
        ? `${m.filled_groups.join(' · ')} da ${m.filled_from}`
        : 'nessuna copia: tutti e quattro trovati'],
      ['artefatti', m.output_dir],
    ]);
    if ((m.missing_groups || []).length) {
      panel.append(el('p', { class: 'hint' },
        `${m.missing_groups.join(' e ')} non compaiono in nessuna immagine: sono stati ` +
        'tolti dagli orientamenti disponibili nella pagina codici, e in #16 il loro posto ' +
        `porta una copia di ${m.filled_from} — la convenzione dei file legacy. ` +
        'Se invece quelle acquisizioni dovrebbero esserci, rimettili dalla pagina codici.'));
    }
  }

  if (step.id === 'depth_scale') {
    if (stages.depth) {
      const d = stages.depth;
      card('modulo depth', [
        ['esito', `${d.status}${d.error ? ' — ' + d.error : ''}`],
        ['immagini', `${d.accepted} accettate su ${d.images} (${d.acceptance_ratio})`],
        ['depth lette (mm)', (d.depths_mm || []).join(' · ') || '—'],
        ['artefatti', d.output_dir],
      ]);
    }
    if (stages.su_giu) {
      const s = stages.su_giu;
      card('orientamento su/giu (serve alla scala)', [
        ['conteggi', JSON.stringify(s.counts || {})],
        ['maggioranza', s.majority],
        ['confidenza media', s.mean_confidence],
      ]);
    }
    if (stages.scale) {
      const s = stages.scale;
      card('modulo scala', [
        ['esito', `${s.status}${s.error ? ' — ' + s.error : ''}`],
        ['profilo vendor', s.profile],
        ['depth', `${s.depths_accepted} accettate su ${s.depths_total} (${s.acceptance_ratio})`],
        ['#18 VECT_DEPTH', (s.lines || {}).line_18_vect_depth || '(vuota)'],
        ['#19 PIXEL_RATIO_X', (s.lines || {}).line_19_pixel_ratio_x || '(vuota)'],
        ['#20 PIXEL_RATIO_Y', (s.lines || {}).line_20_pixel_ratio_y || '(vuota)'],
        ['#21 SCALE_LINE', ((s.lines || {}).line_21_scale_line || '(vuota)').slice(0, 90)],
        ['artefatti', s.output_dir],
      ]);
      if ((s.review_reasons || []).length) {
        panel.append(el('ul', { class: 'problems' },
          s.review_reasons.map((reason) => el('li', {}, reason))));
      }
    }
  }

  panel.append(el('h3', {}, 'valore dello step'));
  valoreDiModulo(panel, step);
}

/* Il valore di uno step scritto dai moduli si guarda, non si riscrive a mano.

   Il pannello ne tiene una copia di quando e' stato disegnato, e li' dentro vivono la
   rilettura della depth su tutta la cartella, i riquadri stretti sul numero e le
   correzioni. Un «Conferma step» che rispedisce quella copia riporta indietro tutto il
   lavoro fatto nel frattempo — ed e' esattamente quello che succedeva. Qui si conferma lo
   stato e basta; il valore resta quello che c'e' sul disco. */
function valoreDiModulo(panel, step) {
  const stored = (state.project.steps[step.id] || {}).value || {};
  const testo = JSON.stringify(stored, null, 1);
  const dettaglio = el('details', {},
    el('summary', { class: 'hint' }, `il valore com'e' adesso (${testo.length} caratteri)`),
    el('pre', { class: 'out' }, testo.length > 20000 ? testo.slice(0, 20000) + '\n…' : testo));
  panel.append(dettaglio);
  panel.append(el('p', { class: 'hint' },
    'lo scrivono i moduli: confermare non lo tocca, marca solo lo step come confermato.'));
  const confermato = (state.project.steps[step.id] || {}).status === 'confirmed';
  panel.append(el('div', { class: 'row' },
    confermaInDueTempi(confermato ? 'Togli la conferma' : 'Conferma step',
      confermato ? 'lo step torna a essere una proposta.'
        : 'lo step vale come confermato. Il valore non viene toccato.',
      async () => {
        try {
          const esito = await api(`/projects/${state.projectId}/steps/${step.id}/confirm`,
            { body: confermato ? { reset: true } : {} });
          toast(esito.confirmed ? 'step confermato' : 'conferma tolta');
          await reload();
        } catch (error) { toast(error.message, true); }
      })));
}

/* --- step generico: editor del valore --- */
function panelGeneric(panel, step) {
  const stored = (state.project.steps[step.id] || {}).value || {};
  const area = el('textarea', {});
  area.value = JSON.stringify(stored, null, 1);
  panel.append(el('p', { class: 'hint' },
    'gli stadi di inferenza non sono ancora agganciati a questo step: il valore si inserisce a mano, ' +
    'nella forma che il writer .fss si aspetta.'));
  panel.append(el('pre', { class: 'out' }, VALUE_HINTS[step.id] || '{}'));
  panel.append(area);
  panel.append(el('p', { class: 'hint' }, BLOCK_HINT));
  panel.append(saveRow(step.id, () => JSON.parse(area.value || '{}')));
}

function anagraficaCard(codes) {
  const card = el('div', { class: 'card' });
  card.append(el('h3', { style: 'margin-top:0' }, 'Anagrafica — encoding_struct'));
  card.append(el('p', { class: 'hint' },
    'da modello ecografo e modello sonda ricava ID ecografo (#02), ID sonda (#03), ID NDG (#05), ' +
    'tipo sonda (#04) e ID setup. Se la combinazione non esiste, la riga si crea qui.'));

  const recognised = ((state.project.analysis || {}).vendor || {}).vendor || '';
  const echo = el('input', {
    type: 'text', value: codes.echo_model || recognised, placeholder: 'es. Esaote, MyLabSigma',
  });
  if (!codes.echo_model && recognised) {
    card.append(el('p', { class: 'hint' }, `ecografo riconosciuto dalla rete: ${recognised}`));
  }
  const probe = el('input', { type: 'text', value: codes.probe_model || '', placeholder: 'es. TLC3-13' });
  card.append(el('div', { class: 'grid' },
    el('div', {}, el('label', {}, 'Modello ecografo'), echo),
    el('div', {}, el('label', {}, 'Modello sonda'), probe)));

  const results = el('div', {});
  const search = async () => {
    results.innerHTML = '';
    try {
      const data = await api('/anagrafica/resolve', {
        body: { echo_model: echo.value, probe_model: probe.value },
      });
      renderAnagrafica(results, data, echo.value, probe.value, search);
    } catch (error) { toast(error.message, true); }
  };

  card.append(el('div', { class: 'row' },
    el('button', { onclick: search }, 'Cerca in encoding_struct'),
    el('button', {
      class: 'ghost',
      onclick: () => applyCodes({ echo_model: echo.value, probe_model: probe.value }, 'modelli salvati'),
    }, 'Salva i modelli')));
  card.append(results);
  card.append(candidatesPicker(codes));
  return card;
}

/* La rete riconosce la marca, non il modello: la stessa sonda vive su decine di macchine
   diverse, quindi ID ecografo si scegle da una lista, non si indovina. */
function candidatesPicker(codes) {
  const host = el('div', {});
  const analysis = state.project.analysis || {};
  const probeId = (analysis.probe || {}).probe_id;
  if (probeId == null) return host;
  const ambiguous = (analysis.combination || {}).ambiguous;
  if (!ambiguous && codes.id_echo) return host;

  host.append(el('h3', {}, 'scegli il modello di ecografo'));
  host.append(el('p', { class: 'hint' },
    `la rete ha riconosciuto la marca (${(analysis.vendor || {}).vendor || '?'}) e la sonda ` +
    `(ID ${probeId}), ma in anagrafica la stessa sonda compare su piu' macchine: ` +
    'ID ecografo, ID NDG e ID setup si prendono dalla riga che scegli qui.'));

  const select = el('select', { style: 'max-width:100%' });
  select.append(el('option', { value: '' }, 'caricamento...'));
  let rows = [];
  const apply = el('button', {}, 'Usa questo setup');
  apply.disabled = true;

  (async () => {
    try {
      const data = await api('/anagrafica/candidates', {
        body: { probe_id: probeId, probe_type: codes.probe_type || null },
      });
      rows = data.candidates || [];
      select.innerHTML = '';
      select.append(el('option', { value: '' }, `— ${rows.length} setup registrati —`));
      rows.forEach((row, index) => {
        select.append(el('option', { value: String(index) },
          `${row.echo_model}  →  eco ${row.id_echo} · ndg ${row.id_ndg} · setup ${row.id_setup}`));
      });
      apply.disabled = false;
    } catch (error) {
      select.innerHTML = '';
      select.append(el('option', { value: '' }, error.message));
    }
  })();

  apply.addEventListener('click', () => {
    const row = rows[parseInt(select.value, 10)];
    if (!row) return toast('scegli un setup', true);
    const patch = { echo_model: row.echo_model };
    if (row.id_echo) patch.id_echo = parseInt(row.id_echo, 10);
    if (row.id_ndg) patch.id_ndg = parseInt(row.id_ndg, 10);
    if (row.id_setup) patch.id_setup = parseInt(row.id_setup, 10);
    if (row.probe_type) patch.probe_type = parseInt(row.probe_type, 10);
    applyCodes(patch, 'setup scelto dall\'anagrafica');
  });

  host.append(select);
  host.append(el('div', { class: 'row' }, apply));
  return host;
}

function renderAnagrafica(host, data, echoModel, probeModel, refresh) {
  const proposal = data.proposal || {};
  const found = Object.entries(proposal).filter(([, value]) => value !== null && value !== undefined);

  if (found.length) {
    host.append(el('h3', {}, 'trovato nell\'anagrafica'));
    const table = el('table', { class: 'lines' });
    const labels = {
      id_echo: 'ID ecografo (#02)', id_probe: 'ID sonda (#03)', id_ndg: 'ID NDG (#05)',
      probe_type: 'Tipo sonda (#04)', id_setup: 'ID setup',
    };
    for (const [key, value] of found) {
      table.append(el('tr', {},
        el('td', { class: 'name' }, labels[key] || key),
        el('td', { class: 'val' }, String(value)),
        el('td', { class: 'name' }, data.sources[key] || '')));
    }
    host.append(table);
    const patch = {};
    for (const [key, value] of found) patch[key] = value;
    patch.echo_model = echoModel;
    patch.probe_model = probeModel;
    host.append(el('div', { class: 'row' },
      el('button', { onclick: () => applyCodes(patch, 'codici presi dall\'anagrafica') },
        'Usa questi codici')));
  }

  const combos = data.combination_matches || [];
  if (combos.length) {
    host.append(el('h3', {}, 'combinazioni eco+sonda simili (foglio FSS)'));
    const table = el('table', { class: 'lines' });
    for (const row of combos) {
      table.append(el('tr', {},
        el('td', { class: 'n' }, row.score),
        el('td', { class: 'val' }, `${row.echo_model} + ${row.probe_model}`),
        el('td', { class: 'name' },
          `tipo ${row.probe_type || '?'} · ndg ${row.id_ndg || '?'} · fss ${row.fss || '?'}`)));
    }
    host.append(table);
  }

  for (const item of data.missing || []) {
    const label = item.sheet === 'ECO' ? 'ecografo' : 'sonda';
    host.append(el('h3', {}, `manca in ${item.sheet}`));
    host.append(el('p', { class: 'hint' },
      `"${item.query || '(vuoto)'}" non e' in anagrafica: la nuova riga prenderebbe id ${item.next_id}.`));
    host.append(el('div', { class: 'row' },
      el('button', {
        class: 'ghost',
        onclick: async () => {
          if (!item.query) return toast('scrivi prima il modello', true);
          if (!confirm(`Aggiungere "${item.query}" al foglio ${item.sheet} con id ${item.next_id}?\n\n` +
            'Il file encoding_struct viene modificato (con backup automatico).')) return;
          try {
            const result = await api('/anagrafica/new_row', {
              body: { kind: item.sheet === 'ECO' ? 'echo' : 'probe', model: item.query },
            });
            toast(`riga ${result.row} creata con id ${result.id} · backup: ${result.backup.split('/').pop()}`);
            await refresh();
          } catch (error) { toast(error.message, true); }
        },
      }, `Crea la riga ${label} con id ${item.next_id}`)));
  }
}

function saveRow(stepId, getValue) {
  return el('div', { class: 'row' },
    el('button', {
      onclick: async () => {
        let value;
        try { value = getValue(); } catch (error) { return toast('JSON non valido: ' + error.message, true); }
        try {
          // Il pannello ha in mano una copia di quando e' stato disegnato. Se nel frattempo
          // qualcosa ha scritto su quello step — un modulo, un job — rispedire la copia lo
          // riporterebbe indietro senza dirlo. Meglio fermarsi e far ricaricare.
          const fresco = (await api(`/projects/${state.projectId}`)).project || {};
          const prima = JSON.stringify(((state.project.steps || {})[stepId] || {}).value || {});
          const adesso = JSON.stringify(((fresco.steps || {})[stepId] || {}).value || {});
          if (prima !== adesso) {
            toast('lo step e\' cambiato mentre era aperto: ricarico, poi riprova', true);
            await reload();
            return;
          }
          const result = await api(`/projects/${state.projectId}/steps/${stepId}`, { body: { value } });
          toast(result.stale.length ? 'salvato · tornati in review: ' + result.stale.join(', ') : 'salvato');
          await reload();
        } catch (error) { toast(error.message, true); }
      },
    }, 'Conferma step'),
  );
}

/* --- ultimo step: generazione --- */
async function panelGenerate(panel) {
  const preview = await api(`/projects/${state.projectId}/preview`);

  panel.append(el('h3', {}, preview.problems.length ? 'da sistemare prima di generare' : 'controlli'));
  if (preview.problems.length) {
    panel.append(el('ul', { class: 'problems' }, preview.problems.map((p) => el('li', {}, p))));
  } else {
    panel.append(el('p', { class: 'ok-badge' }, 'nessun problema di validazione'));
  }

  panel.append(el('h3', {}, `file ${preview.is_biplane ? 'biplana (26 righe)' : 'sonda singola (23 righe)'}`));
  const table = el('table', { class: 'lines' });
  for (const line of preview.lines) {
    table.append(el('tr', { class: line.value ? '' : 'empty' },
      el('td', { class: 'n' }, '#' + String(line.n).padStart(2, '0')),
      el('td', { class: 'name' }, line.name),
      el('td', { class: 'val' }, line.value || '(vuota)')));
  }
  panel.append(table);

  const reference = el('input', {
    type: 'text', style: 'width:520px',
    placeholder: 'percorso di un .fss legacy di riferimento per il quality gate',
  });
  const output = el('pre', { class: 'out', style: 'display:none' });

  panel.append(el('div', { class: 'row' },
    el('button', {
      onclick: async () => {
        const result = await api(`/projects/${state.projectId}/generate`, { body: { force: false } });
        if (!result.written) return toast('non generato: ' + result.problems.length + ' problemi aperti', true);
        toast('scritto ' + result.path);
        await reload();
      },
    }, 'Genera .fss'),
    el('button', {
      class: 'ghost',
      onclick: async () => {
        const result = await api(`/projects/${state.projectId}/generate`, { body: { force: true } });
        toast(result.written ? 'scritto comunque: ' + result.path : 'non scritto', !result.written);
      },
    }, 'Genera comunque'),
  ));
  panel.append(el('div', { style: 'margin-top:10px' }, el('label', {}, 'Quality gate'), reference));
  panel.append(el('div', { class: 'row' },
    el('button', {
      class: 'ghost',
      onclick: async () => {
        try {
          const result = await api(`/projects/${state.projectId}/compare`, { body: { reference: reference.value } });
          output.style.display = 'block';
          output.textContent = (result.compatible ? '[compatibile] ' : '[differenze] ') +
            '\n' + (result.stdout || '') + (result.stderr || '');
        } catch (error) { toast(error.message, true); }
      },
    }, 'Confronta con il legacy (compare_fss.py)'),
    el('span', { class: 'hint' }, 'exit 0 = semanticamente compatibile'),
  ));
  panel.append(output);
}

boot().catch((error) => toast(error.message, true));


/* --- scegliere la cartella sfogliando, invece di incollare il percorso ------------------

   Il browser non puo' dare il percorso vero di una cartella scelta con `<input type=file>`:
   restituisce nomi relativi, e all'importer serve il percorso sul disco. Ma il server gira
   sulla stessa macchina, quindi e' lui a sfogliare, e la finestra qui sotto mostra il suo
   elenco. Accanto a ogni cartella c'e' quante immagini contiene e quante sottocartelle ha:
   e' quello che serve per capire, senza aprirla, se e' l'acquisizione giusta. */
function apriSelettoreCartella({ start, onPick }) {
  const velo = el('div', { class: 'fullscreen selettore' });
  const briciole = el('div', { class: 'selettore-briciole' });
  const elenco = el('div', { class: 'selettore-elenco' });
  const scorciatoie = el('div', { class: 'selettore-lati' });
  const stato = el('span', { class: 'hint' });
  const campo = el('input', { type: 'text', style: 'flex:1;min-width:260px' });

  let corrente = '';
  const chiudi = () => velo.remove();

  const scegli = () => {
    const percorso = campo.value.trim();
    if (!percorso) { toast('nessuna cartella scelta', true); return; }
    onPick(percorso);
    chiudi();
  };

  const vai = async (percorso) => {
    stato.textContent = 'leggo…';
    try {
      const q = percorso ? `?path=${encodeURIComponent(percorso)}` : '';
      const dati = await api(`/browse${q}`);
      corrente = dati.path || '';
      campo.value = corrente;
      stato.textContent = corrente
        ? `${dati.entries.length} sottocartelle · ${dati.images_here} immagini qui dentro`
        : 'scegli da dove partire';

      briciole.innerHTML = '';
      if (dati.parent) {
        const su = el('button', { class: 'ghost sq' }, '↑ cartella superiore');
        su.addEventListener('click', () => vai(dati.parent));
        briciole.append(su);
      }
      briciole.append(el('span', { class: 'selettore-qui' }, corrente || 'partenza'));

      scorciatoie.innerHTML = '';
      for (const radice of dati.roots || []) {
        const voce = el('button', { class: 'selettore-radice' },
          el('span', { class: 'selettore-icona' },
            radice.kind === 'volume' ? '⧉' : radice.kind === 'recente' ? '↺' : '⌂'),
          el('span', {}, radice.label));
        voce.addEventListener('click', () => vai(radice.path));
        scorciatoie.append(voce);
      }

      elenco.innerHTML = '';
      if (!corrente) {
        elenco.append(el('p', { class: 'hint' },
          'parti da una delle voci a sinistra: la home, un disco esterno, o una cartella '
          + 'gia\' usata in un altro progetto.'));
        return;
      }
      if (!dati.entries.length) {
        elenco.append(el('p', { class: 'hint' },
          dati.images_here
            ? `nessuna sottocartella: qui dentro ci sono ${dati.images_here} immagini, `
              + 'e\' probabilmente questa la cartella da importare.'
            : 'cartella vuota.'));
        return;
      }
      for (const voce of dati.entries) {
        const riga = el('div', { class: 'selettore-riga' },
          el('span', { class: 'selettore-icona' }, '▸'),
          el('span', { class: 'selettore-nome', title: voce.name }, voce.name),
          el('span', { class: 'hint' },
            voce.images < 0 ? 'non leggibile'
              : `${voce.images} immagini${voce.folders ? ` · ${voce.folders} sottocartelle` : ''}`));
        riga.addEventListener('click', () => vai(voce.path));
        elenco.append(riga);
      }
    } catch (errore) {
      stato.textContent = errore.message;
      toast(errore.message, true);
    }
  };

  const usa = el('button', {}, 'Usa questa cartella');
  usa.addEventListener('click', scegli);
  const annulla = el('button', { class: 'ghost' }, 'Annulla');
  annulla.addEventListener('click', chiudi);

  velo.append(
    el('div', { class: 'fullscreen-bar' },
      el('strong', {}, 'Scegli la cartella delle acquisizioni'), stato,
      el('span', { style: 'flex:1' }), annulla),
    briciole,
    el('div', { class: 'selettore-corpo' }, scorciatoie, elenco),
    el('div', { class: 'row selettore-scelta' }, campo, usa),
  );
  velo.addEventListener('keydown', (e) => { if (e.key === 'Escape') chiudi(); });
  campo.addEventListener('keydown', (e) => { if (e.key === 'Enter') vai(campo.value.trim()); });
  document.body.append(velo);
  vai(start && start.trim() ? start.trim() : '');
}
