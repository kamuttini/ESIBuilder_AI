/* Revisione snella — vanilla JS, no build step.
 *
 * Three views over one idea: run the official pipeline on the folders you choose, then look at
 * every module's output on every image side by side, and say what is wrong where you see it.
 * Comments and corrections go to the same inbox with the context filled in by the server.
 */

'use strict';

const S = {
  catalog: null,
  view: 'home',
  roots: [],
  root: '',
  folders: [],
  selected: new Set(),
  history: [],
  run: null,
  runPollTimer: null,
  eventsSince: 0,
  logLines: [],
  study: null,
  studyRef: null,          // {runId, slug, folderName}
  sampleIndex: 0,
  gridSize: 220,
  detailIndex: -1,
  overlayOff: new Set(),
  drawer: null,
  feedback: { entries: [], stats: null },
};

/* ------------------------------------------------------------------ utils */
const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key === 'html') node.innerHTML = value;
      else if (key.startsWith('on') && typeof value === 'function') {
        node.addEventListener(key.slice(2), value);
      } else node.setAttribute(key, value);
    }
  }
  for (const child of [].concat(children || [])) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
  }
  return node;
}

async function api(path, options) {
  const response = await fetch(path, Object.assign({
    headers: { 'Content-Type': 'application/json' },
  }, options || {}));
  let payload = null;
  try { payload = await response.json(); } catch (err) { payload = null; }
  if (!response.ok) {
    const message = (payload && payload.error) || `errore ${response.status}`;
    throw new Error(message);
  }
  return payload;
}

let toastTimer = null;
function toast(message, isError) {
  const node = $('#toast');
  node.textContent = message;
  node.className = 'toast' + (isError ? ' err' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.add('hidden'), 3600);
}

const num = (value, digits) => (value === null || value === undefined || value === '')
  ? '—'
  : (typeof value === 'number' ? value.toFixed(digits === undefined ? 2 : digits) : String(value));
const pct = (value) => (value === null || value === undefined || value === '')
  ? '—' : `${(Number(value) * 100).toFixed(0)}%`;
const txt = (value) => (value === null || value === undefined || value === '') ? '—' : String(value);

function imageUrl(path, width) {
  return width
    ? `/api/thumb?w=${width}&path=${encodeURIComponent(path)}`
    : `/api/image?path=${encodeURIComponent(path)}`;
}

function cropUrl(path, box, zoom, pad) {
  const q = new URLSearchParams({
    path: path,
    left: Math.round(box.left), top: Math.round(box.top),
    right: Math.round(box.right), bottom: Math.round(box.bottom),
    zoom: String(zoom || 3), pad: String(pad === undefined ? 24 : pad),
  });
  return `/api/crop?${q.toString()}`;
}

function kv(pairs) {
  const list = el('dl', { class: 'kv' });
  for (const [key, value] of pairs) {
    if (value === undefined) continue;
    list.appendChild(el('dt', { text: key }));
    list.appendChild(el('dd', { text: value === null ? '—' : String(value) }));
  }
  return list;
}

/* ------------------------------------------------------------------ views */
function setView(name) {
  S.view = name;
  $$('.view').forEach((node) => node.classList.toggle('active', node.id === `view-${name}`));
  $$('.tab').forEach((node) => node.classList.toggle('active', node.dataset.view === name));
  if (name === 'feedback') loadFeedback();
  if (name === 'orchestrator') loadOrchestrator();
}

/* ================================================================ HOME */
async function loadRoots() {
  try {
    const data = await api('/api/roots');
    S.roots = data.roots || [];
    const list = $('#root-list');
    list.innerHTML = '';
    S.roots.forEach((root) => list.appendChild(el('option', { value: root.path })));
  } catch (err) { /* the picker still accepts a typed path */ }
}

async function loadFolders() {
  const root = $('#root-input').value.trim();
  if (!root) { toast('scrivi o scegli una cartella radice', true); return; }
  $('#folders-hint').textContent = 'lettura in corso…';
  try {
    const data = await api(`/api/folders?root=${encodeURIComponent(root)}`);
    S.root = data.root;
    S.folders = data.folders || [];
    S.selected.clear();
    $('#folder-picker').classList.remove('hidden');
    $('#folders-hint').textContent = `${S.folders.length} cartelle`;
    renderFolders();
  } catch (err) {
    $('#folders-hint').textContent = '';
    toast(err.message, true);
  }
}

function renderFolders() {
  const filter = $('#folder-filter').value.trim().toLowerCase();
  const list = $('#folder-list');
  list.innerHTML = '';
  const visible = S.folders.filter((f) => !filter || f.name.toLowerCase().includes(filter));
  visible.forEach((folder) => {
    const box = el('input', {
      type: 'checkbox',
      onchange: (event) => {
        if (event.target.checked) S.selected.add(folder.name);
        else S.selected.delete(folder.name);
        updateSelectionCount();
      },
    });
    box.checked = S.selected.has(folder.name);
    list.appendChild(el('li', {}, [
      box,
      el('span', { class: 'fname', title: folder.path, text: folder.name }),
      el('span', { class: 'muted small', text: `${folder.images}${folder.images_capped ? '+' : ''} img` }),
    ]));
  });
  if (!visible.length) list.appendChild(el('li', { class: 'muted', text: 'nessuna cartella' }));
  updateSelectionCount();
}

function updateSelectionCount() {
  $('#folder-selected-count').textContent = String(S.selected.size);
  $('#btn-start').disabled = S.selected.size === 0;
}

function selectFolders(mode) {
  const filter = $('#folder-filter').value.trim().toLowerCase();
  const visible = S.folders.filter((f) => !filter || f.name.toLowerCase().includes(filter));
  visible.forEach((folder) => {
    if (mode === 'all') S.selected.add(folder.name);
    else if (mode === 'none') S.selected.delete(folder.name);
    else if (mode === 'invert') {
      if (S.selected.has(folder.name)) S.selected.delete(folder.name);
      else S.selected.add(folder.name);
    }
  });
  renderFolders();
}

function runOptions() {
  return {
    sample_per_folder: Number($('#opt-sample').value || 40),
    depth_max_images: Number($('#opt-depth-images').value || 0),
    scale_max_frames: Number($('#opt-scale-frames').value || 48),
    previews: $('#opt-previews').checked,
    disable_depth: $('#opt-depth').checked,
    disable_scale: $('#opt-scale').checked,
    disable_lt: $('#opt-lt').checked,
  };
}

async function startRun() {
  $('#start-error').textContent = '';
  const body = {
    dataset_root: S.root,
    folders: Array.from(S.selected),
    options: runOptions(),
  };
  try {
    const data = await api('/api/runs', { method: 'POST', body: JSON.stringify(body) });
    S.run = data.run;
    S.eventsSince = 0;
    S.logLines = [];
    setView('run');
    renderRun();
    startRunPolling();
    loadHistory();
  } catch (err) {
    $('#start-error').textContent = err.message;
  }
}

async function loadHistory() {
  try {
    const data = await api('/api/runs');
    S.history = data.runs || [];
    renderHistory();
    if (data.active && (!S.run || S.run.run_id !== data.active)) {
      const active = await api(`/api/runs/${data.active}`);
      S.run = active.run;
      startRunPolling();
    }
  } catch (err) { /* history is not critical */ }
}

function renderHistory() {
  const box = $('#history');
  box.innerHTML = '';
  if (!S.history.length) {
    box.appendChild(el('div', { class: 'muted', text: 'ancora nessuna run' }));
    return;
  }
  S.history.forEach((entry) => {
    const state = el('span', { class: `pill ${entry.state}`, text: entry.state });
    const row = el('div', {
      class: 'hrow',
      onclick: () => openRun(entry.run_id),
    }, [
      el('div', {}, [
        el('strong', { text: entry.run_id }),
        el('div', { class: 'hmeta', text: `${entry.created} · ${entry.dataset_root}` }),
        el('div', { class: 'hfolders', text: entry.folder_names.slice(0, 4).join(', ')
          + (entry.folder_names.length > 4 ? ` +${entry.folder_names.length - 4}` : '') }),
      ]),
      el('div', { class: 'stack' }, [
        state,
        el('span', { class: 'muted small', text: `${entry.folders_done}/${entry.folders_total} cartelle` }),
        entry.feedback_count
          ? el('span', { class: 'muted small', text: `${entry.feedback_count} feedback` })
          : null,
      ]),
    ]);
    box.appendChild(row);
  });
}

async function openRun(runId) {
  try {
    const data = await api(`/api/runs/${runId}`);
    S.run = data.run;
    S.eventsSince = 0;
    S.logLines = [];
    setView('run');
    renderRun();
    startRunPolling();
  } catch (err) { toast(err.message, true); }
}

/* ================================================================ RUN */
function startRunPolling() {
  clearInterval(S.runPollTimer);
  pollRun();
  S.runPollTimer = setInterval(pollRun, 1500);
}

async function pollRun() {
  if (!S.run) { clearInterval(S.runPollTimer); return; }
  try {
    const data = await api(`/api/runs/${S.run.run_id}/events?since=${S.eventsSince}`);
    S.eventsSince = data.next;
    S.run.state = data.state;
    S.run.folders = data.folders;
    (data.events || []).forEach((event) => {
      if (event.type === 'log') S.logLines.push(`[${event.folder || ''}] ${event.line}`);
      if (event.type === 'folder_failed') S.logLines.push(`ERRORE ${event.folder}: ${event.error}`);
    });
    if (S.logLines.length > 400) S.logLines = S.logLines.slice(-400);
    renderRun();
    if (data.state !== 'running') {
      clearInterval(S.runPollTimer);
      loadHistory();
    }
  } catch (err) {
    clearInterval(S.runPollTimer);
  }
}

function stageSummary(stage, payload) {
  if (!payload) return '';
  switch (stage) {
    case 'dedup':
      return `${payload.images_unique} usate, ${payload.duplicates_removed} rimosse`;
    case 'rotazione':
      return `${payload.rotation_deg_clockwise}°`;
    case 'vendor':
      return `${payload.vendor} ${pct(payload.confidence)}`;
    case 'probe':
      return `#${payload.probe_id} ${pct(payload.confidence)}`;
    case 'rect':
      return `${payload.line_11 || '—'}${payload.provisional ? ' (mediana)' : ''}`;
    case 'orientamento':
      return `${payload.su_giu_majority || '—'} / ${payload.lr_marker_majority || '—'}`;
    case 'depth':
      return `${payload.accepted}/${payload.images_predicted} ok`;
    case 'scala':
      return `${payload.depths_accepted}/${payload.depths_total} depth`;
    case 'folder_done':
      return payload.status || 'fatto';
    default:
      return '';
  }
}

function renderRun() {
  if (!S.run) return;
  $('#run-title').textContent = `${S.run.run_id} — ${S.run.dataset_root}`;
  $('#run-status').textContent = S.run.state;
  $('#run-status').className = `pill ${S.run.state}`;
  $('#btn-cancel-run').classList.toggle('hidden', S.run.state !== 'running');

  const box = $('#run-folders');
  box.innerHTML = '';
  (S.run.folders || []).forEach((job) => {
    const stages = el('div', { class: 'stages' });
    (S.catalog.stage_order || []).forEach((stage) => {
      const payload = (job.stages || {})[stage];
      const node = el('div', { class: 'stage' + (payload ? ' done' : '') }, [
        el('span', { text: S.catalog.stage_labels[stage] || stage }),
        el('span', { class: 'sv', text: payload ? stageSummary(stage, payload) : '…' }),
      ]);
      stages.appendChild(node);
    });
    const canOpen = job.state === 'done' || Object.keys(job.stages || {}).length > 0;
    box.appendChild(el('div', { class: 'rf' }, [
      el('div', { class: 'rf-head' }, [
        el('div', {}, [
          el('strong', { text: job.name }),
          el('div', { class: 'muted small', text: job.error || job.path }),
        ]),
        el('div', { class: 'row' }, [
          el('span', { class: `pill ${job.state}`, text: job.state }),
          canOpen ? el('button', {
            class: 'small',
            text: 'apri revisione',
            onclick: () => openStudy(S.run.run_id, job.slug, job.name),
          }) : null,
        ]),
      ]),
      stages,
    ]));
  });
  $('#run-log').textContent = S.logLines.join('\n');
}

/* ================================================================ REVIEW */
async function openStudy(runId, slug, folderName) {
  setView('review');
  $('#review-empty').classList.remove('hidden');
  $('#review-empty').textContent = 'caricamento…';
  $('#review-body').classList.add('hidden');
  try {
    const data = await api(`/api/runs/${runId}/folders/${slug}/study`);
    if (!data.ready) {
      $('#review-empty').textContent = data.message
        + ' — la revisione si apre quando la cartella ha finito.';
      return;
    }
    S.study = data;
    S.studyRef = { runId: runId, slug: slug, folderName: folderName || data.folder.folder_name };
    S.sampleIndex = 0;
    $('#review-empty').classList.add('hidden');
    $('#review-body').classList.remove('hidden');
    renderReview();
  } catch (err) {
    $('#review-empty').textContent = err.message;
  }
}

function feedbackFor(area, imageId) {
  return (S.study.feedback || []).filter((entry) => {
    if (area && entry.area !== area) return false;
    const target = entry.target || {};
    if (imageId) return String(target.image_id || '') === String(imageId);
    return !target.image_id;
  });
}

function feedbackCountForImage(imageId) {
  return (S.study.feedback || []).filter(
    (entry) => String((entry.target || {}).image_id || '') === String(imageId)).length;
}

function renderReview() {
  const folder = S.study.folder;
  $('#rv-folder').textContent = folder.folder_name;
  $('#rv-dedup').textContent = folder.dedup.text;

  const headline = $('#rv-headline');
  headline.innerHTML = '';
  const items = [
    ['Vendor', folder.vendor.value || '—', `conf ${pct(folder.vendor.confidence)}`,
      folder.vendor.confidence >= folder.vendor.threshold ? 'ok' : 'warn'],
    ['Sonda', folder.probe.value ? `#${folder.probe.value}` : '—',
      folder.probe.name || `conf ${pct(folder.probe.confidence)}`,
      folder.probe.confidence >= folder.probe.threshold ? 'ok' : 'warn'],
    ['#11 rect', folder.rect.line_11 || '—', folder.rect.method || '', 'ok'],
    ['Depth', `${folder.depth.accepted}/${folder.depth.images}`,
      folder.depth.majority_mode || folder.depth.status || '',
      folder.depth.status === 'ok' ? 'ok' : 'warn'],
    ['Scala', `${folder.scala.depths_accepted}/${folder.scala.depths_total}`,
      folder.scala.source || folder.scala.status || '',
      folder.scala.status === 'ok' ? 'ok' : 'warn'],
    ['Esito', folder.status || '—', (folder.review_reasons || []).join(', '),
      folder.status === 'ok' ? 'ok' : 'warn'],
  ];
  items.forEach(([label, value, sub, level]) => {
    headline.appendChild(el('div', { class: `hl ${level}` }, [
      el('span', { class: 'muted', text: label }),
      el('b', { text: String(value) }),
      sub ? el('span', { class: 'muted', text: sub }) : null,
    ]));
  });

  const checks = $('#rv-folder-checks');
  checks.innerHTML = '';
  const folderChecks = S.study.folder_checks || [];
  if (!folderChecks.length) {
    checks.appendChild(el('div', { class: 'chk ok' }, [
      el('span', { text: 'Nessuna incoerenza fra aree su questa cartella.' }),
    ]));
  }
  folderChecks.forEach((check) => {
    checks.appendChild(el('div', { class: `chk ${check.level}` }, [
      el('span', { class: 'areas', text: check.areas }),
      el('span', { text: check.message }),
      el('button', {
        class: 'ghost small',
        text: 'commenta',
        onclick: () => openDrawer({
          area: check.areas.includes('+') ? 'orchestratore' : check.areas,
          scope: 'folder',
          prefill: check.message,
        }),
      }),
    ]));
  });

  renderSample();
  renderGrid();
}

/* ---------------------------------------------------------------- overlays */
function overlayBox(box, cls, label, width, height) {
  if (!box || !width || !height) return null;
  const left = Number(box.left), top = Number(box.top);
  const right = Number(box.right), bottom = Number(box.bottom);
  if ([left, top, right, bottom].some((v) => !isFinite(v))) return null;
  const node = el('div', {
    class: `ov ${cls}`,
    style: `left:${(left / width) * 100}%;top:${(top / height) * 100}%;`
      + `width:${((right - left) / width) * 100}%;height:${((bottom - top) / height) * 100}%`,
  });
  if (label) node.appendChild(el('span', { class: 'ov-label', text: label }));
  return node;
}

/** An image with overlays scaled to it, plus optional drag/click capture for corrections. */
function buildCanvas(container, image, overlays, interaction) {
  container.innerHTML = '';
  const path = image.image_path;
  const img = el('img', { src: imageUrl(path, 1100), alt: image.image_id });
  container.appendChild(img);

  const draw = () => {
    const width = image.image_width || img.naturalWidth;
    const height = image.image_height || img.naturalHeight;
    $$('.ov, .ov-line, .ov-pin', container).forEach((node) => node.remove());
    (overlays || []).forEach((entry) => {
      if (S.overlayOff.has(entry.key)) return;
      if (entry.kind === 'box') {
        const node = overlayBox(entry.box, entry.cls, entry.label, width, height);
        if (node) container.appendChild(node);
      } else if (entry.kind === 'vline' && isFinite(entry.x)) {
        container.appendChild(el('div', {
          class: 'ov-line vert',
          style: `left:${(entry.x / width) * 100}%;top:0;height:100%`,
        }));
      } else if (entry.kind === 'hline' && isFinite(entry.y)) {
        container.appendChild(el('div', {
          class: 'ov-line horiz',
          style: `top:${(entry.y / height) * 100}%;left:0;width:100%`,
        }));
      } else if (entry.kind === 'segment' && isFinite(entry.x) && isFinite(entry.y1)
          && isFinite(entry.y2)) {
        const top = Math.min(entry.y1, entry.y2), bottom = Math.max(entry.y1, entry.y2);
        container.appendChild(el('div', {
          class: 'ov-line vert',
          style: `left:${(entry.x / width) * 100}%;top:${(top / height) * 100}%;`
            + `height:${((bottom - top) / height) * 100}%;border-left-style:solid`,
        }));
        container.appendChild(el('div', {
          class: 'ov-pin',
          style: `left:${(entry.x / width) * 100}%;top:${(entry.y1 / height) * 100}%`,
        }));
      }
    });
  };

  if (img.complete) draw(); else img.addEventListener('load', draw);

  if (interaction) attachInteraction(container, img, image, interaction);
  return { redraw: draw };
}

function attachInteraction(container, img, image, interaction) {
  const layer = el('div', { class: 'drag-layer' });
  container.appendChild(layer);
  const toImage = (event) => {
    const rect = layer.getBoundingClientRect();
    const width = image.image_width || img.naturalWidth || rect.width;
    const height = image.image_height || img.naturalHeight || rect.height;
    return {
      x: Math.round(((event.clientX - rect.left) / rect.width) * width),
      y: Math.round(((event.clientY - rect.top) / rect.height) * height),
    };
  };

  if (interaction.mode === 'box') {
    let start = null;
    let ghost = null;
    layer.addEventListener('mousedown', (event) => {
      start = toImage(event);
      ghost = el('div', { class: 'drag-box' });
      layer.appendChild(ghost);
      event.preventDefault();
    });
    layer.addEventListener('mousemove', (event) => {
      if (!start || !ghost) return;
      const rect = layer.getBoundingClientRect();
      const now = toImage(event);
      const width = image.image_width || img.naturalWidth;
      const height = image.image_height || img.naturalHeight;
      const left = Math.min(start.x, now.x), top = Math.min(start.y, now.y);
      ghost.style.left = `${(left / width) * 100}%`;
      ghost.style.top = `${(top / height) * 100}%`;
      ghost.style.width = `${(Math.abs(now.x - start.x) / width) * 100}%`;
      ghost.style.height = `${(Math.abs(now.y - start.y) / height) * 100}%`;
    });
    const finish = (event) => {
      if (!start) return;
      const now = toImage(event);
      const box = {
        left: Math.min(start.x, now.x), top: Math.min(start.y, now.y),
        right: Math.max(start.x, now.x), bottom: Math.max(start.y, now.y),
      };
      start = null;
      if (ghost) { ghost.remove(); ghost = null; }
      if (box.right - box.left > 3 && box.bottom - box.top > 3) interaction.onBox(box);
    };
    layer.addEventListener('mouseup', finish);
    layer.addEventListener('mouseleave', () => { if (ghost) { ghost.remove(); ghost = null; } start = null; });
  } else if (interaction.mode === 'point') {
    layer.addEventListener('click', (event) => interaction.onPoint(toImage(event)));
  }
}

/* ---------------------------------------------------------------- sample */
function renderSample() {
  const images = S.study.images || [];
  if (!images.length) return;
  S.sampleIndex = Math.max(0, Math.min(S.sampleIndex, images.length - 1));
  const image = images[S.sampleIndex];
  $('#sample-label').textContent = `${S.sampleIndex + 1}/${images.length} — ${image.image_id}`;

  const folder = S.study.folder;
  buildCanvas($('#sample-canvas'), image, [
    { kind: 'box', key: 'rect', box: folder.rect.box, cls: 'rect', label: '#11 di cartella' },
    { kind: 'box', key: 'rect-img', box: image.rect_per_image, cls: 'rect-img',
      label: 'rect di questa immagine' },
  ]);

  const cards = $('#sample-cards');
  cards.innerHTML = '';
  cards.appendChild(areaCard('vendor', folder.vendor, [
    ['valore', folder.vendor.value],
    ['confidenza', `${num(folder.vendor.confidence)} (soglia ${num(folder.vendor.threshold)})`],
    ['margine top1-top2', num(folder.vendor.margin)],
    ['come', folder.vendor.source],
    ['OCR', folder.vendor.ocr_used === 'True' ? (folder.vendor.ocr_text || 'usato') : 'non usato'],
  ], folder.vendor.top3));
  cards.appendChild(areaCard('probe', folder.probe, [
    ['ID sonda', folder.probe.value],
    ['nome', folder.probe.name],
    ['confidenza', `${num(folder.probe.confidence)} (soglia ${num(folder.probe.threshold)})`],
    ['#04 PROBETYPE', folder.probe.probe_type],
    ['come', folder.probe.probe_type_source],
  ], folder.probe.top3));
  cards.appendChild(areaCard('rect', folder.rect, [
    ['#11', folder.rect.line_11],
    ['metodo', folder.rect.method],
    ['routing', folder.rect.source],
    ['gruppo vincente', folder.rect.red_winner_group],
  ]));
  cards.appendChild(areaCard('rotazione', folder.rotazione, [
    ['gradi', `${folder.rotazione.deg}°`],
    ['come', folder.rotazione.source],
    ['consenso', pct(folder.rotazione.vote_ratio)],
  ]));
  cards.appendChild(areaCard('dedup', folder.dedup, [
    ['immagini nella cartella', folder.dedup.images_raw],
    ['duplicati rimossi', `${folder.dedup.removed} (${folder.dedup.removed_pct}%)`],
    ['immagini studiate', folder.dedup.images_unique],
  ]));
  cards.appendChild(areaCard('line13', folder.line13, [
    ['#13', folder.line13.value],
    ['come', folder.line13.source],
    ['supporto', num(folder.line13.support)],
  ]));
  cards.appendChild(areaCard('fss', folder.fss, (folder.fss.lines || [])
    .map((line) => [line.line, line.value])));
}

function areaCard(areaId, prediction, pairs, topk) {
  const area = (S.catalog.areas || []).find((entry) => entry.id === areaId) || { label: areaId };
  const existing = feedbackFor(areaId, null);
  const body = el('div', { class: 'acard-body' }, [kv(pairs)]);
  if (topk && topk.length) {
    const box = el('div', { class: 'topk' });
    topk.forEach((item) => {
      box.appendChild(el('div', { class: 'tk' }, [
        el('span', { text: `${item.label} ${num(item.prob)}` }),
        el('div', { class: 'bar' }, [el('i', { style: `width:${Math.max(2, (item.prob || 0) * 100)}%` })]),
      ]));
    });
    body.appendChild(box);
  }
  if (existing.length) {
    body.appendChild(el('div', { class: 'chips', style: 'margin-top:8px' },
      existing.map((entry) => el('span', {
        class: `chip ${entry.verdict === 'ok' ? 'ok' : 'warn'}`,
        title: entry.comment,
        text: entry.tag || entry.comment.slice(0, 28),
      }))));
  }
  return el('div', { class: 'acard' }, [
    el('div', { class: 'acard-head' }, [
      el('div', {}, [
        el('span', { class: 'acard-title', text: area.label }),
        el('span', { class: 'lines', text: (area.lines || []).join(' ') }),
      ]),
      el('button', {
        class: 'small',
        text: existing.length ? `💬 ${existing.length}` : '💬',
        onclick: () => openDrawer({ area: areaId, scope: 'folder', prediction: prediction }),
      }),
    ]),
    body,
  ]);
}

/* ---------------------------------------------------------------- grid */
function imageProblems(image) {
  return (image.checks || []).filter((c) => c.level === 'warn' || c.level === 'error');
}

function statusClass(status) {
  const value = String(status || '').toLowerCase();
  if (value === 'accepted' || value === 'ok') return 'ok';
  if (value === 'review') return 'warn';
  if (value === 'reject') return 'error';
  return '';
}

function renderGrid() {
  const filter = $('#grid-filter').value.trim().toLowerCase();
  const mode = $('#grid-status').value;
  S.gridSize = Number($('#grid-size').value || 220);
  const grid = $('#image-grid');
  grid.style.gridTemplateColumns = `repeat(auto-fill, minmax(${S.gridSize}px, 1fr))`;
  grid.innerHTML = '';

  const visible = (S.study.images || []).filter((image) => {
    if (filter && !image.image_id.toLowerCase().includes(filter)) return false;
    const depth = image.depth || {}, scala = image.scala || {};
    switch (mode) {
      case 'problem': return imageProblems(image).length > 0;
      case 'depth_review': return String(depth.status) === 'review';
      case 'depth_reject': return String(depth.status) === 'reject';
      case 'scala_review': return String(scala.status) === 'review';
      case 'scala_reject': return String(scala.status) === 'reject';
      case 'commented': return feedbackCountForImage(image.image_id) > 0;
      default: return true;
    }
  });
  $('#grid-count').textContent = `${visible.length} di ${(S.study.images || []).length}`;

  visible.forEach((image) => {
    const depth = image.depth || {}, scala = image.scala || {};
    const problems = imageProblems(image);
    const fbCount = feedbackCountForImage(image.image_id);
    const chips = el('div', { class: 'chips' }, [
      el('span', {
        class: `chip ${statusClass(depth.status)}`,
        title: `depth ${txt(depth.status)} · ${txt(depth.mode)} · ${txt(depth.reason)}`,
        text: `D ${depth.depth_mm !== null && depth.depth_mm !== undefined ? depth.depth_mm : '—'}`,
      }),
      el('span', {
        class: `chip ${statusClass(scala.status)}`,
        title: `scala ${txt(scala.status)} · mm/px ${txt(scala.mm_per_px)} · ${txt(scala.reason)}`,
        text: `S ${scala.mm_per_px ? Number(scala.mm_per_px).toFixed(3) : '—'}`,
      }),
      (image.su_giu || {}).label ? el('span', {
        class: 'chip', title: `su/giù conf ${num((image.su_giu || {}).confidence)}`,
        text: image.su_giu.label,
      }) : null,
      (image.lt || {}).label ? el('span', {
        class: 'chip', title: `L/T conf ${num((image.lt || {}).confidence)}`,
        text: image.lt.label,
      }) : null,
      (image.lr_marker || {}).label ? el('span', {
        class: `chip ${(image.lr_marker || {}).status === 'ok' ? '' : 'warn'}`,
        title: `marker ${txt((image.lr_marker || {}).status)} score ${num((image.lr_marker || {}).score)}`,
        text: image.lr_marker.label === 'lr_flipped' ? 'L/R ↔' : 'L/R =',
      }) : null,
    ]);

    const flags = [];
    if (problems.some((p) => p.level === 'error')) flags.push(['error', String(problems.length)]);
    else if (problems.length) flags.push(['warn', String(problems.length)]);
    if (fbCount) flags.push(['fb', `💬${fbCount}`]);

    grid.appendChild(el('div', {
      class: 'igc',
      onclick: () => openDetail(S.study.images.indexOf(image)),
    }, [
      el('div', { class: 'thumb' }, [
        el('img', { src: imageUrl(image.image_path, S.gridSize * 2), loading: 'lazy',
          alt: image.image_id }),
        ...flags.map(([cls, label]) => el('span', { class: `flag ${cls}`, text: label })),
      ]),
      el('div', { class: 'name', title: image.image_id, text: image.image_id }),
      chips,
    ]));
  });

  if (!visible.length) {
    grid.appendChild(el('div', { class: 'muted', text: 'nessuna immagine con questo filtro' }));
  }
}

/* ---------------------------------------------------------------- detail */
function openDetail(index) {
  const images = S.study.images || [];
  if (index < 0 || index >= images.length) return;
  S.detailIndex = index;
  $('#detail-modal').classList.remove('hidden');
  renderDetail();
}

function closeDetail() {
  $('#detail-modal').classList.add('hidden');
  S.detailIndex = -1;
}

function renderDetail() {
  const image = (S.study.images || [])[S.detailIndex];
  if (!image) return;
  const folder = S.study.folder;
  const depth = image.depth || {}, scala = image.scala || {};

  $('#detail-title').textContent =
    `${image.image_id} — ${S.detailIndex + 1}/${S.study.images.length}`;

  const overlays = [
    { kind: 'box', key: 'rect', box: folder.rect.box, cls: 'rect', label: '#11 cartella' },
    { kind: 'box', key: 'rect-img', box: image.rect_per_image, cls: 'rect-img', label: 'rect immagine' },
    { kind: 'box', key: 'depth', box: depth.box, cls: 'depth',
      label: `depth ${txt(depth.depth_mm)}` },
    { kind: 'box', key: 'marker', box: (image.lr_marker || {}).box, cls: 'marker', label: 'marker' },
  ];
  if (scala.x !== null && scala.x !== undefined) {
    overlays.push({ kind: 'segment', key: 'scala', x: Number(scala.x),
      y1: Number(scala.y_zero), y2: Number(scala.y_far) });
  }
  buildCanvas($('#detail-canvas'), image, overlays);

  const toggles = $('#detail-toggles');
  toggles.innerHTML = '';
  [['rect', '#11 cartella'], ['rect-img', 'rect immagine'], ['depth', 'box depth'],
    ['marker', 'marker L/R'], ['scala', 'righello']].forEach(([key, label]) => {
    toggles.appendChild(el('span', {
      class: `chip clickable ${S.overlayOff.has(key) ? '' : 'selected'}`,
      text: label,
      onclick: () => {
        if (S.overlayOff.has(key)) S.overlayOff.delete(key); else S.overlayOff.add(key);
        renderDetail();
      },
    }));
  });

  // Zoomed crops: how a depth label or a ruler is actually judged.
  const crops = $('#detail-crops');
  crops.innerHTML = '';
  if (depth.box && depth.box.left !== null && depth.box.right) {
    crops.appendChild(el('div', { class: 'crop' }, [
      el('div', { class: 'cl', text: `box depth — OCR: ${txt(depth.ocr_text)}` }),
      el('img', { src: cropUrl(image.image_path, depth.box, 4, 18), alt: 'zoom depth' }),
    ]));
  }
  if (scala.x !== null && scala.x !== undefined && scala.y_zero !== null && scala.y_far !== null) {
    const top = Math.min(Number(scala.y_zero), Number(scala.y_far));
    const bottom = Math.max(Number(scala.y_zero), Number(scala.y_far));
    crops.appendChild(el('div', { class: 'crop' }, [
      el('div', { class: 'cl', text: `righello — ${txt(scala.n_ticks)} tacche, ${txt(scala.n_labels)} numeri` }),
      el('img', {
        src: cropUrl(image.image_path,
          { left: Number(scala.x) - 46, top: top, right: Number(scala.x) + 46, bottom: bottom },
          1.8, 8),
        alt: 'zoom righello',
      }),
    ]));
  }

  const box = $('#detail-areas');
  box.innerHTML = '';

  const checks = image.checks || [];
  const checkBox = el('div', { class: 'checks' });
  if (!checks.length) {
    checkBox.appendChild(el('div', { class: 'chk ok' },
      [el('span', { text: 'nessun incrocio segnalato' })]));
  }
  checks.forEach((check) => {
    checkBox.appendChild(el('div', { class: `chk ${check.level}` }, [
      el('span', { class: 'areas', text: check.areas }),
      el('span', { text: check.message }),
    ]));
  });
  box.appendChild(el('div', { class: 'acard' }, [
    el('div', { class: 'acard-head' }, [
      el('span', { class: 'acard-title', text: 'Incroci fra aree su questa immagine' }),
      el('button', {
        class: 'small', text: '💬',
        onclick: () => openDrawer({ area: 'orchestratore', scope: 'image', image: image,
          prediction: { checks: checks } }),
      }),
    ]),
    checkBox,
  ]));

  box.appendChild(imageAreaCard('depth', image, [
    ['valore', depth.depth_mm !== null && depth.depth_mm !== undefined ? `${depth.depth_mm} mm` : '—'],
    ['esito', depth.status],
    ['modalità', depth.mode],
    ['punteggio', `${num(depth.score)} (ranker ${num(depth.ranker_score)})`],
    ['OCR', depth.ocr_text],
    ['perché', depth.reason],
    ['candidati', depth.candidates],
  ]));

  box.appendChild(imageAreaCard('scala', image, [
    ['esito', scala.status],
    ['mm per pixel', scala.mm_per_px],
    ['colonna x', scala.x],
    ['zero / estremo', `${txt(scala.y_zero)} → ${txt(scala.y_far)}`],
    ['depth di riferimento', scala.depth_mm],
    ['verso', `${txt(scala.direction)} (da ${txt(scala.sugiu_source)})`],
    ['tacche', `${txt(scala.n_ticks)} passo ${txt(scala.tick_pitch_px)} px = ${txt(scala.tick_step_mm)} mm`],
    ['numeri letti', scala.labels_json],
    ['calibrazione', scala.calib_source],
    ['fuori banda', scala.out_of_band],
    ['controprove', `zero ${txt(scala.zero_check)} · depth ${txt(scala.depth_check)}`],
    ['perché', scala.reason],
  ]));

  box.appendChild(imageAreaCard('su_giu', image, [
    ['verso', (image.su_giu || {}).label],
    ['confidenza', num((image.su_giu || {}).confidence)],
    ['prob su / giù', `${num((image.su_giu || {}).prob_su)} / ${num((image.su_giu || {}).prob_giu)}`],
    ['crop', (image.su_giu || {}).crop_source],
  ]));

  box.appendChild(imageAreaCard('lr_marker', image, [
    ['etichetta', (image.lr_marker || {}).label_it || (image.lr_marker || {}).label],
    ['esito', (image.lr_marker || {}).status],
    ['score', num((image.lr_marker || {}).score)],
    ['lato trovato', (image.lr_marker || {}).detected_side],
    ['strategia', (image.lr_marker || {}).search_strategy],
    ['gruppo', (image.lr_marker || {}).orientation_group],
    ['template', (image.lr_marker || {}).template],
    ['motivo review', (image.lr_marker || {}).review_reason],
  ]));

  box.appendChild(imageAreaCard('lt', image, [
    ['piano', (image.lt || {}).label],
    ['confidenza', num((image.lt || {}).confidence)],
    ['prob L / T', `${num((image.lt || {}).prob_l)} / ${num((image.lt || {}).prob_t)}`],
  ]));

  const rectVs = image.rect_vs_folder || {};
  box.appendChild(imageAreaCard('rect', image, [
    ['rect immagine', image.rect_per_image
      ? `${image.rect_per_image.top}|${image.rect_per_image.left}|${image.rect_per_image.bottom}|${image.rect_per_image.right}`
      : '—'],
    ['#11 di cartella', folder.rect.line_11],
    ['IoU con #11', num(rectVs.iou)],
    ['distanza centri', num(rectVs.center_distance_norm, 3)],
  ]));
}

function imageAreaCard(areaId, image, pairs) {
  const area = (S.catalog.areas || []).find((entry) => entry.id === areaId) || { label: areaId };
  const existing = feedbackFor(areaId, image.image_id);
  const body = el('div', { class: 'acard-body' }, [kv(pairs)]);
  if (existing.length) {
    body.appendChild(el('div', { class: 'chips', style: 'margin-top:8px' },
      existing.map((entry) => el('span', {
        class: `chip ${entry.verdict === 'ok' ? 'ok' : 'warn'}`,
        title: entry.comment,
        text: entry.tag || entry.comment.slice(0, 26),
      }))));
  }
  return el('div', { class: 'acard' }, [
    el('div', { class: 'acard-head' }, [
      el('div', {}, [
        el('span', { class: 'acard-title', text: area.label }),
        el('span', { class: 'lines', text: (area.lines || []).join(' ') }),
      ]),
      el('div', { class: 'row' }, [
        el('button', {
          class: 'small', text: existing.length ? `💬 ${existing.length}` : '💬',
          onclick: () => openDrawer({ area: areaId, scope: 'image', image: image }),
        }),
        el('button', {
          class: 'small', text: 'tutta la cartella',
          title: 'lo stesso commento vale per tutte le immagini',
          onclick: () => openDrawer({ area: areaId, scope: 'folder', image: image }),
        }),
      ]),
    ]),
    body,
  ]);
}

/* ================================================================ DRAWER */
function openDrawer(config) {
  const area = (S.catalog.areas || []).find((entry) => entry.id === config.area);
  if (!area) { toast(`area sconosciuta: ${config.area}`, true); return; }
  S.drawer = {
    area: area,
    scope: config.scope || 'folder',
    image: config.image || null,
    prediction: config.prediction || null,
    tag: '',
    correction: {},
  };
  $('#fb-drawer').classList.remove('hidden');
  $('#fbd-title').textContent = `${area.label} — ${config.scope === 'image' ? 'questa immagine' : 'tutta la cartella'}`;
  $('#fbd-target').textContent = [
    S.studyRef ? S.studyRef.folderName : '',
    config.image ? config.image.image_id : '',
    (area.lines || []).join(' '),
  ].filter(Boolean).join(' · ');
  $('#fbd-comment').value = config.prefill || '';
  $('#fbd-error').textContent = '';

  const prediction = config.prediction
    || (config.scope === 'image' && config.image
      ? (config.image[config.area === 'rect' ? 'rect_per_image' : config.area] || {})
      : ((S.study && S.study.folder[config.area]) || {}));
  $('#fbd-prediction').textContent = JSON.stringify(prediction, null, 1);

  fillSelect($('#fbd-verdict'), S.catalog.verdicts, '');
  fillSelect($('#fbd-severity'), S.catalog.severities, 'major');
  const scopes = (area.scopes || ['folder']).map((scope) => scope);
  fillSelect($('#fbd-scope'), scopes, S.drawer.scope);
  $('#fbd-scope').onchange = () => { S.drawer.scope = $('#fbd-scope').value; };

  renderChips();
  renderCorrection();
}

function fillSelect(node, values, selected) {
  node.innerHTML = '';
  values.forEach((value) => {
    const option = el('option', { value: value, text: value });
    if (value === selected) option.selected = true;
    node.appendChild(option);
  });
}

function renderChips() {
  const box = $('#fbd-chips');
  box.innerHTML = '';
  (S.drawer.area.suggestions || []).forEach((suggestion) => {
    const node = el('button', {
      class: 'sugg' + (suggestion.verdict === 'ok' ? ' ok' : '')
        + (S.drawer.tag === suggestion.tag ? ' selected' : ''),
      onclick: () => {
        S.drawer.tag = S.drawer.tag === suggestion.tag ? '' : suggestion.tag;
        if (S.drawer.tag) {
          $('#fbd-verdict').value = suggestion.verdict || '';
          $('#fbd-severity').value = suggestion.severity || 'major';
          if (!$('#fbd-comment').value.trim()) $('#fbd-comment').value = suggestion.text;
        }
        renderChips();
        renderCorrection();
      },
    }, [
      el('span', { text: suggestion.text }),
      el('span', { class: 'tag', text: suggestion.tag
        + (suggestion.needs_correction ? ' · chiede la correzione' : '') }),
      suggestion.hint ? el('span', { class: 'hint', text: suggestion.hint }) : null,
    ]);
    box.appendChild(node);
  });
}

function renderCorrection() {
  const block = $('#fbd-correction-block');
  const box = $('#fbd-correction');
  box.innerHTML = '';
  const fields = S.drawer.area.correction || [];
  if (!fields.length) { block.classList.add('hidden'); return; }
  block.classList.remove('hidden');

  const suggestion = (S.drawer.area.suggestions || [])
    .find((entry) => entry.tag === S.drawer.tag);
  if (suggestion && suggestion.needs_correction) {
    box.appendChild(el('div', { class: 'corr-hint',
      text: 'Questo suggerimento chiede il valore giusto: senza di esso il commento vale metà.' }));
  }

  fields.forEach((field) => {
    const row = el('div', { class: 'corr-row' });
    row.appendChild(el('span', { class: 'cl', text: field.label + (field.optional ? '' : ' *') }));
    const current = S.drawer.correction[field.field];

    if (field.kind === 'select') {
      const options = field.options
        || (S.catalog.options_sources || {})[field.options_source] || [];
      const select = el('select', {
        onchange: (event) => { S.drawer.correction[field.field] = event.target.value; },
      }, [el('option', { value: '', text: '— scegli —' }),
        ...options.map((value) => el('option', { value: value, text: value }))]);
      if (current) select.value = current;
      row.appendChild(select);
      if (field.allow_custom) {
        row.appendChild(el('input', {
          placeholder: 'oppure scrivi…',
          oninput: (event) => {
            if (event.target.value.trim()) S.drawer.correction[field.field] = event.target.value.trim();
          },
        }));
      }
    } else if (field.kind === 'number') {
      row.appendChild(el('input', {
        type: 'number', step: field.step || 'any', value: current === undefined ? '' : current,
        oninput: (event) => {
          const value = event.target.value.trim();
          if (value === '') delete S.drawer.correction[field.field];
          else S.drawer.correction[field.field] = Number(value);
        },
      }));
    } else if (field.kind === 'text') {
      row.appendChild(el('input', {
        value: current === undefined ? '' : current,
        oninput: (event) => { S.drawer.correction[field.field] = event.target.value; },
      }));
    } else if (field.kind === 'box' || field.kind === 'point_x' || field.kind === 'point_y') {
      const label = el('span', { class: 'corr-value',
        text: current !== undefined ? JSON.stringify(current) : 'non impostato' });
      row.appendChild(el('button', {
        class: 'small',
        text: field.kind === 'box' ? 'disegna sull\'immagine' : 'clicca sull\'immagine',
        onclick: () => startPick(field, label),
      }));
      row.appendChild(label);
    }
    box.appendChild(row);
    if (field.hint) box.appendChild(el('div', { class: 'corr-hint', text: field.hint }));
  });
}

/** Pick a box or a point on the image the drawer is about: no coordinates typed by hand. */
function startPick(field, label) {
  const image = S.drawer.image
    || (S.study.images || [])[S.detailIndex >= 0 ? S.detailIndex : S.sampleIndex];
  if (!image) { toast('serve un\'immagine per disegnare', true); return; }

  const overlay = el('div', { class: 'modal' });
  const wrap = el('div', { class: 'canvas-wrap' });
  const card = el('div', { class: 'modal-card wide' }, [
    el('div', { class: 'row spread' }, [
      el('h3', { class: 'tight', text: field.kind === 'box'
        ? `${field.label} — trascina il rettangolo` : `${field.label} — clicca il punto` }),
      el('button', { class: 'ghost small', text: 'chiudi ✕', onclick: () => overlay.remove() }),
    ]),
    wrap,
    el('div', { class: 'muted small',
      text: 'Le coordinate salvate sono in pixel dell\'immagine originale.' }),
  ]);
  overlay.appendChild(card);
  document.body.appendChild(overlay);

  const done = (value) => {
    S.drawer.correction[field.field] = value;
    label.textContent = JSON.stringify(value);
    overlay.remove();
  };

  const existing = [];
  const depth = image.depth || {}, scala = image.scala || {};
  if (depth.box) existing.push({ kind: 'box', key: 'depth', box: depth.box, cls: 'depth', label: 'depth attuale' });
  if (image.rect_per_image) {
    existing.push({ kind: 'box', key: 'rect-img', box: image.rect_per_image, cls: 'rect-img',
      label: 'rect attuale' });
  }
  if (scala.x !== null && scala.x !== undefined) {
    existing.push({ kind: 'segment', key: 'scala', x: Number(scala.x), y1: Number(scala.y_zero),
      y2: Number(scala.y_far) });
  }

  buildCanvas(wrap, image, existing, field.kind === 'box'
    ? { mode: 'box', onBox: done }
    : { mode: 'point', onPoint: (point) => done(field.kind === 'point_x' ? point.x : point.y) });
}

async function saveFeedback() {
  if (!S.drawer) return;
  const comment = $('#fbd-comment').value.trim();
  const payload = {
    area: S.drawer.area.id,
    scope: S.drawer.scope,
    tag: S.drawer.tag,
    comment: comment,
    verdict: $('#fbd-verdict').value,
    severity: $('#fbd-severity').value,
    correction: S.drawer.correction,
    target: {
      run_id: S.studyRef ? S.studyRef.runId : null,
      slug: S.studyRef ? S.studyRef.slug : null,
      folder: S.studyRef ? S.studyRef.folderName : null,
      folder_path: S.study ? S.study.folder.folder_path : null,
      image_id: S.drawer.scope === 'image' && S.drawer.image ? S.drawer.image.image_id : null,
      image_path: S.drawer.scope === 'image' && S.drawer.image ? S.drawer.image.image_path : null,
    },
    prediction: S.drawer.prediction || undefined,
  };
  try {
    const data = await api('/api/feedback', { method: 'POST', body: JSON.stringify(payload) });
    if (S.study) S.study.feedback = (S.study.feedback || []).concat([data.entry]);
    $('#fb-drawer').classList.add('hidden');
    S.drawer = null;
    toast('feedback salvato');
    refreshBadge();
    if (S.study) { renderReview(); if (S.detailIndex >= 0) renderDetail(); }
  } catch (err) {
    $('#fbd-error').textContent = err.message;
  }
}

/* ================================================================ FEEDBACK VIEW */
async function loadFeedback() {
  const params = new URLSearchParams();
  const status = $('#fb-filter-status').value;
  const area = $('#fb-filter-area').value;
  if (status) params.set('status', status);
  if (area) params.set('area', area);
  try {
    const data = await api(`/api/feedback?${params.toString()}`);
    S.feedback = data;
    renderFeedback();
  } catch (err) { toast(err.message, true); }
}

function renderFeedback() {
  const stats = S.feedback.stats || { by_area: {}, by_tag: {} };
  const statsBox = $('#fb-stats');
  statsBox.innerHTML = '';
  statsBox.appendChild(el('div', { class: 'fb-stat' }, [
    el('b', { text: String(stats.total || 0) }), el('span', { text: 'totali' }),
  ]));
  statsBox.appendChild(el('div', { class: 'fb-stat' }, [
    el('b', { text: String(stats.open || 0) }), el('span', { text: 'aperti' }),
  ]));
  Object.entries(stats.by_area || {}).forEach(([area, counts]) => {
    statsBox.appendChild(el('div', { class: 'fb-stat' }, [
      el('b', { text: String(counts.total) }),
      el('span', { text: `${area} · ${counts.wrong} sbagliati, ${counts.ok} ok` }),
    ]));
  });

  const list = $('#fb-list');
  list.innerHTML = '';
  const entries = (S.feedback.entries || []).slice().reverse();
  if (!entries.length) list.appendChild(el('div', { class: 'muted', text: 'nessun feedback' }));
  entries.forEach((entry) => {
    const target = entry.target || {};
    const context = entry.context || {};
    list.appendChild(el('div', { class: `fb-item ${entry.severity}` }, [
      el('div', { class: 'fh' }, [
        el('div', {}, [
          el('strong', { text: `${entry.area}${entry.tag ? ' · ' + entry.tag : ''}` }),
          el('div', { class: 'ft', text: `${entry.id} · ${entry.ts} · ${entry.kind} · ${entry.verdict || '—'} · ${entry.status}` }),
        ]),
        el('div', { class: 'row' }, [
          entry.status === 'open' ? el('button', {
            class: 'ghost small', text: 'presa in carico',
            onclick: () => resolveFeedback(entry.id, 'ack'),
          }) : null,
          el('button', {
            class: 'ghost small danger', text: 'elimina',
            onclick: () => deleteFeedback(entry.id),
          }),
        ]),
      ]),
      el('div', { class: 'fc', text: entry.comment || '(nessun commento)' }),
      el('div', { class: 'fx', text: [
        target.folder ? `cartella ${target.folder}` : null,
        target.image_id ? `immagine ${target.image_id}` : null,
        context.vendor ? `vendor ${context.vendor}` : null,
        Object.keys(entry.correction || {}).length
          ? `correzione ${JSON.stringify(entry.correction)}` : null,
      ].filter(Boolean).join(' · ') }),
    ]));
  });
}

async function resolveFeedback(entryId, status) {
  try {
    await api(`/api/feedback/${entryId}/resolve`,
      { method: 'POST', body: JSON.stringify({ status: status, by: 'camilla' }) });
    loadFeedback();
    refreshBadge();
  } catch (err) { toast(err.message, true); }
}

async function deleteFeedback(entryId) {
  try {
    await api(`/api/feedback/${entryId}`, { method: 'DELETE' });
    loadFeedback();
    refreshBadge();
  } catch (err) { toast(err.message, true); }
}

async function refreshBadge() {
  try {
    const data = await api('/api/feedback?status=open');
    $('#badge-feedback').textContent = String((data.entries || []).length);
  } catch (err) { /* not critical */ }
}

/* ================================================================ ORCHESTRATOR */
async function loadOrchestrator(refresh) {
  const body = $('#orch-body');
  body.textContent = 'calcolo in corso…';
  try {
    const data = await api(`/api/orchestrator${refresh ? '?refresh=1' : ''}`);
    renderOrchestrator(data);
  } catch (err) { body.textContent = err.message; }
}

function renderOrchestrator(data) {
  const body = $('#orch-body');
  body.innerHTML = '';
  body.appendChild(el('div', { class: 'muted small',
    text: `${data.folders_seen} cartelle lette, ${data.feedback_seen} feedback, generato ${data.generated}` }));

  const suggestions = data.suggestions || [];
  body.appendChild(el('h3', { text: `Cosa suggerisce l'incrocio (${suggestions.length})` }));
  const sBox = el('div', { class: 'osugg' });
  if (!suggestions.length) {
    sBox.appendChild(el('div', { class: 'muted',
      text: 'ancora poca evidenza: servono più run e più feedback.' }));
  }
  suggestions.forEach((item) => {
    sBox.appendChild(el('div', { class: 'os' }, [
      el('div', { text: item.text }),
      el('div', { class: 'om', text: `${item.kind} · ${item.vendor} · ${item.area} · supporto ${item.support}` }),
    ]));
  });
  body.appendChild(sBox);

  const cross = data.cross_vendor || {};
  body.appendChild(el('h3', { text: 'Problema di modulo o di vendor' }));
  const crossBox = el('div', { class: 'ovendor' });
  const rows = [
    ['tag su più vendor (modulo)', cross.module_wide_tags],
    ['tag di un solo vendor (profilo)', cross.vendor_specific_tags],
    ['incroci su più vendor (modulo)', cross.module_wide_checks],
    ['incroci di un solo vendor', cross.vendor_specific_checks],
  ];
  rows.forEach(([label, mapping]) => {
    const entries = Object.entries(mapping || {});
    crossBox.appendChild(el('div', { class: 'small' }, [
      el('strong', { text: `${label}: ` }),
      el('span', { class: 'muted', text: entries.length
        ? entries.map(([key, vendors]) => `${key} (${vendors.join(', ')})`).join(' · ')
        : 'niente' }),
    ]));
  });
  body.appendChild(crossBox);

  body.appendChild(el('h3', { text: 'Storico per vendor' }));
  Object.entries(data.vendors || {}).forEach(([name, info]) => {
    const table = el('table', { class: 'mini' });
    const add = (key, value) => {
      table.appendChild(el('tr', {}, [
        el('th', { text: key }),
        el('td', { text: typeof value === 'string' ? value : JSON.stringify(value) }),
      ]));
    };
    add('cartelle viste', info.folders_seen);
    add('sonde', info.probe_ids);
    add('rect: metodi', (info.rect || {}).methods);
    add('rect: prior', (info.rect || {}).priors_by_geometry);
    add('depth: modalità', (info.depth || {}).modes);
    add('depth: valori', (info.depth || {}).values_seen);
    add('scala: righello x', (info.scala || {}).ruler_x_by_geometry);
    add('scala: mm/px', (info.scala || {}).mm_per_px_by_geometry);
    add('scala: verso', (info.scala || {}).directions);
    add('orientamento', info.orientation);
    add('incroci falliti', (info.coherence || {}).checks);
    add('coppie di aree', (info.coherence || {}).area_pairs);
    add('umano: per area', (info.human || {}).by_area);
    add('umano: per tag', (info.human || {}).by_tag);
    body.appendChild(el('div', { class: 'ovendor' }, [
      el('h4', { text: name }),
      table,
    ]));
  });
}

/* ================================================================ boot */
async function boot() {
  S.catalog = await api('/api/catalog');
  const areaSelect = $('#fb-filter-area');
  (S.catalog.areas || []).forEach((area) => {
    areaSelect.appendChild(el('option', { value: area.id, text: area.label }));
  });

  $$('.tab').forEach((tab) => tab.addEventListener('click', () => setView(tab.dataset.view)));
  $('#btn-load-folders').addEventListener('click', loadFolders);
  $('#root-input').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') loadFolders();
  });
  $('#folder-filter').addEventListener('input', renderFolders);
  $$('[data-select]').forEach((button) => button.addEventListener('click',
    () => selectFolders(button.dataset.select)));
  $('#btn-start').addEventListener('click', startRun);
  $('#btn-refresh-run').addEventListener('click', pollRun);
  $('#btn-cancel-run').addEventListener('click', async () => {
    if (!S.run) return;
    await api(`/api/runs/${S.run.run_id}/cancel`, { method: 'POST' });
    toast('annullamento richiesto');
  });
  $('#btn-sample-prev').addEventListener('click', () => { S.sampleIndex -= 1; renderSample(); });
  $('#btn-sample-next').addEventListener('click', () => { S.sampleIndex += 1; renderSample(); });
  $('#grid-filter').addEventListener('input', renderGrid);
  $('#grid-status').addEventListener('change', renderGrid);
  $('#grid-size').addEventListener('change', renderGrid);
  $('#detail-close').addEventListener('click', closeDetail);
  $('#detail-prev').addEventListener('click', () => openDetail(S.detailIndex - 1));
  $('#detail-next').addEventListener('click', () => openDetail(S.detailIndex + 1));
  $('#fbd-close').addEventListener('click', () => {
    $('#fb-drawer').classList.add('hidden');
    S.drawer = null;
  });
  $('#fbd-save').addEventListener('click', saveFeedback);
  $('#fb-refresh').addEventListener('click', loadFeedback);
  $('#fb-filter-status').addEventListener('change', loadFeedback);
  $('#fb-filter-area').addEventListener('change', loadFeedback);
  $('#orch-refresh').addEventListener('click', () => loadOrchestrator(true));
  $('#orch-write').addEventListener('click', async () => {
    try {
      const data = await api('/api/orchestrator/write', { method: 'POST' });
      toast(`salvato in ${data.path}`);
    } catch (err) { toast(err.message, true); }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      if (!$('#detail-modal').classList.contains('hidden')) closeDetail();
      else if (!$('#fb-drawer').classList.contains('hidden')) $('#fbd-close').click();
    }
    if (S.detailIndex >= 0 && !$('#detail-modal').classList.contains('hidden')) {
      if (event.key === 'ArrowRight') openDetail(S.detailIndex + 1);
      if (event.key === 'ArrowLeft') openDetail(S.detailIndex - 1);
    }
  });

  setView('home');
  await loadRoots();
  await loadHistory();
  refreshBadge();
}

boot().catch((err) => {
  document.body.insertBefore(
    el('div', { class: 'card error', text: `avvio fallito: ${err.message}` }),
    document.body.firstChild);
});
