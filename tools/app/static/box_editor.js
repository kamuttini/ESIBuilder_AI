/* Editor dei box sull'immagine: trascinamento, maniglie, slider, schermo intero.
   I box sono di cartella, non di immagine: qualunque modifica vale per tutte le immagini. */

const BOX_SPECS = [
  // The proposal as it came out of the network: dashed, never editable, so the starting point
  // stays visible even after the box has been moved by hand.
  { key: 'rect_echo_original', label: 'originario (proposta rete)', color: '#9aa7b8',
    readonly: true },
  // The margin is *added* around the echo rectangle: at 0 the two coincide, above 0 what gets
  // saved in #11 is a slightly larger box. Computed, never dragged.
  { key: 'rect_echo_saved', label: 'con margine: e\' questo che viene salvato in #11',
    color: '#3fb950', readonly: true },
  { key: 'rect_echo', label: 'rettangolo ecografico', color: '#ff6040', margins: true },
  { key: 'rect_name_echo', label: 'template ecografo #13', color: '#40d0ff' },
  { key: 'rect_name_probe', label: 'template sonda #14', color: '#d29922' },
];

/* Il margine e' una percentuale del lato del rettangolo, aggiunta su entrambi i lati. */
function expandBox(box, margins, maxWidth, maxHeight) {
  if (!box) return null;
  const dx = Math.round(((margins?.x || 0) / 100) * (box.right - box.left));
  const dy = Math.round(((margins?.y || 0) / 100) * (box.bottom - box.top));
  return {
    left: Math.max(0, box.left - dx),
    right: Math.min(maxWidth - 1, box.right + dx),
    top: Math.max(0, box.top - dy),
    bottom: Math.min(maxHeight - 1, box.bottom + dy),
    clampedX: box.left - dx < 0 || box.right + dx > maxWidth - 1,
    clampedY: box.top - dy < 0 || box.bottom + dy > maxHeight - 1,
  };
}

const SIDES = ['top', 'left', 'bottom', 'right'];

function cloneBoxes(source) {
  const out = {};
  for (const spec of BOX_SPECS) {
    const box = source[spec.key];
    if (box && SIDES.every((side) => Number.isFinite(Number(box[side])))) {
      out[spec.key] = { top: +box.top, left: +box.left, bottom: +box.bottom, right: +box.right };
    }
  }
  return out;
}

/* Un editor su una singola immagine. `boxes` viene mutato in posto; `onChange` avvisa. */
function createBoxEditor({ imageSrc, boxes, sampleSize, onChange, height, onDoubleClick,
                           margins }) {
  const marginState = margins || { x: 0, y: 0 };
  const root = el('div', { class: 'editor' });
  const stage = el('div', { class: 'editor-stage' });
  const image = el('img', { src: imageSrc, alt: '' });
  if (height) image.style.maxHeight = height;
  stage.append(image);
  root.append(stage);

  // la selezione salta i box in sola lettura: non si modificano
  let selected = BOX_SPECS.find((spec) => boxes[spec.key] && !spec.readonly)?.key
    || BOX_SPECS[0].key;
  const overlays = {};
  const sliderRows = {};

  const originalWidth = () => (sampleSize && sampleSize[0]) || image.naturalWidth || 1;
  const originalHeight = () => (sampleSize && sampleSize[1]) || image.naturalHeight || 1;
  const scaleX = () => image.clientWidth / originalWidth();
  const scaleY = () => image.clientHeight / originalHeight();

  function clampBox(box) {
    const maxX = originalWidth() - 1;
    const maxY = originalHeight() - 1;
    box.left = Math.max(0, Math.min(Math.round(box.left), maxX - 1));
    box.top = Math.max(0, Math.min(Math.round(box.top), maxY - 1));
    box.right = Math.max(box.left + 2, Math.min(Math.round(box.right), maxX));
    box.bottom = Math.max(box.top + 2, Math.min(Math.round(box.bottom), maxY));
    return box;
  }

  function paint() {
    if (boxes.rect_echo) {
      const expanded = expandBox(boxes.rect_echo, marginState, originalWidth(), originalHeight());
      boxes.rect_echo_saved = (marginState.x || marginState.y) ? expanded : null;
    }
    for (const spec of BOX_SPECS) {
      const box = boxes[spec.key];
      const node = overlays[spec.key];
      if (!node) continue;
      if (!box) { node.style.display = 'none'; continue; }
      node.style.display = 'block';
      node.style.left = `${box.left * scaleX()}px`;
      node.style.top = `${box.top * scaleY()}px`;
      node.style.width = `${(box.right - box.left) * scaleX()}px`;
      node.style.height = `${(box.bottom - box.top) * scaleY()}px`;
      node.classList.toggle('selected', spec.key === selected);
      const readout = sliderRows[spec.key];
      if (readout) {
        readout.size.textContent =
          `${box.right - box.left} x ${box.bottom - box.top} px` +
          `  ·  top ${box.top} left ${box.left} bottom ${box.bottom} right ${box.right}`;
        for (const side of SIDES) {
          readout.inputs[side].value = box[side];
          readout.numbers[side].value = box[side];
        }
        if (readout.margins) {
          readout.margins.x.range.value = marginState.x;
          readout.margins.x.number.value = marginState.x;
          readout.margins.y.range.value = marginState.y;
          readout.margins.y.number.value = marginState.y;
          const dx = Math.round((marginState.x / 100) * (box.right - box.left));
          const dy = Math.round((marginState.y / 100) * (box.bottom - box.top));
          const saved = expandBox(box, marginState, originalWidth(), originalHeight());
          const parts = [];
          if (!marginState.x && !marginState.y) {
            parts.push('margini a zero: viene salvato esattamente il rettangolo ecografico');
          } else {
            parts.push(`margine aggiunto: ${dx} px per lato in orizzontale, ${dy} px in verticale`);
            parts.push(`salvato ${saved.right - saved.left} x ${saved.bottom - saved.top} px ` +
                       `invece di ${box.right - box.left} x ${box.bottom - box.top}`);
            if (saved.clampedX || saved.clampedY) {
              parts.push('il margine tocca il bordo dell\'immagine ed e\' stato tagliato');
            }
          }
          readout.margins.note.textContent = parts.join(' · ');
        }
      }
    }
  }

  function drag(spec, mode, event) {
    event.preventDefault();
    event.stopPropagation();
    selected = spec.key;
    const box = boxes[spec.key];
    if (!box) return;
    const start = { x: event.clientX, y: event.clientY, ...box };
    const move = (moveEvent) => {
      const dx = (moveEvent.clientX - start.x) / scaleX();
      const dy = (moveEvent.clientY - start.y) / scaleY();
      if (mode === 'move') {
        const width = start.right - start.left;
        const height = start.bottom - start.top;
        box.left = start.left + dx;
        box.top = start.top + dy;
        box.right = box.left + width;
        box.bottom = box.top + height;
      } else {
        if (mode.includes('n')) box.top = start.top + dy;
        if (mode.includes('s')) box.bottom = start.bottom + dy;
        if (mode.includes('w')) box.left = start.left + dx;
        if (mode.includes('e')) box.right = start.right + dx;
      }
      clampBox(box);
      paint();
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      if (onChange) onChange(boxes);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  }

  for (const spec of BOX_SPECS) {
    const node = el('div', {
      class: 'editor-box' + (spec.readonly ? ' readonly' : ''),
      style: `--box-color:${spec.color}`,
    });
    if (!spec.readonly) {
      node.addEventListener('pointerdown', (event) => drag(spec, 'move', event));
      for (const corner of ['nw', 'ne', 'sw', 'se', 'n', 's', 'w', 'e']) {
        const handle = el('div', { class: `handle handle-${corner}` });
        handle.addEventListener('pointerdown', (event) => drag(spec, corner, event));
        node.append(handle);
      }
    }
    overlays[spec.key] = node;
    stage.append(node);
  }

  /* Slider e campi numerici per il controllo fine */
  for (const spec of BOX_SPECS) {
    if (!boxes[spec.key] || spec.readonly) continue;
    const block = el('div', { class: 'editor-controls' });
    block.append(el('div', { class: 'editor-legend' },
      el('span', { class: 'swatch', style: `background:${spec.color}` }), spec.label));
    const size = el('div', { class: 'hint' });
    const inputs = {};
    const numbers = {};
    for (const side of SIDES) {
      const max = ['top', 'bottom'].includes(side) ? originalHeight() : originalWidth();
      const slider = el('input', { type: 'range', min: '0', max: String(max), step: '1' });
      const number = el('input', { type: 'number', style: 'width:84px' });
      const set = (raw) => {
        const box = boxes[spec.key];
        box[side] = parseInt(raw || '0', 10) || 0;
        clampBox(box);
        paint();
        if (onChange) onChange(boxes);
      };
      slider.addEventListener('input', () => set(slider.value));
      number.addEventListener('change', () => set(number.value));
      inputs[side] = slider;
      numbers[side] = number;
      block.append(el('div', { class: 'slider-row' },
        el('label', {}, side), slider, number));
    }
    block.append(size);
    sliderRows[spec.key] = { inputs, numbers, size };

    if (spec.margins) {
      // Two extra sliders: one margin for left+right, one for top+bottom. Moving them makes
      // the box symmetric about the image centre, which is what ESI draws.
      const note = el('div', { class: 'hint' });
      const margins = { note };
      for (const axis of ['x', 'y']) {
        const label = axis === 'x' ? 'margine sx/dx %' : 'margine alto/basso %';
        // Percentages of the rectangle's own side, added on both sides: at 0 nothing changes.
        const range = el('input', { type: 'range', min: '0', max: '25', step: '0.1' });
        const number = el('input', { type: 'number', min: '0', max: '25', step: '0.1',
                                     style: 'width:84px' });
        const set = (raw) => {
          marginState[axis] = Math.max(0, Math.min(25, parseFloat(raw) || 0));
          paint();
          if (onChange) onChange(boxes, marginState);
        };
        range.addEventListener('input', () => set(range.value));
        number.addEventListener('change', () => set(number.value));
        margins[axis] = { range, number };
        block.append(el('div', { class: 'slider-row margin-row' },
          el('label', {}, label), range, number, el('span', { class: 'hint' }, '%')));
      }
      block.append(note);
      sliderRows[spec.key].margins = margins;
    }

    root.append(block);
  }

  if (onDoubleClick) {
    stage.style.cursor = 'zoom-in';
    stage.addEventListener('dblclick', (event) => { event.preventDefault(); onDoubleClick(); });
    stage.title = 'doppio clic per aprire a schermo intero';
  }
  image.addEventListener('load', paint);
  window.addEventListener('resize', paint);
  if (image.complete) setTimeout(paint, 0);

  return {
    root,
    paint,
    setImage(src) { image.src = src; },
    boxes,
  };
}

/* Schermo intero: stessa immagine ingrandita, si scorre la cartella e si salva per tutte. */
async function openFullscreenEditor({ projectId, startValue, sampleSize, startImage, onSave,
                                      extraBoxes, margins }) {
  const data = await api(`/projects/${projectId}/images`);
  const names = data.names || [];
  let index = Math.max(0, names.indexOf(startImage));
  const boxes = { ...cloneBoxes(startValue), ...(extraBoxes || {}) };

  const overlay = el('div', { class: 'fullscreen' });
  const counter = el('span', { class: 'hint' });
  const url = () => `/api/projects/${projectId}/image?name=${encodeURIComponent(names[index] || startImage)}&w=1600`;

  const marginState = { x: 0, y: 0, ...(margins || {}) };
  const editor = createBoxEditor({
    imageSrc: url(), boxes, sampleSize, margins: marginState,
    height: 'calc(100vh - 320px)',
    onChange: () => { /* i box sono di cartella: si salvano alla chiusura */ },
  });

  const show = () => {
    counter.textContent = names.length
      ? `immagine ${index + 1} di ${names.length} — ${names[index]}`
      : startImage;
    editor.setImage(url());
  };
  const step = (delta) => {
    if (!names.length) return;
    index = (index + delta + names.length) % names.length;
    show();
  };

  const close = () => {
    window.removeEventListener('keydown', onKey);
    overlay.remove();
  };
  const onKey = (event) => {
    if (event.key === 'ArrowRight') step(1);
    else if (event.key === 'ArrowLeft') step(-1);
    else if (event.key === 'Escape') close();
  };
  window.addEventListener('keydown', onKey);

  overlay.append(el('div', { class: 'fullscreen-bar' },
    el('button', { class: 'ghost', onclick: () => step(-1) }, '‹ precedente'),
    el('button', { class: 'ghost', onclick: () => step(1) }, 'successiva ›'),
    counter,
    el('span', { class: 'hint' }, '· frecce per scorrere, Esc per chiudere'),
    el('div', { style: 'margin-left:auto;display:flex;gap:8px' },
      el('button', {
        onclick: () => {
          onSave(cloneBoxes(boxes), marginState);
          close();
        },
      }, 'Salva per tutte le immagini'),
      el('button', { class: 'ghost', onclick: close }, 'Annulla')),
  ));
  overlay.append(editor.root);
  document.body.append(overlay);
  show();
}
