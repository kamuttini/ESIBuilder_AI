/* La depth immagine per immagine: il valore, e soprattutto **da dove** e' stato preso.
   Il modulo lo legge in due modi — dall'etichetta scritta a schermo o dedotto dalla scala
   ecografica — e la differenza conta: sulla scala basta una tacca letta male e il numero
   cambia. Qui il riquadro e' colorato secondo il metodo, e accanto ci sono i punteggi di
   entrambe le strade, per vedere perche' ha scelto quella. */

async function createDepthViewer(projectId, sampleSize) {
  const data = await api(`/projects/${projectId}/depth`);
  const rows = data.rows || [];
  const modes = data.modes || {};
  const size = sampleSize || [0, 0];

  const root = el('div', {});
  if (!rows.length) {
    root.append(el('p', { class: 'hint' }, 'nessuna depth calcolata su questa cartella'));
    return root;
  }

  const byName = new Map(rows.map((r) => [r.name, r]));
  let names = rows.map((r) => r.name);
  let index = 0;
  let filtro = '';
  /* Il riquadro si stringe solo dove la depth e' scritta a schermo: la label sta sempre
     nello stesso posto e cambia solo di larghezza, quindi un riquadro stretto sul numero
     vale per tutta la cartella. Sulla scala il numero cambia posto ad ogni immagine. */
  let modiBox = data.box_modes || ['direct_label'];
  let modello = data.box_template || null;
  let conteggi = { by_mode: data.by_mode || {}, by_status: data.by_status || {} };
  let bozza = null;
  /* Il riquadro si puo' stringere dove la depth sta nell'interfaccia. Un'immagine che il
     modulo non ha esaminato non ha un metodo, ma la cartella si': se li' la depth e' una
     label, la label c'e' anche in questa, e il riquadro ci si puo' mettere. */
  const propagabile = () => !!(data.coverage || {}).can_propagate;
  const modificabile = () => {
    const r = byName.get(names[index]) || {};
    if (modiBox.includes(r.mode)) return true;
    // Il criterio e' il riquadro, non lo stato: un'immagine che il modulo non ha esaminato
    // resta senza riquadro anche dopo che lei le ha scritto la depth a mano, e allora non
    // e' piu' `missing` ma `corrected`. Guardare lo stato la lasciava fuori.
    return !r.box && propagabile();
  };

  let cerca = '';
  const passa = (name) => {
    const r = byName.get(name);
    if (cerca && !name.toLowerCase().includes(cerca)) return false;
    if (!filtro) return true;
    if (filtro.startsWith('mode:')) return r.mode === filtro.slice(5);
    if (filtro.startsWith('stato:')) return r.status === filtro.slice(6);
    return true;
  };
  const visibili = () => names.filter(passa);

  const stage = el('div', { class: 'editor-stage' });
  const image = el('img', { alt: '' });
  const boxNode = el('div', { class: 'editor-box' });
  boxNode.style.display = 'none';
  const maniglie = ['nw', 'n', 'ne', 'w', 'e', 'sw', 's', 'se'].map((dove) => {
    const h = el('div', { class: `handle handle-${dove}` });
    h.addEventListener('pointerdown', (e) => trascina(e, dove));
    return h;
  });
  boxNode.append(...maniglie);
  boxNode.addEventListener('pointerdown', (e) => {
    if (e.target !== boxNode) return;
    trascina(e, 'move');
  });
  stage.append(image, boxNode);

  /* L'area ingrandita. A grandezza naturale il numero della depth e' alto 13 px e il riquadro
     ci finisce sopra: non si vede piu' che cosa si sta ritagliando. Qui la stessa striscia
     arriva ingrandita, con il riquadro come overlay HTML trascinabile. La finestra del
     ritaglio resta ferma mentre si trascina, altrimenti scapperebbe sotto al cursore. */
  let INGRANDIMENTO = 6;
  const zoomStage = el('div', { class: 'editor-stage zoom-stage' });
  const zoomImg = el('img', { alt: '' });
  const zoomBox = el('div', { class: 'editor-box zoom-box' });
  const zoomManiglie = ['nw', 'n', 'ne', 'w', 'e', 'sw', 's', 'se'].map((dove) => {
    const h = el('div', { class: `handle handle-${dove}` });
    h.addEventListener('pointerdown', (e) => trascina(e, dove, scalaZoom));
    return h;
  });
  zoomBox.append(...zoomManiglie);
  zoomBox.addEventListener('pointerdown', (e) => {
    if (e.target !== zoomBox) return;
    trascina(e, 'move', scalaZoom);
  });
  zoomStage.append(zoomImg, zoomBox);
  let finestra = null;   // [x0, y0, x1, y1] in coordinate native, gia' ritagliate sull'immagine
  let trascinando = false;
  const scalaZoom = () => {
    const larghezza = finestra ? finestra[2] - finestra[0] : 1;
    return { x: (zoomImg.clientWidth || 1) / larghezza, y: (zoomImg.clientWidth || 1) / larghezza };
  };
  /* La finestra del ritaglio. Si sta larghi in orizzontale, dove le cifre crescono e dove
     serve spazio per trascinare, e stretti in verticale, dove non c'e' niente da vedere: le
     righe di interfaccia distano una ventina di pixel.

     Va ritagliata sui bordi dell'immagine **con la stessa regola del server**. Quando la
     depth sta contro il bordo destro — su `prova` il numero e' a 1250-1275 di 1280 — il
     server restituiva una striscia di 64 px mentre il client ne assumeva 93: il riquadro
     finiva fuori posto e il trascinamento andava piu' lento del cursore, di 93/64. */
  const nuovaFinestra = (box) => {
    const [w, h] = nativo();
    const margine = Math.max(60, Math.round(1.6 * (box.right - box.left)));
    let x0 = Math.round(box.left - margine);
    let x1 = Math.round(box.right + margine);
    let y0 = Math.round(box.top - 8);
    let y1 = Math.round(box.bottom + 8);
    // Contro un bordo si sposta la finestra invece di stringerla: la larghezza deve restare
    // quella che il server ritagliera' davvero.
    const larghezza = Math.min(w, x1 - x0);
    if (x0 < 0) { x0 = 0; x1 = larghezza; }
    if (x1 > w) { x1 = w; x0 = w - larghezza; }
    const altezza = Math.min(h, y1 - y0);
    if (y0 < 0) { y0 = 0; y1 = altezza; }
    if (y1 > h) { y1 = h; y0 = h - altezza; }
    return [Math.max(0, x0), Math.max(0, y0), Math.min(w, x1), Math.min(h, y1)];
  };
  const caricaZoom = (nome, box) => {
    finestra = nuovaFinestra(box);
    const [x0, y0, x1, y1] = finestra;
    zoomImg.src = `/api/projects/${projectId}/depth/crop?name=${encodeURIComponent(nome)}`
      + `&zoom=${INGRANDIMENTO}&raw=1&x0=${x0}&y0=${y0}&x1=${x1}&y1=${y1}`;
  };
  const disegnaZoom = () => {
    const r = byName.get(names[index]);
    const box = modificabile() ? bozza : (r && r.box);
    if (!box || !finestra) { zoomBox.style.display = 'none'; return; }
    // Se il riquadro esce dalla finestra la si rifa': succede solo tirandolo lontano.
    if (!trascinando
        && (box.left < finestra[0] + 2 || box.right > finestra[2] - 2
            || box.top < finestra[1] + 1 || box.bottom > finestra[3] - 1)) {
      // Mai mentre si trascina: la striscia scapperebbe sotto al cursore.
      caricaZoom(names[index], box);
    }
    if ((zoomImg.clientWidth || 0) < 10) {
      // Il ritaglio non ha ancora una larghezza: disegnarlo ora vorrebbe dire piazzare il
      // riquadro a coordinate assurde per un istante. Ci ripensa `load` o il ResizeObserver.
      zoomBox.style.display = 'none';
      return;
    }
    const s = scalaZoom();
    zoomBox.style.display = 'block';
    zoomBox.className = 'editor-box zoom-box' + (modificabile() ? '' : ' readonly');
    zoomBox.style.left = `${(box.left - finestra[0]) * s.x}px`;
    zoomBox.style.top = `${(box.top - finestra[1]) * s.y}px`;
    zoomBox.style.width = `${(box.right - box.left) * s.x}px`;
    zoomBox.style.height = `${(box.bottom - box.top) * s.y}px`;
    for (const h of zoomManiglie) h.style.display = modificabile() ? 'block' : 'none';
  };

  /* Le frecce stanno qui, accanto all'area ingrandita: e' li' che si guarda, ed e' li' che
     si vuole passare all'immagine dopo per controllare che il riquadro tenga. Rispondono
     anche i tasti freccia, tranne mentre si scrive in un campo. */
  const zoomEtichetta = el('span', { class: 'hint' });
  const zoomSel = el('select', { style: 'width:74px' },
    ...[4, 6, 8, 10].map((z) => el('option', { value: String(z), ...(z === 6 ? { selected: '' } : {}) }, `${z}×`)));
  zoomSel.addEventListener('change', () => {
    INGRANDIMENTO = parseInt(zoomSel.value, 10) || 6;
    const r = byName.get(names[index]);
    const box = (modificabile() ? bozza : r && r.box);
    if (box) caricaZoom(names[index], box);
    disegnaZoom();
    mostra();
  });
  const zoomTesta = el('div', { class: 'ov-nav zoom-testa' },
    el('button', { class: 'ghost sq', onclick: () => passo(-1) }, '‹'),
    el('button', { class: 'ghost sq', onclick: () => passo(1) }, '›'),
    zoomSel, zoomEtichetta);

  const caption = el('div', { class: 'hint' });
  const dettaglio = el('div', { class: 'hint' });
  const confronto = el('div', { class: 'hint' });

  const scala = () => ({
    x: image.clientWidth / ((size[0] || image.naturalWidth) || 1),
    y: image.clientHeight / ((size[1] || image.naturalHeight) || 1),
  });
  const nativo = () => [size[0] || image.naturalWidth || 1, size[1] || image.naturalHeight || 1];
  const normalizza = (b) => {
    const [w, h] = nativo();
    const box = {
      left: Math.round(Math.min(b.left, b.right - 3)),
      right: Math.round(Math.max(b.right, b.left + 3)),
      top: Math.round(Math.min(b.top, b.bottom - 3)),
      bottom: Math.round(Math.max(b.bottom, b.top + 3)),
    };
    box.left = Math.max(0, Math.min(box.left, w - 4));
    box.top = Math.max(0, Math.min(box.top, h - 4));
    box.right = Math.max(box.left + 3, Math.min(box.right, w));
    box.bottom = Math.max(box.top + 3, Math.min(box.bottom, h));
    return box;
  };

  /* Spostare non e' ridimensionare. Contro un bordo `normalizza` fermerebbe un lato e
     lascerebbe correre l'altro, e il riquadro si schiaccerebbe: su `prova` la depth sta a
     1251-1275 di 1280, cioe' a cinque pixel dal bordo destro, e trascinandolo a destra si
     appiattiva invece di fermarsi. Qui la traslazione viene limitata tutta intera. */
  const sposta = (box, dx, dy) => {
    const [w, h] = nativo();
    const larghezza = box.right - box.left;
    const altezza = box.bottom - box.top;
    const left = Math.max(0, Math.min(Math.round(box.left + dx), w - larghezza));
    const top = Math.max(0, Math.min(Math.round(box.top + dy), h - altezza));
    return { left, top, right: left + larghezza, bottom: top + altezza };
  };

  function trascina(event, quale, scalaFn) {
    if (!modificabile() || !bozza) return;
    event.preventDefault();
    event.stopPropagation();
    const s = (scalaFn || scala)();
    const da = { x: event.clientX, y: event.clientY, box: { ...bozza } };
    const muovi = (e) => {
      const dx = (e.clientX - da.x) / (s.x || 1);
      const dy = (e.clientY - da.y) / (s.y || 1);
      if (quale === 'move') {
        bozza = sposta(da.box, dx, dy);
      } else {
        const b = { ...da.box };
        if (quale.includes('w')) b.left += dx;
        if (quale.includes('e')) b.right += dx;
        if (quale.includes('n')) b.top += dy;
        if (quale.includes('s')) b.bottom += dy;
        bozza = normalizza(b);
      }
      disegna();
      disegnaZoom();
      renderRiquadro();
    };
    const molla = () => {
      window.removeEventListener('pointermove', muovi);
      window.removeEventListener('pointerup', molla);
      trascinando = false;
      disegnaZoom();
    };
    trascinando = true;
    window.addEventListener('pointermove', muovi);
    window.addEventListener('pointerup', molla);
  }

  /* Cosa vede la lente: il riquadro di questa immagine, col valore letto. */
  const contestoLente = () => {
    const r = byName.get(names[index]) || {};
    const box = (modificabile() ? bozza : r.box) || r.box;
    return {
      source: 'depth',
      projectId, name: names[index], size: nativo(),
      boxes: box ? [{ box, color: (modes[r.mode] || {}).color || '#3fb950',
                      label: r.depth_mm == null ? 'depth' : `${r.depth_mm} mm` }] : [],
      caption: r.ocr_text ? `letto «${r.ocr_text}»` : '',
      // Trascinabile solo dove il riquadro si puo' davvero cambiare: sulle immagini dove
      // la depth viene dalla scala non c'e' un riquadro da spostare.
      onChange: (box && modificabile()) ? (nuovo) => {
        bozza = normalizza({ ...nuovo });
        disegna();
        disegnaZoom();
        renderRiquadro();
      } : null,
    };
  };

  const disegna = () => {
    const r = byName.get(names[index]);
    const box = modificabile() ? bozza : (r && r.box);
    if (!r || !box) {
      boxNode.style.display = 'none';
      // Anche il vuoto va detto alla lente, se no resta il riquadro dell'immagine di prima.
      if (Lente.viva()) Lente.aggiorna(contestoLente());
      return;
    }
    const s = scala();
    const colore = (modes[r.mode] || {}).color || '#ffffff';
    boxNode.style.display = 'block';
    boxNode.className = 'editor-box readonly';
    boxNode.style.setProperty('--box-color', colore);
    boxNode.style.left = `${box.left * s.x}px`;
    boxNode.style.top = `${box.top * s.y}px`;
    boxNode.style.width = `${(box.right - box.left) * s.x}px`;
    boxNode.style.height = `${(box.bottom - box.top) * s.y}px`;
    for (const h of maniglie) h.style.display = 'none';
    if (Lente.viva()) Lente.aggiorna(contestoLente());
  };

  const mostra = () => {
    const lista = visibili();
    if (!lista.length) { caption.textContent = 'nessuna immagine con questo filtro'; return; }
    if (!lista.includes(names[index])) index = names.indexOf(lista[0]);
    const r = byName.get(names[index]);
    const m = modes[r.mode] || {};
    image.src = `/api/projects/${projectId}/image`
      + `?name=${encodeURIComponent(names[index])}&w=980`;
    caption.innerHTML = '';
    caption.append(
      `immagine ${lista.indexOf(names[index]) + 1} di ${lista.length} — ${names[index]} · `,
      el('strong', { style: `color:${r.depth_mm == null ? 'var(--muted)' : m.color || 'var(--text)'}` },
        r.depth_mm == null ? 'nessuna depth' : `${r.depth_mm} mm`),
      el('span', {}, ` · ${m.label || r.mode || '—'}`),
    );
    if (r.corrected) {
      caption.append(el('span', { class: 'score-fixed', style: 'margin-left:6px' }, 'corretta'));
    }
    if (r.from_box) {
      caption.append(el('span', { class: 'hint', style: 'margin-left:6px' }, 'dal riquadro'));
    }
    zoomEtichetta.innerHTML = '';
    if (!r.box && !(modello && modello.box)) {
      zoomEtichetta.append(
        `${lista.indexOf(names[index]) + 1} di ${lista.length} · `,
        el('strong', { style: 'color:var(--muted)' }, 'nessun riquadro su questa immagine'),
        el('span', { class: 'hint' }, r.depth_mm == null
          ? ' — il modulo non l\'ha esaminata: scrivi la depth qui sotto'
          : ` — la depth (${r.depth_mm} mm) l'hai scritta tu; premi «Stringi sul numero» `
            + 'per darle il riquadro di cartella'));
    } else zoomEtichetta.append(
      `${lista.indexOf(names[index]) + 1} di ${lista.length} · `,
      el('strong', { style: `color:${r.depth_mm == null ? 'var(--muted)' : m.color || 'var(--text)'}` },
        r.depth_mm == null ? 'nessuna depth' : `${r.depth_mm} mm`),
      el('span', {}, ` · trascina il riquadro o le maniglie · ← → cambia immagine, `
        + 'shift+frecce sposta di un pixel, shift+alt+← → il lato destro'),
    );
    dettaglio.innerHTML = '';
    dettaglio.append(el('div', {},
      `stato ${r.status}${r.score != null ? ` · punteggio ${r.score.toFixed(3)}` : ''}`
      + `${r.ocr_text ? ` · letto «${r.ocr_text}»` : ''}`
      + `${r.candidates ? ` · ${r.candidates} candidati` : ''}`));
    if (m.what) dettaglio.append(el('div', { style: `color:${m.color}` }, m.what));
    if (r.reason) dettaglio.append(el('div', { style: 'color:var(--muted)' }, r.reason));
    const stona = sospetti()[names[index]];
    if (stona) {
      const riga = el('div', { class: 'depth-stona' }, stona.reason);
      if (stona.suggested_mm != null) {
        const usa = el('button', { class: 'ghost sq2', style: 'margin-left:8px' },
          `usa ${stona.suggested_mm} mm`);
        usa.addEventListener('click', async () => {
          try {
            await api(`/projects/${projectId}/depth/correct`,
              { body: { name: names[index], depth_mm: stona.suggested_mm } });
            toast('valore corretto');
            await rileggiTutto();
          } catch (errore) { toast(errore.message, true); }
        });
        riga.append(usa);
      }
      dettaglio.append(riga);
    }
    confronto.innerHTML = '';
    const d = r.direct || {};
    const s2 = r.scale || {};
    confronto.append(el('div', {},
      `le due strade — etichetta: ${d.score != null ? d.score.toFixed(3) : '—'}`
      + `${d.text ? ` («${d.text}»)` : ''}`
      + ` · scala: ${s2.score != null ? s2.score.toFixed(3) : '—'}`
      + `${s2.value_mm != null ? ` (${s2.value_mm} mm)` : ''}`));
    // Senza riquadro proprio si parte da quello di cartella: la label sta nello stesso
    // posto in ogni fotogramma, quindi e' gia' quasi giusto e basta aggiustarlo.
    const prestato = (!r.box && modello && modello.box) ? modello.box : null;
    bozza = r.box ? { ...r.box } : (prestato ? { ...prestato } : null);
    if (bozza) {
      caricaZoom(names[index], bozza);
    } else {
      // Senza riquadro non c'e' niente da ingrandire, e lasciare il ritaglio di prima e'
      // peggio del vuoto: sembra questa immagine e non lo e'.
      finestra = null;
      zoomImg.removeAttribute('src');
      zoomBox.style.display = 'none';
    }
    disegna();
    disegnaZoom();
    aggiornaAmbito();
    renderLista();
    renderCorrezione();
    renderRiquadro();
  };

  const passo = (delta) => {
    const lista = visibili();
    if (!lista.length) return;
    const corrente = lista.indexOf(names[index]);
    index = names.indexOf(lista[((corrente < 0 ? 0 : corrente) + delta + lista.length) % lista.length]);
    mostra();
  };

  /* L'elenco: valore, metodo e punteggio, per trovare subito le sbagliate. */
  const listBox = el('div', { class: 'score-list' });
  const renderLista = () => {
    listBox.innerHTML = '';
    for (const name of visibili().slice(0, 400)) {
      const r = byName.get(name);
      const m = modes[r.mode] || {};
      const riga = el('div', {
        class: 'score-row' + (name === names[index] ? ' current' : ''),
      },
        el('span', { class: 'score-value', style: `color:${m.color || 'var(--muted)'}` },
          r.depth_mm == null ? '—' : `${r.depth_mm}`),
        el('span', { class: 'score-group', title: m.label || '' },
          r.mode === 'direct_label' ? 'etich.' : r.mode === 'scale' ? 'scala' : 'altro'),
        el('span', { class: 'score-name', title: name }, name.split('/').pop()),
        el('span', { class: 'hint' }, r.score != null ? r.score.toFixed(2) : ''),
        r.corrected ? el('span', { class: 'score-fixed' }, 'corretta') : null,
        r.status === 'missing' ? el('span', { class: 'hint', title: 'il modulo non l\'ha esaminata' }, 'da fare') : null,
        (!r.corrected && sospetti()[name]) ? el('span', { class: 'depth-stona-segno', title: 'lettura che stona' }, '!') : null,
      );
      riga.addEventListener('click', () => { index = names.indexOf(name); mostra(); });
      listBox.append(riga);
      if (name === names[index]) {
        requestAnimationFrame(() => riga.scrollIntoView({ block: 'nearest' }));
      }
    }
  };

  /* Il riquadro: si stringe qui, si applica a tutta la cartella.

     Vale solo per la depth letta dall'interfaccia. Il match e' tanto piu' buono quanto piu'
     il riquadro sta stretto sul solo numero, e siccome la label non si sposta lo stesso
     riquadro serve tutte le immagini: da li' il valore viene riletto una per una, lasciando
     spazio alle cifre che crescono. */
  const riquadro = el('div', {});
  const totaleCartella = data.images_total || rows.length;

  /* --- quante immagini hanno una depth, perche' le altre no, e la conferma ------------ */
  const coperturaBox = el('div', { class: 'depth-copertura' });
  const strettoBox = el('div', { class: 'depth-stretto' });
  const coerenzaBox = el('div', { class: 'depth-coerenza' });
  const sospetti = () => ((data.coherence || {}).suspects || {});
  const renderCoerenza = () => {
    coerenzaBox.innerHTML = '';
    const c = data.coherence || {};
    const elenco = Object.entries(c.suspects || {});
    if (!elenco.length) { coerenzaBox.style.display = 'none'; return; }
    coerenzaBox.style.display = '';
    const conProposta = elenco.filter(([, v]) => v.suggested_mm != null);
    coerenzaBox.append(
      el('strong', {}, `${elenco.length} letture stonano rispetto alla cartella`),
      el('span', { class: 'hint' },
        ` — qui la depth e\' scritta «${c.format}» e va da ${(c.range || [])[0]} a `
        + `${(c.range || [])[1]} mm. Chi devia di solito e\' la stessa depth letta male.`));
    if (conProposta.length) {
      const tutti = el('button', { class: 'ghost' },
        `Correggi i ${conProposta.length} valori fuori scala`);
      tutti.addEventListener('click', async () => {
        tutti.disabled = true;
        try {
          for (const [nome, v] of conProposta) {
            await api(`/projects/${projectId}/depth/correct`,
              { body: { name: nome, depth_mm: v.suggested_mm } });
          }
          toast(`corretti ${conProposta.length} valori`);
          await rileggiTutto();
        } catch (errore) { toast(errore.message, true); }
        finally { tutti.disabled = false; }
      });
      coerenzaBox.append(el('div', { class: 'row', style: 'margin-top:6px' }, tutti,
        el('span', { class: 'hint' },
          conProposta.map(([n, v]) => `${n.split('/').pop()} → ${v.suggested_mm}`).join(' · '))));
    }
    const senza = elenco.filter(([, v]) => v.suggested_mm == null);
    if (senza.length) {
      coerenzaBox.append(el('div', { class: 'hint', style: 'margin-top:4px' },
        `${senza.length} da guardare a mano: ` + senza.map(([n]) => n.split('/').pop()).join(', ')));
    }
  };
  /* Stringere i riquadri sul solo numero, su tutta la cartella.

     Il riquadro che il modulo consegna e' quello di parola dell'OCR: comprende l'unita',
     le lettere attaccate e la tacca del righello se il numero le sta accanto. Per leggere
     va bene; per andare a match su un'altra immagine no, perche' quel contorno cambia da
     un fotogramma all'altro. Il valore non si tocca mai: si verifica che il riquadro
     stretto legga le stesse cifre, e chi non lo conferma resta com'e'. */
  /* Stringere e' un gesto di cartella, non di immagine: l'etichetta sta nello stesso
     posto in tutti i fotogrammi, e non ha senso rifilarne uno alla volta. Da qualunque
     tasto si parta, il lavoro e' lo stesso. */
  const stringiTutti = async (bottone, stato) => {
    if (bottone) bottone.disabled = true;
    try {
      const { job_id } = await api(`/projects/${projectId}/depth/tighten`,
        { body: { scope: 'all' } });
      const job = await pollJob(job_id, stato);
      if (stato) stato.textContent = '';
      const esito = job.result || {};
      toast(`${esito.tightened} riquadri su ${esito.targets} stretti sul numero`
        + (esito.folder_box ? ` · ${esito.folder_box} col riquadro di cartella` : ''));
      await rileggiTutto();
    } catch (errore) {
      if (stato) stato.textContent = errore.message;
      toast(errore.message, true);
    } finally { if (bottone) bottone.disabled = false; }
  };

  const renderStretto = () => {
    strettoBox.innerHTML = '';
    const conRiquadro = names.filter((n) => (byName.get(n) || {}).box).length;
    if (!conRiquadro) { strettoBox.style.display = 'none'; return; }
    strettoBox.style.display = '';
    const fatto = data.box_tightened;
    strettoBox.append(el('strong', {}, 'Riquadri stretti sul solo numero'));
    strettoBox.append(el('span', { class: 'hint' },
      ' — via unita\', lettere e tacche: resta il numero, che e\' l\'unica cosa uguale in '
      + 'tutte le immagini. La depth non cambia: se il riquadro stretto legge cifre diverse, '
      + 'quell\'immagine resta com\'era.'));
    const stato = el('span', { class: 'hint' });
    const via = el('button', { class: 'ghost' },
      fatto ? `Rifai su ${conRiquadro} riquadri` : `Stringi i ${conRiquadro} riquadri`);
    via.addEventListener('click', () => stringiTutti(via, stato));
    strettoBox.append(el('div', { class: 'row', style: 'margin-top:6px' }, via, stato));
    if (fatto) {
      const b = fatto.box;
      strettoBox.append(el('div', { class: 'hint' },
        `${fatto.tightened} su ${fatto.targets} stretti rileggendo il numero`
        + (fatto.folder_box
          ? `; ${fatto.folder_box} non si rileggevano e hanno preso il riquadro di cartella`
            + (b ? ` (${b.right - b.left}x${b.bottom - b.top} px, dove il numero sta in tutte le altre)` : '')
          : '')
        + ` — ${fatto.at.replace('T', ' ')}.`));
    }
    // Le immagini rimaste indietro. Spesso non sono un fallimento: il riquadro stretto
    // legge «3.0» dove la parola larga aveva letto «30cm», cioe' ritrova il punto che si
    // era perso. Si mostra cosa ha letto, e la correzione resta un gesto suo.
    const rimaste = (fatto && fatto.unchanged) || [];
    if (rimaste.length) {
      const prestate = rimaste.filter((r) => r.folder_box).length;
      strettoBox.append(el('div', { class: 'hint', style: 'margin-top:6px' },
        `${rimaste.length} non si sono lasciate rileggere`
        + (prestate ? ` (${prestate} hanno preso il riquadro di cartella, che e' comunque `
                      + 'sul solo numero)' : '')
        + ' — ecco cosa ha letto il riquadro stretto:'));
      const elenco = el('div', { class: 'depth-rimaste' });
      for (const riga of rimaste.slice(0, 12)) {
        const vai = el('button', { class: 'ghost link' }, riga.name.split('/').pop());
        vai.addEventListener('click', () => {
          const dove = names.indexOf(riga.name);
          if (dove >= 0) { index = dove; filtro = ''; rifaiChips(); mostra(); }
        });
        const voce = el('div', {}, vai, el('span', { class: 'hint' },
          ` legge «${riga.text || '—'}»`
          + (riga.value_mm != null ? ` = ${riga.value_mm} mm` : '')
          + (riga.folder_box ? ' · riquadro di cartella' : '')));
        // Il tasto solo se il valore letto sta nella scala della cartella: su prova_3
        // un riquadro legge «50» = 500 mm, e offrire di applicarlo sarebbe un tranello.
        const scala = ((data.coherence || {}).range) || [];
        const plausibile = riga.value_mm != null && (scala.length !== 2
          || (riga.value_mm >= scala[0] * 0.9 && riga.value_mm <= scala[1] * 1.1));
        if (plausibile && riga.value_mm !== (byName.get(riga.name) || {}).depth_mm) {
          const usa = el('button', { class: 'ghost sq2', style: 'margin-left:6px' },
            `usa ${riga.value_mm} mm`);
          usa.addEventListener('click', async () => {
            try {
              await api(`/projects/${projectId}/depth/correct`,
                { body: { name: riga.name, depth_mm: riga.value_mm } });
              toast('depth corretta');
              await rileggiTutto();
            } catch (errore) { toast(errore.message, true); }
          });
          voce.append(usa);
        }
        elenco.append(voce);
      }
      strettoBox.append(elenco);
    }
  };

  const renderCopertura = () => {
    const c = data.coverage || {};
    coperturaBox.innerHTML = '';
    coperturaBox.className = 'depth-copertura'
      + (data.confirmed ? ' confermata' : (c.with_depth < c.total ? ' parziale' : ''));
    coperturaBox.append(el('strong', {},
      `${c.with_depth} immagini su ${c.total} hanno una depth`));
    if (c.reason) coperturaBox.append(el('span', { class: 'hint' }, ` — ${c.reason}.`));
    const riga = el('div', { class: 'row', style: 'margin-top:6px' });
    const mancanti = (conteggi.by_status || {}).missing || 0;
    if (mancanti) {
      const vedi = el('button', { class: 'ghost' }, `Vedi le ${mancanti} senza depth`);
      vedi.addEventListener('click', () => {
        filtro = 'stato:missing';
        vista = 'singola';
        rifaiChips();
        applicaVista();
      });
      riga.append(vedi);
    }
    if (c.with_depth < c.total && !c.can_propagate) {
      const tutte = el('button', { class: 'ghost' }, `Cerca la depth su tutte le ${c.total}`);
      const stato = el('span', { class: 'hint' }, 'il modulo gira su ogni immagine: qualche minuto');
      tutte.addEventListener('click', async () => {
        tutte.disabled = true;
        try {
          const { job_id } = await api(`/projects/${projectId}/depth/run`, { body: { sample: 0 } });
          await pollJob(job_id, stato);
          toast('depth cercata su tutte le immagini');
          await rileggiTutto();
        } catch (errore) { toast(errore.message, true); stato.textContent = errore.message; }
        finally { tutte.disabled = false; }
      });
      riga.append(tutte, stato);
    }
    riga.append(confermaInDueTempi(
      data.confirmed ? 'Togli la conferma' : 'Conferma la depth e passa alla scala',
      data.confirmed
        ? 'la depth torna a essere una proposta.'
        : 'queste depth valgono come confermate: lo studio della scala le usera\' come dato certo.',
      async () => {
        try {
          await api(`/projects/${projectId}/depth/confirm`,
            { body: data.confirmed ? { reset: true } : {} });
          toast(data.confirmed ? 'conferma tolta' : 'depth confermata');
          await rileggiTutto();
        } catch (errore) { toast(errore.message, true); }
      }));
    coperturaBox.append(riga);
  };
  const avanzamento = el('span', { class: 'hint' });
  /* L'ambito: il modulo gira su un campione, ma la label sta nell'interfaccia di ogni
     fotogramma. Rileggere un numero dentro a un riquadro gia' noto costa poco, quindi la
     cartella intera e' il default. */
  const ambitoSel = el('select', {},
    el('option', { value: 'all' }, `tutte le immagini della cartella (${totaleCartella})`),
    el('option', { value: 'names' }, 'solo le immagini elencate ora'),
    el('option', { value: 'run' }, `solo quelle del campione (${rows.length})`));
  /* Il filtro per nome esiste per questo: una cartella puo' avere piu' di un'interfaccia.
     Su `prova` le immagini `_trans` hanno la riga della depth 22 px piu' in basso, e con un
     riquadro solo leggevano «646» invece di 15, 20, 23... Si scrive `_trans`, si mette il
     riquadro al posto giusto su una di quelle, e si applica alle sole elencate. */
  const campoCerca = el('input', {
    type: 'search', placeholder: 'filtra per nome file, es. _trans', style: 'width:230px',
  });
  campoCerca.addEventListener('input', () => {
    cerca = campoCerca.value.trim().toLowerCase();
    aggiornaAmbito();
    if (vista === 'riepilogo') renderRiepilogo(); else mostra();
  });
  const aggiornaAmbito = () => {
    ambitoSel.children[1].textContent = `solo le immagini elencate ora (${visibili().length})`;
  };
  const rileggiTutto = async () => {
    const fresca = await api(`/projects/${projectId}/depth`);
    const corrente = names[index];
    byName.clear();
    for (const riga of fresca.rows || []) byName.set(riga.name, riga);
    // La rilettura porta dentro anche le immagini fuori dal campione del modulo: l'elenco
    // e' quindi piu' lungo di prima, non solo aggiornato.
    names = (fresca.rows || []).map((r) => r.name);
    index = Math.max(0, names.indexOf(corrente));
    modello = fresca.box_template || null;
    data.coverage = fresca.coverage; data.confirmed = fresca.confirmed;
    data.coherence = fresca.coherence;
    data.images_total = fresca.images_total;
    conteggi = { by_mode: fresca.by_mode || {}, by_status: fresca.by_status || {} };
    rifaiChips();
    renderCopertura();
    renderCoerenza();
    renderStretto();
    mostra();
  };
  const renderRiquadro = () => {
    riquadro.innerHTML = '';
    const r = byName.get(names[index]);
    if (!r) return;
    if (!modificabile()) {
      riquadro.append(el('div', { class: 'hint' },
        'depth dedotta dalla scala: il riquadro non si puo\' propagare, il numero cambia '
        + 'posto ad ogni immagine.'));
      if (modello) {
        riquadro.append(el('div', { class: 'hint' },
          `riquadro di cartella attivo su ${modello.applied} immagini con etichetta.`));
      }
      // Propagare no, stringere si': il riquadro resta dov'e', gli si toglie il contorno.
      if (r.box) {
        const soloNumero = el('button', { class: 'ghost' },
          'Stringi sul numero, tutte le immagini');
        const dice = el('span', { class: 'hint' });
        soloNumero.addEventListener('click', () => stringiTutti(soloNumero, dice));
        riquadro.append(el('div', { class: 'row' }, soloNumero, dice));
      }
      return;
    }
    if (!bozza) {
      riquadro.append(el('p', { class: 'hint' },
        r.status === 'missing'
          ? 'nessun riquadro ancora: stringine uno su un\'immagine dove la depth e\' stata '
            + 'trovata, e da li\' si applica anche a questa.'
          : 'nessun riquadro su questa immagine'));
      return;
    }
    if (!r.box) {
      riquadro.append(el('div', { class: 'depth-prestito' },
        'questa immagine un riquadro suo non ce l\'ha: il modulo non l\'ha esaminata. '
        + `Quello qui sopra e' il riquadro di cartella, preso da ${modello.from}: `
        + 'spostalo se serve, poi applicalo.'));
    }
    const misura = el('span', { class: 'hint' },
      `${bozza.right - bozza.left} x ${bozza.bottom - bozza.top} px`);
    /* Il riquadro dell'OCR e' quello di *parola*, e la parola comprende cio' che sta
       attaccato al numero: l'unita', una sigla, la tacca del righello. Il tasto le fa
       tutte: l'etichetta e' nello stesso posto in ogni fotogramma, e stringerne una alla
       volta sarebbe solo fatica. */
    const stringi = el('button', { class: 'ghost' }, 'Stringi sul numero, tutte le immagini');
    const esitoStretto = el('span', { class: 'hint' });
    stringi.addEventListener('click', () => stringiTutti(stringi, esitoStretto));
    const campi = el('div', { class: 'row' });
    for (const lato of ['left', 'top', 'right', 'bottom']) {
      const campo = el('input', {
        type: 'number', step: '1', style: 'width:78px', value: String(bozza[lato]),
      });
      campo.addEventListener('change', () => {
        bozza = normalizza({ ...bozza, [lato]: parseInt(campo.value, 10) || 0 });
        disegna();
        disegnaZoom();
        renderRiquadro();
      });
      campi.append(el('span', { class: 'hint' }, lato), campo);
    }
    riquadro.append(
      el('div', { class: 'hint' },
        'trascina il riquadro o le maniglie per stringerlo sul solo numero: piu\' e\' stretto, '
        + 'piu\' il match tiene su tutte le immagini.'),
      el('div', { class: 'row' }, misura,
        Lente.bottone(contestoLente),
        el('span', { class: 'hint' }, 'finestra a parte, per il secondo schermo')),
      el('div', { class: 'row' }, stringi, esitoStretto),
      campi,
      el('div', { class: 'row' }, ambitoSel,
        el('span', { class: 'hint' }, `${totaleCartella} immagini nella cartella, `
          + `${rows.length} nel campione del modulo`)),
      el('div', { class: 'row' }, confermaInDueTempi(
        'Applica il riquadro e rileggi la depth',
        'lo stesso riquadro viene usato su ogni immagine dell\'ambito scelto e il valore '
        + 'viene riletto li\'. Sulla cartella intera ci vuole qualche minuto.',
        async () => {
          try {
            const corpoRichiesta = { name: names[index], box: bozza, scope: ambitoSel.value };
            if (ambitoSel.value === 'names') corpoRichiesta.names = visibili();
            const avvio = await api(`/projects/${projectId}/depth/box`, { body: corpoRichiesta });
            const job = await pollJob(avvio.job_id, avanzamento);
            const esito = job.result || {};
            avanzamento.textContent = '';
            await rileggiTutto();
            toast(`riquadro applicato a ${esito.applied} immagini su ${esito.targets}`
              + (esito.failed ? ` — ${esito.failed} senza numero` : ''));
          } catch (error) { avanzamento.textContent = ''; toast(error.message, true); }
        }), avanzamento),
    );
    if (modello) {
      riquadro.append(el('div', { class: 'hint' },
        `riquadro di cartella preso da ${modello.from}, applicato a ${modello.applied} immagini`
        + `${modello.unit_factor === 10 ? ' (letto in cm)' : ''}`
        + `${(modello.failed || []).length ? `, ${modello.failed.length} senza numero` : ''}.`));
      riquadro.append(el('div', { class: 'row' }, confermaInDueTempi(
        'Torna ai riquadri del modulo', 'il riquadro di cartella viene tolto.',
        async () => {
          try {
            await api(`/projects/${projectId}/depth/box`, { body: { reset: true } });
            await rileggiTutto();
            toast('riquadro di cartella tolto');
          } catch (error) { toast(error.message, true); }
        })));
    }
  };

  /* La correzione qui e' un numero, non un riquadro: la depth sbagliata si riscrive. */
  const correzione = el('div', { class: 'row' });
  const renderCorrezione = () => {
    correzione.innerHTML = '';
    const r = byName.get(names[index]);
    if (!r) return;
    const campo = el('input', {
      type: 'number', step: '0.5', style: 'width:110px',
      value: r.depth_mm != null ? String(r.depth_mm) : '',
    });
    const nuova = r.depth_mm == null;
    correzione.append(
      el('span', { class: 'hint' }, nuova ? 'depth di questa immagine (mm)' : 'depth corretta (mm)'),
      campo);
    // Le depth gia' viste nella cartella: quasi sempre quella giusta e' una di queste, e
    // sceglierla da un elenco e' piu' svelto e piu' sicuro che riscriverla.
    const noti = Object.keys(data.values_mm || {})
      .map(Number).filter((v) => !Number.isNaN(v)).sort((a, b) => a - b);
    if (nuova && noti.length) {
      const scorciatoie = el('span', { class: 'row', style: 'gap:4px' },
        el('span', { class: 'hint' }, 'gia\' nella cartella:'));
      for (const v of noti.slice(0, 8)) {
        const b = el('button', { class: 'ghost sq2' }, `${v}`);
        b.addEventListener('click', () => { campo.value = String(v); campo.dispatchEvent(new Event('input')); });
        scorciatoie.append(b);
      }
      correzione.append(scorciatoie);
    }
    correzione.append(confermaInDueTempi(nuova ? 'Indica questa depth' : 'Salva la depth',
      'vale solo per questa immagine.',
      async () => {
        try {
          await api(`/projects/${projectId}/depth/correct`,
            { body: { name: names[index], depth_mm: parseFloat(campo.value) } });
          const fresca = await api(`/projects/${projectId}/depth`);
          for (const riga of fresca.rows || []) byName.set(riga.name, riga);
          toast('depth corretta');
          mostra();
        } catch (error) { toast(error.message, true); }
      }));
    // In blocco sulle immagini che il modulo non ha esaminato: quando la depth e' la
    // stessa (ed e' il caso normale di una cartella) riscriverla una per una e' solo fatica.
    const senza = visibili().filter((n) => (byName.get(n) || {}).depth_mm == null);
    if (senza.length > (nuova ? 1 : 0)) {
      const altre = nuova ? senza.length - 1 : senza.length;
      correzione.append(confermaInDueTempi(
        nuova ? `Assegna anche alle altre ${altre} senza depth`
              : `Assegna questo valore alle ${altre} senza depth`,
        `lo stesso valore viene scritto sulle ${senza.length} immagini senza depth fra `
        + 'quelle elencate ora. Restano correzioni tue: si tolgono una per una.',
        async () => {
          const valore = parseFloat(campo.value);
          if (Number.isNaN(valore)) { toast('scrivi prima la depth in millimetri', true); return; }
          try {
            const esito = await api(`/projects/${projectId}/depth/correct`,
              { body: { names: senza, depth_mm: valore } });
            toast(`${valore} mm su ${esito.applied} immagini`);
            await rileggiTutto();
          } catch (error) { toast(error.message, true); }
        }));
    }
    if (r.corrected) {
      correzione.append(confermaInDueTempi('Torna al valore del modulo', 'la correzione viene tolta.',
        async () => {
          try {
            await api(`/projects/${projectId}/depth/correct`,
              { body: { name: names[index], reset: true } });
            const fresca = await api(`/projects/${projectId}/depth`);
            for (const riga of fresca.rows || []) byName.set(riga.name, riga);
            toast('correzione tolta');
            mostra();
          } catch (error) { toast(error.message, true); }
        }));
    }
  };

  // --- filtri: per metodo e per stato, con il conto
  const chips = el('div', { class: 'ov-chips' });
  const etichetteStato = {
    accepted: 'accettate', review: 'da rivedere', reject: 'scartate',
    corrected: 'corrette', box: 'dal riquadro', missing: 'senza depth',
  };
  const rifaiChips = () => {
    chips.innerHTML = '';
    const filtri = [
      ['', `tutte ${names.length}`],
      ...Object.entries(modes).map(([id, m]) => [`mode:${id}`,
        `${m.label} ${(conteggi.by_mode || {})[id] || 0}`]),
      ...Object.keys(etichetteStato).map((s) => [`stato:${s}`,
        `${etichetteStato[s]} ${(conteggi.by_status || {})[s] || 0}`]),
    ];
    const bottoni = [];
    for (const [chiave, etichetta] of filtri) {
      const b = el('button', { class: 'chip' + (chiave === filtro ? ' on' : '') }, etichetta);
      if (chiave.startsWith('mode:')) b.style.borderColor = (modes[chiave.slice(5)] || {}).color;
      b.addEventListener('click', () => {
        filtro = chiave;
        for (const [altro, k] of bottoni) altro.className = 'chip' + (k === filtro ? ' on' : '');
        if (vista === 'riepilogo') renderRiepilogo(); else mostra();
      });
      bottoni.push([b, chiave]);
      chips.append(b);
    }
  };
  rifaiChips();

  /* Il riepilogo: tutte le depth trovate, una scheda per valore, con il ritaglio ingrandito
     dell'interfaccia e il riquadro disegnato sopra. A grandezza naturale il numero e' alto
     13 px e non si giudica: qui si vede subito se il riquadro si e' fermato sulle cifre. */
  const riepilogo = el('div', { class: 'depth-gallery' });
  const espansi = new Set();
  const ritaglio = (nome) => el('img', {
    class: 'depth-crop', loading: 'lazy', alt: '',
    src: `/api/projects/${projectId}/depth/crop?name=${encodeURIComponent(nome)}&zoom=3`,
  });
  const renderRiepilogo = () => {
    riepilogo.innerHTML = '';
    const perValore = new Map();
    for (const nome of visibili()) {
      const r = byName.get(nome);
      if (r.depth_mm == null) continue;
      if (!perValore.has(r.depth_mm)) perValore.set(r.depth_mm, []);
      perValore.get(r.depth_mm).push(r);
    }
    if (!perValore.size) {
      riepilogo.append(el('p', { class: 'hint' }, 'nessuna depth con questo filtro'));
      return;
    }
    riepilogo.append(el('p', { class: 'hint' },
      `${perValore.size} valori distinti su ${visibili().length} immagini. `
      + 'Il riquadro verde e\' quello da cui il numero e\' stato letto, ingrandito 3 volte.'));
    const griglia = el('div', { class: 'depth-grid' });
    for (const [valore, elenco] of [...perValore.entries()].sort((a, b) => a[0] - b[0])) {
      const primo = elenco[0];
      const b = primo.box || {};
      const aperta = espansi.has(valore);
      // La scheda aperta prende tutta la riga: e' li' che si scorrono le decine di ritagli.
      const scheda = el('div', { class: 'depth-card' + (aperta ? ' aperta' : '') });
      const cartelle = new Set(elenco.map((r) => r.name.split('/').slice(0, -1).join('/')));
      scheda.append(
        el('div', { class: 'depth-value' }, `${valore} mm`,
          el('span', { class: 'hint' }, ` · ${elenco.length} immagini`)),
        ritaglio(primo.name),
        el('div', { class: 'hint' },
          `riquadro ${(b.right - b.left) || '?'}×${(b.bottom - b.top) || '?'} px @ ${b.left},${b.top}`
          + ` · ${cartelle.size} cartell${cartelle.size === 1 ? 'a' : 'e'}`),
      );
      const azioni = el('div', { class: 'row' },
        el('button', { class: 'ghost sq2' }, aperta ? 'nascondi le altre' : `vedi tutte (${elenco.length})`),
        el('button', { class: 'ghost sq2' }, 'apri questa immagine'));
      azioni.children[0].addEventListener('click', () => {
        if (aperta) espansi.delete(valore); else espansi.add(valore);
        renderRiepilogo();
      });
      azioni.children[1].addEventListener('click', () => {
        index = names.indexOf(primo.name);
        vista = 'singola';
        applicaVista();
      });
      scheda.append(azioni);
      if (aperta) {
        const tutte = el('div', { class: 'depth-strip' });
        for (const r of elenco.slice(0, 120)) {
          const voce = el('div', { class: 'depth-strip-item', title: r.name },
            ritaglio(r.name),
            el('div', { class: 'hint' }, r.name.split('/').pop()));
          voce.addEventListener('click', () => {
            index = names.indexOf(r.name); vista = 'singola'; applicaVista();
          });
          tutte.append(voce);
        }
        if (elenco.length > 120) {
          tutte.append(el('div', { class: 'hint' }, `… e altre ${elenco.length - 120}`));
        }
        scheda.append(tutte);
      }
      griglia.append(scheda);
    }
    riepilogo.append(griglia);
  };

  // --- le due viste: una per una, oppure tutte le depth trovate in un colpo d'occhio
  let vista = 'singola';
  const corpo = el('div', { class: 'ov-body' },
    el('div', { class: 'ov-main' }, zoomTesta, zoomStage, stage,
      el('div', { class: 'ov-under' },
        el('div', { class: 'ov-under-text' }, dettaglio, confronto, riquadro, correzione))),
    el('div', { class: 'ov-side' }, listBox));
  const vistaBtn = el('div', { class: 'ov-chips' },
    el('button', { class: 'chip on' }, 'una per una'),
    el('button', { class: 'chip' }, 'riepilogo per valore'));
  const applicaVista = () => {
    vistaBtn.children[0].className = 'chip' + (vista === 'singola' ? ' on' : '');
    vistaBtn.children[1].className = 'chip' + (vista === 'singola' ? '' : ' on');
    corpo.style.display = vista === 'singola' ? '' : 'none';
    barra.style.display = vista === 'singola' ? '' : 'none';
    riepilogo.style.display = vista === 'singola' ? 'none' : '';
    if (vista === 'singola') mostra(); else renderRiepilogo();
  };
  vistaBtn.children[0].addEventListener('click', () => { vista = 'singola'; applicaVista(); });
  vistaBtn.children[1].addEventListener('click', () => { vista = 'riepilogo'; applicaVista(); });

  const nav = el('div', { class: 'ov-nav' },
    el('button', { class: 'ghost sq', onclick: () => passo(-1) }, '‹'),
    el('button', { class: 'ghost sq', onclick: () => passo(1) }, '›'),
    caption);
  const barra = el('div', { class: 'ov-bar' }, nav);

  root.append(
    el('p', { class: 'hint' },
      `depth calcolata su ${rows.length} immagini delle ${data.images_total} della cartella. `
      + 'Il riquadro sull\'immagine e\' il posto da cui il numero e\' stato preso, colorato '
      + 'secondo il metodo.'),
    coperturaBox,
    coerenzaBox,
    strettoBox,
    chips,
    el('div', { class: 'row' }, campoCerca,
      el('span', { class: 'hint' }, 'filtra le immagini per nome: serve per applicare un '
        + 'secondo riquadro a una parte sola della cartella')),
    vistaBtn,
    barra,
    corpo,
    riepilogo,
  );

  riepilogo.style.display = 'none';
  renderCopertura();
  renderCoerenza();
  renderStretto();
  const daTastiera = (event) => {
    if (vista !== 'singola' || !root.isConnected) return;
    const dove = event.target;
    if (dove && ['INPUT', 'TEXTAREA', 'SELECT'].includes(dove.tagName)) return;
    const frecce = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    const delta = frecce[event.key];
    if (!delta) return;
    event.preventDefault();
    if (event.shiftKey) {
      // Un pixel per volta, per la posa fine che col mouse non viene.
      if (!modificabile() || !bozza) return;
      // shift+alt allarga o stringe il lato destro: e' il lato che si muove quando le cifre
      // cambiano, ed e' quello che si sbaglia piu' spesso.
      bozza = event.altKey
        ? normalizza({ ...bozza, right: bozza.right + delta[0] })
        : sposta(bozza, delta[0], delta[1]);
      disegna();
      disegnaZoom();
      renderRiquadro();
      return;
    }
    if (event.key === 'ArrowLeft') passo(-1);
    if (event.key === 'ArrowRight') passo(1);
  };
  window.addEventListener('keydown', daTastiera);
  zoomImg.addEventListener('load', disegnaZoom);
  if (window.ResizeObserver) new ResizeObserver(() => disegnaZoom()).observe(zoomImg);
  image.addEventListener('load', disegna);
  window.addEventListener('resize', disegna);
  if (window.ResizeObserver) new ResizeObserver(() => disegna()).observe(image);
  mostra();
  return root;
}
