/* La lente: una finestra a parte dove si vede ingrandito il riquadro che si sta disegnando.

   A grandezza naturale un riquadro di 30x18 px non si giudica - e' il problema che si
   ripresenta a ogni box, dalla depth al template dell'ecografo. Lo zoom dentro alla pagina
   aiuta, ma ruba spazio proprio all'immagine su cui si sta lavorando. Con due schermi la
   risposta giusta e' un'altra finestra: si trascina sul secondo monitor e resta li'.

   Il ritaglio non si ricarica a ogni pixel di trascinamento. Si tiene una finestra piu'
   larga del riquadro e la si rifa' solo quando il riquadro ne esce: dentro, a muoversi e'
   soltanto il rettangolo disegnato sopra, che costa niente.

   La scala del ritaglio si **misura** (larghezza dell'immagine diviso larghezza della
   finestra in pixel nativi) invece di darla per buona: il server la finestra la taglia sui
   bordi, e assumerla porta il rettangolo fuori posto proprio dove serve di piu'. */

const Lente = (() => {
  let win = null;
  let ctx = null;         // {projectId, name, size, boxes:[{box,color,label}], caption}
  let finestra = null;    // [x0, y0, x1, y1] in coordinate native
  let nomeInFinestra = '';

  const viva = () => !!(win && !win.closed && win.document && win.document.getElementById('crop'));

  const DOC = `<!doctype html><html lang="it"><head><meta charset="utf-8">
<title>Lente — ESIBuilder AI</title><style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #0d1117; color: #c9d1d9;
         font: 13px ui-monospace, Menlo, monospace; overflow: hidden; }
  #testa { padding: 6px 10px; border-bottom: 1px solid #30363d; display: flex;
           gap: 10px; align-items: baseline; justify-content: space-between; }
  #titolo { font-weight: 600; }
  #dettaglio { color: #8b949e; }
  #scena { position: relative; display: inline-block; }
  /* Il ritaglio non deve mai uscire dalla finestra: se lei la rimpicciolisce, si rimpicciolisce
     anche lui. La scala del rettangolo disegnato sopra non si rompe perche' viene misurata
     ogni volta, non calcolata dallo zoom chiesto al server. */
  #crop { display: block; image-rendering: pixelated;
          max-width: 100vw; max-height: calc(100vh - 46px); }
  .riq { position: absolute; border: 2px solid #3fb950; box-sizing: border-box;
         box-shadow: 0 0 0 1px rgba(0,0,0,.75); pointer-events: none; }
  /* Il riquadro su cui si sta lavorando si trascina anche da qui: e' il posto dove si vede
     davvero dove cade il bordo, quindi e' il posto dove ha senso spostarlo. */
  .riq.viva { pointer-events: auto; cursor: move; }
  .man { position: absolute; width: 14px; height: 14px; margin: -7px 0 0 -7px;
         border: 2px solid #0d1117; border-radius: 3px; background: #3fb950;
         pointer-events: auto; }
  .man.n, .man.s { cursor: ns-resize; }
  .man.w, .man.e { cursor: ew-resize; }
  .man.nw, .man.se { cursor: nwse-resize; }
  .man.ne, .man.sw { cursor: nesw-resize; }
  .riq b { position: absolute; top: -17px; left: -2px; font-size: 11px; font-weight: 600;
           background: #0d1117; padding: 0 3px; white-space: nowrap; }
  #vuoto { padding: 18px; color: #8b949e; }
</style></head><body>
<div id="testa"><span id="titolo">lente</span><span id="dettaglio"></span></div>
<div id="scena"><img id="crop" alt=""></div>
<div id="vuoto">niente da ingrandire: scegli un riquadro nella pagina principale.</div>
</body></html>`;

  const apri = () => {
    if (viva()) { win.focus(); return true; }
    win = window.open('', 'esibuilder-lente',
      'width=1040,height=780,menubar=no,toolbar=no,location=no,status=no');
    if (!win) return false;
    win.document.open();
    win.document.write(DOC);
    win.document.close();
    finestra = null;
    nomeInFinestra = '';
    // La finestra puo' essere ridimensionata o spostata su un altro schermo: il ritaglio
    // si rifa' per riempirla, se no si resta con l'ingrandimento della vecchia dimensione.
    win.addEventListener('resize', () => { finestra = null; disegna(); });
    win.addEventListener('unload', () => { win = null; });
    disegna();
    return true;
  };

  const chiudi = () => { if (viva()) win.close(); win = null; };

  /* La finestra del ritaglio: larga attorno al riquadro, e ritagliata sui bordi
     dell'immagine **spostandola**, non stringendola - la larghezza deve restare quella che
     il server usera' davvero, se no la scala misurata e quella vera non coincidono. */
  const nuovaFinestra = (box, size) => {
    const [w, h] = size;
    const largo = box.right - box.left;
    const alto = box.bottom - box.top;
    const mx = Math.max(40, Math.round(1.4 * largo));
    const my = Math.max(24, Math.round(1.4 * alto));
    let x0 = Math.round(box.left - mx);
    let x1 = Math.round(box.right + mx);
    let y0 = Math.round(box.top - my);
    let y1 = Math.round(box.bottom + my);
    const lw = Math.min(w, x1 - x0);
    if (x0 < 0) { x0 = 0; x1 = lw; }
    if (x1 > w) { x1 = w; x0 = w - lw; }
    const lh = Math.min(h, y1 - y0);
    if (y0 < 0) { y0 = 0; y1 = lh; }
    if (y1 > h) { y1 = h; y0 = h - lh; }
    return [Math.max(0, x0), Math.max(0, y0), Math.min(w, x1), Math.min(h, y1)];
  };

  let trascinando = false;
  let nodi = [];          // i rettangoli disegnati, in ordine: servono a non rifarli

  /* Il trascinamento dentro alla lente. Il delta si porta in coordinate native dividendo
     per la scala **misurata**, la stessa con cui il rettangolo e' stato disegnato: se si
     usasse lo zoom chiesto al server, contro un bordo il riquadro scapperebbe sotto al
     cursore. Mentre si trascina la finestra del ritaglio non si rifa', se no la striscia
     si sposterebbe sotto le dita. */
  const attaccaTrascinamento = (nodo, lato) => {
    const d = win.document;
    nodo.addEventListener('pointerdown', (ev) => {
      if (!ctx || !ctx.onChange) return;
      ev.preventDefault();
      ev.stopPropagation();
      const img = d.getElementById('crop');
      const s = (img.clientWidth || 1) / Math.max(1, finestra[2] - finestra[0]);
      const partenza = { x: ev.clientX, y: ev.clientY };
      const box = { ...(ctx.boxes[0] || {}).box };
      const [w, h] = ctx.size || [0, 0];
      trascinando = true;
      nodo.setPointerCapture(ev.pointerId);
      const muovi = (e) => {
        const dx = Math.round((e.clientX - partenza.x) / (s || 1));
        const dy = Math.round((e.clientY - partenza.y) / (s || 1));
        const n = { ...box };
        if (lato === 'move') {
          // Il riquadro si sposta intero: i lati si limitano insieme, se no contro un bordo
          // si schiaccerebbe invece di fermarsi.
          const largo = box.right - box.left;
          const alto = box.bottom - box.top;
          n.left = Math.max(0, Math.min(box.left + dx, (w || box.right) - largo));
          n.top = Math.max(0, Math.min(box.top + dy, (h || box.bottom) - alto));
          n.right = n.left + largo;
          n.bottom = n.top + alto;
        } else {
          if (lato.includes('w')) n.left = Math.min(box.left + dx, box.right - 2);
          if (lato.includes('e')) n.right = Math.max(box.right + dx, box.left + 2);
          if (lato.includes('n')) n.top = Math.min(box.top + dy, box.bottom - 2);
          if (lato.includes('s')) n.bottom = Math.max(box.bottom + dy, box.top + 2);
          n.left = Math.max(0, n.left);
          n.top = Math.max(0, n.top);
          if (w) n.right = Math.min(n.right, w);
          if (h) n.bottom = Math.min(n.bottom, h);
        }
        ctx.onChange({ left: Math.round(n.left), top: Math.round(n.top),
                       right: Math.round(n.right), bottom: Math.round(n.bottom) });
      };
      const molla = () => {
        trascinando = false;
        nodo.removeEventListener('pointermove', muovi);
        nodo.removeEventListener('pointerup', molla);
        nodo.removeEventListener('pointercancel', molla);
        disegna();
      };
      nodo.addEventListener('pointermove', muovi);
      nodo.addEventListener('pointerup', molla);
      nodo.addEventListener('pointercancel', molla);
    });
  };

  const fuoriFinestra = (box) => !finestra
    || box.left < finestra[0] + 1 || box.right > finestra[2] - 1
    || box.top < finestra[1] + 1 || box.bottom > finestra[3] - 1;

  const disegna = () => {
    if (!viva() || !ctx) return;
    const d = win.document;
    const img = d.getElementById('crop');
    const scena = d.getElementById('scena');
    const vuoto = d.getElementById('vuoto');
    const principale = (ctx.boxes || []).find((b) => b.box) || null;
    if (!principale) {
      scena.style.display = 'none'; vuoto.style.display = '';
      d.getElementById('titolo').textContent = ctx.name || 'lente';
      d.getElementById('dettaglio').textContent = '';
      return;
    }
    scena.style.display = ''; vuoto.style.display = 'none';
    const size = ctx.size || [img.naturalWidth || 1, img.naturalHeight || 1];

    // Mentre si trascina la finestra resta ferma: rifarla farebbe scappare la striscia
    // sotto al cursore. Fuori dal trascinamento si insegue il riquadro come prima.
    if (!trascinando && (nomeInFinestra !== ctx.name || fuoriFinestra(principale.box))) {
      finestra = nuovaFinestra(principale.box, size);
      nomeInFinestra = ctx.name;
      const largo = Math.max(1, finestra[2] - finestra[0]);
      const alto = Math.max(1, finestra[3] - finestra[1]);
      // Ingrandimento per riempire la finestra, entro il tetto del server.
      const zoom = Math.max(1, Math.min(8, Math.floor(Math.min(
        (win.innerWidth - 8) / largo, (win.innerHeight - 46) / alto))));
      // Il ridisegno dopo il caricamento si aggancia una volta sola, qui: metterlo nel
      // ramo "non ho ancora una larghezza" ne accumulava uno per ogni trascinamento.
      img.onload = () => disegna();
      img.src = `/api/projects/${ctx.projectId}/crop?name=${encodeURIComponent(ctx.name)}`
        + `&raw=1&zoom=${zoom}&x0=${finestra[0]}&y0=${finestra[1]}`
        + `&x1=${finestra[2]}&y1=${finestra[3]}`;
    }

    // La scala si misura: e' l'unico modo di essere sicuri che il rettangolo cada dove deve.
    const larghezzaVista = img.clientWidth || 0;
    if (larghezzaVista < 4) return;   // si ridisegna da solo al `load`
    const s = larghezzaVista / Math.max(1, finestra[2] - finestra[0]);
    const dentro = (ctx.boxes || []).filter((v) => v.box);

    const posiziona = (nodo, box) => {
      nodo.style.left = `${(box.left - finestra[0]) * s}px`;
      nodo.style.top = `${(box.top - finestra[1]) * s}px`;
      nodo.style.width = `${(box.right - box.left) * s}px`;
      nodo.style.height = `${(box.bottom - box.top) * s}px`;
    };

    // Mentre si trascina i rettangoli si spostano, non si rifanno: rifarli butterebbe via
    // il nodo che ha la presa del puntatore, e il trascinamento morirebbe al primo pixel.
    if (trascinando && nodi.length === dentro.length) {
      dentro.forEach((voce, k) => posiziona(nodi[k], voce.box));
      scrividettaglio(d, principale.box);
      return;
    }

    for (const vecchio of [...scena.querySelectorAll('.riq')]) vecchio.remove();
    nodi = [];
    for (const voce of dentro) {
      const primo = voce === principale;
      const n = d.createElement('div');
      n.className = 'riq' + (primo && ctx.onChange ? ' viva' : '');
      n.style.borderColor = voce.color || '#3fb950';
      posiziona(n, voce.box);
      if (voce.label) {
        const b = d.createElement('b');
        b.textContent = voce.label;
        b.style.color = voce.color || '#3fb950';
        n.append(b);
      }
      if (primo && ctx.onChange) {
        attaccaTrascinamento(n, 'move');
        for (const [lato, sx, sy] of [['nw', 0, 0], ['n', 0.5, 0], ['ne', 1, 0],
                                      ['w', 0, 0.5], ['e', 1, 0.5],
                                      ['sw', 0, 1], ['s', 0.5, 1], ['se', 1, 1]]) {
          const m = d.createElement('div');
          m.className = `man ${lato}`;
          m.style.left = `${sx * 100}%`;
          m.style.top = `${sy * 100}%`;
          m.style.background = voce.color || '#3fb950';
          attaccaTrascinamento(m, lato);
          n.append(m);
        }
      }
      scena.append(n);
      nodi.push(n);
    }
    d.getElementById('titolo').textContent = (ctx.name || '').split('/').pop();
    scrividettaglio(d, principale.box);
  };

  const scrividettaglio = (d, b) => {
    d.getElementById('dettaglio').textContent =
      `${b.right - b.left} x ${b.bottom - b.top} px · top ${b.top} left ${b.left} `
      + `bottom ${b.bottom} right ${b.right}`
      + (ctx.caption ? ` · ${ctx.caption}` : '')
      + (ctx.onChange ? ' · trascinabile da qui' : '');
  };

  const aggiorna = (nuovo) => { ctx = nuovo; disegna(); };

  /* Il tasto che la apre. `window.open` vuole un gesto dell'utente: da qui in poi gli
     aggiornamenti arrivano da soli. */
  const bottone = (dammiContesto, etichetta) => {
    const b = el('button', { class: 'ghost' }, etichetta || 'Lente su un\'altra finestra');
    b.addEventListener('click', () => {
      ctx = dammiContesto();
      if (!apri()) {
        toast('il browser ha bloccato la finestra: permetti i popup per questo sito', true);
        return;
      }
      b.textContent = 'Lente aperta — portala sull\'altro schermo';
    });
    return b;
  };

  return { apri, chiudi, aggiorna, viva, bottone };
})();
