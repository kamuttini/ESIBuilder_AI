/* La lente: una finestra a parte dove si lavora sull'immagine ingrandita.

   Prima era una finestrella ritagliata attorno al bersaglio: il server mandava una striscia
   di pochi pixel e la vista inseguiva il riquadro o il puntatore. Rispondeva bene a «guarda
   qui», malissimo a «fammi girare per l'immagine»: ogni spostamento chiedeva un ritaglio
   nuovo, lo zoom era limitato a quella striscia, e la vista tornava dove voleva lei.

   Adesso e' un visore dell'immagine **intera**. Si carica una volta a grandezza naturale e
   da li' in poi tutto succede nel browser: pan trascinando, zoom con la rotella sul punto
   sotto al cursore, riquadro che si ridisegna da capo o si aggiusta per i bordi. Nessuna
   richiesta al server mentre si lavora, e nessuna vista che si sposta da sola.

   La geometria e' diretta: un punto nativo (x, y) sta a (x·zoom, y·zoom) dentro la scena.
   Niente finestra di ritaglio da sottrarre, niente scala da misurare a posteriori. */

const Lente = (() => {
  let win = null;
  let ctx = null;         // {projectId, name, size, boxes:[{box,color,label}], ...}
  let zoom = null;        // null = adatta l'immagine intera alla finestra
  let nomeCaricato = '';
  let erroreImmagine = '';
  let latoScelto = null;  // il bordo che le frecce muovono
  let miraScelta = '';    // quale bersaglio della barra e' acceso
  let padrone = '';       // quale sezione la sta usando
  let sospesa = false;    // la sezione e' cambiata e la nuova non si e' ancora fatta viva
  let trascinando = false;
  let nodi = [];          // i rettangoli disegnati: si spostano invece di rifarli
  /* La lente che segue il puntatore dell'altra finestra. Comoda per cercare, ma in mezzo
     mentre si lavora: adesso parte spenta e la si accende quando serve. */
  let segueIlPuntatore = false;
  let seguitoInAttesa = null;
  let frameChiesto = false;

  const NOMI_LATO = { n: 'bordo alto', s: 'bordo basso', w: 'bordo sx', e: 'bordo dx',
                      nw: 'angolo ↖', ne: 'angolo ↗', sw: 'angolo ↙', se: 'angolo ↘',
                      move: 'tutto il riquadro' };

  const ALTEZZA_TESTE = 84;   // barre in cima: quanto spazio tolgono alla vista

  const viva = () => !!(win && !win.closed && win.document && win.document.getElementById('crop'));

  const DOC = `<!doctype html><html lang="it"><head><meta charset="utf-8">
<title>Lente — ESIBuilder AI</title><style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #0d1117; color: #c9d1d9;
         font: 13px ui-monospace, Menlo, monospace; overflow: auto; }
  #barre { position: sticky; top: 0; z-index: 5; background: #0d1117; }
  #testa { padding: 6px 10px; border-bottom: 1px solid #30363d; display: flex;
           gap: 10px; align-items: baseline; justify-content: space-between;
           flex-wrap: wrap; }
  #titolo { font-weight: 600; }
  #dettaglio { color: #8b949e; }
  #scena { position: relative; }
  /* I pixel si vedono: a 800% serve il pixel, non la sfocatura dell'interpolazione. */
  #crop { display: block; image-rendering: pixelated; user-select: none; -webkit-user-drag: none; }
  #zoombar { display: flex; gap: 4px; align-items: center; }
  #mire { display: flex; gap: 3px; flex-wrap: wrap; padding: 5px 10px;
          border-bottom: 1px solid #30363d; }
  #mire button { background: #161b22; color: #8b949e; border: 1px solid #30363d;
                 border-radius: 999px; padding: 1px 9px; font: inherit; font-size: 11.5px;
                 cursor: pointer; }
  #mire button:hover { background: #21262d; color: #c9d1d9; }
  #mire button.on { background: #1f6feb; border-color: #1f6feb; color: #fff; }
  #coord { color: #6e7681; font-variant-numeric: tabular-nums; }
  #scena { cursor: crosshair; }
  #scena.sposta { cursor: grab; }
  #scena.trascina { cursor: grabbing; }
  #zoombar button { background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
                    border-radius: 5px; padding: 1px 8px; font: inherit; cursor: pointer; }
  #zoombar button:hover { background: #30363d; }
  #zoombar button.on { background: #1f6feb; border-color: #1f6feb; color: #fff; }
  #fattore { min-width: 46px; text-align: right; color: #8b949e; }
  .riq { position: absolute; border: 2px solid #3fb950; box-sizing: border-box;
         box-shadow: 0 0 0 1px rgba(0,0,0,.75); pointer-events: none; }
  .riq.viva { pointer-events: auto; cursor: move; }
  .riq.disegno { border: 2px dashed #58a6ff; background: rgba(88,166,255,.12); }
  .man { position: absolute; width: 14px; height: 14px; margin: -7px 0 0 -7px;
         border: 2px solid #0d1117; border-radius: 3px; background: #3fb950;
         pointer-events: auto; }
  .man.n, .man.s { cursor: ns-resize; }
  .man.w, .man.e { cursor: ew-resize; }
  .man.nw, .man.se { cursor: nwse-resize; }
  .man.ne, .man.sw { cursor: nesw-resize; }
  /* I lati si afferrano lungo tutto il bordo, non solo nelle maniglie: a 800% le maniglie
     degli angoli finiscono fuori dalla vista e resterebbe un riquadro che non si prende. */
  .lato { position: absolute; pointer-events: auto; }
  .lato.n, .lato.s { left: 0; right: 0; height: 11px; cursor: ns-resize; }
  .lato.n { top: -5px; } .lato.s { bottom: -5px; }
  .lato.w, .lato.e { top: 0; bottom: 0; width: 11px; cursor: ew-resize; }
  .lato.w { left: -5px; } .lato.e { right: -5px; }
  .lato.scelto { background: rgba(31, 111, 235, .35); }
  .riq b { position: absolute; top: -17px; left: -2px; font-size: 11px; font-weight: 600;
           background: #0d1117; padding: 0 3px; white-space: nowrap; }
  .asse { position: absolute; pointer-events: none; }
  .asse.v { top: 0; bottom: 0; width: 0; border-left: 1px dashed currentColor; }
  .asse.o { left: 0; right: 0; height: 0; border-top: 1px dashed currentColor; }
  .corda { position: absolute; height: 0; border-top: 3px solid currentColor;
           pointer-events: none; }
  .corda.tratteggio { border-top-style: dashed; }
  .corda i, .asse i { position: absolute; top: -16px; left: 0; font-style: normal;
                      font-size: 11px; font-weight: 600; color: #0d1117;
                      background: currentColor; padding: 0 3px; white-space: nowrap; }
  #vuoto { padding: 18px; color: #8b949e; }
  #sospeso { display: none; padding: 7px 10px; background: rgba(210, 153, 34, .14);
             border-bottom: 1px solid #d29922; color: #d29922; font-size: 12px; }
  body.sospesa #scena { opacity: .45; pointer-events: none; }
  body.sospesa #sospeso { display: block; }
</style></head><body>
<div id="barre">
  <div id="sospeso">questa e' la sezione di prima: aprine una che usa la lente, oppure
  torna indietro.</div>
  <div id="mire"></div>
  <div id="testa"><span id="titolo">lente</span>
    <span id="zoombar"><button id="meno" title="rimpicciolisci (rotella o −)">−</button>
      <span id="fattore"></span>
      <button id="piu" title="ingrandisci (rotella o +)">+</button>
      <button id="tutta" title="tutta l'immagine (tasto 0)">tutta</button>
      <button id="vaialbox" title="torna sul riquadro (tasto b)">al riquadro</button>
      <button id="segui" title="segue il puntatore sull'altra finestra (tasto f)">segue</button></span>
    <span id="coord"></span>
    <span id="dettaglio"></span></div>
</div>
<div id="scena"><img id="crop" alt="" draggable="false"></div>
<div id="vuoto">niente da ingrandire: scegli un riquadro nella pagina principale.</div>
</body></html>`;

  const PASSI_ZOOM = [0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32];

  const misura = () => (ctx && ctx.size) || [0, 0];

  /* Lo zoom che fa stare l'immagine intera nella finestra: e' il punto di partenza e quello
     a cui si torna con «tutta». */
  const zoomTutta = () => {
    const [W, H] = misura();
    if (!viva() || !W || !H) return 1;
    return Math.max(0.05, Math.min(
      (win.innerWidth - 4) / W, (win.innerHeight - ALTEZZA_TESTE) / H));
  };

  const fattore = () => zoom || zoomTutta();

  const scena = () => win.document.getElementById('scena');
  const immagine = () => win.document.getElementById('crop');

  /* Il punto nativo sotto a un evento del puntatore. */
  const puntoNativo = (ev) => {
    if (!viva()) return null;
    const r = immagine().getBoundingClientRect();
    const s = fattore();
    if (!s) return null;
    return { x: (ev.clientX - r.left) / s, y: (ev.clientY - r.top) / s };
  };

  /* Porta un punto nativo sotto a una posizione della finestra, scorrendo. */
  const portaSotto = (nat, clientX, clientY) => {
    if (!viva()) return;
    const r = immagine().getBoundingClientRect();
    const s = fattore();
    win.scrollBy(Math.round(r.left + nat.x * s - clientX),
                 Math.round(r.top + nat.y * s - clientY));
  };

  const centraSu = (box) => {
    if (!viva() || !box) return;
    const s = fattore();
    const cx = ((box.left + box.right) / 2) * s;
    const cy = ((box.top + box.bottom) / 2) * s;
    win.scrollTo(Math.max(0, cx - win.innerWidth / 2),
                 Math.max(0, cy - (win.innerHeight - ALTEZZA_TESTE) / 2));
  };

  /* Quanto ingrandire per vedere bene un riquadro: sta comodo in circa un terzo della
     finestra, senza esagerare su un riquadro gia' grande. */
  const zoomPerBox = (box) => {
    if (!viva() || !box) return 4;
    const largo = Math.max(4, box.right - box.left);
    const alto = Math.max(4, box.bottom - box.top);
    return Math.max(0.2, Math.min(16,
      Math.min((win.innerWidth * 0.55) / largo, (win.innerHeight * 0.45) / alto)));
  };

  const vaiAlRiquadro = () => {
    const principale = ((ctx && ctx.boxes) || []).find((b) => b.box);
    const mira = (principale && principale.box) || (ctx && ctx.focus);
    if (!mira) return;
    zoom = zoomPerBox(mira);
    disegna();
    centraSu(mira);
  };

  const cambiaZoom = (verso, ev) => {
    const ora = fattore();
    const nat = ev ? puntoNativo(ev) : null;
    const clientX = ev ? ev.clientX : (viva() ? win.innerWidth / 2 : 0);
    const clientY = ev ? ev.clientY : (viva() ? win.innerHeight / 2 : 0);
    const vicino = PASSI_ZOOM.reduce((a, b) => (Math.abs(b - ora) < Math.abs(a - ora) ? b : a));
    const k = PASSI_ZOOM.indexOf(vicino);
    // Fra due passi, il primo scatto porta sul passo dalla parte giusta invece di saltarlo.
    let prossimo = k + verso;
    if (verso > 0 && vicino > ora) prossimo = k;
    if (verso < 0 && vicino < ora) prossimo = k;
    zoom = PASSI_ZOOM[Math.max(0, Math.min(PASSI_ZOOM.length - 1, prossimo))];
    const centro = nat || (() => {
      // Senza puntatore si tiene fermo il centro della vista.
      const r = immagine().getBoundingClientRect();
      const s = ora;
      return { x: (win.innerWidth / 2 - r.left) / s, y: (win.innerHeight / 2 - r.top) / s };
    })();
    disegna();
    portaSotto(centro, clientX, clientY);
  };

  /* Spostare la vista trascinando lo sfondo, col tasto centrale o con alt. Il tasto
     sinistro serve a disegnare: e' il gesto che si fa cento volte. */
  const attaccaSpostamento = (nodo) => {
    nodo.addEventListener('pointerdown', (ev) => {
      if (!viva() || !ctx) return;
      if (ev.target !== nodo && ev.target.id !== 'crop') return;
      const spostaComunque = ev.button === 1 || ev.altKey || ev.shiftKey;
      const puoDisegnare = !!(ctx.onDraw || ctx.onChange) && !sospesa;
      ev.preventDefault();
      if (ev.button === 0 && puoDisegnare && !spostaComunque) {
        disegnaNuovo(nodo, ev);
        return;
      }
      if (ev.button === 0 && !puoDisegnare && ctx.onPunto && !spostaComunque) {
        indicaPunto(nodo, ev);
        return;
      }
      const da = { x: ev.clientX, y: ev.clientY };
      nodo.classList.add('trascina');
      try { nodo.setPointerCapture(ev.pointerId); } catch (_) { /* pazienza */ }
      const muovi = (e) => {
        win.scrollBy(da.x - e.clientX, da.y - e.clientY);
        da.x = e.clientX;
        da.y = e.clientY;
      };
      const molla = () => {
        nodo.classList.remove('trascina');
        nodo.removeEventListener('pointermove', muovi);
        nodo.removeEventListener('pointerup', molla);
        nodo.removeEventListener('pointercancel', molla);
      };
      nodo.addEventListener('pointermove', muovi);
      nodo.addEventListener('pointerup', molla);
      nodo.addEventListener('pointercancel', molla);
    });
  };

  /* Un clic che indica un punto (lo zero del righello, una tacca): vale solo dove la
     sezione lo chiede, e solo se il puntatore non si e' mosso. */
  const indicaPunto = (nodo, ev) => {
    const partenza = { x: ev.clientX, y: ev.clientY };
    let mosso = false;
    const muovi = (e) => {
      if (Math.abs(e.clientX - partenza.x) > 2 || Math.abs(e.clientY - partenza.y) > 2) {
        mosso = true;
        win.scrollBy(partenza.x - e.clientX, partenza.y - e.clientY);
        partenza.x = e.clientX;
        partenza.y = e.clientY;
      }
    };
    const molla = (e) => {
      nodo.removeEventListener('pointermove', muovi);
      nodo.removeEventListener('pointerup', molla);
      nodo.removeEventListener('pointercancel', molla);
      if (mosso || !e || e.type !== 'pointerup' || !ctx || !ctx.onPunto) return;
      const nat = puntoNativo(e);
      if (!nat) return;
      ctx.onPunto({ x: Math.round(nat.x), y: Math.round(nat.y),
                    alt: !!e.altKey, shift: !!e.shiftKey, target: miraScelta });
    };
    nodo.addEventListener('pointermove', muovi);
    nodo.addEventListener('pointerup', molla);
    nodo.addEventListener('pointercancel', molla);
  };

  /* Ridisegnare il riquadro da capo: si tira un rettangolo sull'immagine e quello diventa
     il box. E' il gesto piu' veloce per rimettere a posto un box sbagliato di molto -
     spostare quattro bordi uno a uno, no. */
  const disegnaNuovo = (nodoScena, ev) => {
    const d = win.document;
    const partenza = puntoNativo(ev);
    if (!partenza) return;
    let ultimo = partenza;
    const nodo = d.createElement('div');
    nodo.className = 'riq disegno';
    nodoScena.append(nodo);
    const posa = () => {
      const s = fattore();
      nodo.style.left = `${Math.min(partenza.x, ultimo.x) * s}px`;
      nodo.style.top = `${Math.min(partenza.y, ultimo.y) * s}px`;
      nodo.style.width = `${Math.abs(ultimo.x - partenza.x) * s}px`;
      nodo.style.height = `${Math.abs(ultimo.y - partenza.y) * s}px`;
    };
    posa();
    try { nodoScena.setPointerCapture(ev.pointerId); } catch (_) { /* pazienza */ }
    const muovi = (e) => { ultimo = puntoNativo(e) || ultimo; posa(); };
    const molla = () => {
      nodoScena.removeEventListener('pointermove', muovi);
      nodoScena.removeEventListener('pointerup', molla);
      nodoScena.removeEventListener('pointercancel', molla);
      nodo.remove();
      const [W, H] = misura();
      const box = {
        left: Math.round(Math.max(0, Math.min(partenza.x, ultimo.x))),
        right: Math.round(Math.min(W || 1e9, Math.max(partenza.x, ultimo.x))),
        top: Math.round(Math.max(0, Math.min(partenza.y, ultimo.y))),
        bottom: Math.round(Math.min(H || 1e9, Math.max(partenza.y, ultimo.y))),
      };
      // Sotto i tre pixel e' un clic, non un rettangolo: cancellare il box per un clic di
      // troppo sarebbe il modo piu' veloce di perdere il lavoro.
      if (box.right - box.left < 3 || box.bottom - box.top < 3) return;
      if (ctx && ctx.onDraw) ctx.onDraw(box);
      else if (ctx && ctx.onChange) ctx.onChange(box);
    };
    nodoScena.addEventListener('pointermove', muovi);
    nodoScena.addEventListener('pointerup', molla);
    nodoScena.addEventListener('pointercancel', molla);
  };

  const attaccaCoordinate = (nodo) => {
    nodo.addEventListener('pointermove', (ev) => {
      if (!viva()) return;
      const nat = puntoNativo(ev);
      if (!nat) return;
      win.document.getElementById('coord').textContent =
        `x ${Math.round(nat.x)} · y ${Math.round(nat.y)}`;
    });
    nodo.addEventListener('pointerleave', () => {
      if (viva()) win.document.getElementById('coord').textContent = '';
    });
  };

  const commutaSegui = () => {
    segueIlPuntatore = !segueIlPuntatore;
    pittaSegui();
  };

  const pittaSegui = () => {
    if (!viva()) return;
    const b = win.document.getElementById('segui');
    if (!b) return;
    b.className = segueIlPuntatore ? 'on' : '';
    b.textContent = segueIlPuntatore ? 'segue' : 'ferma';
  };

  /* Le frecce: muovono il lato scelto di un pixel (shift dieci), oppure scorrono la vista
     quando non c'e' nessun lato scelto. */
  const attaccaTasti = (d) => {
    d.addEventListener('keydown', (ev) => {
      if (!viva()) return;
      if (ev.key === '+' || ev.key === '=') { ev.preventDefault(); cambiaZoom(+1); return; }
      if (ev.key === '-' || ev.key === '_') { ev.preventDefault(); cambiaZoom(-1); return; }
      if (ev.key === '0') { ev.preventDefault(); zoom = null; disegna(); win.scrollTo(0, 0); return; }
      if (ev.key === 'b' || ev.key === 'B') { ev.preventDefault(); vaiAlRiquadro(); return; }
      if (ev.key === 'f' || ev.key === 'F') { ev.preventDefault(); commutaSegui(); return; }
      const passo = ev.shiftKey ? 10 : 1;
      const delta = { ArrowLeft: [-passo, 0], ArrowRight: [passo, 0],
                      ArrowUp: [0, -passo], ArrowDown: [0, passo] }[ev.key];
      if (!delta) return;
      ev.preventDefault();
      if (!ctx || !ctx.onChange || !latoScelto || sospesa) {
        win.scrollBy(delta[0] * 12, delta[1] * 12);
        return;
      }
      const box = { ...((ctx.boxes[0] || {}).box) };
      const [W, H] = misura();
      const n = { ...box };
      if (latoScelto === 'move') {
        const largo = box.right - box.left, alto = box.bottom - box.top;
        n.left = Math.max(0, Math.min(box.left + delta[0], (W || box.right) - largo));
        n.top = Math.max(0, Math.min(box.top + delta[1], (H || box.bottom) - alto));
        n.right = n.left + largo; n.bottom = n.top + alto;
      } else {
        if (latoScelto.includes('w')) n.left = Math.min(box.left + delta[0], box.right - 2);
        if (latoScelto.includes('e')) n.right = Math.max(box.right + delta[0], box.left + 2);
        if (latoScelto.includes('n')) n.top = Math.min(box.top + delta[1], box.bottom - 2);
        if (latoScelto.includes('s')) n.bottom = Math.max(box.bottom + delta[1], box.top + 2);
        n.left = Math.max(0, n.left); n.top = Math.max(0, n.top);
        if (W) n.right = Math.min(n.right, W);
        if (H) n.bottom = Math.min(n.bottom, H);
      }
      ctx.onChange({ left: Math.round(n.left), top: Math.round(n.top),
                     right: Math.round(n.right), bottom: Math.round(n.bottom) });
    });
  };

  const apri = () => {
    if (viva()) { win.focus(); return true; }
    win = window.open('', 'esibuilder-lente',
      'width=1120,height=820,menubar=no,toolbar=no,location=no,status=no');
    if (!win) return false;
    win.document.open();
    win.document.write(DOC);
    win.document.close();
    nomeCaricato = '';
    erroreImmagine = '';
    win.addEventListener('resize', () => { if (!zoom) disegna(); });
    win.addEventListener('unload', () => { win = null; sincronizzaBottoni(); });
    const d = win.document;
    const sc = d.getElementById('scena');
    attaccaSpostamento(sc);
    attaccaCoordinate(sc);
    attaccaTasti(d);
    d.body.tabIndex = 0;
    d.body.focus();
    d.getElementById('piu').addEventListener('click', () => cambiaZoom(+1));
    d.getElementById('meno').addEventListener('click', () => cambiaZoom(-1));
    d.getElementById('tutta').addEventListener('click', () => {
      zoom = null; disegna(); win.scrollTo(0, 0);
    });
    d.getElementById('vaialbox').addEventListener('click', vaiAlRiquadro);
    pittaSegui();
    d.getElementById('segui').addEventListener('click', commutaSegui);
    // La rotella ingrandisce sul punto sotto al cursore; con shift resta lo scorrimento.
    d.addEventListener('wheel', (ev) => {
      if (ev.shiftKey) return;
      ev.preventDefault();
      cambiaZoom(ev.deltaY < 0 ? +1 : -1, ev);
    }, { passive: false });
    disegna();
    return true;
  };

  const chiudi = () => { if (viva()) win.close(); win = null; };

  /* Il trascinamento del riquadro e dei suoi bordi. La scala e' lo zoom: non c'e' piu'
     niente da misurare a posteriori. */
  const attaccaTrascinamento = (nodo, lato) => {
    nodo.addEventListener('pointerdown', (ev) => {
      if (!ctx || !ctx.onChange || sospesa) return;
      ev.preventDefault();
      ev.stopPropagation();
      const s = fattore();
      const partenza = { x: ev.clientX, y: ev.clientY };
      const box = { ...((ctx.boxes[0] || {}).box) };
      const [W, H] = misura();
      trascinando = true;
      latoScelto = lato;
      try { nodo.setPointerCapture(ev.pointerId); } catch (_) { /* pazienza */ }
      const muovi = (e) => {
        const dx = Math.round((e.clientX - partenza.x) / (s || 1));
        const dy = Math.round((e.clientY - partenza.y) / (s || 1));
        const n = { ...box };
        if (lato === 'move') {
          const largo = box.right - box.left;
          const alto = box.bottom - box.top;
          n.left = Math.max(0, Math.min(box.left + dx, (W || box.right) - largo));
          n.top = Math.max(0, Math.min(box.top + dy, (H || box.bottom) - alto));
          n.right = n.left + largo;
          n.bottom = n.top + alto;
        } else {
          if (lato.includes('w')) n.left = Math.min(box.left + dx, box.right - 2);
          if (lato.includes('e')) n.right = Math.max(box.right + dx, box.left + 2);
          if (lato.includes('n')) n.top = Math.min(box.top + dy, box.bottom - 2);
          if (lato.includes('s')) n.bottom = Math.max(box.bottom + dy, box.top + 2);
          n.left = Math.max(0, n.left);
          n.top = Math.max(0, n.top);
          if (W) n.right = Math.min(n.right, W);
          if (H) n.bottom = Math.min(n.bottom, H);
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

  const disegna = () => {
    if (!viva() || !ctx) return;
    const d = win.document;
    const img = immagine();
    const sc = scena();
    const vuoto = d.getElementById('vuoto');

    if (!ctx.name || !ctx.projectId) {
      sc.style.display = 'none';
      vuoto.style.display = '';
      vuoto.textContent = 'niente da ingrandire: scegli un riquadro nella pagina principale.';
      d.getElementById('titolo').textContent = ctx.name || 'lente';
      return;
    }
    if (erroreImmagine && erroreImmagine === ctx.name) {
      sc.style.display = 'none';
      vuoto.style.display = '';
      vuoto.textContent = `non riesco a caricare ${erroreImmagine.split('/').pop()}: `
        + 'l\'immagine non e\' raggiungibile (disco scollegato o cartella spostata).';
      return;
    }
    sc.style.display = '';
    vuoto.style.display = 'none';

    // L'immagine si carica una volta sola, a grandezza naturale: da li' in poi lo zoom e'
    // solo CSS e non si tocca piu' la rete.
    if (nomeCaricato !== ctx.name) {
      nomeCaricato = ctx.name;
      erroreImmagine = '';
      const [W] = misura();
      img.onload = () => { erroreImmagine = ''; disegna(); };
      img.onerror = () => { erroreImmagine = ctx.name || '?'; disegna(); };
      img.src = `/api/projects/${ctx.projectId}/image?name=${encodeURIComponent(ctx.name)}`
        + `&w=${Math.max(320, W || 1920)}`;
    }

    const s = fattore();
    const [W, H] = misura();
    img.style.width = `${Math.round((W || img.naturalWidth || 1) * s)}px`;
    sc.style.width = img.style.width;
    sc.style.height = `${Math.round((H || img.naturalHeight || 1) * s)}px`;
    d.getElementById('fattore').textContent = `${Math.round(s * 100)}%`;
    d.getElementById('titolo').textContent = (ctx.name || '').split('/').pop();
    disegnaMire(d);

    const principale = (ctx.boxes || []).find((b) => b.box) || null;
    const dentro = (ctx.boxes || []).filter((v) => v.box);
    const posiziona = (nodo, box) => {
      nodo.style.left = `${box.left * s}px`;
      nodo.style.top = `${box.top * s}px`;
      nodo.style.width = `${(box.right - box.left) * s}px`;
      nodo.style.height = `${(box.bottom - box.top) * s}px`;
    };

    // Mentre si trascina i rettangoli si spostano, non si rifanno: rifarli butterebbe via
    // il nodo che ha la presa del puntatore e il trascinamento morirebbe al primo pixel.
    if (trascinando && nodi.length === dentro.length) {
      dentro.forEach((voce, k) => posiziona(nodi[k], voce.box));
      if (principale) scrividettaglio(d, principale.box);
      return;
    }

    for (const vecchio of [...sc.querySelectorAll('.riq, .asse, .corda')]) vecchio.remove();
    nodi = [];

    for (const linea of (ctx.lines || [])) {
      const n = d.createElement('div');
      const verticale = linea.x != null;
      n.className = `asse ${verticale ? 'v' : 'o'}`;
      n.style.color = linea.color || '#3fb950';
      if (verticale) n.style.left = `${linea.x * s}px`;
      else n.style.top = `${linea.y * s}px`;
      if (linea.label) {
        const b = d.createElement('i');
        b.textContent = linea.label;
        n.append(b);
      }
      sc.append(n);
    }
    for (const seg of (ctx.segments || [])) {
      const n = d.createElement('div');
      n.className = 'corda' + (seg.dashed ? ' tratteggio' : '');
      n.style.color = seg.color || '#ff6040';
      n.style.left = `${seg.x1 * s}px`;
      n.style.top = `${seg.y * s}px`;
      n.style.width = `${(seg.x2 - seg.x1) * s}px`;
      if (seg.label) {
        const b = d.createElement('i');
        b.textContent = seg.label;
        n.append(b);
      }
      sc.append(n);
    }
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
        for (const lato of ['n', 's', 'w', 'e']) {
          const striscia = d.createElement('div');
          striscia.className = `lato ${lato}` + (latoScelto === lato ? ' scelto' : '');
          attaccaTrascinamento(striscia, lato);
          n.append(striscia);
        }
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
      sc.append(n);
      nodi.push(n);
    }
    if (principale) scrividettaglio(d, principale.box);
    else {
      // Senza riquadro (lo studio del righello, per esempio) il cartiglio dice comunque
      // cosa fa il clic: restare con la scritta della sezione di prima confonde e basta.
      d.getElementById('dettaglio').textContent = (ctx.caption || '')
        + (ctx.onPunto ? ` · clic: ${ctx.puntoLabel || 'indica il punto'}` : '')
        + ' · rotella: zoom · trascina: sposta · 0: tutta l\'immagine';
    }
  };

  /* I bersagli: un clic e la lente va li'. */
  const disegnaMire = (d) => {
    const barra = d.getElementById('mire');
    const elenco = (ctx && ctx.targets) || [];
    if (!elenco.length) {
      barra.style.display = 'none';
      barra.innerHTML = '';
      barra.dataset.firma = '';
      return;
    }
    barra.style.display = '';
    const firma = elenco.map((v) => v.id).join('|') + '#' + miraScelta;
    if (barra.dataset.firma === firma) return;
    barra.dataset.firma = firma;
    barra.innerHTML = '';
    for (const voce of elenco) {
      const b = d.createElement('button');
      b.textContent = voce.label;
      if (voce.id === miraScelta) b.className = 'on';
      b.addEventListener('click', () => {
        miraScelta = voce.id;
        latoScelto = voce.side || null;
        if (ctx.onTarget) ctx.onTarget(voce);
        // Il fuoco resterebbe sul chip, e le frecce scorrerebbero invece di muovere il lato.
        d.body.focus();
        if (voce.box) { zoom = zoomPerBox(voce.box); disegna(); centraSu(voce.box); }
        else disegna();
      });
      barra.append(b);
    }
  };

  const scrividettaglio = (d, b) => {
    if (!b) return;
    d.getElementById('dettaglio').textContent =
      `${b.right - b.left} x ${b.bottom - b.top} px · top ${b.top} left ${b.left} `
      + `bottom ${b.bottom} right ${b.right}`
      + (ctx.caption ? ` · ${ctx.caption}` : '')
      + (ctx.onChange || ctx.onDraw ? ' · trascina sull\'immagine: ridisegna il riquadro' : '')
      + (ctx.onChange ? ' · i bordi e le maniglie lo aggiustano' : '')
      + (latoScelto ? ` · frecce: ${NOMI_LATO[latoScelto] || latoScelto} (shift = 10 px)` : '')
      + (ctx.onPunto ? ` · clic: ${ctx.puntoLabel || 'indica il punto'}` : '')
      + ' · rotella: zoom · tasto centrale, alt o shift: sposta · 0: tutta · b: al riquadro';
  };

  /* Il contesto arriva da una sezione, e ogni sezione ha il suo `source`. Cambiandolo si
     riparte: l'immagine e' un'altra, il bersaglio acceso non esiste piu', il lato nemmeno.
     Restando nella stessa sezione invece non si tocca **niente** della vista: lo zoom e la
     posizione sono dell'utente, e rubarglieli a ogni aggiornamento era il motivo per cui
     la lente sembrava muoversi da sola. */
  const aggiorna = (nuovo, riparti) => {
    const chi = (nuovo && nuovo.source) || '';
    const cambiaSezione = riparti || chi !== padrone;
    const primaImmagine = ctx ? ctx.name : '';
    ctx = nuovo;
    if (cambiaSezione) {
      padrone = chi;
      miraScelta = '';
      latoScelto = null;
      nodi = [];
      zoom = null;
      const d = viva() ? win.document : null;
      if (d) d.getElementById('mire').dataset.firma = '';
    }
    sospesa = false;
    if (viva()) win.document.body.classList.remove('sospesa');
    disegna();
    // Entrando in una sezione nuova (o su un'altra immagine) si parte inquadrando il
    // riquadro: e' quello che si e' venuti a guardare. Da li' in poi comanda l'utente.
    if (viva() && ctx && (cambiaSezione || (ctx.name && ctx.name !== primaImmagine))) {
      const img = immagine();
      const parti = () => vaiAlRiquadro();
      if (img.complete && img.naturalWidth) parti();
      else img.addEventListener('load', parti, { once: true });
    }
  };

  const attiva = (source) => viva() && !sospesa && ((source || '') === padrone);
  const aggiornaSeAttiva = (nuovo) => {
    if (!nuovo || !attiva(nuovo.source)) return false;
    aggiorna(nuovo);
    return true;
  };

  const sospendi = () => {
    if (!ctx) return;
    sospesa = true;
    if (viva()) win.document.body.classList.add('sospesa');
  };

  /* La sezione nuova si presenta disegnando il suo pulsante, e la lente passa a lei. */
  const offri = (dammiContesto) => {
    if (!viva() || !sospesa || typeof dammiContesto !== 'function') return;
    let nuovo = null;
    try {
      nuovo = dammiContesto();
    } catch (_) {
      return;
    }
    if (!nuovo) return;
    aggiorna(nuovo, ((nuovo.source || '') !== padrone));
  };

  const bottoni = [];
  const sincronizzaBottoni = () => {
    const aperta = viva();
    for (const [b, testo] of bottoni) {
      if (!b.isConnected) continue;
      b.textContent = aperta ? 'Lente aperta — portala sull\'altro schermo' : testo;
    }
  };

  const bottone = (dammiContesto, etichetta) => {
    const testo = etichetta || 'Lente su un\'altra finestra';
    const b = el('button', { class: 'ghost' }, testo);
    offri(dammiContesto);
    b.addEventListener('click', () => {
      const nuovo = dammiContesto();
      const eraViva = viva();
      const riparti = !eraViva || sospesa || ((nuovo && nuovo.source) || '') !== padrone;
      if (!eraViva) aggiorna(nuovo, true);
      if (!apri()) {
        toast('il browser ha bloccato la finestra: permetti i popup per questo sito', true);
        return;
      }
      if (eraViva) aggiorna(nuovo, riparti);
      win.focus();
      sincronizzaBottoni();
    });
    bottoni.push([b, testo]);
    if (bottoni.length > 24) {
      const vivi = bottoni.filter(([n]) => n.isConnected);
      bottoni.length = 0;
      bottoni.push(...vivi);
    }
    return b;
  };

  /* Il puntatore si muove sull'altra finestra: qui si scorre, ma solo se l'inseguimento
     e' acceso. A passo di frame, se no si scorre cento volte al secondo. */
  const segui = (x, y, source) => {
    if (!attiva(source) || !segueIlPuntatore) return;
    seguitoInAttesa = { x: Math.round(x), y: Math.round(y) };
    if (frameChiesto) return;
    frameChiesto = true;
    requestAnimationFrame(() => {
      frameChiesto = false;
      const punto = seguitoInAttesa;
      seguitoInAttesa = null;
      if (!punto || !viva() || !segueIlPuntatore) return;
      centraSu({ left: punto.x, right: punto.x, top: punto.y, bottom: punto.y });
    });
  };
  const segueOra = () => segueIlPuntatore;

  return {
    apri, chiudi, aggiorna, aggiornaSeAttiva, attiva, sospendi, offri, viva, bottone,
    segui, segueOra,
  };
})();
