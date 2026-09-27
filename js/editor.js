// Alignment editor: place each inner image precisely inside its parent.
// Toggle with E (or the Align button). Works on the pair the camera is between.

const $ = (id) => document.getElementById(id);

export class Editor {
  constructor(app) {
    this.app = app;
    this.on = false;
    this.pairIdx = -1;
    this.drag = null;
    this.panel = $('editor');
    $('btn-editor').addEventListener('click', () => this.toggle());
    this.panel.querySelector('[data-close]').addEventListener('click', () => this.toggle(false));

    const bind = (id, fn) => $(id).addEventListener('input', (e) => { fn(e.target); this.#apply(); });
    bind('ed-x', (t) => (this.#raw().at = [num(t.value, 0.5), this.#raw().at?.[1] ?? 0.5]));
    bind('ed-y', (t) => (this.#raw().at = [this.#raw().at?.[0] ?? 0.5, num(t.value, 0.5)]));
    bind('ed-w', (t) => { const v = Number(t.value); if (v > 0) this.#raw().width = v; });
    bind('ed-r', (t) => (this.#raw().rot = num(t.value, 0)));
    bind('ed-f', (t) => (this.#raw().feather = num(t.value, 0.45)));
    bind('ed-fa', (t) => (this.#raw().fadeIn = [num(t.value, 0), this.#layer().fadeIn[1]]));
    bind('ed-fb', (t) => (this.#raw().fadeIn = [this.#layer().fadeIn[0], num(t.value, 0.2)]));
    bind('ed-cm', (t) => (this.#raw().colorMatch = num(t.value, 0.8)));
    $('ed-ghost').addEventListener('change', () => (app.dirty = true));
    $('ed-export').addEventListener('click', () => this.#export());
    $('ed-copy').addEventListener('click', () => {
      navigator.clipboard?.writeText(JSON.stringify(this.#raw(), null, 2));
    });

    const c = app.canvas;
    c.addEventListener('pointerdown', (e) => {
      if (!this.on || this.pairIdx < 0) return;
      c.setPointerCapture(e.pointerId);
      this.drag = { x: e.clientX, y: e.clientY };
      app.frozenCam = { ...app.cam };
    });
    c.addEventListener('pointermove', (e) => {
      if (!this.drag) return;
      const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y;
      this.drag = { x: e.clientX, y: e.clientY };
      this.#nudge(dx, dy);
    });
    const end = () => {
      if (!this.drag) return;
      this.drag = null;
      app.frozenCam = null;
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    addEventListener('wheel', (e) => {
      if (!this.on || this.pairIdx < 0 || !(e.altKey || e.shiftKey)) return;
      e.preventDefault();
      const raw = this.#raw();
      const d = e.deltaY || e.deltaX;
      if (e.altKey) raw.width *= Math.exp(-d * 0.0015);
      else raw.rot = (raw.rot || 0) + d * 0.05;
      this.#apply();
      this.#fill();
    }, { passive: false });
  }

  toggle(force) {
    this.on = force ?? !this.on;
    this.panel.hidden = !this.on;
    document.body.classList.toggle('editing', this.on);
    this.pairIdx = -1;
    this.app.dirty = true;
  }

  ghostFor(anchor) {
    return this.on && $('ed-ghost').checked && anchor === this.pairIdx - 1;
  }

  #layer() {
    return this.app.scene.layers[this.pairIdx];
  }

  #raw() {
    return this.#layer().raw;
  }

  update(cam) {
    if (!this.on) return;
    const L = this.app.scene.layers;
    const idx = Math.min(L.length - 1, cam.anchor + 1);
    if (idx !== this.pairIdx && !this.drag) {
      this.pairIdx = idx;
      $('ed-pair').textContent = `${L[idx - 1]?.label ?? '—'} → ${L[idx].label}`;
      this.#fill();
    }
  }

  #fill() {
    const l = this.#layer();
    if (!l) return;
    const raw = l.raw;
    $('ed-x').value = (raw.at?.[0] ?? 0.5).toFixed(4);
    $('ed-y').value = (raw.at?.[1] ?? 0.5).toFixed(4);
    $('ed-w').value = Number(raw.width).toExponential(4);
    $('ed-r').value = raw.rot || 0;
    $('ed-f').value = l.feather;
    $('ed-fa').value = l.fadeIn[0];
    $('ed-fb').value = l.fadeIn[1];
    $('ed-cm').value = l.colorMatch;
  }

  #apply() {
    const l = this.#layer();
    const raw = l.raw;
    l.feather = raw.feather ?? l.feather;
    l.fadeIn = raw.fadeIn ?? l.fadeIn;
    l.colorMatch = raw.colorMatch ?? l.colorMatch;
    this.app.relayout();
  }

  // screen drag (CSS px) -> move inner image inside its parent
  #nudge(dx, dy) {
    const app = this.app;
    const cam = app.cam;
    const l = this.#layer();
    const p = app.scene.layers[this.pairIdx - 1];
    if (!p) return;
    const q = [dx / innerWidth, dy / innerWidth];
    const c = Math.cos(cam.rot), s = Math.sin(cam.rot);
    const wx = (c * q[0] - s * q[1]) * cam.Vw, wy = (s * q[0] + c * q[1]) * cam.Vw;
    // camera is anchored in the parent's frame while between these two layers
    const at = l.raw.at || [0.5, 0.5];
    l.raw.at = [at[0] + wx / p.W, at[1] + wy / p.H];
    this.#apply();
    this.#fill();
  }

  #export() {
    const app = this.app;
    const out = { ...app.data, layers: app.scene.layers.map((l) => l.raw) };
    const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'scene.json';
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
}

function num(v, d) {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}
