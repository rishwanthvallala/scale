import { Scene, CameraPath, Sim, clamp, smoothstep } from './scene.js';
import { Renderer } from './renderer.js';
import { TextureStore, regionStats } from './textures.js';
import { formatLength, powerOfTen, niceLength, shortUnit } from './units.js';
import { Editor } from './editor.js';

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);

const IDENT_STATS = { mean: [0, 0, 0], std: [1, 1, 1] };

class App {
  constructor(data) {
    this.data = data;
    this.scene = new Scene(data);
    this.path = new CameraPath(this.scene);
    this.canvas = $('gl');
    this.renderer = new Renderer(this.canvas);
    const r = this.renderer;
    // phones/tablets get 2K textures: 4K layers cost ~90 MB of GPU memory each
    const lowMem = navigator.deviceMemory && navigator.deviceMemory <= 4;
    const phone = matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) < 900;
    this.small = params.has('lite') || (!params.has('full') && (r.maxTexture < 4096 || lowMem || phone));
    this.store = new TextureStore(r.gl, { useSmall: this.small, maxLive: this.small ? 10 : 7, onChange: () => (this.dirty = true) });
    this.statsCache = new Map();
    this.labelEls = new Map();
    this.sSmooth = 0;
    this.lastLg = null;
    this.lgVel = 0;
    this.time = 0;
    this.dirty = true;
    this.frozenCam = null;
    this.debug = params.has('debug');
    this.#buildStars();
    this.#buildRuler();
    this.#buildCredits();
    this.editor = new Editor(this);
    this.resize();
    addEventListener('resize', () => this.resize());
    addEventListener('scroll', () => (this.dirty = true), { passive: true });
    addEventListener('keydown', (e) => {
      if (e.target.closest('input, textarea, select') || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === 'e' || e.key === 'E') this.editor.toggle();
      // Space plays/pauses, unless a button has keyboard focus (then Space presses that button)
      if (e.key === ' ' && !e.target.closest('button')) { e.preventDefault(); this.setPlaying(!this.playing); }
      else if (['ArrowDown', 'ArrowUp', 'PageDown', 'PageUp', 'Home', 'End'].includes(e.key)) this.setPlaying(false);
    });
    for (const ev of ['wheel', 'touchstart']) addEventListener(ev, () => this.setPlaying(false), { passive: true });
    $('btn-play').addEventListener('click', () => this.setPlaying(!this.playing));
    // mouse clicks shouldn't leave focus on a button (keyboard focus still works normally)
    document.addEventListener('click', (e) => { const b = e.target.closest('button'); if (b && e.detail > 0) b.blur(); });
    // deep links: ?at=<layer id>[&f=0..1 toward the next layer] or ?s=<scroll units>
    const at = params.get('at') && this.scene.byId.get(params.get('at'));
    // add &pin to hold the view there (used for screenshots)
    let s0 = null;
    if (at) {
      const a = this.path.scrollForLayer(at.index);
      const b = this.path.knotsS[Math.min(this.path.knotsS.length - 1, at.index + 2)];
      s0 = a + (b - a) * Number(params.get('f') || 0);
    } else if (params.has('s')) {
      s0 = Number(params.get('s'));
    }
    if (s0 != null) {
      if (params.has('pin')) this.pinnedS = s0;
      else scrollTo(0, s0 * this.unitPx);
    }
    this.sSmooth = this.targetS();
    requestAnimationFrame((t) => this.frame(t));
    // warm the HTTP cache in zoom order after the first frames
    setTimeout(() => this.store.prefetchAll([...this.scene.layers, ...this.scene.extras.filter((e) => e.src)]), 1500);
  }

  // ------------------------------------------------------------------ layout
  resize() {
    const dpr = Math.min(2, devicePixelRatio || 1);
    const w = Math.round(innerWidth * dpr), h = Math.round(innerHeight * dpr);
    const s = this.path.length ? this.targetS() : 0;
    this.renderer.resize(w, h);
    this.dpr = dpr;
    this.aspect = w / h;
    this.path.layout(this.aspect);
    this.unitPx = Math.max(380, innerHeight * 0.75);
    $('scroller').style.height = `${Math.ceil(this.path.length * this.unitPx + innerHeight)}px`;
    if (s && this.pinnedS == null) scrollTo({ top: s * this.unitPx });
    this.#layoutRuler();
    this.dirty = true;
  }

  setPlaying(on) {
    if (on === this.playing) return;
    this.playing = on;
    if (on && this.targetS() >= this.path.length - 0.01) scrollTo({ top: 0 });
    this.playS = this.targetS();
    const b = $('btn-play');
    b.textContent = on ? 'Pause' : 'Play';
    b.setAttribute('aria-pressed', String(on));
  }

  targetS() {
    if (this.pinnedS != null) return this.pinnedS;
    return clamp(scrollY / this.unitPx, 0, this.path.length);
  }

  relayout() {
    this.scene.rebuild();
    this.path.layout(this.aspect);
    this.statsCache.clear();
    this.dirty = true;
  }

  // ------------------------------------------------------------------ frame
  frame(now) {
    const dt = this.lastNow ? Math.min(0.1, (now - this.lastNow) / 1000) : 1 / 60;
    this.lastNow = now;
    this.time += dt;
    if (this.playing) {
      // a steady, video-like fall; the scroll position follows so the user can take over anytime
      this.playS = Math.min(this.path.length, this.playS + dt * 0.42);
      scrollTo(0, this.playS * this.unitPx);
      if (this.playS >= this.path.length) this.setPlaying(false);
    }
    const target = this.playing ? this.playS : this.targetS();
    const k = 1 - Math.exp(-dt * 6.5);
    this.sSmooth += (target - this.sSmooth) * k;
    if (Math.abs(target - this.sSmooth) < 1e-5) this.sSmooth = target;

    const lg = this.path.logAt(this.sSmooth);
    const v = this.lastLg == null ? 0 : (lg - this.lastLg) / dt;
    this.lastLg = lg;
    this.lgVel += (v - this.lgVel) * (1 - Math.exp(-dt * 10));

    const cam = this.frozenCam || this.path.cameraAt(lg);
    this.cam = cam;
    this.#request(cam);
    // skip redundant frames when nothing moves (saves battery; grain simply holds still)
    const key = `${cam.anchor}|${cam.lg}|${cam.cx}|${cam.cy}|${cam.rot}`;
    if (this.dirty || key !== this.lastKey || Math.abs(this.lgVel) > 1e-3) {
      this.dirty = false;
      this.lastKey = key;
      this.#render(cam);
      this.#hud(cam);
    }
    this.editor.update(cam);
    requestAnimationFrame((t) => this.frame(t));
  }

  // which textures we need, nearest first
  #request(cam) {
    const L = this.scene.layers;
    const i = cam.anchor;
    const wanted = [];
    const order = [0, 1, -1, 2, 3, -2, 4];
    order.forEach((d, rank) => {
      const j = i + d;
      if (j >= 0 && j < L.length && L[j].src) wanted.push({ item: L[j], priority: rank });
    });
    for (const e of this.scene.extras) {
      if (e.kind === 'image' && e.src && Math.abs(e.frame - i) <= 2) wanted.push({ item: e, priority: 2.5 + Math.abs(e.frame - i) });
    }
    this.store.want(wanted);
    const loading = this.store.pending();
    $('loading').classList.toggle('on', loading > 0);
    $('loading').firstElementChild.style.width = loading ? `${100 / (1 + loading)}%` : '100%';
  }

  // camera helpers --------------------------------------------------------
  #proj(cam) {
    const cr = Math.cos(-cam.rot), sr = Math.sin(-cam.rot);
    const a = this.aspect, Vw = cam.Vw;
    return (x, y) => {
      const dx = x - cam.cx, dy = y - cam.cy;
      const qx = (cr * dx - sr * dy) / Vw;
      const qy = (sr * dx + cr * dy) / Vw;
      return [2 * qx, -2 * qy * a];
    };
  }

  ndcToPx(n) {
    return [((n[0] + 1) / 2) * this.canvas.width, ((n[1] + 1) / 2) * this.canvas.height];
  }

  // Screen transform for an image of size W×H (meters) whose frame is M (frame -> anchor)
  #affine(cam, M, W, H) {
    const a = this.aspect;
    const ang = Sim.angle(M) - cam.rot;
    const c = Math.cos(ang), s = Math.sin(ang);
    const Vw = cam.Vw;
    const a00 = (2 * c * W) / Vw, a01 = (-2 * s * H) / Vw;
    const a10 = (-2 * a * s * W) / Vw, a11 = (-2 * a * c * H) / Vw;
    const b = this.#proj(cam)(M.x, M.y);
    const det = a00 * a11 - a01 * a10;
    const i00 = a11 / det, i01 = -a01 / det, i10 = -a10 / det, i11 = a00 / det;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [u, w] of [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]) {
      const x = a00 * u + a01 * w + b[0], y = a10 * u + a11 * w + b[1];
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
    const rect = [Math.max(-1, x0), Math.max(-1, y0), Math.min(1, x1), Math.min(1, y1)];
    const visible = rect[0] < rect[2] && rect[1] < rect[3];
    const inv = (nx, ny) => [i00 * (nx - b[0]) + i01 * (ny - b[1]), i10 * (nx - b[0]) + i11 * (ny - b[1])];
    return { ainv: [i00, i10, i01, i11], b, rect, visible, inv };
  }

  #rangeFade(range, lg, soft = 0.25) {
    if (!range) return 1;
    const lo = Math.log10(range[0]), hi = Math.log10(range[1]);
    return smoothstep(lo - soft, lo + soft, lg) * (1 - smoothstep(hi - soft, hi + soft, lg));
  }

  #stats(j) {
    if (this.statsCache.has(j)) return this.statsCache.get(j);
    const L = this.scene.layers;
    const l = L[j], p = L[j - 1];
    const ct = this.store.thumb(l), pt = p && this.store.thumb(p);
    if (!ct || !pt || !ct.thumb || !pt.thumb) return null;
    const at = l.raw.at || [0.5, 0.5];
    const hw = (l.W / p.W) * 0.3, hh = (l.H / p.H) * 0.3;
    const src = regionStats(ct.thumb, 0.2, 0.2, 0.8, 0.8);
    const dst = regionStats(pt.thumb, at[0] - hw, at[1] - hh, at[0] + hw, at[1] + hh);
    // match against the parent as it's displayed: a desaturated parent must not pass its hidden colour on
    const ps = p.grade?.saturation;
    if (dst && ps != null && ps < 1) {
      const lum = (v) => 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
      const gm = lum(dst.mean), gs = lum(dst.std);
      dst.mean = dst.mean.map((v) => gm + (v - gm) * ps);
      dst.std = dst.std.map((v) => gs + (v - gs) * ps);
    }
    const st = src && dst ? {
      srcMean: src.mean, srcStd: src.std, dstMean: dst.mean,
      dstStd: dst.std.map((d, c) => src.std[c] * clamp(d / src.std[c], 0.6, 1.6)),
    } : null;
    this.statsCache.set(j, st);
    return st;
  }

  // ------------------------------------------------------------------ render
  #render(cam) {
    const r = this.renderer;
    const L = this.scene.layers;
    const i = cam.anchor;
    const W = this.canvas.width;
    const lg = Math.log10(cam.Vw);
    const proj = this.#proj(cam);
    r.begin();

    // gather candidate layers
    const draws = [];
    const lo = Math.max(0, i - 2), hi = Math.min(L.length - 1, i + 4);
    for (let j = lo; j <= hi; j++) {
      const l = L[j];
      const M = this.scene.frameToFrame(j, i);
      const sizeFrac = l.W / cam.Vw;
      const coverFrac = this.path.cover(l) / cam.Vw;
      const pxW = sizeFrac * W;
      draws.push({ j, l, M, sizeFrac, coverFrac, pxW });
    }

    // per-layer opacity + coverage culling
    let start = 0;
    const ghost = this.editor.ghostFor(i);
    for (let n = 0; n < draws.length; n++) {
      const d = draws[n];
      const { l, j } = d;
      const tex = l.src ? this.store.get(l) : null;
      d.tex = tex;
      // 'grow': perfectly aligned imagery (Earth, space) shows up early as a sharper patch.
      // 'dissolve': images that don't line up cross-fade over the whole frame instead, and only
      // take hard edges once they fill the screen, so no rectangle is ever visible.
      const dissolve = l.transition === 'dissolve';
      let alpha = j === 0 ? 1 : smoothstep(l.fadeIn[0], l.fadeIn[1], dissolve ? d.coverFrac : d.sizeFrac);
      const texW = tex ? tex.w : l.px[0];
      d.mag = d.pxW / texW;
      if (l.blend === 'lighten') {
        alpha *= 1 - smoothstep(l.maxUpscale, l.maxUpscale * 3, d.mag);
        alpha *= 1 / (1 + 0.35 * Math.max(0, Math.log2(d.mag) - 1)); // a blown-up glow shouldn't out-shine the stars
      }
      const child = L[j + 1];
      if (child && child.src && l.blend === 'lighten' && this.store.get(child)) {
        // the child is a sharper picture of the same region: don't let our blown-up pixels haze over it
        alpha *= 1 - smoothstep(0.55, 1.0, this.path.cover(child) / cam.Vw);
      }
      if (child && child.blend === 'lighten' && l.blend === 'normal') {
        alpha *= 1 - smoothstep(0.2, 0.85, this.path.cover(child) / cam.Vw);
      }
      d.grow = dissolve ? smoothstep(0.82, 1.0, d.coverFrac) : smoothstep(0.3, 0.92, d.coverFrac);
      if (ghost && j === i + 1) { alpha = 0.5; d.grow = 1; }
      d.alpha = alpha;
      if (!tex || !l.src) continue;
      d.aff = this.#affine(cam, d.M, l.W, l.H);
      const aspect = [l.W / Math.min(l.W, l.H), l.H / Math.min(l.W, l.H)];
      d.aspect = aspect;
      // a normal layer that fills the screen hides everything behind it, even while it fades out
      // for a glowing child (otherwise hugely blown-up ancestors show through)
      if (l.blend === 'normal' && (alpha > 0.999 || d.coverFrac >= 1) && d.grow > 0.999 && !ghost) {
        const rf = 0.05;
        const ok = [[-1, -1], [1, -1], [-1, 1], [1, 1]].every(([x, y]) => {
          const q = d.aff.inv(x, y);
          return Math.abs(q[0]) <= 0.5 * (1 - rf / aspect[0]) && Math.abs(q[1]) <= 0.5 * (1 - rf / aspect[1]);
        });
        if (ok) start = n;
      }
    }

    for (let n = start; n < draws.length; n++) {
      const d = draws[n];
      const { l, j } = d;
      if (!d.tex || d.alpha <= 0.002 || !d.aff || !d.aff.visible) continue;
      if (d.pxW < 1.5) continue;
      d.drawn = true;
      if (d.tex.video && this.store.refreshVideo(d.tex)) this.dirty = true;
      const st = l.colorMatch > 0 && j > 0 ? this.#stats(j) : null;
      const match = st ? l.colorMatch * (l.transition === 'dissolve' ? 1 - smoothstep(0.5, 1.0, d.coverFrac) : 1 - d.grow) : 0;
      const s = st || { srcMean: [0, 0, 0], srcStd: [1, 1, 1], dstMean: [0, 0, 0], dstStd: [1, 1, 1] };
      const g = l.grade || {};
      r.drawLayer({
        tex: d.tex.tex,
        rect: d.aff.rect,
        ainv: d.aff.ainv,
        b: d.aff.b,
        aspect: d.aspect,
        alpha: d.alpha,
        feather: l.feather,
        grow: d.grow,
        // soft edge while growing, hard edge once it fills the screen (no parent bleeding at the borders)
        rectFeather: ghost && j === i + 1 ? 0.001 : Math.max(0.001, 0.05 * (1 - smoothstep(0.88, 1.0, d.coverFrac))),
        match,
        srcMean: s.srcMean, srcStd: s.srcStd, dstMean: s.dstMean, dstStd: s.dstStd,
        grade: [g.exposure ?? 0, g.contrast ?? 1, g.saturation ?? 1],
        blend: l.blend,
        // blown-up space images: read from a smoother mip level so they turn into a soft glow
        // instead of showing enlarged JPEG blocks
        bias: l.blend === 'lighten' && d.mag > 1.6 ? 1.5 * Math.log2(d.mag) - 0.7 : 0,
      });
    }

    // stars (procedural, anchored to a frame — the Sun)
    this.#drawStars(cam, lg);

    // extras: orbits, glows, images
    for (const e of this.scene.extras) {
      if (Math.abs(e.frame - i) > 4) continue;
      const fade = this.#rangeFade(e.range, lg);
      if (fade <= 0.002) continue;
      const M = this.scene.frameToFrame(e.frame, i);
      const [x, y] = Sim.apply(M, e.pos?.[0] ?? 0, e.pos?.[1] ?? 0);
      const c = this.ndcToPx(proj(x, y));
      if (e.kind === 'ring') {
        const rp = (e.radius / cam.Vw) * W;
        if (rp < 2 || rp > 2e7) continue;
        const col = e.color || [0.6, 0.75, 1];
        r.drawRing({ center: c, r: rp, width: (e.widthPx ?? 1.2) * this.dpr, color: [col[0], col[1], col[2], (e.alpha ?? 0.35) * fade] });
      } else if (e.kind === 'glow') {
        const rp = Math.max(0.9 * this.dpr, ((e.radius || 0) / cam.Vw) * W);
        const col = (e.color || [1, 1, 1]).map((v) => v * fade);
        r.drawGlow({ center: c, radius: rp, halo: (e.halo ?? 3) * this.dpr, color: col, intensity: e.intensity ?? 1, spikes: e.spikes ?? 0 });
      } else if (e.kind === 'image' && e.src) {
        const tex = this.store.get(e);
        if (!tex) continue;
        const H = e.width * (tex.h / tex.w);
        const Mi = Sim.compose(M, Sim.fromAngle(e.rot || 0, e.pos[0], e.pos[1]));
        const aff = this.#affine(cam, Mi, e.width, H);
        const pxW = (e.width / cam.Vw) * W;
        if (!aff.visible || pxW < 1) continue;
        r.drawLayer({
          tex: tex.tex, rect: aff.rect, ainv: aff.ainv, b: aff.b, aspect: [1, H / e.width],
          alpha: fade, feather: e.feather ?? 0.12, grow: 0, rectFeather: 0.05, match: 0,
          srcMean: [0, 0, 0], srcStd: [1, 1, 1], dstMean: [0, 0, 0], dstStd: [1, 1, 1],
          grade: [0, 1, 1], blend: e.blend === 'normal' ? 'normal' : 'lighten',
        });
      }
    }

    // tiny layers become points of light ("pale blue dot")
    for (const d of draws) {
      const dot = d.l.dot;
      if (!dot) continue;
      const maxView = dot.maxView ?? d.l.W * 3e6;
      const a = (1 - smoothstep(3 * this.dpr, 14 * this.dpr, d.pxW)) * (1 - smoothstep(Math.log10(maxView) - 0.4, Math.log10(maxView), lg));
      if (a <= 0.002) continue;
      const c = this.ndcToPx(proj(d.M.x, d.M.y));
      r.drawGlow({ center: c, radius: Math.max(0.9 * this.dpr, d.pxW / 2), halo: (dot.halo ?? 2.2) * this.dpr, color: dot.color.map((v) => v * a), intensity: dot.intensity ?? 1.4, spikes: 0 });
    }

    const blur = clamp(Math.abs(this.lgVel) * 0.022, 0, 0.07);
    r.end({ blur: params.has('noblur') ? 0 : blur, time: this.time, grain: 0.022, vignette: 0.38 });
    this.draws = draws;
  }

  // ------------------------------------------------------------------ stars
  #buildStars() {
    const sf = this.scene.starfield;
    if (!sf || sf.frame < 0) return;
    let seed = sf.seed ?? 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const data = [];
    this.octaves = [];
    const [lo, hi] = sf.octaves; // log10 meters
    const step = sf.step ?? 0.25;
    const palette = [
      [0.62, 0.72, 1.0], [0.8, 0.86, 1.0], [1.0, 0.98, 0.95], [1.0, 0.93, 0.8], [1.0, 0.82, 0.6], [1.0, 0.7, 0.5],
    ];
    const weights = [0.03, 0.1, 0.27, 0.3, 0.2, 0.1];
    for (let e = lo; e <= hi + 1e-9; e += step) {
      const S = Math.pow(10, e);
      const first = data.length / 6;
      const n = sf.perOctave ?? 420;
      for (let k = 0; k < n; k++) {
        const rr = Math.sqrt(rnd());
        const th = rnd() * Math.PI * 2;
        if (rr * S < (sf.void ?? 3.8e16)) continue;
        let pick = rnd(), ci = 0;
        while (ci < weights.length - 1 && pick > weights[ci]) { pick -= weights[ci]; ci++; }
        const col = palette[ci];
        const b = Math.pow(rnd(), 3.2);
        const bright = 0.3 + b * 1.9;
        data.push(rr * Math.cos(th), rr * Math.sin(th), col[0] * bright, col[1] * bright, col[2] * bright, 1.3 + b * 3.4);
      }
      this.octaves.push({ S, first, count: data.length / 6 - first });
    }
    this.renderer.setStars(new Float32Array(data));

    const warp = [];
    for (let k = 0; k < (sf.warpCount ?? 1400); k++) {
      let pick = rnd(), ci = 0;
      while (ci < weights.length - 1 && pick > weights[ci]) { pick -= weights[ci]; ci++; }
      const col = palette[ci];
      const b = 0.35 + Math.pow(rnd(), 2.5) * 1.1;
      warp.push(rnd() * 2 - 1, rnd() * 2 - 1, col[0] * b, col[1] * b, col[2] * b, 0.9 + rnd() * 1.4, rnd());
    }
    this.renderer.setWarp(new Float32Array(warp));
  }

  #drawStars(cam, lg) {
    const sf = this.scene.starfield;
    if (sf && sf.warpRange) {
      this.renderer.drawWarp({
        phase: -lg * (sf.warpRate ?? 0.9),
        aspect: this.aspect,
        alpha: (sf.warpAlpha ?? 0.55) * this.#rangeFade(sf.warpRange, lg, 0.5),
        px: 2.1 * this.dpr,
      });
    }
    if (!sf || !this.octaves || Math.abs(sf.frame - cam.anchor) > 5) return;
    const fade = this.#rangeFade(sf.range, lg, 0.4);
    if (fade <= 0.002) return;
    const M = this.scene.frameToFrame(sf.frame, cam.anchor);
    const off = this.#proj(cam)(M.x, M.y);
    const ang = Sim.angle(M) - cam.rot;
    const c = Math.cos(ang), s = Math.sin(ang), a = this.aspect;
    const batches = [];
    for (const o of this.octaves) {
      const sc = o.S / cam.Vw;
      if (sc < 0.06 || sc > 80 || !o.count) continue;
      const alpha = smoothstep(0.08, 0.4, sc) * fade;
      batches.push({ first: o.first, count: o.count, off, m: [2 * c * sc, -2 * a * s * sc, -2 * s * sc, -2 * a * c * sc], alpha });
    }
    this.renderer.drawStars(batches, 1.7 * this.dpr);
  }

  // ------------------------------------------------------------------ HUD
  #hud(cam) {
    const L = this.scene.layers;
    const lg = Math.log10(cam.Vw);
    const logs = this.path.coverLog;
    // nearest layer in log-space drives the story text
    let best = 0, bd = Infinity;
    for (let j = 0; j < logs.length; j++) {
      const d = Math.abs(lg - logs[j]);
      if (d < bd) { bd = d; best = j; }
    }
    const side = lg > logs[best] ? best - 1 : best + 1;
    const gap = side >= 0 && side < logs.length ? Math.abs(logs[side] - logs[best]) / 2 : 0.6;
    const intro = 1 - smoothstep(0.05, 0.55, this.sSmooth);
    let op = 1 - smoothstep(0.55, 0.95, bd / Math.max(0.05, gap));
    op *= 1 - intro;
    const l = L[best];
    if (this.storyIdx !== best) {
      this.storyIdx = best;
      $('story-label').textContent = l.label;
      $('story-caption').textContent = l.caption;
      $('story-power').textContent = `${powerOfTen(l.W)} · ${formatLength(l.W)} across`;
    }
    $('story-label').parentElement.style.opacity = op.toFixed(3);
    $('intro').style.opacity = intro.toFixed(3);

    const cssW = innerWidth;
    const len = niceLength(cam.Vw * 0.16);
    $('scale-bar').style.width = `${((len / cam.Vw) * cssW).toFixed(1)}px`;
    $('scale-len').textContent = formatLength(len);
    $('fov').textContent = formatLength(cam.Vw);
    $('pow').textContent = powerOfTen(cam.Vw);

    const rm = this.ruler;
    if (rm) {
      const f = clamp((rm.top - lg) / (rm.top - rm.bottom), 0, 1);
      rm.marker.style.top = `${(f * 100).toFixed(3)}%`;
    }
    this.#labels(cam);
  }

  #labels(cam) {
    const lg = Math.log10(cam.Vw);
    const proj = this.#proj(cam);
    const seen = new Set();
    const put = (key, text, x, y, alpha) => {
      let el = this.labelEls.get(key);
      if (!el) {
        el = document.createElement('div');
        el.className = 'obj-label';
        el.textContent = text;
        $('labels').appendChild(el);
        this.labelEls.set(key, el);
      }
      seen.add(key);
      const cx = ((x + 1) / 2) * innerWidth, cy = ((1 - y) / 2) * innerHeight;
      el.style.transform = `translate(${(cx + 6).toFixed(1)}px, ${(cy - 9).toFixed(1)}px)`;
      el.style.opacity = alpha.toFixed(3);
    };
    const i = cam.anchor;
    for (const e of this.scene.extras) {
      if (!e.label || Math.abs(e.frame - i) > 4) continue;
      const a = this.#rangeFade(e.labelRange || e.range, lg, 0.2);
      if (a < 0.01) continue;
      const M = this.scene.frameToFrame(e.frame, i);
      const [x, y] = Sim.apply(M, e.pos?.[0] ?? 0, e.pos?.[1] ?? 0);
      const n = proj(x, y);
      if (Math.abs(n[0]) > 1.05 || Math.abs(n[1]) > 1.05) continue;
      put(`e${e.index}`, e.label, n[0], n[1], a);
    }
    for (const d of this.draws || []) {
      const dl = d.l.raw.dotLabel;
      if (!dl) continue;
      const a = this.#rangeFade(dl.range, lg, 0.2);
      if (a < 0.01) continue;
      const n = proj(d.M.x, d.M.y);
      if (Math.abs(n[0]) > 1.05 || Math.abs(n[1]) > 1.05) continue;
      put(`l${d.j}`, dl.text, n[0], n[1], a);
    }
    for (const [k, el] of this.labelEls) if (!seen.has(k)) el.style.opacity = '0';
  }

  #buildRuler() {
    const nav = $('ruler');
    nav.tabIndex = 0;
    nav.addEventListener('click', (ev) => {
      const r = nav.getBoundingClientRect();
      const f = clamp((ev.clientY - r.top) / r.height, 0, 1);
      const lg = this.ruler.top - f * (this.ruler.top - this.ruler.bottom);
      const s = this.path.scrollForLog(lg);
      scrollTo({ top: s * this.unitPx, behavior: 'smooth' });
    });
  }

  #layoutRuler() {
    const nav = $('ruler');
    nav.innerHTML = '';
    const logs = this.path.coverLog;
    const top = Math.ceil(this.path.knotsL[0]);
    const bottom = Math.floor(logs[logs.length - 1]);
    this.ruler = { top, bottom };
    for (let e = top; e >= bottom; e--) {
      const f = (top - e) / (top - bottom);
      const t = document.createElement('div');
      const major = e % 3 === 0;
      t.className = `tick${major ? ' major' : ''}`;
      t.style.top = `${f * 100}%`;
      nav.appendChild(t);
      if (major) {
        const lab = document.createElement('div');
        lab.className = 'tick-label';
        lab.style.top = `${f * 100}%`;
        lab.textContent = shortUnit(e);
        nav.appendChild(lab);
      }
    }
    const m = document.createElement('div');
    m.className = 'marker';
    nav.appendChild(m);
    this.ruler.marker = m;
  }

  #buildCredits() {
    const list = $('credits-list');
    const items = [...this.scene.layers.map((l) => l.raw), ...(this.data.extras || [])].filter((r) => r.src && r.credit);
    const seen = new Set();
    for (const r of items) {
      const key = `${r.credit}|${r.sourceUrl}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const li = document.createElement('li');
      li.innerHTML = `<b></b> — <span></span>${r.license ? ` · <i></i>` : ''}${r.sourceUrl ? '<br><a target="_blank" rel="noopener"></a>' : ''}`;
      li.querySelector('b').textContent = r.label || r.id;
      li.querySelector('span').textContent = r.credit;
      if (r.license) li.querySelector('i').textContent = r.license;
      if (r.sourceUrl) { const a = li.querySelector('a'); a.href = r.sourceUrl; a.textContent = r.sourceUrl; }
      list.appendChild(li);
    }
    $('btn-credits').addEventListener('click', () => ($('credits').hidden = !$('credits').hidden));
    document.querySelectorAll('[data-close]').forEach((b) => b.addEventListener('click', () => (b.closest('.panel').hidden = true)));
  }
}

async function boot() {
  try {
    const url = params.get('scene') || 'scene.json';
    const data = await fetch(url, { cache: 'no-cache' }).then((r) => {
      if (!r.ok) throw new Error(`${url}: ${r.status}`);
      return r.json();
    });
    window.app = new App(data);
  } catch (err) {
    console.error(err);
    const d = document.createElement('div');
    d.className = 'fatal';
    d.textContent = location.protocol === 'file:'
      ? 'Open this through a local web server (run start.bat), not by double-clicking index.html.'
      : `Could not start: ${err.message}`;
    document.body.appendChild(d);
  }
}

boot();
