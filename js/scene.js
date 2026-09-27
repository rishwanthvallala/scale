// Scene geometry: a chain of nested layers, each with its own frame in meters.
// Frame i: origin at the layer's center, +x right, +y down (image space), units = meters.
// layer.toParent maps frame i -> frame i-1 (rotation + translation, no scale).
// All of this runs in float64 in JS; only screen-relative numbers reach the GPU,
// which is what lets one continuous zoom span ~42 orders of magnitude.

export const Sim = {
  identity: () => ({ c: 1, s: 0, x: 0, y: 0 }),
  fromAngle(deg, x = 0, y = 0) {
    const a = (deg * Math.PI) / 180;
    return { c: Math.cos(a), s: Math.sin(a), x, y };
  },
  apply(t, px, py) {
    return [t.c * px - t.s * py + t.x, t.s * px + t.c * py + t.y];
  },
  // a∘b: apply b first, then a
  compose(a, b) {
    return {
      c: a.c * b.c - a.s * b.s,
      s: a.s * b.c + a.c * b.s,
      x: a.c * b.x - a.s * b.y + a.x,
      y: a.s * b.x + a.c * b.y + a.y,
    };
  },
  inverse(t) {
    return { c: t.c, s: -t.s, x: -(t.c * t.x + t.s * t.y), y: -(-t.s * t.x + t.c * t.y) };
  },
  angle(t) {
    return Math.atan2(t.s, t.c);
  },
};

const DEFAULTS = {
  normal: { feather: 0.45, fadeIn: [0.035, 0.2], colorMatch: 0.8, maxUpscale: 1e9 },
  lighten: { feather: 0.3, fadeIn: [0, 0], colorMatch: 0, maxUpscale: 4 },
  none: { feather: 0, fadeIn: [0, 0], colorMatch: 0, maxUpscale: 1e9 },
};

export class Scene {
  constructor(data) {
    this.data = data;
    this.layers = data.layers.map((raw, i) => this.#makeLayer(raw, i));
    this.byId = new Map(this.layers.map((l) => [l.id, l]));
    this.extras = (data.extras || []).map((e, i) => ({ ...e, index: i, frame: this.byId.get(e.frame)?.index ?? -1 }))
      .filter((e) => e.frame >= 0);
    this.starfield = data.starfield ? { ...data.starfield, frame: this.byId.get(data.starfield.frame)?.index ?? -1 } : null;
    this.rebuild();
  }

  #makeLayer(raw, index) {
    const blend = raw.blend === 'add' ? 'lighten' : raw.blend || (raw.src ? 'normal' : 'none');
    const transition = raw.transition || 'grow';
    const d = transition === 'dissolve' && blend === 'normal' ? { ...DEFAULTS.normal, feather: 0.6, fadeIn: [0.3, 0.8] } : DEFAULTS[blend];
    const px = raw.px || [4096, 4096];
    return {
      raw,
      index,
      id: raw.id,
      label: raw.label || raw.id,
      caption: raw.caption || '',
      src: raw.src || null,
      src2k: raw.src2k || null,
      px,
      W: raw.width,
      H: raw.width * (px[1] / px[0]),
      blend,
      transition,
      feather: raw.feather ?? d.feather,
      fadeIn: raw.fadeIn ?? d.fadeIn,
      colorMatch: raw.colorMatch ?? d.colorMatch,
      maxUpscale: raw.maxUpscale ?? d.maxUpscale,
      dot: raw.dot || null,
      grade: raw.grade || null,
      scroll: raw.scroll,
      toParent: Sim.identity(),
    };
  }

  // Recompute transforms from the raw placement data ("at" = center of this layer
  // in the parent's normalized image coordinates, "rot" = degrees).
  rebuild() {
    for (let i = 1; i < this.layers.length; i++) {
      const l = this.layers[i];
      const p = this.layers[i - 1];
      const at = l.raw.at || [0.5, 0.5];
      l.toParent = Sim.fromAngle(l.raw.rot || 0, (at[0] - 0.5) * p.W, (at[1] - 0.5) * p.H);
      l.W = l.raw.width;
      l.H = l.raw.width * (l.px[1] / l.px[0]);
    }
    this.layers[0].W = this.layers[0].raw.width;
    this.layers[0].H = this.layers[0].W * (this.layers[0].px[1] / this.layers[0].px[0]);
    this._cache = new Map();
  }

  // Transform mapping coordinates in frame j to frame k.
  frameToFrame(j, k) {
    if (j === k) return Sim.identity();
    const key = j * 4096 + k;
    let t = this._cache.get(key);
    if (t) return t;
    if (j > k) {
      t = Sim.identity();
      for (let m = k + 1; m <= j; m++) t = Sim.compose(t, this.layers[m].toParent);
    } else {
      t = Sim.inverse(this.frameToFrame(k, j));
    }
    this._cache.set(key, t);
    return t;
  }
}

// ---------------------------------------------------------------------------
// Camera path: scroll -> log10(view width) via a monotone cubic through one knot
// per layer (the moment that layer exactly covers the viewport), then the view
// width tells us which segment we're in and how far along it we are.

export class CameraPath {
  constructor(scene) {
    this.scene = scene;
    this.aspect = 16 / 9;
  }

  cover(l) {
    return Math.min(l.W, l.H * this.aspect);
  }

  contain(l) {
    return Math.max(l.W, l.H * this.aspect);
  }

  layout(aspect) {
    this.aspect = aspect;
    const L = this.scene.layers;
    const logs = L.map((l) => Math.log10(this.cover(l)));
    // enforce strictly decreasing view sizes
    for (let i = 1; i < logs.length; i++) if (logs[i] > logs[i - 1] - 0.05) logs[i] = logs[i - 1] - 0.05;
    this.coverLog = logs;

    const knotsS = [];
    const knotsL = [];
    // intro: first layer fully visible with some margin
    const introLog = Math.log10(this.contain(L[0]) * 1.12);
    let s = 0;
    knotsS.push(s);
    knotsL.push(Math.max(introLog, logs[0] + 0.05));
    s += 0.9;
    knotsS.push(s);
    knotsL.push(logs[0]);
    for (let i = 1; i < L.length; i++) {
      const decades = logs[i - 1] - logs[i];
      const w = L[i - 1].scroll ?? clamp(0.55 + 0.42 * decades, 0.75, 2.2);
      s += w;
      knotsS.push(s);
      knotsL.push(logs[i]);
    }
    this.knotsS = knotsS;
    this.knotsL = knotsL;
    this.length = s;
    this.tangents = monotoneTangents(knotsS, knotsL);
  }

  // scroll units -> log10(view width)
  logAt(s) {
    const S = this.knotsS, Lg = this.knotsL, m = this.tangents;
    if (s <= S[0]) return Lg[0];
    if (s >= S[S.length - 1]) return Lg[Lg.length - 1];
    let i = 0;
    while (i < S.length - 2 && s > S[i + 1]) i++;
    const h = S[i + 1] - S[i];
    const t = (s - S[i]) / h;
    const t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * Lg[i] + (t3 - 2 * t2 + t) * h * m[i] + (-2 * t3 + 3 * t2) * Lg[i + 1] + (t3 - t2) * h * m[i + 1];
  }

  // inverse of logAt (for jump-to-scale); bisection is plenty fast
  scrollForLog(lg) {
    let a = 0, b = this.length;
    for (let k = 0; k < 60; k++) {
      const mid = (a + b) / 2;
      if (this.logAt(mid) > lg) a = mid; else b = mid;
    }
    return (a + b) / 2;
  }

  scrollForLayer(i) {
    return this.knotsS[i + 1];
  }

  // Camera for a given log10(view width). Returns anchor frame index + pose in that frame.
  cameraAt(lg) {
    const L = this.scene.layers;
    const logs = this.coverLog;
    const n = L.length;
    let i = 0;
    while (i < n - 1 && lg <= logs[i + 1]) i++;
    const Vw = Math.pow(10, lg);
    if (i >= n - 1) return { anchor: n - 1, seg: n - 1, t: 0, cx: 0, cy: 0, rot: 0, Vw, lg };
    const t = clamp((logs[i] - lg) / (logs[i] - logs[i + 1]), 0, 1);
    const child = L[i + 1].toParent;
    const Px = child.x, Py = child.y;
    const Vw0 = Math.pow(10, logs[i]);
    const e = smoothstep(0, 1, t);
    const rot = Sim.angle(child) * e;
    // screen-space offset of the target, shrinking to 0 over the segment
    const sx = (Px / Vw0) * (1 - e);
    const sy = (Py / Vw0) * (1 - e);
    const c = Math.cos(rot), s = Math.sin(rot);
    const ox = (c * sx - s * sy) * Vw;
    const oy = (s * sx + c * sy) * Vw;
    return { anchor: i, seg: i, t, cx: Px - ox, cy: Py - oy, rot, Vw, lg };
  }
}

function monotoneTangents(x, y) {
  const n = x.length;
  const d = [], m = new Array(n);
  for (let i = 0; i < n - 1; i++) d.push((y[i + 1] - y[i]) / (x[i + 1] - x[i]));
  m[0] = d[0];
  m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (d[i] === 0) { m[i] = 0; m[i + 1] = 0; continue; }
    const a = m[i] / d[i], b = m[i + 1] / d[i];
    const h = a * a + b * b;
    if (h > 9) {
      const tau = 3 / Math.sqrt(h);
      m[i] = tau * a * d[i];
      m[i + 1] = tau * b * d[i];
    }
  }
  return m;
}

export function clamp(v, a, b) {
  return v < a ? a : v > b ? b : v;
}

export function smoothstep(a, b, v) {
  if (a === b) return v < a ? 0 : 1;
  const t = clamp((v - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
}
