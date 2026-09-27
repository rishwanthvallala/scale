// Streams layer images into GPU textures around the current zoom position.
// Only a small window of layers lives on the GPU at once (4K textures are ~90 MB
// each with mipmaps); everything else stays in the browser's HTTP cache.

const THUMB = 128;

export class TextureStore {
  constructor(gl, { useSmall = false, maxLive = 7, onChange = () => {} } = {}) {
    this.gl = gl;
    this.useSmall = useSmall;
    this.maxLive = maxLive;
    this.onChange = onChange;
    this.entries = new Map(); // key -> { state, tex, w, h, thumb, stats }
    this.queue = [];
    this.inflight = 0;
    this.maxInflight = 2;
    this.aniso = gl.getExtension('EXT_texture_filter_anisotropic');
    this.prefetched = new Set();
  }

  urlFor(item) {
    return (this.useSmall && item.src2k) || item.src;
  }

  get(item) {
    const e = this.entries.get(this.urlFor(item));
    return e && e.state === 'ready' ? e : null;
  }

  thumb(item) {
    const e = this.entries.get(this.urlFor(item));
    return e && e.thumb ? e : null;
  }

  // wanted: array of { item, priority } (lower = sooner)
  want(wanted) {
    const keep = new Set();
    wanted.sort((a, b) => a.priority - b.priority);
    for (const { item } of wanted) {
      const url = this.urlFor(item);
      if (!url) continue;
      keep.add(url);
      if (!this.entries.has(url)) {
        this.entries.set(url, { state: 'queued', url, tex: null, thumb: null });
        this.queue.push(url);
      }
    }
    // reorder queue by current priority
    const order = new Map(wanted.map((w, i) => [this.urlFor(w.item), i]));
    this.queue = this.queue.filter((u) => keep.has(u)).sort((a, b) => (order.get(a) ?? 1e9) - (order.get(b) ?? 1e9));
    for (const [url, e] of this.entries) {
      if (!keep.has(url) && e.state === 'queued') this.entries.delete(url);
    }
    // evict GPU textures outside the window (keep thumbs: they're tiny and used for color matching)
    const live = [...this.entries.values()].filter((e) => e.tex);
    if (live.length > this.maxLive) {
      for (const e of live) {
        if (!keep.has(e.url) && e.tex) {
          this.gl.deleteTexture(e.tex);
          e.tex = null;
          if (e.video) { e.video.pause(); e.video.removeAttribute('src'); e.video.load(); e.video = null; }
          e.state = 'evicted';
        }
      }
    }
    for (const [url, e] of this.entries) {
      if (keep.has(url) && e.state === 'evicted') {
        e.state = 'queued';
        this.queue.push(url);
      }
    }
    this.pump();
  }

  // Warm the HTTP cache for everything, nearest first, one at a time.
  async prefetchAll(items) {
    for (const item of items) {
      const url = this.urlFor(item);
      if (!url || this.prefetched.has(url)) continue;
      this.prefetched.add(url);
      try {
        await fetch(url, { priority: 'low' }).then((r) => r.blob());
      } catch { /* ignore */ }
    }
  }

  pending() {
    let n = 0;
    for (const e of this.entries.values()) if (e.state === 'queued' || e.state === 'loading') n++;
    return n;
  }

  pump() {
    while (this.inflight < this.maxInflight && this.queue.length) {
      const url = this.queue.shift();
      const e = this.entries.get(url);
      if (!e || e.state !== 'queued') continue;
      e.state = 'loading';
      this.inflight++;
      this.#load(e).finally(() => {
        this.inflight--;
        this.pump();
      });
    }
  }

  // Video layers: refresh the texture from the current frame (call only for layers being drawn).
  refreshVideo(entry) {
    const v = entry.video;
    if (!v) return false;
    if (v.paused) v.play().catch(() => {});
    if (v.readyState < 2) return true;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, entry.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, v);
    return true;
  }

  async #loadVideo(e) {
    const v = document.createElement('video');
    Object.assign(v, { muted: true, loop: true, playsInline: true, preload: 'auto', crossOrigin: 'anonymous' });
    v.src = e.url;
    await new Promise((res, rej) => {
      v.onloadeddata = res;
      v.onerror = () => rej(new Error(`video failed: ${e.url}`));
    });
    v.play().catch(() => {});
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, v);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    Object.assign(e, { tex, video: v, w: v.videoWidth, h: v.videoHeight, state: 'ready' });
    if (!e.thumb) e.thumb = makeThumb(v, v.videoWidth, v.videoHeight);
    this.onChange();
  }

  async #load(e) {
    try {
      if (/\.(mp4|webm|mov)(\?|$)/i.test(e.url)) return await this.#loadVideo(e);
      const res = await fetch(e.url);
      if (!res.ok) throw new Error(`${res.status} ${e.url}`);
      const blob = await res.blob();
      const bmp = await createImageBitmap(blob, { premultiplyAlpha: 'none' });
      if (!this.entries.has(e.url)) { bmp.close(); return; }
      const gl = this.gl;
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      if (this.aniso) gl.texParameterf(gl.TEXTURE_2D, this.aniso.TEXTURE_MAX_ANISOTROPY_EXT, 4);
      e.tex = tex;
      e.w = bmp.width;
      e.h = bmp.height;
      if (!e.thumb) e.thumb = makeThumb(bmp, bmp.width, bmp.height);
      bmp.close();
      e.state = 'ready';
      this.onChange();
    } catch (err) {
      console.warn('texture load failed', err);
      e.state = 'failed';
    }
  }
}

function makeThumb(bmp, w, h) {
  const c = document.createElement('canvas');
  c.width = THUMB;
  c.height = Math.max(1, Math.round((THUMB * h) / w));
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bmp, 0, 0, c.width, c.height);
  return { w: c.width, h: c.height, data: ctx.getImageData(0, 0, c.width, c.height).data };
}

// Mean / std per channel of a thumb region (normalized rect, 0..1).
export function regionStats(thumb, x0, y0, x1, y1) {
  const { w, h, data } = thumb;
  const ax = Math.max(0, Math.floor(x0 * w)), bx = Math.min(w, Math.ceil(x1 * w));
  const ay = Math.max(0, Math.floor(y0 * h)), by = Math.min(h, Math.ceil(y1 * h));
  if (bx <= ax || by <= ay) return null;
  let n = 0;
  const s = [0, 0, 0], q = [0, 0, 0];
  for (let y = ay; y < by; y++) {
    for (let x = ax; x < bx; x++) {
      const k = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) {
        const v = data[k + c] / 255;
        s[c] += v;
        q[c] += v * v;
      }
      n++;
    }
  }
  const mean = s.map((v) => v / n);
  const std = q.map((v, c) => Math.sqrt(Math.max(1e-6, v / n - mean[c] * mean[c])));
  return { mean, std, n };
}
