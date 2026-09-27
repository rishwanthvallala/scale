// WebGL2 renderer: image layers (masked, color-matched), glows, orbit rings,
// procedural stars, then a post pass (zoom blur, vignette, grain) that glues
// sources of different origin into one "camera".

const QUAD_VS = `#version 300 es
in vec2 a_pos;
uniform vec4 u_rect;
void main() { gl_Position = vec4(mix(u_rect.xy, u_rect.zw, a_pos), 0.0, 1.0); }`;

const LAYER_FS = `#version 300 es
precision highp float;
uniform sampler2D u_tex;
uniform vec2 u_view;
uniform mat2 u_ainv;
uniform vec2 u_b;
uniform vec2 u_aspect;
uniform float u_alpha;
uniform float u_feather;
uniform float u_grow;
uniform float u_rectFeather;
uniform vec3 u_srcMean, u_srcStd, u_dstMean, u_dstStd;
uniform float u_match;
uniform vec3 u_grade;
uniform float u_bias;
out vec4 o;
void main() {
  vec2 ndc = gl_FragCoord.xy / u_view * 2.0 - 1.0;
  vec2 q = u_ainv * (ndc - u_b);
  if (abs(q.x) > 0.5 || abs(q.y) > 0.5) discard;
  vec3 c = texture(u_tex, q + 0.5, u_bias).rgb;
  vec3 m = (c - u_srcMean) / u_srcStd * u_dstStd + u_dstMean;
  c = mix(c, clamp(m, 0.0, 1.0), u_match);
  c *= exp2(u_grade.x);
  c = (c - 0.5) * u_grade.y + 0.5;
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  c = clamp(mix(vec3(l), c, u_grade.z), 0.0, 1.0);
  float rr = length(q * 2.0);
  float mRad = 1.0 - smoothstep(1.0 - u_feather, 1.0, rr);
  vec2 e = (1.0 - abs(q * 2.0)) * u_aspect;
  float mRect = smoothstep(0.0, u_rectFeather, min(e.x, e.y));
  float mask = mix(mRad, mRect, u_grow) * u_alpha;
  o = vec4(c * mask, mask);
}`;

const GLOW_FS = `#version 300 es
precision highp float;
uniform vec2 u_center;
uniform float u_radius;
uniform float u_halo;
uniform vec3 u_color;
uniform float u_intensity;
uniform float u_spikes;
uniform float u_ext;
out vec4 o;
void main() {
  vec2 p = gl_FragCoord.xy - u_center;
  float d = length(p);
  float win = 1.0 - smoothstep(u_ext * 0.45, u_ext, d);
  float disk = 1.0 - smoothstep(u_radius - 0.8, u_radius + 0.8, d);
  float halo = exp(-d / u_halo) * 0.65 + exp(-d / (u_halo * 5.0)) * 0.12;
  float sp = u_spikes * (exp(-abs(p.x) * 0.9) + exp(-abs(p.y) * 0.9)) * exp(-d / (u_halo * 7.0));
  o = vec4(u_color * (disk + (halo + sp) * u_intensity * win), 0.0);
}`;

const RING_FS = `#version 300 es
precision highp float;
uniform vec2 u_center;
uniform float u_r;
uniform float u_w;
uniform vec4 u_color;
out vec4 o;
void main() {
  float d = abs(length(gl_FragCoord.xy - u_center) - u_r);
  float a = 1.0 - smoothstep(u_w * 0.5 - 0.5, u_w * 0.5 + 0.8, d);
  o = vec4(u_color.rgb * a * u_color.a, 0.0);
}`;

const STAR_VS = `#version 300 es
in vec2 a_p;
in vec4 a_c;
uniform vec2 u_off;
uniform mat2 u_m;
uniform float u_alpha;
uniform float u_px;
out vec3 v_c;
void main() {
  gl_Position = vec4(u_off + u_m * a_p, 0.0, 1.0);
  gl_PointSize = a_c.a * u_px;
  v_c = a_c.rgb * u_alpha;
}`;

const STAR_FS = `#version 300 es
precision highp float;
in vec3 v_c;
out vec4 o;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float d2 = dot(p, p);
  float core = exp(-d2 * 22.0);
  float halo = exp(-sqrt(d2) * 6.0) * 0.22;
  o = vec4(v_c * (core + halo), 0.0);
}`;

// "Flying through space" stars: each star cycles through depth as the zoom decade changes,
// so zooming in carries you forward through them. Screen-space, very subtle.
const WARP_VS = `#version 300 es
in vec2 a_p;
in vec4 a_c;
in float a_z;
uniform float u_phase;
uniform float u_aspect;
uniform float u_alpha;
uniform float u_px;
out vec3 v_c;
void main() {
  float z = max(fract(a_z - u_phase), 0.002);
  vec2 p = a_p * 0.32 / z;
  gl_Position = vec4(p.x, p.y * u_aspect, 0.0, 1.0);
  float near = 1.0 - z;
  gl_PointSize = u_px * a_c.a * (0.8 + 1.6 * near * near);
  v_c = a_c.rgb * u_alpha * smoothstep(1.0, 0.78, z) * (0.35 + 0.65 * near);
}`;

const POST_FS = `#version 300 es
precision highp float;
uniform sampler2D u_scene;
uniform vec2 u_view;
uniform float u_blur;
uniform float u_time;
uniform float u_grain;
uniform float u_vignette;
out vec4 o;
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
void main() {
  vec2 uv = gl_FragCoord.xy / u_view;
  vec3 c;
  if (u_blur > 0.0004) {
    vec2 dir = uv - 0.5;
    float j = hash(gl_FragCoord.xy + u_time * 13.0);
    c = vec3(0.0);
    for (int i = 0; i < 12; i++) {
      float k = (float(i) + j) / 12.0;
      c += texture(u_scene, 0.5 + dir * (1.0 - u_blur * k)).rgb;
    }
    c /= 12.0;
  } else {
    c = texture(u_scene, uv).rgb;
  }
  vec2 v = (uv - 0.5) * vec2(u_view.x / u_view.y, 1.0);
  c *= mix(1.0, smoothstep(1.35, 0.3, length(v)), u_vignette);
  float g = hash(gl_FragCoord.xy + fract(u_time * 7.13) * 917.0) - 0.5;
  c += g * u_grain;
  o = vec4(c, 1.0);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) + '\n' + src);
  return s;
}

function program(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.bindAttribLocation(p, 0, 'a_pos');
  gl.bindAttribLocation(p, 0, 'a_p');
  gl.bindAttribLocation(p, 1, 'a_c');
  gl.bindAttribLocation(p, 2, 'a_z');
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const info = gl.getActiveUniform(p, i);
    u[info.name] = gl.getUniformLocation(p, info.name);
  }
  return { p, u };
}

export class Renderer {
  constructor(canvas) {
    const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, premultipliedAlpha: false, powerPreference: 'high-performance' });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.gl = gl;
    this.canvas = canvas;
    this.layerProg = program(gl, QUAD_VS, LAYER_FS);
    this.glowProg = program(gl, QUAD_VS, GLOW_FS);
    this.ringProg = program(gl, QUAD_VS, RING_FS);
    this.postProg = program(gl, QUAD_VS, POST_FS);
    this.starProg = program(gl, STAR_VS, STAR_FS);
    this.warpProg = program(gl, WARP_VS, STAR_FS);

    this.quadVao = gl.createVertexArray();
    gl.bindVertexArray(this.quadVao);
    const qb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, qb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    this.maxTexture = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.fbo = null;
  }

  resize(w, h) {
    const gl = this.gl;
    if (this.canvas.width === w && this.canvas.height === h && this.fbo) return;
    this.canvas.width = w;
    this.canvas.height = h;
    if (this.fbo) {
      gl.deleteFramebuffer(this.fbo);
      gl.deleteTexture(this.fboTex);
    }
    this.fboTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.fboTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.fboTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  begin() {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.bindVertexArray(this.quadVao);
  }

  #blend(mode) {
    const gl = this.gl;
    if (mode === 'lighten') {
      gl.blendEquation(gl.MAX);
      gl.blendFunc(gl.ONE, gl.ONE);
    } else if (mode === 'add') {
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.ONE, gl.ONE);
    } else {
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    }
  }

  // p: { tex, rect:[x0,y0,x1,y1] ndc, ainv:[a,b,c,d] (column-major), b:[x,y], aspect:[ax,ay],
  //      alpha, feather, grow, rectFeather, match, srcMean, srcStd, dstMean, dstStd, grade:[e,c,s], blend }
  drawLayer(p) {
    const gl = this.gl;
    const { p: prog, u } = this.layerProg;
    gl.useProgram(prog);
    this.#blend(p.blend);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, p.tex);
    gl.uniform1i(u.u_tex, 0);
    gl.uniform4fv(u.u_rect, p.rect);
    gl.uniform2f(u.u_view, this.canvas.width, this.canvas.height);
    gl.uniformMatrix2fv(u.u_ainv, false, p.ainv);
    gl.uniform2fv(u.u_b, p.b);
    gl.uniform2fv(u.u_aspect, p.aspect);
    gl.uniform1f(u.u_alpha, p.alpha);
    gl.uniform1f(u.u_feather, p.feather);
    gl.uniform1f(u.u_grow, p.grow);
    gl.uniform1f(u.u_rectFeather, p.rectFeather);
    gl.uniform1f(u.u_match, p.match);
    gl.uniform3fv(u.u_srcMean, p.srcMean);
    gl.uniform3fv(u.u_srcStd, p.srcStd);
    gl.uniform3fv(u.u_dstMean, p.dstMean);
    gl.uniform3fv(u.u_dstStd, p.dstStd);
    gl.uniform3fv(u.u_grade, p.grade);
    gl.uniform1f(u.u_bias, p.bias || 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // center in device px (GL coords, y up), radius/halo in px
  drawGlow({ center, radius, halo, color, intensity = 1, spikes = 0 }) {
    const gl = this.gl;
    const { p: prog, u } = this.glowProg;
    const ext = radius + halo * 16;
    const W = this.canvas.width, H = this.canvas.height;
    const x0 = ((center[0] - ext) / W) * 2 - 1, x1 = ((center[0] + ext) / W) * 2 - 1;
    const y0 = ((center[1] - ext) / H) * 2 - 1, y1 = ((center[1] + ext) / H) * 2 - 1;
    if (x1 < -1 || x0 > 1 || y1 < -1 || y0 > 1) return;
    gl.useProgram(prog);
    this.#blend('add');
    gl.uniform4f(u.u_rect, Math.max(-1, x0), Math.max(-1, y0), Math.min(1, x1), Math.min(1, y1));
    gl.uniform2fv(u.u_center, center);
    gl.uniform1f(u.u_radius, radius);
    gl.uniform1f(u.u_halo, halo);
    gl.uniform3fv(u.u_color, color);
    gl.uniform1f(u.u_intensity, intensity);
    gl.uniform1f(u.u_spikes, spikes);
    gl.uniform1f(u.u_ext, ext);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  drawRing({ center, r, width, color }) {
    const gl = this.gl;
    const { p: prog, u } = this.ringProg;
    gl.useProgram(prog);
    this.#blend('add');
    gl.uniform4f(u.u_rect, -1, -1, 1, 1);
    gl.uniform2fv(u.u_center, center);
    gl.uniform1f(u.u_r, r);
    gl.uniform1f(u.u_w, width);
    gl.uniform4fv(u.u_color, color);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // stars: Float32Array [x, y, r, g, b, size] per star (x,y in octave units)
  setStars(data) {
    const gl = this.gl;
    this.starVao = gl.createVertexArray();
    gl.bindVertexArray(this.starVao);
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 24, 8);
    gl.bindVertexArray(null);
  }

  // batches: [{ first, count, off:[x,y], m:[a,b,c,d], alpha }]
  drawStars(batches, pxScale) {
    if (!this.starVao || !batches.length) return;
    const gl = this.gl;
    const { p: prog, u } = this.starProg;
    gl.useProgram(prog);
    this.#blend('add');
    gl.bindVertexArray(this.starVao);
    gl.uniform1f(u.u_px, pxScale);
    for (const bt of batches) {
      gl.uniform2fv(u.u_off, bt.off);
      gl.uniformMatrix2fv(u.u_m, false, bt.m);
      gl.uniform1f(u.u_alpha, bt.alpha);
      gl.drawArrays(gl.POINTS, bt.first, bt.count);
    }
    gl.bindVertexArray(this.quadVao);
  }

  // data: Float32Array [x, y, r, g, b, size, z0] per star
  setWarp(data) {
    const gl = this.gl;
    this.warpVao = gl.createVertexArray();
    gl.bindVertexArray(this.warpVao);
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 8);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 28, 24);
    gl.bindVertexArray(null);
    this.warpCount = data.length / 7;
  }

  drawWarp({ phase, aspect, alpha, px }) {
    if (!this.warpVao || alpha <= 0.002) return;
    const gl = this.gl;
    const { p: prog, u } = this.warpProg;
    gl.useProgram(prog);
    this.#blend('add');
    gl.bindVertexArray(this.warpVao);
    gl.uniform1f(u.u_phase, phase);
    gl.uniform1f(u.u_aspect, aspect);
    gl.uniform1f(u.u_alpha, alpha);
    gl.uniform1f(u.u_px, px);
    gl.drawArrays(gl.POINTS, 0, this.warpCount);
    gl.bindVertexArray(this.quadVao);
  }

  end({ blur = 0, time = 0, grain = 0.02, vignette = 0.35 }) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.disable(gl.BLEND);
    const { p: prog, u } = this.postProg;
    gl.useProgram(prog);
    gl.bindVertexArray(this.quadVao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.fboTex);
    gl.uniform1i(u.u_scene, 0);
    gl.uniform4f(u.u_rect, -1, -1, 1, 1);
    gl.uniform2f(u.u_view, this.canvas.width, this.canvas.height);
    gl.uniform1f(u.u_blur, blur);
    gl.uniform1f(u.u_time, time);
    gl.uniform1f(u.u_grain, grain);
    gl.uniform1f(u.u_vignette, vignette);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}
