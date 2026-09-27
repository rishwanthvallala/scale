// End-to-end check that behaves like a user: real mouse-wheel scrolling from the top to
// the bottom, screenshots along the way, Play / ruler / credits / editor clicks, a phone
// viewport with touch scrolling, frame-rate sampling and console-error capture.
//
//   node tools/qa.mjs [url] [outDir]
//   default url: http://localhost:8080/   (run start.bat first)
//
// Needs Google Chrome or Microsoft Edge installed. Writes PNG/JPEG shots + report.json.

import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { join, resolve } from 'node:path';

const URL_ = process.argv[2] || 'http://localhost:8080/';
const OUT = resolve(process.argv[3] || 'qa-out');
const PORT = 9333 + Math.floor(Math.random() * 500);
mkdirSync(OUT, { recursive: true });

const browsers = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
];
const exe = browsers.find((b) => existsSync(b));
if (!exe) throw new Error('No Chrome/Edge found');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const gpu = !process.env.QA_SOFTWARE;
const proc = spawn(exe, [
  '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${join(OUT, '.profile')}`,
  '--window-size=1600,900', '--hide-scrollbars', '--no-first-run', '--no-default-browser-check',
  '--autoplay-policy=no-user-gesture-required',
  ...(gpu ? ['--enable-gpu', '--use-angle=d3d11', '--ignore-gpu-blocklist'] : ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']),
  'about:blank',
], { stdio: 'ignore' });

let ws, nextId = 1;
const pending = new Map();
const events = [];
const listeners = [];

async function connect() {
  for (let k = 0; k < 50; k++) {
    try {
      const list = await fetch(`http://127.0.0.1:${PORT}/json/list`).then((r) => r.json());
      const page = list.find((t) => t.type === 'page');
      if (page) return page.webSocketDebuggerUrl;
    } catch { /* not up yet */ }
    await sleep(200);
  }
  throw new Error('Chrome did not start');
}

function send(method, params = {}) {
  const id = nextId++;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((res, rej) => pending.set(id, { res, rej, method }));
}

async function evaluate(expr) {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
  if (r.exceptionDetails) throw new Error(`${expr}: ${r.exceptionDetails.text} ${r.exceptionDetails.exception?.description || ''}`);
  return r.result.value;
}

let shotN = 0;
const shots = [];
async function shot(name, note = {}) {
  const r = await send('Page.captureScreenshot', { format: 'jpeg', quality: 85 });
  const file = `${String(shotN++).padStart(2, '0')}-${name}.jpg`;
  writeFileSync(join(OUT, file), Buffer.from(r.data, 'base64'));
  shots.push({ file, ...note });
  return file;
}

const STATE = `(() => {
  const a = window.app; if (!a || !a.cam) return null;
  return { s: +a.sSmooth.toFixed(3), target: +a.targetS().toFixed(3), len: +a.path.length.toFixed(2),
    fov: document.getElementById('fov').textContent, label: document.getElementById('story-label').textContent,
    storyOpacity: document.getElementById('story-label').parentElement.style.opacity,
    anchor: a.scene.layers[a.cam.anchor].id, pending: a.store.pending(), playing: !!a.playing,
    // sharpness: screen pixels per image pixel for the image that fills the screen (>2 soft, >3 blurry)
    mag: (() => { const d = (a.draws || []).filter((d) => d.drawn && d.alpha > 0.5 && d.coverFrac >= 0.9).pop(); return d ? +d.mag.toFixed(2) : null; })(),
    dominant: (() => { const d = (a.draws || []).filter((d) => d.drawn && d.alpha > 0.5 && d.coverFrac >= 0.9).pop(); return d ? d.l.id : null; })(),
    scrollY: Math.round(scrollY), maxScroll: document.documentElement.scrollHeight - innerHeight };
})()`;

async function waitSettled(maxMs = 8000) {
  const t0 = Date.now();
  let st;
  while (Date.now() - t0 < maxMs) {
    st = await evaluate(STATE);
    if (st && Math.abs(st.s - st.target) < 0.002 && st.pending === 0) return st;
    await sleep(150);
  }
  return { ...st, timedOut: true };
}

async function wheel(notches, x = 800, y = 450, gap = 16) {
  for (let k = 0; k < notches; k++) {
    await send('Input.dispatchMouseEvent', { type: 'mouseWheel', x, y, deltaX: 0, deltaY: 100 });
    await sleep(gap);
  }
}

async function click(selector) {
  const box = await evaluate(`(() => { const r = document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; })()`);
  for (const type of ['mousePressed', 'mouseReleased']) {
    await send('Input.dispatchMouseEvent', { type, x: box[0], y: box[1], button: 'left', clickCount: 1 });
  }
}

async function key(k, code, keyCode) {
  await send('Input.dispatchKeyEvent', { type: 'keyDown', key: k, code, windowsVirtualKeyCode: keyCode });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', key: k, code, windowsVirtualKeyCode: keyCode });
}

async function fps(ms = 2000, whileScrolling = true) {
  await evaluate(`window.__f = 0; window.__run = true; (function t(){ if (!window.__run) return; window.__f++; requestAnimationFrame(t); })(); 0`);
  const t0 = Date.now();
  if (whileScrolling) { while (Date.now() - t0 < ms) await wheel(1, 800, 450, 40); } else await sleep(ms);
  const frames = await evaluate('window.__run = false, window.__f');
  return +(frames / ((Date.now() - t0) / 1000)).toFixed(1);
}

async function main() {
  ws = new WebSocket(await connect());
  await new Promise((r) => (ws.onopen = r));
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id && pending.has(msg.id)) {
      const p = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? p.rej(new Error(`${p.method}: ${msg.error.message}`)) : p.res(msg.result);
    } else if (msg.method) {
      events.push(msg);
      listeners.forEach((f) => f(msg));
    }
  };
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Log.enable');
  await send('Network.enable');

  const report = { url: URL_, gpu, desktop: [], mobile: [], checks: {}, errors: [], failedRequests: [] };
  listeners.push((m) => {
    if (m.method === 'Runtime.exceptionThrown') report.errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
    if (m.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(m.params.type)) report.errors.push(`console.${m.params.type}: ${m.params.args.map((a) => a.value ?? a.description).join(' ')}`);
    if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') report.errors.push(`log: ${m.params.entry.text} ${m.params.entry.url || ''}`);
    if (m.method === 'Network.responseReceived' && m.params.response.status >= 400) report.failedRequests.push(`${m.params.response.status} ${m.params.response.url}`);
  });

  // ---------------------------------------------------------------- desktop
  await send('Page.navigate', { url: URL_ });
  await sleep(2500);
  report.renderer = await evaluate(`(() => { const gl = window.app?.renderer.gl; if (!gl) return 'no app'; const d = gl.getExtension('WEBGL_debug_renderer_info'); return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); })()`);
  report.lite = await evaluate('window.app?.small');
  let st = await waitSettled();
  await shot('landing', st);
  report.desktop.push(st);

  // scroll through everything the way a person with a mouse wheel would: bursts of notches
  let guard = 0;
  while (guard++ < 400) {
    await wheel(6);
    st = await waitSettled();
    report.desktop.push(st);
    await shot(`scroll-${(st.anchor || 'x').replace(/[^a-z0-9-]/gi, '')}`, st);
    if (st.scrollY >= st.maxScroll - 2) break;
  }
  report.checks.reachedEnd = st.scrollY >= st.maxScroll - 2;

  // frame rate while scrolling at a few places
  report.fps = {};
  for (const frac of [0.1, 0.45, 0.8]) {
    await evaluate(`scrollTo(0, ${frac} * (document.documentElement.scrollHeight - innerHeight)); 0`);
    await waitSettled();
    report.fps[`at${Math.round(frac * 100)}%`] = await fps(2000, true);
  }
  report.fps.idle = await fps(1500, false);

  // Play button runs the zoom by itself
  await evaluate('scrollTo(0, 0); 0');
  await waitSettled();
  await click('#btn-play');
  await sleep(4000);
  const afterPlay = await evaluate(STATE);
  await shot('play-4s', afterPlay);
  report.checks.playAdvances = afterPlay.s > 1 && afterPlay.playing;
  await key(' ', 'Space', 32);
  await sleep(500);
  const paused = await evaluate(STATE);
  report.checks.spacePauses = !paused.playing;

  // ruler jump
  const rulerBox = await evaluate(`(() => { const r = document.getElementById('ruler').getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height * 0.7]; })()`);
  for (const type of ['mousePressed', 'mouseReleased']) await send('Input.dispatchMouseEvent', { type, x: rulerBox[0], y: rulerBox[1], button: 'left', clickCount: 1 });
  await sleep(2500);
  const afterRuler = await waitSettled();
  await shot('ruler-jump-70pct', afterRuler);
  report.checks.rulerJumps = afterRuler.scrollY > paused.scrollY + 100;

  await click('#btn-credits');
  await sleep(300);
  await shot('credits-panel');
  report.checks.creditsItems = await evaluate('document.querySelectorAll("#credits-list li").length');
  await click('#credits [data-close]');
  await key('e', 'KeyE', 69);
  await sleep(400);
  await shot('editor-panel');
  report.checks.editorOpensAfterClickingButtons = await evaluate('!document.getElementById("editor").hidden');
  await key('e', 'KeyE', 69);

  // ---------------------------------------------------------------- phone
  await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 3, mobile: true });
  await send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 5 });
  await send('Page.navigate', { url: URL_ });
  await sleep(2500);
  report.mobileLite = await evaluate('window.app?.small');
  st = await waitSettled();
  await shot('phone-landing', st);
  report.mobile.push(st);
  for (let k = 0; k < 8; k++) {
    // four quick finger drags upward per step (synthesizeScrollGesture doesn't scroll in headless)
    for (let d = 0; d < 4; d++) {
      await send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: 195, y: 720 }] });
      for (let m = 1; m <= 12; m++) {
        await send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: 195, y: 720 - m * 45 }] });
        await sleep(12);
      }
      await send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
      await sleep(60);
    }
    st = await waitSettled();
    report.mobile.push(st);
    await shot(`phone-swipe-${k}`, st);
  }
  report.checks.phoneSwipeScrolls = report.mobile[report.mobile.length - 1].scrollY > 1000;
  report.checks.phoneUsesLiteTextures = report.mobileLite === true;
  report.checks.phoneNoHorizontalScroll = await evaluate('document.documentElement.scrollWidth <= innerWidth + 1');

  report.shots = shots;
  writeFileSync(join(OUT, 'report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ renderer: report.renderer, checks: report.checks, fps: report.fps, errors: report.errors.slice(0, 20), failedRequests: report.failedRequests.slice(0, 20), shots: shots.length }, null, 2));
}

main()
  .catch((e) => { console.error('QA FAILED:', e.message); process.exitCode = 1; })
  .finally(() => { try { ws?.close(); } catch {} proc.kill(); });
