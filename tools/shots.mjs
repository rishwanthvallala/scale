// Screenshots at chosen stops of the zoom, after every texture has loaded.
//   node tools/shots.mjs <outDir> <url> <stop> [<stop> ...]
// A stop is "<layer id>" or "<layer id>@<0..1 toward the next layer>" or "s=<scroll units>".
// Example: node tools/shots.mjs shots http://localhost:8080/ s=0 milky-way milky-way@0.5

import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { join, resolve } from 'node:path';

const [outArg, base, ...stops] = process.argv.slice(2);
const OUT = resolve(outArg || 'shots');
mkdirSync(OUT, { recursive: true });
const W = Number(process.env.SHOT_W || 1920), H = Number(process.env.SHOT_H || 1080);
const exe = ['C:/Program Files/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'].find(existsSync);
const PORT = 9800 + Math.floor(Math.random() * 100);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const proc = spawn(exe, ['--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${join(OUT, '.profile')}`,
  `--window-size=${W},${H}`, '--hide-scrollbars', '--enable-gpu', '--use-angle=d3d11', '--ignore-gpu-blocklist', 'about:blank'], { stdio: 'ignore' });

let wsUrl;
for (let k = 0; k < 50 && !wsUrl; k++) {
  try { wsUrl = (await fetch(`http://127.0.0.1:${PORT}/json/list`).then((r) => r.json())).find((t) => t.type === 'page')?.webSocketDebuggerUrl; } catch {}
  await sleep(200);
}
const ws = new WebSocket(wsUrl);
await new Promise((r) => (ws.onopen = r));
let id = 1;
const pend = new Map();
ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && pend.has(d.id)) { pend.get(d.id)(d); pend.delete(d.id); } };
const send = (method, params = {}) => new Promise((r) => { const i = id++; pend.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (e) => (await send('Runtime.evaluate', { expression: e, returnByValue: true })).result.result?.value;

try {
  for (const [n, stop] of stops.entries()) {
    const q = stop.startsWith('s=') ? stop : `at=${stop.split('@')[0]}&f=${stop.split('@')[1] || 0}`;
    await send('Page.navigate', { url: `${base}?${q}&pin` });
    await sleep(1200);
    for (let k = 0; k < 120; k++) {
      if ((await ev('window.app && window.app.store.pending() === 0 && document.fonts.status === "loaded"')) === true) break;
      await sleep(250);
    }
    await sleep(700);
    const r = await send('Page.captureScreenshot', { format: 'jpeg', quality: 90 });
    const file = join(OUT, `${String(n + 1).padStart(2, '0')}-${stop.replace(/[^a-z0-9.-]/gi, '_')}.jpg`);
    writeFileSync(file, Buffer.from(r.result.data, 'base64'));
    console.log(file, '|', await ev(`document.getElementById('story-label').textContent + ' | ' + document.getElementById('fov').textContent`));
  }
} finally {
  ws.close();
  proc.kill();
}
