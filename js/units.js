const LY = 9.4607e15;
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };

export function sup(n) {
  return String(n).split('').map((ch) => SUP[ch] ?? ch).join('');
}

export function powerOfTen(m) {
  return `10${sup(Math.floor(Math.log10(m) + 1e-9))} m`;
}

function sig(x, digits = 3) {
  const v = Number(x.toPrecision(digits));
  return v.toLocaleString('en-US', { maximumFractionDigits: 3 });
}

function words(x, unit) {
  if (x >= 1e12) return `${sig(x / 1e12)} trillion ${unit}`;
  if (x >= 1e9) return `${sig(x / 1e9)} billion ${unit}`;
  if (x >= 1e6) return `${sig(x / 1e6)} million ${unit}`;
  return `${sig(x)} ${unit}`;
}

export function formatLength(m) {
  if (m >= 0.1 * LY) return words(m / LY, 'light-years');
  if (m >= 1e6) return words(m / 1e3, 'km');
  if (m >= 1e3) return `${sig(m / 1e3)} km`;
  if (m >= 1) return `${sig(m)} m`;
  if (m >= 1e-2) return `${sig(m * 1e2)} cm`;
  if (m >= 1e-3) return `${sig(m * 1e3)} mm`;
  if (m >= 1e-6) return `${sig(m * 1e6)} µm`;
  if (m >= 1e-9) return `${sig(m * 1e9)} nm`;
  if (m >= 1e-12) return `${sig(m * 1e12)} pm`;
  return `${sig(m * 1e15)} fm`;
}

// Largest 1/2/5 × 10^n not exceeding x.
export function niceLength(x) {
  const p = Math.pow(10, Math.floor(Math.log10(x)));
  const f = x / p;
  return (f >= 5 ? 5 : f >= 2 ? 2 : 1) * p;
}

// Short ruler labels
export function shortUnit(exp) {
  const m = Math.pow(10, exp);
  if (m >= 1e7) return `10${sup(exp)} m`;
  if (m >= 1e3) return `${sig(m / 1e3)} km`;
  if (m >= 1) return `${sig(m)} m`;
  if (m >= 1e-3) return `${sig(m * 1e3)} mm`;
  if (m >= 1e-6) return `${sig(m * 1e6)} µm`;
  if (m >= 1e-9) return `${sig(m * 1e9)} nm`;
  if (m >= 1e-12) return `${sig(m * 1e12)} pm`;
  return `${sig(m * 1e15)} fm`;
}
