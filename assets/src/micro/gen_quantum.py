"""Generate the sub-molecular levels (all on black, meant for additive blending):
  atom     : carbon atom electron density (1s2 2s2 2p2, hydrogen-like orbitals with Clementi-Raimondi
             effective charges), volumetric column-density glow + tiny nucleus point.
  nucleus  : carbon-12 nucleus, 6 protons + 6 neutrons, relaxed close packing, sphere renderer + glow.
  proton   : illustrative proton: 3 valence quarks (colour charges), Y-shaped gluon flux tubes,
             faint sea-quark pairs, inside a fuzzy glowing boundary.
py gen_quantum.py [--preview] [atom|nucleus|proton ...]
"""
import json, os, sys, time
import numpy as np
import cv2
from sphere_render import Scene, render, normalize, euler

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "gen"); os.makedirs(GEN, exist_ok=True)
PREVIEW = "--preview" in sys.argv
W, H = (960, 540) if PREVIEW else (3840, 2160)
BOHR = 0.529177  # Angstrom


def save(img, lid):
    img8 = (np.clip(img, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)
    p = os.path.join(GEN, lid + (".preview.jpg" if PREVIEW else ".jpg"))
    cv2.imwrite(p, cv2.cvtColor(img8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("  ->", p)
    return p


def colormap(t, stops):
    """t in 0..1 -> RGB via piecewise-linear stops [(t, (r,g,b)), ...]."""
    t = np.clip(t, 0, 1)
    xs = np.array([s[0] for s in stops]); cs = np.array([s[1] for s in stops], np.float32)
    out = np.stack([np.interp(t, xs, cs[:, k]) for k in range(3)], -1)
    return out.astype(np.float32)


# ------------------------------------------------------------------------------------------------
# ATOM
# ------------------------------------------------------------------------------------------------
def carbon_parts(x, y, z):
    """Electron density of ground-state carbon (bohr^-3), positions in bohr, split into core and valence.
    Hydrogen-like radial functions with effective Z: 1s 5.673, 2s 3.217, 2p 3.136 (Clementi & Raimondi 1963);
    the two 2p electrons occupy p_x and p_y (3P ground state)."""
    r = np.sqrt(x * x + y * y + z * z) + 1e-9
    Z1, Z2s, Z2p = 5.673, 3.217, 3.136
    core = 2 * (Z1 ** 3 / np.pi) * np.exp(-2 * Z1 * r)
    s2 = 2 * (Z2s ** 3 / (32 * np.pi)) * (2 - Z2s * r) ** 2 * np.exp(-Z2s * r)
    p2 = (Z2p ** 5 / (32 * np.pi)) * (x * x + y * y) * np.exp(-Z2p * r)
    return core, s2 + p2


def gen_atom(width_A=6.0, edge_A=1.75):
    t0 = time.time()
    Wc, Hc = (W // 2, H // 2) if not PREVIEW else (W, H)
    Rv = euler(58, 0, 28)  # tilt the 2p plane so the (slight) anisotropy reads in 3D
    eu, ew, ev = Rv[:, 0], Rv[:, 1], Rv[:, 2]
    s = width_A / Wc
    us = (np.arange(Wc) + 0.5 - Wc / 2) * s
    ws = -(np.arange(Hc) + 0.5 - Hc / 2) * s
    ts = np.linspace(-3.0, 3.0, 181)
    dt = ts[1] - ts[0]
    colC = np.zeros((Hc, Wc)); colV = np.zeros((Hc, Wc))
    for j0 in range(0, Hc, 16):
        wj = ws[j0:j0 + 16]
        U, Wg, T = np.meshgrid(us, wj, ts, indexing="xy")
        px = (U * eu[0] + Wg * ew[0] + T * ev[0]) / BOHR
        py = (U * eu[1] + Wg * ew[1] + T * ev[1]) / BOHR
        pz = (U * eu[2] + Wg * ew[2] + T * ev[2]) / BOHR
        c, v = carbon_parts(px, py, pz)
        colC[j0:j0 + 16] = c.sum(-1) * dt; colV[j0:j0 + 16] = v.sum(-1) * dt
    # valence glow: log mapping whose floor sits where the column density has fallen at r = edge_A
    v = colV / colV.max()
    row = v[Hc // 2]; xr = np.abs(us)
    D = -np.log10(np.interp(edge_A, xr[Wc // 2:], row[Wc // 2:]))
    t = np.clip((np.log10(v + 1e-12) + D) / D, 0, 1)
    cm = colormap(t, [(0.0, (0, 0, 0)), (0.22, (0.07, 0.02, 0.20)), (0.45, (0.11, 0.14, 0.62)),
                      (0.68, (0.16, 0.45, 1.0)), (0.86, (0.50, 0.82, 1.0)), (1.0, (0.86, 0.96, 1.0))])
    img = cm * (t ** 1.35)[..., None] * 0.95
    # the 1s core: compact and bright
    c = colC / colC.max()
    img += (c ** 0.55)[..., None] * np.array([0.85, 0.93, 1.0], np.float32) * 0.8
    img = img.astype(np.float32)
    if not PREVIEW:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
    # subtle granular texture (a probability cloud, not a solid ball)
    rng = np.random.default_rng(3)
    g = fbm(W, H, rng, 7)
    fine = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.2 * W / 3840)
    img *= (1 + 0.10 * g + 0.10 * fine / (fine.std() + 1e-6))[..., None]
    # nucleus: tiny bright point with a soft glare (the real nucleus would be 1/50000 of this width)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rr = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (W / 3840)
    img += (np.exp(-(rr / 3.5) ** 2) * 2.0 + np.exp(-(rr / 16.0) ** 2) * 0.35)[..., None]         * np.array([1.0, 0.9, 0.8], np.float32)
    img = np.clip(img, 0, None)
    print(f"  atom rendered in {time.time() - t0:.1f}s (log range {D:.2f} decades)")
    save(img, "atom")
    return dict(width_A=width_A)


# ------------------------------------------------------------------------------------------------
# NUCLEUS
# ------------------------------------------------------------------------------------------------
def pack_nucleons(n=12, r=0.86, seed=5):
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 1.0, (n, 3))
    for it in range(4000):
        d = p[:, None, :] - p[None, :, :]
        dist = np.linalg.norm(d, axis=-1) + np.eye(n)
        over = np.clip(2 * r * 0.93 - dist, 0, None)  # nucleons overlap slightly (they are fuzzy)
        np.fill_diagonal(over, 0)
        push = (d / dist[..., None] * over[..., None]).sum(1) * 0.5
        p += push - p * 0.02  # push apart + pull to centre
        p -= p.mean(0)
    return p


def black_bg(Wi, bh, y0=0, Hfull=None):
    return np.zeros((bh, Wi, 3), np.float32)


def bloom(img, sigmas=(6, 24, 80), weights=(0.18, 0.12, 0.08)):
    out = img.copy()
    sc = img.shape[1] / 3840
    for s, w in zip(sigmas, weights):
        out += w * cv2.GaussianBlur(img, (0, 0), s * sc)
    return out


def gen_nucleus(width_fm=12.0):
    p = pack_nucleons()
    n = len(p)
    rng = np.random.default_rng(11)
    kinds = np.array([1] * 6 + [0] * 6); rng.shuffle(kinds)
    # make sure a proton sits front-and-centre-ish (it becomes the focus for the proton level)
    Rv = euler(-18, 24, 0)
    cam = p @ Rv.T
    front = np.argsort(-cam[:, 2])[:4]
    best = front[np.argmin(np.linalg.norm(cam[front, :2] - [0.35, 0.25], axis=1))]
    if kinds[best] == 0:
        swap = np.flatnonzero(kinds == 1)[0]
        kinds[best], kinds[swap] = 1, 0
    col = np.where(kinds[:, None] == 1, np.array([0.60, 0.085, 0.05]), np.array([0.20, 0.25, 0.36])).astype(np.float32)
    rad = np.full(n, 0.86, np.float32)
    sc = Scene(p, rad, col, ao_cutoff=4.0)
    img, info = render(sc, Rv, np.zeros(3), width_fm, W, H, ss=1.5, bgfun=black_bg, ambient=0.24,
                       key_int=1.0, fill_int=0.28, spec_int=0.20, shininess=20, rim_int=0.45,
                       shadow_soft=0.12, ao_strength=0.9, light=(-0.5, 0.55, 0.67), fill=(0.7, -0.3, 0.6))
    # soft "fuzzy" nucleon edges + glow
    img = bloom(img, sigmas=(4, 16, 55), weights=(0.20, 0.14, 0.05))
    # radial falloff so the frame edges are black
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rr = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (W / width_fm)  # fm
    img *= np.clip((5.2 - rr) / 2.0, 0, 1)[..., None]  # fade to black toward the corners (additive blending)
    save(img, "nucleus")
    focus_xy = cam[best, :2]
    focus = (0.5 + focus_xy[0] / width_fm, 0.5 - focus_xy[1] / (width_fm * H / W))
    return dict(focus=[round(float(focus[0]), 4), round(float(focus[1]), 4)], width_fm=width_fm)


# ------------------------------------------------------------------------------------------------
# PROTON (illustrative)
# ------------------------------------------------------------------------------------------------
def fbm(Wn, Hn, rng, octaves=5):
    out = np.zeros((Hn, Wn), np.float32); amp = 1.0; tot = 0
    for o in range(octaves):
        f = 2 ** (o + 2)
        g = rng.normal(0, 1, (f * Hn // Wn + 2, f + 2)).astype(np.float32)
        out += amp * cv2.resize(g, (Wn, Hn), interpolation=cv2.INTER_CUBIC)
        tot += amp; amp *= 0.55
    return out / tot


def gen_proton(width_fm=3.6):
    """Illustrative proton: fuzzy lit sphere (charge radius 0.84 fm) filled with a turbulent gluon field,
    three valence quarks in the three colour charges joined by a Y-shaped flux tube, plus faint sea quarks."""
    rng = np.random.default_rng(21)
    sc = W / width_fm  # px per fm
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    X = (xx - W / 2) / sc; Y = -(yy - H / 2) / sc
    Rr = np.sqrt(X * X + Y * Y)
    Rp = 0.84
    # --- body: soft-edged sphere, lit from the upper left like the nucleons in the nucleus level
    rn = np.clip(Rr / 0.9, 0, 1)
    nz = np.sqrt(np.clip(1 - rn ** 2, 0, 1))
    L = normalize([-0.5, 0.55, 0.67])
    lam = np.clip((-X / 0.9) * -L[0] * -1 + (Y / 0.9) * L[1] + nz * L[2], 0, 1) if False else         np.clip((X / 0.9) * L[0] + (Y / 0.9) * L[1] + nz * L[2], 0, 1)
    edge = 1 / (1 + np.exp((Rr - 0.86) / 0.07))  # fuzzy boundary
    n1 = cv2.GaussianBlur(fbm(W, H, rng, 6), (0, 0), 0.02 * sc)
    n2 = cv2.GaussianBlur(fbm(W, H, rng, 6), (0, 0), 0.02 * sc)
    n1 /= n1.std() + 1e-6; n2 /= n2.std() + 1e-6
    turb = np.clip(0.85 + 0.22 * n1 + 0.10 * np.abs(n2), 0.3, 2.0)
    body = edge * (0.30 + 0.70 * lam) * turb
    img = body[..., None] * np.array([0.50, 0.085, 0.05], np.float32)
    # faint magenta/orange swirls = gluon field fluctuations
    sw = np.clip(0.5 + 0.25 * n2, 0, 1) ** 3 * edge
    img += sw[..., None] * np.array([0.35, 0.08, 0.20], np.float32) * 0.5
    img += (np.exp(-(Rr / 1.0) ** 2) * 0.05)[..., None] * np.array([1.0, 0.4, 0.25], np.float32)
    # --- valence quarks (u, u, d) on a triangle
    ang0 = np.radians(105)
    qpos = [np.array([np.cos(ang0 + k * 2 * np.pi / 3), np.sin(ang0 + k * 2 * np.pi / 3)]) * 0.40 +
            rng.normal(0, 0.04, 2) for k in range(3)]
    qcol = [np.array([1.0, 0.18, 0.12]), np.array([0.20, 1.0, 0.30]), np.array([0.22, 0.42, 1.0])]
    cen = np.mean(qpos, axis=0) + np.array([0.02, -0.03])
    # --- gluon flux tubes: coiled springs from the Y junction to each quark
    tube = np.zeros((H, W, 3), np.float32)
    for k in range(3):
        a, b = cen, qpos[k]
        Lk = np.linalg.norm(b - a); d = (b - a) / Lk; nrm = np.array([-d[1], d[0]])
        ts = np.linspace(0, 1, 1500)
        coil = 0.028 * np.sin(ts * 2 * np.pi * 7.5) * np.sin(np.pi * np.clip(ts * 1.15, 0, 1))
        pts = a[None] + ts[:, None] * (b - a)[None] + coil[:, None] * nrm[None]
        pix = np.stack([pts[:, 0] * sc + W / 2, -pts[:, 1] * sc + H / 2], -1)
        m = np.zeros((H, W), np.uint8)
        cv2.polylines(m, [np.round(pix * 16).astype(np.int32).reshape(-1, 1, 2)], False, 255,
                      thickness=max(1, int(0.012 * sc)), lineType=cv2.LINE_AA, shift=4)
        mf = m.astype(np.float32) / 255
        c = 0.45 * qcol[k] + 0.55 * np.array([1.0, 0.80, 0.55])
        tube += mf[..., None] * c
    tube = cv2.GaussianBlur(tube, (0, 0), 0.006 * sc) * 1.2 + cv2.GaussianBlur(tube, (0, 0), 0.035 * sc) * 1.6         + cv2.GaussianBlur(tube, (0, 0), 0.09 * sc) * 1.2
    img += tube * 0.8
    # --- sea quark / antiquark pairs: small faint glows
    for i in range(18):
        r = Rp * np.sqrt(rng.uniform(0.05, 1)) * 0.9; th = rng.uniform(0, 2 * np.pi)
        c = np.array([r * np.cos(th), r * np.sin(th)])
        cc = [qcol[0], qcol[1], qcol[2], np.array([1.0, 0.9, 0.6])][rng.integers(4)]
        sz = rng.uniform(0.012, 0.024)
        for off in (-1, 1):  # a pair
            o = c + off * np.array([np.cos(th + 1.3), np.sin(th + 1.3)]) * sz * 1.8
            g = np.exp(-(((X - o[0]) ** 2 + (Y - o[1]) ** 2) / (sz ** 2)))
            img += g[..., None] * cc * rng.uniform(0.18, 0.35)
    # --- valence quark cores
    for k in range(3):
        d2 = (X - qpos[k][0]) ** 2 + (Y - qpos[k][1]) ** 2
        img += (np.exp(-d2 / 0.030 ** 2) * 2.6)[..., None] * (0.30 * qcol[k] + 0.70)
        img += (np.exp(-d2 / 0.070 ** 2) * 0.9)[..., None] * qcol[k]
        img += (np.exp(-d2 / 0.16 ** 2) * 0.16)[..., None] * qcol[k]
    img = bloom(img, sigmas=(6, 30, 110), weights=(0.10, 0.08, 0.05))
    img *= np.clip((1.55 - Rr) / 0.45, 0, 1)[..., None]  # black beyond ~1.5 fm
    save(img, "proton")
    return dict(width_fm=width_fm)


if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or ["atom", "nucleus", "proton"]
    meta = {}
    mp = os.path.join(GEN, "quantum_meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    for w in which:
        t0 = time.time()
        meta[w] = {"atom": gen_atom, "nucleus": gen_nucleus, "proton": gen_proton}[w]()
        print(f"{w} done in {time.time() - t0:.1f}s")
    if not PREVIEW:
        json.dump(meta, open(mp, "w"), indent=2)
