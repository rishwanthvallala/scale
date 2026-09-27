"""Space-filling (CPK) sphere renderer in numpy/OpenCV.

Orthographic camera, exact per-pixel sphere z-buffer, world-space analytic ambient occlusion
(sphere-occlusion approximation over neighbour atoms, so it is identical at every zoom level),
shadow map for the key light with soft PCF, Blinn-Phong specular, rim light, depth fog and a dark
vignetted background. All lighting lengths are in scene units (Angstrom), so several views of the
same scene at different zooms match exactly. Shading runs in row bands to bound memory.
"""
import numpy as np
import cv2
from scipy.spatial import cKDTree


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def euler(ax=0, ay=0, az=0):
    """Rotation matrix from Euler angles (deg) about x, then y, then z."""
    ax, ay, az = np.radians([ax, ay, az])
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


ID_BITS = 22


def splat_keys(px, py, pz, pr, W, H, max_frag=4_000_000):
    """Exact sphere z-buffer. px,py pixel centres (x right, y down); pz depth px (bigger = closer);
    pr radius px. Returns int64 key buffer (H*W,), -1 where empty; key = (depth_q << ID_BITS) | index."""
    n = len(px)
    assert n < (1 << ID_BITS)
    keys = np.full(H * W, -1, dtype=np.int64)
    if n == 0:
        return keys
    zmin = float((pz - pr).min())
    Rint = np.ceil(pr).astype(np.int64) + 1
    order = np.argsort(Rint, kind="stable")
    Rs = Rint[order]
    groups = np.split(order, np.flatnonzero(np.diff(Rs)) + 1)
    for g in groups:
        R = int(Rint[g[0]])
        off = np.arange(-R, R + 1)
        dx, dy = np.meshgrid(off, off)
        dx = dx.ravel(); dy = dy.ravel()
        m = max(1, max_frag // len(dx))
        for s in range(0, len(g), m):
            ids = g[s:s + m]
            cx = px[ids][:, None]; cy = py[ids][:, None]
            ix = np.floor(cx).astype(np.int64) + dx[None, :]
            iy = np.floor(cy).astype(np.int64) + dy[None, :]
            fx = ix + 0.5 - cx; fy = iy + 0.5 - cy
            d2 = fx * fx + fy * fy
            r2 = (pr[ids] ** 2)[:, None]
            ok = (d2 < r2) & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            if not ok.any():
                continue
            z = pz[ids][:, None] + np.sqrt(np.maximum(r2 - d2, 0))
            zq = ((z - zmin) * 8.0).astype(np.int64)
            key = (zq << ID_BITS) | ids[:, None].astype(np.int64)
            np.maximum.at(keys, (iy * W + ix)[ok], key[ok])
    return keys


def background(W, H, c_center=(0.075, 0.08, 0.10), c_edge=(0.010, 0.011, 0.016), y0=0, Hfull=None):
    Hf = Hfull or H
    yy, xx = np.mgrid[y0:y0 + H, 0:W].astype(np.float32)
    rx = (xx - W / 2) / (W / 2); ry = (yy - Hf / 2) / (W / 2)
    rr = np.clip(np.sqrt(rx * rx + ry * ry) / 1.15, 0, 1) ** 1.5
    c0 = np.array(c_center, np.float32); c1 = np.array(c_edge, np.float32)
    return c0 * (1 - rr[..., None]) + c1 * rr[..., None]


class Scene:
    def __init__(self, pos, rad, col, ao_cutoff=7.0, kmax=40):
        self.pos = np.asarray(pos, np.float64)
        self.rad = np.asarray(rad, np.float32)
        self.col = np.asarray(col, np.float32)
        tree = cKDTree(self.pos)
        d, idx = tree.query(self.pos, k=kmax + 1, distance_upper_bound=ao_cutoff)
        idx = idx[:, 1:]; d = d[:, 1:]
        idx[~np.isfinite(d)] = -1
        self.nb = idx.astype(np.int64)


def render(scene, R, center, width, W, H, ss=1.5, light=(-0.45, 0.62, 0.64), fill=(0.75, -0.25, 0.55),
           ambient=0.30, key_int=0.95, fill_int=0.22, spec_int=0.30, shininess=36, rim_int=0.22,
           ao_strength=1.0, shadow=True, shadow_soft=0.35, fog_ref=None, fog_range=None, fog_amount=0.6,
           bgfun=background, margin=4.0, band=384, verbose=True):
    """Render `scene` with camera rotation R (rows = camera axes in world), orthographic, `width` units wide,
    centred at world point `center`. Returns (float32 RGB HxWx3 linear, info)."""
    R = np.asarray(R, np.float64)
    center = np.asarray(center, np.float64)
    Wi, Hi = int(round(W * ss)), int(round(H * ss))
    s = Wi / width
    Pw = scene.pos @ R.T                         # world rotated into camera axes
    cen = center @ R.T
    P = (Pw - cen).astype(np.float32)            # camera coords relative to centre
    rad = scene.rad
    halfw = width / 2; halfh = width * Hi / Wi / 2
    vid = np.flatnonzero((np.abs(P[:, 0]) < halfw + rad + 0.1) & (np.abs(P[:, 1]) < halfh + rad + 0.1))
    px = P[vid, 0].astype(np.float64) * s + Wi / 2
    py = -P[vid, 1].astype(np.float64) * s + Hi / 2
    pz = P[vid, 2].astype(np.float64) * s
    pr = rad[vid].astype(np.float64) * s
    if verbose:
        print(f"  render {W}x{H} ss={ss} width={width:.2f} spheres={len(vid)} r_px~{pr.mean():.1f}", flush=True)
    keys = splat_keys(px, py, pz, pr, Wi, Hi).reshape(Hi, Wi)
    mask_id = (1 << ID_BITS) - 1

    Lk = normalize(light).astype(np.float32)
    Lf = normalize(fill).astype(np.float32)
    Hk = normalize(Lk + np.array([0, 0, 1.0])).astype(np.float32)

    def band_geom(y0, y1):
        kb = keys[y0:y1]
        fg = kb >= 0
        gid = vid[np.where(fg, kb & mask_id, 0)]
        yy, xx = np.mgrid[y0:y1, 0:Wi].astype(np.float32)
        X = (xx + 0.5 - Wi / 2) / s
        Y = -(yy + 0.5 - Hi / 2) / s
        C = P[gid]; r = rad[gid]
        dx = X - C[..., 0]; dy = Y - C[..., 1]
        dz = np.sqrt(np.maximum(r * r - dx * dx - dy * dy, 0))
        Z = C[..., 2] + dz
        N = np.stack([dx / r, dy / r, dz / r], -1)
        return fg, gid, X, Y, Z, N

    # ---- light frame + shadow map
    zl = Lk.astype(np.float64)
    up = np.array([0, 1.0, 0]) if abs(zl[1]) < 0.9 else np.array([1.0, 0, 0])
    xl = normalize(np.cross(up, zl)); yl = np.cross(zl, xl)
    Rl = np.stack([xl, yl, zl]).astype(np.float32)
    if shadow:
        lo = np.array([np.inf, np.inf]); hi = -lo.copy(); zlo = np.inf
        for y0 in range(0, Hi, band):
            fg, gid, X, Y, Z, N = band_geom(y0, min(Hi, y0 + band))
            if not fg.any():
                continue
            pts = np.stack([X[fg], Y[fg], Z[fg]], -1) @ Rl.T
            lo = np.minimum(lo, pts[:, :2].min(0)); hi = np.maximum(hi, pts[:, :2].max(0))
            zlo = min(zlo, pts[:, 2].min())
        lo = lo - margin; hi = hi + margin
        Pl = P @ Rl.T
        cast = np.flatnonzero((Pl[:, 0] > lo[0] - rad) & (Pl[:, 0] < hi[0] + rad) &
                              (Pl[:, 1] > lo[1] - rad) & (Pl[:, 1] < hi[1] + rad) & (Pl[:, 2] > zlo - 3.0))
        ext = hi - lo
        sres = s * 0.6
        SW = int(min(6144, max(64, ext[0] * sres))); SH = int(min(6144, max(64, ext[1] * sres)))
        sres = min(SW / ext[0], SH / ext[1])
        lx = (Pl[cast, 0] - lo[0]) * sres; ly = (hi[1] - Pl[cast, 1]) * sres
        lz = Pl[cast, 2] * sres; lr = rad[cast] * sres
        skeys = splat_keys(lx.astype(np.float64), ly.astype(np.float64), lz.astype(np.float64),
                           lr.astype(np.float64), SW, SH).reshape(SH, SW)
        sfg = skeys >= 0
        sid = np.where(sfg, skeys & mask_id, 0)
        del skeys
        gx = (np.arange(SW, dtype=np.float32) + 0.5)[None, :]
        gy = (np.arange(SH, dtype=np.float32) + 0.5)[:, None]
        dd = np.maximum(lr[sid] ** 2 - (gx - lx[sid]) ** 2 - (gy - ly[sid]) ** 2, 0)
        sdepth = np.where(sfg, (lz[sid] + np.sqrt(dd)) / sres, -1e9).astype(np.float32)
        del sid, dd, sfg
        rad_t = shadow_soft * sres
        samples = [(0.0, 0.0)] + [(np.cos(a) * rad_t * k, np.sin(a) * rad_t * k)
                                  for k in (0.33, 0.66, 1.0) for a in np.linspace(0, 2 * np.pi, 9, endpoint=False) + k]

    out = np.empty((Hi, Wi, 3), np.float32)
    nb = scene.nb
    for y0 in range(0, Hi, band):
        y1 = min(Hi, y0 + band)
        fg, gid, X, Y, Z, N = band_geom(y0, y1)
        bh = y1 - y0
        ao = np.ones((bh, Wi), np.float32)
        sh = np.ones((bh, Wi), np.float32)
        fidx = np.flatnonzero(fg.ravel())
        if len(fidx):
            Pc = np.stack([X.ravel()[fidx], Y.ravel()[fidx], Z.ravel()[fidx]], -1)
            Nf = N.reshape(-1, 3)[fidx]
            g = gid.ravel()[fidx]
            # AO over neighbour spheres (world space => zoom independent)
            aoc = np.empty(len(fidx), np.float32)
            for c0 in range(0, len(fidx), 150_000):
                sl = slice(c0, c0 + 150_000)
                nbi = nb[g[sl]]
                valid = nbi >= 0
                nsafe = np.where(valid, nbi, 0)
                d = P[nsafe] - Pc[sl][:, None, :]
                dist2 = np.maximum((d * d).sum(-1), 1e-6)
                cosang = np.maximum((d * Nf[sl][:, None, :]).sum(-1) / np.sqrt(dist2), 0)
                rr = rad[nsafe]
                occ = np.where(valid, cosang * np.minimum(rr * rr / dist2, 1.0), 0).sum(-1)
                aoc[sl] = np.exp(-1.15 * ao_strength * occ)
            ao.ravel()[fidx] = aoc
            if shadow:
                pts = Pc @ Rl.T
                u = (pts[:, 0] - lo[0]) * sres; v = (hi[1] - pts[:, 1]) * sres
                acc = np.zeros(len(u), np.float32)
                # per-pixel random rotation of the PCF pattern turns banding into fine noise
                rot = (np.sin(fidx.astype(np.float64) * 12.9898 + y0 * 78.233) * 43758.5453) % 1.0 * 2 * np.pi
                cr = np.cos(rot).astype(np.float32); sr = np.sin(rot).astype(np.float32)
                for (ou, ov) in samples:
                    ru = ou * cr - ov * sr; rv = ou * sr + ov * cr
                    ui = np.clip((u + ru).astype(np.int64), 0, SW - 1)
                    vi = np.clip((v + rv).astype(np.int64), 0, SH - 1)
                    acc += (sdepth[vi, ui] <= pts[:, 2] + 0.08).astype(np.float32)
                sh.ravel()[fidx] = acc / len(samples)
        alb = scene.col[gid]
        ndl = (N * Lk).sum(-1)
        diff = np.clip((ndl + 0.12) / 1.12, 0, 1) * sh
        ndf = np.clip((N * Lf).sum(-1), 0, 1)
        ndh = np.clip((N * Hk).sum(-1), 0, 1)
        spec = (ndh ** shininess) * sh * spec_int
        rim = (1 - np.clip(N[..., 2], 0, 1)) ** 3 * rim_int
        lit = alb * (ambient * ao + key_int * diff * (0.55 + 0.45 * ao))[..., None] \
            + alb * (fill_int * ndf * ao)[..., None] * np.array([0.85, 0.92, 1.05], np.float32) \
            + spec[..., None] * np.array([1.0, 0.97, 0.92], np.float32) \
            + (rim * ao)[..., None] * np.array([0.55, 0.65, 0.85], np.float32)
        bgimg = bgfun(Wi, bh, y0=y0, Hfull=Hi)
        if fog_ref is not None and fog_range:
            zw = Z + cen[2]
            fog = np.clip((fog_ref - zw) / fog_range, 0, 1) ** 1.2 * fog_amount
            lit = lit * (1 - fog[..., None]) + bgimg * fog[..., None]
        out[y0:y1] = np.where(fg[..., None], lit, bgimg)
    info = dict(keys=keys, vid=vid, s=s, Wi=Wi, Hi=Hi, P=P)
    if ss != 1:
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)
    return out, info


def to_srgb8(img, exposure=1.0, gamma=1 / 2.2):
    x = np.clip(img * exposure, 0, None)
    x = x / (1 + 0.12 * x)
    x = np.clip(x, 0, 1) ** gamma
    return (x * 255 + 0.5).astype(np.uint8)


def project(R, center, width, W, H, world_pt):
    """Normalised image coords (0..1, y down) of a world point in an orthographic view."""
    p = (np.asarray(world_pt, np.float64) - np.asarray(center, np.float64)) @ np.asarray(R).T
    return (p[0] / width + 0.5, 0.5 - p[1] / (width * H / W))
