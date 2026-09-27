"""Build all-heavy-atom B-DNA along arbitrary 3D curves, using base pairs taken from PDB 1BNA
(Drew & Dickerson 1981, B-DNA dodecamer) and the helical screw operation fitted from 1BNA itself.
"""
import os
import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
PDB = os.path.join(HERE, "data", "1BNA.pdb")
BACKBONE = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]

# van der Waals radii (Angstrom, Bondi) and a muted CPK palette (linear RGB)
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80}
COLORS = {
    "C": (0.34, 0.34, 0.355),
    "N": (0.14, 0.24, 0.56),
    "O": (0.56, 0.115, 0.09),
    "P": (0.82, 0.43, 0.08),
}


def parse_pdb(path=PDB):
    res = {}
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip(); ch = line[21]; rn = int(line[22:26]); el = line[76:78].strip()
        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        res.setdefault((ch, rn), []).append((name, el, np.array(xyz)))
    return res


def kabsch(A, B):
    """R, t minimising |R A + t - B|."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def fit_screw(res):
    src, dst = [], []
    for k in range(2, 11):
        for (a, b) in ((("A", k), ("A", k + 1)), (("B", 25 - k), ("B", 24 - k))):
            na = {n: x for n, e, x in res[a]}; nb = {n: x for n, e, x in res[b]}
            for n in BACKBONE:
                if n in na and n in nb:
                    src.append(na[n]); dst.append(nb[n])
    R, t = kabsch(np.array(src), np.array(dst))
    w, V = np.linalg.eig(R)
    u = np.real(V[:, np.argmin(np.abs(w - 1))]); u /= np.linalg.norm(u)
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    # sign of rotation about u
    tmp = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(u, tmp); e1 /= np.linalg.norm(e1)
    if np.dot(np.cross(e1, R @ e1), u) < 0:
        u = -u
    rise = float(np.dot(t, u))
    # point on axis: (I - R) p = t - rise u
    p0, *_ = np.linalg.lstsq(np.eye(3) - R, t - rise * u, rcond=None)
    return dict(R=R, t=t, u=u, angle=ang, rise=rise, p0=p0)


def templates():
    """Return list of base-pair templates in the helix frame at step index 0.
    Each template: (xyz (n,3) with z along the helix axis, elements list, is_backbone mask)."""
    res = parse_pdb()
    sc = fit_screw(res)
    u = sc["u"]; p0 = sc["p0"]
    e1 = np.cross(u, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
    F = np.stack([e1, e2, u])  # world -> helix frame
    th = np.radians(sc["angle"]); rise = sc["rise"]
    out = []
    for j in range(2, 12):  # skip terminal pairs (their 5' nucleotides lack phosphate)
        atoms = res[("A", j)] + res[("B", 25 - j)]
        xyz = np.array([x for n, e, x in atoms])
        el = [e for n, e, x in atoms]
        bb = np.array([n in BACKBONE for n, e, x in atoms])
        h = (xyz - p0) @ F.T
        # undo j screw steps: rotate by -th*j about z and shift -rise*j
        a = -th * j
        c, s = np.cos(a), np.sin(a)
        x = c * h[:, 0] - s * h[:, 1]; y = s * h[:, 0] + c * h[:, 1]; z = h[:, 2] - rise * j
        out.append((np.stack([x, y, z], -1), el, bb))
    zc = np.mean([t[0][:, 2].mean() for t in out])
    out = [(t[0] - [0, 0, zc], t[1], t[2]) for t in out]
    return out, sc


def rmf_frames(C, T):
    """Rotation-minimising frames along sampled curve C (n,3) with unit tangents T."""
    n = len(C)
    N = np.zeros_like(C); B = np.zeros_like(C)
    a = np.array([0, 0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0, 0])
    N[0] = np.cross(a, T[0]); N[0] /= np.linalg.norm(N[0]); B[0] = np.cross(T[0], N[0])
    for i in range(n - 1):  # double reflection method (Wang et al. 2008)
        v1 = C[i + 1] - C[i]; c1 = v1 @ v1
        rL = N[i] - (2 / c1) * (v1 @ N[i]) * v1
        tL = T[i] - (2 / c1) * (v1 @ T[i]) * v1
        v2 = T[i + 1] - tL; c2 = v2 @ v2
        N[i + 1] = rL - (2 / c2) * (v2 @ rL) * v2 if c2 > 1e-12 else rL
        N[i + 1] /= np.linalg.norm(N[i + 1])
        B[i + 1] = np.cross(T[i + 1], N[i + 1])
    return N, B


def dna_along_curve(ctrl, tmpl, sc, rng, phase=0.0):
    """Place base pairs every `rise` along a Catmull-Rom-ish cubic spline through control points."""
    ctrl = np.asarray(ctrl, float)
    tpar = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ctrl, axis=0), axis=1))])
    cs = CubicSpline(tpar, ctrl, bc_type="natural")
    tt = np.linspace(0, tpar[-1], int(tpar[-1] * 4) + 2)
    pts = cs(tt)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    rise = sc["rise"]; th = np.radians(sc["angle"])
    nbp = int(arc[-1] / rise)
    sgrid = np.arange(nbp) * rise
    Cc = np.stack([np.interp(sgrid, arc, pts[:, k]) for k in range(3)], -1)
    d1 = cs(np.interp(sgrid, arc, tt), 1)
    Tt = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
    N, B = rmf_frames(Cc, Tt)
    xyz_all, el_all, bb_all, bp_all = [], [], [], []
    for i in range(nbp):
        x, el, bb = tmpl[rng.integers(len(tmpl))]
        a = th * i + phase
        c, s = np.cos(a), np.sin(a)
        lx = c * x[:, 0] - s * x[:, 1]; ly = s * x[:, 0] + c * x[:, 1]; lz = x[:, 2]
        w = Cc[i] + lx[:, None] * N[i] + ly[:, None] * B[i] + lz[:, None] * Tt[i]
        xyz_all.append(w); el_all += el; bb_all.append(bb); bp_all.append(np.full(len(el), i))
    return np.concatenate(xyz_all), el_all, np.concatenate(bb_all), Cc, Tt


def atom_arrays(el):
    rad = np.array([VDW.get(e, 1.7) for e in el], np.float32)
    col = np.array([COLORS.get(e, (0.5, 0.5, 0.5)) for e in el], np.float32)
    return rad, col


if __name__ == "__main__":
    tm, sc = templates()
    print("screw angle %.2f deg, rise %.3f A, %d templates" % (sc["angle"], sc["rise"], len(tm)))
    for x, el, bb in tm[:3]:
        r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        print(len(el), "z range %.2f..%.2f" % (x[:, 2].min(), x[:, 2].max()), "r max %.2f" % r.max())
