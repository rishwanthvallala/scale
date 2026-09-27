"""Generate the three nested molecular levels from ONE all-atom DNA scene:
  dna-strands (60 nm)  ->  dna-helix (8 nm)  ->  atoms (1.6 nm)
Same geometry, same lights, orthographic camera => the zooms nest exactly.

py gen_dna.py [--preview]
"""
import json, os, sys, time
import numpy as np
import cv2
from dna_model import templates, dna_along_curve, atom_arrays
from sphere_render import Scene, render, to_srgb8, project, background

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "gen"); os.makedirs(GEN, exist_ok=True)
PREVIEW = "--preview" in sys.argv
W, H = (960, 540) if PREVIEW else (3840, 2160)
SS = 1.5

R_CAM = np.eye(3)  # camera axes == world axes (x right, y up, z toward viewer)

VIEWS = {  # id: (width in Angstrom)
    "dna-strands": 600.0,
    "dna-helix": 80.0,
    "atoms": 16.0,
}


def bgfun(Wi, bh, y0=0, Hfull=None):
    return background(Wi, bh, c_center=(0.040, 0.045, 0.060), c_edge=(0.006, 0.007, 0.011), y0=y0, Hfull=Hfull)


def build_scene(seed=7):
    tm, sc = templates()
    rng = np.random.default_rng(seed)
    ang = np.radians(-24)
    d = np.array([np.cos(ang), np.sin(ang), 0.12]); d /= np.linalg.norm(d)
    # hero strand: straight through the origin, gently curving far from the centre
    hero = [(-430 * d) + [0, -60, -70], -170 * d, -60 * d, 60 * d, 170 * d, 430 * d + [0, 80, 40]]
    curves = [np.array(hero)]
    # other strands: smooth random curves, kept away from the central (helix) window
    def ok(c, pts_other):
        tt = np.linspace(0, 1, 400)
        seg = np.concatenate([np.linspace(c[i], c[i + 1], 60) for i in range(len(c) - 1)])
        ex = (seg[:, 0] / 150.0) ** 2 + (seg[:, 1] / 95.0) ** 2
        if (ex < 1).any():
            return False
        for o in pts_other:
            dd = np.linalg.norm(seg[:, None, :] - o[None, ::3, :], axis=-1)
            if dd.min() < 34:
                return False
        return True
    sampled = [np.concatenate([np.linspace(hero[i], hero[i + 1], 60) for i in range(len(hero) - 1)])]
    tries = 0
    specs = [(-150, -60), (-230, -120), (40, 110), (-110, -40), (-260, -170), (30, 90), (-190, -90)]
    for (z0, z1) in specs:
        while tries < 4000:
            tries += 1
            side = rng.integers(4)
            a = rng.uniform(0, 2 * np.pi)
            start = np.array([rng.uniform(-420, 420), rng.uniform(-260, 260), rng.uniform(z0, z1)])
            # direction mostly across the frame
            th = rng.uniform(0, np.pi)
            dirv = np.array([np.cos(th), np.sin(th) * 0.8, rng.uniform(-0.15, 0.15)])
            dirv /= np.linalg.norm(dirv)
            n = 6
            L = rng.uniform(700, 950)
            pts = [start - dirv * L / 2 + dirv * L * k / (n - 1) for k in range(n)]
            perp = np.cross(dirv, [0, 0, 1.0]); perp /= np.linalg.norm(perp)
            pts = [p + perp * rng.normal(0, 55) + [0, 0, rng.normal(0, 15)] for p in pts]
            pts = np.array(pts)
            pts[:, 2] = np.clip(pts[:, 2], z0 - 30, z1 + 30)
            if ok(pts, sampled):
                curves.append(pts)
                sampled.append(np.concatenate([np.linspace(pts[i], pts[i + 1], 60) for i in range(len(pts) - 1)]))
                break
    print(f"  {len(curves)} strands ({tries} tries)")
    X, E = [], []
    for k, c in enumerate(curves):
        xyz, el, bb, Cc, Tt = dna_along_curve(c, tm, sc, rng, phase=rng.uniform(0, 6.28))
        X.append(xyz); E += el
    xyz = np.concatenate(X)
    rad, col = atom_arrays(E)
    return xyz, rad, col, E, curves


def save(img8, lid):
    p = os.path.join(GEN, lid + (".preview.jpg" if PREVIEW else ".jpg"))
    cv2.imwrite(p, cv2.cvtColor(img8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return p


def main():
    t0 = time.time()
    xyz, rad, col, E, curves = build_scene()
    print(f"  scene: {len(xyz)} atoms, built in {time.time() - t0:.1f}s")
    scene = Scene(xyz, rad, col)
    common = dict(ss=SS, fog_ref=25.0, fog_range=320.0, fog_amount=0.62, bgfun=bgfun, shadow_soft=0.5,
                  ao_strength=1.35)

    # --- choose the atoms-view centre: a front-facing sugar carbon of the hero strand near the origin
    near = np.flatnonzero(np.linalg.norm(xyz[:, :2] - [4.0, -6.0], axis=1) < 9.0)
    # pick carbon with the largest z (most exposed toward the camera) among those
    cands = [i for i in near if E[i] == "C"]
    zs = xyz[cands, 2]
    # exposure: no other atom in front covering it
    def exposed(i):
        d2 = ((xyz[:, :2] - xyz[i, :2]) ** 2).sum(1)
        front = (xyz[:, 2] > xyz[i, 2] + 0.3) & (d2 < (rad + rad[i] * 0.55) ** 2)
        return not front.any()
    cands = [c for c in np.array(cands)[np.argsort(-zs)] if exposed(c)]
    atom_i = int(cands[0])
    atom_c = xyz[atom_i].copy()
    print("  focus atom", atom_i, E[atom_i], atom_c)

    centers = {"dna-strands": np.zeros(3), "dna-helix": np.zeros(3),
               "atoms": np.array([atom_c[0], atom_c[1], 0.0])}
    # focus of each view = where the next view's centre projects
    order = ["dna-strands", "dna-helix", "atoms"]
    focus = {}
    for a, b in zip(order, order[1:]):
        focus[a] = project(R_CAM, centers[a], VIEWS[a], W, H, centers[b])
    focus["atoms"] = project(R_CAM, centers["atoms"], VIEWS["atoms"], W, H, atom_c)
    meta = {}
    for lid in order:
        t1 = time.time()
        img, info = render(scene, R_CAM, centers[lid], VIEWS[lid], W, H, **common)
        p = save(to_srgb8(img, exposure=1.05), lid)
        print(f"  {lid}: {time.time() - t1:.1f}s -> {p}  focus={focus[lid]}")
        meta[lid] = dict(focus=[round(float(focus[lid][0]), 4), round(float(focus[lid][1]), 4)])
    json.dump(dict(meta=meta, atom=dict(index=atom_i, element=E[atom_i], xyz=atom_c.tolist())),
              open(os.path.join(GEN, "dna_views.json"), "w"), indent=2)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
