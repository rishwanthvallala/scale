"""Assemble scene.json from the per-scale manifests.

    py tools/build_scene.py

Reads assets/manifest/{space,earth,micro}.json (each ordered largest -> smallest),
chains them together (each layer is placed at the previous layer's `focus`),
inserts the virtual Solar System / Earth-Moon stops with orbits, the Sun and the Moon,
writes 2048px variants for phones to assets/levels/2k/, and prints warnings for
gaps that are too large to dissolve cleanly.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MAN = ROOT / "assets" / "manifest"
SMALL = ROOT / "assets" / "levels" / "2k"
GRADED = ROOT / "assets" / "levels" / "graded"

# Simulation renders come in whatever colour map each project picked (white/gold, green,
# grey...). Remapping their brightness onto one palette makes them read as one camera,
# ending in the warm-white/blue tones of the real JWST and Milky Way images that follow.
# CDC's false-colour micrographs, shown in their native greyscale so the whole
# skin -> hair -> bacteria -> virus stretch reads as one electron microscope
SEM_GREY = {"staph-cluster", "bacterium"}

COSMIC_PALETTE_IDS = {"cosmic-web", "large-scale-structure", "supercluster", "galaxy-cluster"}
# per-image brightness target for the median lit pixel (gas renders have big saturated plateaus)
COSMIC_TARGET = {"large-scale-structure": 0.08, "galaxy-cluster": 0.1}
# images whose renderer saturated into flat plateaus: map the plateau below white
COSMIC_HI_SCALE = {"large-scale-structure": 1.8}
COSMIC_STOPS = [
    (0.00, (0.00, 0.00, 0.00)),
    (0.10, (0.04, 0.05, 0.13)),
    (0.30, (0.20, 0.25, 0.55)),
    (0.55, (0.55, 0.60, 0.88)),
    (0.78, (0.92, 0.86, 0.86)),
    (0.92, (1.00, 0.90, 0.72)),
    (1.00, (1.00, 0.98, 0.94)),
]
AU = 1.495978707e11

# Planet data: orbit radius (AU), radius (m), color, angle on orbit (deg)
PLANETS = [
    ("Mercury", 0.387, 2.44e6, (0.75, 0.72, 0.68), 200),
    ("Venus", 0.723, 6.05e6, (1.0, 0.93, 0.78), 120),
    ("Earth", 1.0, 6.371e6, (0.55, 0.72, 1.0), 38),
    ("Mars", 1.524, 3.39e6, (1.0, 0.62, 0.45), 300),
    ("Jupiter", 5.203, 6.99e7, (1.0, 0.88, 0.72), 150),
    ("Saturn", 9.537, 5.82e7, (1.0, 0.9, 0.7), 250),
    ("Uranus", 19.19, 2.54e7, (0.7, 0.9, 1.0), 70),
    ("Neptune", 30.07, 2.46e7, (0.5, 0.65, 1.0), 330),
]
EARTH_ANGLE = 38
MOON_ANGLE = -35


def load(name):
    p = MAN / f"{name}.json"
    if not p.exists():
        print(f"  (no {p.relative_to(ROOT)})")
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def small_variant(src):
    src_path = ROOT / src
    out = SMALL / src_path.name
    if not out.exists() or out.stat().st_mtime < src_path.stat().st_mtime:
        SMALL.mkdir(parents=True, exist_ok=True)
        im = Image.open(src_path).convert("RGB")
        s = 2048 / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        im.save(out, "JPEG", quality=88, optimize=True, progressive=True)
    return out.relative_to(ROOT).as_posix()


def cosmic_grade(src):
    src_path = ROOT / src
    out = GRADED / src_path.name
    if out.exists() and out.stat().st_mtime >= src_path.stat().st_mtime:
        return out.relative_to(ROOT).as_posix()
    GRADED.mkdir(parents=True, exist_ok=True)
    a = np.asarray(Image.open(src_path).convert("RGB"), dtype=np.float32) / 255.0
    lum = a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    lit = lum[lum > 0.02]
    hi = np.percentile(lit, 99.7) if lit.size else 1.0
    hi *= COSMIC_HI_SCALE.get(src_path.stem, 1.0)
    t = np.clip(lum / max(hi, 1e-3), 0, 1)
    # put the median lit pixel at a common (dim) brightness so every level has similar density and dark voids
    med = np.median(t[t > 0.02 / max(hi, 1e-3)]) if lit.size else 0.3
    # filmic shoulder f(): bright cores roll off instead of clipping into flat white blobs;
    # aim the gamma at f⁻¹(target) so the shoulder doesn't lift the dark voids again
    k = 2.4
    target = COSMIC_TARGET.get(src_path.stem, 0.13)
    pre = -np.log(1 - target * (1 - np.exp(-k))) / k
    gamma = np.log(pre) / np.log(np.clip(med, 0.02, 0.95))
    t = t ** gamma
    t = (1 - np.exp(-k * t)) / (1 - np.exp(-k))
    xs = np.array([k for k, _ in COSMIC_STOPS], dtype=np.float32)
    out_rgb = np.stack([np.interp(t, xs, [c[ch] for _, c in COSMIC_STOPS]) for ch in range(3)], axis=-1)
    out_rgb = np.clip(out_rgb, 0, 1)
    Image.fromarray((out_rgb * 255 + 0.5).astype(np.uint8)).save(out, "JPEG", quality=92, optimize=True)
    return out.relative_to(ROOT).as_posix()


def image_layer(e, at):
    if e["id"] in COSMIC_PALETTE_IDS:
        e = {**e, "file": cosmic_grade(e["file"])}
    path = ROOT / e["file"]
    if not path.exists():
        print(f"  !! missing image {e['file']} — skipped")
        return None
    with Image.open(path) as im:
        px = [im.width, im.height]
    layer = {
        "id": e["id"],
        "label": e.get("label", e["id"]),
        "caption": e.get("caption", ""),
        "src": e["file"],
        "src2k": small_variant(e["file"]),
        "px": px,
        "width": float(e["width_m"]),
        "at": at,
        "blend": e.get("blend", "normal"),
        "credit": e.get("credit", ""),
        "license": e.get("license", ""),
        "sourceUrl": e.get("source_url", ""),
    }
    for k in ("rot", "feather", "fadeIn", "colorMatch", "maxUpscale", "grade", "scroll", "dot", "dotLabel", "transition"):
        if k in e:
            layer[k] = e[k]
    return layer


def main():
    space = load("space")
    earth = load("earth")
    micro = load("micro")

    layers, extras = [], []
    focus = None

    def push(layer, next_focus):
        nonlocal focus
        if layer:
            layers.append(layer)
            focus = next_focus

    # ---- cosmic scales
    moon = next((e for e in space if e.get("role") == "extra" or e["id"] == "moon"), None)
    for e in space:
        if e is moon:
            continue
        push(image_layer(e, focus or [0.5, 0.5]), e.get("focus", [0.5, 0.5]))

    # ---- Solar System (virtual stop: the Sun, orbits and planets are drawn procedurally)
    ss_w = 1.25e13
    layers.append({
        "id": "solar-system",
        "label": "Solar System",
        "caption": "The Sun holds 99.8% of all the mass in the Solar System.",
        "width": ss_w, "px": [1, 1], "at": focus or [0.5, 0.5], "blend": "none",
    })
    extras.append({
        "kind": "glow", "frame": "solar-system", "pos": [0, 0], "radius": 6.957e8,
        "halo": 5, "color": [1.0, 0.93, 0.82], "intensity": 2.4, "spikes": 0.35,
        "range": [1e9, 4e19], "label": "Sun", "labelRange": [2e11, 3e16],
    })
    for name, r_au, radius, col, ang in PLANETS:
        R = r_au * AU
        a = math.radians(ang)
        extras.append({"kind": "ring", "frame": "solar-system", "pos": [0, 0], "radius": R,
                       "color": [0.55, 0.7, 1.0], "alpha": 0.28, "widthPx": 1.1,
                       "range": [R / 14, R * 400]})
        if name == "Earth":
            continue
        extras.append({"kind": "glow", "frame": "solar-system", "pos": [R * math.cos(a), R * math.sin(a)],
                       "radius": radius, "halo": 1.8, "color": list(col), "intensity": 1.2,
                       "range": [R / 14, R * 60], "label": name, "labelRange": [R / 10, R * 25]})

    # ---- Earth & Moon (virtual stop)
    ea = math.radians(EARTH_ANGLE)
    em_w = 1.05e9
    layers.append({
        "id": "earth-moon",
        "label": "Earth & Moon",
        "caption": "Every other planet in the Solar System could fit between Earth and the Moon.",
        "width": em_w, "px": [1, 1],
        "at": [0.5 + AU * math.cos(ea) / ss_w, 0.5 + AU * math.sin(ea) / ss_w],
        "blend": "none",
    })
    ma = math.radians(MOON_ANGLE)
    mpos = [3.844e8 * math.cos(ma), 3.844e8 * math.sin(ma)]
    if moon and (ROOT / moon["file"]).exists():
        extras.append({"kind": "image", "frame": "earth-moon", "pos": mpos, "width": float(moon["width_m"]),
                       "src": moon["file"], "src2k": small_variant(moon["file"]), "blend": "add",
                       "feather": 0.08, "range": [2e6, 4e9], "label": "Moon", "labelRange": [5e8, 5e10],
                       "credit": moon.get("credit", ""), "license": moon.get("license", ""),
                       "sourceUrl": moon.get("source_url", ""), "id": "moon"})
    extras.append({"kind": "glow", "frame": "earth-moon", "pos": mpos, "radius": 1.7374e6, "halo": 1.4,
                   "color": [0.8, 0.8, 0.78], "intensity": 1.0, "range": [2e9, 4e11],
                   **({} if moon else {"label": "Moon", "labelRange": [5e8, 5e10]})})
    focus = [0.5, 0.5]

    # ---- Earth, from the globe down to a person on the grass
    for e in earth:
        layer = image_layer(e, focus)
        if layer and e["id"].startswith("person"):
            layer.setdefault("transition", "dissolve")
        if layer and e["id"] == "earth-globe":
            layer.setdefault("dot", {"color": [0.55, 0.72, 1.0], "halo": 2.2, "intensity": 1.5})
            layer.setdefault("dotLabel", {"text": "Earth", "range": [3e9, 4e13]})
        push(layer, e.get("focus", [0.5, 0.5]))

    # ---- the body, cells, molecules, atoms
    # These come from unrelated photos and micrographs, so they cross-fade instead of growing in as patches.
    for e in micro:
        layer = image_layer(e, focus)
        if layer:
            layer.setdefault("transition", "dissolve")
            if e["id"] in SEM_GREY:
                layer.setdefault("grade", {"saturation": 0.0})
        push(layer, e.get("focus", [0.5, 0.5]))

    # ---- sanity checks
    print("\nchain:")
    for i, l in enumerate(layers):
        ratio = layers[i - 1]["width"] / l["width"] if i else float("nan")
        flag = ""
        if i and l.get("blend") == "normal" and layers[i - 1].get("blend") == "normal" and ratio > 15:
            flag = "  <-- big gap for two opaque images: add an intermediate image"
        if i and ratio <= 1.05:
            flag = "  <-- child is not smaller than its parent!"
        print(f"  {i:2d} {l['id']:<18} {l['width']:.3e} m  x{ratio:8.1f}  {l.get('blend')}{flag}")

    scene = {
        "title": "Scale of Everything",
        "layers": layers,
        "extras": extras,
        "starfield": {"frame": "solar-system", "octaves": [16.5, 20.75], "step": 0.25, "perOctave": 3500,
                      "void": 3.8e16, "range": [2e15, 5e20], "seed": 11,
                      "warpRange": [3e13, 2e20], "warpRate": 0.9, "warpAlpha": 1.0, "warpCount": 1600},
    }
    (ROOT / "scene.json").write_text(json.dumps(scene, indent=1), encoding="utf-8")
    print(f"\nwrote scene.json with {len(layers)} layers, {len(extras)} extras")


if __name__ == "__main__":
    sys.exit(main())
