"""
person_photos.py -- the bridge from the Earth levels to the human body.

Reads assets/src/earth/person/person.json (a list, widest photo first), e.g.

  [{"id": "person-wide", "src": "candidates/B01_x.jpg", "rotate_deg": 0,
    "crop_px": [500, 0, 3500, 2250], "width_m": 9.0,
    "label": "...", "caption": "...", "credit": "...", "license": "...",
    "source_url": "...", "notes": "..."},
   {"id": "person", "src": "candidates/A01_y.jpg", "rotate_deg": 12.5,
    "crop_px": [...], "width_m": 3.5, "focus_px": [x, y], ...}]

  * rotate_deg (counter-clockwise, about the image centre) is applied first; center_px and
    size_px describe the square crop in the ORIGINAL image's pixel coordinates (the crop centre is
    rotated with the image), focus_px (the back of a hand, last photo only) likewise.
  * The body centre must be at the crop centre: every level is centred in its parent.

prepare() crops/rotates/resizes (<= 4096 px) and gently grades each photo so its grass matches
the lawn at the centre of the natural earth-160m render (Lab mean fully-ish, contrast gently);
the person keeps most of its own colour.  earth_levels.py then chains the photos below
earth-160m (injection + residual matching exactly like the Earth levels).
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent
PERSON_DIR = ROOT / 'assets' / 'src' / 'earth' / 'person'
CONFIG = PERSON_DIR / 'person.json'
MAX_PX = 4096


def _lab(rgb):
    return cv2.cvtColor(np.clip(rgb / 255.0, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)


def _rgb(lab):
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB) * 255.0, 0, 255)


def grass_mask(rgb):
    lab = _lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    m = ((a < -5) & (b > 4) & (L > 12) & (L < 92)).astype(np.float32)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return m


def lawn_stats(rgb):
    lab = _lab(rgb)
    m = grass_mask(rgb) > 0.5
    if m.sum() < 100:
        m = np.ones(m.shape, bool)
    px = lab[m]
    return px.mean(0), px.std(0)


def grade(rgb, target, mean_k=0.85, std_k=0.35, person_share=0.3):
    """Reinhard-style Lab transfer of the photo's grass statistics toward `target` (mean, std);
    applied fully on grass, partly (person_share) elsewhere."""
    lab = _lab(rgb)
    m = grass_mask(rgb)
    px = lab[m > 0.5] if (m > 0.5).sum() > 100 else lab.reshape(-1, 3)
    mu_s, sd_s = px.mean(0), np.maximum(px.std(0), 1e-3)
    mu_t, sd_t = target
    ratio = (np.maximum(sd_t, 1e-3) / sd_s) ** std_k
    new = (lab - mu_s) * ratio + mu_s + (mu_t - mu_s) * mean_k
    w = person_share + (1 - person_share) * cv2.GaussianBlur(m, (0, 0), max(2, rgb.shape[0] / 400))
    out = lab + (new - lab) * w[..., None]
    return _rgb(out), dict(src_mean=[round(float(v), 1) for v in mu_s],
                           target_mean=[round(float(v), 1) for v in mu_t])


def _load_crop(entry):
    """Rotate (optional) and crop.  crop_px = [x0, y0, x1, y1] in ORIGINAL pixel coordinates
    (after rotation the crop keeps its size and follows its rotated centre)."""
    im = ImageOps.exif_transpose(Image.open(PERSON_DIR / entry['src'])).convert('RGB')
    if entry.get('cleanup_px'):          # remove small litter etc.: [[x, y, radius], ...]
        a = np.asarray(im).copy()
        mask = np.zeros(a.shape[:2], np.uint8)
        for x, y, r in entry['cleanup_px']:
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        a = cv2.inpaint(a, mask, 7, cv2.INPAINT_TELEA)
        im = Image.fromarray(a)
    w0, h0 = im.size
    x0, y0, x1, y1 = [float(v) for v in entry['crop_px']]
    cx, cy, cw, ch = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    fx, fy = entry.get('focus_px', [cx, cy])
    ang = float(entry.get('rotate_deg', 0.0))
    if ang:
        im = im.rotate(ang, resample=Image.BICUBIC, expand=True)
        w1, h1 = im.size
        t = math.radians(ang)

        def rot(x, y):   # PIL rotates counter-clockwise; image y axis points down
            dx, dy = x - w0 / 2, y - h0 / 2
            return (w1 / 2 + dx * math.cos(t) + dy * math.sin(t),
                    h1 / 2 - dx * math.sin(t) + dy * math.cos(t))
        cx, cy = rot(cx, cy)
        fx, fy = rot(fx, fy)
    box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
    if box[0] < -0.5 or box[1] < -0.5 or box[2] > im.size[0] + 0.5 or box[3] > im.size[1] + 0.5:
        raise ValueError(f"{entry['id']}: crop {box} outside image {im.size}")
    k = min(1.0, MAX_PX / max(cw, ch))
    out = (int(round(cw * k)), int(round(ch * k)))
    crop = im.transform(out, Image.EXTENT, box, resample=Image.BICUBIC)
    focus = [(fx - box[0]) / cw, (fy - box[1]) / ch]
    return np.asarray(crop).astype(np.float32), focus


def prepare(ref160, ref_width=160.0):
    """ref160: natural earth-160m image (float RGB).  Returns the photo levels, widest first:
    dicts with id, width, img (graded float32 RGB), focus, manifest fields."""
    if not CONFIG.exists():
        raise FileNotFoundError(f'{CONFIG} not found')
    entries = json.loads(CONFIG.read_text(encoding='utf-8'))
    n = ref160.shape[0]
    out = []
    for e in entries:
        img, focus = _load_crop(e)
        # lawn statistics of the aerial image inside this photo's footprint (min 20 m)
        fw = max(float(e['width_m']), 20.0)
        half = int(round(n * fw / ref_width / 2))
        region = ref160[n // 2 - half:n // 2 + half, n // 2 - half:n // 2 + half]
        grass = float(grass_mask(region).mean())
        if grass < 0.5:
            print(f"  WARNING [{e['id']}]: only {grass:.0%} of the aerial image around the target looks "
                  f"like grass -- the lawn photos may not fit this place (pick a lawn with --lat/--lon)")
        target = lawn_stats(region)
        graded, info = grade(img, target)
        out.append(dict(id=e['id'], width=float(e['width_m']), img=graded, focus=focus,
                        entry=e, grade=info))
        print(f"  [{e['id']}] {img.shape[1]}x{img.shape[0]} px, {e['width_m']} m wide, "
              f"graded grass {info['src_mean']} -> {info['target_mean']}")
    return out
