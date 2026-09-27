"""Preview a level with a labelled 0.1 grid and the footprint of the next level at its focus.
py focus_preview.py <id> [out.jpg]  (reads assets/manifest/micro.json)
Also writes a side-by-side of [footprint crop | next image] to check visual continuity.
"""
import json, os, sys
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
man = json.load(open(os.path.join(ROOT, "assets", "manifest", "micro.json"), encoding="utf-8"))
ids = [m["id"] for m in man]
i = ids.index(sys.argv[1])
L = man[i]
im = Image.open(os.path.join(ROOT, L["file"])).convert("RGB")
W, H = im.size
d = ImageDraw.Draw(im)
for k in range(1, 10):
    x = W * k / 10; y = H * k / 10
    d.line([(x, 0), (x, H)], fill=(255, 255, 0), width=max(1, W // 800))
    d.line([(0, y), (W, y)], fill=(255, 255, 0), width=max(1, W // 800))
    d.text((x + 3, 3), f"{k/10:.1f}", fill=(255, 255, 0))
    d.text((3, y + 3), f"{k/10:.1f}", fill=(255, 255, 0))
side = None
if i + 1 < len(man):
    N = man[i + 1]
    fw = N["width_m"] / L["width_m"] * W
    fh = fw * N["px"][1] / N["px"][0]
    cx, cy = L["focus"][0] * W, L["focus"][1] * H
    box = (cx - fw / 2, cy - fh / 2, cx + fw / 2, cy + fh / 2)
    orig = Image.open(os.path.join(ROOT, L["file"])).convert("RGB")
    crop = orig.crop(tuple(int(round(v)) for v in box)).resize((600, int(600 * fh / fw)))
    nxt = Image.open(os.path.join(ROOT, N["file"])).convert("RGB").resize((600, int(600 * fh / fw)))
    side = Image.new("RGB", (1210, crop.height), (0, 0, 0))
    side.paste(crop, (0, 0)); side.paste(nxt, (610, 0))
    d.rectangle(box, outline=(255, 0, 0), width=max(2, W // 400))
im.thumbnail((1200, 1200))
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "prev", f"focus_{L['id']}.jpg")
im.save(out, quality=85)
if side is not None:
    side.save(out.replace(".jpg", "_pair.jpg"), quality=85)
print(out)
