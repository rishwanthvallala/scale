"""Measure SEM scale-bar length: longest horizontal run of bar-colored pixels in a region.
py measure_bar.py file y0 y1 [x0 x1] [mode: white|green|black]"""
import sys
import numpy as np
from PIL import Image
f = sys.argv[1]; y0, y1 = int(sys.argv[2]), int(sys.argv[3])
im = np.asarray(Image.open(f).convert("RGB")).astype(int)
x0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
x1 = int(sys.argv[5]) if len(sys.argv) > 5 else im.shape[1]
mode = sys.argv[6] if len(sys.argv) > 6 else "white"
reg = im[y0:y1, x0:x1]
if mode == "white":
    m = (reg.min(axis=2) > 200)
elif mode == "green":
    m = (reg[..., 1] > 180) & (reg[..., 0] < 120) & (reg[..., 2] < 120)
else:
    m = (reg.max(axis=2) < 50)
best = (0, 0, 0)
for yy in range(m.shape[0]):
    row = m[yy]; run = 0
    for xx in range(len(row)):
        run = run + 1 if row[xx] else 0
        if run > best[0]:
            best = (run, yy + y0, xx - run + 1 + x0)
print("longest run px:", best[0], "at y", best[1], "x from", best[2], "to", best[2] + best[0] - 1)
