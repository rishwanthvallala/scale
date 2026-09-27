import time, numpy as np, cv2
from dna_model import templates, dna_along_curve, atom_arrays
from sphere_render import Scene, render, to_srgb8, euler
tm, sc = templates()
rng = np.random.default_rng(1)
ctrl = [[-60,0,0],[-20,0,0],[20,0,0],[60,0,0]]
xyz, el, bb, Cc, Tt = dna_along_curve(ctrl, tm, sc, rng)
rad, col = atom_arrays(el)
print(len(el), 'atoms')
S = Scene(xyz, rad, col)
R = euler(0, 0, 25)  # camera rotation: rotate world
t=time.time()
img, info = render(S, R.T, np.zeros(3), 80.0, 960, 540, ss=1.5)
print('t', time.time()-t)
cv2.imwrite('prev/test_helix.jpg', cv2.cvtColor(to_srgb8(img), cv2.COLOR_RGB2BGR))
