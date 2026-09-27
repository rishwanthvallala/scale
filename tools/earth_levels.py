#!/usr/bin/env python3
"""
earth_levels.py -- build the EARTH part of the Powers-of-Ten zoom.

Every level is rendered in the SAME orthographic projection (sphere R = 6371008.8 m), centred on
the target, north-up and square.  For an N px level of plane width W (metres), output pixel
(col j, row i) has plane coordinates

    x = (j + 0.5 - N/2) * W/N        (east)
    y = (N/2 - i - 0.5) * W/N        (north)

so the target is exactly the image centre of every level, and every level sits exactly in the
centre of its parent (focus [0.5, 0.5]).  Each output pixel is inverse-projected to lat/lon and
the source imagery is sampled there (super-sampled + area-prefiltered, so nothing aliases).

    py tools/earth_levels.py --lat 40.7716 --lon -73.9749 [--size 4096] [--provider usgs|esri]

Sources (downloaded politely, <= 4 concurrent requests, retries with backoff, cached on disk under
assets/src/earth/tiles):
  globe .............. NASA Blue Marble Next Generation (July 2004, 21600x10800) + NASA Blue
                       Marble cloud layer + a simple atmosphere model (limb haze, glow)
  5000 km / 1600 km .. NASA GIBS Blue Marble NG tiles (500 m) (+ fading clouds at 5000 km)
  500 km / 160 km .... NASA GIBS Global WELD Landsat annual true-colour mosaic (30 m)
  50 km / 16 km ...... USGS The National Map USGSImageryOnly (NAIP), holes filled from WELD
  5 km ... 160 m ..... NYC OTI 2018 orthoimagery (6 in / 15 cm, maps.nyc.gov tiles), USGS fallback
  --provider esri .... Esri World Imagery for 500 km and below (NOT public domain, see note)

After rendering, adjacent levels are harmonised from the finest level upwards: each coarser level
gets a gentle per-channel Lab tone mapping fitted so that, inside the footprint of its child, it
matches the child; then the (downsampled) child is feathered into the parent's centre.  Hence
when the engine cross-dissolves parent -> child there is no colour pop.

Outputs: assets/levels/earth-*.jpg (RGB JPEG q=90), assets/levels/person*.jpg,
assets/manifest/earth.json, previews in assets/src/earth/previews/.
"""
import argparse
import io
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(4)

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / 'assets' / 'src' / 'earth'
TILE_DIR = SRC_DIR / 'tiles'
WORK_DIR = SRC_DIR / 'work'
PREVIEW_DIR = SRC_DIR / 'previews'
PERSON_DIR = SRC_DIR / 'person'
OUT_DIR = ROOT / 'assets' / 'levels'
MANIFEST = ROOT / 'assets' / 'manifest' / 'earth.json'

R_EARTH = 6371008.8
WEBMERC_Z0 = 156543.03392804097          # m/px at zoom 0, equator, 256 px tiles
UA = 'ScaleZoomSite/0.1 (personal non-commercial project)'
MAX_WORKERS = 4
RENDER_VERSION = 'v3'                    # bump to invalidate cached natural renders

DEFAULT_LAT, DEFAULT_LON = 40.77184, -73.97518   # centre of Sheep Meadow lawn (>100 m to trees)


# ----------------------------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------------------------
_tls = threading.local()


def _session():
    s = getattr(_tls, 's', None)
    if s is None:
        s = requests.Session()
        s.headers['User-Agent'] = UA
        _tls.s = s
    return s


def http_get(url, tries=7, timeout=60):
    """GET with retries + exponential backoff.  Returns bytes, or None if the resource does not
    exist (404/204/400)."""
    delay = 1.0
    err = ''
    for attempt in range(tries):
        try:
            r = _session().get(url, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
            if r.status_code in (204, 400, 404):
                return None
            err = f'HTTP {r.status_code}'
        except requests.RequestException as e:
            err = type(e).__name__
        if attempt < tries - 1:
            time.sleep(delay * (1 + 0.3 * np.random.rand()))
            delay = min(delay * 2, 30)
    raise IOError(f'giving up on {url}: {err}')


def download_file(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.part')
    print(f'  downloading {url}')
    for attempt in range(7):
        try:
            with _session().get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, 'wb') as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            print(f'    retry {attempt + 1}: {e}')
            time.sleep(2 ** attempt)
    raise IOError(f'could not download {url}')


# ----------------------------------------------------------------------------------------------
# Projection helpers
# ----------------------------------------------------------------------------------------------
def inv_ortho(x, y, lat0, lon0):
    """Inverse orthographic projection (sphere).  x,y metres -> lat, lon (deg; lon unwrapped
    around lon0), inside-disk mask, mu = cos(angular distance) (1 at centre, 0 at limb)."""
    phi0, lam0 = math.radians(lat0), math.radians(lon0)
    rho = np.hypot(x, y)
    inside = rho <= R_EARTH
    s = np.clip(rho / R_EARTH, 0.0, 1.0)
    c = np.arcsin(s)
    sc, cc = s, np.cos(c)
    safe = np.where(rho > 0, rho, 1.0)
    t = np.where(rho > 0, y * sc / safe, 0.0)
    lat = np.arcsin(np.clip(cc * math.sin(phi0) + t * math.cos(phi0), -1.0, 1.0))
    dlon = np.arctan2(x * sc, rho * cc * math.cos(phi0) - y * sc * math.sin(phi0))
    return np.degrees(lat), lon0 + np.degrees(dlon), inside, cc


def merc_px(lat, lon, z):
    """Global Web-Mercator pixel coordinates (256 px tiles) at zoom z; pixel i spans [i, i+1)."""
    n = 256.0 * (2 ** z)
    gx = (lon + 180.0) / 360.0 * n
    s = np.sin(np.radians(np.clip(lat, -85.05112878, 85.05112878)))
    gy = (0.5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return gx, gy


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class View:
    """One square orthographic level."""

    def __init__(self, width, n, lat0, lon0, ss=2):
        self.width, self.n, self.lat0, self.lon0, self.ss = float(width), int(n), lat0, lon0, ss
        self.mpp = self.width / self.n

    def plane_coords(self, rows):
        ss, n, px = self.ss, self.n, self.mpp
        sub = (np.arange(ss) + 0.5) / ss
        xs = ((np.arange(n)[:, None] + sub[None, :]).ravel() - n / 2) * px
        ys = (n / 2 - (rows[:, None] + sub[None, :]).ravel()) * px
        return xs, ys

    def grid(self, g):
        """(g+1)x(g+1) grid of lat/lon over the whole view (for tile planning)."""
        t = np.linspace(-self.width / 2, self.width / 2, g + 1)
        X, Y = np.meshgrid(t, -t)
        return inv_ortho(X, Y, self.lat0, self.lon0)

    def pixel_plane(self):
        """plane x (row vector) and y (column vector) of output pixel centres."""
        c = (np.arange(self.n) + 0.5 - self.n / 2) * self.mpp
        return c[None, :], -c[:, None]


def sample_view(view, coord_fn, img, valid=None, interp=cv2.INTER_LINEAR,
                border=cv2.BORDER_CONSTANT):
    """Sample `img` for every output pixel of `view` (ss x ss super-sampling, coverage-weighted).
    coord_fn(lat, lon) -> (u, v, ok) in img pixel coordinates (pixel centres at integers).
    Returns colour float32 (N,N,C) (un-premultiplied) and coverage float32 (N,N) in 0..1."""
    n, ss = view.n, view.ss
    ch = 1 if img.ndim == 2 else img.shape[2]
    out = np.zeros((n, n, ch), np.float32)
    cov = np.zeros((n, n), np.float32)
    rows_per = max(4, int(1_500_000 // (n * ss * ss)))
    for r0 in range(0, n, rows_per):
        rows = np.arange(r0, min(n, r0 + rows_per))
        xs, ys = view.plane_coords(rows)
        X, Y = np.meshgrid(xs, ys)
        lat, lon, inside, _ = inv_ortho(X, Y, view.lat0, view.lon0)
        u, v, ok = coord_fn(lat, lon)
        ok = ok & inside
        u = np.where(ok, u, -1e5).astype(np.float32)
        v = np.where(ok, v, -1e5).astype(np.float32)
        c = cv2.remap(img, u, v, interp, borderMode=border, borderValue=0).astype(np.float32)
        if valid is not None:
            m = cv2.remap(valid, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0).astype(np.float32) / 255.0
            m *= ok
        else:
            m = ok.astype(np.float32)
        if c.ndim == 2:
            c = c[..., None]
        nr = len(rows)
        cw = (c * m[..., None]).reshape(nr, ss, n, ss, ch).sum((1, 3))
        mw = m.reshape(nr, ss, n, ss).sum((1, 3))
        out[r0:r0 + nr] = cw / np.maximum(mw, 1e-6)[..., None]
        cov[r0:r0 + nr] = mw / (ss * ss)
    return out, cov


# ----------------------------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------------------------
class TileSource:
    """XYZ Web-Mercator tile source with an on-disk cache."""

    def __init__(self, key, url, maxz, minz=0, ext='jpg', nodata_black=False, native_mpp=None,
                 credit='', license='', source_url='', note=''):
        self.key, self.url, self.maxz, self.minz, self.ext = key, url, maxz, minz, ext
        self.nodata_black = nodata_black
        self.native_mpp = native_mpp        # finest real ground resolution (m/px) of the data
        self.credit, self.license, self.source_url, self.note = credit, license, source_url, note

    def tile_path(self, z, x, y):
        return TILE_DIR / self.key / str(z) / str(x) / f'{y}.{self.ext}'

    def pick_zoom(self, mpp, lat, quality=1.25):
        c = WEBMERC_Z0 * math.cos(math.radians(lat))
        z = math.ceil(math.log2(c * quality / mpp))
        return int(min(max(z, self.minz), self.maxz))

    def fetch(self, z, tiles):
        n = 2 ** z
        todo = []
        for (x, y) in tiles:
            p = self.tile_path(z, x % n, y)
            if not p.exists() and not p.with_suffix('.404').exists():
                todo.append((x % n, y))
        if not todo:
            return
        print(f'    {self.key} z{z}: fetching {len(todo)} tiles ({len(tiles) - len(todo)} cached)')
        done = [0]
        lock = threading.Lock()

        def job(t):
            x, y = t
            p = self.tile_path(z, x, y)
            try:
                b = http_get(self.url.format(z=z, x=x, y=y))
            except IOError as e:
                with lock:
                    failed.append(str(e))
                return              # not cached -> retried on the next run; fallback fills it
            p.parent.mkdir(parents=True, exist_ok=True)
            ok = False
            if b is not None:
                try:
                    Image.open(io.BytesIO(b)).verify()
                    ok = True
                except Exception:  # noqa: BLE001  (error page instead of an image)
                    ok = False
            if ok:
                tmp = p.with_name(p.name + '.part')
                tmp.write_bytes(b)
                tmp.replace(p)
            else:
                p.with_suffix('.404').write_bytes(b'')
            with lock:
                done[0] += 1
                if done[0] % 200 == 0:
                    print(f'      {done[0]}/{len(todo)}')

        failed = []
        with ThreadPoolExecutor(MAX_WORKERS) as ex:
            list(ex.map(job, todo))
        if failed:
            print(f'    WARNING: {len(failed)} {self.key} tiles failed after retries '
                  f'(e.g. {failed[0]}); re-run the script to retry them')

    def load(self, z, x, y):
        p = self.tile_path(z, x % (2 ** z), y)
        if not p.exists():
            return None
        try:
            im = Image.open(p)
            if im.mode == 'P':
                im = im.convert('RGBA')
            im = im.convert('RGB')
        except Exception:  # noqa: BLE001
            return None
        a = np.asarray(im)
        if a.shape[:2] != (256, 256):
            a = cv2.resize(a, (256, 256), interpolation=cv2.INTER_AREA)
        return a

    def mosaic(self, z, tiles):
        xs = [t[0] for t in tiles]
        ys = [t[1] for t in tiles]
        tx0, ty0 = min(xs), min(ys)
        w, h = (max(xs) - tx0 + 1) * 256, (max(ys) - ty0 + 1) * 256
        img = np.zeros((h, w, 3), np.uint8)
        val = np.zeros((h, w), np.uint8)
        for (x, y) in tiles:
            a = self.load(z, x, y)
            if a is None:
                continue
            oy, ox = (y - ty0) * 256, (x - tx0) * 256
            img[oy:oy + 256, ox:ox + 256] = a
            val[oy:oy + 256, ox:ox + 256] = 255
        if self.nodata_black:
            dark = (img.max(axis=2) <= 10).astype(np.uint8)
            dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
            dark = cv2.dilate(dark, np.ones((7, 7), np.uint8))
            val[dark > 0] = 0
        return img, val, tx0 * 256, ty0 * 256

    def render(self, view, need=None, quality=1.25):
        z = self.pick_zoom(view.mpp, view.lat0, quality)
        while True:
            tiles = tiles_for_view(view, z, need)
            if not tiles:
                return None
            xs = [t[0] for t in tiles]
            ys = [t[1] for t in tiles]
            if max(max(xs) - min(xs), max(ys) - min(ys)) * 256 <= 28000 or z <= self.minz:
                break
            z -= 1
        self.fetch(z, tiles)
        img, val, ox, oy = self.mosaic(z, tiles)
        src_mpp = WEBMERC_Z0 * math.cos(math.radians(view.lat0)) / 2 ** z
        r = view.mpp / src_mpp                      # >1: source finer than output
        d = r / view.ss
        sx = sy = 1.0
        if d > 1.05:                                # area-prefilter so samples don't skip pixels
            h, w = img.shape[:2]
            nw, nh = max(1, int(round(w / d))), max(1, int(round(h / d)))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            val = cv2.resize(val, (nw, nh), interpolation=cv2.INTER_AREA)
            sx, sy = nw / w, nh / h
        elif self.native_mpp and self.native_mpp > src_mpp * 1.1:
            # tiles are finer than the real data (server up-sampled, blocky) -> smooth the blocks
            sg = 0.5 * self.native_mpp / src_mpp
            img = cv2.GaussianBlur(img, (0, 0), sg)
        interp = cv2.INTER_CUBIC if r < 0.9 else cv2.INTER_LINEAR

        def coord_fn(lat, lon):
            gx, gy = merc_px(lat, lon, z)
            ok = np.abs(lat) < 85.05
            return (gx - ox) * sx - 0.5, (gy - oy) * sy - 0.5, ok

        rgb, cov = sample_view(view, coord_fn, img, val, interp)
        eff = max(src_mpp, self.native_mpp or 0)
        info = dict(source=self.key, zoom=z, src_mpp=round(src_mpp, 3), eff_mpp=round(eff, 3),
                    tiles=len(tiles))
        return rgb, cov, info


def tiles_for_view(view, z, need=None, g=128):
    lat, lon, inside, _ = view.grid(g)
    gx, gy = merc_px(lat, lon, z)
    ok = inside & (np.abs(lat) < 85.05)
    corners_x = np.stack([gx[:-1, :-1], gx[1:, :-1], gx[:-1, 1:], gx[1:, 1:]])
    corners_y = np.stack([gy[:-1, :-1], gy[1:, :-1], gy[:-1, 1:], gy[1:, 1:]])
    cok = np.stack([ok[:-1, :-1], ok[1:, :-1], ok[:-1, 1:], ok[1:, 1:]]).any(0)
    if need is not None:
        n = need.shape[0]
        idx = (np.arange(g + 1) * n / g).astype(int)
        cellneed = np.zeros((g, g), bool)
        integral = np.pad(need.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        i0, i1 = idx[:-1], np.maximum(idx[1:], idx[:-1] + 1)
        s = (integral[i1][:, i1] - integral[i0][:, i1] - integral[i1][:, i0] + integral[i0][:, i0])
        # dilate by one cell so that feathering has data
        cellneed = cv2.dilate((s > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        cok &= cellneed
    x0 = np.floor((corners_x.min(0) - 3) / 256).astype(int)
    x1 = np.floor((corners_x.max(0) + 3) / 256).astype(int)
    y0 = np.floor((corners_y.min(0) - 3) / 256).astype(int)
    y1 = np.floor((corners_y.max(0) + 3) / 256).astype(int)
    nmax = 2 ** z - 1
    tiles = set()
    for i, j in zip(*np.nonzero(cok)):
        for tx in range(x0[i, j], x1[i, j] + 1):
            for ty in range(max(0, y0[i, j]), min(nmax, y1[i, j]) + 1):
                tiles.add((tx, ty))
    return tiles


class EquirectSource:
    """Global plate-carree image (e.g. Blue Marble NG 21600x10800)."""

    def __init__(self, key, url, fname, credit='', license='', source_url='', note='',
                 gray=False):
        self.key, self.url, self.fname, self.gray = key, url, fname, gray
        self.credit, self.license, self.source_url, self.note = credit, license, source_url, note
        self._img = None

    def image(self):
        if self._img is None:
            p = download_file(self.url, SRC_DIR / self.fname)
            if self.gray:
                cache = SRC_DIR / (Path(self.fname).stem + '_gray.png')
                if not cache.exists():
                    a = np.asarray(Image.open(p).convert('RGB')).astype(np.float32).mean(2)
                    cv2.imwrite(str(cache), np.clip(a + 0.5, 0, 255).astype(np.uint8))
                self._img = cv2.imread(str(cache), cv2.IMREAD_GRAYSCALE)
            else:
                a = cv2.imread(str(p), cv2.IMREAD_COLOR)
                cv2.cvtColor(a, cv2.COLOR_BGR2RGB, dst=a)
                self._img = a
        return self._img

    def render(self, view, need=None, quality=None, interp=None):
        img = self.image()
        h, w = img.shape[:2]
        src_mpp = 2 * math.pi * R_EARTH / w
        if interp is None:
            interp = cv2.INTER_CUBIC if view.mpp / src_mpp < 0.9 else cv2.INTER_LINEAR

        def coord_fn(lat, lon):
            lonw = np.mod(lon + 180.0, 360.0)
            u = lonw / 360.0 * w - 0.5
            v = np.clip((90.0 - lat) / 180.0 * h - 0.5, 0, h - 1)
            return u, v, np.ones(lat.shape, bool)

        rgb, cov = sample_view(view, coord_fn, img, None, interp, border=cv2.BORDER_WRAP)
        info = dict(source=self.key, src_mpp=round(src_mpp, 1), eff_mpp=round(src_mpp, 1))
        return rgb, cov, info


GIBS = 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best'
SOURCES = {
    'bmng_eq': EquirectSource(
        'bmng_eq',
        'https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74092/world.200407.3x21600x10800.jpg',
        'world.200407.3x21600x10800.jpg',
        credit='NASA Earth Observatory - Blue Marble: Next Generation (R. Stockli), July 2004',
        license='Public domain (NASA)',
        source_url='https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74092/world.200407.3x21600x10800.jpg'),
    'clouds': EquirectSource(
        'clouds',
        'https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57747/cloud_combined_8192.tif',
        'cloud_combined_8192.tif', gray=True,
        credit='NASA Goddard / Blue Marble: Clouds (R. Stockli, MODIS)',
        license='Public domain (NASA)',
        source_url='https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57747/cloud_combined_8192.tif'),
    'gibs_bmng': TileSource(
        'gibs_bmng',
        GIBS + '/BlueMarble_NextGeneration/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg',
        maxz=8, native_mpp=463.0,
        credit='NASA Blue Marble: Next Generation via NASA GIBS / Worldview',
        license='Public domain (NASA; GIBS imagery has no use restrictions)',
        source_url=GIBS + '/BlueMarble_NextGeneration/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg'),
    'weld': TileSource(
        'weld',
        GIBS + '/Landsat_WELD_CorrectedReflectance_TrueColor_Global_Annual/default/2000-12-01/'
               'GoogleMapsCompatible_Level12/{z}/{y}/{x}.jpeg',
        maxz=12, nodata_black=True, native_mpp=30.0,
        credit='NASA/USGS Global Web-Enabled Landsat Data (GWELD) annual 2000 true colour, via NASA GIBS',
        license='Public domain (NASA/USGS Landsat)',
        source_url=GIBS + '/Landsat_WELD_CorrectedReflectance_TrueColor_Global_Annual/default/2000-12-01/'
                          'GoogleMapsCompatible_Level12/{z}/{y}/{x}.jpeg'),
    'water': TileSource(
        'water',
        GIBS + '/MODIS_Water_Mask/default/GoogleMapsCompatible_Level9/{z}/{y}/{x}.png',
        maxz=9, ext='png', native_mpp=250.0,
        credit='NASA MODIS 250 m land/water mask (MOD44W) via NASA GIBS',
        license='Public domain (NASA)',
        source_url=GIBS + '/MODIS_Water_Mask/default/GoogleMapsCompatible_Level9/{z}/{y}/{x}.png'),
    'usgs': TileSource(
        'usgs',
        'https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}',
        maxz=16, nodata_black=True, native_mpp=0.6,
        credit='USGS The National Map: Orthoimagery (USDA NAIP)',
        license='Public domain (USGS/USDA)',
        source_url='https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer'),
    'nyc2018': TileSource(
        'nyc2018',
        'https://maps.nyc.gov/xyz/1.0.0/photo/2018/{z}/{x}/{y}.png8',
        maxz=20, minz=8, ext='png', nodata_black=True, native_mpp=0.1524,
        credit='NYC Office of Technology & Innovation (OTI), 2018 Orthoimagery (6-inch, Apr-May 2018)',
        license='CC BY 4.0 (NYC Open Data)',
        source_url='https://data.cityofnewyork.us/City-Government/2018-Orthoimagery-Manhattan/hxws-3mbm',
        note='Tiles from maps.nyc.gov/xyz/1.0.0/photo/2018 (8-bit palette PNG).'),
    'esri': TileSource(
        'esri',
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        maxz=19, native_mpp=0.3,
        credit='Esri World Imagery (Esri, Maxar, Earthstar Geographics, and the GIS User Community)',
        license='Esri Terms of Use - attribution required; personal/non-commercial use only',
        source_url='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'),
}

# ----------------------------------------------------------------------------------------------
# Levels
# ----------------------------------------------------------------------------------------------
LEVELS = [
    ('earth-globe', 1.45e7), ('earth-5000km', 5e6), ('earth-1600km', 1.6e6),
    ('earth-500km', 5e5), ('earth-160km', 1.6e5), ('earth-50km', 5e4), ('earth-16km', 1.6e4),
    ('earth-5km', 5e3), ('earth-1600m', 1.6e3), ('earth-500m', 500.0), ('earth-160m', 160.0),
]


def level_sources(lid, provider):
    table = {
        'earth-globe': ['bmng_eq'],
        'earth-5000km': ['gibs_bmng', 'bmng_eq'],
        'earth-1600km': ['gibs_bmng', 'bmng_eq'],
        'earth-500km': ['weld', 'gibs_bmng'],
        'earth-160km': ['weld', 'gibs_bmng'],
        'earth-50km': ['usgs', 'weld', 'gibs_bmng'],
        'earth-16km': ['usgs', 'weld', 'gibs_bmng'],
        'earth-5km': ['nyc2018', 'usgs', 'weld'],
        'earth-1600m': ['nyc2018', 'usgs', 'weld'],
        'earth-500m': ['nyc2018', 'usgs', 'weld'],
        'earth-160m': ['nyc2018', 'usgs', 'weld'],
    }
    s = list(table[lid])
    if provider == 'esri' and lid not in ('earth-globe', 'earth-5000km', 'earth-1600km'):
        s = ['esri', 'weld', 'gibs_bmng']
    return s


# Zoom story for the default (Sheep Meadow) target.  Captions <= 90 chars.
NYC_TEXT = {
    'earth-globe': ('Earth', 'Earth is 12,742 km across. We are diving toward New York City.'),
    'earth-5000km': ('North America',
                     "The Great Lakes hold about a fifth of the world's fresh surface water."),
    'earth-1600km': ('US East Coast',
                     'About 50 million people live along the Boston-to-Washington corridor.'),
    'earth-500km': ('New York region',
                    'The Hudson River meets the Atlantic at one of the largest natural harbors.'),
    'earth-160km': ('New York metro area',
                    'Around 20 million people live in the New York metropolitan area.'),
    'earth-50km': ('New York City',
                   "NYC's five boroughs cover about 780 km² and are home to roughly 8.5 million people."),
    'earth-16km': ('Manhattan', 'Manhattan is 21.6 km long and at most 3.7 km wide.'),
    'earth-5km': ('Central Park',
                  'Central Park: 341 hectares of green, 4 km long and 0.8 km wide.'),
    'earth-1600m': ('Southern Central Park',
                    'About 18,000 trees grow in Central Park.'),
    'earth-500m': ('Sheep Meadow',
                   'Sheep Meadow: 15 acres of lawn where real sheep grazed until 1934.'),
    'earth-160m': ('On the lawn',
                   'Sheep Meadow is a designated quiet zone, made for sunbathing and picnics.'),
}


def generic_text(lid, width):
    labels = {
        'earth-globe': 'Earth', 'earth-5000km': 'A continent', 'earth-1600km': 'A region',
        'earth-500km': 'A landscape', 'earth-160km': 'A metro area', 'earth-50km': 'A city',
        'earth-16km': 'A district', 'earth-5km': 'A neighbourhood', 'earth-1600m': 'A park',
        'earth-500m': 'A lawn', 'earth-160m': 'On the ground'}
    km = width / 1000
    cap = (f'This view is {km:,.0f} km across.' if km >= 1 else f'This view is {width:,.0f} m across.')
    return labels.get(lid, lid), cap


# ----------------------------------------------------------------------------------------------
# Colour (per-channel Lab tone curves)
# ----------------------------------------------------------------------------------------------
LUT_DOM = [np.linspace(0, 100, 1001), np.linspace(-128, 128, 1025), np.linspace(-128, 128, 1025)]


def to_lab(rgb):
    return cv2.cvtColor(np.clip(rgb / 255.0, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)


def from_lab(lab):
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB) * 255.0, 0, 255)


def lut_identity():
    return [d.copy() for d in LUT_DOM]


def lut_blend(lut, w):
    return [d + w * (l - d) for d, l in zip(LUT_DOM, lut)]


def lut_apply_lab(lab, lut):
    out = np.empty_like(lab)
    for c in range(3):
        out[..., c] = np.interp(lab[..., c], LUT_DOM[c], lut[c])
    return out


def lut_apply(rgb, lut):
    return from_lab(lut_apply_lab(to_lab(rgb), lut))


def fit_lut(src, ref, max_slope=2.0, min_slope=0.5):
    """Monotone per-channel curves mapping the distribution of src (M,3 Lab) onto ref (K,3 Lab):
    70 % quantile mapping + 30 % mean/std mapping, slope-limited, linear beyond the data."""
    P = np.linspace(1, 99, 50)
    lut = []
    for c in range(3):
        d = LUT_DOM[c]
        s, r = src[:, c], ref[:, c]
        qs, qr = np.percentile(s, P), np.percentile(r, P)
        qs = qs + np.arange(len(qs)) * 1e-4          # strictly increasing
        qr = np.maximum.accumulate(qr)
        sd_s, sd_r = max(s.std(), 1e-3), max(r.std(), 1e-3)
        k = float(np.clip(sd_r / sd_s, min_slope, max_slope))
        lin = r.mean() + (d - s.mean()) * k
        q = np.interp(d, qs, qr)
        q = np.where(d < qs[0], qr[0] + (d - qs[0]) * k, q)
        q = np.where(d > qs[-1], qr[-1] + (d - qs[-1]) * k, q)
        y = 0.7 * q + 0.3 * lin
        # slope limit, re-integrate around the median
        dy = np.clip(np.diff(y) / np.diff(d), min_slope, max_slope) * np.diff(d)
        y2 = np.concatenate([[0], np.cumsum(dy)])
        m = np.interp(np.median(s), d, y) - np.interp(np.median(s), d, y2)
        lut.append(y2 + m)
    return lut


def footprint_resample(child, child_w, parent_shape, parent_w):
    """Resample a (finer, centred) child image into the parent pixel grid.  Works for square or
    rectangular images; widths are the horizontal extents in metres.  Returns (img, cov, box)
    where box = (i0, i1, j0, j1) are the parent rows/cols covered by the child's footprint."""
    ph, pw_px = parent_shape[:2]
    ch, cw_px = child.shape[:2]
    ppx = parent_w / pw_px                              # parent m/px
    cpx = child_w / cw_px                               # child m/px
    s = ppx / cpx                                       # child px per parent px (>1)
    if s > 1.01:
        w2, h2 = max(8, int(round(cw_px / s))), max(8, int(round(ch / s)))
        small = cv2.resize(child.astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA)
    else:
        small = child.astype(np.float32)
    sh, sw = small.shape[:2]
    child_h = child_w * ch / cw_px
    kx, ky = ppx / (child_w / sw), ppx / (child_h / sh)
    hx, hy = child_w / 2 / ppx, child_h / 2 / ppx
    j0, j1 = max(0, int(math.floor(pw_px / 2 - hx))), min(pw_px, int(math.ceil(pw_px / 2 + hx)))
    i0, i1 = max(0, int(math.floor(ph / 2 - hy))), min(ph, int(math.ceil(ph / 2 + hy)))
    # parent px j -> small coord u = (j + 0.5 - W/2) * kx + sw/2 - 0.5 (same for rows)
    tx = (j0 + 0.5 - pw_px / 2) * kx + sw / 2 - 0.5
    ty = (i0 + 0.5 - ph / 2) * ky + sh / 2 - 0.5
    M = np.array([[kx, 0, tx], [0, ky, ty]], np.float64)
    size = (j1 - j0, i1 - i0)
    img = cv2.warpAffine(small, M, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ones = np.ones((sh, sw), np.float32)
    cov = cv2.warpAffine(ones, M, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, cov, (i0, i1, j0, j1)


def cheb_q(parent_shape, box, parent_w, child_w, child_aspect=1.0):
    """Normalised Chebyshev distance from the centre in units of the child's half extents
    (child_aspect = child height / width)."""
    ph, pw_px = parent_shape[:2]
    i0, i1, j0, j1 = box
    ppx = parent_w / pw_px
    cx = (np.arange(j0, j1) + 0.5 - pw_px / 2) * ppx / (child_w / 2)
    cy = (np.arange(i0, i1) + 0.5 - ph / 2) * ppx / (child_w * child_aspect / 2)
    return np.maximum(np.abs(cx)[None, :], np.abs(cy)[:, None])


# ----------------------------------------------------------------------------------------------
# Clouds + atmosphere (identical, position-based functions for every level -> consistent)
# ----------------------------------------------------------------------------------------------
CLOUD_D0, CLOUD_D1 = 1.25e6, 3.3e6          # ground distance (m) where clouds fade in / are full
HAZE_RGB = np.array([0.50, 0.66, 1.00], np.float32)


def choose_cloud_shift(lat0, lon0):
    """Rotate the (artistic, undated) cloud layer in longitude so that the target region is
    naturally clear.  Returns the shift in degrees."""
    img = SOURCES['clouds'].image()
    small = cv2.resize(img, (720, 360), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    lat = 90 - (np.arange(360) + 0.5) / 2
    lon = -180 + (np.arange(720) + 0.5) / 2
    LA, LO = np.meshgrid(np.radians(lat), np.radians(lon), indexing='ij')
    p0, l0 = math.radians(lat0), math.radians(lon0)
    best = None
    for shift in range(0, 360, 2):
        lo = LO + math.radians(shift)
        cosd = (np.sin(LA) * math.sin(p0) + np.cos(LA) * math.cos(p0) * np.cos(lo - l0))
        d = np.arccos(np.clip(cosd, -1, 1)) * R_EARTH
        wa = np.cos(LA)
        inner = d < CLOUD_D0 * 1.3
        ring = (d > CLOUD_D0) & (d < 3.5e6)
        a_in = (small * wa)[inner].sum() / wa[inner].sum()
        a_ring = (small * wa)[ring].sum() / wa[ring].sum()
        score = a_in - 0.25 * min(a_ring, 0.3)
        if shift == 0:
            natural = (score, shift, a_in, a_ring)
        if best is None or score < best[0]:
            best = (score, shift, a_in, a_ring)
    if natural[2] <= 0.12:          # real cloud positions are fine -> don't rotate
        best = natural
    print(f'  cloud layer shifted by {best[1]} deg lon (inner cover {best[2]:.2f}, '
          f'ring {best[3]:.2f})')
    return best[1]


def remove_ice(cloud):
    """The Blue Marble cloud layer also contains the bright Greenland/Antarctic ice sheets; when
    the layer is rotated those would become fake 'clouds', so blank them (BMNG ice mask)."""
    h, w = cloud.shape
    bm = cv2.resize(SOURCES['bmng_eq'].image(), (w, h), interpolation=cv2.INTER_AREA)
    lat = 90 - (np.arange(h) + 0.5) / h * 180
    ice = (bm.min(2) > 150) & (np.abs(lat)[:, None] > 55)
    ice = cv2.dilate(ice.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    out = cloud.copy()
    out[ice] = 0
    return out


def render_clouds(view, shift):
    """Cloud opacity (N,N) for this view, faded out near the target."""
    img = SOURCES['clouds'].image()
    h, w = img.shape
    if shift % 360:
        img = remove_ice(img)
    src_mpp = 2 * math.pi * R_EARTH / w
    interp = cv2.INTER_CUBIC if view.mpp < src_mpp else cv2.INTER_LINEAR

    def coord_fn(lat, lon):
        lonw = np.mod(lon + 180.0 - shift, 360.0)
        return lonw / 360.0 * w - 0.5, np.clip((90.0 - lat) / 180.0 * h - 0.5, 0, h - 1), \
            np.ones(lat.shape, bool)

    a, cov = sample_view(view, coord_fn, img, None, interp, border=cv2.BORDER_WRAP)
    a = np.clip(a[..., 0] / 255.0, 0, 1)
    # gentle contrast so up-sampled clouds keep defined edges; suppress faint haze
    a = smoothstep(0.18, 0.95, a) ** 0.9
    x, y = view.pixel_plane()
    rho = np.hypot(x, y)
    d = R_EARTH * np.arcsin(np.clip(rho / R_EARTH, 0, 1))
    return (a * smoothstep(CLOUD_D0, CLOUD_D1, d)).astype(np.float32)


def apply_clouds(rgb, alpha):
    if alpha is None or alpha.max() <= 0:
        return rgb
    cloud = np.array([246, 247, 250], np.float32)
    # slightly grey, thicker cloud cores stay bright; thin cloud lets surface through
    a = alpha[..., None]
    return rgb * (1 - a) + cloud * a


def atmosphere(rgb, view, strength=1.0):
    """Limb darkening + blue haze as a function of mu (cos of angle from the target).
    Identity at mu = 1, so small levels are untouched while globe/5000 km stay consistent."""
    x, y = view.pixel_plane()
    rho = np.hypot(x, y)
    mu = np.sqrt(np.clip(1 - (rho / R_EARTH) ** 2, 1e-4, 1))
    f = 1 / mu - 1                                      # extra air mass relative to nadir
    if f.max() < 1e-4:
        return rgb
    trans = np.exp(-0.10 * strength * f) * (0.35 + 0.65 * mu ** 0.35)
    haze = (1 - np.exp(-0.30 * strength * f)) * 0.62
    out = rgb * trans[..., None] + (255 * HAZE_RGB)[None, None, :] * haze[..., None]
    return out


def glow(view):
    """Additive thin atmospheric rim outside the disk (black space background)."""
    x, y = view.pixel_plane()
    rho = np.hypot(x, y)
    h = np.maximum(rho - R_EARTH, 0)
    g = 0.62 * np.exp(-h / 28e3) + 0.10 * np.exp(-h / 120e3)
    g = np.where(rho > R_EARTH, g, 0)
    return (255 * HAZE_RGB)[None, None, :] * g[..., None]


# ----------------------------------------------------------------------------------------------
# Rendering pipeline
# ----------------------------------------------------------------------------------------------
def soften(cov, px):
    """Turn a coverage mask into a feathered weight that is 0 at gaps and ramps up inside."""
    if px <= 0:
        return cov
    v = (cov > 0.98).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    v = cv2.erode(v, k).astype(np.float32)
    v = cv2.GaussianBlur(v, (0, 0), px / 2)
    return np.clip(np.minimum(v, cov), 0, 1)


def render_natural(lid, view, sources, globe=False):
    n = view.n
    acc = np.zeros((n, n, 3), np.float32)
    cover = np.zeros((n, n), np.float32)
    infos = []
    feather = max(2, n // 256)
    for i, key in enumerate(sources):
        need = None
        if i > 0:
            need = cover < 0.999
            if globe:
                x, y = view.pixel_plane()
                need &= np.hypot(x, y) <= R_EARTH * 1.001
            if not need.any():
                break
        print(f'  [{lid}] source {key}' + ('' if need is None else f' (filling {need.mean():.1%})'))
        res = SOURCES[key].render(view, need)
        if res is None:
            continue
        rgb, cov, info = res
        info['coverage'] = float(round((cov > 0.5).mean(), 4))
        if i > 0 and cover.max() > 0:
            ov = (cover > 0.99) & (cov > 0.99)
            if ov.sum() > 20000:
                sel = np.flatnonzero(ov.ravel())
                sel = sel[:: max(1, len(sel) // 400000)]
                cur = acc.reshape(-1, 3)[sel] / cover.ravel()[sel, None]
                lut = fit_lut(to_lab(rgb.reshape(-1, 3)[sel][None])[0], to_lab(cur[None])[0])
                rgb = lut_apply(rgb, lut)
                info['colour_matched_to_primary'] = True
        if i == 0:
            primary_cov = cov
        w = (cov if globe or i == len(sources) - 1 else soften(cov, feather)) * (1 - cover)
        acc += rgb * w[..., None]
        cover += w
        infos.append(info)
    out = acc / np.maximum(cover, 1e-6)[..., None]
    if sources[0] in ('weld', 'usgs', 'esri') and view.width >= 3e4:
        out, cover = calm_open_water(lid, view, out, cover, infos, primary_cov,
                                     shore_m=float(np.clip(view.width / 40, 800, 2500)),
                                     smooth_m=max(9000.0, view.width / 8))
    if not globe:
        out[cover < 1e-3] = 0
        gap = float((cover < 0.5).mean())
        if gap > 0:
            print(f'  WARNING [{lid}]: {gap:.2%} of the image has no imagery')
    return np.clip(out, 0, 255), cover, infos


def calm_open_water(lid, view, out, cover, infos, primary_cov, shore_m=2500.0, smooth_m=9000.0):
    """Landsat mosaics show scene seams / sun-glint patches / no-data corners over open water.
    Using the MODIS 250 m water mask, water more than `shore_m` from any shore is replaced by a
    heavily smoothed (normalised-convolution) version of the Landsat water colour itself, so broad
    natural colour variations stay but seams disappear; no-data gets the median water colour."""
    res = SOURCES['water'].render(view, None)
    if res is None:
        return out, cover
    wm, wcov, _ = res
    water = ((wm[..., 1] > 128) & (wcov > 0.5)).astype(np.uint8)
    if water.mean() < 0.002:
        return out, cover
    n = view.n
    e = int(max(2, round(shore_m / view.mpp)))
    core = cv2.erode(water, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)))
    if core.mean() < 0.001:
        return out, cover
    w = cv2.GaussianBlur(core.astype(np.float32), (0, 0), max(1.0, e / 2)) * core
    w = cv2.GaussianBlur(w, (0, 0), max(1.0, e / 3))
    # Landsat water samples (away from the shore by ~500 m, valid, not black)
    e2 = int(max(1, round(500 / view.mpp)))
    wat2 = cv2.erode(water, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e2 + 1, 2 * e2 + 1)))
    kv = int(max(2, view.n // 400))                     # stay clear of no-data edge artefacts
    valid = cv2.erode((primary_cov > 0.99).astype(np.uint8), np.ones((2 * kv + 1, 2 * kv + 1), np.uint8))
    m = ((wat2 > 0) & (valid > 0) & (out.max(2) > 12)).astype(np.float32)
    calm0 = np.median(out[m > 0], axis=0) if m.sum() > 500 else np.array([22, 38, 52], np.float32)
    # normalised convolution at reduced resolution
    f = max(1, int(round((smooth_m / view.mpp) / 12)))
    sm = max(8, n // f)
    ms = cv2.resize(m, (sm, sm), interpolation=cv2.INTER_AREA)
    cs = cv2.resize(out * m[..., None], (sm, sm), interpolation=cv2.INTER_AREA)
    sig = smooth_m / view.mpp * sm / n
    ms_b = cv2.GaussianBlur(ms, (0, 0), sig)
    cs_b = cv2.GaussianBlur(cs, (0, 0), sig)
    calm = cs_b / np.maximum(ms_b, 1e-6)[..., None]
    trust = smoothstep(0.02, 0.15, ms_b)[..., None]
    calm = calm * trust + calm0[None, None, :] * (1 - trust)
    calm = cv2.resize(calm, (n, n), interpolation=cv2.INTER_CUBIC)
    # water where the primary mosaic has no data is always replaced (even close to shore), with
    # a LOCAL water colour (normalised convolution over ~1.5 km) and a wide feather
    nod = ((water > 0) & (primary_cov < 0.5)).astype(np.float32)
    if nod.any():
        fl = max(1, int(round((1500 / view.mpp) / 4)))
        sl = max(8, n // fl)
        mls = cv2.resize(m, (sl, sl), interpolation=cv2.INTER_AREA)
        cls = cv2.resize(out * m[..., None], (sl, sl), interpolation=cv2.INTER_AREA)
        ls = max(1.0, 1500 / view.mpp * sl / n)
        mb = cv2.GaussianBlur(mls, (0, 0), ls)
        loc = cv2.GaussianBlur(cls, (0, 0), ls) / np.maximum(mb, 1e-6)[..., None]
        lt = smoothstep(0.02, 0.2, mb)[..., None]
        loc = loc * lt + cv2.resize(calm, (sl, sl), interpolation=cv2.INTER_AREA) * (1 - lt)
        loc = cv2.resize(loc, (n, n), interpolation=cv2.INTER_CUBIC)
        fw = max(2.0, n / 200)
        nods = np.clip(cv2.GaussianBlur(cv2.dilate(nod, np.ones((5, 5), np.uint8)), (0, 0), fw) * 1.6, 0, 1)
        nods *= cv2.GaussianBlur(water.astype(np.float32), (0, 0), 2)
        lw = (nods * (1 - w))[..., None]          # local colour only near shore; open water stays calm
        calm = calm * (1 - lw) + loc * lw
        w = np.maximum(w, nods)
    out = out * (1 - w[..., None]) + calm * w[..., None]
    cover = np.maximum(cover, core.astype(np.float32))
    infos.append(dict(source='water', role='open-water mask', calm_rgb=[round(float(c), 1) for c in calm0],
                      eff_mpp=250.0, coverage=float(round(core.mean(), 4))))
    print(f'  [{lid}] open water calmed: {core.mean():.1%} of image, median colour {np.round(calm0, 1)}')
    return out, cover


def family(level_meta):
    """Imagery family of a level = its dominant source (Blue Marble variants count as one)."""
    srcs = [s for s in level_meta['sources'] if s.get('role') is None]
    key = max(srcs, key=lambda s: s.get('coverage', 0))['source']
    return 'bmng' if key in ('bmng_eq', 'gibs_bmng') else key


def push_pull(dw, w, eps=1e-6):
    """Smooth scattered-data fill (pyramid push-pull).  dw = values * weight (pre-multiplied),
    w = weights.  Returns a field equal to dw/w where w ~ 1, smoothly extended elsewhere."""
    pyr = [(dw, w)]
    while min(pyr[-1][1].shape) > 4:
        a, b = pyr[-1]
        size = (max(1, a.shape[1] // 2), max(1, a.shape[0] // 2))
        pyr.append((cv2.resize(a, size, interpolation=cv2.INTER_AREA),
                    cv2.resize(b, size, interpolation=cv2.INTER_AREA)))
    a, b = pyr[-1]
    est = a / np.maximum(b, eps)[..., None]
    for a, b in reversed(pyr[:-1]):
        up = cv2.resize(est, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        up = cv2.GaussianBlur(up, (0, 0), 1.0)
        conf = np.clip(b * 2.0, 0, 1)[..., None]
        est = (a / np.maximum(b, eps)[..., None]) * conf + up * (1 - conf)
    return est


def extend_field(dw, w, iters=(400, 250, 150)):
    """Smoothly extend a field known where w ~ 1 (dw = values * w) to the whole grid: a
    membrane (harmonic) fill, coarse-to-fine, initialised with push-pull.  The result is
    continuous at the edge of the known region and has no seams -- this removes the square
    'halo' a plain average would leave around a footprint."""
    H, W = w.shape
    if not (w > 0.5).any():
        return np.zeros_like(dw)
    est = push_pull(dw, w)
    for f, it in zip((4, 2, 1), iters):
        h, ww = max(4, H // f), max(4, W // f)
        est = cv2.resize(est, (ww, h), interpolation=cv2.INTER_LINEAR)
        ws = cv2.resize(w, (ww, h), interpolation=cv2.INTER_AREA)
        kv = cv2.resize(dw, (ww, h), interpolation=cv2.INTER_AREA) / np.maximum(ws, 1e-6)[..., None]
        k = ws > 0.98
        for _ in range(it):
            est = cv2.blur(est, (3, 3), borderType=cv2.BORDER_REPLICATE)
            est[k] = kv[k]
    return est


def lut_compose(f, g):
    """x -> f(g(x))"""
    return [np.interp(g[c], LUT_DOM[c], f[c]) for c in range(3)]


RELAX = 0.6        # share of the accumulated colour correction kept per same-source step upwards


def harmonise(nat, width, cnat, csurf, cw, clut, same, globe, inject=True, use_lut=True,
              child_blur=0.0, feather_q0=0.72):
    """Colour-harmonise a parent level with its (already final) child.

    1. M  = tone curves mapping the parent's natural colours onto the child's natural colours in
       the child's footprint.  Parent LUT = child LUT o M  (full match at a source switch; at a
       same-source step only RELAX of the accumulated correction is kept, so the coarse levels
       drift gently back to their natural look -- the globe stays close to real Blue Marble).
    2. Low-frequency residual (child - parent) inside the footprint is extrapolated smoothly
       outwards (no square halo), then the down-sampled child is feathered into the centre, so
       the parent's footprint equals the child and the dissolve has no pop."""
    H, Wd = nat.shape[:2]
    asp = cnat.shape[0] / cnat.shape[1]
    img_n, cov, box = footprint_resample(cnat, cw, nat.shape, width)
    i0, i1, j0, j1 = box
    q = cheb_q(nat.shape, box, width, cw, asp)
    region = nat[i0:i1, j0:j1]
    sel = (q < 0.97) & (cov > 0.99)
    if globe:
        sel &= region.sum(2) > 0
    if use_lut:
        a = to_lab(region[sel][None])[0]
        b = to_lab(img_n[sel][None])[0]
        step = max(1, len(a) // 400000)
        M = fit_lut(a[::step], b[::step])
        full = lut_compose(clut, M)
        lut = lut_blend(full, RELAX) if same else full
        surf = lut_apply(nat, lut)
    else:
        lut = clut
        surf = nat.copy()
    # residual low-frequency correction, extrapolated outwards
    img, cov, _ = footprint_resample(csurf, cw, nat.shape, width)
    if child_blur > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), child_blur)
    fp = min(i1 - i0, j1 - j0)
    m = ((q < 0.97) & (cov > 0.99)).astype(np.float32)
    if globe:
        m *= (surf[i0:i1, j0:j1].sum(2) > 0)
    sig = max(2.0, fp / 14)
    mb = cv2.GaussianBlur(m, (0, 0), sig)
    d = (cv2.GaussianBlur(img * m[..., None], (0, 0), sig)
         - cv2.GaussianBlur(surf[i0:i1, j0:j1] * m[..., None], (0, 0), sig)) / np.maximum(mb, 1e-4)[..., None]
    d *= (mb > 0.3)[..., None]
    wgt = (mb > 0.3).astype(np.float32)
    D = np.zeros((H, Wd, 3), np.float32)
    Wt = np.zeros((H, Wd), np.float32)
    D[i0:i1, j0:j1] = d * wgt[..., None]
    Wt[i0:i1, j0:j1] = wgt
    f = max(1, max(H, Wd) // 512)
    size = (max(1, Wd // f), max(1, H // f))
    Ds = cv2.resize(D, size, interpolation=cv2.INTER_AREA)
    Ws = cv2.resize(Wt, size, interpolation=cv2.INTER_AREA)
    Dext = cv2.resize(extend_field(Ds, Ws), (Wd, H), interpolation=cv2.INTER_CUBIC)
    qf = cheb_q(nat.shape, (0, H, 0, Wd), width, cw, asp)
    fade = np.exp(-np.maximum(qf - 1, 0) ** 2 / 0.8 ** 2)
    surf = np.clip(surf + Dext * fade[..., None], 0, 255)
    resid = float(np.abs(d[wgt > 0]).mean()) if wgt.any() else 0.0
    if inject:
        mm = smoothstep(1.0, feather_q0, q) * np.clip(cov, 0, 1)
        surf[i0:i1, j0:j1] = surf[i0:i1, j0:j1] * (1 - mm[..., None]) + img * mm[..., None]
    shift_nat = float(np.abs(to_lab(surf[::8, ::8]) - to_lab(nat[::8, ::8])).mean())
    return surf, lut, dict(mean_abs_lab_change_vs_natural=round(shift_nat, 2),
                           residual_rgb_after_lut=round(resid, 2))


def registration_check(parent, pw, child, cw):
    """Measure the residual mis-registration between a parent's footprint and its (down-sampled)
    child with phase correlation of high-passed luminance.  Returns (dx_m, dy_m, response):
    how far the parent's content sits east/north of the child's."""
    img, cov, box = footprint_resample(child, cw, parent.shape, pw)
    i0, i1, j0, j1 = box
    reg = parent[i0:i1, j0:j1]
    k = int(min(i1 - i0, j1 - j0) * 0.8)
    ci, cj = (i1 - i0) // 2, (j1 - j0) // 2
    k -= k % 2
    r0, c0 = ci - k // 2, cj - k // 2
    a = np.ascontiguousarray(reg[r0:r0 + k, c0:c0 + k].mean(2), np.float32)
    b = np.ascontiguousarray(img[r0:r0 + k, c0:c0 + k].mean(2), np.float32)
    hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 4)
    win = cv2.createHanningWindow((k, k), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(hp(b), hp(a), win)
    ppx = pw / parent.shape[1]
    return round(dx * ppx, 2), round(-dy * ppx, 2), round(float(resp), 3)


def save_jpg(path, rgb, q=90):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb + 0.5, 0, 255).astype(np.uint8)).save(
        path, 'JPEG', quality=q, optimize=True, progressive=True)


def save_preview(lid, rgb, size=1024):
    h, w = rgb.shape[:2]
    small = cv2.resize(np.clip(rgb, 0, 255).astype(np.float32),
                       (size, max(1, int(round(size * h / w)))), interpolation=cv2.INTER_AREA)
    save_jpg(PREVIEW_DIR / f'{lid}.jpg', small, 88)


def pair_sheet(parent_id, parent, pw, child_id, child, cw, size=768):
    """Side-by-side: parent's footprint crop (up-scaled) | child (down-scaled) | 50/50 mix."""
    img, cov, (i0, i1, j0, j1) = footprint_resample(child, cw, parent.shape, pw)
    h = max(1, int(round(size * (i1 - i0) / (j1 - j0))))
    crop = cv2.resize(parent[i0:i1, j0:j1].astype(np.float32), (size, h), interpolation=cv2.INTER_CUBIC)
    ch = cv2.resize(child.astype(np.float32), (size, h), interpolation=cv2.INTER_AREA)
    mix = 0.5 * crop + 0.5 * ch
    bar = np.full((h, 8, 3), 255, np.float32)
    save_jpg(PREVIEW_DIR / 'pairs' / f'{parent_id}__{child_id}.jpg',
             np.concatenate([crop, bar, ch, bar, mix], axis=1), 88)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lat', type=float, default=DEFAULT_LAT)
    ap.add_argument('--lon', type=float, default=DEFAULT_LON)
    ap.add_argument('--size', type=int, default=4096)
    ap.add_argument('--provider', choices=['usgs', 'esri'], default='usgs')
    ap.add_argument('--force', action='store_true', help='ignore cached natural renders')
    ap.add_argument('--only-natural', action='store_true', help='stop after natural renders')
    ap.add_argument('--levels', default='', help='comma list of level ids to (re)render naturally')
    ap.add_argument('--no-person', action='store_true', help='skip the person photos')
    ap.add_argument('--no-inject', action='store_true', help="don't feather children into parents")
    ap.add_argument('--cloud-shift', default='0',
                    help="rotate the cloud layer in longitude (deg) or 'auto' = pick a clear target area;"
                         " default 0 = real cloud positions (clouds always fade out near the target)")
    args = ap.parse_args()

    if args.provider == 'esri':
        print('NOTE: --provider esri uses Esri World Imagery. It requires the attribution '
              '"Esri, Maxar, Earthstar Geographics, and the GIS User Community" and is for '
              'personal / non-commercial use only (Esri terms of use).')

    lat0, lon0, n = args.lat, args.lon, args.size
    is_nyc = (abs(lat0 - DEFAULT_LAT) < 0.02 and abs(lon0 - DEFAULT_LON) < 0.02)
    tag = f'{RENDER_VERSION}_{args.provider}_{n}_{lat0:.6f}_{lon0:.6f}'
    only = set(filter(None, args.levels.split(',')))
    for d in (WORK_DIR, PREVIEW_DIR, OUT_DIR, MANIFEST.parent, TILE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---- 1. natural renders (cached) -----------------------------------------------------
    meta = {}
    for lid, width in LEVELS:
        globe = lid == 'earth-globe'
        view = View(width, n, lat0, lon0, ss=3 if width >= 5e6 else 2)
        cache = WORK_DIR / f'{lid}_{tag}.png'
        mcache = WORK_DIR / f'{lid}_{tag}.json'
        if cache.exists() and mcache.exists() and not args.force and (not only or lid not in only):
            meta[lid] = json.loads(mcache.read_text())
            continue
        t0 = time.time()
        print(f'[{lid}] rendering natural image, width {width:g} m, {view.mpp:.4g} m/px')
        rgb, cover, infos = render_natural(lid, view, level_sources(lid, args.provider), globe)
        cv2.imwrite(str(cache), cv2.cvtColor(np.clip(rgb + 0.5, 0, 255).astype(np.uint8),
                                             cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if globe:
            cv2.imwrite(str(WORK_DIR / f'{lid}_{tag}_cover.png'),
                        np.clip(cover * 255 + 0.5, 0, 255).astype(np.uint8))
        meta[lid] = dict(sources=infos)
        mcache.write_text(json.dumps(meta[lid], indent=1))
        save_preview(lid + '.natural', rgb)
        print(f'  done in {time.time() - t0:.0f}s: {infos}')
    # registration of the natural renders (independent sources) -- diagnostic only
    for (p, pw), (c, cw) in zip(LEVELS[:-1], LEVELS[1:]):
        pa = cv2.cvtColor(cv2.imread(str(WORK_DIR / f'{p}_{tag}.png')), cv2.COLOR_BGR2RGB).astype(np.float32)
        ca = cv2.cvtColor(cv2.imread(str(WORK_DIR / f'{c}_{tag}.png')), cv2.COLOR_BGR2RGB).astype(np.float32)
        dx, dy, resp = registration_check(pa, pw, ca, cw)
        meta[p]['registration_vs_child_m'] = [dx, dy, resp]
        print(f'  registration {p} vs {c}: parent offset {dx:+.2f} m E, {dy:+.2f} m N '
              f'({abs(dx) / (pw / n) if pw else 0:.2f}/{abs(dy) / (pw / n):.2f} parent px, response {resp})')
    if args.only_natural:
        return

    # ---- 2. photo levels below earth-160m (person-wide, person) ---------------------------
    photos = []
    if not args.no_person:
        try:
            import person_photos
            ref = cv2.cvtColor(cv2.imread(str(WORK_DIR / f'{LEVELS[-1][0]}_{tag}.png')), cv2.COLOR_BGR2RGB)
            photos = person_photos.prepare(ref.astype(np.float32), LEVELS[-1][1])
        except FileNotFoundError as e:
            print(f'  person photos skipped: {e}')

    # ---- 3. harmonise fine -> coarse, inject children, clouds/atmosphere, write -------------
    if args.cloud_shift == 'auto':
        shift = choose_cloud_shift(lat0, lon0)
    else:
        shift = int(float(args.cloud_shift)) % 360
    chain = [dict(id=l, width=w, kind='earth') for l, w in LEVELS]
    chain += [dict(id=p['id'], width=p['width'], kind='photo', photo=p) for p in photos]
    child = None            # (id, width, surface image float32, natural image, LUT, kind)
    finals = {}
    for lev in reversed(chain):
        lid, width, kind = lev['id'], lev['width'], lev['kind']
        globe = lid == 'earth-globe'
        if kind == 'photo':
            nat = lev['photo']['img']
            meta[lid] = dict(sources=[], grade=lev['photo']['grade'])
        else:
            nat = cv2.cvtColor(cv2.imread(str(WORK_DIR / f'{lid}_{tag}.png')), cv2.COLOR_BGR2RGB)
            nat = nat.astype(np.float32)
        view = View(width, nat.shape[0], lat0, lon0)
        cov_img = None
        if globe:
            cov_img = cv2.imread(str(WORK_DIR / f'{lid}_{tag}_cover.png'),
                                 cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        surf, lut = nat, lut_identity()
        if child is not None:
            cid, cw, csurf, cnat, clut, ckind = child
            photo_link = kind == 'photo' or ckind == 'photo'
            same = photo_link or family(meta[lid]) == family(meta[cid])
            blur = 0.0
            if kind == 'earth' and ckind == 'photo':
                # make the (sharp) photo look like the aerial where it is feathered in
                eff = max(s['eff_mpp'] for s in meta[lid]['sources'][:1])
                blur = 0.25 * eff / (width / nat.shape[1])
            surf, lut, stats = harmonise(nat, width, cnat, csurf, cw, clut, same, globe,
                                         inject=not args.no_inject, use_lut=not photo_link,
                                         child_blur=blur, feather_q0=0.85 if photo_link else 0.72)
            if globe:
                surf *= (cov_img > 0)[..., None]
            meta[lid]['matched_to'] = cid
            meta[lid]['harmonise'] = stats
            print(f'[{lid}] harmonised to {cid} ({"same family / photo" if same else "source switch"}): '
                  f'{stats}')
        final = surf
        if kind == 'earth':
            # clouds + atmosphere (position-based -> identical in every level)
            half_diag = width / 2 * math.sqrt(2)
            if half_diag > CLOUD_D0:
                final = apply_clouds(final, render_clouds(View(width, n, lat0, lon0, ss=2), shift))
                meta[lid]['clouds'] = True
            final = atmosphere(final, view)
            if globe:
                final = final * cov_img[..., None] + glow(view)
        final = np.clip(final, 0, 255)
        save_jpg(OUT_DIR / f'{lid}.jpg', final)
        save_preview(lid, final)
        fh, fw = final.shape[:2]
        finals[lid] = cv2.resize(final, (1536, int(round(1536 * fh / fw))), interpolation=cv2.INTER_AREA)
        meta[lid]['px'] = [final.shape[1], final.shape[0]]
        print(f'[{lid}] written')
        child = (lid, width, surf, nat, lut, kind)

    for p, c in zip(chain[:-1], chain[1:]):
        pair_sheet(p['id'], finals[p['id']], p['width'], c['id'], finals[c['id']], c['width'])

    # ---- 4. manifest -----------------------------------------------------------------------
    entries = []
    for lid, width in LEVELS:
        label, caption = NYC_TEXT[lid] if is_nyc else generic_text(lid, width)
        srcs = meta[lid]['sources']
        used = [s['source'] for s in srcs]
        credit = '; '.join(dict.fromkeys(SOURCES[k].credit for k in used))
        lic = '; '.join(dict.fromkeys(SOURCES[k].license for k in used))
        urls = ' | '.join(dict.fromkeys(SOURCES[k].source_url for k in used))
        eff = max(width / n, min(s['eff_mpp'] for s in srcs[:1]))
        notes = (f'Orthographic projection centred on {lat0:.6f}, {lon0:.6f} (sphere R=6371008.8 m), '
                 f'north-up; {width / n:.4g} m/px (effective source resolution ~{eff:.3g} m/px). '
                 f'Sources: ' + ', '.join(f"{s['source']}" + (f" z{s['zoom']}" if 'zoom' in s else '')
                                          + f" ({s['coverage']:.0%})" for s in srcs) + '.')
        mt = meta[lid].get('matched_to')
        if mt and mt.startswith('earth-'):
            notes += (f' Colours tone-matched to {mt} inside its footprint; {mt} (down-sampled) is '
                      f'feathered into the centre so the dissolve has no pop.')
        elif mt:
            notes += (f' The {mt} drone photo (down-sampled, softened to the aerial resolution) is '
                      f'feathered into the centre, so the person is already visible and the dissolve is seamless.')
        reg = meta[lid].get('registration_vs_child_m')
        if reg:
            notes += (f' Measured registration of the raw sources vs the next level: {reg[0]:+.1f} m E, '
                      f'{reg[1]:+.1f} m N.')
        if meta[lid].get('clouds'):
            notes += (' NASA Blue Marble cloud layer' + (f' (rotated {shift} deg in longitude)' if shift else '')
                      + f', faded out within {CLOUD_D0 / 1e3:.0f} km of the target so the dive is clear.')
            if 'clouds' not in used:
                credit += '; ' + SOURCES['clouds'].credit
                urls += ' | ' + SOURCES['clouds'].source_url
        if lid == 'earth-globe':
            notes += ' Simple atmosphere: limb darkening, blue haze and rim glow; black space.'
        entries.append(dict(id=lid, file=f'assets/levels/{lid}.jpg', px=[n, n], label=label,
                            caption=caption, width_m=width, focus=[0.5, 0.5],
                            blend='add' if lid == 'earth-globe' else 'normal',
                            credit=credit, license=lic, source_url=urls, notes=notes))
    for p in photos:
        e = p['entry']
        notes = e.get('notes', '')
        pw_, ph_ = meta[p['id']]['px']
        rot = f", rotated {e.get('rotate_deg')} deg" if e.get('rotate_deg') else ''
        notes += (f" {pw_}x{ph_} crop{rot};"
                  f" grass colour-graded toward the Sheep Meadow lawn of earth-160m (Lab mean "
                  f"{p['grade']['src_mean']} -> {p['grade']['target_mean']}).")
        if meta[p['id']].get('matched_to'):
            notes += (f" The {meta[p['id']]['matched_to']} photo (same shoot, SIFT-aligned) is feathered "
                      f"into the centre for a seamless dissolve.")
        entries.append(dict(id=p['id'], file=f"assets/levels/{p['id']}.jpg", px=meta[p['id']]['px'],
                            label=e['label'], caption=e['caption'], width_m=p['width'],
                            focus=[round(p['focus'][0], 4), round(p['focus'][1], 4)] if p is photos[-1] else [0.5, 0.5],
                            blend='normal', credit=e['credit'], license=e['license'],
                            source_url=e['source_url'], notes=notes.strip()))
    MANIFEST.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'manifest: {MANIFEST} ({len(entries)} entries)')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
