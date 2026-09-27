# Scale of Everything

A scroll-driven zoom from the cosmic web to the atomic nucleus: one continuous camera move
through ~42 orders of magnitude, built from real photographs, micrographs and scientific
renders instead of 3D models.

```
start.bat            → serves the folder on http://localhost:8080 and opens it
```
(Any static web server works. Opening `index.html` directly from disk will not work, because browsers block `fetch()` on `file://`.)

Controls: scroll or swipe to zoom · **Space** plays it like a video · click the ruler on the right to jump to a scale ·
**E** opens the alignment editor · `?lite` forces the 2K textures used on phones.

Testing (needs Chrome or Edge, with `start.bat` running):

```
node tools/qa.mjs http://localhost:8080/ qa-out                            # scrolls like a user, clicks, phone test, fps, errors
node tools/shots.mjs shots http://localhost:8080/ milky-way milky-way@0.5  # full-HD stills at chosen stops
```

---

## Why 2D images and not 3D models

The viral "zoom from the universe to an atom" videos (Powers of Ten, Cosmic Eye, the AI zoom
edits) aren't 3D scenes. They're **chains of flat images**. Each photo or render is placed
inside the previous one and the camera keeps zooming while they dissolve into each other.
Everything on screen is a real photo (or a photoreal still), so the quality is the same at
every scale.

When you mix separate 3D models, each scale has its own lighting, materials and polygon
budget, and the seams show. A browser can't render a photoreal human, hair or bacteria in
real time anyway. So this site does what the videos do, but interactively:

| Piece | What it does |
|---|---|
| **Nested image chain** (`scene.json`) | Each layer has a real physical width in meters and sits at a point inside its parent. Scroll maps to log₁₀(field of view), so the zoom rate stays constant across all 42 decades. |
| **Float64 camera math** (`js/scene.js`) | Positions are resolved in the parent frame on the CPU, and only screen-relative numbers go to the GPU. That's how 10²⁶ m and 10⁻¹⁵ m share one camera without precision loss. |
| **Seam hiding** (`js/renderer.js`) | A small inner image fades in with a soft radial mask that becomes a rectangle as it fills the screen. It's temporarily color-matched to the patch of the parent it covers and eased back to its true colors as it takes over. Space images use a "lighten" blend over black, so they have no edges at all. |
| **One virtual camera** | Zoom blur that scales with scroll speed, film grain and a vignette, applied to everything. This is the same trick video editors use to hide cuts between sources. |
| **Procedural space** | Stars stream past between the galaxy and the Sun. Orbits, the Sun's glare and the "pale blue dot" are drawn procedurally, so they stay sharp at any zoom. |
| **Streaming** | Only about 7 layers around the current zoom are kept on the GPU. Phones get 2048 px versions automatically. |

## The chain

`tools/build_scene.py` stitches three manifests (each ordered largest to smallest) into `scene.json`:

1. `assets/manifest/space.json`: cosmic web → galaxy groups → Milky Way (and the Moon)
2. *(generated)* Solar System and Earth–Moon stops, with orbits, planets, the Sun and the Moon
3. `assets/manifest/earth.json`: globe → continent → New York → Central Park → a person on the grass
4. `assets/manifest/micro.json`: hand → skin → hair → bacteria → virus → DNA → atoms → nucleus → proton

Every Earth level comes from `tools/earth_levels.py`. It renders all levels in the same
orthographic projection, centred on one latitude/longitude, so each level sits exactly in the
middle of the previous one. The default target is the middle of the Sheep Meadow lawn in Central Park:

```
py tools/earth_levels.py --lat 40.77184 --lon -73.97518 --size 4096 --provider usgs
py tools/build_scene.py
```

Imagery by scale:

| Scale | Source | Licence |
|---|---|---|
| Globe, 5,000 km, 1,600 km | NASA Blue Marble and the NASA cloud layer | public domain |
| 500 km, 160 km | NASA/USGS Landsat mosaic | public domain |
| 50 km, 16 km | USGS/USDA NAIP aerial photos | public domain |
| 5 km and closer | NYC 2018 6-inch aerial photos | CC BY 4.0, needs credit |

To zoom into somewhere else, pass that place's latitude and longitude. The high-resolution
levels only exist in the US. For anywhere else, `--provider esri` uses Esri World Imagery, which
requires attribution and is for personal, non-commercial use only. Then replace the two
person photos.

The first run downloads about 6,000 map tiles (roughly 45 minutes). Later runs reuse the cache
in `assets/src/earth/tiles` and take about 10 minutes. `assets/src/earth/work` only speeds up
re-runs, so you can delete it.

## Tuning the joins: the alignment editor

Press **E** (or click *Align*) while you're between two layers:

- drag on the view to move the inner image; **Alt + wheel** scales it; **Shift + wheel** rotates it
- *Ghost* shows the inner image at 50% with hard edges, which makes lining up features easy
- *Feather*, *Fade in* and *Color match* control how the dissolve looks
- **Download scene.json** and replace the file in the project folder

`scene.json` layer fields: `src`, `px`, `width` (meters across the image), `at` ([x, y] of this
layer's center inside the parent, 0–1), `rot` (degrees), `blend` (`normal` | `add`), `feather`,
`fadeIn` ([start, end] as a fraction of the screen width), `colorMatch` (0–1), `maxUpscale`,
`grade` ({exposure, contrast, saturation}), `dot`, `dotLabel`, `credit`, `license`, `sourceUrl`.

## Making it even more seamless

Real imagery doesn't nest perfectly. A hair micrograph and a bacteria micrograph were never
the same sample. The engine hides that well, but the most seamless viral edits use one of
these methods. They plug straight into this engine, because every method outputs the same
thing: a still image plus its width in meters and where its child sits.

1. **AI outpainting chain (what most viral zooms use).** Start from the smallest image and
   repeatedly *zoom out* with outpainting (Midjourney "Zoom Out", Photoshop Generative
   Expand, Flux Fill / SDXL inpainting in ComfyUI), changing the prompt as the scale changes.
   Each new image literally contains the previous one, shrunk, in its center, so the joins
   are pixel-perfect. Use about 2–4× per step, keep the lighting and film look in the
   prompt, and set `at: [0.5, 0.5]` with `width = previous_width × zoom_factor`.
2. **Bridge frames.** Keep the real images, but for each join, paste the downscaled child
   into the center of the parent and run a low-strength img2img or inpaint over the border
   ring, so the parent's content flows into the child.
3. **Blender stills.** For anything you model (DNA, a virus, a cell), render stills at 4K with
   the same camera, lens, depth of field and color management at every scale, then add them as layers.
   Don't try to run Blender scenes in the browser.

Images should be at least 4096 px on the long side and never enlarged. Keep neighbors
within about 10× of each other in width (Earth levels use about 3.16×). Bigger jumps
between opaque images look blurry before the child takes over, and `build_scene.py` warns
about them. Space images need a truly black background.

## Deploying

The whole thing is static files: upload the folder to Netlify, Cloudflare Pages, GitHub
Pages or any host. For production you'd convert the JPEGs to KTX2/Basis GPU textures (about
6× less video memory) and maybe AVIF for download size.

## Credits

Every image's creator, license and source link is in `scene.json` and appears in the site
under **Image credits**. Keep those if you publish.
