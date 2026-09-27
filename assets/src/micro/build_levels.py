"""Process raw micro-scale sources into assets/levels/*.jpg and write assets/manifest/micro.json.

Run from anywhere:  py assets/src/micro/build_levels.py
Raw downloads live in assets/src/micro/raw, generated renders in assets/src/micro/gen.
"""
import json, os
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RAW = os.path.join(HERE, "raw")
GEN = os.path.join(HERE, "gen")
OUT = os.path.join(ROOT, "assets", "levels")
MAN = os.path.join(ROOT, "assets", "manifest", "micro.json")
MAXS = 4096


def warm(im, amount):
    """Gentle white-balance shift toward warm skin tones (amount 0..1)."""
    a = np.asarray(im).astype(np.float32)
    gains = np.array([1 + 0.10 * amount, 1 + 0.02 * amount, 1 - 0.10 * amount])
    a = np.clip(a * gains, 0, 255)
    return Image.fromarray(a.astype(np.uint8))


LEVELS = [
    dict(
        id="skin-macro", src="raw/human_skin_closeup.jpg", crop=None,
        label="Skin", caption="The back of a hand: a landscape of ridges, furrows and fine hairs.",
        width_from=lambda w_px: 2.3e-2,
        focus=[0.585, 0.47], blend="normal",
        credit="Montavius Howard (TongCreator), via Pixabay / Wikimedia Commons",
        license="CC0 1.0",
        source_url="https://commons.wikimedia.org/wiki/File:Human_skin_close-up.jpg",
        notes=("Phone macro (LG G4). No scale bar: width estimated from the vellus hair (~5.7 px thick at 5312 px, "
               "assumed ~25 um => ~4.4 um/px => ~23 mm) cross-checked with skin-furrow polygon size (~1.5 mm). "
               "Uncertain by ~+/-40%. Focus: sharp furrow-crossing just below the root of the visible hair."),
    ),
    dict(
        id="skin-closeup", src="raw/hand_skin_closeup.jpg", crop=None, post=lambda im: warm(im, 0.6),
        label="Skin surface", caption="Dry skin flakes: the outer layer is made of dead, flattened cells.",
        width_from=lambda w_px: 5.5e-3,
        focus=[0.64, 0.42], blend="normal",
        credit="Clump, via Wikimedia Commons",
        license="CC0 1.0",
        source_url="https://commons.wikimedia.org/wiki/File:Hand_skin_closeup.jpg",
        notes=("Close-up photo of dry skin on the back of a hand. No scale bar: width estimated from furrow spacing "
               "(~350 px between primary furrows, taken as ~1 mm) => ~5.5 mm. Colour warmed slightly toward "
               "skin-macro. Focus on a flaky plateau beside a diagonal furrow (resembles the NCI SEM)."),
    ),
    dict(
        id="skin-sem", src="raw/wetzel_skin.jpg", crop=(0, 0, 1800, 2150),
        label="Skin (electron microscope)", caption="Under an electron microscope, skin looks like cracked, flaking terrain.",
        width_from=lambda w_px: w_px / 359 * 100e-6,
        focus=[0.72, 0.30], blend="normal",
        credit="Bruce Wetzel & Harry Schaefer, National Cancer Institute (NIH)",
        license="Public domain",
        source_url="https://commons.wikimedia.org/wiki/File:Skin_surface_(human).jpg",
        notes=("SEM of human skin surface. 100 um scale bar = 359 px (measured) on 1800 px width => 501 um. "
               "Cropped bottom 99 px (scale bar/label). Focus on the steep diagonal skin ridge crest (upper right), "
               "which lines up with the hair shaft in hair-sem."),
    ),
    dict(
        id="hair-sem", src="raw/muse_hair_2000x.tif", crop=(0, 0, 2048, 1240),
        label="Human hair", caption="A single hair is armoured with overlapping scales called the cuticle.",
        width_from=lambda w_px: w_px / 136 * 10e-6,
        focus=[0.6, 0.42], blend="normal",
        credit="Nicola Angeli / MUSE - Museo delle Scienze, Trento (Wikimedia Commons)",
        license="CC BY-SA 3.0",
        source_url="https://commons.wikimedia.org/wiki/File:Human_hair_2000X_-_SEM_MUSE.tif",
        notes=("ZEISS SEM, 2000x. 10 um scale bar = 136 px on 2048 px => 150.6 um wide. Cropped to y<1240 to remove "
               "the data bar and the green in-image scale bar. The hair runs the same diagonal as the skin ridge "
               "in skin-sem. Focus on cuticle scale edges mid-shaft."),
    ),
    dict(
        id="bacteria", src="raw/mrsa7820.jpg", crop=(0, 0, 2835, 1670),
        label="Bacteria", caption="Staphylococcus bacteria: millions live on your skin right now.",
        width_from=lambda w_px: w_px / 560 * 5e-6,
        focus=[0.5, 0.5], blend="normal",
        credit="CDC / Janice Carr, Deepak Mandhalapu (Public Health Image Library #7820)",
        license="Public domain",
        source_url="https://commons.wikimedia.org/wiki/File:MRSA7820.jpg",
        notes="SEM 4780x of MRSA (Staphylococcus aureus). 5 um bar = 560 px on 2835 px => 25.3 um. Cropped data bar (y>=1670).",
    ),
    dict(
        id="staph-cluster", src="raw/staph_visa_20k.jpg", crop=(0, 0, 1420, 940),
        label="Staph cluster", caption="Staphylococcus grows in grape-like bunches; each ball is one cell.",
        width_from=lambda w_px: w_px / 256 * 1e-6,
        focus=[0.7, 0.6], blend="normal",
        credit="CDC / Matthew J. Arduino, DrPH (Public Health Image Library)",
        license="Public domain",
        source_url="https://commons.wikimedia.org/wiki/File:Staphylococcus_aureus_VISA_2.jpg",
        notes="Colorized SEM 20000x of VISA S. aureus. 1 um bar = 256 px on 1420 px => 5.55 um. Cropped data bar (y>=940).",
    ),
    dict(
        id="bacterium", src="raw/staph_visa_50k.jpg", crop=(0, 0, 1420, 945),
        label="Bacterium", caption="One bacterium is about a thousandth of a millimetre across.",
        width_from=lambda w_px: w_px / 317 * 0.5e-6,
        focus=[0.34, 0.47], blend="normal",
        credit="CDC / Matthew J. Arduino, DrPH (Public Health Image Library)",
        license="Public domain",
        source_url="https://commons.wikimedia.org/wiki/File:Staphylococcus_aureus_VISA.jpg",
        notes="Colorized SEM 50000x of the same VISA S. aureus sample. 500 nm bar = 317 px on 1420 px => 2.24 um. Cropped data bar (y>=945).",
    ),
    dict(
        id="phages", src="raw/beards_phage.jpg", crop=None,
        label="Bacteriophages", caption="Viruses that hunt bacteria latch on and inject their DNA.",
        width_from=lambda w_px: 1.0e-6,
        focus=[0.23, 0.604], blend="normal",
        credit="Graham Beards, via Wikimedia Commons",
        license="CC BY-SA 3.0",
        source_url="https://commons.wikimedia.org/wiki/File:Phage.jpg",
        notes=("TEM (~200,000x) of phages attached to a bacterial cell wall. No scale bar: phage heads are ~95 px; "
               "taking a typical ~65 nm capsid gives ~0.68 nm/px => ~1.0 um wide (+/-30%)."),
    ),
    dict(
        id="virus", src="raw/t2_phage.jpg", crop=(0, 0, 2021, 1870),
        label="T2 bacteriophage", caption="A virus is a protein shell packed with genes, plus a tail to inject them.",
        width_from=lambda w_px: w_px / 646 * 100e-9,
        focus=[0.605, 0.338], blend="normal",
        credit="SnaxMikn, via Wikimedia Commons",
        license="CC BY-SA 4.0",
        source_url="https://commons.wikimedia.org/wiki/File:Enterobacteria_phage_T2_transmission_electron_micrograph.jpg",
        notes="Negative-stain TEM of phage T2. 100 nm bar = 646 px on 2021 px => 313 nm. Cropped the scale bar (y>=1870).",
    ),
]


def _gmeta(name, key, default):
    try:
        d = json.load(open(os.path.join(GEN, name), encoding="utf-8"))
        for k in key:
            d = d[k]
        return d
    except Exception:
        return default


GEN_CREDIT = "Generated for this project (numpy sphere/volume renderer in assets/src/micro)"


def generated_levels():
    pdb = " Atomic coordinates: base pairs from PDB 1BNA (Drew et al. 1981, B-DNA dodecamer) placed along smooth 3D curves."
    return [
        dict(id="dna-strands", src="gen/dna-strands.jpg", label="DNA",
             caption="DNA strands: 2 nm wide, yet each of your cells packs about 2 metres of it.",
             width_m=6.0e-8, focus=_gmeta("dna_views.json", ["meta", "dna-strands", "focus"], [0.5, 0.5]),
             blend="normal", credit=GEN_CREDIT + "; coordinates from RCSB PDB 1BNA",
             license="CC0 (original render; PDB data is public domain)",
             source_url="https://www.rcsb.org/structure/1BNA",
             notes=("Render, 60 nm field (orthographic, exact). All-heavy-atom B-DNA, ~90k atoms, CPK space-filling, "
                    "world-space AO + shadow map." + pdb + " Same scene/lighting as dna-helix and atoms, so the zoom "
                    "nests exactly; focus = centre of the dna-helix view.")),
        dict(id="dna-helix", src="gen/dna-helix.jpg", label="Double helix",
             caption="The double helix: two sugar-phosphate backbones twisted around paired bases.",
             width_m=8.0e-9, focus=_gmeta("dna_views.json", ["meta", "dna-helix", "focus"], [0.5, 0.5]),
             blend="normal", credit=GEN_CREDIT + "; coordinates from RCSB PDB 1BNA",
             license="CC0 (original render; PDB data is public domain)",
             source_url="https://www.rcsb.org/structure/1BNA",
             notes=("Render, 8 nm field of the same scene as dna-strands (hero strand through the origin). "
                    "Screw fitted from 1BNA: 35.4 deg/bp, 3.37 A rise. Focus = projected centre of the atoms view.")),
        dict(id="atoms", src="gen/atoms.jpg", label="Atoms",
             caption="Each ball is one atom: carbon (grey), oxygen (red), nitrogen (blue), phosphorus (gold).",
             width_m=1.6e-9, focus=_gmeta("dna_views.json", ["meta", "atoms", "focus"], [0.5, 0.5]),
             blend="normal", credit=GEN_CREDIT + "; coordinates from RCSB PDB 1BNA",
             license="CC0 (original render; PDB data is public domain)",
             source_url="https://www.rcsb.org/structure/1BNA",
             notes=("Render, 1.6 nm field of the same scene; spheres at van der Waals radii (C 1.70, N 1.55, O 1.52, "
                    "P 1.80 A). Centred on a front-facing sugar carbon; focus = that carbon (becomes the atom level).")),
        dict(id="atom", src="gen/atom.jpg", label="Carbon atom",
             caption="A carbon atom: six electrons in a fuzzy cloud around a nucleus 50,000x smaller.",
             width_m=6.0e-10, focus=[0.5, 0.5], blend="add", credit=GEN_CREDIT,
             license="CC0 (original render)", source_url="",
             notes=("Generated. Projected electron density of ground-state carbon (1s2 2s2 2p2; hydrogen-like orbitals "
                    "with Clementi-Raimondi effective charges), log tone-mapped so the glow fades at ~1.75 A (the vdW "
                    "radius of the carbon sphere in 'atoms'). 0.6 nm field. Nucleus drawn as a tiny bright point "
                    "(true size would be ~1/50,000 of the cloud). Black background for additive blending.")),
        dict(id="nucleus", src="gen/nucleus.jpg", label="Nucleus",
             caption="The carbon nucleus: 6 protons and 6 neutrons carrying 99.97% of the atom's mass.",
             width_m=1.2e-14, focus=_gmeta("quantum_meta.json", ["nucleus", "focus"], [0.5, 0.5]),
             blend="add", credit=GEN_CREDIT, license="CC0 (original render)", source_url="",
             notes=("Generated. Carbon-12: 12 nucleons (r = 0.86 fm) relaxed into a close-packed cluster, sphere "
                    "renderer + bloom; protons warm red, neutrons grey-blue. 12 fm field (nucleus ~5.5 fm across). "
                    "Focus = a front-facing proton.")),
        dict(id="proton", src="gen/proton.jpg", label="Proton",
             caption="Inside a proton: three quarks bound by gluons, the strongest force in nature.",
             width_m=3.6e-15, focus=[0.5, 0.5], blend="add", credit=GEN_CREDIT,
             license="CC0 (original render)", source_url="",
             notes=("Illustrative render (no real image exists). Fuzzy sphere of charge radius 0.84 fm, three valence "
                    "quarks in the three colour charges joined by a Y-shaped gluon flux tube, faint sea-quark pairs. "
                    "3.6 fm field.")),
    ]


def load_generated():
    out = []
    for g in generated_levels():
        p = os.path.join(HERE, g["src"])
        if not os.path.exists(p):
            print("missing", g["src"]); continue
        im = Image.open(p).convert("RGB")
        if max(im.size) > MAXS:
            s = MAXS / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        im.save(os.path.join(OUT, g["id"] + ".jpg"), quality=90, optimize=True)
        e = dict(id=g["id"], file=f"assets/levels/{g['id']}.jpg", px=[im.width, im.height], label=g["label"],
                 caption=g["caption"], width_m=g["width_m"], focus=[round(float(v), 4) for v in g["focus"]],
                 blend=g["blend"], credit=g["credit"], license=g["license"], source_url=g["source_url"],
                 notes=g["notes"])
        out.append(e)
    return out


def process(L):
    im = Image.open(os.path.join(HERE, L["src"]))
    im = im.convert("RGB")
    if L.get("crop"):
        im = im.crop(L["crop"])
    if L.get("mirror"):
        im = ImageOps.mirror(im)
    w_px = im.width  # physical width is defined on the cropped, pre-resize image
    width_m = L["width_from"](w_px)
    if L.get("post"):
        im = L["post"](im)
    if max(im.size) > MAXS:
        s = MAXS / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    dst = os.path.join(OUT, L["id"] + ".jpg")
    im.save(dst, quality=90, optimize=True)
    return dict(id=L["id"], file=f"assets/levels/{L['id']}.jpg", px=[im.width, im.height], label=L["label"],
                caption=L["caption"], width_m=float(f"{width_m:.4g}"), focus=L["focus"], blend=L["blend"],
                credit=L["credit"], license=L["license"], source_url=L["source_url"], notes=L["notes"])


def main():
    os.makedirs(OUT, exist_ok=True)
    entries = []
    for L in LEVELS:
        if not os.path.exists(os.path.join(HERE, L["src"])):
            print("missing", L["src"]); continue
        e = process(L); entries.append(e)
        print(f"{e['id']:14s} {e['px']} width_m={e['width_m']:.4g}")
    for g in load_generated():
        entries.append(g)
        print(f"{g['id']:14s} {g['px']} width_m={g['width_m']:.4g} (generated)")
    for e in entries:
        assert len(e["caption"]) <= 90, (e["id"], len(e["caption"]))
    # sanity: each next level must be smaller
    for a, b in zip(entries, entries[1:]):
        if not b["width_m"] < a["width_m"]:
            print("WARNING: width not decreasing", a["id"], b["id"])
        else:
            print(f"   {a['id']} -> {b['id']}: x{a['width_m'] / b['width_m']:.1f}")
    json.dump(entries, open(MAN, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("wrote", MAN)


if __name__ == "__main__":
    main()
