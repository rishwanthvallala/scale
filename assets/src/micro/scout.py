"""Scouting helper: search Wellcome Collection or Wikimedia Commons and build a
labelled contact sheet of thumbnails for visual review.

py scout.py wc "query" sheet.jpg [n] [license-filter e.g. cc-by,cc-0,pdm]
py scout.py cm "query" sheet.jpg [n]
"""
import sys, json, re, io, urllib.request, urllib.parse, urllib.error
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
UA = "ScaleZoomSite/0.1 (personal non-commercial project)"
# Wellcome's IIIF CDN rejects non-browser user agents; used only for their openly licensed images.
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"


def get(url, timeout=60, tries=4):
    import time
    for k in range(tries):
        try:
            ua = BROWSER_UA if "iiif.wellcomecollection.org" in url else UA
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and k < tries - 1:
                time.sleep(5 * (k + 1))
                continue
            raise


def wc(q, n, lic_ok=None):
    url = ("https://api.wellcomecollection.org/catalogue/v2/images?" +
           urllib.parse.urlencode({"query": q, "pageSize": 100}))
    d = json.loads(get(url))
    out = []
    for r in d["results"]:
        loc = r["locations"][0]
        lic = loc.get("license", {}).get("id")
        if lic_ok and lic not in lic_ok:
            continue
        base = loc["url"].rsplit("/info.json", 1)[0]
        out.append(dict(id=r["id"], title=r.get("source", {}).get("title", ""),
                        lic=lic, thumb=base + "/full/400,/0/default.jpg",
                        base=base, ar=r.get("aspectRatio")))
        if len(out) >= n:
            break
    return out


def cm(q, n):
    if q.startswith("Category:"):
        params = {"action": "query", "generator": "categorymembers", "gcmtitle": q,
                  "gcmtype": "file", "gcmlimit": str(n), "prop": "imageinfo",
                  "iiprop": "url|size|extmetadata", "iiurlwidth": "300", "format": "json"}
        d = json.loads(get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)))
        sub = json.loads(get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(
            {"action": "query", "list": "categorymembers", "cmtitle": q, "cmtype": "subcat",
             "cmlimit": "50", "format": "json"})))
        for c in sub["query"]["categorymembers"]:
            print("SUBCAT", c["title"])
        return _cm_parse(d)
    params = {"action": "query", "generator": "search", "gsrsearch": q + " filetype:bitmap",
              "gsrnamespace": "6", "gsrlimit": str(n), "prop": "imageinfo",
              "iiprop": "url|size|extmetadata", "iiurlwidth": "300", "format": "json"}
    d = json.loads(get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)))
    return _cm_parse(d)


def _cm_parse(d):
    out = []
    for p in sorted(d.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0)):
        ii = (p.get("imageinfo") or [{}])[0]
        md = ii.get("extmetadata", {})
        lic = re.sub(r"<[^>]+>", "", md.get("LicenseShortName", {}).get("value", ""))
        out.append(dict(id=p["title"], title=p["title"], lic=lic, thumb=ii.get("thumburl"),
                        base=ii.get("url"), ar=(ii.get("width") or 1) / (ii.get("height") or 1),
                        size=f"{ii.get('width')}x{ii.get('height')}"))
    return out


def _fetch(u):
    try:
        return get(u, timeout=40)
    except Exception as e:
        return e


def sheet(items, path, cols=5, tw=300, th=230):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(3) as ex:
        blobs = list(ex.map(_fetch, [it["thumb"] for it in items]))
    rows = (len(items) + cols - 1) // cols
    S = Image.new("RGB", (cols * tw, rows * th), (30, 30, 30))
    dr = ImageDraw.Draw(S)
    for i, it in enumerate(items):
        x, y = (i % cols) * tw, (i // cols) * th
        try:
            b = blobs[i]
            if isinstance(b, Exception):
                raise b
            im = Image.open(io.BytesIO(b)).convert("RGB")
            im.thumbnail((tw - 6, th - 30))
            S.paste(im, (x + 3, y + 3))
        except Exception as e:
            dr.text((x + 5, y + 50), "ERR " + str(e)[:40], fill=(255, 80, 80))
        dr.rectangle([x, y + th - 26, x + tw, y + th], fill=(0, 0, 0))
        dr.text((x + 4, y + th - 24), f"[{i}] {it['lic']} {it.get('size','')}", fill=(255, 255, 0))
        dr.text((x + 4, y + th - 12), it["title"][:48], fill=(220, 220, 220))
    if items:
        S.save(path, quality=85)


if __name__ == "__main__":
    src, q, path = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    lic_ok = sys.argv[5].split(",") if len(sys.argv) > 5 else None
    items = wc(q, n, lic_ok) if src == "wc" else cm(q, n)
    for i, it in enumerate(items):
        print(i, "|", it["id"], "|", it["lic"], "|", it.get("size", it.get("ar")), "|", it["title"][:90])
        print("     ", it["base"])
    sheet(items, path)
