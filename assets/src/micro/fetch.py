"""Download Commons originals by file title: py fetch.py "File:X.jpg" outname"""
import sys, json, urllib.request, urllib.parse, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
UA = "ScaleZoomSite/0.1 (personal non-commercial project)"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"

def get(url, tries=5):
    for k in range(tries):
        try:
            ua = BROWSER_UA if "iiif.wellcomecollection.org" in url else UA
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and k < tries - 1:
                time.sleep(8 * (k + 1)); continue
            raise

def info(title):
    p = {"action": "query", "titles": title, "prop": "imageinfo",
         "iiprop": "url|size|extmetadata", "format": "json"}
    d = json.loads(get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(p)))
    pg = list(d["query"]["pages"].values())[0]
    return pg["imageinfo"][0]

if __name__ == "__main__":
    title, out = sys.argv[1], sys.argv[2]
    ii = info(title)
    md = ii["extmetadata"]
    import re
    for k in ["Artist", "Credit", "LicenseShortName", "ImageDescription", "DateTimeOriginal"]:
        v = re.sub(r"<[^>]+>", "", str(md.get(k, {}).get("value", "")))
        print(f"{k}: {v[:600]}")
    print("url:", ii["url"]); print("page:", ii["descriptionurl"]); print("size:", ii["width"], ii["height"])
    open(out, "wb").write(get(ii["url"]))
    print("saved", out)
