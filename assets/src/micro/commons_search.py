"""Search Wikimedia Commons files and print title / size / license / url.
Usage: py commons_search.py "search terms" [limit]
"""
import sys, json, re, urllib.request, urllib.parse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

UA = "ScaleZoomSite/0.1 (personal non-commercial project)"


def search(q, limit=30):
    params = {
        "action": "query", "generator": "search", "gsrsearch": q + " filetype:bitmap",
        "gsrnamespace": "6", "gsrlimit": str(limit), "prop": "imageinfo",
        "iiprop": "url|size|extmetadata", "format": "json",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    pages = data.get("query", {}).get("pages", {})
    out = []
    for p in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        ii = (p.get("imageinfo") or [{}])[0]
        md = ii.get("extmetadata", {})
        def g(k):
            v = md.get(k, {}).get("value", "")
            return re.sub(r"<[^>]+>", "", str(v)).strip()
        out.append(dict(title=p["title"], w=ii.get("width"), h=ii.get("height"),
                        url=ii.get("url"), page=ii.get("descriptionurl"),
                        lic=g("LicenseShortName"), artist=g("Artist")[:80],
                        desc=g("ImageDescription")[:300]))
    return out


if __name__ == "__main__":
    q = sys.argv[1]
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    for r in search(q, lim):
        print(f"{r['w']}x{r['h']} | {r['lic']} | {r['title']}")
        print(f"    by: {r['artist'][:40]} | {r['desc'][:160]}")
