"""Fetch small int8-calibration image sets from Wikimedia Commons into
convert/calib_images/<category>/ (gitignored — rerun to regenerate).

Usage: python convert/fetch_calib.py [category ...]   # default: all
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "calib_images")
UA = {"User-Agent": "executorch-convert-calib/1.0 (model calibration; contact: none)"}

CATEGORIES = {
    # category -> Commons search query (filetype:bitmap keeps out SVG/maps)
    "portrait": "filetype:bitmap portrait photograph person standing",
    "street": "filetype:bitmap city street traffic photograph",
    "general": "filetype:bitmap outdoor landscape photograph",
}
N = 10
WIDTH = 1024


def fetch(category, query):
    d = os.path.join(OUT, category)
    os.makedirs(d, exist_ok=True)
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": query, "gsrnamespace": 6, "gsrlimit": N * 2,
        "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": WIDTH,
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    pages = json.load(urllib.request.urlopen(req)).get("query", {}).get("pages", {})
    got = 0
    for p in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        if got >= N:
            break
        info = (p.get("imageinfo") or [{}])[0]
        if info.get("mime") not in ("image/jpeg", "image/png"):
            continue
        thumb = info.get("thumburl") or info.get("url")
        if not thumb:
            continue
        thumb = thumb.split("?", 1)[0]  # utm params trip the robot policy
        ext = ".png" if thumb.lower().endswith(".png") else ".jpg"
        dst = os.path.join(d, f"{category}_{got:02d}{ext}")
        for attempt in range(3):
            try:
                r = urllib.request.Request(thumb, headers=UA)
                with urllib.request.urlopen(r, timeout=30) as resp, open(dst, "wb") as f:
                    f.write(resp.read())
                got += 1
                break
            except Exception as e:
                print(f"  retry {thumb}: {e}")
                time.sleep(10 * (attempt + 1))
        time.sleep(3)  # stay under Wikimedia rate limits
    print(f"{category}: {got} images -> {d}")


def fill_picsum(category):
    """Top up a category to N with Lorem Picsum (diverse real photos, no rate
    limit). Not domain-targeted — fine for 'general', weaker for 'street'."""
    d = os.path.join(OUT, category)
    os.makedirs(d, exist_ok=True)
    have = len(os.listdir(d))
    for i in range(have, N):
        url = f"https://picsum.photos/seed/etcal-{category}-{i}/1024/1024"
        dst = os.path.join(d, f"{category}_picsum_{i:02d}.jpg")
        try:
            r = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(r, timeout=30) as resp, open(dst, "wb") as f:
                f.write(resp.read())
        except Exception as e:
            print(f"  picsum skip: {e}")
    print(f"{category}: topped up to {len(os.listdir(d))}")


if __name__ == "__main__":
    cats = sys.argv[1:] or list(CATEGORIES)
    for c in cats:
        fetch(c, CATEGORIES[c])
        fill_picsum(c)
