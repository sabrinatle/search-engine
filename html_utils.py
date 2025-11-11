from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse 

def parse_html_zones(html: str):
    zones = {"title": "", "headings": [], "bold": [], "body": ""}
    if not html:
        return zones
    try:
        soup = BeautifulSoup(html, "html5lib")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        zones["title"] = soup.title.string
    for tag in soup.find_all(["h1", "h2", "h3"]):
        zones["headings"].append(tag.get_text(" ", strip=True))
    for tag in soup.find_all(["b", "strong"]):
        zones["bold"].append(tag.get_text(" ", strip=True))
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    zones["body"] = soup.get_text(" ", strip=True)
    return zones


def extract_anchors(html: str, base_url: str):
    """Return list of (href_abs, anchor_text). Robust to malformed hrefs/base URL."""
    out = []
    if not html:
        return out
    try:
        soup = BeautifulSoup(html, "html5lib")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # Normalize base URL; if it's malformed (e.g., [YOUR_IP]), we will still try to join hrefs
    # but we won't let a ValueError crash the indexer.
    def safe_join(base_url: str, href: str):
        if not href:
            return None
        href = href.strip()
        low = href.lower()

        # Drop non-document schemes early
        if low.startswith(("javascript:", "mailto:", "data:", "tel:")):
            return None

        # Try to join; Python 3.13 may raise on bracketed invalid hosts
        try:
            abs_url = urljoin(base_url, href)
        except Exception:
            # Fallback: if href is absolute, keep it; otherwise skip
            abs_url = href

        # Validate resulting URL
        try:
            pr = urlparse(abs_url)
        except Exception:
            return None

        # Keep only http/https
        if pr.scheme not in ("http", "https"):
            return None

        # Python 3.13 can still choke on illegal bracketed hosts; guard by re-parsing host
        # If netloc contains brackets, ensure it's valid IPv6 or a hostname without brackets
        netloc = pr.netloc
        if "[" in netloc or "]" in netloc:
            # If it's bracketed but not a valid IPv6, drop it
            # (Avoid importing ipaddress and raising again; just skip)
            return None

        if not pr.netloc:
            return None

        return abs_url

    for a in soup.find_all("a"):
        href = a.get("href")
        abs_href = safe_join(base_url, href)
        if not abs_href:
            continue
        text = a.get_text(" ", strip=True) or ""
        out.append((abs_href, text))
    return out

