import re
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str):
    if not text:
        return []
    return [t.lower() for t in TOKEN_RE.findall(text)]

def maybe_stemmer(disable: bool = False):
    if disable:
        return None
    try:
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        return ps.stem
    except Exception:
        return None

def make_ngrams(tokens, n):
    if n <= 1:
        return []
    return ["_".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
