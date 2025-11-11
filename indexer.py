#!/usr/bin/env python3
import os, sys, json, gzip, math, time, argparse, pathlib, collections
from typing import Dict
from html_utils import parse_html_zones, extract_anchors
from text_utils import tokenize, maybe_stemmer, make_ngrams
from storage import PartialMerger, write_jsonl, read_jsonl


def delta_encode(int_list):
    if not int_list:
        return []
    out = [int_list[0]]
    for i in range(1, len(int_list)):
        out.append(int_list[i] - int_list[i - 1])
    return out


def flush_partial(out_dir: pathlib.Path, partial: Dict[str, dict], idx: int):
    """Write one sorted partial file: each line is 'term\\t{df, postings}' as JSON, gzipped."""
    path = out_dir / "partials" / f"part_{idx:03d}.gz"
    (out_dir / "partials").mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for term in sorted(partial.keys()):  # terms sorted lexicographically
            node = partial[term]
            df = len(node["dfset"])
            # (optional but tidy) keep postings sorted by docID
            node["postings"].sort(key=lambda p: p["d"])
            rec = {"df": df, "postings": node["postings"]}
            f.write(f"{term}\t{json.dumps(rec, separators=(',',':'))}\n")
    print(f"[FLUSH] wrote {path.name} terms={len(partial)}")


def build_index(args):
    t0 = time.time()
    corpus_dir = pathlib.Path(args.corpus_dir)
    out_dir = pathlib.Path(args.out_dir)
    (out_dir / "partials").mkdir(parents=True, exist_ok=True)

    # URL <-> docID maps
    docid_by_url: Dict[str, int] = {}
    url_by_docid: Dict[int, str] = {}
    docstats: Dict[int, dict] = {}

    # Temp store for anchor extraction (src, href, tokens)
    anchors_tmp = out_dir / "anchors_tmp.jsonl"
    if anchors_tmp.exists():
        anchors_tmp.unlink()

    # In-memory partial inverted index:
    # term -> {"dfset": set(docids), "postings": [ {d, tf, pos, zones{t,h,b}} ]}
    partial: Dict[str, dict] = {}
    partial_posting_count = 0
    partial_idx = 0

    # Stemmer (optional)
    stem = maybe_stemmer(disable=bool(args.no_stem))

    # Enumerate JSON pages
    json_files = list(corpus_dir.rglob("*.json"))
    json_files.sort()
    total_docs = 0

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8", errors="ignore") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[WARN] Skipping unreadable JSON: {jf} ({e})", file=sys.stderr)
            continue

        url = obj.get("url", "").split("#")[0]
        html = obj.get("content", "")
        if not url:
            continue

        docid = len(docid_by_url)
        docid_by_url[url] = docid
        url_by_docid[docid] = url
        total_docs += 1

        # Parse zones and anchors
        zones = parse_html_zones(html)  # {"title": str, "headings": [str], "bold":[str], "body": str}
        anchors = extract_anchors(html, base_url=url)  # list of (href_abs, anchor_text)

        # Tokenize + (optional) stem per zone; build one content stream for positions
        content_positions = []
        term_tf = collections.Counter()
        term_positions = collections.defaultdict(list)
        zone_flags = collections.defaultdict(lambda: {"t": 0, "h": 0, "b": 0})

        def add_zone(text: str, flag_key: str):
            tokens = tokenize(text)
            if stem:
                tokens = [stem(t) for t in tokens]
            base = len(content_positions)
            content_positions.extend(tokens)
            for i, tok in enumerate(tokens):
                term_tf[tok] += 1
                term_positions[tok].append(base + i)
                if flag_key in ("t", "h", "b"):
                    zone_flags[tok][flag_key] = 1

        add_zone(zones.get("title", ""), "t")
        add_zone(" ".join(zones.get("headings", [])), "h")
        add_zone(" ".join(zones.get("bold", [])), "b")
        add_zone(zones.get("body", ""), "b0")  # 'b0' only for position continuity; not a scoring flag

        # n-grams (start-index positions)
        uni_tokens = content_positions
        if args.enable_bigrams:
            for i in range(len(uni_tokens) - 1):
                gram = uni_tokens[i] + "_" + uni_tokens[i + 1]
                term_tf[gram] += 1
                term_positions[gram].append(i)
        if args.enable_trigrams:
            for i in range(len(uni_tokens) - 2):
                gram = uni_tokens[i] + "_" + uni_tokens[i + 1] + "_" + uni_tokens[i + 2]
                term_tf[gram] += 1
                term_positions[gram].append(i)

        # Anchor extraction (write temp; attribution happens after crawling)
        for href, atext in anchors:
            toks = tokenize(atext)
            if stem:
                toks = [stem(t) for t in toks]
            if toks:
                write_jsonl(anchors_tmp, {"src": url, "href": href, "tokens": toks})

        # Add postings into the in-memory partial map
        doc_len = len(uni_tokens)
        for term, tf in term_tf.items():
            pos = term_positions[term]
            pos.sort()
            dpos = delta_encode(pos)
            z = zone_flags.get(term, {"t": 0, "h": 0, "b": 0})
            entry = {
                "d": docid,
                "tf": tf,
                "pos": dpos,
                "zones": {"t": z.get("t", 0), "h": z.get("h", 0), "b": z.get("b", 0)},
            }
            node = partial.get(term)
            if node is None:
                partial[term] = {"dfset": {docid}, "postings": [entry]}
            else:
                node["dfset"].add(docid)
                node["postings"].append(entry)
            partial_posting_count += 1

        # Doc-level stats (norm computed later)
        docstats[docid] = {
            "url": url,
            "length": doc_len,
            "title_terms": sum(zone_flags[t]["t"] for t in zone_flags),
            "heading_terms": sum(zone_flags[t]["h"] for t in zone_flags),
            "bold_terms": sum(zone_flags[t]["b"] for t in zone_flags),
        }

        # Flush partial if too big
        if partial_posting_count >= args.partial_flush_every:
            flush_partial(out_dir, partial, partial_idx)
            partial = {}
            partial_posting_count = 0
            partial_idx += 1

    # Final flush if anything remains
    if partial_posting_count > 0:
        flush_partial(out_dir, partial, partial_idx)
        partial = {}
        partial_idx += 1

    # -------- Step 1 fix: build a fast anchor index --------
    # href_anchor_counts: href_url -> Counter(term)
    href_anchor_counts = collections.defaultdict(collections.Counter)
    if anchors_tmp.exists():
        for record in read_jsonl(anchors_tmp):
            href = (record.get("href") or "").split("#")[0]
            toks = record.get("tokens", [])
            if href and toks:
                href_anchor_counts[href].update(toks)

    # Build term -> {docid: anchor_tf} so we can O(1) lookup during merge
    anchor_index = collections.defaultdict(collections.Counter)
    if href_anchor_counts:
        for href, cnts in href_anchor_counts.items():
            did = docid_by_url.get(href)
            if did is None:
                continue
            for term_, c in cnts.items():
                anchor_index[term_][did] += c
    # -------------------------------------------------------

    # Merge partials into final index, injecting anchor counts
    merger = PartialMerger(out_dir / "partials")
    merged_path = out_dir / "final_index.gz"
    lexicon_path = out_dir / "lexicon.json"

    with gzip.open(merged_path, "wt", encoding="utf-8") as fout:
        lexicon = {}
        terms_processed = 0
        for term, postings in merger.merge():
            # ensure 'anc' field exists and apply O(1) anchor lookups
            term_anc = anchor_index.get(term)
            if term_anc:
                for p in postings:
                    p["anc"] = p.get("anc", 0) + term_anc.get(p["d"], 0)
            else:
                for p in postings:
                    if "anc" not in p:
                        p["anc"] = 0

            # compute df and write line
            df = len({p["d"] for p in postings})
            lexicon[term] = {
                "df": df,
                "idf": 0.0,  # filled after we know N
                "postings_count": len(postings),
            }
            fout.write(f"{term}\t{json.dumps({'df': df, 'postings': postings}, separators=(',',':'))}\n")

            terms_processed += 1
            if terms_processed % 50000 == 0:
                print(f"[MERGE] terms merged: {terms_processed}")

    # Compute IDF for all terms
    N = total_docs if total_docs > 0 else 1
    for term, meta in lexicon.items():
        df = max(1, meta["df"])
        meta["idf"] = math.log(N / df + 1e-12)

    # Save lexicon
    with open(lexicon_path, "w", encoding="utf-8") as f:
        json.dump(lexicon, f)

    # Compute doc norms: sqrt(sum( (log1p(tf)*idf)^2 )) over all postings
    print("[NORMS] computing document norms...")
    norms = collections.defaultdict(float)
    line_ct = 0
    with gzip.open(merged_path, "rt", encoding="utf-8") as fin:
        for line in fin:
            line_ct += 1
            if line_ct % 100000 == 0:
                print(f"[NORMS] processed {line_ct} terms")
            try:
                term, payload = line.rstrip("\n").split("\t", 1)
            except ValueError:
                continue
            node = json.loads(payload)
            idf = lexicon.get(term, {}).get("idf", 0.0)
            for p in node.get("postings", []):
                w = math.log1p(p.get("tf", 0)) * idf
                norms[p.get("d", -1)] += w * w

    for did in range(total_docs):
        stats = docstats.setdefault(did, {
            "url": url_by_docid.get(did, ""),
            "length": 0,
            "title_terms": 0,
            "heading_terms": 0,
            "bold_terms": 0,
        })
        stats["norm"] = math.sqrt(norms.get(did, 0.0))

    with open(out_dir / "docstats.json", "w", encoding="utf-8") as f:
        json.dump(docstats, f)

    manifest = {
        "args": vars(args),
        "total_docs": total_docs,
        "partials_written": partial_idx,
        "elapsed_sec": time.time() - t0,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Indexed {total_docs} docs → {out_dir}")
    print(f"  Partials: {partial_idx}")
    print(f"  Time: {manifest['elapsed_sec']:.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ICS Search Indexer (extra credit features + fast anchors)")
    ap.add_argument("--corpus_dir", required=True, help="Path to corpus root directory")
    ap.add_argument("--out_dir", required=True, help="Output directory for index files")
    ap.add_argument("--partial_flush_every", type=int, default=150000, help="Flush partial after this many postings")
    ap.add_argument("--enable_bigrams", action="store_true")
    ap.add_argument("--enable_trigrams", action="store_true")
    ap.add_argument("--no_stem", action="store_true", help="Disable stemming")
    args = ap.parse_args()
    build_index(args)
