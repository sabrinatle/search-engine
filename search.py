"""
CS 121 Search Engine - Search Component
Implements TF-IDF ranking with zone-based importance weighting.
Designed to work with disk-based index (no full loading into memory).
"""
import json
import gzip
import math
import time
import sys
import argparse
from collections import defaultdict, Counter
from pathlib import Path
from text_utils import tokenize, maybe_stemmer


class SearchEngine:
    """
    Search engine that reads index from disk and ranks documents using TF-IDF.
    Does NOT load the entire index into memory - reads postings on demand.
    """

    def __init__(self, index_dir: str, disable_stem: bool = False):
        self.index_dir = Path(index_dir)
        self.stem = maybe_stemmer(disable=disable_stem)

        # Load metadata (these are small enough to keep in memory)
        print(f"[LOAD] Loading metadata from {index_dir}...", file=sys.stderr)

        with open(self.index_dir / "lexicon.json", "r") as f:
            self.lexicon = json.load(f)

        with open(self.index_dir / "docstats.json", "r") as f:
            self.docstats = json.load(f)
            # Convert string keys to int for docstats
            self.docstats = {int(k): v for k, v in self.docstats.items()}

        with open(self.index_dir / "manifest.json", "r") as f:
            self.manifest = json.load(f)

        self.total_docs = self.manifest["total_docs"]
        self.index_path = self.index_dir / "final_index.gz"

        # EXTRA CREDIT: Load HITS scores
        hits_path = self.index_dir / "hits_scores.json"
        if hits_path.exists():
            with open(hits_path, "r") as f:
                hits_data = json.load(f)
                self.auth_scores = {int(k): v for k, v in hits_data.get("auth_scores", {}).items()}
                self.hub_scores = {int(k): v for k, v in hits_data.get("hub_scores", {}).items()}
            # Compute max HITS authority for normalization
            self.max_auth = max(self.auth_scores.values()) if self.auth_scores else 1.0
            print(f"[HITS] Loaded HITS scores for {len(self.auth_scores)} documents", file=sys.stderr)
        else:
            self.auth_scores = {}
            self.hub_scores = {}
            self.max_auth = 1.0
            print(f"[WARN] No HITS scores found. Run hits.py to generate them for better ranking.", file=sys.stderr)

        # EXTRA CREDIT: Load PageRank scores 
        pagerank_path = self.index_dir / "pagerank_scores.json"
        if pagerank_path.exists():
            with open(pagerank_path, "r") as f:
                pr_data = json.load(f)
                self.pagerank_scores = {int(k): v for k, v in pr_data.get("pagerank_scores", {}).items()}
            # Compute max PageRank for normalization
            self.max_pagerank = max(self.pagerank_scores.values()) if self.pagerank_scores else 1.0
            print(f"[PageRank] Loaded PageRank scores for {len(self.pagerank_scores)} documents", file=sys.stderr)
        else:
            self.pagerank_scores = {}
            self.max_pagerank = 1.0
            print(f"[WARN] No PageRank scores found. Run pagerank.py to generate them for better ranking.", file=sys.stderr)

        # Load chunk manifest if it exists (for fast lookups)
        chunks_manifest_path = self.index_dir / "chunks_manifest.json"
        if chunks_manifest_path.exists():
            with open(chunks_manifest_path, "r") as f:
                self.chunks_manifest = json.load(f)
            self.use_chunks = True
            self.chunks_dir = self.index_dir / "chunks"
            print(f"[OK] Loaded index with {self.total_docs} docs, {len(self.lexicon)} terms", file=sys.stderr)
            print(f"[INFO] Using {len(self.chunks_manifest)} chunked index files for fast lookup", file=sys.stderr)
        else:
            self.use_chunks = False
            print(f"[OK] Loaded index with {self.total_docs} docs, {len(self.lexicon)} terms", file=sys.stderr)
            print(f"[WARN] No chunks found - using slow linear search. Run split_index.py first!", file=sys.stderr)

    def _read_postings(self, term: str):
        """
        Read postings for a specific term from disk.
        Uses chunked index with binary search if available, otherwise linear search.
        """
        # Quick check: is term even in the index?
        if term not in self.lexicon:
            return None

        if self.use_chunks:
            return self._read_postings_from_chunk(term)
        else:
            return self._read_postings_linear(term)

    def _read_postings_from_chunk(self, term: str):
        """Read postings using chunked index with linear search (fast for small chunks)."""
        # Find which chunk contains this term (use same hash as splitter)
        chunk_id = str(sum(ord(c) for c in term) % len(self.chunks_manifest))

        if chunk_id not in self.chunks_manifest:
            return None

        chunk_info = self.chunks_manifest[chunk_id]
        chunk_path = self.chunks_dir / chunk_info["file"]

        # Linear search within the chunk file (efficient for ~180k terms per chunk)
        try:
            with open(chunk_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        t, payload = line.rstrip("\n").split("\t", 1)
                        if t == term:
                            node = json.loads(payload)
                            return node.get("postings", [])
                    except Exception:
                        continue
        except Exception:
            pass

        return None

    def _read_postings_linear(self, term: str):
        """Fallback: linear search through the gzipped index file."""
        with gzip.open(self.index_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    t, payload = line.rstrip("\n").split("\t", 1)
                    if t == term:
                        node = json.loads(payload)
                        return node.get("postings", [])
                except Exception:
                    continue
        return None

    def _process_query(self, query: str):
        """Tokenize and stem query terms."""
        tokens = tokenize(query)
        if self.stem:
            tokens = [self.stem(t) for t in tokens]
        return tokens

    def _calculate_scores(self, query_terms):
        """
        Calculate TF-IDF scores for all documents matching query terms.
        Incorporates zone-based importance (title, headings, bold).

        Returns: dict of docid -> score
        """
        # Weights for zone importance
        ZONE_WEIGHTS = {
            "title": 3.0,      # Title is most important
            "heading": 2.0,    # Headings are important
            "bold": 1.5,       # Bold text is somewhat important
            "anchor": 2.5,     # Anchor text is very important
        }

        scores = defaultdict(float)
        query_term_counts = Counter(query_terms)

        # Process each unique query term
        for term in set(query_terms):
            # Get IDF from lexicon
            term_meta = self.lexicon.get(term)
            if not term_meta:
                continue  # Term not in index

            idf = term_meta.get("idf", 0.0)

            # Read postings from disk (NOT loading entire index)
            postings = self._read_postings(term)
            if not postings:
                continue

            # Calculate contribution to each document
            for posting in postings:
                docid = posting.get("d", -1)
                if docid < 0:
                    continue

                tf = posting.get("tf", 0)
                if tf == 0:
                    continue

                # Base TF-IDF score (log-normalized TF)
                base_score = math.log1p(tf) * idf

                # Zone boost: check if term appears in important zones
                zones = posting.get("zones", {})
                zone_boost = 1.0

                if zones.get("t", 0):  # Title
                    zone_boost += ZONE_WEIGHTS["title"]
                if zones.get("h", 0):  # Headings
                    zone_boost += ZONE_WEIGHTS["heading"]
                if zones.get("b", 0):  # Bold
                    zone_boost += ZONE_WEIGHTS["bold"]

                # Anchor text boost
                anc_count = posting.get("anc", 0)
                if anc_count > 0:
                    zone_boost += ZONE_WEIGHTS["anchor"] * math.log1p(anc_count)

                # Apply zone boost
                weighted_score = base_score * zone_boost

                # Normalize by document length (cosine normalization)
                doc_norm = self.docstats.get(docid, {}).get("norm", 1.0)
                if doc_norm > 0:
                    weighted_score /= doc_norm

                scores[docid] += weighted_score

        return scores

    def search(self, query: str, top_k: int = 10):
        """
        Search for query and return top-k ranked results.
        Integrates HITS authority and PageRank scores if available.

        Returns: list of (url, score) tuples
        """
        if not query or not query.strip():
            return []

        start_time = time.time()

        # Process query
        query_terms = self._process_query(query)
        if not query_terms:
            return []

        # Calculate TF-IDF scores
        scores = self._calculate_scores(query_terms)

        # Apply link analysis boosts if available
        # Combine TF-IDF with HITS and PageRank
        # Strategy: Normalize each component to 0-1, then combine with weights
        TFIDF_WEIGHT = 0.70
        HITS_WEIGHT = 0.15
        PAGERANK_WEIGHT = 0.15

        # Normalize TF-IDF scores to 0-1 range
        if scores:
            max_tfidf = max(scores.values())
            if max_tfidf > 0:
                for docid in scores:
                    tfidf_score = scores[docid] / max_tfidf  # 0-1 range

                    # Start with normalized TF-IDF
                    combined_score = tfidf_score * TFIDF_WEIGHT

                    # EXTRA CREDIT: Add HITS authority if available
                    if self.auth_scores:
                        auth_score = self.auth_scores.get(docid, 0.0)
                        # Normalize HITS to 0-1 range by dividing by max
                        auth_score_normalized = auth_score / self.max_auth if self.max_auth > 0 else 0.0
                        combined_score += auth_score_normalized * HITS_WEIGHT

                    # EXTRA CREDIT: Add PageRank if available 
                    if self.pagerank_scores:
                        pr_score = self.pagerank_scores.get(docid, 0.0)
                        # Normalize PageRank to 0-1 range by dividing by max
                        pr_score_normalized = pr_score / self.max_pagerank if self.max_pagerank > 0 else 0.0
                        combined_score += pr_score_normalized * PAGERANK_WEIGHT

                    scores[docid] = combined_score

        # Sort by score (descending)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Get top-k results with URLs
        results = []
        for docid, score in ranked[:top_k]:
            url = self.docstats.get(docid, {}).get("url", "")
            if url:
                results.append((url, score))

        elapsed = (time.time() - start_time) * 1000  # Convert to ms

        return results, elapsed

    def interactive_search(self):
        """Interactive console search interface."""
        print("=" * 70)
        print("ICS Search Engine - Interactive Mode")
        print("=" * 70)
        print(f"Index loaded: {self.total_docs} documents, {len(self.lexicon)} terms")
        print("Type your query and press Enter. Type 'quit' or 'exit' to stop.")
        print("=" * 70)
        print()

        while True:
            try:
                query = input("Search> ").strip()

                if not query:
                    continue

                if query.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break

                # Perform search
                results, elapsed = self.search(query, top_k=10)

                # Display results
                print()
                if not results:
                    print("No results found.")
                else:
                    print(f"Found {len(results)} results (search took {elapsed:.1f}ms)")
                    print("-" * 70)
                    for i, (url, score) in enumerate(results, 1):
                        print(f"{i}. {url}")
                        print(f"   Score: {score:.4f}")

                print()

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)


def batch_search(engine: SearchEngine, queries_file: str, output_file: str = None):
    """
    Process a batch of queries from a file.
    Each line in queries_file should be one query.
    """
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    results_all = []
    total_time = 0

    print(f"Processing {len(queries)} queries...", file=sys.stderr)

    for i, query in enumerate(queries, 1):
        results, elapsed = engine.search(query, top_k=10)
        total_time += elapsed

        results_all.append({
            "query": query,
            "results": [{"url": url, "score": score} for url, score in results],
            "elapsed_ms": elapsed
        })

        print(f"[{i}/{len(queries)}] '{query}' -> {len(results)} results ({elapsed:.1f}ms)", file=sys.stderr)

    avg_time = total_time / len(queries) if queries else 0
    print(f"\nAverage query time: {avg_time:.1f}ms", file=sys.stderr)

    # Save results if output file specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_all, f, indent=2)
        print(f"Results saved to {output_file}", file=sys.stderr)

    return results_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS 121 Search Engine - Search Interface")
    parser.add_argument("--index_dir", required=True, help="Directory containing index files")
    parser.add_argument("--no_stem", action="store_true", help="Disable stemming (must match indexer)")
    parser.add_argument("--batch", help="Batch mode: process queries from file")
    parser.add_argument("--output", help="Output file for batch results (JSON)")
    parser.add_argument("--query", help="Single query mode (non-interactive)")

    args = parser.parse_args()

    # Initialize search engine
    engine = SearchEngine(args.index_dir, disable_stem=args.no_stem)

    if args.batch:
        # Batch mode
        batch_search(engine, args.batch, args.output)
    elif args.query:
        # Single query mode
        results, elapsed = engine.search(args.query, top_k=10)
        print(f"Query: {args.query}")
        print(f"Time: {elapsed:.1f}ms")
        print(f"Results: {len(results)}")
        print()
        for i, (url, score) in enumerate(results, 1):
            print(f"{i}. {url} (score: {score:.4f})")
    else:
        # Interactive mode
        engine.interactive_search()