import json
import pathlib
import argparse


def compute_pagerank(link_graph_path, num_iterations=20, damping_factor=0.85):
    """
    Compute PageRank scores for all documents.

    Args:
        link_graph_path: Path to link_graph.json file
        num_iterations: Number of iterations for convergence (default: 20)
        damping_factor: Damping factor (default: 0.85)

    Returns:
        pagerank_scores: Dict[int, float] - PageRank score for each document
    """
    # Load the link graph (forward links: docid -> [target docids])
    with open(link_graph_path, 'r', encoding='utf-8') as f:
        link_graph = json.load(f)

    # Convert string keys to integers and collect all docids
    forward_links = {}
    all_docids = set()
    for src_str, targets in link_graph.items():
        src = int(src_str)
        forward_links[src] = targets
        all_docids.add(src)
        all_docids.update(targets)

    N = len(all_docids)
    print(f"[PageRank] Computing PageRank for {N} documents")

    # Initialize PageRank scores uniformly
    pagerank = {docid: 1.0 / N for docid in all_docids}

    # Build reverse link graph (incoming links)
    incoming_links = {docid: [] for docid in all_docids}
    for src, targets in forward_links.items():
        for target in targets:
            incoming_links[target].append(src)

    # Compute outgoing link counts
    outgoing_counts = {}
    for docid in all_docids:
        outgoing = forward_links.get(docid, [])
        outgoing_counts[docid] = len(outgoing) if outgoing else 0

    # PageRank iterative algorithm
    for iteration in range(num_iterations):
        new_pagerank = {}

        for docid in all_docids:
            # Base probability (random jump)
            rank = (1 - damping_factor) / N

            # Add contributions from incoming links
            for incoming_docid in incoming_links[docid]:
                out_count = outgoing_counts[incoming_docid]
                if out_count > 0:
                    rank += damping_factor * (pagerank[incoming_docid] / out_count)

            new_pagerank[docid] = rank

        # Normalize to maintain probability distribution
        total = sum(new_pagerank.values())
        if total > 0:
            for docid in new_pagerank:
                new_pagerank[docid] /= total

        pagerank = new_pagerank
        print(f"[PageRank] Iteration {iteration + 1}/{num_iterations} completed")

    return pagerank


def save_pagerank_scores(out_dir, pagerank_scores):
    """Save PageRank scores to JSON file."""
    pr_data = {
        "pagerank_scores": {str(k): v for k, v in pagerank_scores.items()}
    }

    out_path = pathlib.Path(out_dir) / "pagerank_scores.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(pr_data, f)

    print(f"[PageRank] Saved scores to {out_path}")
    print(f"[PageRank] Total documents with scores: {len(pagerank_scores)}")

    # Print top 10 PageRank scores
    top_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\n[PageRank] Top 10 PageRank Scores:")
    for docid, score in top_pagerank:
        print(f"  DocID {docid}: {score:.8f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute PageRank scores")
    ap.add_argument("--index_dir", required=True, help="Directory containing link_graph.json")
    ap.add_argument("--iterations", type=int, default=20, help="Number of PageRank iterations (default: 20)")
    ap.add_argument("--damping", type=float, default=0.85, help="Damping factor (default: 0.85)")
    args = ap.parse_args()

    index_dir = pathlib.Path(args.index_dir)
    link_graph_path = index_dir / "link_graph.json"

    if not link_graph_path.exists():
        print(f"[ERROR] Link graph not found: {link_graph_path}")
        print("Please run the indexer first to generate link_graph.json")
        exit(1)

    print(f"[PageRank] Loading link graph from {link_graph_path}")
    pagerank_scores = compute_pagerank(link_graph_path, num_iterations=args.iterations, damping_factor=args.damping)

    save_pagerank_scores(index_dir, pagerank_scores)
    print("\n[OK] PageRank computation complete!")