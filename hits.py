import json
import pathlib
import argparse
import collections


def compute_hits(link_graph_path, num_iterations=10):
    """
    Compute HITS (Hyperlink-Induced Topic Search) hub and authority scores.

    Args:
        link_graph_path: Path to link_graph.json file
        num_iterations: Number of iterations for convergence (default: 10)

    Returns:
        hub_scores: Dict[int, float] - hub score for each document
        auth_scores: Dict[int, float] - authority score for each document
    """
    # Load the link graph (forward links: docid -> [target docids])
    with open(link_graph_path, 'r', encoding='utf-8') as f:
        link_graph = json.load(f)

    # Convert string keys to integers
    forward_links = {}
    all_docids = set()
    for src_str, targets in link_graph.items():
        src = int(src_str)
        forward_links[src] = targets
        all_docids.add(src)
        all_docids.update(targets)

    # Build reverse link graph (backward links: docid -> [source docids that link to it])
    backward_links = collections.defaultdict(list)
    for src, targets in forward_links.items():
        for target in targets:
            backward_links[target].append(src)

    # Initialize hub and authority scores to 1.0
    hub_scores = {docid: 1.0 for docid in all_docids}
    auth_scores = {docid: 1.0 for docid in all_docids}

    # HITS iterative algorithm
    for iteration in range(num_iterations):
        new_auth = {}
        new_hub = {}

        # Authority update: auth(p) = sum of hub scores of pages pointing to p
        for docid in all_docids:
            incoming = backward_links.get(docid, [])
            new_auth[docid] = sum(hub_scores.get(src, 0.0) for src in incoming)

        # Hub update: hub(p) = sum of authority scores of pages p points to
        for docid in all_docids:
            outgoing = forward_links.get(docid, [])
            new_hub[docid] = sum(new_auth.get(target, 0.0) for target in outgoing)

        # Normalize scores
        auth_norm = sum(v * v for v in new_auth.values()) ** 0.5
        hub_norm = sum(v * v for v in new_hub.values()) ** 0.5

        if auth_norm > 0:
            for docid in new_auth:
                new_auth[docid] /= auth_norm

        if hub_norm > 0:
            for docid in new_hub:
                new_hub[docid] /= hub_norm

        auth_scores = new_auth
        hub_scores = new_hub

        print(f"[HITS] Iteration {iteration + 1}/{num_iterations} completed")

    return hub_scores, auth_scores


def save_hits_scores(out_dir, hub_scores, auth_scores):
    """Save HITS scores to JSON file."""
    hits_data = {
        "hub_scores": {str(k): v for k, v in hub_scores.items()},
        "auth_scores": {str(k): v for k, v in auth_scores.items()}
    }

    out_path = pathlib.Path(out_dir) / "hits_scores.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(hits_data, f)

    print(f"[HITS] Saved scores to {out_path}")
    print(f"[HITS] Total documents with scores: {len(hub_scores)}")

    # Print top 10 hubs and authorities
    top_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    top_auths = sorted(auth_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\n[HITS] Top 10 Hub Scores:")
    for docid, score in top_hubs:
        print(f"  DocID {docid}: {score:.6f}")

    print("\n[HITS] Top 10 Authority Scores:")
    for docid, score in top_auths:
        print(f"  DocID {docid}: {score:.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute HITS hub and authority scores")
    ap.add_argument("--index_dir", required=True, help="Directory containing link_graph.json")
    ap.add_argument("--iterations", type=int, default=10, help="Number of HITS iterations (default: 10)")
    args = ap.parse_args()

    index_dir = pathlib.Path(args.index_dir)
    link_graph_path = index_dir / "link_graph.json"

    if not link_graph_path.exists():
        print(f"[ERROR] Link graph not found: {link_graph_path}")
        print("Please run the indexer first to generate link_graph.json")
        exit(1)

    print(f"[HITS] Loading link graph from {link_graph_path}")
    hub_scores, auth_scores = compute_hits(link_graph_path, num_iterations=args.iterations)

    save_hits_scores(index_dir, hub_scores, auth_scores)
    print("\n[OK] HITS computation complete!")