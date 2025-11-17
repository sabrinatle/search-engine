#!/usr/bin/env python3
"""
Split the merged index into alphabetical chunks for fast disk-based lookup.
This allows O(log n) binary search within small files instead of O(n) linear search.
"""
import gzip
import json
import sys
from pathlib import Path


def split_index(index_path, output_dir, num_chunks=100):
    """
    Split index into num_chunks files based on term prefix.
    Each chunk is small enough to binary search quickly.
    """
    output_dir = Path(output_dir)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    print(f"[SPLIT] Reading index from {index_path}...", file=sys.stderr)

    # Read all terms and group by prefix
    chunks = [[] for _ in range(num_chunks)]

    with gzip.open(index_path, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if line_num % 100000 == 0:
                print(f"[SPLIT] Processed {line_num} terms...", file=sys.stderr)

            try:
                term, payload = line.rstrip("\n").split("\t", 1)
                # Assign to chunk based on consistent hash (first char or deterministic hash)
                # Use a simple deterministic hash: sum of ord values
                chunk_id = sum(ord(c) for c in term) % num_chunks
                chunks[chunk_id].append((term, payload))
            except Exception:
                continue

    # Write each chunk as a sorted file
    print(f"[SPLIT] Writing {num_chunks} chunk files...", file=sys.stderr)
    chunk_manifest = {}

    for chunk_id, chunk_data in enumerate(chunks):
        if not chunk_data:
            continue

        # Sort by term for binary search
        chunk_data.sort(key=lambda x: x[0])

        chunk_file = chunks_dir / f"chunk_{chunk_id:03d}.txt"
        with open(chunk_file, "w", encoding="utf-8") as f:
            for term, payload in chunk_data:
                f.write(f"{term}\t{payload}\n")

        # Record which terms are in this chunk
        terms_in_chunk = [term for term, _ in chunk_data]
        chunk_manifest[chunk_id] = {
            "file": f"chunk_{chunk_id:03d}.txt",
            "num_terms": len(terms_in_chunk),
            "first_term": terms_in_chunk[0],
            "last_term": terms_in_chunk[-1]
        }

        if (chunk_id + 1) % 10 == 0:
            print(f"[SPLIT] Written {chunk_id + 1}/{num_chunks} chunks", file=sys.stderr)

    # Save manifest
    with open(output_dir / "chunks_manifest.json", "w") as f:
        json.dump(chunk_manifest, f, indent=2)

    print(f"[OK] Split index into {len(chunk_manifest)} chunks", file=sys.stderr)
    return chunk_manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", required=True, help="Path to final_index.gz")
    parser.add_argument("--output_dir", required=True, help="Output directory (same as index dir)")
    parser.add_argument("--num_chunks", type=int, default=100, help="Number of chunks")
    args = parser.parse_args()

    split_index(args.index_path, args.output_dir, args.num_chunks)