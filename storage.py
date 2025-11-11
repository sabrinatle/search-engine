import os, gzip, json, heapq, pathlib

def write_jsonl(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

class PartialReader:
    def __init__(self, path):
        self.path = path
        self.fin = gzip.open(path, "rt", encoding="utf-8")
        self._next = None
        self._advance()

    def _advance(self):
        line = self.fin.readline()
        if not line:
            self._next = None
            return
        term, payload = line.rstrip("\n").split("\t", 1)
        self._next = (term, json.loads(payload))

    def peek(self):
        return self._next

    def pop(self):
        cur = self._next
        self._advance()
        return cur

    def close(self):
        try:
            self.fin.close()
        except Exception:
            pass

class PartialMerger:
    def __init__(self, partials_dir):
        self.partial_paths = sorted(pathlib.Path(partials_dir).glob("part_*.gz"))

    def merge(self):
        readers = [PartialReader(p) for p in self.partial_paths]
        heap = []
        for i, r in enumerate(readers):
            if r.peek() is not None:
                term, node = r.peek()
                heap.append((term, i))
        heapq.heapify(heap)

        current_term = None
        accum_postings = []

        while heap:
            term, i = heapq.heappop(heap)
            r = readers[i]
            t2, node = r.pop()
            assert t2 == term
            if r.peek() is not None:
                heapq.heappush(heap, (r.peek()[0], i))

            if current_term is None:
                current_term = term
                accum_postings = node["postings"]
            elif term == current_term:
                accum_postings.extend(node["postings"])
            else:
                accum_postings.sort(key=lambda p: p["d"])
                yield current_term, accum_postings
                current_term = term
                accum_postings = node["postings"]

        if current_term is not None:
            accum_postings.sort(key=lambda p: p["d"])
            yield current_term, accum_postings

        for r in readers:
            r.close()
