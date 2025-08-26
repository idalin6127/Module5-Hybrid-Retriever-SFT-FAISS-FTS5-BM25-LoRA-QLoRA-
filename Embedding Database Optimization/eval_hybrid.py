# eval_hybrid.py
from typing import List, Dict, Set, Tuple
import sqlite3
from pathlib import Path
from search_hybrid import HybridSearcher

# ---------- Section you can edit: write gold labels using document titles ----------
# Tip: Titles are stored in documents.title during build (usually filename without extension)
QUERIES_TITLES = [
    # Transformers
    {"q": "what is attention in transformers", "relevant_titles": {"01_transformers"}},
    {"q": "why are transformers good for long texts", "relevant_titles": {"01_transformers"}},
    {"q": "explain self-attention in simple words", "relevant_titles": {"01_transformers"}},
    {"q": "transformer vs rnn for long sequences", "relevant_titles": {"01_transformers"}},

    # BM25
    {"q": "how does bm25 rank search results", "relevant_titles": {"02_bm25"}},
    {"q": "difference between bm25 and tfidf", "relevant_titles": {"02_bm25"}},
    {"q": "do search engines use bm25 by default", "relevant_titles": {"02_bm25"}},
    {"q": "keyword search scoring with bm25", "relevant_titles": {"02_bm25"}},

    # FAISS
    {"q": "cosine similarity in faiss", "relevant_titles": {"03_faiss"}},
    {"q": "how does faiss speed up similarity search", "relevant_titles": {"03_faiss"}},
    {"q": "library for large scale vector search", "relevant_titles": {"03_faiss"}},

    # Multiple valid answers (BM25 + FAISS)
    {"q": "difference between keyword search and vector search", "relevant_titles": {"02_bm25", "03_faiss"}},

    # Slightly trickier queries (test hybrid capabilities)
    {"q": "rank results with length normalization", "relevant_titles": {"02_bm25"}},   # hints BM25 properties
    {"q": "semantic neighbors with inner product", "relevant_titles": {"03_faiss"}}, # hints vector retrieval
    {"q": "attention lets model focus on important tokens", "relevant_titles": {"01_transformers"}},
]

# ---------- Paths (consistent with search_hybrid.py) ----------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "mydata.db")

def load_title_to_id() -> Dict[str, int]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT doc_id, title FROM documents").fetchall()
    conn.close()
    m = {r[1]: int(r[0]) for r in rows}
    if not m:
        raise RuntimeError("No documents found in DB. Did you run build_index.py?")
    return m

def convert_titles_to_ids(queries_titles: List[Dict]) -> List[Dict]:
    title2id = load_title_to_id()
    converted = []
    for item in queries_titles:
        titles = item["relevant_titles"]
        missing = [t for t in titles if t not in title2id]
        if missing:
            raise ValueError(f"Titles not found in DB: {missing}. "
                             f"Available: {sorted(title2id.keys())}")
        converted.append({"q": item["q"], "relevant_doc_ids": {title2id[t] for t in titles}})
    return converted

def doc_level_hit(results: List[Dict], gold_doc_ids: Set[int]) -> bool:
    """If ANY returned chunk belongs to a relevant doc_id -> hit (doc-level)."""
    return any(r["doc_id"] in gold_doc_ids for r in results)

def eval_strategy(searcher: HybridSearcher, queries: List[Dict], method: str, k: int = 3) -> float:
    hit = 0
    for item in queries:
        q = item["q"]
        gold = item["relevant_doc_ids"]
        if method == "vector":
            res = searcher.hydrate(searcher.search_vector(q, k))
        elif method == "keyword":
            res = searcher.hydrate(searcher.search_keyword(q, k))
        elif method == "sum":
            res = searcher.hydrate(searcher.hybrid_sum(q, k))
        elif method == "rrf":
            res = searcher.hydrate(searcher.hybrid_rrf(q, k))
        else:
            raise ValueError("Unknown method")

        if doc_level_hit(res, gold):
            hit += 1
    return hit / len(queries)

def debug_print_examples(searcher: HybridSearcher, queries: List[Dict], k: int = 3, n_show: int = 3) -> None:
    """Print a few examples to inspect what's being returned."""
    print("\n--- Debug examples ---")
    for item in queries[:n_show]:
        q = item["q"]
        gold = item["relevant_doc_ids"]
        v = searcher.hydrate(searcher.search_vector(q, k))
        t = searcher.hydrate(searcher.search_keyword(q, k))
        s = searcher.hydrate(searcher.hybrid_sum(q, k))
        print(f"\nQ: {q}")
        print(f"Gold doc_ids: {sorted(gold)}")
        print("Vector:", [(x['doc_id'], x['title']) for x in v])
        print("Keyword:", [(x['doc_id'], x['title']) for x in t])
        print("Hybrid-sum:", [(x['doc_id'], x['title']) for x in s])

def main():
    # 1) Convert gold labels based on titles into current DB doc_ids
    QUERIES = convert_titles_to_ids(QUERIES_TITLES)

    # 2) Initialize the searcher
    searcher = HybridSearcher(alpha=0.6)

    # 3) Optional: print a few examples for manual inspection
    debug_print_examples(searcher, QUERIES, k=3, n_show=3)

    # 4) Evaluate Recall@3
    for m in ["vector", "keyword", "sum", "rrf"]:
        r = eval_strategy(searcher, QUERIES, method=m, k=3)
        print(f"{m:8s} Recall@3 = {r:.3f}")

if __name__ == "__main__":
    main()