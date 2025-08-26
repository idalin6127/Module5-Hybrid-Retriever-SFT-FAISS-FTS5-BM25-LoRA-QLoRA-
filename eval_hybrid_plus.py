# eval_hybrid_plus.py — richer evaluation with Recall/MRR/nDCG, alpha&k sweeps, and artifacts
from typing import List, Dict, Set, Tuple
from pathlib import Path
import sqlite3, json, csv, math, datetime
from search_hybrid import HybridSearcher

# ---------- 1) Your gold in titles (edit this to your case) ----------
QUERIES_TITLES = [
    # Transformers
    {"q": "what is attention in transformers", "relevant_titles": {"01_transformers"}},
    {"q": "why are transformers good for long texts", "relevant_titles": {"01_transformers"}},
    {"q": "self-attention explained simply", "relevant_titles": {"01_transformers"}},

    # BM25
    {"q": "how does bm25 rank search results", "relevant_titles": {"02_bm25"}},
    {"q": "difference between bm25 and tfidf", "relevant_titles": {"02_bm25"}},
    {"q": "do search engines use bm25", "relevant_titles": {"02_bm25"}},

    # FAISS
    {"q": "cosine similarity in faiss", "relevant_titles": {"03_faiss"}},
    {"q": "how does faiss speed up similarity search", "relevant_titles": {"03_faiss"}},
    {"q": "library for large-scale vector search", "relevant_titles": {"03_faiss"}},

    # Multi-answer
    {"q": "difference between keyword search and vector search", "relevant_titles": {"02_bm25", "03_faiss"}},
]

# ---------- 2) Paths ----------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "mydata.db")
ARTIFACTS_DIR = BASE_DIR / "eval_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ---------- 3) Helpers ----------
def load_title_to_id() -> Dict[str, int]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT doc_id, title FROM documents").fetchall()
    conn.close()
    m = {r[1]: int(r[0]) for r in rows}
    if not m:
        raise RuntimeError("No documents found in DB. Run build_index.py first.")
    return m

def convert_titles_to_ids(queries_titles: List[Dict]) -> List[Dict]:
    title2id = load_title_to_id()
    converted = []
    for item in queries_titles:
        titles = item["relevant_titles"]
        missing = [t for t in titles if t not in title2id]
        if missing:
            raise ValueError(f"Titles not found: {missing}. Available: {sorted(title2id.keys())}")
        converted.append({"q": item["q"], "relevant_doc_ids": {title2id[t] for t in titles}})
    return converted

# Metrics (doc-level)
def recall_at_k(doc_list: List[int], gold: Set[int], k: int) -> float:
    return 1.0 if any(d in gold for d in doc_list[:k]) else 0.0

def mrr_at_k(doc_list: List[int], gold: Set[int], k: int) -> float:
    for i, d in enumerate(doc_list[:k], start=1):
        if d in gold:
            return 1.0 / i
    return 0.0

def ndcg_at_k(doc_list: List[int], gold: Set[int], k: int) -> float:
    # rel = 1 if doc is relevant, else 0
    dcg = 0.0
    for i, d in enumerate(doc_list[:k], start=1):
        rel = 1.0 if d in gold else 0.0
        if rel:
            dcg += (2**rel - 1) / math.log2(i + 1)
    # Ideal DCG: put all relevant docs at the top
    ideal_rels = min(len(gold), k)
    idcg = sum((2**1 - 1) / math.log2(i + 1) for i in range(1, ideal_rels + 1))
    return (dcg / idcg) if idcg > 0 else 0.0

def docs_from_results(results: List[Dict]) -> List[int]:
    return [r["doc_id"] for r in results]

def evaluate_one(searcher: HybridSearcher, queries: List[Dict], method: str, k: int) -> Dict[str, float]:
    r_sum = mrr_sum = ndcg_sum = 0.0
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

        doc_list = docs_from_results(res)
        r_sum     += recall_at_k(doc_list, gold, k)
        mrr_sum   += mrr_at_k(doc_list, gold, k)
        ndcg_sum  += ndcg_at_k(doc_list, gold, k)

    n = len(queries)
    return {
        f"Recall@{k}": r_sum / n,
        f"MRR@{k}": mrr_sum / n,
        f"nDCG@{k}": ndcg_sum / n
    }

def dump_examples_md(searcher: HybridSearcher, queries: List[Dict], k: int = 3, n_show: int = 3):
    path = ARTIFACTS_DIR / "qualitative_examples.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# Qualitative Examples\n\n")
        f.write(f"_Generated: {datetime.datetime.now().isoformat()}_\n\n")
        for item in queries[:n_show]:
            q = item["q"]
            gold = sorted(item["relevant_doc_ids"])
            v = searcher.hydrate(searcher.search_vector(q, k))
            t = searcher.hydrate(searcher.search_keyword(q, k))
            s = searcher.hydrate(searcher.hybrid_sum(q, k))
            f.write(f"## Q: {q}\n")
            f.write(f"Gold doc_ids: {gold}\n\n")
            def lines(name, res):
                rows = [f"- doc_id={r['doc_id']} | title={r['title']} | score={r['score']:.4f} | snippet=\"{r['snippet'][:120]}\""
                        for r in res]
                return f"**{name}**\n" + "\n".join(rows) + "\n\n"
            f.write(lines("Vector", v))
            f.write(lines("Keyword", t))
            f.write(lines("Hybrid-sum", s))
    print(f"[OK] Wrote qualitative examples to {path}")

def dump_scores_csv(rows: List[Dict[str, float]], out_path: Path):
    keys = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[OK] Wrote scores to {out_path}")

def save_config(config: Dict):
    (ARTIFACTS_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[OK] Saved config to {ARTIFACTS_DIR / 'config.json'}")

# ---------- 4) Main ----------
def main():
    # Convert title gold to doc_id gold for current DB
    QUERIES = convert_titles_to_ids(QUERIES_TITLES)

    # Reproducibility config
    config = {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunking": {"size": 500, "overlap": 50},  # keep aligned with build_index.py
        "fusion": {"methods": ["sum", "rrf"], "alpha_default": 0.6},
        "k_values": [1, 3, 5],
        "alpha_sweep": [0.3, 0.5, 0.7],
        "db_path": DB_PATH
    }
    save_config(config)

    # Init searcher (uses same model & paths as build time)
    searcher = HybridSearcher(alpha=config["fusion"]["alpha_default"])

    # A) Evaluate across methods and k ∈ {1,3,5}
    methods = ["vector", "keyword", "sum", "rrf"]
    table = []
    for m in methods:
        row = {"method": m}
        for k in config["k_values"]:
            scores = evaluate_one(searcher, QUERIES, method=m, k=k)
            row.update(scores)
        table.append(row)

    dump_scores_csv(table, ARTIFACTS_DIR / "scores_by_method_and_k.csv")

    # B) Alpha sweep for weighted-sum at k=3
    k_target = 3
    alpha_rows = []
    for a in config["alpha_sweep"]:
        searcher.alpha = a
        scores = evaluate_one(searcher, QUERIES, method="sum", k=k_target)
        alpha_rows.append({"alpha": a, **scores})
    dump_scores_csv(alpha_rows, ARTIFACTS_DIR / f"alpha_sweep_k{k_target}.csv")

    # C) Qualitative examples (first 3 queries)
    dump_examples_md(searcher, QUERIES, k=3, n_show=3)

    print("\n=== Summary ===")
    for r in table:
        print(r)
    print("\nAlpha sweep @k=3:")
    for r in alpha_rows:
        print(r)

if __name__ == "__main__":
    main()
