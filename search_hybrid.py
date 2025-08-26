from pathlib import Path
from typing import List, Tuple, Dict, Any
import sqlite3
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss


# ---------- Paths (absolute, relative to this file) ----------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "mydata.db")
FAISS_PATH = str(BASE_DIR / "index.faiss")


class HybridSearcher:
    def __init__(self, alpha: float = 0.6, use_fts_first: bool = True):
        """
        alpha: weight for vector score in weighted-sum fusion (0~1).
        use_fts_first: try to use FTS5 for keyword search; if unavailable, fallback to BM25.
        """
        self.alpha = alpha

        # Single DB connection used everywhere (allow usage across threads for FastAPI)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Sentence embeddings model (must match the one used during indexing)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # FAISS index
        self.index = faiss.read_index(FAISS_PATH)

        # Keyword channel: FTS5 or BM25 fallback
        self.use_fts = self._has_fts5() if use_fts_first else False
        # Also ensure the FTS table actually exists; otherwise fall back
        if self.use_fts and not self._table_exists("chunks_fts"):
            self.use_fts = False
        # Lazy BM25 state; build on first need
        self._bm25 = None
        self._bm25_ids = []
        self._bm25_tok = []
        if not self.use_fts:
            self._ensure_bm25()

    # ------------- Utilities -------------

    def _has_fts5(self) -> bool:
        try:
            self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __t USING fts5(x)")
            self.conn.execute("DROP TABLE __t")
            return True
        except Exception:
            return False

    def _table_exists(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return row is not None

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "rank-bm25 is required for keyword search fallback. Install with `pip install rank-bm25`."
            ) from e
        rows = self.conn.execute(
            "SELECT chunk_id, content FROM chunks ORDER BY chunk_id"
        ).fetchall()
        self._bm25_ids = [r["chunk_id"] for r in rows]
        corpus = [r["content"] for r in rows]
        self._bm25_tok = [doc.lower().split() for doc in corpus]
        self._bm25 = BM25Okapi(self._bm25_tok)
        print("[INFO] Using BM25 (rank_bm25).")

    @staticmethod
    def _minmax(d: Dict[int, float]) -> Dict[int, float]:
        if not d:
            return {}
        vals = np.asarray(list(d.values()), dtype=float)
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            # all the same; return zeros so they don't dominate
            return {k: 0.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    # ------------- Vector (semantic) search -------------

    def search_vector(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        Returns a list of (chunk_id, score). Score is cosine-like (IP on normalized vectors).
        """
        q = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        D, I = self.index.search(q, k)
        out = []
        for cid, s in zip(I[0], D[0]):
            if cid != -1:
                out.append((int(cid), float(s)))
        return out

    # ------------- Keyword search (FTS5 or BM25) -------------

    def search_keyword(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        Returns (chunk_id, score) for keyword results.
        - If FTS5 is available: use MATCH and assign simple descending scores (1.0, 0.9, ...)
          because not all SQLite builds expose bm25() function consistently.
        - Else: use rank_bm25 scores.
        """
        if self.use_fts:
            try:
                # Wrap as a phrase to avoid tokenizer interpreting bare tokens as columns
                fts_query = f'"{query}"'
                rows = self.conn.execute(
                    """
                    SELECT rowid AS chunk_id
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ?
                    LIMIT ?
                    """,
                    (fts_query, k),
                ).fetchall()
                if rows:
                    # simple heuristic scores
                    return [(int(r["chunk_id"]), float(1.0 - 0.1 * i)) for i, r in enumerate(rows)]
            except sqlite3.OperationalError:
                # FTS MATCH failed (e.g., malformed query or tokenizer issue). Fall back.
                pass
            # FTS returned no hits or failed; fall back to BM25
            self._ensure_bm25()
        # BM25 path
        toks = query.lower().split()
        scores = self._bm25.get_scores(toks)
        idx = np.argsort(-scores)[:k]
        return [(int(self._bm25_ids[i]), float(scores[i])) for i in idx if scores[i] > 0.0]

    # ------------- Fusion methods -------------

    def hybrid_sum(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        Weighted-sum fusion after min-max normalization.
        final = alpha * vec + (1 - alpha) * keyword
        """
        v = self.search_vector(query, k)
        t = self.search_keyword(query, k)

        v_dic = {cid: s for cid, s in v}
        t_dic = {cid: s for cid, s in t}

        v_n = self._minmax(v_dic)
        t_n = self._minmax(t_dic)

        all_ids = set(v_dic) | set(t_dic)
        fused = []
        for cid in all_ids:
            vs = v_n.get(cid, 0.0)
            ts = t_n.get(cid, 0.0)
            fused.append((cid, self.alpha * vs + (1 - self.alpha) * ts))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:k]

    def hybrid_rrf(self, query: str, k: int = 5, C: int = 60) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion (RRF).
        score = sum(1 / (C + rank_i)) across lists.
        """
        v = self.search_vector(query, k)
        t = self.search_keyword(query, k)

        rrf: Dict[int, float] = {}
        for rank, (cid, _) in enumerate(v, start=1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (C + rank)
        for rank, (cid, _) in enumerate(t, start=1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (C + rank)

        fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
        return fused

    # ------------- Hydration -------------

    def hydrate(self, pairs: List[Tuple[int, float]], k: int = None) -> List[Dict[str, Any]]:
        """
        Convert (chunk_id, score) pairs into rich objects with doc_id/title/snippet.
        Always uses self.conn (absolute DB path) — do NOT open new connections here.
        """
        if k is None:
            k = len(pairs)

        results: List[Dict[str, Any]] = []
        for cid, score in pairs[:k]:
            row = self.conn.execute(
                """
                SELECT chunks.chunk_id, chunks.content, documents.doc_id, documents.title
                FROM chunks 
                JOIN documents ON chunks.doc_id = documents.doc_id
                WHERE chunks.chunk_id = ?
                """,
                (cid,),
            ).fetchone()

            if row:
                results.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "doc_id": row["doc_id"],
                        "title": row["title"],
                        "snippet": row["content"][:300],
                        "score": float(score),
                    }
                )

        return results



