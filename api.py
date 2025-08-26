# api.py
from pathlib import Path
import os
import sqlite3
import faiss

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List

from search_hybrid import HybridSearcher  # make sure search_hybrid.py is the final version

app = FastAPI(title="Hybrid Search API")
searcher = HybridSearcher(alpha=0.6)  # uses absolute DB/FAISS paths inside the class

# ---------- Debug endpoint to verify paths and loaded data ----------
BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "mydata.db"
FAISS_PATH = BASE / "index.faiss"

@app.get("/debug_state")
def debug_state():
    info = {
        "cwd": os.getcwd(),
        "api_file_dir": str(BASE),
        "db_path": str(DB_PATH),
        "faiss_path": str(FAISS_PATH),
        "db_exists": DB_PATH.exists(),
        "faiss_exists": FAISS_PATH.exists(),
    }
    if DB_PATH.exists():
        conn = sqlite3.connect(str(DB_PATH))
        info["docs"] = conn.execute(
            "SELECT doc_id, title FROM documents ORDER BY doc_id"
        ).fetchall()
        info["num_chunks"] = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()
    if FAISS_PATH.exists():
        try:
            idx = faiss.read_index(str(FAISS_PATH))
            info["faiss_ntotal"] = idx.ntotal
        except Exception as e:
            info["faiss_error"] = str(e)
    return info

# ---------- Response models ----------
class Hit(BaseModel):
    doc_id: int
    title: str
    chunk_id: int
    score: float
    snippet: str

class SearchResponse(BaseModel):
    query: str
    strategy: str
    results: List[Hit]

def _to_hits(items):
    return [
        Hit(
            doc_id=x["doc_id"],
            title=x["title"],
            chunk_id=x["chunk_id"],
            score=float(x["score"]),
            snippet=x["snippet"],
        )
        for x in items
    ]

# ---------- Main search endpoint ----------
@app.get("/hybrid_search", response_model=SearchResponse)
def hybrid_search(q: str = Query(..., alias="query"), k: int = 3, method: str = "sum"):
    try:
        if method == "sum":
            pairs = searcher.hybrid_sum(q, k)
            hyd = searcher.hydrate(pairs)
            return SearchResponse(query=q, strategy="weighted-sum", results=_to_hits(hyd))

        elif method == "rrf":
            pairs = searcher.hybrid_rrf(q, k)
            hyd = searcher.hydrate(pairs)
            return SearchResponse(query=q, strategy="rrf", results=_to_hits(hyd))

        elif method == "vector":
            pairs = searcher.search_vector(q, k)
            hyd = searcher.hydrate(pairs)
            return SearchResponse(query=q, strategy="vector-only", results=_to_hits(hyd))

        elif method == "keyword":
            pairs = searcher.search_keyword(q, k)
            hyd = searcher.hydrate(pairs)
            return SearchResponse(query=q, strategy="keyword-only", results=_to_hits(hyd))

        else:
            return SearchResponse(query=q, strategy="unsupported", results=[])
    except ImportError as e:
        # Typically missing rank-bm25 for keyword fallback
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Surface the exception message for faster debugging
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

