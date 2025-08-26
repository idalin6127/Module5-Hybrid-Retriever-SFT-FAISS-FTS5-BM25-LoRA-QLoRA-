# build_index.py — always reset tables and vector index, then rebuild everything
import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "mydata.db"
FAISS_PATH = "index.faiss"

# If True, remove old vector index files before rebuilding
CLEAR_VECTOR_FILES = True


def has_fts5(conn: sqlite3.Connection) -> bool:
    """Check whether SQLite has FTS5 enabled."""
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __t USING fts5(x)")
        conn.execute("DROP TABLE __t")
        return True
    except Exception:
        return False


def drop_tables(conn: sqlite3.Connection) -> None:
    """Hard reset: drop tables if they exist."""
    conn.executescript("""
    DROP TABLE IF EXISTS chunks_fts;
    DROP TABLE IF EXISTS chunks;
    DROP TABLE IF EXISTS documents;
    """)
    conn.commit()


def create_schema(conn: sqlite3.Connection, enable_fts5: bool) -> None:
    """Create schema. doc_id/chunk_id are AUTOINCREMENT to avoid PK collisions."""
    conn.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        title     TEXT,
        author    TEXT,
        year      INTEGER,
        keywords  TEXT
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id    INTEGER,
        content   TEXT,
        FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
    );
    """)
    if enable_fts5:
        conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(content, content='chunks', content_rowid='chunk_id');
        """)
    conn.commit()


def hard_reset_db(conn: sqlite3.Connection, enable_fts5: bool) -> None:
    """Preferred reset: drop then recreate tables."""
    drop_tables(conn)
    create_schema(conn, enable_fts5)


def soft_reset_db(conn: sqlite3.Connection) -> None:
    """Alternative: only delete rows (keep structure). Use if you don’t want to drop tables."""
    conn.executescript("""
    DELETE FROM chunks;
    DELETE FROM documents;
    """)
    try:
        conn.execute("DELETE FROM chunks_fts;")
    except Exception:
        pass
    conn.commit()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Simple character-based chunking with overlap."""
    out = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(text):
        piece = text[i:i + chunk_size].strip()
        if piece:
            out.append(piece)
        i += step
    return out


def ensure_docs_with_examples(folder: str = "docs") -> None:
    """Create docs/ and 3 example files if none exist."""
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)

    has_txt = any(f.endswith(".txt") for f in os.listdir(folder))
    if not has_txt:
        ex1 = (
            "Transformers are a type of deep learning model widely used in NLP.\n"
            "They rely on self-attention to focus on important words in a sentence "
            "no matter where they appear. Compared to RNNs, transformers handle long texts better."
        )
        ex2 = (
            "BM25 is a ranking function used in search engines.\n"
            "It improves upon TF-IDF using term frequency saturation and document length normalization.\n"
            "When you search for products like shoes online, BM25 ranks relevant results."
        )
        ex3 = (
            "FAISS is a library for efficient similarity search on large vector datasets.\n"
            "It is used for semantic search, recommendations, and image retrieval using cosine similarity or inner product."
        )
        with open(os.path.join(folder, "01_transformers.txt"), "w", encoding="utf-8") as f:
            f.write(ex1)
        with open(os.path.join(folder, "02_bm25.txt"), "w", encoding="utf-8") as f:
            f.write(ex2)
        with open(os.path.join(folder, "03_faiss.txt"), "w", encoding="utf-8") as f:
            f.write(ex3)


def ingest_folder(conn: sqlite3.Connection, folder: str = "docs") -> None:
    """Insert documents + chunks from docs/*.txt into SQLite."""
    ensure_docs_with_examples(folder)

    for fname in sorted(os.listdir(folder)):  # stable order
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(folder, fname)
        title = os.path.splitext(fname)[0]
        author, year, keywords = None, None, None

        cur = conn.execute(
            "INSERT INTO documents(title, author, year, keywords) VALUES(?,?,?,?)",
            (title, author, year, keywords)
        )
        doc_id = cur.lastrowid

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        chunks = chunk_text(text)
        for c in chunks:
            conn.execute("INSERT INTO chunks(doc_id, content) VALUES(?, ?)", (doc_id, c))

    conn.commit()


def backfill_fts(conn: sqlite3.Connection, enable_fts5: bool) -> None:
    """Populate chunks_fts from chunks if FTS5 is available."""
    if not enable_fts5:
        print("[INFO] FTS5 not available; keyword search will fall back to BM25 in your retrieval code.")
        return
    conn.execute("""
        INSERT INTO chunks_fts(rowid, content)
        SELECT chunk_id, content
        FROM chunks
        WHERE chunk_id NOT IN (SELECT rowid FROM chunks_fts);
    """)
    conn.commit()


def build_faiss(conn: sqlite3.Connection, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
    """Encode chunks and build a FAISS cosine-similarity index (IP on normalized vectors)."""
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "faiss-cpu is not installed. Run `pip install faiss-cpu`, "
            "or switch your retrieval code to use hnswlib as a fallback."
        ) from e

    rows = conn.execute("SELECT chunk_id, content FROM chunks ORDER BY chunk_id").fetchall()
    if not rows:
        raise RuntimeError("No chunks found. Place .txt files in docs/ or let this script create examples.")

    ids = [int(r[0]) for r in rows]
    texts = [r[1] for r in rows]

    print(f"[INFO] Encoding {len(texts)} chunks...")
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    dim = emb.shape[1]

    if CLEAR_VECTOR_FILES and os.path.exists(FAISS_PATH):
        try:
            os.remove(FAISS_PATH)
        except Exception:
            pass

    print(f"[INFO] Building FAISS index (dim={dim})...")
    index = faiss.IndexFlatIP(dim)  # normalized vectors + inner product ≈ cosine similarity
    idmap = faiss.IndexIDMap2(index)
    idmap.add_with_ids(emb.astype(np.float32), np.asarray(ids, dtype=np.int64))
    faiss.write_index(idmap, FAISS_PATH)
    print(f"[OK] FAISS index saved -> {FAISS_PATH}")


def print_doc_mapping(conn: sqlite3.Connection) -> None:
    """Print (doc_id, title) for writing gold labels later."""
    mapping = conn.execute("SELECT doc_id, title FROM documents ORDER BY doc_id").fetchall()
    print("Doc mapping:", mapping)


def main() -> None:
    # 1) Connect DB + detect FTS5
    conn = sqlite3.connect(DB_PATH)
    enable_fts5 = has_fts5(conn)

    # 2) Hard reset DB (drop & recreate). Use soft_reset_db(conn) instead if you prefer.
    hard_reset_db(conn, enable_fts5=enable_fts5)
    # Alternative:
    # create_schema(conn, enable_fts5)
    # soft_reset_db(conn)

    # 3) Ingest docs/*.txt into documents + chunks
    ingest_folder(conn, folder="docs")

    # 4) Populate FTS (if available)
    backfill_fts(conn, enable_fts5=enable_fts5)

    # 5) Build FAISS index
    build_faiss(conn)

    # 6) Show doc_id ↔ title mapping
    print_doc_mapping(conn)

    conn.close()


if __name__ == "__main__":
    main()
