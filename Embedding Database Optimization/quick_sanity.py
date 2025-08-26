# quick_sanity.py
from pathlib import Path
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "mydata.db"
FAISS_PATH = BASE / "index.faiss"

print("BASE:", BASE)
print("DB exists:", DB_PATH.exists(), DB_PATH)
print("FAISS exists:", FAISS_PATH.exists(), FAISS_PATH)

# 1) DB sanity
conn = sqlite3.connect(str(DB_PATH))
docs = conn.execute("SELECT doc_id, title FROM documents ORDER BY doc_id").fetchall()
chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
print("Docs:", docs)
print("Num chunks:", chunks)

# 2) FAISS sanity
index = faiss.read_index(str(FAISS_PATH))
print("FAISS ntotal:", index.ntotal)

# 3) Vector query manually (bypass your API/searcher)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
q = model.encode(["what is attention in transformers"], normalize_embeddings=True).astype("float32")
D, I = index.search(q, 3)
print("Top IDs:", I[0], "Scores:", D[0])

# 4) Hydrate one hit to see text (if any)
if I[0][0] != -1:
    row = conn.execute("""
        SELECT chunks.chunk_id, chunks.content, documents.doc_id, documents.title
        FROM chunks JOIN documents ON chunks.doc_id = documents.doc_id
        WHERE chunks.chunk_id = ?
    """, (int(I[0][0]),)).fetchone()
    print("Hit snippet:", row)
conn.close()
