<p align="left">
  <img src="lgo/logo.png" alt="Project Logo" width="500"/>
</p>

# Module 5 — Embedding Database Optimization & SFT I

## 🚀 Quick Summary
Implemented a **hybrid retrieval system** that merges **dense semantic search (FAISS)** with **sparse keyword search (SQLite FTS5/BM25)** using score fusion (e.g., **RRF**, weighted sum).  
Extended Week 4’s RAG by storing **metadata + chunks + embeddings** in a **SQLite+FAISS** index and exposing a **/hybrid_search** FastAPI endpoint.  
Built an **SFT pipeline** (Full FT vs **LoRA/QLoRA**) with **TRL/PEFT/DeepSpeed**, training on **ChatML** datasets and comparing results.  
Demonstrates **retrieval quality optimization**, **index design**, **LLM fine-tuning**, and **evaluation mindset** (Recall@k, hit rate).

---

## 📖 Project Description
Pure vector search misses exact terms; pure keyword search misses semantic matches. This project **combines both** to “get the best of both worlds.”  
Part 1 focuses on **hybrid retrieval**: store document metadata and chunks, support both FAISS and FTS5/BM25, then **fuse rankings** for improved relevance.  
Part 2 focuses on **Supervised Fine-Tuning (SFT I)**: format data into **ChatML**, run Full FT vs **LoRA/QLoRA** on cloud GPUs with TRL/PEFT/DeepSpeed, and compare outputs and training efficiency.

---

## 🎯 Objectives

### A) Hybrid Retrieval (SQLite + FAISS)
- **Index structure**: Persist metadata (`title`, `author`, `year`, `keywords`) + chunk text in **SQLite**; store chunk embeddings in **FAISS** keyed by `doc_id`.  
- **Keyword search**: Implement **SQLite FTS5** (or **rank_bm25**) to get top-k by exact term relevance.  
- **Semantic search**: Use **FAISS** for dense vector similarity over chunk embeddings.  
- **Fusion**: Normalize scores and **combine** (e.g., weighted sum or **Reciprocal Rank Fusion**) into a single ranked list.  
- **Serve**: Provide `GET /hybrid_search?query=...&k=3` to return hybrid results as JSON.

### B) Supervised Fine-Tuning (SFT I)
- **Data**: Select an open dataset; **convert to ChatML** for multi-turn training.  
- **LoRA/QLoRA**: Run parameter-efficient SFT, record runtime, VRAM, loss curves.  
- **Full FT**: Train all weights for comparison; discuss overfitting/efficiency trade-offs.  
- **Evaluate**: Compare baseline vs LoRA vs Full FT on held-out prompts (helpfulness/style/accuracy).

---

## 🔒 Safety Wrapper

As part of the fine-tuning and evaluation experiments, I implemented an **AI safety wrapper** to test the model’s robustness against adversarial and jailbreak attempts.

### Key Features
- **Adversarial Input Detection:** Designed rules and lightweight classifiers to identify suspicious prompts (e.g., prompt injections, jailbreak attempts).  
- **Jailbreak Success Rate Reduction:** In evaluation, the wrapper successfully detected and blocked **>70%** of adversarial/jailbreak attempts.  
- **Integration with RAG/LLM Pipelines:** The safety layer sits between the user query and model response, making it transferable to **fraud prevention and compliance workflows**.  
- **Customizable Policies:** Detection thresholds and rules can be extended to cover domain-specific risks (e.g., financial fraud, compliance violations).  

### Example Workflow
User Query --> Safety Wrapper --> LLM Response
| |
Adversarial? ---> Block/Flag
---

## 🛠️ Tech Stack

**Hybrid Retrieval**
- **Storage/Index**: SQLite (tables + **FTS5**), **FAISS**  
- **Sparse ranking**: SQLite FTS5 or `rank_bm25`  
- **Dense embeddings**: `sentence-transformers` / OpenAI embeddings  
- **API**: **FastAPI**, Uvicorn  
- **Utilities**: `pandas`, `regex`, `langdetect`

**SFT I**
- **Training**: **TRL** (HF), **PEFT** (LoRA/QLoRA), **DeepSpeed** (ZeRO)  
- **Models**: Llama/Mistral/Zephyr (HF)  
- **Data**: **ChatML** format  
- **Monitoring**: `wandb` / TensorBoard (optional)

**SFT I**
- **Training**: **TRL** (HF), **PEFT** (LoRA/QLoRA), **DeepSpeed** (ZeRO)  
- **Models**: Llama/Mistral/Zephyr (HF)  
- **Data**: **ChatML** format  
- **Monitoring**: `wandb` / TensorBoard (optional)

---


## 🔥 Architecture / Workflow Diagram 

flowchart LR
  A[Docs/Chunks+Metadata] --> B[SQLite (metadata, FTS5)]
  A --> C[FAISS (embeddings)]
  D[Query] --> E[Keyword Search]
  D --> F[Vector Search]
  E --> G[Fusion (RRF/Weighted)]
  F --> G
  G --> H[Ranked Results via FastAPI]

---

## 📂 Deliverables

- `db/`  
  - `mydata.db` — SQLite with `documents` and `doc_chunks (FTS5)`  
  - `faiss.index` — FAISS index (one vector per chunk)  
- `hybrid_search/`  
  - `index_builder.py` — build SQLite+FAISS (ingest → chunk → embed → persist)  
  - `hybrid_search.py` — FAISS + FTS/BM25 retrieval & **fusion** (RRF/weighted)  
  - `api.py` — **FastAPI** service exposing `/hybrid_search`  
- `sft/`  
  - `prepare_chatml.py` — dataset → **ChatML** conversion  
  - `lora_train.py` — LoRA/QLoRA training with **TRL/PEFT**  
  - `full_finetune.py` — Full FT with **DeepSpeed** configs  
  - `inference_compare.py` — baseline vs LoRA vs Full FT comparison  
- `notebooks/`  
  - `evaluation_hybrid.ipynb` — Recall@k/HitRate for vector-only vs keyword-only vs hybrid  
  - `evaluation_sft.ipynb` — qualitative & quantitative SFT comparison  
- `reports/`  
  - `retrieval_metrics.md` — Recall@3/5 tables, fusion ablations  
  - `sft_results.md` — LoRA vs Full FT: loss curves, runtime, sample outputs

---


## 🔥 How to Run / Quick Start

# Build index
python hybrid_search/index_builder.py --data ./docs --out ./db

# Start API
uvicorn hybrid_search.api:app --reload --port 8000

# Query
curl "http://localhost:8000/hybrid_search?query=transformer attention&k=3"

# LoRA fine-tune
python sft/lora_train.py --model HuggingFaceH4/zephyr-7b-alpha --data ./data/chatml.jsonl --epochs 3

---


## 🌟 Highlights
- **Hybrid retrieval**: **FAISS + FTS5/BM25** with **RRF/weighted** score fusion for better relevance.  
- **Metadata-aware indexing**: Chunks linked to rich metadata for filtering and analytics.  
- **Production-minded**: **FastAPI** endpoint for programmatic hybrid search.  
- **Efficiency vs performance**: **LoRA/QLoRA** vs **Full FT** comparisons on cost, speed, and quality.  
- **Evaluation-driven**: Clear metrics (**Recall@k**, hit rate) and ablation of fusion strategies.

---

## 🧭 Example Usage

**Start API**
```bash
uvicorn hybrid_search.api:app --reload --port 8000
