# 🔬 AI Scientific Intelligence Platform — V1

![Python](https://img.shields.io/badge/Python-3.11-blue)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-green)
![Status](https://img.shields.io/badge/Status-V1_Stable-brightgreen)

---

## 🧠 Overview

**Version 1** is a baseline semantic retrieval system built for AI/ML research papers.

It demonstrates a complete Retrieval-Augmented Generation (RAG) pipeline using:

- SentenceTransformer embeddings
- ChromaDB vector database
- Semantic similarity search
- Basic retrieval evaluation

This version focuses purely on **dense retrieval (vector search)** without hybrid ranking or reranking.

---

## 🏗️ Architecture (V1)

```text
User Query
    │
    ▼
Query Normalization
    │
    ▼
SentenceTransformer Embedding
    │
    ▼
ChromaDB Vector Search
    │
    ▼
Top-K Most Similar Documents
```

---

## ⚙️ Core Components

### 🧠 Embeddings
- Model: `all-MiniLM-L6-v2`
- Converts text into dense vector representations
- Captures semantic meaning of queries and documents

### 🗄️ Vector Database
- ChromaDB (persistent storage)
- Stores ~41,000 AI/ML paper chunks
- Enables fast similarity search

### 🔍 Retrieval
- Cosine similarity via vector distance
- Top-K document retrieval
- Ranking based on semantic closeness

### 📊 Evaluation
- Query-based manual evaluation
- Basic Recall@K-style inspection
- Qualitative relevance checking

---

## 📊 Dataset

- Source: `jamescalam/ai-arxiv-chunked`
- ~41,584 pre-chunked AI research documents
- Domain: Machine Learning, NLP, Deep Learning

---

## 🚀 Features (V1)

### ✅ Implemented
- Semantic search using embeddings
- ChromaDB vector indexing
- Persistent local storage
- Basic retrieval evaluation pipeline
- Clean modular architecture

### ❌ Not included (intentionally)
- BM25 / keyword search
- Hybrid retrieval
- Score normalization
- Reranking models
- Advanced evaluation metrics (MRR, nDCG)

---

## 📦 Project Structure (V1)

```text
app/
│
├── rag/
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── loader.py
│
├── utils/
│   └── query.py
│
scripts/
│
├── build_index.py
├── evaluate_retrieval.py
```

---

## 🎯 Purpose of V1

V1 is designed as a **foundation layer** for modern retrieval systems:

- Understand embedding-based retrieval
- Build vector database pipelines
- Establish evaluation baselines
- Prepare for hybrid retrieval evolution (V2)

---

## 🔮 Next Version (V2)

V2 improves retrieval quality using:

- BM25 lexical search
- Hybrid fusion (Dense + Sparse)
- Score normalization
- Improved ranking stability
- Better evaluation insights

---

## 📌 Key Insight

V1 is intentionally simple — it represents the **semantic retrieval baseline** that modern RAG systems are built on.
