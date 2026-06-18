# 🔬 AI Scientific Intelligence Platform
## Production-Grade Hybrid Retrieval System for AI Research Papers

![Python](https://img.shields.io/badge/Python-3.11-blue)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-green)
![Status](https://img.shields.io/badge/Status-V1_Stable-brightgreen)

---

# 🚀 Overview

AI Scientific Intelligence Platform is a production-oriented Retrieval-Augmented Generation (RAG) retrieval system built on a corpus of more than 41,000 AI research paper chunks.

The project demonstrates how modern retrieval systems combine:

🧠 Dense Semantic Search
🔍 Sparse Keyword Search (BM25)
⚖️ Score Normalization
🔗 Hybrid Score Fusion
📚 Scientific Document Retrieval

to produce more robust and accurate search results.

## 🎯 Key Highlights
✅ 41K+ AI Research Paper Chunks
✅ SentenceTransformer Embeddings
✅ ChromaDB Vector Database
✅ BM25 Lexical Retrieval
✅ Hybrid Retrieval Architecture
✅ Score Normalization
✅ Modular Production-Style Codebase
✅ Local-First (No Paid APIs)
✅ Python 3.11

## 🏗️ Architecture

### 🔹 Version 1 — Baseline Semantic Retrieval

The initial system used dense embeddings and vector similarity search to retrieve relevant AI research papers.

```text
                     User Query
                          │
                          ▼

                 Query Normalization
                          │
                          ▼

                 SentenceTransformer
                          │
                          ▼

                   Query Embedding
                          │
                          ▼

                     ChromaDB
                    Vector Search
                          │
                          ▼

                   Top-K Results
```

### ✅ V1 Components

* Semantic Retrieval
* SentenceTransformer Embeddings
* ChromaDB Vector Database
* Retrieval Evaluation Pipeline
* Persistent Local Storage

---

### 🔹 Version 2 — Hybrid Retrieval System

V2 combines dense semantic search with sparse lexical search (BM25) to improve retrieval quality and ranking robustness.

```text
                        User Query
                             │
                             ▼

                  Query Normalization
                             │

            ┌────────────────┴────────────────┐
            │                                 │
            ▼                                 ▼

      Dense Retrieval                  Sparse Retrieval
   (SentenceTransformer)                   (BM25)

            │                                 │
            ▼                                 ▼

      Vector Scores                    BM25 Scores

            └──────────────┬───────────────┘
                           ▼

                 Score Normalization
                           ▼

                     Hybrid Fusion
                           ▼

                    Ranked Results
```

### ✅ V2 Improvements

* BM25 Lexical Retrieval
* Hybrid Retrieval (Dense + Sparse)
* Score Normalization
* Weighted Score Fusion
* Improved Retrieval Stability
* Modular Retriever Design

---

## 📊 V1 vs V2

| Feature                        | V1 | V2 |
| ------------------------------ | -- | -- |
| SentenceTransformer Embeddings | ✅  | ✅  |
| ChromaDB                       | ✅  | ✅  |
| Semantic Search                | ✅  | ✅  |
| BM25 Retrieval                 | ❌  | ✅  |
| Hybrid Retrieval               | ❌  | ✅  |
| Score Normalization            | ❌  | ✅  |
| Retrieval Evaluation           | ✅  | ✅  |
| Production-Oriented Design     | ✅  | ✅  |

---

## 🚀 Retrieval Pipeline

```text
Query
  │
  ▼

Normalize Query
  │
  ▼

Dense Search + BM25 Search
  │
  ▼

Score Normalization
  │
  ▼

Hybrid Fusion
  │
  ▼

Ranked Documents
  │
  ▼

Context for Future RAG Generation
```

---

## 🛠️ Technology Stack

| Category         | Technology           |
| ---------------- | -------------------- |
| Language         | Python 3.11          |
| Embeddings       | SentenceTransformers |
| Vector Database  | ChromaDB             |
| Sparse Retrieval | rank-bm25            |
| Dataset          | Hugging Face         |
| Environment      | Virtual Environment  |
| IDE              | VS Code              |
| Version Control  | Git & GitHub         |

---

## 🎯 Future Roadmap

### Version 3

* Cross-Encoder Reranking
* Recall@K Evaluation
* Mean Reciprocal Rank (MRR)
* FastAPI Service Layer
* Docker Support
* GitHub Actions CI/CD
* Automated Benchmarking
* Full RAG Generation Pipeline

---

## 📌 Project Goals

This project is designed to demonstrate:

* AI Engineering
* Retrieval-Augmented Generation (RAG)
* Information Retrieval Systems
* Vector Databases
* Software Engineering Best Practices
* Production-Oriented ML Development
* MLOps Readiness
