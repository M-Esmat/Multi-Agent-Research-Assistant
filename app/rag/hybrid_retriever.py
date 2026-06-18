# class HybridRetriever:
#     def __init__(self, vector_retriever, bm25_retriever, embedder, alpha=0.7):
#         self.vector = vector_retriever
#         self.bm25 = bm25_retriever
#         self.embedder = embedder
#         self.alpha = alpha

#     def search(self, query, top_k=10):

#         # VECTOR SEARCH (ChromaDB)
        
        
#         query_vec = self.embedder.embed_text(query)

#         vector_results = self.vector.query(
#             query_vec,
#             top_k=top_k * 2
#         )

#         vector_docs = vector_results["documents"][0]
#         vector_meta = vector_results["metadatas"][0]
#         vector_dist = vector_results["distances"][0]

#         vector_scores = {}

#         for meta, dist in zip(vector_meta, vector_dist):
#             doc_id = meta.get("paper_id", meta.get("id"))
#             vector_scores[doc_id] = 1 - dist  # similarity

#         # BM25 SEARCH
#         bm25_results = self.bm25.search(query, top_k=top_k * 2)

#         bm25_scores = {}

#         for item in bm25_results:
#             doc_id = item["id"]
#             score = item["score"]
#             bm25_scores[doc_id] = score

        
#         # HYBRID FUSION
        
#         scores = {}

#         all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())

#         for doc_id in all_ids:
#             v_score = vector_scores.get(doc_id, 0.0)
#             b_score = bm25_scores.get(doc_id, 0.0)

#             scores[doc_id] = (
#                 self.alpha * v_score +
#                 (1 - self.alpha) * b_score
#             )

#         ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

#         return ranked[:top_k]

class HybridRetriever:
    def __init__(self, vector_retriever, bm25_retriever, embedder, alpha=0.7):
        self.vector = vector_retriever
        self.bm25 = bm25_retriever
        self.embedder = embedder
        self.alpha = alpha

    def search(self, query, top_k=10):

        # -------------------------
        # VECTOR SEARCH
        # -------------------------
        query_vec = self.embedder.embed_text(query)

        vector_results = self.vector.query(
            query_vec,
            top_k=top_k * 2
        )

        vector_meta = vector_results["metadatas"][0]
        vector_dist = vector_results["distances"][0]

        vector_scores_raw = {}

        for meta, dist in zip(vector_meta, vector_dist):
            doc_id = meta.get("paper_id", meta.get("id"))
            score = 1 - dist
            vector_scores_raw[doc_id] = score

        # normalize vector scores
        if vector_scores_raw:
            v_min = min(vector_scores_raw.values())
            v_max = max(vector_scores_raw.values())

            vector_scores = {
                k: (v - v_min) / (v_max - v_min + 1e-8)
                for k, v in vector_scores_raw.items()
            }
        else:
            vector_scores = {}

        # -------------------------
        # BM25 SEARCH
        # -------------------------
        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        bm25_scores_raw = {
            item["id"]: item["score"]
            for item in bm25_results
        }

        # normalize BM25 scores
        if bm25_scores_raw:
            b_min = min(bm25_scores_raw.values())
            b_max = max(bm25_scores_raw.values())

            bm25_scores = {
                k: (v - b_min) / (b_max - b_min + 1e-8)
                for k, v in bm25_scores_raw.items()
            }
        else:
            bm25_scores = {}

        # -------------------------
        # HYBRID FUSION
        # -------------------------
        scores = {}

        all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())

        for doc_id in all_ids:
            v_score = vector_scores.get(doc_id, 0.0)
            b_score = bm25_scores.get(doc_id, 0.0)

            scores[doc_id] = (
                self.alpha * v_score +
                (1 - self.alpha) * b_score
            )

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return ranked[:top_k]