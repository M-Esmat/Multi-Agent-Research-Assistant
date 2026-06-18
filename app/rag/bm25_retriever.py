from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents

        self.corpus = [doc["chunk"].lower().split() for doc in documents]

        # IMPORTANT: keep stable mapping
        self.ids = [doc["id"] for doc in documents]

        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query, top_k=5):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for i in ranked_indices:
            results.append({
                "id": self.ids[i],
                "score": float(scores[i]),
                "doc": self.documents[i]
            })

        return results