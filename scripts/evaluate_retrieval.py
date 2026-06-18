from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore
from app.utils.query import normalize_query
# from app.rag.vector_store import convert_distance_to_similarity

TEST_QUERIES = [
    "What is DistilBERT?",
    "How does knowledge distillation work?",
    "What is BERT used for in NLP?",
    "Explain transformer architecture",
    "What is retrieval augmented generation?"
]


def print_results(query, results):
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    similarities = [1 - d for d in distances]

    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
        print(f"\nRank {i+1}")
        print(f"Distance: {dist:.4f}")
        print(f"Similarity: {1 - dist:.3f}")
        print(f"Title: {meta.get('title', 'N/A')}")
        print(f"Paper ID: {meta.get('paper_id', 'N/A')}")
        print(f"Snippet: {doc[:300]}")
        print("-" * 60)


def main():
    embedder = EmbeddingService()
    store = VectorStore()

    for query in TEST_QUERIES:
        query_vec = embedder.embed_text(query)

        results = store.query(query_vec, top_k=5)

        print_results(query, results)


if __name__ == "__main__":
    main()