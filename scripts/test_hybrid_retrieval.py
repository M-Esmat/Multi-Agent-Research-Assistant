from app.rag.vector_store import VectorStore
from app.rag.embeddings import EmbeddingService
from app.rag.bm25_retriever import BM25Retriever
from app.rag.hybrid_retriever import HybridRetriever


TEST_QUERIES = [
    "What is DistilBERT?",
    "How does knowledge distillation work?",
    "What is BERT used for in NLP?",
    "Explain transformer architecture",
    "What is retrieval augmented generation?"
]


def main():
    print("\n Initializing components...")

    embedder = EmbeddingService()
    vector_store = VectorStore()

    # NOTE: assumes vector store can expose raw documents
    
    
    documents = vector_store.get_all_documents()

    bm25 = BM25Retriever(documents)
    hybrid = HybridRetriever(vector_store, bm25, embedder, alpha=0.7)

    print("\n🔍 Running hybrid retrieval tests...\n")

    for query in TEST_QUERIES:
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        results = hybrid.search(query, top_k=5)

        for rank, item in enumerate(results, 1):
            doc_id, score = item

            print(f"\nRank {rank}")
            print(f"Doc ID: {doc_id}")
            print(f"Score: {score}")
            print("-" * 50)

if __name__ == "__main__":
    main()