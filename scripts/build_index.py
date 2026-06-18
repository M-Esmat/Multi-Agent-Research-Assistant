from datasets import load_dataset

from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore


def main():
    dataset = load_dataset("jamescalam/ai-arxiv-chunked")["train"]

    embedder = EmbeddingService()
    store = VectorStore()

    batch_size = 64

    for i in range(0, 500, batch_size):  # start small for testing
        batch = dataset[i:i+batch_size]

        texts = batch["chunk"]

        embeddings = embedder.embed_batch(texts)

        ids = [
            f"{batch['id'][j]}-{batch['chunk-id'][j]}"
            for j in range(len(texts))
        ]

        metadatas = [
            {
                "title": batch["title"][j],
                "paper_id": batch["id"][j],
                "source": batch["source"][j],
                "category": batch["primary_category"][j],
            }
            for j in range(len(texts))
        ]

        store.add_documents(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        print(f"Indexed batch {i}-{i+batch_size}")

    print("Indexing complete!")


if __name__ == "__main__":
    main()
    
    
    
## 1. Real embedding pipeline text chunks --> sentenceTransformer -> 384-d vectors
## 2. persistent vector DB: chromadb collection: arxiv_chunks
# 3. batch ingestion system: chunked processing (64 batch size) - metaadaataaa stored - scalaable ingestion loop

# 4. real indexing pipeline --> 500+ scientific chunks indexed successfully

## dataset -> embeddingservice(MiniLLM) --> ChromaDB ( persistent storage ) --> indexed Vector knowledge base

