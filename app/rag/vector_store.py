import chromadb
from chromadb.config import Settings

from app.core.config import settings
from app.utils.query import normalize_query

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name="arxiv_chunks"
        )

    def add_documents(self, ids, embeddings, documents, metadatas):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    # def query(self, query_embedding, top_k=5):
    #     return self.collection.query(
    #         query_embeddings=[query_embedding],
    #         n_results=top_k
    #     )

    ## TRY UPDATING QUERY METHOD FOR BETTER RESULTS:
    def query(self, query_embedding, top_k=5):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        return {
            "documents": results["documents"],
            "metadatas": results["metadatas"],
            "distances": results["distances"]
        }
    
    
    ## SIMILARITY SCORE HELPER FUNCTION
    def convert_distance_to_similarity(self, distances):
        return [
            [1 - d for d in batch]
            for batch in distances
        ]
    
    def get_all_documents(self):
        data = self.collection.get(include=["documents", "metadatas"])

        docs = []

        for doc, meta in zip(data["documents"], data["metadatas"]):
            docs.append({
                "id": meta.get("paper_id", meta.get("id")),
                "chunk": doc,
                "metadata": meta
            })

        return docs
        
