import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.rag.embeddings import EmbeddingService


embedding_service = EmbeddingService()

vector = embedding_service.embed_text(
    "What is retrieval augmented generation?"
)

print(type(vector))
print(len(vector))