import uuid
from functools import lru_cache

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from qdrant_client.models import Distance, VectorParams

from app.configs.settings import get_settings
from app.services.embedding_service import get_embedder


@lru_cache
def get_qdrant_client() -> QdrantClient:
    setting = get_settings()
    return QdrantClient(
        url=setting.qdrant_url,
    )


class VectorStoreRepository:

    def __init__(self, collection_name: str | None = None):
        setting = get_settings()
        self.client = get_qdrant_client()
        self.collection_name: str = collection_name or setting.collection_name
        self._vector_store: QdrantVectorStore | None = None
        self._ensure_collection()

    @property
    def vector_store(self) -> QdrantVectorStore:
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=get_embedder(),
            )
        return self._vector_store

    def _ensure_collection(self) -> None:

        if self.client.collection_exists(self.collection_name):
            return

        print(f"Collection {self.collection_name} not found. Creating...")

        vector_size = len(get_embedder().embed_query("dimension probe"))
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        if not documents:
            return []

        ids = [str(uuid.uuid4()) for _ in documents]
        self.vector_store.add_documents(documents, ids=ids)
        return ids

    def delete_collection(self) -> None:
        self._vector_store = None
        self.client.delete_collection(self.collection_name)

    def search(self, query: str, k: int | None = None) -> list[Document]:
        settings = get_settings()
        k = k or settings.retrieval_k
        return self.vector_store.similarity_search(query, k=k)

    def search_with_score(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        settings = get_settings()
        k = k or settings.retrieval_k
        return self.vector_store.similarity_search_with_score(query, k=k)

    def health_check(self) -> bool:
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
