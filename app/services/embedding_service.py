from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.configs.settings import get_settings


@lru_cache
def get_embedding_client() -> Embeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(model=settings.embedding_model)
