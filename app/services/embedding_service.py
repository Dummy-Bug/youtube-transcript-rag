from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.configs.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache
def get_embedder() -> Embeddings:
    settings = get_settings()
    logger.info(f"Loading embedding model {settings.embedding_model}")
    return HuggingFaceEmbeddings(model=settings.embedding_model)
