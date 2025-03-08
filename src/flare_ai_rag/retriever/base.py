from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    @abstractmethod
    def semantic_search(self, query: str, top_k: int = 5) -> list[float]:
        """Perform semantic search using vector embeddings."""

    @abstractmethod
    def keyword_search(self, query: str, top_k: int = 5) -> tuple[list[int], list[float]]:
        """Perform keyword search using vector embeddings."""

    @abstractmethod
    def hybrid_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Perform hybrid search by combining multiple search types."""