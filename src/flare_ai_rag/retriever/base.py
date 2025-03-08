from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    @abstractmethod
    def semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Perform semantic search using vector embeddings."""
        
    # semantic search will try to understand the 'vibes' behind the user input. Can it read between the lines?
    # ok may have been implemented alr 