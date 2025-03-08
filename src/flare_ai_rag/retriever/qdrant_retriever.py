from typing import override

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, FusionQuery, Fusion, Prefetch

from flare_ai_rag.ai import EmbeddingTaskType, GeminiDenseEmbedding, ModelSparseEmbedding
from flare_ai_rag.retriever.base import BaseRetriever
from flare_ai_rag.retriever.config import RetrieverConfig

class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        client: QdrantClient,
        retriever_config: RetrieverConfig,
        dense_embedding_client: GeminiDenseEmbedding,
        sparse_embedding_client: ModelSparseEmbedding,
    ) -> None:
        """Initialize the QdrantRetriever."""
        self.client = client
        self.retriever_config = retriever_config
        self.dense_embedding_client = dense_embedding_client
        self.sparse_embedding_client = sparse_embedding_client

    @override
    def semantic_search(self, query: str, top_k: int = 25) -> list[float]:
        """
        Perform semantic search by converting the query into a dense vector
        and searching in Qdrant.

        :param query: The input query.
        :param top_k: Number of top results to return.
        :return: The dense vector.
        """
        # Convert the query into a vector embedding using Gemini
        query_vector = self.dense_embedding_client.embed_content(
            embedding_model=self.retriever_config.dense_embedding_model,
            contents=query,
            task_type=EmbeddingTaskType.RETRIEVAL_QUERY,
        )

        return query_vector
    
    @override
    def keyword_search(self, query: str, top_k: int = 25) -> tuple[list[int], list[float]]:
        """
        Perform keyword search by converting the query into a sparse vector
        and searching in Qdrant.

        :param query: The input query.
        :param top_k: Number of top results to return.
        :return: The sparse vector
        """
        # Convert the query into a vector embedding using Gemini
        query_vector = self.sparse_embedding_client.embed_content(
            contents=query,
        )

        return query_vector.indices.tolist(), query_vector.values.tolist()

    @override
    def hybrid_search(self, query: str, top_k: int = 25, limit: int = 5) -> list[dict]:
        """
        Perform hybrid search by combining dense and sparse embeddings with RRF.

        :param query: The input query
        :param top_k: Number of top results to return for semantic and keyword searches
        :param limit: Number of top results to return

        :return: A list of dictionaries, each representing a retrieved document.
        """
        semantic_vector = self.semantic_search(query, top_k)
        keyword_indices, keyword_values = self.keyword_search(query, top_k)
        keyword_vector = SparseVector(
            indices=keyword_indices,
            values=keyword_values,
        )

        prefetch = [
            Prefetch(
                query=semantic_vector,
                using="dense",
                limit=top_k//2,
            ),
            Prefetch(
                query=keyword_vector,
                using="sparse",
                limit=top_k//2
            ),
        ]

        results = self.client.query_points(
            collection_name=self.retriever_config.collection_name,
            prefetch=prefetch,
            query=FusionQuery(
                fusion=Fusion.RRF,
            ),
            with_payload=True,
            limit=top_k//2,
        )

        results = list(map(lambda point: point.model_dump()['payload'], results.points))[:limit]
        print(results)
        return results