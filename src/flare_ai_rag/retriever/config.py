from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrieverConfig:
    """Configuration for the embedding model used in the retriever."""

    dense_embedding_model: str
    sparse_embedding_model: str
    collection_name: str
    vector_size: int
    host: str
    port: int

    @staticmethod
    def load(retriever_config: dict[str, Any]) -> "RetrieverConfig":
        return RetrieverConfig(
            dense_embedding_model=retriever_config["dense_embedding_model"],
            sparse_embedding_model=retriever_config["sparse_embedding_model"],
            collection_name=retriever_config["collection_name"],
            vector_size=retriever_config["vector_size"],
            host=retriever_config["host"],
            port=retriever_config["port"],
        )
