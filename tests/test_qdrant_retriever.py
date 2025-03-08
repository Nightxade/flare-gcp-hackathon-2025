import structlog
from qdrant_client import QdrantClient

from flare_ai_rag.ai import GeminiDenseEmbedding, ModelSparseEmbedding
from flare_ai_rag.retriever import QdrantRetriever, RetrieverConfig
from flare_ai_rag.settings import settings
from flare_ai_rag.utils import load_json

logger = structlog.get_logger(__name__)


def main() -> None:
    # Load Qdrant config
    config_json = load_json(settings.input_path / "input_parameters.json")
    retriever_config = RetrieverConfig.load(config_json["retriever_config"])

    # Initialize Qdrant client
    client = QdrantClient(host=retriever_config.host, port=retriever_config.port)

    # Initialize Gemini client
    dense_embedding_client = GeminiDenseEmbedding(api_key=settings.gemini_api_key)
    sparse_embedding_client = ModelSparseEmbedding(
        retriever_config.sparse_embedding_model
    )

    # Initialize the retriever.
    retriever = QdrantRetriever(
        client=client,
        retriever_config=retriever_config,
        dense_embedding_client=dense_embedding_client,
        sparse_embedding_client=sparse_embedding_client,
    )

    # Define a sample query.
    query = "What is Flare?"

    # Perform semantic search.
    results = retriever.semantic_search(query, top_k=5)

    # Print out the search results.
    for result in results:
        logger.info("Search Results:", result=result)


if __name__ == "__main__":
    main()
