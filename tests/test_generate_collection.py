import pandas as pd
import structlog
from qdrant_client import QdrantClient

from flare_ai_rag.ai import GeminiDenseEmbedding, ModelSparseEmbedding
from flare_ai_rag.retriever.config import RetrieverConfig
from flare_ai_rag.retriever.qdrant_collection import generate_collection
from flare_ai_rag.settings import settings
from flare_ai_rag.utils import load_json

logger = structlog.get_logger(__name__)


def main() -> None:
    # Load Qdrant config
    config_json = load_json(settings.input_path / "input_parameters.json")
    retriever_config = RetrieverConfig.load(config_json["retriever_config"])

    # Load the CSV file.
    df_docs = pd.read_csv(settings.data_path / "docs.csv", delimiter=",")
    logger.info("Loaded CSV Data.", num_rows=len(df_docs))

    # Initialize Qdrant client.
    client = QdrantClient(host=retriever_config.host, port=retriever_config.port)

    # Initialize Gemini client
    dense_embedding_client = GeminiDenseEmbedding(api_key=settings.gemini_api_key)
    sparse_embedding_client = ModelSparseEmbedding(
        retriever_config.sparse_embedding_model
    )

    generate_collection(
        df_docs,
        client,
        retriever_config,
        dense_embedding_client=dense_embedding_client,
        sparse_embedding_client=sparse_embedding_client,
    )


if __name__ == "__main__":
    main()
