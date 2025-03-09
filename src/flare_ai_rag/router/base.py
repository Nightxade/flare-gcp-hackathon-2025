from abc import ABC, abstractmethod
from typing import Any

from flare_ai_rag.ai import BaseAIProvider, BaseClient

from .config import RouterConfig


class BaseQueryRouter(ABC):
    """
    An abstract base class defining the interface for query routings.
    """

    @abstractmethod
    def __init__(
        self, client: BaseAIProvider | BaseClient, config: RouterConfig
    ) -> None:
        """
        Constructor
        """

    @abstractmethod
    def route_query(
        self,
        prompt: str,
        response_mime_type: str | None = None,
        response_schema: Any | None = None,
    ) -> str:
        """
        Determine the type of the query: ANSWER, CLARIFY, or REJECT.
        """
