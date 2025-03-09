from typing import Any, override

from flare_ai_rag.ai import GeminiProvider, OpenRouterClient
from flare_ai_rag.responder import BaseResponder, ResponderConfig
from flare_ai_rag.utils import parse_chat_response


class GeminiResponder(BaseResponder):
    def __init__(
        self, client: GeminiProvider, responder_config: ResponderConfig
    ) -> None:
        """
        Initialize the responder with a GeminiProvider.

        :param client: An instance of OpenRouterClient.
        :param model: The model identifier to be used by the API.
        """
        self.client = client
        self.responder_config = responder_config

    @override
    def generate_response(self, query: str, retrieved_documents: list[dict]) -> str:
        """
        Generate a final answer using the query and the retrieved context.

        :param query: The input query.
        :param retrieved_documents: A list of dictionaries containing retrieved docs.
        :return: The generated answer as a string.
        """

        # Build Context from response history
        history_context = """
        List of previous 5 or less responses.
        Response 1 is the most recent response, and its tokens should be weighted more heavily.
        As the index of the response increases, its recency decreases, and the weight on its tokens should similarly decrease.
        Here is the list:
        """

        for idx, chat in enumerate(self.client.chat_history, start=1):
            history_context += f"Response {idx}:\n{chat}\n\n"


        # Build context from the retrieved documents.
        doc_context = "List of retrieved documents:\n"

        for idx, doc in enumerate(retrieved_documents[::-1], start=1):
            identifier = doc.get("filename", f"Doc{idx}")
            doc_context += f"Document {identifier}:\n{doc.get('text', '')}\n\n"

        # Compose the prompt
        prompt = (
            history_context
            + doc_context
            + f"User query: {query}\n"
            + self.responder_config.query_prompt
        )

        print(f'\n{prompt}\n')

        # Use the generate method of GeminiProvider to obtain a response.
        response = self.client.generate(
            prompt,
            response_mime_type=None,
            response_schema=None,
        )

        self.client.chat_history.append(response.text)
        if len(self.client.chat_history) > self.responder_config.context_size:
            self.client.chat_history = self.client.chat_history[1:]

        return response.text


class OpenRouterResponder(BaseResponder):
    def __init__(
        self, client: OpenRouterClient, responder_config: ResponderConfig
    ) -> None:
        """
        Initialize the responder with an OpenRouter client and the model to use.

        :param client: An instance of OpenRouterClient.
        :param model: The model identifier to be used by the API.
        """
        self.client = client
        self.responder_config = responder_config

    @override
    def generate_response(self, query: str, retrieved_documents: list[dict]) -> str:
        """
        Generate a final answer using the query and the retrieved context,
        and include citations.

        :param query: The input query.
        :param retrieved_documents: A list of dictionaries containing retrieved docs.
        :return: The generated answer as a string.
        """
        context = "List of retrieved documents:\n"

        # Build context from the retrieved documents.
        for idx, doc in enumerate(retrieved_documents, start=1):
            identifier = doc.get("metadata", {}).get("filename", f"Doc{idx}")
            context += f"Document {identifier}:\n{doc.get('text', '')}\n\n"

        # Compose the prompt
        prompt = context + f"User query: {query}\n" + self.responder_config.query_prompt
        # Prepare the payload for the completion endpoint.
        payload: dict[str, Any] = {
            "model": self.responder_config.model.model_id,
            "messages": [
                {"role": "system", "content": self.responder_config.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        if self.responder_config.model.max_tokens is not None:
            payload["max_tokens"] = self.responder_config.model.max_tokens
        if self.responder_config.model.temperature is not None:
            payload["temperature"] = self.responder_config.model.temperature

        # Send the prompt to the OpenRouter API.
        response = self.client.send_chat_completion(payload)

        return parse_chat_response(response)
