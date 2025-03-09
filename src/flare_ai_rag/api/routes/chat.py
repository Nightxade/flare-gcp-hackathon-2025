import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from flare_ai_rag.ai import GeminiProvider
from flare_ai_rag.attestation import Vtpm, VtpmAttestationError
from flare_ai_rag.prompts import PromptService, SemanticRouterResponse
from flare_ai_rag.responder import GeminiResponder
from flare_ai_rag.retriever import QdrantRetriever
from flare_ai_rag.router import BaseQueryRouter

from flare_ai_rag.api.middleware import scrape

logger = structlog.get_logger(__name__)
router = APIRouter()


class ChatMessage(BaseModel):
    """
    Pydantic model for chat message validation.

    Attributes:
        message (str): The chat message content, must not be empty
    """

    message: str = Field(..., min_length=1)


class ChatRouter:
    """
    A simple chat router that processes incoming messages using the RAG pipeline.

    It wraps the existing query classification, document retrieval, and response
    generation components to handle a conversation in a single endpoint.
    """

    def __init__(  # noqa: PLR0913
        self,
        router: APIRouter,
        ai: GeminiProvider,
        query_router: BaseQueryRouter,
        query_improvement_router: BaseQueryRouter,
        retriever: QdrantRetriever,
        responder: GeminiResponder,
        attestation: Vtpm,
        prompts: PromptService,
    ) -> None:
        """
        Initialize the ChatRouter.

        Args:
            router (APIRouter): FastAPI router to attach endpoints.
            ai (GeminiProvider): AI client used by a simple semantic router
                to determine if an attestation was requested or if RAG
                pipeline should be used.
            query_router: RAG Component that classifies the query.
            retriever: RAG Component that retrieves relevant documents.
            responder: RAG Component that generates a response.
            attestation (Vtpm): Provider for attestation services
            prompts (PromptService): Service for managing prompts
        """
        self._router = router
        self.ai = ai
        self.query_router = query_router
        self.query_improvement_router = query_improvement_router
        self.retriever = retriever
        self.responder = responder
        self.attestation = attestation
        self.prompts = prompts
        self.logger = logger.bind(router="chat")
        self._setup_routes()

    def _setup_routes(self) -> None:
        """
        Set up FastAPI routes for the chat endpoint.
        """

        @self._router.post("/")
        async def chat(message: ChatMessage) -> dict[str, str] | None:  # pyright: ignore [reportUnusedFunction]
            """
            Process a chat message through the RAG pipeline.
            Returns a response containing the query classification and the answer.
            """
            try:
                self.logger.debug("Received chat message", message=message.message)

                # If attestation has previously been requested:
                if self.attestation.attestation_requested:
                    try:
                        resp = self.attestation.get_token([message.message])
                    except VtpmAttestationError as e:
                        resp = f"The attestation failed with  error:\n{e.args[0]}"
                    self.attestation.attestation_requested = False
                    return {"response": resp}

                route = await self.get_semantic_route(message.message)
                return await self.route_message(route, message.message)

            except Exception as e:
                self.logger.exception("Chat processing failed", error=str(e))
                raise HTTPException(status_code=500, detail=str(e)) from e

    @property
    def router(self) -> APIRouter:
        """Return the underlying FastAPI router with registered endpoints."""
        return self._router

    async def get_semantic_route(self, message: str) -> SemanticRouterResponse:
        """
        Determine the semantic route for a message using AI provider.

        Args:
            message: Message to route

        Returns:
            SemanticRouterResponse: Determined route for the message
        """
        try:
            prompt, mime_type, schema = self.prompts.get_formatted_prompt(
                "semantic_router", user_input=message
            )
            prompt = self.responder.client.history_context() + prompt
            print(prompt)
            route_response = self.ai.generate(
                prompt=prompt, response_mime_type=mime_type, response_schema=schema
            )
            print(route_response.text)
            return SemanticRouterResponse(route_response.text)
        except Exception as e:
            self.logger.exception("routing_failed", error=str(e))
            return SemanticRouterResponse.CONVERSATIONAL

    async def route_message(
        self, route: SemanticRouterResponse, message: str
    ) -> dict[str, str]:
        """
        Route a message to the appropriate handler based on semantic route.

        Args:
            route: Determined semantic route
            message: Original message to handle

        Returns:
            dict[str, str]: Response from the appropriate handler
        """
        handlers = {
            SemanticRouterResponse.RAG_ROUTER: self.handle_rag_pipeline,
            SemanticRouterResponse.SCRAPE: self.handle_scrape,
            SemanticRouterResponse.REQUEST_ATTESTATION: self.handle_attestation,
            SemanticRouterResponse.CONVERSATIONAL: self.handle_conversation,
        }

        handler = handlers.get(route)
        if not handler:
            return {"response": "Unsupported route"}

        return await handler(message)

    async def handle_rag_pipeline(self, query: str) -> dict[str, str]:
        """
        Handle attestation requests.

        Args:
            _: User query

        Returns:
            dict[str, str]: Response containing attestation request
        """
        # Step 1. Improve the user query with Gemini

        # Build Context from response history
        history_context = f"""
        Also, here is a list of your previous {len(self.responder.client.chat_history)} chat responses to the user.
        USE THIS CONTEXT TO HELP YOU INTERPRET AND REWRITE THE USER'S QUERY.
        Response 1 is the most recent response, and its tokens should be weighted more heavily.
        As the index of the response increases, its recency decreases, and the weight on its tokens should similarly decrease.
        Here is the list:
        """
        for idx, chat in enumerate(self.responder.client.chat_history, start=1):
            history_context += f"Response {idx}:\n{chat}\n\n"
        query = history_context + query

        prompt, mime_type, schema = self.prompts.get_formatted_prompt(
            "query_improvement", user_input=query
        )
        improved_query = self.query_improvement_router.route_query(
            prompt=prompt, response_mime_type=mime_type, response_schema=schema
        )
        self.logger.info("Query improved", improved_query=improved_query)

        # Step 2. Classify the user query.
        prompt, mime_type, schema = self.prompts.get_formatted_prompt(
            "rag_router", user_input=improved_query
        )
        classification = self.query_router.route_query(
            prompt=prompt, response_mime_type=mime_type, response_schema=schema
        )
        self.logger.info("Query classified", classification=classification)

        if classification == "ANSWER":
            # Step 3. Retrieve relevant documents.
            retrieved_docs = self.retriever.hybrid_search(query)
            self.logger.info("Documents retrieved")

            # Step 4. Generate the final answer.
            answer = self.responder.generate_response(query, retrieved_docs)
            self.logger.info("Response generated", answer=answer)
            return {"classification": classification, "response": answer}

        # Map static responses for CLARIFY and REJECT.
        static_responses = {
            "CLARIFY": "Please provide additional context.",
            "REJECT": "The query is out of scope.",
        }

        if classification in static_responses:
            return {
                "classification": classification,
                "response": static_responses[classification],
            }

        self.logger.exception("RAG Routing failed")
        raise ValueError(classification)

    async def handle_attestation(self, _: str) -> dict[str, str]:
        """
        Handle attestation requests.

        Args:
            _: Unused message parameter

        Returns:
            dict[str, str]: Response containing attestation request
        """
        prompt = self.prompts.get_formatted_prompt("request_attestation")[0]
        request_attestation_response = self.ai.generate(prompt=prompt)
        self.attestation.attestation_requested = True
        return {"response": request_attestation_response.text}

    async def handle_conversation(self, message: str) -> dict[str, str]:
        """
        Handle general conversation messages.

        Args:
            message: Message to process

        Returns:
            dict[str, str]: Response from AI provider
        """
        response = self.ai.send_message(message)
        return {"response": response.text}
    
    async def handle_scrape(self, query: str) -> dict[str, str]:
        prompt = f"Find the ticker in the following query, return only the ticker: {query}"
        ticker = self.ai.generate(prompt=prompt).text
        data = scrape(ticker)
        data = [{'date': 'Mar 9, 2025', 'open': '86,186.64', 'high': '86,425.25', 'low': '82,257.23', 'close': '82,573.92', 'volume': '21,896,366,080'}, {'date': 'Mar 8, 2025', 'open': '86,742.66', 'high': '86,847.27', 'low': '85,247.48', 'close': '86,154.59', 'volume': '18,206,118,081'}, {'date': 'Mar 7, 2025', 'open': '89,963.28', 'high': '91,191.05', 'low': '84,717.68', 'close': '86,742.67', 'volume': '65,945,677,657'}, {'date': 'Mar 6, 2025', 'open': '90,622.36', 'high': '92,804.94', 'low': '87,852.14', 'close': '89,961.73', 'volume': '47,749,810,486'}, {'date': 'Mar 5, 2025', 'open': '87,222.95', 'high': '90,998.24', 'low': '86,379.77', 'close': '90,623.56', 'volume': '50,498,988,027'}, {'date': 'Mar 4, 2025', 'open': '86,064.07', 'high': '88,911.27', 'low': '81,529.24', 'close': '87,222.20', 'volume': '68,095,241,474'}, {'date': 'Mar 3, 2025', 'open': '94,248.42', 'high': '94,429.75', 'low': '85,081.30', 'close': '86,065.67', 'volume': '70,072,228,536'}, {'date': 'Mar 2, 2025', 'open': '86,036.26', 'high': '95,043.44', 'low': '85,040.21', 'close': '94,248.35', 'volume': '58,398,341,092'}, {'date': 'Mar 1, 2025', 'open': '84,373.87', 'high': '86,522.30', 'low': '83,794.23', 'close': '86,031.91', 'volume': '29,190,628,396'}, {'date': 'Feb 28, 2025', 'open': '84,705.63', 'high': '85,036.32', 'low': '78,248.91', 'close': '84,373.01', 'volume': '83,610,570,576'}, {'date': 'Feb 27, 2025', 'open': '84,076.86', 'high': '87,000.78', 'low': '83,144.96', 'close': '84,704.23', 'volume': '52,659,591,954'}, {'date': 'Feb 26, 2025', 'open': '88,638.89', 'high': '89,286.25', 'low': '82,131.90', 'close': '84,347.02', 'volume': '64,597,492,134'}, {'date': 'Feb 25, 2025', 'open': '91,437.12', 'high': '92,511.08', 'low': '86,008.23', 'close': '88,736.17', 'volume': '92,139,104,128'}, {'date': 'Feb 24, 2025', 'open': '96,277.96', 'high': '96,503.45', 'low': '91,371.74', 'close': '91,418.17', 'volume': '44,046,480,529'}, {'date': 'Feb 23, 2025', 'open': '96,577.80', 'high': '96,671.88', 'low': '95,270.45', 'close': '96,273.92', 'volume': '16,999,478,976'}]
        prompt = f"Summarize and generate insights for the data. Present it to the user in a clear and simple manner. Do not make up information. {data}"
        response = self.ai.generate(prompt=prompt)
        return {'response':response.text}