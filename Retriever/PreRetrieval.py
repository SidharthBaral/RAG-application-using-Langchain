from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate


class PreRetrievalResult(BaseModel):
    """Structured output for pre-retrieval processing."""
    
    rewritten_query: str = Field(
        description="A clear, unambiguous, search-optimized rewrite of the user query."
    )
    multi_queries: List[str] = Field(
        description="2 to 3 distinct variations/perspectives of the query to capture different semantic aspects."
    )
    domain_route: str = Field(
        description="The identified target domain or dataset category for routing."
    )
    suggested_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional key-value metadata filters extracted from the query (e.g., {'speaker': 'guest_name'})."
    )


class PreRetrievalPipeline:
    """Pre-retrieval pipeline responsible for:

    1. Query Rewriting (Refining vague queries for dense search)
    2. Multi-Query Generation (Boosting retrieval recall)
    3. Domain-Aware Routing (Directing queries to specific collections/filters)
    """

    def __init__(
        self, 
        llm: BaseChatModel, 
        allowed_domains: Optional[List[str]] = None
    ):
        """Parameters:

        - llm: LangChain ChatModel instance (e.g., ChatGroq, ChatOpenAI)
        - allowed_domains: List of target collections or domain routes available in your system.
        """
        self.llm = llm
        self.allowed_domains = allowed_domains or [
            "technical_ai", 
            "podcast_interviews", 
            "general_qna",
            "teacher"
        ]
        
        # Enforce structured output schema
        self.structured_llm = self.llm.with_structured_output(PreRetrievalResult)
        self.chain = self._build_chain()

    def _build_chain(self):
        system_prompt = (
            "You are an expert search engine pre-retrieval optimizer for a RAG system.\n"
            "Your task is to analyze the user's input query and perform three operations:\n\n"
            "1. **Query Rewriting**: Refine vague terms, expand context, remove noise, and convert "
            "it into a clear, standalone search query optimized for vector similarity search.\n"
            "2. **Multi-Query Generation**: Generate 3 alternative versions of the query from "
            "different angles (e.g., conceptual, keyword-focused, specific sub-topic) to maximize retrieval recall.\n"
            "3. **Domain Routing**: Classify the query into EXACTLY ONE of these allowed domains: {allowed_domains}.\n"
            "4. **Metadata Extraction**: Extract key metadata filters if explicitly mentioned in the query (e.g., language, speaker, dates).\n\n"
            "Provide your response adhering strictly to the required schema."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "User Query: {query}\nConversation Context (if any): {context}")
        ])

        return prompt | self.structured_llm

    def run(self, query: str, context: Optional[str] = None) -> PreRetrievalResult:
        """Executes pre-retrieval operations on a user query."""
        response = self.chain.invoke({
            "query": query,
            "context": context or "None",
            "allowed_domains": ", ".join(self.allowed_domains)
        })
        return response