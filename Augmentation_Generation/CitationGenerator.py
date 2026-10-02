from typing import List
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate


class CitationReference(BaseModel):
    """Metadata and quote for a specific source citation."""
    citation_id: int = Field(description="Numerical ID of the citation used in text, e.g. 1, 2")
    source: str = Field(description="Source identifier, URL, or document title")
    quoted_snippet: str = Field(description="Exact snippet or key sentence from context supporting the claim")


class GenerationWithCitations(BaseModel):
    """Structured response containing the grounded answer and structured citations."""
    answer: str = Field(
        description="The factual answer containing inline numerical citations like [1], [2] where facts are stated."
    )
    citations: List[CitationReference] = Field(
        default_factory=list,
        description="List of cited source references corresponding to the inline citation markers."
    )


class GenerationPipeline:
    """Pure Generation Component: Receives an augmented payload and 

    generates a grounded answer with structured citations.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(GenerationWithCitations)
        self.chain = self._build_chain()

    def _build_chain(self):
        system_instruction = (
            "You are a factual assistant generating answers for a RAG system.\n"
            "STRICT RULES FOR CITATION AND GROUNDING:\n"
            "1. Answer the query using ONLY the information provided in the SOURCES list.\n"
            "2. Attach inline numerical citations (e.g., [1], [2]) directly after stating any fact derived from Source 1, Source 2, etc.\n"
            "3. If a fact draws from multiple sources, include multiple markers like [1][2].\n"
            "4. For every citation used in your answer, extract the precise supporting snippet/quote into the citations list.\n"
            "5. If the sources do not contain enough facts to answer, explicitly state that context is insufficient."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", "SOURCES:\n{formatted_context}\n\nUSER QUESTION: {query}")
        ])

        return prompt | self.structured_llm

    def generate(self, augmented_payload: dict) -> GenerationWithCitations:
        """Executes generation against the LLM using the augmented payload."""
        formatted_context = augmented_payload.get("formatted_context", "")
        query = augmented_payload.get("query", "")

        if not formatted_context:
            return GenerationWithCitations(
                answer="I cannot answer this question because no relevant context was found.",
                citations=[]
            )

        result = self.chain.invoke({
            "formatted_context": formatted_context,
            "query": query
        })

        return result