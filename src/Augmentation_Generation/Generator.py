import re
from typing import List, Iterator, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate


class CitationReference(BaseModel):
    citation_id: int = Field(description="Numerical ID of the citation, e.g. 1, 2")
    source: str = Field(description="Source identifier, URL, or document title")
    quoted_snippet: str = Field(description="Exact snippet or key sentence from context")


class GenerationWithCitations(BaseModel):
    answer: str = Field(description="Answer containing inline citations like [1], [2].")
    citations: List[CitationReference] = Field(default_factory=list)


class GenerationPipeline:
    """Unified Generation Pipeline supporting both Pydantic Structured Output 
    and Real-Time Token Streaming with deterministic citation resolution.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        
        # 1. Plain prompt for streaming
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a factual assistant for a RAG system.\n"
                "STRICT GROUNDING RULES:\n"
                "1. Answer using ONLY facts from the SOURCES list.\n"
                "2. Place inline markers like [1], [2] immediately after each stated fact.\n"
                "3. If facts draw from multiple sources, use multiple markers like [1][2].\n"
                "4. If context is insufficient, state that context is insufficient."
            ),
            ("human", "SOURCES:\n{formatted_context}\n\nUSER QUESTION: {query}")
        ])
        
        # 2. Structured chain (Your original logic)
        self.structured_chain = self.prompt | self.llm.with_structured_output(GenerationWithCitations)
        
        # 3. Streaming chain
        self.streaming_chain = self.prompt | self.llm

    def generate(self, augmented_payload: dict) -> GenerationWithCitations:
        """Executes full structured generation (Original non-streaming behavior)."""
        formatted_context = augmented_payload.get("formatted_context", "")
        query = augmented_payload.get("query", "")

        if not formatted_context:
            return GenerationWithCitations(
                answer="I cannot answer this question because no relevant context was found.",
                citations=[]
            )

        return self.structured_chain.invoke({
            "formatted_context": formatted_context,
            "query": query
        })

    def stream_generate(self, augmented_payload: dict) -> Iterator[str]:
        """Streams text tokens progressively to st.write_stream."""
        formatted_context = augmented_payload.get("formatted_context", "")
        query = augmented_payload.get("query", "")

        if not formatted_context:
            yield "I cannot answer this question because no relevant context was found."
            return

        for chunk in self.streaming_chain.stream({
            "formatted_context": formatted_context,
            "query": query
        }):
            if chunk.content:
                yield chunk.content

    def extract_citations_from_stream(
        self, completed_answer: str, augmented_payload: dict
    ) -> List[CitationReference]:
        """Resolves structured citations from the completed streamed text using the retrieved documents."""
        used_docs = augmented_payload.get("used_documents", [])
        if not used_docs or not completed_answer:
            return []

        # Find all cited numbers in the streamed answer: e.g., [1], [2]
        cited_indices = set(map(int, re.findall(r"\[(\d+)\]", completed_answer)))
        
        citations = []
        for idx in sorted(cited_indices):
            # Context indices are 1-based ([1] corresponds to doc 0)
            if 1 <= idx <= len(used_docs):
                doc = used_docs[idx - 1]
                citations.append(
                    CitationReference(
                        citation_id=idx,
                        source=doc.metadata.get("source", f"Source {idx}"),
                        quoted_snippet=doc.page_content[:250].strip() + "..."
                    )
                )
        return citations