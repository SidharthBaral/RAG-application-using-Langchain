from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# --- Ingestion Schemas ---
class IngestRequest(BaseModel):
    source_type: str = Field(default="YouTube Transcript", description="Data extractor type (e.g., 'YouTube Transcript', 'Web Article')")
    source_url: str = Field(..., description="Target URL or local filepath to extract")


class IngestResponse(BaseModel):
    success: bool
    message: str
    total_chunks: int


# --- Citation & Query Schemas ---
class CitationModel(BaseModel):
    citation_id: int
    source: str
    quoted_snippet: str


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question or follow-up")
    session_id: str = Field(default="default_session", description="Conversation session ID")
    enable_input_guardrail: bool = True
    enable_output_guardrail: bool = True
    enable_pre_retrieval: bool = True
    enable_compression: bool = True
    enable_lost_in_middle: bool = True
    top_n_rerank: int = 4
    max_context_chars: int = 4000


class QueryResponse(BaseModel):
    success: bool
    session_id: str
    answer: str
    citations: List[CitationModel] = Field(default_factory=list)
    rewritten_query: Optional[str] = None
    retrieved_chunk_count: int = 0
    guardrail_status: Dict[str, Any] = Field(default_factory=dict)


# --- History Schemas ---
class ChatMessageModel(BaseModel):
    role: str
    message: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessageModel]