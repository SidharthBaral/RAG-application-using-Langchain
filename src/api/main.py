import sys
import json
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from langchain_core.documents import Document

# Ensure project root is available in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from Models.ResponseGeneration import TextGeneration
from Models.EmbeddingModel import MultilingualEmbeddingGenerator
from Indexing.data_extraction.DataExtractorFactory import DataExtractorFactory
from Indexing.preprocessor.TextPreprocessor import TextPreprocessor
from Indexing.preprocessor.VectorDB import VectorDB
from RAGPipeline import Pipeline
from Memory.SQLiteMemory import SQLiteChatMemory
from api.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    CitationModel,
    HistoryResponse,
    ChatMessageModel,
)


# ==============================================================================
# APPLICATION LIFECYCLE (Model & DB Preloading)
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models and persistent vector DB connections on startup."""
    print("🚀 Initializing RAG System Backend Models...")

    # 1. Initialize Active LLM
    app.state.llm = TextGeneration(
        provider="groq",
        model_name="openai/gpt-oss-120b",
        temperature=0.0,
    ).llm

    # 2. Initialize Embeddings Generator
    app.state.embedder_service = MultilingualEmbeddingGenerator(
        provider="huggingface",
        model_name="intfloat/multilingual-e5-small",
        device="cpu",
    )

    # 3. Initialize ChromaDB Vector Store
    app.state.vector_db = VectorDB(
        embedding_function=app.state.embedder_service.embedder,
        collection_name="youtube_video_transcripts",
        persist_directory="./my_chroma_data",
    )

    # 4. Initialize SQLite Memory Manager
    app.state.memory = SQLiteChatMemory(db_path="conversations.db")

    # In-memory document storage for BM25 hybrid search
    app.state.indexed_documents = []

    print("✅ RAG Backend Ready to Serve Requests.")
    yield
    print("🛑 Shutting down RAG Backend.")


# ==============================================================================
# FASTAPI APP SETUP
# ==============================================================================
app = FastAPI(
    title="Enterprise Multilingual RAG API",
    version="1.0.0",
    description="REST API for Hybrid RAG with Guardrails, Citations, and Multi-Turn SQLite Memory.",
    lifespan=lifespan,
)

# Enable CORS for external web frontends (React, Next.js, Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper: Build Pipeline instance per request
def _get_pipeline(req: QueryRequest) -> Pipeline:
    if not app.state.indexed_documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been indexed yet. Call /api/v1/ingest first.",
        )
    return Pipeline(
        llm=app.state.llm,
        vector_db=app.state.vector_db,
        indexed_documents=app.state.indexed_documents,
        db_path="conversations.db",
        enable_input_guardrail=req.enable_input_guardrail,
        enable_output_guardrail=req.enable_output_guardrail,
        enable_pre_retrieval=req.enable_pre_retrieval,
        enable_compression=req.enable_compression,
        enable_lost_in_middle=req.enable_lost_in_middle,
        max_context_chars=req.max_context_chars,
        top_n_rerank=req.top_n_rerank,
    )


# ==============================================================================
# API ENDPOINTS
# ==============================================================================
@app.get("/health", tags=["Health"])
async def health_check():
    """Health status and service availability."""
    return {
        "status": "healthy",
        "indexed_chunks": len(app.state.indexed_documents),
    }


@app.post("/api/v1/ingest", response_model=IngestResponse, tags=["Document Ingestion"])
async def ingest_document(req: IngestRequest):
    """Extracts, preprocesses, chunks, and indexes a source URL or file."""
    try:
        extractor = DataExtractorFactory.get_extractor(req.source_type)
        preprocessor = TextPreprocessor(embeddings=app.state.embedder_service.embedder)

        # Run extraction & chunking
        documents = preprocessor.process(extractor, source=req.source_url)
        if not documents:
            raise HTTPException(status_code=400, detail="Extractor yielded zero document chunks.")

        # Persist to Vector Store
        app.state.vector_db.add_documents(documents)
        app.state.indexed_documents.extend(documents)

        return IngestResponse(
            success=True,
            message=f"Successfully indexed document into vector storage.",
            total_chunks=len(documents),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/api/v1/query", response_model=QueryResponse, tags=["Query Execution"])
def query_rag(req: QueryRequest):
    """Synchronous one-shot execution returning grounded answer, citations, and metadata."""
    try:
        pipeline = _get_pipeline(req)
        result = pipeline.run(raw_query=req.query, session_id=req.session_id)

        logs = result.get("logs", {})
        pr = logs.get("pre_retrieval")
        ig = logs.get("input_guardrail")
        og = logs.get("output_guardrail")

        # Serialize structured citations
        citations_data = []
        for cit in result.get("citations", []):
            citations_data.append(
                CitationModel(
                    citation_id=getattr(cit, "citation_id", 0),
                    source=getattr(cit, "source", "Unknown"),
                    quoted_snippet=getattr(cit, "quoted_snippet", ""),
                )
            )

        return QueryResponse(
            success=result.get("success", False),
            session_id=req.session_id,
            answer=result.get("answer", ""),
            citations=citations_data,
            rewritten_query=getattr(pr, "rewritten_query", None) if pr else None,
            retrieved_chunk_count=len(logs.get("retrieved_documents", [])),
            guardrail_status={
                "input_safe": getattr(ig, "is_safe", True) if ig else True,
                "output_grounded": getattr(og, "is_grounded", True) if og else True,
            },
        )
    except Exception as e:
        import traceback
        traceback.print_exc()  # Prints the full stack trace in your terminal
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {str(e)}",
        )


@app.post("/api/v1/query/stream", tags=["Query Execution"])
async def stream_rag(req: QueryRequest):
    """Server-Sent Events (SSE) streaming endpoint delivering token-by-token output."""
    pipeline = _get_pipeline(req)

    # 1. Run Pre-retrieval, search, and augmentation
    is_safe, payload, logs = pipeline.prepare_context(
        raw_query=req.query, session_id=req.session_id
    )

    if not is_safe:
        async def rejection_generator():
            yield f"data: {json.dumps({'type': 'error', 'message': payload.get('error')})}\n\n"
        return StreamingResponse(rejection_generator(), media_type="text/event-stream")

    async def event_generator():
        collected_tokens = []
        try:
            # 2. Stream answer tokens
            for token in pipeline.stream_answer(payload):
                collected_tokens.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                await asyncio.sleep(0.01)  # Yield control to event loop

            final_answer = "".join(collected_tokens)

            # 3. Persist completed turn to SQLite
            pipeline.save_turn(
                session_id=req.session_id,
                user_query=req.query,
                assistant_answer=final_answer,
            )

            # 4. Post-generation Output Faithfulness Guardrail
            guardrail_passed = True
            if req.enable_output_guardrail and payload.get("formatted_context"):
                check = pipeline.audit_output(
                    context=payload["formatted_context"], answer=final_answer
                )
                if check and not check.is_grounded:
                    guardrail_passed = False
                    yield f"data: {json.dumps({'type': 'guardrail_alert', 'claims': check.hallucinated_claims})}\n\n"

            # 5. Emit extracted citations
            citations = []
            if hasattr(pipeline.generator, "extract_citations_from_stream"):
                citations = pipeline.generator.extract_citations_from_stream(final_answer, payload)

            serialized_citations = [
                {
                    "citation_id": getattr(c, "citation_id", 0),
                    "source": getattr(c, "source", ""),
                    "quoted_snippet": getattr(c, "quoted_snippet", ""),
                }
                for c in citations
            ]

            yield f"data: {json.dumps({'type': 'metadata', 'citations': serialized_citations, 'guardrail_passed': guardrail_passed})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/v1/history/{session_id}", response_model=HistoryResponse, tags=["Conversation Memory"])
async def get_chat_history(session_id: str, limit: int = 10):
    """Fetches chronological chat history for a session."""
    messages = app.state.memory.get_messages(session_id=session_id, limit=limit)
    return HistoryResponse(
        session_id=session_id,
        messages=[ChatMessageModel(role=m["role"], message=m["message"]) for m in messages],
    )


@app.delete("/api/v1/history/{session_id}", tags=["Conversation Memory"])
async def delete_chat_history(session_id: str):
    """Clears SQLite history for a session."""
    app.state.memory.clear_session(session_id=session_id)
    return {"status": "success", "message": f"History deleted for session '{session_id}'."}