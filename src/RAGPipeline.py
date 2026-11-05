import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
from langchain_core.documents import Document

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from Indexing.preprocessor.VectorDB import VectorDB
from Augmentation_Generation.RAGGuardrails import RAGGuardrails
from Retriever.PreRetrieval import PreRetrievalPipeline
from Retriever.AdvancedRetrieval import AdvancedRetriever
from Retriever.PostRetrieval import ContextualCompressor
from Augmentation_Generation.Augmentation import AugmentationPipeline
from Augmentation_Generation.Generator import GenerationPipeline, CitationReference
from Memory.SQLiteMemory import SQLiteChatMemory


class Pipeline:
    """End-to-End Query Execution Pipeline binding all runtime stages of the Advanced RAG system.

    Features:
    - Persistent SQLite multi-turn chat memory
    - History-aware query condensation (coreference resolution)
    - Consolidated Input Guardrail + Pre-Retrieval in 1 LLM call
    - Async Parallel Hybrid Retrieval (BM25 + Dense MMR) with Cross-Encoder Reranking
    - Batched Post-Retrieval Context Compression
    - Lost-in-the-Middle context augmentation
    - Dual-path generation: One-shot (.run()) & Real-time streaming (.prepare_context() + .stream_answer())
    - Post-generation Output Faithfulness Guardrail
    """

    def __init__(
        self,
        llm,
        vector_db: VectorDB,
        indexed_documents: List[Document],
        db_path: str = "conversations.db",
        history_window: int = 6,
        enable_input_guardrail: bool = True,
        enable_output_guardrail: bool = True,
        enable_pre_retrieval: bool = True,
        enable_compression: bool = True,
        enable_lost_in_middle: bool = True,
        max_context_chars: int = 4000,
        top_n_rerank: int = 4,
        reranker_model_name: str = "BAAI/bge-reranker-base",
    ):
        self.llm = llm
        self.vector_db = vector_db
        self.indexed_documents = indexed_documents
        self.history_window = history_window
        self.enable_input_guardrail = enable_input_guardrail
        self.enable_output_guardrail = enable_output_guardrail
        self.enable_pre_retrieval = enable_pre_retrieval
        self.enable_compression = enable_compression
        self.top_n_rerank = top_n_rerank

        # 1. SQLite Chat Memory Manager
        self.memory = SQLiteChatMemory(db_path=db_path)

        # 2. Consolidated PreRetrieval (Safety Audit + History Condensation + Multi-Query)
        self.pre_retriever = PreRetrievalPipeline(llm=self.llm)
        self.guardrails = RAGGuardrails(llm=self.llm)

        # 3. Hybrid Parallel Retriever & Reranker
        self.advanced_retriever = AdvancedRetriever(
            vector_db=self.vector_db,
            documents=self.indexed_documents,
            reranker_model_name=reranker_model_name,
        )

        # 4. Context Compression, Augmentation & Generation Pipelines
        self.compressor = ContextualCompressor(llm=self.llm)
        self.augmentor = AugmentationPipeline(
            max_context_chars=max_context_chars,
            enable_lost_in_middle=enable_lost_in_middle,
        )
        self.generator = GenerationPipeline(llm=self.llm)

    # =========================================================================
    # SHARED CONTEXT PREPARATION (Stages 1 through 5)
    # =========================================================================
    def prepare_context(
        self, raw_query: str, session_id: str = "default_session"
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """Executes Stages 1 through 5:

        1. Fetches recent conversation history from SQLite.
        2. Runs Pre-Retrieval (Input Guardrail + History-aware coreference resolution).
        3. Parallel Hybrid Retrieval (BM25 + Dense MMR) with Cross-Encoder reranking.
        4. Batched Contextual Compression.
        5. Context window budgeting & Lost-in-the-Middle assembly.

        Returns: (is_safe, augmented_payload, pipeline_log)
        """
        pipeline_log = {}

        # ----------------------------------------------------
        # STAGE 0: RETRIEVE SESSION HISTORY FROM SQLITE
        # ----------------------------------------------------
        history_str = self.memory.get_formatted_history(
            session_id=session_id, limit=self.history_window
        )
        pipeline_log["retrieved_history"] = history_str

        # ----------------------------------------------------
        # STAGES 1 & 2: CONSOLIDATED GUARDRAIL + HISTORY-AWARE PRE-RETRIEVAL
        # ----------------------------------------------------
        if self.enable_pre_retrieval:
            pre_result = self.pre_retriever.run(query=raw_query, history=history_str)
            pipeline_log["pre_retrieval"] = pre_result
            pipeline_log["input_guardrail"] = getattr(pre_result, "safety", None)

            # Block immediately if input guardrail triggers
            if self.enable_input_guardrail and hasattr(pre_result, "safety") and not pre_result.safety.is_safe:
                return False, {"error": f"❌ Query rejected: {pre_result.safety.reason}"}, pipeline_log

            search_query = pre_result.rewritten_query
            queries_to_search = [pre_result.rewritten_query] + pre_result.multi_queries
        else:
            search_query = raw_query
            queries_to_search = [raw_query]
            pipeline_log["pre_retrieval"] = None
            pipeline_log["input_guardrail"] = None

        # ----------------------------------------------------
        # STAGE 3: ASYNC PARALLEL HYBRID RETRIEVAL & RERANKING
        # ----------------------------------------------------
        reranked_docs = self.advanced_retriever.retrieve_and_rerank(
            queries=queries_to_search,
            top_n=self.top_n_rerank,
        )
        pipeline_log["retrieved_documents"] = reranked_docs

        # ----------------------------------------------------
        # STAGE 4: POST-RETRIEVAL CONTEXTUAL COMPRESSION
        # ----------------------------------------------------
        if self.enable_compression and reranked_docs:
            compressed_docs = self.compressor.compress_documents(
                query=search_query,
                documents=reranked_docs,
            )
        else:
            compressed_docs = reranked_docs
        pipeline_log["compressed_documents"] = compressed_docs

        # ----------------------------------------------------
        # STAGE 5: AUGMENTATION (Payload Assembly & Lost-In-The-Middle)
        # ----------------------------------------------------
        augmented_payload = self.augmentor.augment(
            query=search_query,
            documents=compressed_docs,
        )
        pipeline_log["augmented_payload"] = augmented_payload

        return True, augmented_payload, pipeline_log

    # =========================================================================
    # OPTION A: ONE-SHOT SYNCHRONOUS EXECUTION
    # =========================================================================
    def run(self, raw_query: str, session_id: str = "default_session") -> Dict[str, Any]:
        """Executes full 7-stage pipeline synchronously and records turn to SQLite."""
        # 1. Reuse Stages 1 to 5
        is_safe, payload, logs = self.prepare_context(raw_query=raw_query, session_id=session_id)
        if not is_safe:
            return {
                "success": False,
                "error_stage": "Input Guardrail",
                "answer": payload.get("error", "Input query was rejected."),
                "citations": [],
                "logs": logs,
            }

        # 2. Stage 6: Grounded Generation with Structured Citations
        gen_response = self.generator.generate(payload)
        logs["generation"] = gen_response

        # 3. Stage 7: Output Guardrail (Faithfulness Verification)
        final_answer = gen_response.answer
        if self.enable_output_guardrail and payload.get("formatted_context"):
            output_check = self.guardrails.validate_output(
                context=payload["formatted_context"],
                answer=gen_response.answer,
            )
            logs["output_guardrail"] = output_check

            if not output_check.is_grounded:
                final_answer = (
                    "⚠️ Response modified by Output Guardrail: "
                    "The generated answer contains details unverified by the context."
                )

        # 4. Save interaction turn to SQLite
        self.save_turn(
            session_id=session_id,
            user_query=raw_query,
            assistant_answer=final_answer,
        )

        return {
            "success": True,
            "answer": final_answer,
            "citations": gen_response.citations,
            "logs": logs,
        }

    # =========================================================================
    # OPTION B: STREAMING HELPERS (For st.write_stream in Streamlit)
    # =========================================================================
    def stream_answer(self, augmented_payload: Dict[str, Any]) -> Iterator[str]:
        """Yields streaming tokens for real-time rendering."""
        return self.generator.stream_generate(augmented_payload)

    def audit_output(self, context: str, answer: str):
        """Audits generated streamed text for factual grounding."""
        if not self.enable_output_guardrail or not context:
            return None
        return self.guardrails.validate_output(context=context, answer=answer)

    # =========================================================================
    # SQLITE MEMORY CONVENIENCE METHODS
    # =========================================================================
    def save_turn(self, session_id: str, user_query: str, assistant_answer: str):
        """Persists a completed conversation turn to SQLite."""
        self.memory.add_message(session_id=session_id, role="user", message=user_query)
        self.memory.add_message(session_id=session_id, role="assistant", message=assistant_answer)

    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Retrieves raw past messages for UI display."""
        return self.memory.get_messages(session_id=session_id, limit=limit or self.history_window)

    def clear_session(self, session_id: str):
        """Clears memory for a given session."""
        self.memory.clear_session(session_id=session_id)