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


class Pipeline:
    """End-to-End Query Execution Pipeline binding all runtime stages of the Advanced RAG system.
    Supports both full-pass execution (.run()) and real-time streaming (.prepare_context() + .stream_answer()).
    """

    def __init__(
        self,
        llm,
        vector_db: VectorDB,
        indexed_documents: List[Document],
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
        self.enable_input_guardrail = enable_input_guardrail
        self.enable_output_guardrail = enable_output_guardrail
        self.enable_pre_retrieval = enable_pre_retrieval
        self.enable_compression = enable_compression

        # Consolidated single-call PreRetrieval + Input Guardrail
        self.pre_retriever = PreRetrievalPipeline(llm=self.llm)
        self.guardrails = RAGGuardrails(llm=self.llm)
        
        # Async-capable Hybrid Retriever
        self.advanced_retriever = AdvancedRetriever(
            vector_db=self.vector_db,
            documents=self.indexed_documents,
            reranker_model_name=reranker_model_name,
        )
        self.compressor = ContextualCompressor(llm=self.llm)
        self.augmentor = AugmentationPipeline(
            max_context_chars=max_context_chars,
            enable_lost_in_middle=enable_lost_in_middle,
        )
        self.generator = GenerationPipeline(llm=self.llm)
        self.top_n_rerank = top_n_rerank

    # =========================================================================
    # OPTION A: ONE-SHOT EXECUTION (Matches your original code exactly)
    # =========================================================================
    def run(self, raw_query: str) -> Dict[str, Any]:
        """Executes the complete query-time RAG workflow synchronously."""
        pipeline_log = {}

        # STAGES 1 & 2: CONSOLIDATED GUARDRAIL + PRE-RETRIEVAL (1 LLM CALL)
        if self.enable_pre_retrieval:
            pre_result = self.pre_retriever.run(query=raw_query)
            pipeline_log["pre_retrieval"] = pre_result
            pipeline_log["input_guardrail"] = getattr(pre_result, "safety", None)

            if self.enable_input_guardrail and hasattr(pre_result, "safety") and not pre_result.safety.is_safe:
                return {
                    "success": False,
                    "error_stage": "Input Guardrail",
                    "answer": f"❌ Query rejected: {pre_result.safety.reason}",
                    "citations": [],
                    "logs": pipeline_log,
                }

            search_query = pre_result.rewritten_query
            queries_to_search = [pre_result.rewritten_query] + pre_result.multi_queries
        else:
            search_query = raw_query
            queries_to_search = [raw_query]
            pipeline_log["pre_retrieval"] = None
            pipeline_log["input_guardrail"] = None

        # STAGE 3: ASYNC PARALLEL HYBRID RETRIEVAL & CROSS-ENCODER RERANKING
        reranked_docs = self.advanced_retriever.retrieve_and_rerank(
            queries=queries_to_search, 
            top_n=self.top_n_rerank
        )
        pipeline_log["retrieved_documents"] = reranked_docs

        # STAGE 4: POST-RETRIEVAL CONTEXTUAL COMPRESSION
        if self.enable_compression and reranked_docs:
            compressed_docs = self.compressor.compress_documents(
                query=search_query, 
                documents=reranked_docs
            )
        else:
            compressed_docs = reranked_docs
        pipeline_log["compressed_documents"] = compressed_docs

        # STAGE 5: AUGMENTATION
        augmented_payload = self.augmentor.augment(
            query=search_query, 
            documents=compressed_docs
        )
        pipeline_log["augmented_payload"] = augmented_payload

        # STAGE 6: GROUNDED GENERATION WITH STRUCTURED CITATIONS
        gen_response = self.generator.generate(augmented_payload)
        pipeline_log["generation"] = gen_response

        # STAGE 7: OUTPUT GUARDRAIL (Faithfulness Audit)
        final_answer = gen_response.answer
        if self.enable_output_guardrail and augmented_payload.get("formatted_context"):
            output_check = self.guardrails.validate_output(
                context=augmented_payload["formatted_context"],
                answer=gen_response.answer,
            )
            pipeline_log["output_guardrail"] = output_check

            if not output_check.is_grounded:
                final_answer = (
                    "⚠️ Response modified by Output Guardrail: "
                    "The generated answer contains details unverified by the context."
                )

        return {
            "success": True,
            "answer": final_answer,
            "citations": gen_response.citations,
            "logs": pipeline_log,
        }

    # =========================================================================
    # OPTION B: STREAMING EXECUTION (For st.write_stream in Streamlit)
    # =========================================================================
    def prepare_context(self, raw_query: str) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """Executes Stages 1 through 5 up to context augmentation without generating text yet."""
        pipeline_log = {}

        if self.enable_pre_retrieval:
            pre_result = self.pre_retriever.run(query=raw_query)
            pipeline_log["pre_retrieval"] = pre_result
            pipeline_log["input_guardrail"] = getattr(pre_result, "safety", None)

            if self.enable_input_guardrail and hasattr(pre_result, "safety") and not pre_result.safety.is_safe:
                return False, {"error": f"❌ Query rejected: {pre_result.safety.reason}"}, pipeline_log

            search_query = pre_result.rewritten_query
            queries_to_search = [pre_result.rewritten_query] + pre_result.multi_queries
        else:
            search_query = raw_query
            queries_to_search = [raw_query]
            pipeline_log["pre_retrieval"] = None
            pipeline_log["input_guardrail"] = None

        reranked_docs = self.advanced_retriever.retrieve_and_rerank(
            queries=queries_to_search, 
            top_n=self.top_n_rerank
        )
        pipeline_log["retrieved_documents"] = reranked_docs

        if self.enable_compression and reranked_docs:
            compressed_docs = self.compressor.compress_documents(
                query=search_query, 
                documents=reranked_docs
            )
        else:
            compressed_docs = reranked_docs
        pipeline_log["compressed_documents"] = compressed_docs

        augmented_payload = self.augmentor.augment(
            query=search_query, 
            documents=compressed_docs
        )
        pipeline_log["augmented_payload"] = augmented_payload

        return True, augmented_payload, pipeline_log

    def stream_answer(self, augmented_payload: Dict[str, Any]) -> Iterator[str]:
        """Yields streaming tokens for real-time rendering."""
        return self.generator.stream_generate(augmented_payload)

    def audit_output(self, context: str, answer: str):
        """Audits generated streamed text for factual grounding."""
        if not self.enable_output_guardrail or not context:
            return None
        return self.guardrails.validate_output(context=context, answer=answer)