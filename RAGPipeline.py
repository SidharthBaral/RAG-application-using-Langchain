import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
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
from Augmentation_Generation.CitationGenerator import GenerationPipeline


class Pipeline:
    """End-to-End Query Execution Pipeline binding all runtime stages of the Advanced RAG system."""

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

        # Initialize Query Pipeline Sub-components
        self.guardrails = RAGGuardrails(llm=self.llm)
        self.pre_retriever = PreRetrievalPipeline(llm=self.llm)
        
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

    def run(self, raw_query: str) -> Dict[str, Any]:
        """Executes the complete query-time RAG workflow."""
        pipeline_log = {}

        # ----------------------------------------------------
        # STAGE 1: INPUT GUARDRAIL
        # ----------------------------------------------------
        if self.enable_input_guardrail:
            input_check = self.guardrails.validate_input(raw_query)
            pipeline_log["input_guardrail"] = input_check
            if not input_check.is_safe:
                return {
                    "success": False,
                    "error_stage": "Input Guardrail",
                    "answer": f"❌ Query rejected: {input_check.reason}",
                    "citations": [],
                    "logs": pipeline_log,
                }

        # ----------------------------------------------------
        # STAGE 2: PRE-RETRIEVAL (Rewriting & Multi-Query Expansion)
        # ----------------------------------------------------
        if self.enable_pre_retrieval:
            pre_result = self.pre_retriever.run(query=raw_query)
            queries_to_search = [pre_result.rewritten_query] + pre_result.multi_queries
            pipeline_log["pre_retrieval"] = pre_result
        else:
            queries_to_search = [raw_query]
            pipeline_log["pre_retrieval"] = None

        # ----------------------------------------------------
        # STAGE 3: ADVANCED HYBRID RETRIEVAL & CROSS-ENCODER RERANKING
        # ----------------------------------------------------
        reranked_docs = self.advanced_retriever.retrieve_and_rerank(
            queries=queries_to_search, 
            top_n=self.top_n_rerank
        )
        pipeline_log["retrieved_documents"] = reranked_docs

        # ----------------------------------------------------
        # STAGE 4: POST-RETRIEVAL CONTEXTUAL COMPRESSION
        # ----------------------------------------------------
        search_query = queries_to_search[0]
        if self.enable_compression and reranked_docs:
            compressed_docs = self.compressor.compress_documents(
                query=search_query, 
                documents=reranked_docs
            )
        else:
            compressed_docs = reranked_docs

        pipeline_log["compressed_documents"] = compressed_docs

        # ----------------------------------------------------
        # STAGE 5: AUGMENTATION (Context Optimization & Payload Assembly)
        # ----------------------------------------------------
        augmented_payload = self.augmentor.augment(
            query=search_query, 
            documents=compressed_docs
        )
        pipeline_log["augmented_payload"] = augmented_payload

        # ----------------------------------------------------
        # STAGE 6: GROUNDED GENERATION WITH CITATIONS
        # ----------------------------------------------------
        gen_response = self.generator.generate(augmented_payload)
        pipeline_log["generation"] = gen_response

        # ----------------------------------------------------
        # STAGE 7: OUTPUT GUARDRAIL (Faithfulness Audit)
        # ----------------------------------------------------
        final_answer = gen_response.answer
        if self.enable_output_guardrail and augmented_payload["formatted_context"]:
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