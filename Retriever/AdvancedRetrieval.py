from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder


class AdvancedRetriever:
    """Combines Dense Vector Search with MMR (Maximal Marginal Relevance),
    Sparse Keyword Search (BM25), Hybrid Ensemble Fusion, and Cross-Encoder Reranking.
    """

    def __init__(
        self,
        vector_db,
        documents: List[Document],
        reranker_model_name: str = "BAAI/bge-reranker-base",
    ):
        """Parameters:
        - vector_db: Instance of VectorDB class (ChromaDB wrapper)
        - documents: Original preprocessed document chunks required to train BM25
        - reranker_model_name: Multilingual cross-encoder for re-scoring candidates
        """
        self.vector_db = vector_db

        # 1. Initialize Sparse BM25 Retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 5

        # 2. Initialize Dense MMR Vector Retriever (ensures diversity)
        self.dense_retriever = self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7},
        )

        # 3. Hybrid Ensemble Retriever (50% Dense + 50% Sparse weight balance)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.dense_retriever],
            weights=[0.5, 0.5],
        )

        # 4. Initialize Cross-Encoder Reranker
        print(f"Loading Cross-Encoder Reranker: {reranker_model_name}...")
        self.reranker = CrossEncoder(reranker_model_name)

    def retrieve_and_rerank(
        self, queries: List[str], top_n: int = 5
    ) -> List[Document]:
        """Retrieves candidates across all query variations using Hybrid Search,

        deduplicates them, and scores them using Cross-Encoder Reranking.
        """
        # Step A: Execute Hybrid Retrieval across generated multi-queries
        candidate_docs: List[Document] = []
        seen_contents = set()

        for q in queries:
            docs = self.ensemble_retriever.invoke(f"query: {q}")
            for doc in docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    candidate_docs.append(doc)

        if not candidate_docs:
            return []

        # Step B: Prepare Query-Document pairs for Cross-Encoder
        # The primary (rewritten) query is used for precise scoring
        primary_query = queries[0]
        pairs = [[primary_query, doc.page_content] for doc in candidate_docs]

        # Step C: Compute relevance scores using Cross-Encoder
        scores = self.reranker.predict(pairs)

        # Step D: Pair documents with scores and sort descending
        doc_score_pairs = list(zip(candidate_docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Step E: Slice top_n highest-ranked documents
        reranked_docs = [doc for doc, score in doc_score_pairs[:top_n]]
        return reranked_docs