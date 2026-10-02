from typing import List
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate


class ContextualCompressor:
    """Post-retrieval pipeline that compresses retrieved document chunks by extracting

    strictly query-relevant sentences and stripping out irrelevant text and noise.
    """

    def __init__(self, llm: BaseChatModel):
        """Parameters:
        - llm: LangChain ChatModel instance used for contextual sentence extraction.
        """
        self.llm = llm
        self._chain = self._build_compression_chain()

    def _build_compression_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a contextual compressor for a high-precision RAG system.\n"
                    "Given a user query and a context passage, extract ONLY the exact sentences "
                    "or key facts that directly help answer the query.\n"
                    "Do NOT summarize, rephrase extensively, alter facts, or introduce outside knowledge.\n"
                    "If no information in the passage is relevant to the query, respond strictly with 'NO_RELEVANT_INFO'.",
                ),
                ("human", "Query: {query}\n\nPassage:\n{passage}"),
            ]
        )
        return prompt | self.llm

    def compress_documents(
        self, query: str, documents: List[Document]
    ) -> List[Document]:
        """Compresses each document chunk individually to retain strictly relevant context."""
        compressed_docs: List[Document] = []

        for doc in documents:
            response = self._chain.invoke(
                {"query": query, "passage": doc.page_content}
            )

            compressed_text = response.content.strip()

            # Filter out chunks containing no relevant information
            if compressed_text and "NO_RELEVANT_INFO" not in compressed_text:
                new_doc = Document(
                    page_content=compressed_text, metadata=doc.metadata
                )
                compressed_docs.append(new_doc)

        return compressed_docs