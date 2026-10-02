from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document


class AugmentationPipeline:
    """Pure Augmentation Component: Responsible for context reordering, 

    character budgeting, source tag formatting, and payload assembly.
    """

    def __init__(
        self,
        max_context_chars: int = 6000,
        enable_lost_in_middle: bool = True
    ):
        """Parameters:

        - max_context_chars: Character limit budget for the context block.
        - enable_lost_in_middle: Re-orders documents so top relevance hits 
          appear at the start and end of the context payload.
        """
        self.max_context_chars = max_context_chars
        self.enable_lost_in_middle = enable_lost_in_middle

    def _reorder_lost_in_middle(self, documents: List[Document]) -> List[Document]:
        """Mitigates LLM attention degradation by placing high-rank chunks 

        at the top and bottom of the context sequence.
        """
        if len(documents) <= 2 or not self.enable_lost_in_middle:
            return documents

        reordered = [None] * len(documents)
        left, right = 0, len(documents) - 1

        for i, doc in enumerate(documents):
            if i % 2 == 0:
                reordered[left] = doc
                left += 1
            else:
                reordered[right] = doc
                right -= 1

        return reordered

    def augment(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """Prepares and returns an augmented payload containing the query,

        formatted context, and utilized documents.
        """
        if not documents:
            return {
                "query": query,
                "formatted_context": "",
                "used_documents": []
            }

        # Step 1: Re-order context using Lost-in-the-Middle strategy
        optimized_docs = self._reorder_lost_in_middle(documents)

        # Step 2: Apply character budgeting and source tagging
        formatted_sources = []
        used_docs = []
        current_length = 0

        for idx, doc in enumerate(optimized_docs, 1):
            source_tag = doc.metadata.get("source", f"Document_{idx}")
            chunk_str = f"[Source {idx}] (Path/URL: {source_tag})\nContent: {doc.page_content.strip()}\n"

            if current_length + len(chunk_str) > self.max_context_chars:
                break

            formatted_sources.append(chunk_str)
            current_length += len(chunk_str)
            used_docs.append(doc)

        formatted_context = "\n\n".join(formatted_sources)

        # Step 3: Return clean Augmented Payload
        return {
            "query": query,
            "formatted_context": formatted_context,
            "used_documents": used_docs
        }