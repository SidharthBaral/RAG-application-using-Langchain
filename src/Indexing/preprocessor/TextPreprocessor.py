import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import emoji
except ImportError:
    emoji = None

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from Indexing.data_extraction.BaseDataExtractor import BaseDataExtractor


class TextPreprocessor:
    """Preprocesses raw extracted text with safe cleaning and semantic boundary chunking."""

    def __init__(
        self, 
        embeddings: Optional[Embeddings] = None,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: Optional[float] = None
    ):
        """
        Parameters:
        - embeddings: Vector model used to calculate semantic distances. Defaults to BAAI/bge-m3.
        - breakpoint_threshold_type: Method for calculating split points ('percentile', 'standard_deviation', 'interquartile').
        - breakpoint_threshold_amount: Custom threshold value (e.g. 90.0 for percentile).
        """
        # 1. Default to a local multilingual embedding model if none is provided
        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                encode_kwargs={"normalize_embeddings": True}
            )
        
        self.embeddings = embeddings
        
        # 2. Configure the Semantic Chunker
        chunker_kwargs = {"breakpoint_threshold_type": breakpoint_threshold_type}
        if breakpoint_threshold_amount is not None:
            chunker_kwargs["breakpoint_threshold_amount"] = breakpoint_threshold_amount

        self.splitter = SemanticChunker(
            embeddings=self.embeddings,
            **chunker_kwargs
        )

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        # 1. Normalize non-breaking spaces and unicode artifacts
        text = text.replace("\xa0", " ")

        # 2. Strip emojis if library is available
        if emoji:
            text = emoji.replace_emoji(text, replace="")

        # 3. Generic Sound Cue Removal (Any Language)
        text = re.sub(r"\[(?![^\]]*\]\()[^\]]{1,80}\]", "", text)

        # 4. Strip standalone music symbols (♪, ♫, ♬)
        text = re.sub(r"[♪♫♬]+", "", text)

        # 5. Standardize quotes
        text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')

        # 6. Normalize whitespace while preserving double newlines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        
        return text.strip()

    def process(
        self, 
        extractor: BaseDataExtractor, 
        source: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        # 1. Extract raw text
        raw_text = extractor.extract(source)
        
        # 2. Apply structure-preserving cleanup
        cleaned_text = self.clean_text(raw_text)
        
        if not cleaned_text:
            return []

        # 3. Split into semantically coherent chunks based on sentence similarity
        raw_chunks = self.splitter.split_text(cleaned_text)
        
        processed_docs = []
        base_meta = metadata or {}
        
        # 4. Filter quality and package into standard LangChain Document objects
        for i, chunk in enumerate(raw_chunks):
            # Drop low-information chunks
            if len(chunk.strip()) < 20:
                continue
            
            chunk_metadata = base_meta.copy()
            chunk_metadata.update({
                "chunk_index": i, 
                "source": source
            })
            
            processed_docs.append(
                Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
            )
            
        return processed_docs