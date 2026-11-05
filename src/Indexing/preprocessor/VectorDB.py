import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# Automatically load environment variables from .env file
load_dotenv()

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

# Lazy imports for vector stores
try:
    from langchain_chroma import Chroma
except ImportError:
    Chroma = None

try:
    from langchain_pinecone import PineconeVectorStore
except ImportError:
    PineconeVectorStore = None

# Lazy import for Hugging Face API Embeddings
try:
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings as HuggingFaceEndpointEmbeddings
    except ImportError:
        HuggingFaceEndpointEmbeddings = None


class VectorDB:
    """Unified Vector Database abstraction supporting local ChromaDB by default,
    or Pinecone when credentials are provided. Automatically uses Hugging Face 
    API embeddings if no embedding_function is supplied.
    """

    def __init__(
        self,
        embedding_function: Optional[Embeddings] = None,
        collection_name: str = "rag_collection",
        persist_directory: str = "./chroma_db",
        pinecone_api_key: Optional[str] = None,
        pinecone_index_name: Optional[str] = None,
        hf_api_token: Optional[str] = None,
        hf_model_name: str = "BAAI/bge-m3",
        **kwargs: Any,
    ):
        """Parameters:
        - embedding_function: Embeddings instance. If None, initializes HuggingFace API embeddings using .env token.
        - collection_name: Collection / namespace name for Chroma or Pinecone.
        - persist_directory: Path to store local Chroma DB files.
        - pinecone_api_key: Pinecone API Key (Falls back to PINECONE_API_KEY in .env).
        - pinecone_index_name: Pinecone index name (Triggers Pinecone usage if provided).
        - hf_api_token: Hugging Face API Token (Falls back to HUGGINGFACEHUB_API_TOKEN in .env).
        - hf_model_name: Default Hugging Face API embedding model ID.
        """
        # 1. Resolve or initialize the embedding function via Hugging Face API
        if embedding_function is None:
            token = hf_api_token or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
            
            if not token:
                raise ValueError(
                    "No `embedding_function` provided and no `HUGGINGFACEHUB_API_TOKEN` "
                    "or `HF_TOKEN` found in environment variables / .env file."
                )
            
            if HuggingFaceEndpointEmbeddings is None:
                raise ImportError(
                    "Hugging Face embeddings module missing. "
                    "Install with: `pip install langchain-huggingface`"
                )

            # Initialize Hugging Face API hosted embeddings
            self.embedding_function = HuggingFaceEndpointEmbeddings(
                model=hf_model_name,
                huggingfacehub_api_token=token
            )
        else:
            self.embedding_function = embedding_function

        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # 2. Resolve Pinecone API credentials
        effective_pinecone_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")

        # 3. Determine backend vector store (Pinecone vs Chroma)
        if effective_pinecone_key and pinecone_index_name:
            if PineconeVectorStore is None:
                raise ImportError(
                    "langchain-pinecone is missing. Install with: "
                    "`pip install langchain-pinecone pinecone-client`"
                )

            self.db_type = "pinecone"
            self.vector_store = PineconeVectorStore(
                index_name=pinecone_index_name,
                embedding=self.embedding_function,
                pinecone_api_key=effective_pinecone_key,
                namespace=collection_name,
                **kwargs,
            )
        else:
            if Chroma is None:
                raise ImportError(
                    "langchain-chroma is missing. Install with: `pip install langchain-chroma`"
                )

            self.db_type = "chroma"
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_function,
                persist_directory=persist_directory,
                **kwargs,
            )

    def add_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Adds LangChain Document objects to the active vector database."""
        if not documents:
            return []
        return self.vector_store.add_documents(documents, **kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Performs similarity search using query string."""
        return self.vector_store.similarity_search(
            query=query, k=k, filter=filter, **kwargs
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[tuple[Document, float]]:
        """Performs similarity search and returns documents alongside distance scores."""
        return self.vector_store.similarity_search_with_score(
            query=query, k=k, filter=filter, **kwargs
        )

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        """Converts the active vector database into a LangChain retriever for RAG chains."""
        return self.vector_store.as_retriever(**kwargs)