import os
import sys
from pathlib import Path
from typing import List, Union, Optional, Any
from dotenv import load_dotenv

# Automatically load environment variables from .env file
load_dotenv()

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


class MultilingualEmbeddingGenerator:
    """Generates multilingual vector embeddings using dynamic provider credentials and custom models."""

    def __init__(
        self, 
        provider: str = "huggingface_api", 
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        device: str = "cpu",
        **kwargs: Any
    ):
        """
        Parameters:
        - provider: Embedding provider ('huggingface_api', 'huggingface' [local], 'openai', 'cohere').
        - model_name: Custom model ID (e.g., 'intfloat/multilingual-e5-small', 'BAAI/bge-m3').
        - api_key: Explicit API key for cloud providers. Automatically falls back to .env variables if None.
        - device: Target compute device for local models ('cpu', 'cuda', 'mps').
        - **kwargs: Extra arguments passed directly to the underlying LangChain embedding class.
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.device = device
        self.kwargs = kwargs

        self.embedder: Embeddings = self._initialize_embedder()

    def _initialize_embedder(self) -> Embeddings:
        # Option 1: Hugging Face Hosted Inference API (Cloud via langchain_community)
        if self.provider in ["huggingface_api", "hf_api", "huggingface-api"]:
            selected_model = self.model_name or "intfloat/multilingual-e5-small"
            token = (
                self.api_key 
                or os.getenv("HUGGINGFACEHUB_API_TOKEN") 
                or os.getenv("HF_TOKEN")
            )
            
            if not token:
                raise ValueError(
                    "Hugging Face API token not found. Please set `HUGGINGFACEHUB_API_TOKEN` "
                    "or `HF_TOKEN` in your .env file or pass `api_key`."
                )

            try:
                from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
            except ImportError:
                raise ImportError(
                    "The `langchain-community` package is required for Hugging Face API embeddings. "
                    "Install it using: `pip install langchain-community`"
                )

            return HuggingFaceInferenceAPIEmbeddings(
                api_key=token,
                model_name=selected_model,
                **self.kwargs
            )

        # Option 2: Local Hugging Face Execution (Fast Local PyTorch execution)
        elif self.provider in ["huggingface", "bge-m3", "local"]:
            selected_model = self.model_name or "intfloat/multilingual-e5-small"
            
            model_kwargs = self.kwargs.pop("model_kwargs", {})
            model_kwargs.setdefault("device", self.device)
            
            encode_kwargs = self.kwargs.pop("encode_kwargs", {})
            encode_kwargs.setdefault("normalize_embeddings", True)

            return HuggingFaceEmbeddings(
                model_name=selected_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                **self.kwargs
            )

        # Option 3: OpenAI Embeddings
        elif self.provider == "openai":
            selected_model = self.model_name or "text-embedding-3-large"
            openai_args = self.kwargs.copy()
            token = self.api_key or os.getenv("OPENAI_API_KEY")
            if token:
                openai_args["api_key"] = token

            return OpenAIEmbeddings(
                model=selected_model,
                **openai_args
            )

        # Option 4: Cohere Embeddings
        elif self.provider == "cohere":
            try:
                from langchain_cohere import CohereEmbeddings
            except ImportError:
                raise ImportError(
                    "To use Cohere embeddings, install: `pip install langchain-cohere`"
                )
            selected_model = self.model_name or "embed-multilingual-v3.0"
            cohere_args = self.kwargs.copy()
            token = self.api_key or os.getenv("COHERE_API_KEY")
            if token:
                cohere_args["cohere_api_key"] = token

            return CohereEmbeddings(
                model=selected_model,
                **cohere_args
            )

        else:
            raise ValueError(
                f"Unsupported provider: '{self.provider}'. "
                "Choose from 'huggingface_api', 'huggingface', 'openai', or 'cohere'."
            )

    def embed_documents(self, docs: List[Union[Document, str]]) -> List[List[float]]:
        """Generates vector embeddings for a list of LangChain Document objects or plain strings."""
        if not docs:
            return []

        texts = [doc.page_content if isinstance(doc, Document) else doc for doc in docs]
        return self.embedder.embed_documents(texts)

    def embed_query(self, query_text: str) -> List[float]:
        """Embeds a single search query."""
        return self.embedder.embed_query(query_text)