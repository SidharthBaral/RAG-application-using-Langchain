import os
import sys
import httpx
from pathlib import Path
from typing import Optional, Any
from dotenv import load_dotenv

# Automatically load environment variables
load_dotenv()


# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq


class TextGeneration:
    """Manages LLM initialization. Defaults to Groq with 'openai/gpt-oss-120b',

    while allowing custom models, keys, and providers (OpenAI, Anthropic, Ollama).
    """

    def __init__(
        self,
        provider: str = "groq",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        verify_ssl: bool = False,
        **kwargs: Any
    ):
        """Parameters:

        - provider: 'groq' (default), 'openai', 'anthropic', or 'ollama'.
        - model_name: Custom model ID. Defaults to 'openai/gpt-oss-120b' for Groq.
        - api_key: Explicit API key. Falls back to environment variables if None.
        - temperature: Sampling temperature (default 0.0 for deterministic RAG).
        - verify_ssl: Set False to bypass corporate proxy/SSL inspection issues.
        - **kwargs: Additional kwargs passed directly to the underlying Chat model.
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.verify_ssl = verify_ssl
        self.kwargs = kwargs

        # Handle HTTP client for corporate networks
        self.http_client = self.kwargs.pop("http_client", None)
        if self.http_client is None and not self.verify_ssl:
            self.http_client = httpx.Client(verify=False)

        self.llm: BaseChatModel = self._initialize_llm()

    def _initialize_llm(self) -> BaseChatModel:
        # 1. Groq Provider (Default)
        if self.provider == "groq":
            selected_model = self.model_name or "openai/gpt-oss-120b"
            token = self.api_key or os.getenv("GROQ_API_KEY")

            if not token:
                raise ValueError(
                    "Groq API key not found. Please set `GROQ_API_KEY` in your .env file "
                    "or pass `api_key` explicitly."
                )

            groq_kwargs = {
                "model_name": selected_model,
                "api_key": token,
                "temperature": self.temperature,
                **self.kwargs
            }

            if self.http_client:
                groq_kwargs["http_client"] = self.http_client

            return ChatGroq(**groq_kwargs)

        # 2. OpenAI Provider (Custom User Choice)
        elif self.provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(
                    "To use OpenAI models, install: `pip install langchain-openai`"
                )

            selected_model = self.model_name or "gpt-4o"
            token = self.api_key or os.getenv("OPENAI_API_KEY")

            if not token:
                raise ValueError(
                    "OpenAI API key not found. Please set `OPENAI_API_KEY` in your .env file "
                    "or pass `api_key` explicitly."
                )

            openai_kwargs = {
                "model": selected_model,
                "api_key": token,
                "temperature": self.temperature,
                **self.kwargs
            }

            if self.http_client:
                openai_kwargs["http_client"] = self.http_client

            return ChatOpenAI(**openai_kwargs)

        # 3. Anthropic Provider (Custom User Choice)
        elif self.provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError:
                raise ImportError(
                    "To use Anthropic models, install: `pip install langchain-anthropic`"
                )

            selected_model = self.model_name or "claude-3-5-sonnet-20241022"
            token = self.api_key or os.getenv("ANTHROPIC_API_KEY")

            if not token:
                raise ValueError(
                    "Anthropic API key not found. Please set `ANTHROPIC_API_KEY` in your .env file "
                    "or pass `api_key` explicitly."
                )

            return ChatAnthropic(
                model=selected_model,
                api_key=token,
                temperature=self.temperature,
                **self.kwargs
            )

        # 4. Ollama Local Provider (Custom User Choice)
        elif self.provider in ["ollama", "local"]:
            try:
                from langchain_community.chat_models import ChatOllama
            except ImportError:
                raise ImportError(
                    "To use Ollama models, install: `pip install langchain-community`"
                )

            selected_model = self.model_name or "llama3"
            return ChatOllama(
                model=selected_model,
                temperature=self.temperature,
                **self.kwargs
            )

        else:
            raise ValueError(
                f"Unsupported provider: '{self.provider}'. "
                "Choose from 'groq', 'openai', 'anthropic', or 'ollama'."
            )