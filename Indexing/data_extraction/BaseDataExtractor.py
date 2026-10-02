from abc import ABC, abstractmethod
class BaseDataExtractor(ABC):
    """Abstract base class for all data source extractors in the RAG pipeline."""
    
    @abstractmethod
    def extract(self, source: str) -> str:
        """Extract plain text from a given source (URL, File Path, ID, etc.)."""
        pass