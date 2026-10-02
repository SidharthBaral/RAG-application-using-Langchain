import sys
from pathlib import Path
from typing import Dict, Type, Optional

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from Indexing.data_extraction.BaseDataExtractor import BaseDataExtractor
from Indexing.data_extraction.YoutubeTranscriptExtractor import YouTubeTranscriptExtractor
from Indexing.data_extraction.WebExtractor import WebExtractor

# Import additional extractors here as you build them
# from Indexing.data_extraction.WebExtractor import PDFExtractor


class DataExtractorFactory:
    """Factory class to dynamically instantiate extractors based on source type."""

    # Registry mapping source type identifiers to extractor classes
    _EXTRACTORS: Dict[str, Type[BaseDataExtractor]] = {
        "youtube": YouTubeTranscriptExtractor,
         "web": WebExtractor,
    }

    @classmethod
    def register_extractor(cls, source_type: str, extractor_cls: Type[BaseDataExtractor]) -> None:
        """Allows registering new extractors dynamically at runtime."""
        cls._EXTRACTORS[source_type.lower()] = extractor_cls

    @classmethod
    def get_extractor(cls, source_type: str, **kwargs) -> BaseDataExtractor:
        """Instantiates and returns a concrete extractor based on source_type.

        Parameters:
        - source_type: Identifier string (e.g., 'youtube', 'web')
        - **kwargs: Extra arguments passed to the extractor's __init__ method
        """
        key = source_type.lower().strip()
        
        # Friendly mappings for UI strings
        if "youtube" in key:
            key = "youtube"
        elif "web" in key or "url" in key:
            key = "web"

        extractor_cls = cls._EXTRACTORS.get(key)
        
        if not extractor_cls:
            supported = ", ".join(f"'{k}'" for k in cls._EXTRACTORS.keys())
            raise ValueError(
                f"Unsupported source type: '{source_type}'. "
                f"Currently supported options are: {supported}"
            )

        return extractor_cls(**kwargs)