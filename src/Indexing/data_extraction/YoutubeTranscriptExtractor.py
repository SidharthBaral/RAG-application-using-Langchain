import re
from typing import List, Optional
from .BaseDataExtractor import BaseDataExtractor
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

# Import IP block / rate limit exceptions safely across library versions
try:
    from youtube_transcript_api._errors import RequestBlocked, IpBlocked
except ImportError:
    class RequestBlocked(Exception): pass
    class IpBlocked(Exception): pass


class YouTubeTranscriptExtractor(BaseDataExtractor):
    """Extracts transcript text from YouTube videos given a video URL or ID, with timestamps removed."""

    def __init__(self, preferred_languages: Optional[List[str]] = None):
        self.preferred_languages = preferred_languages or ['en']
        self.ytt_api = YouTubeTranscriptApi()

    def _extract_video_id(self, source: str) -> str:
        pattern = r"(?:v=|\/([0-9A-Za-z_-]{11}).*|youtu\.be\/)([0-9A-Za-z_-]{11})"
        match = re.search(pattern, source)
        if match:
            return match.group(1) or match.group(2)
        return source.strip()

    def _get_transcript_list(self, video_id: str):
        if hasattr(self.ytt_api, "list"):
            return self.ytt_api.list(video_id)
        elif hasattr(YouTubeTranscriptApi, "list_transcripts"):
            return YouTubeTranscriptApi.list_transcripts(video_id)
        else:
            raise AttributeError("Installed youtube_transcript_api version has no valid list method.")

    def _remove_timestamps(self, text: str) -> str:
        """Strips timestamp patterns like 00:00, 0:15, 1:23:45, [01:23], (0:15) from text."""
        if not text:
            return ""
        # Matches HH:MM:SS, MM:SS, or M:SS with optional surrounding brackets/parentheses and trailing dashes
        timestamp_pattern = r"[\(\[]?\b(?:\d{1,2}:)?\d{1,2}:\d{2}\b[\]\)]?\s*-?\s*"
        text = re.sub(timestamp_pattern, "", text)
        # Collapse multiple spaces into a single space
        return re.sub(r"\s+", " ", text).strip()

    def extract(self, source: str) -> str:
        video_id = self._extract_video_id(source)

        try:
            transcript_list = self._get_transcript_list(video_id)

            try:
                transcript = transcript_list.find_transcript(self.preferred_languages)
            except NoTranscriptFound:
                transcript = next(iter(transcript_list))

            data = transcript.fetch()

            # Compatible extraction for both FetchedTranscriptSnippet objects and dicts
            text_items = []
            for item in data:
                if hasattr(item, 'text'):
                    raw_str = item.text
                elif isinstance(item, dict):
                    raw_str = item.get('text', '')
                else:
                    raw_str = str(item)

                # Clean timestamps from each item string
                cleaned_item = self._remove_timestamps(raw_str)
                if cleaned_item:
                    text_items.append(cleaned_item)

            full_text = " ".join(text_items)
            return self._remove_timestamps(full_text)

        except TranscriptsDisabled:
            raise RuntimeError(f"Transcripts are disabled for video: {video_id}")
        except VideoUnavailable:
            raise RuntimeError(f"Video is unavailable or private: {video_id}")
        except (RequestBlocked, IpBlocked):
            raise RuntimeError("YouTube IP block or rate limit hit. Consider using residential proxies.")
        except Exception as e:
            raise RuntimeError(f"Failed to extract transcript from '{source}': {str(e)}")