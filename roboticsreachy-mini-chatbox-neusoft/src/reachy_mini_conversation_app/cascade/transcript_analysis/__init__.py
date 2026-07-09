"""Transcript analysis for real-time reaction to user speech."""

from .base import EntityMatch, TriggerMatch, TriggerConfig, ReactionConfig, TranscriptAnalyzer
from .loader import get_profile_reactions
from .manager import NoOpTranscriptManager, TranscriptAnalysisManager
from .keyword_analyzer import KeywordAnalyzer


# EntityAnalyzer is optional (requires gliner extra)
try:
    from .entity_analyzer import EntityAnalyzer

    __all__ = [
        "EntityMatch",
        "TriggerMatch",
        "TriggerConfig",
        "ReactionConfig",
        "TranscriptAnalyzer",
        "KeywordAnalyzer",
        "EntityAnalyzer",
        "TranscriptAnalysisManager",
        "NoOpTranscriptManager",
        "get_profile_reactions",
    ]
except ImportError:
    __all__ = [
        "EntityMatch",
        "TriggerMatch",
        "TriggerConfig",
        "ReactionConfig",
        "TranscriptAnalyzer",
        "KeywordAnalyzer",
        "TranscriptAnalysisManager",
        "NoOpTranscriptManager",
        "get_profile_reactions",
    ]
