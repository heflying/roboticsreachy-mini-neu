"""Tests for KeywordAnalyzer."""

import pytest

from reachy_mini_conversation_app.cascade.transcript_analysis.keyword_analyzer import (
    KeywordAnalyzer,
    _is_glob,
)


# --- _is_glob helper ---


def test_is_glob_star():
    """Detect star wildcard."""
    assert _is_glob("danc*") is True


def test_is_glob_question_mark():
    """Detect question-mark wildcard."""
    assert _is_glob("colo?r") is True


def test_is_glob_plain_word():
    """Reject plain word as non-glob."""
    assert _is_glob("guitar") is False


# --- KeywordAnalyzer.analyze ---


@pytest.fixture
def single_reaction():
    """Create analyzer with one literal trigger word."""
    return KeywordAnalyzer({"music": ["guitar"]})


@pytest.fixture
def glob_reaction():
    """Create analyzer with one glob trigger pattern."""
    return KeywordAnalyzer({"dance": ["danc*"]})


@pytest.mark.asyncio
async def test_literal_word_match(single_reaction):
    """Match exact literal word in text."""
    result = await single_reaction.analyze("I love guitar", is_final=True)
    assert result == {"music": ["guitar"]}


@pytest.mark.asyncio
async def test_literal_case_insensitive(single_reaction):
    """Match literal word regardless of case."""
    result = await single_reaction.analyze("I love GUITAR solos", is_final=True)
    assert result == {"music": ["guitar"]}


@pytest.mark.asyncio
async def test_literal_substring_match(single_reaction):
    """Match literal word as substring of a token."""
    result = await single_reaction.analyze("She is a great guitarist", is_final=True)
    assert result == {"music": ["guitar"]}


@pytest.mark.asyncio
async def test_multi_word_phrase():
    """Match multi-word literal phrase."""
    analyzer = KeywordAnalyzer({"piano": ["grand piano"]})
    result = await analyzer.analyze("I have a grand piano at home", is_final=True)
    assert result == {"piano": ["grand piano"]}


@pytest.mark.asyncio
async def test_glob_pattern_match(glob_reaction):
    """Match glob pattern against whole token."""
    result = await glob_reaction.analyze("I love dancing", is_final=True)
    assert result == {"dance": ["dancing"]}


@pytest.mark.asyncio
async def test_glob_no_substring_match(glob_reaction):
    """Fnmatch matches whole tokens, so 'danc*' should not match 'undanceable'."""
    result = await glob_reaction.analyze("That was undanceable", is_final=True)
    assert result == {}


@pytest.mark.asyncio
async def test_glob_one_match_per_pattern(glob_reaction):
    """Yield only first matching token per glob pattern."""
    result = await glob_reaction.analyze("dancing dancer", is_final=True)
    # Only one match per glob pattern (first token that matches)
    assert result == {"dance": ["dancing"]}


@pytest.mark.asyncio
async def test_multiple_reactions():
    """Fire both reactions when text matches both."""
    analyzer = KeywordAnalyzer({"music": ["guitar"], "dance": ["danc*"]})
    result = await analyzer.analyze("I play guitar while dancing", is_final=True)
    assert "music" in result
    assert "dance" in result


@pytest.mark.asyncio
async def test_no_match_empty_dict(single_reaction):
    """Return empty dict when no words match."""
    result = await single_reaction.analyze("I love cooking", is_final=True)
    assert result == {}


@pytest.mark.asyncio
async def test_empty_text(single_reaction):
    """Return empty dict for empty input text."""
    result = await single_reaction.analyze("", is_final=True)
    assert result == {}


@pytest.mark.asyncio
async def test_mixed_literals_and_globs():
    """Match both literal and glob words in a single reaction."""
    analyzer = KeywordAnalyzer({"groove": ["funk", "groov*"]})
    result = await analyzer.analyze("This funk track is grooving", is_final=True)
    assert "groove" in result
    assert "funk" in result["groove"]
    assert "grooving" in result["groove"]
