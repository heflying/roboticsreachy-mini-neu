"""Tests for TranscriptAnalysisManager."""

from __future__ import annotations
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app.cascade.config import set_config
from reachy_mini_conversation_app.cascade.transcript_analysis.base import (
    EntityMatch,
    TriggerMatch,
    TriggerConfig,
    ReactionConfig,
)
from reachy_mini_conversation_app.cascade.transcript_analysis.manager import (
    TranscriptAnalysisManager,
)


@pytest.fixture(autouse=True)
def _mock_cascade_config():
    """Inject a mock CascadeConfig so entity tests don't need cascade.yaml."""
    mock_cfg = MagicMock()
    mock_cfg.gliner_model = "urchade/gliner_small-v2.1"
    set_config(mock_cfg)
    yield
    set_config(None)


def _make_reaction(
    name: str,
    words: list[str] | None = None,
    repeatable: bool = False,
    params: dict | None = None,
    all_groups: list[list[str]] | None = None,
    entities: list[str] | None = None,
) -> ReactionConfig:
    """Build a ReactionConfig with AsyncMock callback."""
    if all_groups:
        trigger = TriggerConfig(all=[TriggerConfig(words=g) for g in all_groups])
    else:
        trigger = TriggerConfig(
            words=words or [],
            entities=entities or [],
        )
    return ReactionConfig(
        name=name,
        callback=AsyncMock(),
        trigger=trigger,
        params=params or {},
        repeatable=repeatable,
    )


def _cb(r: ReactionConfig) -> AsyncMock:
    """Extract the AsyncMock callback for test assertions."""
    assert isinstance(r.callback, AsyncMock)
    return r.callback


def _make_manager(reactions: list[ReactionConfig], **kwargs) -> TranscriptAnalysisManager:
    """Build a manager with a mock deps and no entity analyzer."""
    deps = MagicMock()
    mgr = TranscriptAnalysisManager(reactions, deps, **kwargs)
    # Disable entity analyzer by default (tests that need it will set it explicitly)
    mgr.entity_analyzer = None
    return mgr


# --- Basic keyword dispatch ---


@pytest.mark.asyncio
async def test_keyword_fires_callback():
    """Dispatch callback when keyword matches."""
    r = _make_reaction("music", words=["guitar"])
    mgr = _make_manager([r])
    await mgr.analyze_final("I love guitar")
    await asyncio.sleep(0)

    _cb(r).assert_called_once()
    _, match = _cb(r).call_args.args[0], _cb(r).call_args.args[1]
    assert isinstance(match, TriggerMatch)
    assert "guitar" in match.words


@pytest.mark.asyncio
async def test_callback_receives_params():
    """Pass reaction params as kwargs to callback."""
    r = _make_reaction("wave", words=["wave"], params={"direction": "left"})
    mgr = _make_manager([r])
    await mgr.analyze_final("let's wave")
    await asyncio.sleep(0)

    _cb(r).assert_called_once()
    call_kwargs = _cb(r).call_args.kwargs
    assert call_kwargs["direction"] == "left"


# --- Deduplication ---


@pytest.mark.asyncio
async def test_non_repeatable_fires_once():
    """Fire non-repeatable reaction only once across analyses."""
    r = _make_reaction("music", words=["guitar"], repeatable=False)
    mgr = _make_manager([r])
    await mgr.analyze_final("I love guitar")
    await asyncio.sleep(0)
    await mgr.analyze_final("guitar solo")
    await asyncio.sleep(0)

    assert _cb(r).call_count == 1


@pytest.mark.asyncio
async def test_repeatable_keyword_fires_every_time():
    """Fire repeatable reaction on every matching analysis."""
    r = _make_reaction("music", words=["guitar"], repeatable=True)
    mgr = _make_manager([r])
    await mgr.analyze_final("I love guitar")
    await asyncio.sleep(0)
    await mgr.analyze_final("guitar solo")
    await asyncio.sleep(0)

    assert _cb(r).call_count == 2


@pytest.mark.asyncio
async def test_reset_clears_dedup():
    """Allow non-repeatable reaction to fire again after reset."""
    r = _make_reaction("music", words=["guitar"], repeatable=False)
    mgr = _make_manager([r])
    await mgr.analyze_final("I love guitar")
    await asyncio.sleep(0)
    assert _cb(r).call_count == 1

    mgr.reset()
    await mgr.analyze_final("guitar again")
    await asyncio.sleep(0)
    assert _cb(r).call_count == 2


# --- Boolean `all` triggers ---


@pytest.mark.asyncio
async def test_all_trigger_fires_when_all_match():
    """Fire when all sub-groups of an all-trigger match."""
    r = _make_reaction("dance_groove", all_groups=[["danc*"], ["groov*"]])
    mgr = _make_manager([r])
    await mgr.analyze_final("I was dancing to a grooving beat")
    await asyncio.sleep(0)

    _cb(r).assert_called_once()


@pytest.mark.asyncio
async def test_all_trigger_no_fire_on_partial():
    """Do not fire when only some sub-groups match."""
    r = _make_reaction("dance_groove", all_groups=[["danc*"], ["groov*"]])
    mgr = _make_manager([r])
    await mgr.analyze_final("I was dancing all night")
    await asyncio.sleep(0)

    _cb(r).assert_not_called()


@pytest.mark.asyncio
async def test_all_trigger_merged_words():
    """Merge matched words from all sub-groups into TriggerMatch."""
    r = _make_reaction("dance_groove", all_groups=[["danc*"], ["groov*"]])
    mgr = _make_manager([r])
    await mgr.analyze_final("dancing to grooving beats")
    await asyncio.sleep(0)

    match = _cb(r).call_args.args[1]
    assert "dancing" in match.words
    assert "grooving" in match.words


# --- Entity dispatch ---


@pytest.mark.asyncio
async def test_entity_dispatch():
    """Dispatch callback when entity analyzer finds a match."""
    r = _make_reaction("person_react", entities=["PERSON"])
    mgr = _make_manager([r])
    # Provide a fake entity analyzer that returns a match
    entity_match = EntityMatch(text="Alice", label="PERSON", confidence=0.9)

    async def fake_entity_analyze(text, is_final):
        return [entity_match]

    mock_entity_analyzer = MagicMock()
    mock_entity_analyzer.analyze = AsyncMock(side_effect=fake_entity_analyze)
    mgr.entity_analyzer = mock_entity_analyzer

    await mgr.analyze_final("I met Alice today")
    await asyncio.sleep(0)

    _cb(r).assert_called_once()
    match = _cb(r).call_args.args[1]
    assert len(match.entities) == 1
    assert match.entities[0].text == "Alice"


@pytest.mark.asyncio
async def test_entity_repeatable_dedup_by_text():
    """Deduplicate repeatable entity reactions by entity text."""
    r = _make_reaction("person_react", entities=["PERSON"], repeatable=True)
    mgr = _make_manager([r])

    async def fake_analyze_alice(text, is_final):
        return [EntityMatch(text="Alice", label="PERSON", confidence=0.9)]

    async def fake_analyze_bob(text, is_final):
        return [EntityMatch(text="Bob", label="PERSON", confidence=0.9)]

    mock_analyzer = MagicMock()
    mock_analyzer.analyze = AsyncMock(side_effect=fake_analyze_alice)
    mgr.entity_analyzer = mock_analyzer

    await mgr.analyze_final("I met Alice")
    await asyncio.sleep(0)
    # Same entity text again — should be deduped
    await mgr.analyze_final("Alice is here")
    await asyncio.sleep(0)
    assert _cb(r).call_count == 1

    # Different entity text — should fire
    mock_analyzer.analyze = AsyncMock(side_effect=fake_analyze_bob)
    await mgr.analyze_final("Bob arrived")
    await asyncio.sleep(0)
    assert _cb(r).call_count == 2


@pytest.mark.asyncio
async def test_non_repeatable_entity_fires_once():
    """Fire non-repeatable entity reaction only once total."""
    r = _make_reaction("person_react", entities=["PERSON"], repeatable=False)
    mgr = _make_manager([r])

    mock_analyzer = MagicMock()
    mock_analyzer.analyze = AsyncMock(
        return_value=[EntityMatch(text="Alice", label="PERSON", confidence=0.9)]
    )
    mgr.entity_analyzer = mock_analyzer

    await mgr.analyze_final("I met Alice")
    await asyncio.sleep(0)
    await mgr.analyze_final("Bob is here")
    await asyncio.sleep(0)

    assert _cb(r).call_count == 1


# --- Partial analysis debouncing ---


@pytest.mark.asyncio
async def test_analyze_partial_debounces():
    """Debounce rapid partial calls so only the first dispatches."""
    r = _make_reaction("music", words=["guitar"], repeatable=True)
    mgr = _make_manager([r])

    # Three rapid calls — only first should dispatch
    await mgr.analyze_partial("guitar riff")
    await mgr.analyze_partial("guitar riff 2")
    await mgr.analyze_partial("guitar riff 3")
    await asyncio.sleep(0.1)  # let tasks complete

    assert _cb(r).call_count == 1


# --- Multiple independent reactions ---


@pytest.mark.asyncio
async def test_multiple_reactions_independent():
    """Fire two independent reactions from the same text."""
    r1 = _make_reaction("music", words=["guitar"])
    r2 = _make_reaction("dance", words=["danc*"])
    mgr = _make_manager([r1, r2])
    await mgr.analyze_final("I play guitar while dancing")
    await asyncio.sleep(0)

    _cb(r1).assert_called_once()
    _cb(r2).assert_called_once()
