"""Tests for transcript analysis loader."""

import types

import yaml

from reachy_mini_conversation_app.cascade.transcript_analysis import loader as loader_mod
from reachy_mini_conversation_app.cascade.transcript_analysis.base import TriggerConfig
from reachy_mini_conversation_app.cascade.transcript_analysis.loader import _parse_trigger


# --- _parse_trigger ---


def test_parse_trigger_words_only():
    """Parse trigger with only words."""
    t = _parse_trigger({"words": ["guitar", "bass"]})
    assert t.words == ["guitar", "bass"]
    assert t.entities == []
    assert t.all == []


def test_parse_trigger_entities_only():
    """Parse trigger with only entities."""
    t = _parse_trigger({"entities": ["PERSON", "ORG"]})
    assert t.entities == ["PERSON", "ORG"]
    assert t.words == []


def test_parse_trigger_all_groups():
    """Parse trigger with nested all sub-triggers."""
    t = _parse_trigger({"all": [{"words": ["danc*"]}, {"words": ["groov*"]}]})
    assert len(t.all) == 2
    assert t.all[0].words == ["danc*"]
    assert t.all[1].words == ["groov*"]
    assert t.words == []


def test_parse_trigger_empty_dict():
    """Parse empty dict into default TriggerConfig."""
    t = _parse_trigger({})
    assert t == TriggerConfig()


# --- _import_callback ---


def test_import_callback_success(monkeypatch):
    """Return function when import and getattr succeed."""
    def fake_fn():
        pass

    fake_module = types.ModuleType("fake")
    setattr(fake_module, "my_callback", fake_fn)

    monkeypatch.setattr("importlib.import_module", lambda path: fake_module)
    result = loader_mod._import_callback("test_profile", "my_callback")
    assert result is fake_fn


def test_import_callback_import_error(monkeypatch):
    """Return None when module import fails."""
    monkeypatch.setattr("importlib.import_module", lambda path: (_ for _ in ()).throw(ImportError("no module")))
    result = loader_mod._import_callback("test_profile", "missing_mod")
    assert result is None


def test_import_callback_missing_attribute(monkeypatch):
    """Return None when module lacks the callback attribute."""
    fake_module = types.ModuleType("fake")
    monkeypatch.setattr("importlib.import_module", lambda path: fake_module)
    result = loader_mod._import_callback("test_profile", "nonexistent")
    assert result is None


# --- get_profile_reactions ---


def _setup_profile(monkeypatch, tmp_path, profile_name, yaml_content=None, callback_fn=None):
    """Set up a fake profile directory with optional YAML and callback."""
    monkeypatch.setattr(loader_mod, "PROFILES_DIRECTORY", tmp_path)

    # Monkeypatch config object
    config_obj = types.SimpleNamespace(REACHY_MINI_CUSTOM_PROFILE=profile_name)
    monkeypatch.setattr(loader_mod, "config", config_obj)

    if profile_name:
        profile_dir = tmp_path / profile_name
        profile_dir.mkdir(parents=True, exist_ok=True)

        if yaml_content is not None:
            reactions_file = profile_dir / "reactions.yaml"
            reactions_file.write_text(yaml.dump(yaml_content))

    if callback_fn is not None:
        monkeypatch.setattr(loader_mod, "_import_callback", callback_fn)


def test_no_profile_returns_none(monkeypatch, tmp_path):
    """Return None when no profile is configured."""
    _setup_profile(monkeypatch, tmp_path, profile_name=None)
    assert loader_mod.get_profile_reactions() is None


def test_no_reactions_yaml_returns_none(monkeypatch, tmp_path):
    """Return None when profile dir exists but has no reactions.yaml."""
    _setup_profile(monkeypatch, tmp_path, profile_name="myprofile")
    assert loader_mod.get_profile_reactions() is None


def test_valid_yaml_loads_reactions(monkeypatch, tmp_path):
    """Load a complete reaction from valid YAML."""
    async def fake_cb(deps, match):
        pass

    _setup_profile(
        monkeypatch, tmp_path, profile_name="myprofile",
        yaml_content=[{
            "name": "greet",
            "callback": "greet",
            "trigger": {"words": ["hello", "hi"]},
        }],
        callback_fn=lambda profile, name: fake_cb,
    )
    reactions = loader_mod.get_profile_reactions()
    assert reactions is not None
    assert len(reactions) == 1
    assert reactions[0].name == "greet"
    assert reactions[0].trigger.words == ["hello", "hi"]
    assert reactions[0].callback is fake_cb


def test_skips_entry_missing_name(monkeypatch, tmp_path):
    """Skip YAML entries that lack a name field."""
    async def fake_cb(deps, match):
        pass

    _setup_profile(
        monkeypatch, tmp_path, profile_name="myprofile",
        yaml_content=[{"callback": "something", "trigger": {"words": ["x"]}}],
        callback_fn=lambda profile, name: fake_cb,
    )
    assert loader_mod.get_profile_reactions() is None


def test_skips_entry_bad_callback(monkeypatch, tmp_path):
    """Skip entries whose callback fails to import."""
    _setup_profile(
        monkeypatch, tmp_path, profile_name="myprofile",
        yaml_content=[{
            "name": "broken",
            "callback": "broken",
            "trigger": {"words": ["x"]},
        }],
        callback_fn=lambda profile, name: None,
    )
    assert loader_mod.get_profile_reactions() is None


def test_empty_yaml_returns_none(monkeypatch, tmp_path):
    """Return None when YAML contains an empty list."""
    _setup_profile(
        monkeypatch, tmp_path, profile_name="myprofile",
        yaml_content=[],
    )
    assert loader_mod.get_profile_reactions() is None


def test_preserves_params_and_repeatable(monkeypatch, tmp_path):
    """Preserve params dict and repeatable flag from YAML."""
    async def fake_cb(deps, match):
        pass

    _setup_profile(
        monkeypatch, tmp_path, profile_name="myprofile",
        yaml_content=[{
            "name": "wave",
            "callback": "wave",
            "trigger": {"words": ["wave"]},
            "params": {"direction": "left", "speed": 2},
            "repeatable": True,
        }],
        callback_fn=lambda profile, name: fake_cb,
    )
    reactions = loader_mod.get_profile_reactions()
    assert reactions is not None
    r = reactions[0]
    assert r.params == {"direction": "left", "speed": 2}
    assert r.repeatable is True
