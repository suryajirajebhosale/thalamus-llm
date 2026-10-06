from pathlib import Path

import pytest

from thalamus import PolicyError, RoutingPolicy

EXAMPLES = Path(__file__).parent.parent / "examples"

DOC = {
    "defaults": {
        "provider": "vendor",
        "timeout": 30,
        "fallback": ["vendor"],
        "models": {"hosted": "small-model", "vendor": "big-model"},
    },
    "tasks": {
        "extraction": {
            "extract": {"provider": "hosted", "timeout": 12},
            "summarize": {"provider": "hosted", "model": "pinned", "temperature": 0},
        },
        "standalone": {"provider": "vendor", "reasoning_effort": "LOW", "tool_calling": True},
    },
}


def test_group_name_becomes_tier_and_defaults_fill_in():
    task = RoutingPolicy.from_dict(DOC).resolve("extract")
    assert task.tier == "extraction"
    assert task.provider == "hosted"
    assert task.model == "small-model"  # from defaults.models
    assert task.timeout == 12
    assert task.fallback == ("vendor",)


def test_pinned_model_and_options_pass_through():
    task = RoutingPolicy.from_dict(DOC).resolve("summarize")
    assert task.model == "pinned"
    assert task.timeout == 30
    assert dict(task.options) == {"temperature": 0}


def test_top_level_task_entry_is_not_a_group():
    task = RoutingPolicy.from_dict(DOC).resolve("standalone")
    assert task.tier is None
    assert task.reasoning_effort == "low"
    assert task.tool_calling is True


def test_unknown_task_resolves_to_defaults_and_is_flagged():
    task = RoutingPolicy.from_dict(DOC).resolve("brand_new_call_site")
    assert task.configured is False
    assert task.provider == "vendor"
    assert task.model == "big-model"


def test_unknown_task_without_default_provider_raises():
    with pytest.raises(PolicyError):
        RoutingPolicy.from_dict({"tasks": {"a": {"provider": "x"}}}).resolve("b")


def test_duplicate_task_ids_rejected():
    doc = {"tasks": {"g1": {"t": {"provider": "a"}}, "g2": {"t": {"provider": "b"}}}}
    with pytest.raises(PolicyError, match="more than once"):
        RoutingPolicy.from_dict(doc)


@pytest.mark.parametrize(
    "entry, match",
    [
        ({"provider": "a", "timeout": -1}, "positive"),
        ({"provider": "a", "timeout": "soon"}, "seconds"),
        ({"provider": "a", "reasoning_effort": "extreme"}, "reasoning_effort"),
        ({"provider": "a", "fallback": 3}, "fallback"),
        ({"model": "m"}, "no provider"),
    ],
)
def test_bad_entries_rejected(entry, match):
    with pytest.raises(PolicyError, match=match):
        RoutingPolicy.from_dict({"tasks": {"t": entry}})


def test_validate_reports_unknown_providers_missing_models_and_tasks():
    doc = {
        "defaults": {"models": {"a": "m"}},
        "tasks": {"t1": {"provider": "a", "fallback": ["a"]}, "t2": {"provider": "ghost"}},
    }
    problems = RoutingPolicy.from_dict(doc).validate(providers=["a"], required_tasks=["t1", "t3"])
    assert "tasks.t1: falls back to its own provider" in problems
    assert "tasks.t2.provider: unknown provider 'ghost'" in problems
    assert "tasks.t2: no model and no defaults.models.ghost" in problems
    assert "tasks.t3: called by code but missing from policy" in problems


def test_example_configs_are_valid():
    policy = RoutingPolicy.from_yaml(EXAMPLES / "thalamus.yaml")
    assert policy.validate(providers=["openai", "claude", "gemini", "local"]) == []
    assert policy.resolve("intent_classifier").options == {"temperature": 0}
    demo = RoutingPolicy.from_yaml(EXAMPLES / "demo.yaml")
    assert demo.validate(providers=["hosted", "openai", "claude"]) == []


def test_reload_reads_file_again(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text("tasks:\n  t:\n    provider: a\n    model: one\n")
    policy = RoutingPolicy.from_yaml(path)
    path.write_text("tasks:\n  t:\n    provider: a\n    model: two\n")
    assert policy.resolve("t").model == "one"
    assert policy.reload().resolve("t").model == "two"
