from __future__ import annotations

from hypothesize import models

ALLOW_LIST = [
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
]


class Denied(Exception):
    def __init__(self, body):
        self.body = body
        super().__init__(repr(body))


def test_real_allow_list_routes_reasoning_to_sonnet_46():
    resolver = models.ModelResolver(ALLOW_LIST)
    assert resolver.resolve("opus") == "us.anthropic.claude-sonnet-4-6"
    assert resolver.assignments()["opus"]["at_floor"] is True


def test_synthesis_selects_strongest_model_inside_its_band():
    resolver = models.ModelResolver(ALLOW_LIST)
    assert resolver.resolve("sonnet") == "us.anthropic.claude-sonnet-4-6"


def test_allow_list_is_parsed_from_governor_error():
    error = Denied({"error": "model_not_allowed", "allowedModels": ALLOW_LIST})
    assert models.is_model_not_allowed(error)
    assert models.parse_allowed_models(error) == ALLOW_LIST


def test_discovery_deliberately_trips_governor_gate():
    def probe(model_id):
        assert model_id == models.PROBE_MODEL
        raise Denied({"error": "model_not_allowed", "allowedModels": ALLOW_LIST})

    assert models.discover_allowed_models(probe) == ALLOW_LIST


def test_learning_invalidates_optimistic_assignment():
    resolver = models.ModelResolver()
    assert resolver.resolve("opus") == "us.anthropic.claude-opus-4-7"
    resolver.learn(ALLOW_LIST)
    assert resolver.resolve("opus") == "us.anthropic.claude-sonnet-4-6"


def test_unusable_model_without_allow_list_fails_instead_of_retrying_forever():
    resolver = models.ModelResolver()
    resolver.mark_unusable("us.anthropic.claude-opus-4-7")
    try:
        resolver.resolve("opus")
    except models.ModelNotAvailable as error:
        assert "did not report an allow-list" in str(error)
    else:
        raise AssertionError("expected an actionable model resolution failure")


def test_manifest_flags_allow_list_models_blocked_by_iam():
    resolver = models.ModelResolver(ALLOW_LIST)
    resolver.mark_unusable("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    resolver.resolve("opus")
    assert "[not invokable — IAM]" in models.render_manifest(resolver)
