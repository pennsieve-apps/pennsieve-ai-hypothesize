from __future__ import annotations

from hypothesize.blackboard import Blackboard


def test_blackboard_round_trips_model_provenance(tmp_path):
    path = tmp_path / "blackboard.json"
    original = Blackboard(
        research_question="question",
        subsystem="pennsieve:530",
        model_assignments={"opus": {"model_id": "us.anthropic.claude-sonnet-4-6"}},
    )
    original.save(path)
    loaded = Blackboard.load(path)
    assert loaded.model_assignments == original.model_assignments


def test_blackboard_preserves_scout_degraded_state(tmp_path):
    path = tmp_path / "blackboard.json"
    original = Blackboard(
        research_question="question",
        subsystem="pennsieve:530",
        degraded=[{"stage": "deep_reader", "error": "timeout"}],
    )
    original.save(path)
    assert Blackboard.load(path).degraded == original.degraded
