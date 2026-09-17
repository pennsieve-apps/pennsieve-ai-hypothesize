"""Entrypoint wiring: the processor contract is honoured before any LLM work."""
from __future__ import annotations

import pytest

from hypothesize import main as entry


@pytest.fixture(autouse=True)
def processor_env(monkeypatch, tmp_path):
    monkeypatch.setenv("INPUT_DIR", str(tmp_path / "in"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("LLM_GOVERNOR_FUNCTION_NAME", "governor-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:8765")
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)


def test_missing_upstream_blackboard_exits_2(capsys):
    assert entry.main() == 2
    assert "Scout did not produce" in capsys.readouterr().err
