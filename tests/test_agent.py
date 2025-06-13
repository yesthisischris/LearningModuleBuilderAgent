"""Basic tests for the MVP agent."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lmb_agent import build_agent  # noqa: E402


class DummyLLM:
    def predict(self, prompt: str) -> str:  # noqa: D401
        """Return a placeholder response."""
        return "dummy"


def test_build_agent():
    graph = build_agent(llm=DummyLLM())
    result = graph.invoke({"topic": "Test"})
    assert "cells" in result
