"""Basic tests for the MVP agent."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lmb_agent import build_agent  # noqa: E402


class DummyLLM:
    def invoke(self, prompt: str):  # noqa: D401
        """Return a dummy response object with ``content`` attribute."""

        class DummyResponse:
            content = "dummy"

        return DummyResponse()


def _approve(state, _llm):
    state["approved"] = True
    return state


def _skip_research(state, _llm):
    state["research_results"] = []
    state["doc_content"] = ""
    state["package_info"] = {}
    return state


def _skip_save(state, _llm):
    state["notebook_file"] = "dummy.ipynb"
    return state


def test_build_agent(monkeypatch):
    monkeypatch.setattr("lmb_agent.nodes.ask_approval", _approve)
    monkeypatch.setattr("lmb_agent.agent.ask_approval", _approve)
    monkeypatch.setattr("lmb_agent.nodes.research_package", _skip_research)
    monkeypatch.setattr("lmb_agent.agent.research_package", _skip_research)
    monkeypatch.setattr("lmb_agent.nodes.save_notebook", _skip_save)
    monkeypatch.setattr("lmb_agent.agent.save_notebook", _skip_save)

    graph = build_agent(llm=DummyLLM())
    result = graph.invoke({"topic": "Test"})
    assert result["notebook_file"] == "dummy.ipynb"
