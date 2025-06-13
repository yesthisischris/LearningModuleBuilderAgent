"""Build and run the LangGraph-powered learning module agent.

This module assembles the reusable node functions into a LangGraph
`StateGraph`. The resulting graph clarifies the desired topic,
plans a short lesson outline, then generates notebook cells.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph

from .nodes import clarify, generate, plan, ask_approval, save_notebook


def build_agent(llm: ChatOpenAI | None = None) -> StateGraph:
    """Return a minimal agent graph."""
    llm = llm or ChatOpenAI(model="gpt-4o")
    graph = StateGraph(dict)

    graph.add_node("clarify", lambda state: clarify(state, llm))
    graph.add_node("plan", lambda state: plan(state, llm))
    graph.add_node("ask_approval", lambda state: ask_approval(state, llm))
    graph.add_node("generate", lambda state: generate(state, llm))
    graph.add_node("save_notebook", lambda state: save_notebook(state, llm))

    graph.add_edge(START, "clarify")
    graph.add_edge("clarify", "plan")
    graph.add_edge("plan", "ask_approval")
    graph.add_edge("ask_approval", "generate")
    graph.add_edge("generate", "save_notebook")
    graph.add_edge("save_notebook", END)
    return graph.compile()
