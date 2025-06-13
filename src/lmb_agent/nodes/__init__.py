"""Reusable LangGraph nodes.

This module contains simple functions implementing each step
of the learning module workflow so they can be composed into
LangGraph graphs.
"""
from __future__ import annotations

from typing import Dict

from langchain_openai import ChatOpenAI


def clarify(state: Dict, llm: ChatOpenAI) -> Dict:
    """Ask the user for a concise topic clarification."""
    question = "Please provide a short topic for the learning module"
    state["topic"] = llm.predict(question)
    return state


def plan(state: Dict, llm: ChatOpenAI) -> Dict:
    """Create a brief lesson outline for the clarified topic."""
    prompt = f"Create an outline for a lesson on {state['topic']}"
    state["outline"] = llm.predict(prompt)
    return state


def generate(state: Dict, llm: ChatOpenAI) -> Dict:
    """Generate notebook cells for the given outline."""
    prompt = f"Generate notebook cells for: {state['outline']}"
    state["cells"] = llm.predict(prompt)
    return state
