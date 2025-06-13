"""Reusable LangGraph nodes.

This module contains simple functions implementing each step
of the learning module workflow so they can be composed into
LangGraph graphs.
"""
from __future__ import annotations

from typing import Dict

from langchain_openai import ChatOpenAI


def clarify(state: Dict, llm: ChatOpenAI) -> Dict:
    """Use the provided topic or ask the user for a concise topic clarification."""
    if "topic" not in state or not state["topic"]:
        question = "Please provide a short topic for the learning module"
        response = llm.invoke(question)
        state["topic"] = response.content
    return state


def plan(state: Dict, llm: ChatOpenAI) -> Dict:
    """Create a brief lesson outline for the clarified topic."""
    prompt = f"Create an outline for a lesson on {state['topic']}"
    response = llm.invoke(prompt)
    state["outline"] = response.content
    return state


def generate(state: Dict, llm: ChatOpenAI) -> Dict:
    """Generate notebook cells for the given outline."""
    prompt = f"Generate notebook cells for: {state['outline']}"
    response = llm.invoke(prompt)
    state["cells"] = response.content
    return state
    return state
