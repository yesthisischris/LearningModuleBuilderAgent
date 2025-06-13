"""Top-level package for Learning Module Builder Agent.

This package exposes simple entry-points for building Jupyter-based
learning modules via a LangGraph agentic workflow.
"""

__all__ = ["build_agent", "clarify", "plan", "generate"]

from .agent import build_agent
from .nodes import clarify, generate, plan
