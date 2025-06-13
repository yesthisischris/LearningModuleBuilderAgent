"""Command-line interface for the Learning Module Builder Agent.

Invoke this module with a lesson topic to run the LangGraph agent
and print the generated notebook cells.
"""
from __future__ import annotations

import argparse

from .agent import build_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the learning module agent")
    parser.add_argument("topic", help="Short lesson topic")
    args = parser.parse_args()

    agent = build_agent()
    result = agent.invoke({"topic": args.topic})
    print(result)


if __name__ == "__main__":
    main()
