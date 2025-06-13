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

    print(f"🚀 Creating learning module for: {args.topic}")
    print("📝 Generating lesson plan...")
    
    agent = build_agent()
    result = agent.invoke({"topic": args.topic})
    
    if result.get("notebook_file"):
        print(f"\n🎉 Learning module complete!")
        print(f"📁 File saved: {result['notebook_file']}")
    else:
        print("\n❌ Learning module creation was cancelled or failed.")
    
    # Don't print the full result anymore since it's verbose
    # print(result)


if __name__ == "__main__":
    main()
