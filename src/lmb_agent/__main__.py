"""Module execution entry point.

Running ``python -m lmb_agent`` is equivalent to invoking the
``lmb_agent.cli`` module.
"""
from .cli import main

if __name__ == "__main__":
    main()
