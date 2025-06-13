# Learning Module Builder Agent

This project demonstrates a minimal LangGraph workflow for building
Jupyter-based learning modules. The agent clarifies the topic,
plans a short outline, then generates notebook cells.

## Features

- Modular nodes in `src/lmb_agent/nodes.py`.
- Agent assembly in `src/lmb_agent/agent.py`.
- Command line interface via `python -m lmb_agent`.

## Quick start

```bash
pip install -e .
python -m lmb_agent "Intro to NumPy"
```

## GitHub Codespaces

The repository includes a devcontainer for quick setup in Codespaces:

1. Click the green **Code** button on GitHub and choose **Create codespace**.
2. Wait for the container to build and install dependencies.
3. Add your OpenAI API key as a Codespace secret named `OPENAI_API_KEY`.
4. Run `python -m lmb_agent "Intro to NumPy"` or the `lmb-agent` command.

The agent reads `OPENAI_API_KEY` from the environment so you can chat with it
immediately once the secret is configured.

## Repository layout

```
learning-module-builder-agent/
├── src/lmb_agent/
│   ├── __init__.py
│   ├── __main__.py      # ``python -m lmb_agent`` entry point
│   ├── agent.py         # LangGraph workflow
│   ├── cli.py           # Command line interface
│   └── nodes/           # Reusable workflow steps
├── tests/               # Basic unit tests
├── pyproject.toml       # Package metadata
└── README.md
```

The example is intentionally lightweight so you can extend the workflow
with your own LangGraph nodes or tooling.
