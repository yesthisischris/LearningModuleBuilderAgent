# Contributor Guidelines

This repository demonstrates a light-weight LangGraph workflow for building
learning modules. Changes should follow these conventions:

- Keep the workflow modular. Each LangGraph node should be a small function that
  accepts and returns a dictionary of state.
- Add a descriptive comment block at the top of every Python file explaining its
  purpose and overall structure.
- Prefer simple, easily testable functions. If logic grows, factor it into
  helpers under `src/lmb_agent`.
- Run `ruff --fix` and `pytest` before committing when tests exist.
- Document new commands or examples in `README.md`.
