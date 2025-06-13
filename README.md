# 📓 `nb‑module‑agent`
_A turnkey Python package that lets an LLM **plan, generate and quality‑check complete Jupyter‑notebook learning modules** (e.g. “Build me a short _NumPy_ tutorial”).  
It blends **LangChain/LangGraph internal tools** with **Model Context Protocol (MCP) tools** for secure, best‑practice extensibility._

---

## ✨ Key Features
| Capability | Implementation |
|------------|----------------|
|**Clarification & planning**|LangGraph state machine in `agent.py` asks follow‑up questions, then produces an outline.|
|**Code execution & validation**|`PythonREPLTool` runs snippets to be sure they work :contentReference[oaicite:0]{index=0}|
|**External integrations (GitHub, Drive, etc.)**|Done via MCP tools exposed on one or more MCP servers using **`langchain‑mcp‑adapters`** :contentReference[oaicite:1]{index=1}|
|**Notebook assembly**|`nbformat` compiles markdown + code cells into a `.ipynb`.|
|**Self‑critique**|A second LLM pass flags missing concepts, bad code, or unsafe content.|
|**Dev‑container**|Reproducible VS Code setup with Docker, Poetry, pre‑commit, Jupyter port‑forward.|
|**CI/CD**|GitHub Actions runs unit tests, lint, mypy and sample‑notebook smoke tests.|

---

## 🔧 Quick‑start

```bash
git clone https://github.com/your‑org/nb‑module‑agent
cd nb‑module‑agent
devcontainer open .          # or: docker compose up dev
poetry install --with dev
python -m nb_module_agent.cli \
   "Create a short **Pandas** beginner module"
Minimal code example
python
Copy
from langchain.chat_models import ChatOpenAI
from langchain.tools.python import PythonREPLTool           # internal Lang tool
from langchain_mcp_adapters.client import (
    MultiServerMCPClient, mcp_tools_from_schema)            # MCP tool bridge
from langgraph.prebuilt import create_react_agent

# 1. Discover external MCP tools (e.g. GitHub, Google Drive)
mcp = MultiServerMCPClient(["https://mcp.my‑org.ai"])
mcp_tools = mcp_tools_from_schema(mcp)

# 2. Build an agent with BOTH internal and MCP tools
tools = [PythonREPLTool()] + mcp_tools
agent = create_react_agent(
    llm=ChatOpenAI(model="gpt‑4o‑mini"),
    tools=tools,
    system_prompt="You are a rigorous notebook‑module creator."
)

# 3. Invoke
agent.invoke("Build a short *Matplotlib* tutorial notebook for beginners.")
🗂️ Repository Layout
bash
Copy
nb-module-agent/
├─ .devcontainer/               # VS Code Dev Container
│  ├─ devcontainer.json         # ports, extensions, postCreate
│  └─ Dockerfile                # Python 3.12‑slim + Jupyter + Poetry
├─ .github/
│  └─ workflows/ci.yml          # lint, type‑check, tests
├─ src/nb_module_agent/
│  ├─ __init__.py               # version, public API
│  ├─ agent.py                  # LangGraph state graph (clarify → plan → gen)
│  ├─ planning.py               # Lesson‑plan templates & validators
│  ├─ notebook_builder.py       # nbformat helpers
│  ├─ tools/
│  │  ├─ repl.py                # Thin wrapper around PythonREPLTool
│  │  └─ mcp_client.py          # Utility to fetch & cache MCP tool schemas
│  └─ cli.py                    # `python -m nb_module_agent` entry‑point
├─ tests/
│  ├─ test_planner.py
│  └─ test_notebook_compile.py
├─ notebooks/                   # Sample outputs (smoke‑tested in CI)
├─ pyproject.toml               # Poetry, type‑stub, Ruff, mypy config
├─ README.md                    # ← you are here
└─ LICENSE
File/Dir highlights
Path	Why it exists
agent.py	Defines a LangGraph StateGraph: Clarify → Plan → Critique → GenerateCells → CompileNotebook
tools/mcp_client.py	One‑liner to wrap any MCP server and expose its tools to the agent.
notebook_builder.py	Transforms validated JSON sections into real notebook cells (nbformat.v4).
tests/	Unit tests must pass ruff + mypy + pytest to merge.

🖥️ Dev‑container
.devcontainer/devcontainer.json

jsonc
Copy
{
  "name": "nb-module-agent",
  "image": "mcr.microsoft.com/devcontainers/python:3.12",
  "features": { "ghcr.io/devcontainers/features/docker-in-docker:2": {} },
  "postCreateCommand": "poetry install --with dev && pre-commit install",
  "forwardPorts": [8888],
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-azuretools.vscode-docker"
      ]
    }
  }
}
.devcontainer/Dockerfile

Dockerfile
Copy
FROM mcr.microsoft.com/devcontainers/python:3.12

# System deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Poetry already baked in; just ensure nbformat + jupyter for local tests
RUN pip install --no-cache-dir "nbformat>=5.10" jupyter
📜 Best‑practice checklist
Domain	What we do	References
Packaging	PEP 621 / pyproject.toml; no setup.py	
Code‑style	Ruff (ruff check --fix) + Black profile	
Types	mypy strict mode; CI fails on Any leakage	
Security	Pinned deps via Poetry lock; automated pip‑audit on PRs	
LLM Ops	Separate system/assistant prompts, deterministic IDs for each graph node; retries + timeouts with lc_retry decorator	
External tools	Use MCP adapters to avoid embedding secrets; tool schemas fetched dynamically 
changelog.langchain.com
Notebook hygiene	All generated notebooks include a pre‑executed kernel UUID & metadata for reproducible CI smoke runs	

🚀 Roadmap
🔍 Add Tavily search tool for “research‑first” lessons

🧪 Integrate pytest‑nb to execute notebooks in CI

🌐 Optional web UI (FastAPI + Mermaid flow viz)

📄 License
MIT — see LICENSE.

The Model Context Protocol (MCP) is an open standard announced by Anthropic for safe, tool‑augmented LLM workflows 
theverge.com
.

makefile
Copy
::contentReference[oaicite:4]{index=4}
