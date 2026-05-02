# ember-agents

## What this package does

All agent logic and LangGraph orchestration for the Ember Bio platform. Contains
the Discovery Agent ("Orion"), Scoring Agent, Report Agent, and the multi-agent
orchestrator that routes queries through the appropriate pipeline.

## Key modules

- `base.py` — Abstract Agent class
- `factory.py` — Agent registry and factory
- `discovery/` — Discovery Agent with search and synthesis tools
- `scoring/` — Candidate scoring agent and criteria framework
- `reporting/` — Report generation agent
- `orchestrator.py` — LangGraph multi-agent flow

## How to run tests

```bash
uv sync --extra dev
uv run pytest --cov
```

## Conventions

- Agents import ember-data directly for BigQuery access (not through API)
- All agents inherit from the abstract Agent base class
- LangGraph manages agent state and routing
- Async-first — use pytest-asyncio for async test fixtures
- Depends on ember-shared and ember-data
- Lint with ruff: `uv run ruff check src/ tests/`
