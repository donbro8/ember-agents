# ember-agents — Submodule Agent Instructions

This file provides cross-provider governance instructions for child agents dispatched to this submodule.

## Governance Context

This repository is managed by the System Management Agent (SMA) from the parent project root. Child agents (module-architect, implementer, reviewer) are dispatched here via task files in `.tasks/`.

## Rules

- Read task files from `.tasks/` for your assignment
- Write implementation code within this repository only
- Do not modify governance artifacts (`status/`, `docs/system/`, `.contracts/`) — those belong to SMA
- Follow the conventions documented in this repo's `CLAUDE.md`
- Report completion by updating your task file status

## Agent Definitions

Agent definitions are in `.claude/agents/`. See `CLAUDE.md` for the routing table.

## Key Paths

| Path | Purpose |
|---|---|
| `.tasks/` | Inbound task files from SMA |
| `.claude/agents/` | Agent definitions |
| `docs/` | Projected architecture and tech-spec docs |
| `docs/detail-spec/` | Detailed specifications |

## Submodule Metadata

| Field | Value |
|---|---|
| Repo | ember-agents |
| Type | service |
| Domain | agents — LangGraph agent orchestration |
