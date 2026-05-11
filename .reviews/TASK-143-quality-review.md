---
task_ref: TASK-143
plan_ref: PLAN-012
review_type: quality
verdict: PASS
reviewed_at: 2026-05-11T14:06:07Z
reviewed_by: SMA
---

# TASK-143 Quality Review: ember-agents

## Verdict

PASS

## Findings

### Resolved: explanation payload shape now matches the data/API contract

`ember-data` added top-level `CandidateResult` fields for `matched_dimensions`, `missed_dimensions`, `concrete_labels`, `component_scores`, `threshold_metadata`, `suppression_metadata`, and `evidence_summary`. The `ember-agents` implementation instead attaches matched/missed dimensions inside a nested `match_explanations` object and does not populate top-level `matched_dimensions`, `missed_dimensions`, or `concrete_labels`.

Impact: downstream API/UI consumers may not receive the fields promised by the phase 4 task and data model contract. The renderer can display nested `match_explanations`, but the cross-repo payload is inconsistent.

Revision completed:

- Top-level `matched_dimensions` and `missed_dimensions` are populated on the result object.
- `concrete_labels` is populated where labels are known.
- `match_explanations` remains additive and is no longer the sole carrier for matched/missed dimensions.
- Tests prove `_candidate_to_result()` exposes the top-level fields used by `ember-data` and `ember-api`.

### Resolved: active dimensions include DIR-007 opportunity query fields

`_expected_dimensions_from_spec()` now includes modality, cell-line class, revenue, patent window, and jurisdiction when those signals are active.

## Verification Evidence

- Child reported `rtk .venv/bin/pytest -q tests/search/test_render.py tests/test_ember_agent.py`: 81 passed.
- Child reported `rtk .venv/bin/ruff check src/ember_agents/agent.py src/ember_agents/render.py tests/search/test_render.py tests/test_ember_agent.py`: all checks passed.
- Revision dispatch `0001N03RVJ9AZTR3` reported `rtk .venv/bin/pytest -q tests/search/test_render.py tests/test_ember_agent.py`: 82 passed.
- Revision dispatch `0001N03RVJ9AZTR3` reported `rtk .venv/bin/ruff check src/ember_agents/agent.py src/ember_agents/render.py tests/search/test_render.py tests/test_ember_agent.py`: all checks passed.
