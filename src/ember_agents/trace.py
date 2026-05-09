"""Execution trace dataclasses for the EmberAgent 6-phase pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceStatus:
    """Per-source fetch status recorded during the Fetch phase.

    Attributes:
        name: Source identifier (e.g. ``"biologic_seed"``, ``"clinicaltrials"``).
        status: Outcome of the fetch — ``"ok"``, ``"empty"``, or ``"error"``.
        result_count: Number of results returned by this source (0 on error/empty).
    """

    name: str
    status: str
    result_count: int


@dataclass
class ExecutionTrace:
    """Structured record of a single EmberAgent pipeline execution.

    Attributes:
        extracted_signals: Raw signals extracted by the Interpret phase, keyed
            by dimension name (e.g. ``"target"``, ``"modality"``).
        query_type: Query classification produced by the Interpret phase
            (``"drug_lookup"``, ``"opportunity_scan"``, ``"biosimilar_screen"``,
            or ``"general"``).
        resolved_classifications: Mapping of dimension names to their resolved
            canonical values from the Classify phase.
        gate_outcome: Human-readable outcome of the Gate check
            (``"passed"`` or a failure reason such as ``"missing_core_fields"``).
        source_statuses: Per-source fetch outcome records from the Fetch phase.
        duration_seconds: Wall-clock time for the full pipeline in seconds.
    """

    extracted_signals: dict = field(default_factory=dict)
    query_type: str = "general"
    resolved_classifications: dict = field(default_factory=dict)
    gate_outcome: str = "passed"
    source_statuses: list[SourceStatus] = field(default_factory=list)
    duration_seconds: float = 0.0
