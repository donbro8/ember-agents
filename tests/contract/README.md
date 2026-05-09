# Contract Tests — Tier 1

This directory contains **Tier 1 contract tests** for the ember-agents package.
Tier 1 tests use recorded JSON fixtures and require no network access, making
them fast, deterministic, and safe to run in CI without credentials.

## Directory layout

```
tests/contract/
├── README.md                  # This file
├── __init__.py
├── conftest.py                # Shared helpers and pytest fixtures
├── fixtures/                  # Recorded JSON responses (source of truth)
│   ├── bigquery_patents_adalimumab.json
│   ├── bigquery_fda.json
│   ├── biologic_seed.json
│   ├── clinicaltrials.json
│   ├── pubmed.json
│   └── uniprot.json
├── test_ember_agent.py
├── test_intent_extractor.py
├── test_schema_compliance.py
└── test_seed_source.py
```

## Contract test strategy

### What "Tier 1" means

Tier 1 tests verify **schema compliance without network access**.  They answer
the question: *given a realistic API response (captured from the live source),
do our domain models parse it correctly?*

This is distinct from:

- **Tier 2 integration tests** — exercise live API calls; require credentials
  and network access; run in CI only when secrets are available.
- **Unit tests** — mock at the function level; live alongside the source code
  in `src/`.

### Reference queries

All fixtures are captured for the reference molecule **adalimumab** (Humira),
chosen because it has well-known, stable data across every source:

| Fixture | Source | Query |
|---|---|---|
| `bigquery_patents_adalimumab.json` | Google Patents (BigQuery) | `"adalimumab"` |
| `bigquery_fda.json` | FDA drug labels (BigQuery) | `"adalimumab"` |
| `clinicaltrials.json` | ClinicalTrials.gov REST API v2 | `"adalimumab"` |
| `pubmed.json` | NCBI PubMed E-utilities | `"adalimumab"` |
| `uniprot.json` | UniProt REST API | accession `P01375` (TNF_HUMAN) |
| `biologic_seed.json` | Internal seed data | adalimumab biosimilar landscape |

### What the tests cover

1. BigQuery patent rows parse to `PatentJurisdiction` with correct fields.
2. Seed fixtures parse to `PatentJurisdiction` with `expiry_date_approximate=False`.
3. `canonical_id` is stable across `adalimumab` / `Humira` when `fda_generic_name` is present.
4. `result_type` is `DRUG` when `drug_name` is present; `TARGET` when only a target is present.
5. `display_label` is always populated.
6. Scoring weights shift correctly per `query_type`.

## Running the contract tests

```bash
# All contract tests (no network, no credentials needed)
uv run pytest tests/contract/ -m contract -v

# Single file
uv run pytest tests/contract/test_schema_compliance.py -v
```

## Fixture management

### When to refresh fixtures

Refresh fixtures when:

- An upstream API changes its response schema.
- A new field is added or removed from a data source.
- You suspect a fixture is stale (check the diff output).
- Preparing a release that depends on stable API shapes.

You do **not** need to refresh fixtures when:
- Adding new contract tests (the existing fixtures are the source of truth).
- Changing internal parsing logic (update the tests instead).

### How to refresh fixtures

Use the capture script in `scripts/`:

```bash
# Refresh all sources (requires GCP credentials + network access)
./scripts/refresh-contract-fixtures.sh

# Refresh a single source
./scripts/refresh-contract-fixtures.sh --source clinicaltrials
./scripts/refresh-contract-fixtures.sh --source pubmed
./scripts/refresh-contract-fixtures.sh --source uniprot
./scripts/refresh-contract-fixtures.sh --source patents
./scripts/refresh-contract-fixtures.sh --source fda

# Dry run (shows what would change without writing files)
./scripts/refresh-contract-fixtures.sh --dry-run
```

Available `--source` values: `patents`, `fda`, `clinicaltrials`, `pubmed`, `uniprot`.

### BigQuery credentials

BigQuery sources require active Google Cloud credentials:

```bash
# One-time setup (developer workstations)
gcloud auth application-default login

# CI/CD — set the service account key file
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### PubMed rate limits

Without an API key, PubMed allows 3 requests/second.  For faster refreshes,
set `NCBI_API_KEY` in your environment (10 req/sec with a key):

```bash
export NCBI_API_KEY=your_key_here
./scripts/refresh-contract-fixtures.sh --source pubmed
```

### Reviewing diffs

The script diffs every refreshed fixture against the existing file and prints
unified diffs to stdout.  A non-empty diff indicates an upstream schema change.

**If the diff shows only value changes** (different dates, counts, titles) —
the schema is stable; the fixture is just stale.  Accept the new file.

**If the diff shows field additions or removals** — the upstream API schema
changed.  Review the diff carefully, then:

1. Accept the new fixture if the new schema is a superset (non-breaking).
2. Update the contract tests in `test_schema_compliance.py` to assert the new
   fields/structure.
3. Update any production parsing code that depended on the removed fields.
4. File a change request if the schema change affects downstream consumers.

### Fixture format

Fixtures are plain JSON files.  Array-type responses (patents, FDA rows) are
stored as JSON arrays.  Object-type responses (ClinicalTrials, PubMed, UniProt)
are stored as JSON objects matching the upstream API envelope shape.

Fixtures do **not** embed a `captured_at` timestamp inside the file itself
(to keep diffs clean).  The capture date is tracked in git history.

## Adding new fixtures

1. Add the live-capture logic to `scripts/refresh-contract-fixtures.sh`.
2. Run the script to produce the initial fixture file.
3. Add a `load_fixture("your_fixture_name")` call in `conftest.py`.
4. Write contract tests in a new or existing test file.
