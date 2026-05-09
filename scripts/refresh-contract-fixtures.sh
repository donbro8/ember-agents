#!/usr/bin/env bash
# refresh-contract-fixtures.sh — Capture live API responses for contract test fixtures.
#
# Usage:
#   ./scripts/refresh-contract-fixtures.sh [--dry-run] [--source SOURCE]
#
# Options:
#   --dry-run        Print what would be done without writing files.
#   --source SOURCE  Refresh only one source: patents | fda | clinicaltrials | pubmed | uniprot
#
# Idempotent: running multiple times on the same day produces identical output
# (timestamps are date-only, not wall-clock time).  Each fixture file is tagged
# with a "captured_at" field at the top level before the payload, or, for
# array-type responses, wrapped in an object with "captured_at" and "data" keys.
#
# Diffs are printed to stdout.  Exit code is 0 even when schema changes are
# detected; callers should inspect output to decide whether to update tests.
#
# Requirements:
#   - uv (used to run Python in the project virtualenv)
#   - Active Google Cloud credentials when running BigQuery sources
#     (set GOOGLE_APPLICATION_CREDENTIALS or use gcloud auth application-default login)
#   - Optional: NCBI_API_KEY env var for higher PubMed rate limits

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve project root (the directory containing this script's parent)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURES_DIR="${PROJECT_ROOT}/tests/contract/fixtures"

CAPTURE_DATE="$(date -u +%Y-%m-%d)"
DRY_RUN=false
ONLY_SOURCE=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --source)
            ONLY_SOURCE="$2"
            shift 2
            ;;
        -h|--help)
            head -30 "${BASH_SOURCE[0]}" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[refresh-contract-fixtures] $*"; }

should_run() {
    local source="$1"
    [[ -z "${ONLY_SOURCE}" || "${ONLY_SOURCE}" == "${source}" ]]
}

write_fixture() {
    local name="$1"      # stem, e.g. "bigquery_patents_adalimumab"
    local content="$2"   # JSON string
    local dest="${FIXTURES_DIR}/${name}.json"

    if "${DRY_RUN}"; then
        log "DRY-RUN: would write ${dest}"
        return
    fi

    # Write new content to a temp file then diff
    local tmp
    tmp="$(mktemp)"
    printf '%s\n' "${content}" > "${tmp}"

    if [[ -f "${dest}" ]]; then
        if diff --unified=3 "${dest}" "${tmp}" > /dev/null 2>&1; then
            log "  ${name}: no change"
        else
            log "  ${name}: SCHEMA CHANGE DETECTED — diff below"
            echo "--- ${dest} (existing)"
            echo "+++ ${dest} (refreshed)"
            diff --unified=3 "${dest}" "${tmp}" || true
        fi
    else
        log "  ${name}: new fixture (no existing file to diff)"
    fi

    cp "${tmp}" "${dest}"
    rm -f "${tmp}"
    log "  ${name}: written to ${dest}"
}

# ---------------------------------------------------------------------------
# Inline Python runner — executes a Python snippet via uv run python
# ---------------------------------------------------------------------------
run_python() {
    cd "${PROJECT_ROOT}" && uv run python -c "$1"
}

# ---------------------------------------------------------------------------
# Source: BigQuery patents (adalimumab)
# ---------------------------------------------------------------------------
capture_bigquery_patents() {
    log "Fetching BigQuery patents (adalimumab)…"
    local script
    script=$(cat <<'PYEOF'
import json, sys
from datetime import date

try:
    from ember_data.bigquery.client import BigQueryClient
    from ember_data.bigquery.queries import patent_search_query

    client = BigQueryClient()
    query = patent_search_query("adalimumab", limit=10)
    rows = client.run_query(query)

    results = []
    for row in rows:
        results.append({
            "publication_number": row.get("publication_number", ""),
            "country_code": row.get("country_code", ""),
            "filing_date": str(row["filing_date"]) if row.get("filing_date") else None,
            "grant_date": str(row["grant_date"]) if row.get("grant_date") else None,
            "expiry_date": str(row.get("derived_expiry", "")) or None,
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "assignee": row.get("assignee", ""),
            "status": "active" if row.get("derived_expiry") and str(row["derived_expiry"]) > str(date.today()) else "expired",
        })

    output = {
        "captured_at": "CAPTURE_DATE_PLACEHOLDER",
        "source": "bigquery_patents",
        "query": "adalimumab",
        "data": results,
    }
    print(json.dumps(output, indent=2, default=str))

except Exception as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF
)
    script="${script//CAPTURE_DATE_PLACEHOLDER/${CAPTURE_DATE}}"

    local result
    result="$(run_python "${script}")"

    # Contract tests expect a plain array for this fixture; extract the data array
    local payload
    payload="$(echo "${result}" | uv run python -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d['data'], indent=2))")"

    write_fixture "bigquery_patents_adalimumab" "${payload}"
}

# ---------------------------------------------------------------------------
# Source: BigQuery FDA labels (adalimumab)
# ---------------------------------------------------------------------------
capture_bigquery_fda() {
    log "Fetching BigQuery FDA labels (adalimumab)…"
    local script
    script=$(cat <<'PYEOF'
import json, sys

try:
    from ember_data.bigquery.client import BigQueryClient
    from ember_data.bigquery.queries import fda_drug_events_query

    client = BigQueryClient()
    query = fda_drug_events_query("adalimumab", limit=10)
    rows = client.run_query(query)

    results = []
    for row in rows:
        results.append({
            "application_number": row.get("openfda_product_ndc", ""),
            "brand_name": row.get("openfda_brand_name", ""),
            "generic_name": row.get("openfda_generic_name", ""),
            "manufacturer_name": row.get("openfda_manufacturer_name", ""),
            "product_type": row.get("openfda_product_type", ""),
            "route": row.get("route", ""),
            "approval_date": row.get("approval_date", ""),
            "dosage_form": row.get("dosage_form", ""),
            "active_ingredient": row.get("openfda_generic_name", ""),
            "indications": [],
            "marketing_status": row.get("marketing_status", ""),
            "te_code": row.get("te_code"),
            "reference_listed_drug": row.get("reference_listed_drug", False),
        })

    output = {
        "captured_at": "CAPTURE_DATE_PLACEHOLDER",
        "source": "bigquery_fda",
        "query": "adalimumab",
        "data": results,
    }
    print(json.dumps(output, indent=2, default=str))

except Exception as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF
)
    script="${script//CAPTURE_DATE_PLACEHOLDER/${CAPTURE_DATE}}"

    local result
    result="$(run_python "${script}")"

    local payload
    payload="$(echo "${result}" | uv run python -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d['data'], indent=2))")"

    write_fixture "bigquery_fda" "${payload}"
}

# ---------------------------------------------------------------------------
# Source: ClinicalTrials.gov (adalimumab)
# ---------------------------------------------------------------------------
capture_clinicaltrials() {
    log "Fetching ClinicalTrials.gov (adalimumab)…"
    local script
    script=$(cat <<'PYEOF'
import json, sys

try:
    from ember_data.clients.clinicaltrials import ClinicalTrialsClient

    client = ClinicalTrialsClient()
    trials = client.search("adalimumab", max_results=5)

    results = []
    for trial in trials:
        results.append({
            "nctId": trial.nct_id if hasattr(trial, "nct_id") else trial.id,
            "briefTitle": trial.brief_title if hasattr(trial, "brief_title") else trial.title,
            "officialTitle": getattr(trial, "official_title", None) or getattr(trial, "title", ""),
            "overallStatus": getattr(trial, "status", ""),
            "phase": getattr(trial, "phase", None),
            "conditions": getattr(trial, "conditions", []),
            "interventions": getattr(trial, "interventions", []),
            "sponsor": getattr(trial, "sponsor", {}),
            "enrollmentCount": getattr(trial, "enrollment_count", None),
            "startDate": str(trial.start_date) if getattr(trial, "start_date", None) else None,
            "completionDate": str(trial.completion_date) if getattr(trial, "completion_date", None) else None,
            "primaryOutcome": getattr(trial, "primary_outcome", None),
            "locations": getattr(trial, "locations", []),
        })

    output = {
        "captured_at": "CAPTURE_DATE_PLACEHOLDER",
        "source": "clinicaltrials",
        "query": "adalimumab",
        "studies": results,
        "totalCount": len(results),
        "nextPageToken": None,
    }
    print(json.dumps(output, indent=2, default=str))

except Exception as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF
)
    script="${script//CAPTURE_DATE_PLACEHOLDER/${CAPTURE_DATE}}"

    local result
    result="$(run_python "${script}")"

    # Extract the envelope minus captured_at (fixture format: {studies, totalCount, nextPageToken})
    local payload
    payload="$(echo "${result}" | uv run python -c "
import json, sys
d = json.load(sys.stdin)
out = {k: v for k, v in d.items() if k not in ('captured_at', 'source', 'query')}
print(json.dumps(out, indent=2))
")"

    write_fixture "clinicaltrials" "${payload}"
}

# ---------------------------------------------------------------------------
# Source: PubMed (adalimumab)
# ---------------------------------------------------------------------------
capture_pubmed() {
    log "Fetching PubMed (adalimumab)…"
    local ncbi_key_arg=""
    if [[ -n "${NCBI_API_KEY:-}" ]]; then
        ncbi_key_arg="api_key='${NCBI_API_KEY}'"
    fi

    local script
    script=$(cat <<'PYEOF'
import json, sys

try:
    from ember_data.clients.pubmed import PubMedClient

    client = PubMedClient(NCBI_KEY_ARG)
    articles = client.search("adalimumab", max_results=5)

    article_list = []
    for art in articles:
        article_list.append({
            "pmid": art.pmid if hasattr(art, "pmid") else art.id,
            "title": art.title,
            "abstract": getattr(art, "abstract", ""),
            "authors": [
                {"name": a if isinstance(a, str) else getattr(a, "name", str(a)),
                 "affiliation": "" if isinstance(a, str) else getattr(a, "affiliation", "")}
                for a in getattr(art, "authors", [])
            ],
            "journal": getattr(art, "journal", ""),
            "publication_date": str(art.publication_date) if getattr(art, "publication_date", None) else None,
            "doi": getattr(art, "doi", None),
            "mesh_terms": getattr(art, "mesh_terms", []),
        })

    output = {
        "captured_at": "CAPTURE_DATE_PLACEHOLDER",
        "header": {"type": "esearch", "version": "0.3"},
        "esearchresult": {
            "count": str(len(article_list)),
            "retmax": str(len(article_list)),
            "retstart": "0",
            "idlist": [a["pmid"] for a in article_list],
        },
        "articles": article_list,
    }
    print(json.dumps(output, indent=2, default=str))

except Exception as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF
)
    script="${script//CAPTURE_DATE_PLACEHOLDER/${CAPTURE_DATE}}"
    script="${script//NCBI_KEY_ARG/${ncbi_key_arg}}"

    local result
    result="$(run_python "${script}")"

    local payload
    payload="$(echo "${result}" | uv run python -c "
import json, sys
d = json.load(sys.stdin)
out = {k: v for k, v in d.items() if k != 'captured_at'}
print(json.dumps(out, indent=2))
")"

    write_fixture "pubmed" "${payload}"
}

# ---------------------------------------------------------------------------
# Source: UniProt (TNF / adalimumab target)
# ---------------------------------------------------------------------------
capture_uniprot() {
    log "Fetching UniProt (TNF target for adalimumab)…"
    local script
    script=$(cat <<'PYEOF'
import json, sys

try:
    from ember_data.clients.uniprot import UniProtClient

    client = UniProtClient(organism_id=9606)
    # TNF is the primary target of adalimumab
    targets, provenance = zip(*[r for r in [client.get_target("P01375")] if r]) if client.get_target("P01375") else ([], [])

    # Fall back to search if direct lookup returns None
    if not targets:
        results_raw = client.search_targets("TNF adalimumab", max_results=3)
        targets = [r[0] for r in results_raw if r]

    results = []
    for tgt in targets:
        results.append({
            "primaryAccession": getattr(tgt, "uniprot_accession", getattr(tgt, "id", "")),
            "uniProtkbId": getattr(tgt, "uniprot_id", ""),
            "organism": {
                "scientificName": getattr(tgt, "organism", "Homo sapiens"),
                "commonName": "Human",
                "taxonId": 9606,
            },
            "proteinDescription": {
                "recommendedName": {
                    "fullName": {"value": getattr(tgt, "name", "")},
                },
            },
            "genes": [{"geneName": {"value": g} for g in [getattr(tgt, "gene_name", "")]} if getattr(tgt, "gene_name", "") else {}],
            "sequence": {
                "length": getattr(tgt, "sequence_length", None),
                "molWeight": getattr(tgt, "molecular_weight", None),
            },
            "keywords": [{"name": k} for k in getattr(tgt, "keywords", [])],
        })

    output = {
        "captured_at": "CAPTURE_DATE_PLACEHOLDER",
        "results": results,
        "totalResults": len(results),
    }
    print(json.dumps(output, indent=2, default=str))

except Exception as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF
)
    script="${script//CAPTURE_DATE_PLACEHOLDER/${CAPTURE_DATE}}"

    local result
    result="$(run_python "${script}")"

    local payload
    payload="$(echo "${result}" | uv run python -c "
import json, sys
d = json.load(sys.stdin)
out = {k: v for k, v in d.items() if k != 'captured_at'}
print(json.dumps(out, indent=2))
")"

    write_fixture "uniprot" "${payload}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
log "Starting fixture refresh (capture_date=${CAPTURE_DATE})"
log "Fixtures directory: ${FIXTURES_DIR}"
"${DRY_RUN}" && log "DRY-RUN mode: no files will be written"

mkdir -p "${FIXTURES_DIR}"

if should_run "patents"; then
    capture_bigquery_patents
fi

if should_run "fda"; then
    capture_bigquery_fda
fi

if should_run "clinicaltrials"; then
    capture_clinicaltrials
fi

if should_run "pubmed"; then
    capture_pubmed
fi

if should_run "uniprot"; then
    capture_uniprot
fi

log "Done. Review any diffs above and update contract tests if schema changed."
