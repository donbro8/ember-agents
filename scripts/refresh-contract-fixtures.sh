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
# (timestamps are date-only, not wall-clock time).
#
# Diffs are printed to stdout.  Exit code is 0 even when schema changes are
# detected; callers should inspect output to decide whether to update tests.
#
# Requirements:
#   - uv (used to run Python in the project virtualenv)
#   - Active Google Cloud credentials when running BigQuery sources
#     (set GOOGLE_APPLICATION_CREDENTIALS or use gcloud auth application-default login)
#   - GCP_PROJECT_ID env var (for BigQuery client)
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
import json, os, sys
from datetime import date

try:
    from ember_data.bigquery.client import BigQueryClient

    project_id = os.environ.get("GCP_PROJECT_ID", "")
    if not project_id:
        print("ERROR: GCP_PROJECT_ID env var not set", file=sys.stderr)
        sys.exit(1)

    client = BigQueryClient(project_id=project_id)
    rows = client.search_patents("adalimumab", limit=10)

    results = []
    for row in rows:
        results.append({
            "publication_number": row.get("publication_number", ""),
            "country_code": row.get("country_code", ""),
            "filing_date": str(row["filing_date"]) if row.get("filing_date") else None,
            "grant_date": str(row["grant_date"]) if row.get("grant_date") else None,
            "expiry_date": str(row.get("derived_expiry", "")) or None,
            "expiry_approximate": row.get("expiry_approximate", True),
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "assignee": row.get("assignee", ""),
            "status": "active" if row.get("derived_expiry") and str(row["derived_expiry"]) > str(date.today()) else "expired",
        })

    print(json.dumps(results, indent=2, default=str))

except Exception as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF
)

    local result
    result="$(run_python "${script}")"
    write_fixture "bigquery_patents_adalimumab" "${result}"
}

# ---------------------------------------------------------------------------
# Source: BigQuery FDA labels (adalimumab)
# ---------------------------------------------------------------------------
capture_bigquery_fda() {
    log "Fetching BigQuery FDA labels (adalimumab)…"
    local script
    script=$(cat <<'PYEOF'
import json, os, sys

try:
    from ember_data.bigquery.client import BigQueryClient

    project_id = os.environ.get("GCP_PROJECT_ID", "")
    if not project_id:
        print("ERROR: GCP_PROJECT_ID env var not set", file=sys.stderr)
        sys.exit(1)

    client = BigQueryClient(project_id=project_id)
    rows = client.query_fda_drug_events("adalimumab", limit=10)

    results = []
    for row in rows:
        results.append({
            "application_number": row.get("openfda_product_ndc", ""),
            "brand_name": row.get("openfda_brand_name", ""),
            "generic_name": row.get("openfda_generic_name", ""),
            "manufacturer_name": row.get("openfda_manufacturer_name", ""),
            "product_type": row.get("openfda_product_type", ""),
            "route": row.get("route", ""),
        })

    print(json.dumps(results, indent=2, default=str))

except Exception as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF
)

    local result
    result="$(run_python "${script}")"
    write_fixture "bigquery_fda" "${result}"
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
            "nctId": getattr(trial, "nct_id", getattr(trial, "id", "")),
            "briefTitle": getattr(trial, "brief_title", getattr(trial, "title", "")),
            "overallStatus": getattr(trial, "status", ""),
            "phase": getattr(trial, "phase", None),
            "conditions": getattr(trial, "conditions", []),
            "startDate": str(trial.start_date) if getattr(trial, "start_date", None) else None,
            "completionDate": str(trial.completion_date) if getattr(trial, "completion_date", None) else None,
        })

    output = {
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

    local result
    result="$(run_python "${script}")"
    write_fixture "clinicaltrials" "${result}"
}

# ---------------------------------------------------------------------------
# Source: PubMed (adalimumab)
# ---------------------------------------------------------------------------
capture_pubmed() {
    log "Fetching PubMed (adalimumab)…"
    local script
    script=$(cat <<'PYEOF'
import json, os, sys

try:
    from ember_data.clients.pubmed import PubMedClient

    api_key = os.environ.get("NCBI_API_KEY")
    client = PubMedClient(api_key=api_key) if api_key else PubMedClient()
    articles = client.search("adalimumab", max_results=5)

    article_list = []
    for art in articles:
        article_list.append({
            "pmid": getattr(art, "pmid", getattr(art, "id", "")),
            "title": getattr(art, "title", ""),
            "abstract": getattr(art, "abstract", ""),
            "authors": [
                {"name": a if isinstance(a, str) else getattr(a, "name", str(a))}
                for a in getattr(art, "authors", [])
            ],
            "journal": getattr(art, "journal", ""),
            "publication_date": str(art.publication_date) if getattr(art, "publication_date", None) else None,
            "doi": getattr(art, "doi", None),
        })

    output = {
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

    local result
    result="$(run_python "${script}")"
    write_fixture "pubmed" "${result}"
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
    entry = client.get_target("P01375")

    results = []
    if entry:
        tgt = entry[0] if isinstance(entry, tuple) else entry
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
            "genes": [{"geneName": {"value": getattr(tgt, "gene_name", "")}}] if getattr(tgt, "gene_name", "") else [],
            "sequence": {
                "length": getattr(tgt, "sequence_length", None),
                "molWeight": getattr(tgt, "molecular_weight", None),
            },
            "keywords": [{"name": k} for k in getattr(tgt, "keywords", [])],
        })

    output = {
        "results": results,
        "totalResults": len(results),
    }
    print(json.dumps(output, indent=2, default=str))

except Exception as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF
)

    local result
    result="$(run_python "${script}")"
    write_fixture "uniprot" "${result}"
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
