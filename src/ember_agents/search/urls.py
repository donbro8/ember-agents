"""URL builders for verified source links."""

from __future__ import annotations

from urllib.parse import quote


def trial_url(nct_id: str) -> str | None:
    """ClinicalTrials.gov study page."""
    if not nct_id:
        return None
    return f"https://clinicaltrials.gov/study/{nct_id}"


def pubmed_url(pmid: str | int | None = None, doi: str | None = None) -> str | None:
    """PubMed article page (prefers PMID, falls back to DOI)."""
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
    if doi:
        return f"https://doi.org/{doi}"
    return None


def uniprot_url(accession: str) -> str | None:
    """UniProt protein page."""
    if not accession:
        return None
    return f"https://www.uniprot.org/uniprotkb/{accession}"


def patent_url(publication_number: str) -> str | None:
    """Google Patents page."""
    if not publication_number:
        return None
    return f"https://patents.google.com/patent/{publication_number}"


def drug_label_url(drug_name: str) -> str | None:
    """FDA DailyMed search."""
    if not drug_name:
        return None
    return f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query={quote(drug_name)}"
