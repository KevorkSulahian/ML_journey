# %% [markdown]
# Legal DSPy Deep Research Planner

# %% [markdown]
# # Legal DSPy Deep Research Planner
#
# This notebook builds a U.S.-federal-first legal research workflow for long-form reasoning, rule synthesis, and application of law to facts. It prefers legal-specific retrieval first, uses `dspy.RLM` only for large authority bundles, and falls back to Tavily only when the primary legal adapters do not return usable authority.

# %%
import os
import re
import json
import html
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Literal
from urllib.parse import urljoin, urlparse

import requests
from dotenv import load_dotenv
import dspy

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is not set. Add it to your environment or .env file.")
    return value


OPENAI_API_KEY = require_env("OPENAI_API_KEY")
GOVINFO_API_KEY = require_env("GOVINFO_API_KEY")
COURTLISTENER_API_KEY = os.getenv("COURTLISTENER_API_KEY")
CONGRESS_API_KEY = os.getenv("CONGRESS_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

MAIN_MODEL_NAME = os.getenv("LEGAL_DSPY_MAIN_MODEL", "openai/gpt-4o-mini")
SUB_MODEL_NAME = os.getenv("LEGAL_DSPY_SUB_MODEL", "openai/gpt-4o-mini")

main_lm = dspy.LM(MAIN_MODEL_NAME, api_key=OPENAI_API_KEY, max_tokens=8000)
sub_lm = dspy.LM(SUB_MODEL_NAME, api_key=OPENAI_API_KEY, max_tokens=4000)
dspy.configure(lm=main_lm)

HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "MLJourneyLegalResearch/1.0"})

print(f"DSPy configured with main={MAIN_MODEL_NAME} sub={SUB_MODEL_NAME}")
print(f"CourtListener token detected: {bool(COURTLISTENER_API_KEY)}")
print(f"Congress token detected: {bool(CONGRESS_API_KEY)}")
print(f"Tavily token detected: {bool(TAVILY_API_KEY)}")

# %%
BUDGETS = {
    "light": {
        "max_issues": 3,
        "queries_per_issue": 2,
        "courtlistener_results": 4,
        "govinfo_packages_per_collection": 6,
        "congress_results": 4,
        "selected_sources_per_issue": 2,
    },
    "medium": {
        "max_issues": 5,
        "queries_per_issue": 3,
        "courtlistener_results": 6,
        "govinfo_packages_per_collection": 10,
        "congress_results": 6,
        "selected_sources_per_issue": 3,
    },
    "deep": {
        "max_issues": 7,
        "queries_per_issue": 4,
        "courtlistener_results": 8,
        "govinfo_packages_per_collection": 14,
        "congress_results": 8,
        "selected_sources_per_issue": 4,
    },
}

LEGAL_SOURCE_THRESHOLDS = {
    "single_doc_chars_for_rlm": 20_000,
    "issue_bundle_chars_for_rlm": 60_000,
}

ARTIFACT_OUTPUT = "output/legal_research_artifact.json"
REPORT_OUTPUT = "output/legal_research_memo.md"

manual_legal_research_request = {
    "legal_question": "What negligence arguments are strongest for the injured plaintiff, and what counterarguments is the defendant likely to raise?",
    "fact_pattern": "A delivery company sent a driver to a downtown office tower during a heavy rainstorm. The building owner knew for weeks that the marble lobby floor became extremely slick when water pooled near the main entrance, and staff had previously reported two near-falls. On the day of the accident, no warning signs or mats were placed near the doorway. A visitor entering the lobby slipped, fractured her wrist, and missed six weeks of work. The building owner argues that the danger was open and obvious because it was raining and the visitor should have watched where she was going.",
    "jurisdiction": "United States (Federal)",
    "requested_work_product": "legal_memo",
    "known_issues": [],
    "preferred_authority_types": ["cases", "statutes", "regulations"],
    "allowed_domains": [],
    "allow_tavily_fallback": True,
    "budget_mode": "deep",
}

DATASET_INPUT_CONFIG = {
    "enabled": True,
    "split": "test",
    "question_contains": "American law",
    "row_index": 0,
}


def load_lexam_request() -> tuple[dict[str, Any], dict[str, Any]]:
    if not DATASET_INPUT_CONFIG["enabled"]:
        return manual_legal_research_request, {
            "request_source": "manual",
            "reason": "Dataset input disabled.",
        }

    if load_dataset is None:
        return manual_legal_research_request, {
            "request_source": "manual",
            "reason": "datasets library is unavailable; using manual fallback.",
        }

    dataset = load_dataset("LEXam-Benchmark/LEXam", "open_question", split=DATASET_INPUT_CONFIG["split"])
    matching_rows = [
        row
        for row in dataset
        if row.get("language") == "en"
        and DATASET_INPUT_CONFIG["question_contains"].lower() in (row.get("question") or "").lower()
    ]
    if not matching_rows:
        return manual_legal_research_request, {
            "request_source": "manual",
            "reason": "No matching LEXam rows found; using manual fallback.",
        }

    selected_row = matching_rows[DATASET_INPUT_CONFIG["row_index"]]
    question = (selected_row.get("question") or "").strip()
    request = {
        "legal_question": question,
        "fact_pattern": question,
        "jurisdiction": "United States (Federal)",
        "requested_work_product": "legal_memo",
        "known_issues": [],
        "preferred_authority_types": ["cases"],
        "allowed_domains": [],
        "allow_tavily_fallback": False,
        "budget_mode": manual_legal_research_request["budget_mode"],
        "reference_answer": selected_row.get("answer"),
        "dataset_source": {
            "provider": "LEXam-Benchmark/LEXam",
            "config": "open_question",
            "split": DATASET_INPUT_CONFIG["split"],
            "id": selected_row.get("id"),
            "course": selected_row.get("course"),
            "language": selected_row.get("language"),
            "area": selected_row.get("area"),
            "raw_jurisdiction": selected_row.get("jurisdiction"),
            "year": selected_row.get("year"),
        },
    }
    return request, {
        "request_source": "lexam",
        "dataset": request["dataset_source"],
    }


legal_research_request, request_source_metadata = load_lexam_request()

budget = BUDGETS[legal_research_request["budget_mode"]]
print(
    json.dumps(
        {
            "request_source_metadata": request_source_metadata,
            "legal_research_request": legal_research_request,
        },
        indent=2,
        ensure_ascii=False,
    )
)

# %%
@dataclass
class SourceCandidate:
    issue: str
    query: str
    title: str
    url: str
    snippet: str
    source_family: str
    source_type: str
    authority_level: str
    retrieval_adapter: str
    jurisdiction: str
    date: Optional[str] = None
    package_id: Optional[str] = None
    opinion_id: Optional[int] = None
    is_fallback: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class SourceDocument:
    issue: str
    title: str
    url: str
    source_family: str
    source_type: str
    authority_level: str
    retrieval_adapter: str
    jurisdiction: str
    content: str
    date: Optional[str] = None
    source_urls: list[str] = field(default_factory=list)
    is_fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthorityDigest:
    issue: str
    title: str
    url: str
    source_family: str
    source_type: str
    authority_level: str
    retrieval_adapter: str
    jurisdiction: str
    digest_mode: Literal["short", "rlm_bundle"]
    source_urls: list[str]
    is_fallback: bool
    governing_rules: list[str]
    key_holdings: list[str]
    application_to_facts: list[str]
    counterarguments: list[str]
    uncertainty_flags: list[str]
    supporting_quotes: list[str]
    digest_summary: str


@dataclass
class IssueAnalysis:
    issue: str
    queries: list[str]
    selected_sources: list[dict]
    analysis_mode: Literal["short", "rlm_bundle", "no_sources"]
    governing_rules: list[str]
    application_to_facts: list[str]
    counterarguments: list[str]
    uncertainty_flags: list[str]
    cited_urls: list[str]
    issue_summary: str


def assert_supported_jurisdiction(jurisdiction: str) -> None:
    normalized = jurisdiction.lower()
    if "united states" not in normalized and "federal" not in normalized:
        raise ValueError(
            "This notebook is federal-first in v1. Use a jurisdiction like 'United States (Federal)' instead of a state-only scope."
        )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def html_to_text(raw_html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return normalize_text(text)


def keyword_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", (text or "").lower())
    stop = {
        "the", "and", "that", "with", "from", "this", "have", "will", "into", "your", "about", "under", "what", "when", "where", "which", "while", "would", "could", "should", "there", "their", "them", "then", "than", "because", "against", "likely", "federal", "united", "states",
    }
    return [token for token in tokens if token not in stop]


def query_score(query: str, text: str) -> int:
    query_terms = keyword_tokens(query)
    haystack = (text or "").lower()
    overlap = sum(1 for term in query_terms if term in haystack)
    phrase_bonus = 2 if query.lower() in haystack else 0
    return overlap + phrase_bonus


def authority_priority(candidate: SourceCandidate) -> tuple:
    authority_rank = {
        "primary": 4,
        "official": 3,
        "legislative": 2,
        "secondary": 1,
    }
    family_rank = {
        "case_law": 4,
        "statute": 4,
        "regulation": 4,
        "legislative_material": 2,
        "secondary": 1,
    }
    return (
        0 if candidate.is_fallback else 1,
        authority_rank.get(candidate.authority_level, 0),
        family_rank.get(candidate.source_family, 0),
        candidate.date or "",
    )


def dedupe_candidates(candidates: list[SourceCandidate]) -> list[SourceCandidate]:
    seen: set[str] = set()
    deduped: list[SourceCandidate] = []
    for candidate in candidates:
        key = candidate.url.split("#")[0]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def maybe_limit_text(text: str, max_chars: int = 60_000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED FOR NOTEBOOK DISPLAY]"


def allowed_domain_match(url: str, allowed_domains: list[str]) -> bool:
    if not allowed_domains:
        return True
    host = urlparse(url).netloc.lower()
    return any(domain.lower() in host for domain in allowed_domains)


def build_source_bundle(documents: list[SourceDocument]) -> str:
    parts = []
    for idx, doc in enumerate(documents, start=1):
        parts.append(
            "\n".join(
                [
                    f"[SOURCE {idx}]",
                    f"TITLE: {doc.title}",
                    f"URL: {doc.url}",
                    f"SOURCE_FAMILY: {doc.source_family}",
                    f"SOURCE_TYPE: {doc.source_type}",
                    f"AUTHORITY_LEVEL: {doc.authority_level}",
                    f"IS_FALLBACK: {doc.is_fallback}",
                    "TEXT:",
                    doc.content,
                ]
            )
        )
    return "\n\n".join(parts)


def print_candidate_table(candidates: list[SourceCandidate]) -> None:
    for candidate in candidates:
        fallback = "fallback" if candidate.is_fallback else "primary"
        print(f"- [{fallback}] {candidate.title} | {candidate.source_family} | {candidate.retrieval_adapter} | {candidate.url}")

# %%
def courtlistener_auth_headers() -> dict[str, str]:
    if not COURTLISTENER_API_KEY:
        return {}
    return {"Authorization": f"Token {COURTLISTENER_API_KEY}"}


def search_courtlistener_cases(issue: str, query: str, limit: int) -> list[SourceCandidate]:
    params = {
        "q": query,
        "type": "o",
        "order_by": "score desc",
    }
    response = HTTP.get("https://www.courtlistener.com/api/rest/v4/search/", params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    candidates: list[SourceCandidate] = []
    for result in payload.get("results", [])[:limit]:
        opinion_meta = (result.get("opinions") or [{}])[0]
        snippet = normalize_text(opinion_meta.get("snippet") or result.get("snippet") or "")
        citations = ", ".join(result.get("citation") or [])
        title = result.get("caseName") or result.get("caseNameFull") or "Untitled opinion"
        if citations:
            title = f"{title} ({citations})"

        candidates.append(
            SourceCandidate(
                issue=issue,
                query=query,
                title=title,
                url=urljoin("https://www.courtlistener.com", result.get("absolute_url", "")),
                snippet=snippet,
                source_family="case_law",
                source_type="case",
                authority_level="primary",
                retrieval_adapter="courtlistener",
                jurisdiction=result.get("court") or "United States (Federal)",
                date=result.get("dateFiled"),
                opinion_id=opinion_meta.get("id"),
                notes=["CourtListener search result"],
            )
        )

    return candidates


def fetch_courtlistener_document(candidate: SourceCandidate) -> SourceDocument:
    content_parts = [candidate.title, candidate.snippet]
    metadata: dict[str, Any] = {
        "retrieval_note": "CourtListener snippet used",
    }

    if candidate.opinion_id and COURTLISTENER_API_KEY:
        detail_url = f"https://www.courtlistener.com/api/rest/v4/opinions/{candidate.opinion_id}/"
        detail_response = HTTP.get(detail_url, headers=courtlistener_auth_headers(), timeout=30)
        if detail_response.ok:
            detail = detail_response.json()
            raw_text = detail.get("plain_text") or detail.get("html_with_citations") or detail.get("html") or detail.get("html_lawbox") or ""
            extracted = html_to_text(raw_text) if "<" in raw_text else normalize_text(raw_text)
            if extracted:
                content_parts.append(extracted)
                metadata["retrieval_note"] = "CourtListener opinion detail used"
                metadata["download_url"] = detail.get("download_url")

    return SourceDocument(
        issue=candidate.issue,
        title=candidate.title,
        url=candidate.url,
        source_family=candidate.source_family,
        source_type=candidate.source_type,
        authority_level=candidate.authority_level,
        retrieval_adapter=candidate.retrieval_adapter,
        jurisdiction=candidate.jurisdiction,
        content=normalize_text("\n\n".join(part for part in content_parts if part)),
        date=candidate.date,
        source_urls=[candidate.url],
        is_fallback=candidate.is_fallback,
        metadata=metadata,
    )


def govinfo_collections_for_request(preferred_authority_types: list[str]) -> list[str]:
    collections: list[str] = []
    if "statutes" in preferred_authority_types:
        collections.extend(["USCODE", "PLAW"])
    if "regulations" in preferred_authority_types:
        collections.extend(["CFR", "FR"])
    if not collections:
        collections = ["USCODE", "CFR", "FR"]
    return collections


def govinfo_window_for_collection(collection: str) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    if collection in {"USCODE", "PLAW"}:
        start = datetime(now.year - 3, 1, 1, tzinfo=timezone.utc)
    else:
        start = now - timedelta(days=365)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), now.strftime("%Y-%m-%dT%H:%M:%SZ")


def search_govinfo_materials(issue: str, query: str, preferred_authority_types: list[str], limit: int) -> list[SourceCandidate]:
    candidates: list[SourceCandidate] = []
    for collection in govinfo_collections_for_request(preferred_authority_types):
        start, end = govinfo_window_for_collection(collection)
        url = f"https://api.govinfo.gov/collections/{collection}/{start}/{end}"
        response = HTTP.get(
            url,
            params={
                "offset": 0,
                "pageSize": limit,
                "api_key": GOVINFO_API_KEY,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        for package in payload.get("packages", []):
            score = query_score(query, f"{package.get('title', '')} {package.get('docClass', '')} {package.get('packageId', '')}")
            if score <= 0:
                continue

            if collection in {"USCODE", "PLAW"}:
                source_family = "statute"
                authority_level = "official"
            elif collection in {"CFR", "FR"}:
                source_family = "regulation"
                authority_level = "official"
            else:
                source_family = "legislative_material"
                authority_level = "official"

            details_url = f"https://www.govinfo.gov/app/details/{package['packageId']}"
            candidates.append(
                SourceCandidate(
                    issue=issue,
                    query=query,
                    title=package.get("title") or package.get("packageId") or "Untitled govinfo package",
                    url=details_url,
                    snippet=f"GovInfo {collection} package {package.get('packageId')}",
                    source_family=source_family,
                    source_type=collection.lower(),
                    authority_level=authority_level,
                    retrieval_adapter="govinfo",
                    jurisdiction="United States (Federal)",
                    date=package.get("dateIssued"),
                    package_id=package.get("packageId"),
                    notes=[f"GovInfo collection {collection}"],
                )
            )

    return candidates


def fetch_govinfo_document(candidate: SourceCandidate) -> SourceDocument:
    if not candidate.package_id:
        raise ValueError("GovInfo candidates require package_id.")

    summary_url = f"https://api.govinfo.gov/packages/{candidate.package_id}/summary"
    summary_response = HTTP.get(summary_url, params={"api_key": GOVINFO_API_KEY}, timeout=30)
    summary_response.raise_for_status()
    summary = summary_response.json()

    download = summary.get("download") or {}
    text_url = download.get("txtLink") or download.get("xmlLink")
    content = candidate.snippet
    if text_url:
        text_response = HTTP.get(text_url, params={"api_key": GOVINFO_API_KEY}, timeout=30)
        text_response.raise_for_status()
        body = text_response.text
        content = html_to_text(body) if "<" in body else normalize_text(body)

    summary_text = json.dumps(
        {
            "title": summary.get("title"),
            "shortTitle": summary.get("shortTitle"),
            "dateIssued": summary.get("dateIssued"),
            "collectionName": summary.get("collectionName"),
            "references": summary.get("references"),
        },
        ensure_ascii=False,
    )

    return SourceDocument(
        issue=candidate.issue,
        title=summary.get("title") or candidate.title,
        url=summary.get("detailsLink") or candidate.url,
        source_family=candidate.source_family,
        source_type=candidate.source_type,
        authority_level=candidate.authority_level,
        retrieval_adapter=candidate.retrieval_adapter,
        jurisdiction=candidate.jurisdiction,
        content=normalize_text(summary_text + "\n\n" + content),
        date=summary.get("dateIssued") or candidate.date,
        source_urls=[summary.get("detailsLink") or candidate.url],
        is_fallback=candidate.is_fallback,
        metadata=summary,
    )


def needs_legislative_materials(request: dict[str, Any]) -> bool:
    text = f"{request['legal_question']} {request['fact_pattern']} {' '.join(request['preferred_authority_types'])}".lower()
    return any(term in text for term in ["bill", "congress", "legislation", "legislative", "statutory amendment"])


def search_congress_materials(issue: str, query: str, limit: int) -> list[SourceCandidate]:
    if not CONGRESS_API_KEY:
        return []

    response = HTTP.get(
        "https://api.congress.gov/v3/bill",
        params={
            "api_key": CONGRESS_API_KEY,
            "format": "json",
            "limit": max(10, limit * 5),
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    candidates: list[SourceCandidate] = []
    for bill in payload.get("bills", []):
        title = bill.get("title") or "Untitled bill"
        if query_score(query, title) <= 0:
            continue
        bill_type = str(bill.get("type", "")).lower()
        number = str(bill.get("number", "")).lower()
        congress = bill.get("congress")
        detail_url = bill.get("url") or f"https://api.congress.gov/v3/bill/{congress}/{bill_type}/{number}?format=json"
        candidates.append(
            SourceCandidate(
                issue=issue,
                query=query,
                title=title,
                url=detail_url,
                snippet=bill.get("latestAction", {}).get("text", ""),
                source_family="legislative_material",
                source_type="bill",
                authority_level="legislative",
                retrieval_adapter="congress",
                jurisdiction="United States (Federal)",
                date=bill.get("updateDate"),
                notes=["Congress.gov bill listing"],
            )
        )
        if len(candidates) >= limit:
            break

    return candidates


def fetch_congress_document(candidate: SourceCandidate) -> SourceDocument:
    if not CONGRESS_API_KEY:
        raise ValueError("Congress adapter requires CONGRESS_API_KEY to fetch bill details.")

    response = HTTP.get(candidate.url, params={"api_key": CONGRESS_API_KEY, "format": "json"}, timeout=30)
    response.raise_for_status()
    payload = response.json().get("bill", {})

    content = {
        "title": payload.get("title"),
        "introducedDate": payload.get("introducedDate"),
        "policyArea": payload.get("policyArea"),
        "latestAction": payload.get("latestAction"),
        "constitutionalAuthorityStatementText": html_to_text(payload.get("constitutionalAuthorityStatementText", "")),
        "subjects": payload.get("subjects"),
        "summaries": payload.get("summaries"),
    }

    return SourceDocument(
        issue=candidate.issue,
        title=payload.get("title") or candidate.title,
        url=payload.get("url") or candidate.url,
        source_family=candidate.source_family,
        source_type=candidate.source_type,
        authority_level=candidate.authority_level,
        retrieval_adapter=candidate.retrieval_adapter,
        jurisdiction=candidate.jurisdiction,
        content=normalize_text(json.dumps(content, ensure_ascii=False)),
        date=payload.get("updateDate") or candidate.date,
        source_urls=[payload.get("url") or candidate.url],
        is_fallback=candidate.is_fallback,
        metadata=payload,
    )


def search_tavily_fallback(issue: str, query: str, allowed_domains: list[str], limit: int) -> list[SourceCandidate]:
    if not TAVILY_API_KEY or TavilyClient is None:
        return []

    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(
        query,
        search_depth="advanced",
        max_results=limit,
        include_domains=allowed_domains or None,
        include_raw_content=False,
    )

    candidates: list[SourceCandidate] = []
    for result in response.get("results", []):
        url = result.get("url", "")
        if not allowed_domain_match(url, allowed_domains):
            continue
        candidates.append(
            SourceCandidate(
                issue=issue,
                query=query,
                title=result.get("title") or url,
                url=url,
                snippet=normalize_text(result.get("content") or ""),
                source_family="secondary",
                source_type="web",
                authority_level="secondary",
                retrieval_adapter="tavily",
                jurisdiction="Unspecified",
                is_fallback=True,
                notes=["Tavily fallback used because legal adapters returned no usable authority"],
            )
        )
    return candidates


def fetch_generic_web_document(candidate: SourceCandidate) -> SourceDocument:
    response = HTTP.get(candidate.url, timeout=30)
    response.raise_for_status()
    body = response.text
    content = html_to_text(body) if "<" in body else normalize_text(body)
    return SourceDocument(
        issue=candidate.issue,
        title=candidate.title,
        url=candidate.url,
        source_family=candidate.source_family,
        source_type=candidate.source_type,
        authority_level=candidate.authority_level,
        retrieval_adapter=candidate.retrieval_adapter,
        jurisdiction=candidate.jurisdiction,
        content=content,
        date=candidate.date,
        source_urls=[candidate.url],
        is_fallback=candidate.is_fallback,
        metadata={"retrieval_note": "Generic web fetch"},
    )


def fetch_source_document(candidate: SourceCandidate) -> SourceDocument:
    if candidate.retrieval_adapter == "courtlistener":
        return fetch_courtlistener_document(candidate)
    if candidate.retrieval_adapter == "govinfo":
        return fetch_govinfo_document(candidate)
    if candidate.retrieval_adapter == "congress":
        return fetch_congress_document(candidate)
    return fetch_generic_web_document(candidate)


def run_citation_lookup(text: str) -> dict[str, Any]:
    if not COURTLISTENER_API_KEY:
        return {
            "enabled": False,
            "reason": "COURTLISTENER_API_KEY not set; skipping citation lookup guardrail.",
        }

    response = HTTP.post(
        "https://www.courtlistener.com/api/rest/v3/citation-lookup/",
        headers=courtlistener_auth_headers(),
        data={"text": text[:64_000]},
        timeout=30,
    )
    response.raise_for_status()
    return {
        "enabled": True,
        "results": response.json(),
    }

# %%
class IssueExtractionSignature(dspy.Signature):
    """Extract the major legal issues that should be researched from the question and fact pattern."""

    legal_question: str = dspy.InputField()
    fact_pattern: str = dspy.InputField()
    user_guidance_summary: str = dspy.InputField()
    known_issues: list[str] = dspy.InputField()
    preferred_authority_types: list[str] = dspy.InputField()
    max_issues: int = dspy.InputField()
    issues_to_research: list[str] = dspy.OutputField()


class ClarifyingSignature(dspy.Signature):
    """Ask the user what to focus on before research begins.

    Always include questions about:
    1. the substantive focus or issue emphasis,
    2. which authority types or source families to use or avoid,
    3. whether GovInfo, Congress, or other sources should be included.
    """

    legal_question: str = dspy.InputField()
    fact_pattern: str = dspy.InputField()
    requested_work_product: str = dspy.InputField()
    number_of_questions: int = dspy.InputField()
    clarifying_questions: list[str] = dspy.OutputField()


class GuidanceNormalizationSignature(dspy.Signature):
    """Convert free-form user answers into concrete research controls.

    Keep approved_authority_types limited to:
    - cases
    - statutes
    - regulations

    Keep approved_source_adapters limited to:
    - courtlistener
    - govinfo
    - congress
    - tavily
    """

    legal_question: str = dspy.InputField()
    fact_pattern: str = dspy.InputField()
    clarifying_questions_and_answers: list[dict] = dspy.InputField()
    current_authority_types: list[str] = dspy.InputField()
    current_source_adapters: list[str] = dspy.InputField()
    focused_issues: list[str] = dspy.OutputField()
    approved_authority_types: list[str] = dspy.OutputField()
    approved_source_adapters: list[str] = dspy.OutputField()
    user_guidance_summary: str = dspy.OutputField()


class QueryPlanningSignature(dspy.Signature):
    """Generate legal-research queries for one issue."""

    legal_question: str = dspy.InputField()
    fact_pattern: str = dspy.InputField()
    issue: str = dspy.InputField()
    jurisdiction: str = dspy.InputField()
    user_guidance_summary: str = dspy.InputField()
    preferred_authority_types: list[str] = dspy.InputField()
    number_of_queries: int = dspy.InputField()
    queries: list[str] = dspy.OutputField()


class SourceSelectionSignature(dspy.Signature):
    """Pick the strongest authorities for a single issue. Prefer primary and official authority over secondary sources."""

    legal_question: str = dspy.InputField()
    fact_pattern: str = dspy.InputField()
    issue: str = dspy.InputField()
    user_guidance_summary: str = dspy.InputField()
    candidate_sources: list[dict] = dspy.InputField()
    max_sources: int = dspy.InputField()
    selected_urls: list[str] = dspy.OutputField()
    selection_rationale: str = dspy.OutputField()


class ShortAuthorityDigestSignature(dspy.Signature):
    """Digest a short legal authority and extract rules, holdings, and fact application."""

    legal_question: str = dspy.InputField()
    fact_pattern: str = dspy.InputField()
    issue: str = dspy.InputField()
    source_title: str = dspy.InputField()
    source_url: str = dspy.InputField()
    source_type: str = dspy.InputField()
    source_text: str = dspy.InputField()
    governing_rules: list[str] = dspy.OutputField()
    key_holdings: list[str] = dspy.OutputField()
    application_to_facts: list[str] = dspy.OutputField()
    counterarguments: list[str] = dspy.OutputField()
    uncertainty_flags: list[str] = dspy.OutputField()
    supporting_quotes: list[str] = dspy.OutputField()
    digest_summary: str = dspy.OutputField()


class IssueSynthesisSignature(dspy.Signature):
    """Synthesize multiple authority digests into one issue-level analysis."""

    legal_question: str = dspy.InputField()
    fact_pattern: str = dspy.InputField()
    issue: str = dspy.InputField()
    authority_digests: list[dict] = dspy.InputField()
    issue_summary: str = dspy.OutputField()
    governing_rules: list[str] = dspy.OutputField()
    application_to_facts: list[str] = dspy.OutputField()
    counterarguments: list[str] = dspy.OutputField()
    uncertainty_flags: list[str] = dspy.OutputField()
    cited_urls: list[str] = dspy.OutputField()


class MemoWriterSignature(dspy.Signature):
    """Write a legal memo in markdown with governing rules, fact application, counterarguments, and uncertainties."""

    legal_question: str = dspy.InputField()
    fact_pattern: str = dspy.InputField()
    jurisdiction: str = dspy.InputField()
    issue_analyses: list[dict] = dspy.InputField()
    memo_markdown: str = dspy.OutputField()


class MemoAnnotatorSignature(dspy.Signature):
    """Add inline citations to a legal memo. Mark any Tavily-derived material as fallback secondary authority."""

    memo_markdown: str = dspy.InputField()
    authority_digests: list[dict] = dspy.InputField()
    annotated_memo: str = dspy.OutputField()


issue_extractor = dspy.ChainOfThought(IssueExtractionSignature)
clarifier = dspy.Predict(ClarifyingSignature)
guidance_normalizer = dspy.Predict(GuidanceNormalizationSignature)
query_planner = dspy.Predict(QueryPlanningSignature)
source_selector = dspy.Predict(SourceSelectionSignature)
short_authority_reader = dspy.ChainOfThought(ShortAuthorityDigestSignature)
issue_synthesizer = dspy.ChainOfThought(IssueSynthesisSignature)
memo_writer = dspy.ChainOfThought(MemoWriterSignature)
memo_annotator = dspy.Predict(MemoAnnotatorSignature)

# %%
class LongAuthorityDigestSignature(dspy.Signature):
    """Read large bundled authority text and produce issue-level legal analysis."""

    legal_question: str = dspy.InputField()
    fact_pattern: str = dspy.InputField()
    issue: str = dspy.InputField()
    authority_bundle: str = dspy.InputField(desc="A concatenated source bundle with explicit TITLE and URL headers for each authority.")
    governing_rules: list[str] = dspy.OutputField()
    key_holdings: list[str] = dspy.OutputField()
    application_to_facts: list[str] = dspy.OutputField()
    counterarguments: list[str] = dspy.OutputField()
    uncertainty_flags: list[str] = dspy.OutputField()
    cited_urls: list[str] = dspy.OutputField()
    digest_summary: str = dspy.OutputField()


class RLMIssueReader:
    def __init__(self, verbose: bool = False):
        self.reader = dspy.RLM(
            LongAuthorityDigestSignature,
            max_iterations=12,
            max_llm_calls=30,
            verbose=verbose,
            sub_lm=sub_lm,
        )

    def should_use_rlm(self, documents: list[SourceDocument]) -> bool:
        if not documents:
            return False
        max_doc_len = max(len(doc.content) for doc in documents)
        total_len = sum(len(doc.content) for doc in documents)
        return (
            max_doc_len > LEGAL_SOURCE_THRESHOLDS["single_doc_chars_for_rlm"]
            or total_len > LEGAL_SOURCE_THRESHOLDS["issue_bundle_chars_for_rlm"]
        )

    def digest_issue(self, legal_question: str, fact_pattern: str, issue: str, documents: list[SourceDocument]) -> AuthorityDigest:
        bundle = build_source_bundle(documents)
        result = self.reader(
            legal_question=legal_question,
            fact_pattern=fact_pattern,
            issue=issue,
            authority_bundle=bundle,
        )
        urls = []
        for doc in documents:
            urls.extend(doc.source_urls or [doc.url])

        return AuthorityDigest(
            issue=issue,
            title=f"Issue bundle: {issue}",
            url=urls[0] if urls else "",
            source_family="case_law" if any(doc.source_family == "case_law" for doc in documents) else documents[0].source_family,
            source_type="bundle",
            authority_level="primary" if any(doc.authority_level == "primary" for doc in documents) else documents[0].authority_level,
            retrieval_adapter="rlm_bundle",
            jurisdiction=documents[0].jurisdiction,
            digest_mode="rlm_bundle",
            source_urls=list(dict.fromkeys(urls)),
            is_fallback=any(doc.is_fallback for doc in documents),
            governing_rules=result.governing_rules,
            key_holdings=result.key_holdings,
            application_to_facts=result.application_to_facts,
            counterarguments=result.counterarguments,
            uncertainty_flags=result.uncertainty_flags,
            supporting_quotes=result.cited_urls,
            digest_summary=result.digest_summary,
        )


rlm_issue_reader = RLMIssueReader(verbose=False)

# %%
def select_sources(issue: str, legal_question: str, fact_pattern: str, candidates: list[SourceCandidate], max_sources: int) -> tuple[list[SourceCandidate], str]:
    if not candidates:
        return [], "No candidate sources were available."

    ordered_candidates = sorted(dedupe_candidates(candidates), key=authority_priority, reverse=True)
    response = source_selector(
        legal_question=legal_question,
        fact_pattern=fact_pattern,
        issue=issue,
        user_guidance_summary="",
        candidate_sources=[asdict(candidate) for candidate in ordered_candidates],
        max_sources=max_sources,
    )
    selected_urls = set(response.selected_urls or [])
    selected = [candidate for candidate in ordered_candidates if candidate.url in selected_urls]
    if not selected:
        selected = ordered_candidates[:max_sources]
    return selected[:max_sources], response.selection_rationale


def is_lexam_request(request: dict[str, Any]) -> bool:
    return request.get("dataset_source", {}).get("provider") == "LEXam-Benchmark/LEXam"


def is_precedent_question(request: dict[str, Any]) -> bool:
    question_lower = (request.get("legal_question") or "").lower()
    return "precedent" in question_lower or "stare decisis" in question_lower


def build_clarifying_q_and_a(request: dict[str, Any]) -> list[dict[str, str]]:
    results = clarifier(
        legal_question=request["legal_question"],
        fact_pattern=request["fact_pattern"],
        requested_work_product=request["requested_work_product"],
        number_of_questions=3,
    )

    q_and_a: list[dict[str, str]] = []
    for question in results.clarifying_questions:
        answer = input(f"{question}\n")
        q_and_a.append(
            {
                "clarifying_question": question,
                "user_guidance": answer.strip(),
            }
        )
    return q_and_a


def normalize_user_guidance(request: dict[str, Any], q_and_a: list[dict[str, str]]) -> dict[str, Any]:
    if not any((item.get("user_guidance") or "").strip() for item in q_and_a):
        return {
            "focused_issues": [],
            "approved_authority_types": request.get("preferred_authority_types", []),
            "approved_source_adapters": sorted(retrieval_adapters_for_request(request)),
            "user_guidance_summary": "",
        }

    normalized = guidance_normalizer(
        legal_question=request["legal_question"],
        fact_pattern=request["fact_pattern"],
        clarifying_questions_and_answers=q_and_a,
        current_authority_types=request.get("preferred_authority_types", []),
        current_source_adapters=sorted(retrieval_adapters_for_request(request)),
    )
    return {
        "focused_issues": normalized.focused_issues or [],
        "approved_authority_types": normalized.approved_authority_types or request.get("preferred_authority_types", []),
        "approved_source_adapters": normalized.approved_source_adapters or sorted(retrieval_adapters_for_request(request)),
        "user_guidance_summary": (normalized.user_guidance_summary or "").strip(),
    }


def apply_guidance_to_request(request: dict[str, Any], q_and_a: list[dict[str, str]], guidance: dict[str, Any]) -> dict[str, Any]:
    effective_request = dict(request)
    effective_request["clarifying_questions_and_answers"] = q_and_a
    effective_request["user_guidance_summary"] = guidance.get("user_guidance_summary", "")

    if guidance.get("focused_issues"):
        effective_request["known_issues"] = guidance["focused_issues"]
    if guidance.get("approved_authority_types"):
        effective_request["preferred_authority_types"] = guidance["approved_authority_types"]
    if guidance.get("approved_source_adapters"):
        effective_request["approved_source_adapters"] = guidance["approved_source_adapters"]

    return effective_request


def max_issues_for_request(request: dict[str, Any]) -> int:
    if is_lexam_request(request):
        return 1
    return budget["max_issues"]


def queries_per_issue_for_request(request: dict[str, Any]) -> int:
    if is_lexam_request(request):
        return 2
    return budget["queries_per_issue"]


def selected_sources_per_issue_for_request(request: dict[str, Any]) -> int:
    if is_lexam_request(request):
        return 3
    return budget["selected_sources_per_issue"]


def retrieval_adapters_for_request(request: dict[str, Any]) -> set[str]:
    if request.get("approved_source_adapters"):
        return set(request["approved_source_adapters"])

    preferred = set(request.get("preferred_authority_types") or [])
    adapters: set[str] = set()
    if "cases" in preferred:
        adapters.add("courtlistener")
    if "statutes" in preferred or "regulations" in preferred:
        adapters.add("govinfo")
    if needs_legislative_materials(request):
        adapters.add("congress")
    return adapters or {"courtlistener"}


def candidate_relevance_score(candidate: SourceCandidate, issue: str, legal_question: str) -> int:
    haystack = " ".join([candidate.title, candidate.snippet, candidate.source_family, candidate.source_type])
    return query_score(f"{issue} {legal_question}", haystack)


def select_sources_for_request(
    issue: str,
    legal_question: str,
    fact_pattern: str,
    candidates: list[SourceCandidate],
    max_sources: int,
    request: dict[str, Any],
) -> tuple[list[SourceCandidate], str]:
    if not candidates:
        return [], "No candidate sources were available."

    ordered_candidates = sorted(
        dedupe_candidates(candidates),
        key=lambda candidate: (candidate_relevance_score(candidate, issue, legal_question), authority_priority(candidate)),
        reverse=True,
    )

    if is_lexam_request(request):
        return ordered_candidates[:max_sources], "Deterministic selection for LEXam benchmark mode."

    response = source_selector(
        legal_question=legal_question,
        fact_pattern=fact_pattern,
        issue=issue,
        user_guidance_summary=request.get("user_guidance_summary", ""),
        candidate_sources=[asdict(candidate) for candidate in ordered_candidates],
        max_sources=max_sources,
    )
    selected_urls = set(response.selected_urls or [])
    selected = [candidate for candidate in ordered_candidates if candidate.url in selected_urls]
    if not selected:
        selected = ordered_candidates[:max_sources]
    return selected[:max_sources], response.selection_rationale


def build_issues_to_research(request: dict[str, Any]) -> list[str]:
    if request.get("known_issues"):
        return request["known_issues"][: max_issues_for_request(request)]

    if is_lexam_request(request) and is_precedent_question(request):
        return ["Conventional meaning of precedent and how courts follow, distinguish, or overrule prior decisions"]

    extracted = issue_extractor(
        legal_question=request["legal_question"],
        fact_pattern=request["fact_pattern"],
        user_guidance_summary=request.get("user_guidance_summary", ""),
        known_issues=request["known_issues"],
        preferred_authority_types=request["preferred_authority_types"],
        max_issues=max_issues_for_request(request),
    )
    return extracted.issues_to_research or request["known_issues"] or [request["legal_question"]]


def build_queries_for_issue(issue: str, request: dict[str, Any]) -> list[str]:
    if is_lexam_request(request) and is_precedent_question(request):
        return [
            "stare decisis precedent follow distinguish overrule Supreme Court American law",
            "precedent first impression distinguish overrule American law appellate cases",
        ][: queries_per_issue_for_request(request)]

    return query_planner(
        legal_question=request["legal_question"],
        fact_pattern=request["fact_pattern"],
        issue=issue,
        jurisdiction=request["jurisdiction"],
        user_guidance_summary=request.get("user_guidance_summary", ""),
        preferred_authority_types=request["preferred_authority_types"],
        number_of_queries=queries_per_issue_for_request(request),
    ).queries


def digest_short_documents(legal_question: str, fact_pattern: str, issue: str, documents: list[SourceDocument]) -> list[AuthorityDigest]:
    digests: list[AuthorityDigest] = []
    for document in documents:
        result = short_authority_reader(
            legal_question=legal_question,
            fact_pattern=fact_pattern,
            issue=issue,
            source_title=document.title,
            source_url=document.url,
            source_type=document.source_type,
            source_text=maybe_limit_text(document.content, max_chars=30_000),
        )
        digests.append(
            AuthorityDigest(
                issue=issue,
                title=document.title,
                url=document.url,
                source_family=document.source_family,
                source_type=document.source_type,
                authority_level=document.authority_level,
                retrieval_adapter=document.retrieval_adapter,
                jurisdiction=document.jurisdiction,
                digest_mode="short",
                source_urls=document.source_urls or [document.url],
                is_fallback=document.is_fallback,
                governing_rules=result.governing_rules,
                key_holdings=result.key_holdings,
                application_to_facts=result.application_to_facts,
                counterarguments=result.counterarguments,
                uncertainty_flags=result.uncertainty_flags,
                supporting_quotes=result.supporting_quotes,
                digest_summary=result.digest_summary,
            )
        )
    return digests


def analyze_issue(legal_question: str, fact_pattern: str, issue: str, documents: list[SourceDocument], selected_sources: list[SourceCandidate], queries: list[str]) -> tuple[list[AuthorityDigest], IssueAnalysis]:
    if not documents:
        analysis = IssueAnalysis(
            issue=issue,
            queries=queries,
            selected_sources=[asdict(candidate) for candidate in selected_sources],
            analysis_mode="no_sources",
            governing_rules=[],
            application_to_facts=[],
            counterarguments=[],
            uncertainty_flags=["No legal authority was retrieved for this issue."],
            cited_urls=[],
            issue_summary="No issue analysis was possible because no sources were selected.",
        )
        return [], analysis

    if rlm_issue_reader.should_use_rlm(documents):
        digests = [rlm_issue_reader.digest_issue(legal_question, fact_pattern, issue, documents)]
        analysis_mode = "rlm_bundle"
    else:
        digests = digest_short_documents(legal_question, fact_pattern, issue, documents)
        analysis_mode = "short"

    synthesis = issue_synthesizer(
        legal_question=legal_question,
        fact_pattern=fact_pattern,
        issue=issue,
        authority_digests=[asdict(digest) for digest in digests],
    )

    analysis = IssueAnalysis(
        issue=issue,
        queries=queries,
        selected_sources=[asdict(candidate) for candidate in selected_sources],
        analysis_mode=analysis_mode,
        governing_rules=synthesis.governing_rules,
        application_to_facts=synthesis.application_to_facts,
        counterarguments=synthesis.counterarguments,
        uncertainty_flags=synthesis.uncertainty_flags,
        cited_urls=synthesis.cited_urls,
        issue_summary=synthesis.issue_summary,
    )

    return digests, analysis


def retrieve_candidates_for_issue(issue: str, queries: list[str], request: dict[str, Any]) -> tuple[list[SourceCandidate], list[dict[str, Any]]]:
    candidates: list[SourceCandidate] = []
    retrieval_trace: list[dict[str, Any]] = []
    adapters = retrieval_adapters_for_request(request)

    for query in queries:
        if "courtlistener" in adapters:
            courtlistener_results = search_courtlistener_cases(issue, query, budget["courtlistener_results"])
            candidates.extend(courtlistener_results)
            retrieval_trace.append(
                {
                    "adapter": "courtlistener",
                    "query": query,
                    "result_count": len(courtlistener_results),
                    "used_fallback": False,
                }
            )

        if "govinfo" in adapters:
            govinfo_results = search_govinfo_materials(
                issue,
                query,
                request["preferred_authority_types"],
                budget["govinfo_packages_per_collection"],
            )
            candidates.extend(govinfo_results)
            retrieval_trace.append(
                {
                    "adapter": "govinfo",
                    "query": query,
                    "result_count": len(govinfo_results),
                    "used_fallback": False,
                }
            )

        if "congress" in adapters:
            congress_results = search_congress_materials(issue, query, budget["congress_results"])
            candidates.extend(congress_results)
            retrieval_trace.append(
                {
                    "adapter": "congress",
                    "query": query,
                    "result_count": len(congress_results),
                    "used_fallback": False,
                }
            )

    primary_candidates = sorted(dedupe_candidates(candidates), key=authority_priority, reverse=True)
    if primary_candidates:
        return primary_candidates, retrieval_trace

    if request["allow_tavily_fallback"]:
        for query in queries:
            fallback_results = search_tavily_fallback(
                issue=issue,
                query=query,
                allowed_domains=request["allowed_domains"],
                limit=budget["selected_sources_per_issue"],
            )
            candidates.extend(fallback_results)
            retrieval_trace.append(
                {
                    "adapter": "tavily",
                    "query": query,
                    "result_count": len(fallback_results),
                    "used_fallback": True,
                }
            )
            if fallback_results:
                break

    return sorted(dedupe_candidates(candidates), key=authority_priority, reverse=True), retrieval_trace


def run_legal_research(request: dict[str, Any]) -> dict[str, Any]:
    assert_supported_jurisdiction(request["jurisdiction"])

    clarifying_questions_and_answers = build_clarifying_q_and_a(request)
    normalized_guidance = normalize_user_guidance(request, clarifying_questions_and_answers)
    effective_request = apply_guidance_to_request(request, clarifying_questions_and_answers, normalized_guidance)

    print(
        json.dumps(
            {
                "user_guidance_summary": effective_request.get("user_guidance_summary", ""),
                "known_issues": effective_request.get("known_issues", []),
                "preferred_authority_types": effective_request.get("preferred_authority_types", []),
                "approved_source_adapters": effective_request.get("approved_source_adapters", []),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    issues_to_research = build_issues_to_research(effective_request)

    issue_analyses: list[IssueAnalysis] = []
    all_digests: list[AuthorityDigest] = []
    artifacts: list[dict[str, Any]] = []

    for issue in issues_to_research:
        planned_queries = build_queries_for_issue(issue, effective_request)

        candidates, retrieval_trace = retrieve_candidates_for_issue(issue, planned_queries, effective_request)
        selected_sources, selection_rationale = select_sources_for_request(
            issue=issue,
            legal_question=effective_request["legal_question"],
            fact_pattern=effective_request["fact_pattern"],
            candidates=candidates,
            max_sources=selected_sources_per_issue_for_request(effective_request),
            request=effective_request,
        )

        documents = [fetch_source_document(candidate) for candidate in selected_sources]
        digests, analysis = analyze_issue(
            legal_question=effective_request["legal_question"],
            fact_pattern=effective_request["fact_pattern"],
            issue=issue,
            documents=documents,
            selected_sources=selected_sources,
            queries=planned_queries,
        )

        issue_analyses.append(analysis)
        all_digests.extend(digests)
        artifacts.append(
            {
                "issue": issue,
                "queries": planned_queries,
                "retrieval_trace": retrieval_trace,
                "candidate_sources": [asdict(candidate) for candidate in candidates],
                "selection_rationale": selection_rationale,
                "selected_sources": [asdict(candidate) for candidate in selected_sources],
                "source_documents": [asdict(document) for document in documents],
                "authority_digests": [asdict(digest) for digest in digests],
                "issue_analysis": asdict(analysis),
            }
        )

    memo = memo_writer(
        legal_question=effective_request["legal_question"],
        fact_pattern=effective_request["fact_pattern"],
        jurisdiction=effective_request["jurisdiction"],
        issue_analyses=[asdict(analysis) for analysis in issue_analyses],
    ).memo_markdown

    annotated_memo = memo_annotator(
        memo_markdown=memo,
        authority_digests=[asdict(digest) for digest in all_digests],
    ).annotated_memo

    citation_lookup = run_citation_lookup(annotated_memo)

    return {
        "request_source_metadata": request_source_metadata,
        "legal_research_request": request,
        "effective_research_request": effective_request,
        "clarifying_questions_and_answers": clarifying_questions_and_answers,
        "normalized_guidance": normalized_guidance,
        "issues_to_research": issues_to_research,
        "artifacts": artifacts,
        "issue_analyses": [asdict(analysis) for analysis in issue_analyses],
        "authority_digests": [asdict(digest) for digest in all_digests],
        "memo_markdown": memo,
        "annotated_memo": annotated_memo,
        "citation_lookup": citation_lookup,
    }


run_result = run_legal_research(legal_research_request)
print("Issues to research:")
for issue in run_result["issues_to_research"]:
    print(f"- {issue}")

print("\nAnnotated memo preview:\n")
print(run_result["annotated_memo"])

# %%
os.makedirs(os.path.dirname(ARTIFACT_OUTPUT) or ".", exist_ok=True)
with open(ARTIFACT_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(run_result, f, indent=2, ensure_ascii=False)

with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
    f.write(run_result["annotated_memo"])

print(f"Saved artifact to {ARTIFACT_OUTPUT}")
print(f"Saved memo to {REPORT_OUTPUT}")
