"""Retrieval orchestrator for chat mode.

The runtime makes all search decisions. The chat model never sees search tools —
it receives pre-fetched evidence injected into its context as [WEB EVIDENCE].

Pipeline:
  User query
    → SearchPolicy.evaluate()     (classify intent, decide whether/how to search)
    → fetch_evidence()             (search + optional page fetch)
    → inject into system prompt
    → LLM answers                  (no tools exposed)

The model reasons. The runtime retrieves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


# ── Decision ──────────────────────────────────────────────────────────────────

@dataclass
class SearchDecision:
    search: bool
    confidence: float
    reason: str
    max_pages: int = 1          # how many top results to fully fetch
    query_type: Literal[
        "product", "person", "current_events", "comparison",
        "factual", "creative", "coding"
    ] = "factual"
    answer_style: str = ""      # prepended to system prompt, empty = no override


# ── Answer style hints ────────────────────────────────────────────────────────

_ANSWER_STYLES: dict[str, str] = {
    "product": (
        "Answer as a product review synthesis.\n"
        "Structure: specs summary → pros → cons → verdict.\n"
        "End with a clear one-sentence recommendation."
    ),
    "comparison": (
        "Answer as a side-by-side comparison.\n"
        "Use a markdown table for the key specs.\n"
        "End with a recommendation that names a winner and explains why."
    ),
    "person": (
        "Answer as a biographical summary.\n"
        "Cover: role, background, key achievements, current status."
    ),
    "current_events": (
        "Answer as a news summary.\n"
        "Cover: what happened, when, key parties involved, why it matters."
    ),
    "creative": (
        "Be creative, engaging, and match the user's tone.\n"
        "Prioritize quality and voice over length."
    ),
    "coding": (
        "Be precise and practical.\n"
        "Include working code examples. Prefer brevity over exhaustive explanation."
    ),
    "factual": (
        "Answer clearly and directly. Lead with the answer, then supporting detail."
    ),
}


# ── Signal sets ───────────────────────────────────────────────────────────────

_CREATIVE = {
    "write me", "schrijf", "maak een gedicht", "maak een verhaal",
    "give me ideas", "geef me ideeën", "brainstorm", "inspiratie voor",
    "help me write", "generate a poem", "create a story",
    "wat zijn leuke", "wat zijn goede ideeën",
}

_CODING = {
    "how do i code", "how do i implement", "syntax for",
    "debug this", "fix this code", "what does this code",
    "hoe gebruik ik", "hoe implementeer ik", "error in my code",
    "traceback", "import error", "how to use the library",
    "how to build", "how to create a function",
}

_KNOWLEDGE = {
    "explain how", "explain what", "uitleg over", "wat betekent",
    "hoe werkt", "what is the concept", "theoretically",
    "generally speaking", "in general terms",
    "what is the difference between", "wat is het verschil",
    "calculate", "bereken", "solve this",
    "what do you think", "jouw mening",
}

_PRODUCT_BRANDS = {
    "msi", "asus", "dell", "hp", "lenovo", "acer", "huawei", "razer",
    "alienware", "apple", "microsoft surface", "samsung", "lg", "sony",
    "nvidia", "amd", "intel", "qualcomm", "corsair", "logitech",
    "canon", "nikon", "fujifilm", "dji",
    "iphone", "macbook", "ipad", "galaxy", "pixel",
}

_PRODUCT_TERMS = {
    "laptop", "notebook", "tablet", "smartphone", "desktop", "pc",
    "gpu", "cpu", "processor", "graphics card", "videocard",
    "ssd", "nvme", "ram", "monitor", "keyboard", "headset",
    "specs", "specifications", "benchmark", "review", "teardown",
}

_NEWS_TERMS = {
    "news", "nieuws", "today", "tonight", "this week", "this month",
    "latest", "recent", "breaking", "what happened", "update on",
    "right now", "live score", "standings", "weather", "forecast",
}

_PERSON_TERMS = {
    "who is", "wie is", "tell me about", "vertel me over",
    "biography", "born in", "age of", "how old is", "net worth",
    "ceo of", "founder of", "who created", "who invented",
}

_COMPARISON_TERMS = {
    " vs ", " versus ", "compared to", "better than",
    "which is better", "should i buy", "or the", "which one",
    "difference between", "vergelijk",
}

# Matches model/product codes: "B7UC", "RTX 3050", "i7-13700H", "RX 6700 XT"
_MODEL_RE = re.compile(
    r'\b([A-Z]{1,3}\d[\w\-]{1,8}|[A-Z]{2,4}\s\d{3,4}[A-Z\s]*\d*)\b'
)


# ── Policy ────────────────────────────────────────────────────────────────────

class SearchPolicy:
    """Classify a query and return a SearchDecision the pipeline will execute."""

    def evaluate(self, query: str) -> SearchDecision:
        q = query.lower()

        def _decide(search: bool, confidence: float, reason: str,
                    max_pages: int = 1, query_type: str = "factual") -> SearchDecision:
            return SearchDecision(
                search=search,
                confidence=confidence,
                reason=reason,
                max_pages=max_pages,
                query_type=query_type,
                answer_style=_ANSWER_STYLES.get(query_type, ""),
            )

        # ── No retrieval needed ──────────────────────────────────────────────
        if any(sig in q for sig in _CREATIVE):
            return _decide(False, 0.90, "creative/brainstorm", query_type="creative")
        if any(sig in q for sig in _CODING):
            return _decide(False, 0.85, "coding/syntax", query_type="coding")
        if any(sig in q for sig in _KNOWLEDGE):
            return _decide(False, 0.80, "general knowledge", query_type="factual")

        # ── Product — search + fetch official/review page ────────────────────
        if (any(brand in q for brand in _PRODUCT_BRANDS)
                or any(term in q for term in _PRODUCT_TERMS)
                or _MODEL_RE.search(query)):
            return _decide(True, 0.90, "product lookup", max_pages=1, query_type="product")

        # ── Person — search + fetch bio/Wikipedia ────────────────────────────
        if any(sig in q for sig in _PERSON_TERMS):
            return _decide(True, 0.85, "person lookup", max_pages=1, query_type="person")

        # ── Current events — search + fetch top article ──────────────────────
        if any(term in q for term in _NEWS_TERMS):
            return _decide(True, 0.90, "current events", max_pages=1, query_type="current_events")

        # ── Comparison — two queries, snippets only ───────────────────────────
        if any(term in q for term in _COMPARISON_TERMS):
            return _decide(True, 0.75, "comparison", max_pages=0, query_type="comparison")

        # ── Factual question words ────────────────────────────────────────────
        _Q = {"who", "what", "where", "when", "why", "how", "which",
              "wie", "wat", "waar", "wanneer", "waarom", "hoe", "welk", "welke",
              "is", "are", "does", "can", "will", "heeft", "is er"}
        if set(q.split()) & _Q:
            return _decide(True, 0.60, "factual question", max_pages=0, query_type="factual")

        # ── Conversational ────────────────────────────────────────────────────
        return _decide(False, 0.70, "conversational", query_type="creative")


# ── Session evidence cache ────────────────────────────────────────────────────

@dataclass
class _CacheEntry:
    evidence: str
    keywords: set[str]        # from the search query, for topic-match


_session_cache: dict[str, _CacheEntry] = {}

_STOP_WORDS = {
    "the", "a", "an", "is", "it", "its", "in", "on", "of", "for",
    "and", "or", "to", "do", "how", "what", "which", "with", "from",
    "de", "het", "een", "is", "in", "op", "van", "voor", "en", "of",
    "hoe", "wat", "welk", "met", "door",
}


def _keywords(text: str) -> set[str]:
    return {w for w in text.lower().split() if w not in _STOP_WORDS and len(w) > 2}


def get_cached_evidence(session_id: str, query: str) -> str | None:
    """Return cached evidence if the new query is topically related to the last search."""
    entry = _session_cache.get(session_id)
    if not entry:
        return None
    curr = _keywords(query)
    overlap = len(entry.keywords & curr) / max(len(entry.keywords | curr), 1)
    return entry.evidence if overlap >= 0.25 else None


def cache_evidence(session_id: str, search_query: str, evidence: str) -> None:
    _session_cache[session_id] = _CacheEntry(
        evidence=evidence,
        keywords=_keywords(search_query),
    )


# ── Evidence result ───────────────────────────────────────────────────────────

@dataclass
class EvidenceResult:
    """Return type of fetch_evidence(). Carries the context block and a quality hint."""
    text: str           # formatted context block; empty string when no search needed
    quality_hint: str   # one-line quality signal prepended to system prompt; "" = none


# ── Quality classifier ────────────────────────────────────────────────────────

# Term pairs that suggest conflicting information across sources
_CONFLICT_PAIRS: list[tuple[set[str], set[str]]] = [
    ({"discontinued", "unavailable", "no longer", "end of life"},
     {"available", "in stock", "buy now", "released"}),
    ({"cancelled", "canceled", "scrapped"},
     {"confirmed", "announced", "launching"}),
    ({"not compatible", "not supported"},
     {"compatible", "supported", "works with"}),
]


def _classify_evidence(snippets: list[dict]) -> str:
    """Return a one-line quality hint, or empty string when evidence is good."""
    if not snippets:
        return "No results found. Answer from training knowledge and state clearly if uncertain."

    if len(snippets) < 2:
        return "Evidence quality: weak — only one source found. State uncertainty clearly."

    texts = [s.get("snippet", "").lower() for s in snippets]
    for neg_set, pos_set in _CONFLICT_PAIRS:
        has_neg = any(any(w in t for w in neg_set) for t in texts)
        has_pos = any(any(w in t for w in pos_set) for t in texts)
        if has_neg and has_pos:
            return "Evidence quality: conflicting — sources disagree. Present both positions clearly."

    return ""   # good evidence needs no special instruction


# ── Pipeline execution ────────────────────────────────────────────────────────

async def fetch_evidence(
    decision: SearchDecision,
    query: str,
    session_id: str = "",
) -> EvidenceResult:
    """Run the retrieval pipeline and return an EvidenceResult for context injection.

    Returns EvidenceResult(text="", quality_hint="") when no search is needed.
    The model never knows whether a search occurred; it only sees "Context:".
    """
    if not decision.search:
        return EvidenceResult("", "")

    # Cache hit — quality hint not stored (re-classify from text length as proxy)
    if session_id:
        cached = get_cached_evidence(session_id, query)
        if cached:
            return EvidenceResult(cached, "")

    from tools import execute_tool

    raw = await execute_tool("web_search", {"query": query, "max_results": 5})
    snippets = _parse_search_results(raw)

    page_text = ""
    page_url = ""
    if decision.max_pages > 0 and snippets:
        page_url = snippets[0]["url"]
        page_raw = await execute_tool("browse_page", {"url": page_url, "max_chars": 3000})
        if not page_raw.startswith("[error]"):
            page_text = page_raw

    quality_hint = _classify_evidence(snippets)
    text = _build_context_block(snippets, page_text, page_url)

    if session_id:
        cache_evidence(session_id, query, text)

    return EvidenceResult(text, quality_hint)


def _parse_search_results(raw: str) -> list[dict]:
    """Parse the text output of tool_web_search into a list of {title, url, snippet}."""
    results = []
    current: dict = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\d+\.", stripped):
            if current:
                results.append(current)
            current = {"title": re.sub(r"^\d+\.\s*", "", stripped), "url": "", "snippet": ""}
        elif stripped.startswith("URL:"):
            current["url"] = stripped[4:].strip()
        elif current:
            current["snippet"] = (current["snippet"] + " " + stripped).strip()
    if current:
        results.append(current)
    return results


def _domain(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return m.group(1) if m else url


def _build_context_block(
    snippets: list[dict],
    page_text: str,
    page_url: str,
) -> str:
    """Build the context block injected into the system prompt.

    Labelled "Context:" — the model doesn't know whether this came from the web,
    a cache, a document, or an API. Provenance belongs to the runtime.
    """
    lines = ["Context:"]
    for s in snippets:
        domain = _domain(s["url"])
        if s["title"]:
            lines.append(f"\n{s['title']} ({domain})")
        if s["snippet"]:
            lines.append(s["snippet"][:300])
    if page_text:
        lines += [f"\nDetailed source ({_domain(page_url)}):", page_text[:2500]]
    lines.append("\nAnswer naturally using the context above.")
    return "\n".join(lines)
