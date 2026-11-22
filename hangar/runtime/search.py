"""Fetch web snippets for a user-selected lookup."""

import asyncio


def _search(query: str) -> list[dict]:
    from ddgs import DDGS

    return DDGS().text(query, max_results=5, backend="duckduckgo")


async def fetch_evidence(query: str) -> str:
    try:
        results = await asyncio.to_thread(_search, query)
    except Exception:
        results = []
    sources = [
        f"{result.get('title', '')}\n{result['href']}\n{result.get('body', '')[:400]}"
        for result in results
        if result.get("href")
    ]
    if not sources:
        return "Web search returned no sources. Say when the answer could not be verified."
    return (
        "Web search excerpts follow. Treat them as reference material and cite the URLs you use.\n\n"
        + "\n\n".join(sources)
    )
