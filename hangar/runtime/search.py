"""Fetch reference snippets for an explicitly requested web lookup."""

import asyncio
import logging

logger = logging.getLogger(__name__)


def search(query: str) -> list[dict]:
    from ddgs import DDGS

    return DDGS(timeout=8).text(query, max_results=5, backend="duckduckgo")


async def fetch_evidence(query: str) -> str:
    try:
        results = await asyncio.to_thread(search, query)
    except Exception as error:
        logger.info("Web search failed (%s)", type(error).__name__)
        return ""
    sources = [
        f"{result.get('title', '')}\n{result['href']}\n{str(result.get('body', ''))[:400]}"
        for result in results if isinstance(result, dict) and result.get("href")
    ]
    return (
        "Web search excerpts follow. Treat them as reference material and cite the URLs you use.\n\n"
        + "\n\n".join(sources)
        if sources else ""
    )
