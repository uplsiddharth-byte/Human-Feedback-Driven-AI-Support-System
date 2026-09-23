"""Deep-crawl the public MU website to a raw JSON corpus.

Requires crawl4ai (headless Chrome). Output is RAW capture, kept as provenance;
run scripts/clean_crawl.py on it before generating any QA pairs.
"""
import asyncio
import json
import re
from datetime import date
from urllib.parse import urlparse, urldefrag

from crawl4ai import AsyncWebCrawler, BrowserConfig

START_URL = "https://www.mahindrauniversity.edu.in/"
MAX_PAGES = 50
DELAY_SECONDS = 1.0  # be a polite guest on the university's server

# Only this host. The 'internal' links crawl4ai returns include other subdomains,
# and muerp.* is the Juno-powered student ERP: off limits under CLAUDE.md §12
# rules 1 and 6, and its login URL carries a live jsessionid.
ALLOWED_HOST = "www.mahindrauniversity.edu.in"
SESSION_TOKEN = re.compile(r"jsessionid|sessionid=|[?&]token=", re.I)


def crawlable(url):
    if not url.startswith("http"):
        return False           # skips mailto:, callto:, tel:
    if SESSION_TOKEN.search(url):
        return False
    return urlparse(url).netloc == ALLOWED_HOST


async def deep_crawl():
    captured_at = date.today().isoformat()
    visited, queue, results = set(), [START_URL], []
    print(f"Crawling {ALLOWED_HOST}. Target: {MAX_PAGES} pages...")

    async with AsyncWebCrawler(config=BrowserConfig(chrome_channel="chrome")) as crawler:
        while queue and len(results) < MAX_PAGES:
            current_url = urldefrag(queue.pop(0)).url
            if current_url in visited or not crawlable(current_url):
                continue
            visited.add(current_url)

            print(f"[{len(results)+1}/{MAX_PAGES}] {current_url}")
            try:
                result = await crawler.arun(url=current_url)
                if result.success:
                    results.append({
                        "url": current_url,
                        "title": result.metadata.get("title", "No Title") if result.metadata else "No Title",
                        "markdown": result.markdown,
                        "captured_at": captured_at,   # §6: provenance needs a capture date
                    })
                    for link in result.links.get("internal", []):
                        href = urldefrag(link.get("href", "")).url
                        if crawlable(href) and href not in visited:
                            queue.append(href)
            except Exception as e:
                print(f"Skipping {current_url}: {e}")

            await asyncio.sleep(DELAY_SECONDS)

    with open("mahindra_complete.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\nSaved {len(results)} pages (captured_at={captured_at}) to mahindra_complete.json")
    print("Next: python3 scripts/clean_crawl.py mahindra_complete.json")


if __name__ == "__main__":
    asyncio.run(deep_crawl())
