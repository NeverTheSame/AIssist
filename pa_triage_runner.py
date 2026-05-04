#!/usr/bin/env python3
"""Run a browser-backed triage pass for new work items."""

import argparse
import asyncio
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from playwright.async_api import Page, async_playwright


DEFAULT_REF_PATTERN = r"\[(?:Ref|Incident):(\d+)\]"


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value


def extract_reference_from_title(title: str, pattern: str = DEFAULT_REF_PATTERN) -> Optional[str]:
    """Extract a reference number from a work item title."""
    match = re.search(pattern, title)
    return match.group(1) if match else None


async def _collect_candidate_ids(page: Page) -> List[str]:
    snapshot = await page.accessibility.snapshot()
    ids_found: List[str] = []

    def collect_ids(node: Optional[Dict[str, Any]]) -> None:
        if not node:
            return
        if node.get("role") == "link":
            text = node.get("name", "") or node.get("text", "")
            if text.isdigit() and 4 <= len(text) <= 8:
                ids_found.append(text)
        for child in node.get("children", []):
            collect_ids(child)

    collect_ids(snapshot)
    return ids_found


async def fetch_work_items_with_playwright(query_url: str, count: int = 1) -> List[Dict[str, Any]]:
    """Fetch new work item IDs from a configured query page."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        print("Navigating to work item query...")
        await page.goto(query_url)
        await page.wait_for_timeout(5000)

        ids_found = await _collect_candidate_ids(page)
        print(f"Found potential work item IDs: {ids_found[:10]}")

        await browser.close()
        return [
            {"id": work_item_id, "title": None, "state": "New"}
            for work_item_id in ids_found[:count]
        ]


async def fetch_reference_details(reference_number: str, detail_url_template: str) -> Dict[str, Any]:
    """Fetch details for a generic reference using a configured URL template."""
    url = detail_url_template.format(reference_number=reference_number)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        print(f"Navigating to reference {reference_number}...")
        await page.goto(url)
        await page.wait_for_timeout(4000)

        snapshot = await page.accessibility.snapshot()
        await browser.close()

    def extract_text(node: Optional[Dict[str, Any]], depth: int = 0) -> str:
        if not node or depth > 10:
            return ""

        texts = []
        node_name = node.get("name", "") or node.get("text", "")
        if node_name:
            texts.append(node_name)

        for child in node.get("children", []):
            child_text = extract_text(child, depth + 1)
            if child_text:
                texts.append(child_text)

        return "\n".join(texts)

    page_text = extract_text(snapshot)
    return {
        "reference_number": reference_number,
        "text": page_text,
        "text_length": len(page_text),
    }


async def run(
    count: int,
    query_url: str,
    ref_pattern: str,
    detail_url_template: Optional[str],
) -> None:
    print("=== Work Item Triage Runner ===")
    print(f"Processing {count} work item(s)\n")

    print("Step 1: Fetching new work items...")
    work_items = await fetch_work_items_with_playwright(query_url, count)

    if not work_items:
        print("No work items found.")
        return

    print(f"Found {len(work_items)} work item(s): {[wi['id'] for wi in work_items]}")

    for work_item in work_items:
        print(f"\n--- Processing work item #{work_item['id']} ---")
        title = work_item.get("title") or ""
        reference_number = extract_reference_from_title(title, ref_pattern)

        if not reference_number:
            print("No reference number found in title.")
            continue

        print(f"Reference number: {reference_number}")
        if detail_url_template:
            details = await fetch_reference_details(reference_number, detail_url_template)
            print(f"Fetched reference details ({details['text_length']} chars)")

    print("\n=== Triage complete ===")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run browser-backed triage for new work items.")
    parser.add_argument("--count", type=int, default=1, help="Maximum number of work items to process")
    parser.add_argument(
        "--query-url",
        default=os.environ.get("PA_QUERY_URL"),
        help="Work item query URL; defaults to PA_QUERY_URL",
    )
    parser.add_argument(
        "--ref-pattern",
        default=os.environ.get("PA_REF_PATTERN", DEFAULT_REF_PATTERN),
        help="Regex with one capture group for a generic reference number",
    )
    parser.add_argument(
        "--detail-url-template",
        default=os.environ.get("PA_DETAIL_URL_TEMPLATE"),
        help="Optional URL template with {reference_number}",
    )
    args = parser.parse_args()

    query_url = args.query_url or _required_env("PA_QUERY_URL")
    asyncio.run(
        run(
            count=args.count,
            query_url=query_url,
            ref_pattern=args.ref_pattern,
            detail_url_template=args.detail_url_template,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
