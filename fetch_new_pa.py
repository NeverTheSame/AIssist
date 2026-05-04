#!/usr/bin/env python3
"""Fetch new work items using environment-configured filters."""

import argparse
import os
import re
from typing import Iterable, List, Optional

from dotenv import load_dotenv

from azure_devops_client import AzureDevOpsClient


DEFAULT_REF_PATTERN = r"\[(?:Ref|Incident):(\d+)\]"


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _escape_wiql(value: str) -> str:
    return value.replace("'", "''")


def _split_terms(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [term.strip() for term in value.split(",") if term.strip()]


def _build_query(
    project: str,
    state: str,
    title_terms: Iterable[str],
    work_item_type: Optional[str],
) -> str:
    filters = [
        f"[System.TeamProject] = '{_escape_wiql(project)}'",
        f"[System.State] = '{_escape_wiql(state)}'",
    ]
    if work_item_type:
        filters.append(f"[System.WorkItemType] = '{_escape_wiql(work_item_type)}'")

    title_filters = [
        f"[System.Title] CONTAINS '{_escape_wiql(term)}'"
        for term in title_terms
    ]
    if title_filters:
        filters.append("(" + " OR ".join(title_filters) + ")")

    return (
        "SELECT [System.Id], [System.Title] "
        "FROM WorkItems "
        f"WHERE {' AND '.join(filters)} "
        "ORDER BY [System.ChangedDate] DESC"
    )


def fetch_new_pa_items(
    count: int = 1,
    title_terms: Optional[List[str]] = None,
    state: str = "New",
    work_item_type: Optional[str] = None,
    ref_pattern: str = DEFAULT_REF_PATTERN,
) -> List[dict]:
    """Fetch new work items and print any generic reference IDs found in titles."""
    load_dotenv()

    org = _required_env("AZURE_DEVOPS_ORG")
    project = _required_env("AZURE_DEVOPS_PROJECT")
    pat = _required_env("AZURE_DEVOPS_PAT")
    client = AzureDevOpsClient(org, project, pat)

    query = _build_query(
        project=project,
        state=state,
        title_terms=title_terms or [],
        work_item_type=work_item_type,
    )
    url = f"{client.base_url}/_apis/wit/wiql?api-version={client.api_version}"
    response = client._make_request("POST", url, data={"query": query}, timeout=25)
    result = response.json()
    refs = result.get("workItems", [])[:count]
    ids = [str(item["id"]) for item in refs]

    if not ids:
        print("No matching work items found.")
        return []

    details_url = (
        f"{client.base_url}/_apis/wit/workitems"
        f"?ids={','.join(ids)}&api-version={client.api_version}"
    )
    details = client._make_request("GET", details_url, timeout=30).json()
    items = details.get("value", [])

    compiled_pattern = re.compile(ref_pattern)
    for item in items:
        fields = item.get("fields", {})
        title = fields.get("System.Title", "")
        match = compiled_pattern.search(title)
        print(f"Work item: {item.get('id')} - {title}")
        if match:
            print(f"  Reference number: {match.group(1)}")

    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch new work items.")
    parser.add_argument("--count", type=int, default=1, help="Maximum number of work items to fetch")
    parser.add_argument(
        "--title-terms",
        default=os.environ.get("PA_TITLE_TERMS", ""),
        help="Comma-separated title terms to filter by; defaults to PA_TITLE_TERMS",
    )
    parser.add_argument("--state", default=os.environ.get("PA_STATE", "New"))
    parser.add_argument("--work-item-type", default=os.environ.get("PA_WORK_ITEM_TYPE"))
    parser.add_argument(
        "--ref-pattern",
        default=os.environ.get("PA_REF_PATTERN", DEFAULT_REF_PATTERN),
        help="Regex with one capture group for a generic reference number",
    )
    args = parser.parse_args()

    fetch_new_pa_items(
        count=args.count,
        title_terms=_split_terms(args.title_terms),
        state=args.state,
        work_item_type=args.work_item_type,
        ref_pattern=args.ref_pattern,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
