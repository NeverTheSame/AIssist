"""Tool-boundary output validation before the one Azure DevOps write path
that accepts LLM-generated content
(azure_devops_client.create_preventative_action_work_item,
patch built at azure_devops_client.py:617-638).

Must run *after* rehydration: the gateway (guard/gateway.py) has already
substituted real values back into `response.choices[*].message.content`
by the time analysis text reaches here, so validating pre-rehydration text
would miss a real value the model reintroduced via an invented placeholder.

`title` and `repair_item_type` come from human input() today
(main.py:1102,:1109) -- only `description` is LLM-generated. This module
is pre-positioned for the automation in pa_triage_runner.py/fetch_new_pa.py
that would remove that gate, not closing a live autonomous-write hole.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .injection import InjectionResult, detect as detect_injection

DEFAULT_REPAIR_ITEM_TYPES = [
    "Product Improvement",
    "Process Enablement",
    "Documentation",
    "Technical Enablement",
    "Diagnostic Tools",
]


@dataclass
class ValidationResult:
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    injection: Optional[InjectionResult] = None


def validate_work_item(
    repair_item_type: str,
    description: str,
    assigned_to: str = "",
    allowed_repair_item_types: Optional[List[str]] = None,
    allowed_assignees: Optional[List[str]] = None,
) -> ValidationResult:
    allowed_repair_item_types = allowed_repair_item_types or DEFAULT_REPAIR_ITEM_TYPES
    result = ValidationResult()

    if repair_item_type and repair_item_type not in allowed_repair_item_types:
        result.errors.append(
            f"repair_item_type {repair_item_type!r} is not in the allowlist {allowed_repair_item_types}"
        )

    if allowed_assignees and assigned_to and assigned_to not in allowed_assignees:
        result.errors.append(f"assigned_to {assigned_to!r} is not in the allowlist")

    injection = detect_injection(description or "")
    result.injection = injection
    if injection.verdict == "block":
        blocking = [f.pattern_id for f in injection.findings if f.severity == "block"]
        result.errors.append(f"description scanned positive for injected control content: {blocking}")
    elif injection.verdict == "warn":
        warned = [f.pattern_id for f in injection.findings]
        result.warnings.append(f"description scanned positive for suspicious content: {warned}")

    result.ok = not result.errors
    return result
