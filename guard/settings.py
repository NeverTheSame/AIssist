"""Env-driven configuration for the guard package.

Deliberately does not import config.py: guard must stay importable (and
therefore testable) with no .env, no Kusto/Azure credentials, and no network
access at all.
"""

import os
from dataclasses import dataclass, field
from typing import List


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class GuardSettings:
    # Master switch. Everything in guard/ is a no-op when this is False, so
    # the layer can ship without changing default behaviour.
    enabled: bool = False

    # Detector error handling: fail-closed (block the call) is the default
    # per the threat model; set GUARD_FAIL_OPEN=true for daily local use.
    fail_open: bool = False

    presidio_enabled: bool = True
    llm_detector_enabled: bool = False

    # Injection *detection* (spotlighting + heuristic scan) vs. injection
    # *blocking*: detection alone is safe to turn on by default once phase 2
    # ships, since it only annotates the audit record; blocking is a
    # separate, more disruptive switch a deployment opts into once it
    # trusts the false-positive rate.
    injection_defense_enabled: bool = False
    injection_block_enabled: bool = False

    # Domain suffixes (comma-separated, lowercase) that mark an email as a
    # corporate UPN rather than a generic external email address.
    upn_domains: List[str] = field(default_factory=list)

    audit_dir: str = ".guard_audit"


def load_settings() -> GuardSettings:
    return GuardSettings(
        enabled=_bool("GUARD_ENABLED", False),
        fail_open=_bool("GUARD_FAIL_OPEN", False),
        presidio_enabled=_bool("GUARD_PRESIDIO_ENABLED", True),
        llm_detector_enabled=_bool("GUARD_LLM_DETECTOR_ENABLED", False),
        injection_defense_enabled=_bool("GUARD_INJECTION_DEFENSE_ENABLED", False),
        injection_block_enabled=_bool("GUARD_INJECTION_BLOCK_ENABLED", False),
        upn_domains=[
            d.strip().lower()
            for d in os.environ.get("GUARD_UPN_DOMAINS", "").split(",")
            if d.strip()
        ],
        audit_dir=os.environ.get("GUARD_AUDIT_DIR", ".guard_audit"),
    )
