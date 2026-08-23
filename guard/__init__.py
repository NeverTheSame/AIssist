"""AIssist security layer: redaction gateway, log scrubbing, prompt-
injection defense, and an audit trail -- all disabled by default (see
GUARD_ENABLED in .env.example) so installing it changes nothing until a
deployment opts in.
"""

import logging
from typing import Any, Iterable, List, Optional

from .context import get_call_site, get_incident_id, set_incident_context
from .detectors import Detector, RegexDetector, create_presidio_detector
from .gateway import GuardBlockedError, wrap_chat_client
from .injection import SPOTLIGHT_SYSTEM_CLAUSE, InjectionResult, detect as detect_injection, spotlight
from .logfilter import install as _install_logfilter
from .settings import GuardSettings, load_settings
from .validation import ValidationResult, validate_work_item
from .vault import PseudonymVault

logger = logging.getLogger(__name__)

__all__ = [
    "wrap_client",
    "install_log_redaction",
    "redact_text",
    "load_settings",
    "GuardSettings",
    "GuardBlockedError",
    "set_incident_context",
    "get_incident_id",
    "get_call_site",
    "spotlight_if_enabled",
    "injection_system_clause_suffix",
    "detect_injection",
    "validate_work_item",
    "ValidationResult",
    "InjectionResult",
]

_shared_vault: Optional[PseudonymVault] = None


def _vault() -> PseudonymVault:
    global _shared_vault
    if _shared_vault is None:
        _shared_vault = PseudonymVault()
    return _shared_vault


def _detectors_for(settings: GuardSettings) -> List[Detector]:
    detectors: List[Detector] = [RegexDetector(upn_domains=settings.upn_domains)]
    if settings.presidio_enabled:
        presidio = create_presidio_detector()
        if presidio is not None:
            detectors.append(presidio)
    return detectors


def wrap_client(client: Any, settings: Optional[GuardSettings] = None) -> Any:
    """Wrap an AzureOpenAI client so chat.completions.create redacts PII
    before it leaves the process and rehydrates it in the response.

    Returns the client unchanged when guard is disabled (GUARD_ENABLED is
    not set) -- the one behavioural switch that must default to a no-op.
    """
    settings = settings or load_settings()
    if not settings.enabled:
        return client
    return wrap_chat_client(client, detectors=_detectors_for(settings), vault=_vault(), settings=settings)


def install_log_redaction(handlers: Iterable[logging.Handler], settings: Optional[GuardSettings] = None) -> None:
    """Attach redaction to the given logging handlers in place. No-op when
    guard is disabled."""
    settings = settings or load_settings()
    if not settings.enabled:
        return
    _install_logfilter(handlers, _detectors_for(settings), _vault())


def redact_text(text: str, settings: Optional[GuardSettings] = None) -> str:
    """Scrub PII from a raw string bound for a debug/diagnostic file that
    doesn't go through the logging module (e.g. the fetcher subprocess
    dump). Returns text unchanged when guard is disabled."""
    settings = settings or load_settings()
    if not settings.enabled or not text:
        return text
    detectors = _detectors_for(settings)
    spans = []
    for detector in detectors:
        try:
            spans.extend(detector.detect(text))
        except Exception:
            logger.debug("redact_text: detector %s failed; skipping", detector.name)
    if not spans:
        return text
    from .detectors import merge_spans

    return _vault().redact(text, merge_spans(spans))


def spotlight_if_enabled(text: str, settings: Optional[GuardSettings] = None) -> str:
    """Wrap untrusted incident text in spotlighting delimiters. No-op text
    passthrough when injection defense is disabled."""
    settings = settings or load_settings()
    if not settings.injection_defense_enabled or not text:
        return text
    return spotlight(text)


def injection_system_clause_suffix(settings: Optional[GuardSettings] = None) -> str:
    """Text to append to the system prompt so the model knows spotlighted
    blocks are untrusted data. Empty string when injection defense is
    disabled, so callers can unconditionally concatenate it."""
    settings = settings or load_settings()
    if not settings.injection_defense_enabled:
        return ""
    return "\n\n" + SPOTLIGHT_SYSTEM_CLAUSE
