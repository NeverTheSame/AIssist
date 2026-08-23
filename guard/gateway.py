"""Client proxy that intercepts chat.completions.create: the one place
every AzureOpenAI call in the app can be redacted before it leaves the
process, regardless of which of the app's 8 call sites (or a
team_knowledge/ module not on disk) made the call.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from . import context as guard_context
from .audit import AuditRecord, new_call_id, write_audit_record
from .detectors import Detector, Span
from .injection import InjectionResult, detect as detect_injection
from .settings import GuardSettings
from .vault import PseudonymVault

logger = logging.getLogger(__name__)


class GuardBlockedError(RuntimeError):
    """Raised when a detector errors and the policy is fail-closed, or when
    injection blocking is enabled and a message scans as malicious."""


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _redact_text(
    text: str, detectors: List[Detector], vault: PseudonymVault, fail_open: bool
) -> Tuple[str, List[Span]]:
    if not text:
        return text, []
    spans: List[Span] = []
    for detector in detectors:
        try:
            spans.extend(detector.detect(text))
        except Exception as exc:
            if fail_open:
                logger.warning("Detector %s failed (fail-open, continuing): %s", detector.name, exc)
                continue
            raise GuardBlockedError(f"Detector {detector.name} failed and policy is fail-closed") from exc
    return vault.redact(text, spans), spans


def _redact_message(
    message: Dict[str, Any], detectors: List[Detector], vault: PseudonymVault, fail_open: bool
) -> Tuple[Dict[str, Any], List[Span], int]:
    content = message.get("content")

    if isinstance(content, str):
        redacted, spans = _redact_text(content, detectors, vault, fail_open)
        return {**message, "content": redacted}, spans, 0

    if isinstance(content, list):
        new_parts = []
        all_spans: List[Span] = []
        image_count = 0
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else None
            if part_type == "image_url":
                # Images are an acknowledged, unredacted egress channel --
                # no OCR-based redaction here. Counted for the audit trail.
                image_count += 1
                new_parts.append(part)
            elif part_type == "text":
                redacted, spans = _redact_text(part.get("text", "") or "", detectors, vault, fail_open)
                all_spans.extend(spans)
                new_parts.append({**part, "text": redacted})
            else:
                new_parts.append(part)
        return {**message, "content": new_parts}, all_spans, image_count

    return message, [], 0


def _rehydrate_response(response: Any, vault: PseudonymVault) -> List[str]:
    unmatched: List[str] = []
    choices = getattr(response, "choices", None) or []
    for choice in choices:
        message = getattr(choice, "message", None)
        if message is None:
            continue
        content = getattr(message, "content", None)
        if isinstance(content, str):
            rehydrated, missing = vault.rehydrate(content)
            unmatched.extend(missing)
            try:
                message.content = rehydrated
            except Exception:
                logger.warning("Could not rehydrate response.choices[].message.content in place")
    return unmatched


class _GuardedCompletions:
    def __init__(
        self,
        raw_completions: Any,
        detectors: List[Detector],
        vault: PseudonymVault,
        settings: GuardSettings,
    ):
        self._raw = raw_completions
        self._detectors = detectors
        self._vault = vault
        self._settings = settings

    def create(self, *args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages")
        if not messages:
            # Nothing to redact; pass through untouched.
            return self._raw.create(*args, **kwargs)

        start = time.monotonic()
        entity_counts: Dict[str, int] = {}
        tiers_fired = set()
        image_count = 0
        redacted_messages = []
        injection_verdict = "clean"
        injection_pattern_ids: List[str] = []

        for message in messages:
            redacted_message, spans, images = _redact_message(
                message, self._detectors, self._vault, self._settings.fail_open
            )
            image_count += images
            for span in spans:
                entity_counts[span.entity_type] = entity_counts.get(span.entity_type, 0) + 1
                tiers_fired.add(span.tier)
            redacted_messages.append(redacted_message)

            if self._settings.injection_defense_enabled and redacted_message.get("role") == "user":
                result: InjectionResult = detect_injection(_extract_text(redacted_message.get("content")))
                if result.worse_than(injection_verdict):
                    injection_verdict = result.verdict
                injection_pattern_ids.extend(f.pattern_id for f in result.findings)

        if injection_verdict == "block" and self._settings.injection_block_enabled:
            raise GuardBlockedError(
                f"Message scanned positive for prompt injection ({injection_pattern_ids}) "
                "and GUARD_INJECTION_BLOCK_ENABLED is set"
            )

        call_kwargs = dict(kwargs)
        call_kwargs["messages"] = redacted_messages

        fail_open_triggered = False
        try:
            response = self._raw.create(*args, **call_kwargs)
        except Exception:
            raise

        unmatched = _rehydrate_response(response, self._vault)

        latency_ms = (time.monotonic() - start) * 1000
        usage = getattr(response, "usage", None)
        record = AuditRecord(
            call_id=new_call_id(),
            incident_id=guard_context.get_incident_id(),
            call_site=guard_context.get_call_site(),
            model=call_kwargs.get("model"),
            entity_counts=entity_counts,
            tiers_fired=sorted(tiers_fired),
            latency_ms=latency_ms,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            injection_verdict=injection_verdict if self._settings.injection_defense_enabled else None,
            rehydration_mismatches=unmatched,
            image_count=image_count,
            fail_open_triggered=fail_open_triggered,
        )
        write_audit_record(self._settings.audit_dir, record)

        return response

    def __getattr__(self, item: str) -> Any:
        return getattr(self._raw, item)


class _GuardedChat:
    def __init__(self, raw_chat: Any, completions: _GuardedCompletions):
        self._raw = raw_chat
        self.completions = completions

    def __getattr__(self, item: str) -> Any:
        return getattr(self._raw, item)


class GuardedClient:
    """Delegates everything except .chat.completions.create to the wrapped
    AzureOpenAI client, so unknown kwargs, other resources (embeddings,
    etc.), and SDK internals keep working unmodified."""

    def __init__(self, raw_client: Any, detectors: List[Detector], vault: PseudonymVault, settings: GuardSettings):
        self._raw = raw_client
        completions = _GuardedCompletions(raw_client.chat.completions, detectors, vault, settings)
        self.chat = _GuardedChat(raw_client.chat, completions)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._raw, item)


def wrap_chat_client(
    raw_client: Any, detectors: List[Detector], vault: PseudonymVault, settings: GuardSettings
) -> GuardedClient:
    return GuardedClient(raw_client, detectors, vault, settings)
