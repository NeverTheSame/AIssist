"""Reversible pseudonymization: detected spans <-> stable placeholders.

Placeholders are keyed by (entity_type, normalized surface form) so the
same person/host/etc. gets the same placeholder across every call in a
process lifetime -- the model can reason about "the same machine" across
turns without ever seeing its real name -- and rehydration on the way back
restores the real values for the analyst.
"""

import re
import threading
from typing import Dict, List, Tuple

from .detectors import Span

PLACEHOLDER_RE = re.compile(r"<([A-Z][A-Z0-9_]*)_(\d+)>")


class PseudonymVault:
    """Per-process, in-memory placeholder mapping. Not thread-safe across
    processes by design -- a fresh vault per run keeps entity numbering
    small and avoids persisting real values to disk."""

    def __init__(self):
        self._lock = threading.Lock()
        self._surface_to_placeholder: Dict[Tuple[str, str], str] = {}
        self._placeholder_to_original: Dict[str, str] = {}
        self._counters: Dict[str, int] = {}

    @staticmethod
    def _normalize(surface: str) -> str:
        return surface.strip().lower()

    def placeholder_for(self, surface: str, entity_type: str) -> str:
        key = (entity_type, self._normalize(surface))
        with self._lock:
            placeholder = self._surface_to_placeholder.get(key)
            if placeholder is not None:
                return placeholder
            self._counters[entity_type] = self._counters.get(entity_type, 0) + 1
            placeholder = f"<{entity_type}_{self._counters[entity_type]}>"
            self._surface_to_placeholder[key] = placeholder
            self._placeholder_to_original[placeholder] = surface
            return placeholder

    def redact(self, text: str, spans: List[Span]) -> str:
        """Replace each span with its placeholder. Spans must be
        non-overlapping; apply in descending start order so earlier
        offsets stay valid as the string shrinks/grows."""
        if not spans:
            return text
        ordered = sorted(spans, key=lambda s: s.start, reverse=True)
        for span in ordered:
            surface = text[span.start:span.end]
            placeholder = self.placeholder_for(surface, span.entity_type)
            text = text[:span.start] + placeholder + text[span.end:]
        return text

    def rehydrate(self, text: str) -> Tuple[str, List[str]]:
        """Substitute placeholders back to real values. Placeholders the
        vault never minted (the model invented one, or echoed a stale one
        from a different process) are left verbatim and reported so the
        caller can flag them in the audit record instead of silently
        dropping or mistranslating them."""
        unmatched: List[str] = []

        def _sub(match: "re.Match") -> str:
            placeholder = match.group(0)
            original = self._placeholder_to_original.get(placeholder)
            if original is None:
                unmatched.append(placeholder)
                return placeholder
            return original

        result = PLACEHOLDER_RE.sub(_sub, text)
        return result, unmatched
