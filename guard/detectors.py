"""Detector tiers: Regex (always on), Presidio (NER, opt-in via install),
LLM (contextual, opt-in via settings).

Follows the house style of kusto_fetcher.remove_img_data_tags(): module-level
compiled patterns, plain functions, no classes where a function will do.
"""

import hashlib
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    entity_type: str
    confidence: float
    tier: str


class Detector(Protocol):
    name: str

    def detect(self, text: str) -> List[Span]:
        ...


def merge_spans(spans: List[Span]) -> List[Span]:
    """Resolve overlaps deterministically: longer match wins, ties broken by
    the order the caller supplied (i.e. put higher-priority detectors/
    patterns first). Keeps placeholder substitution well-defined -- the
    vault requires non-overlapping spans."""
    if not spans:
        return []
    ordered = sorted(enumerate(spans), key=lambda pair: (-(pair[1].end - pair[1].start), pair[0]))
    kept: List[Span] = []
    claimed: List[range] = []
    for _, span in ordered:
        span_range = range(span.start, span.end)
        if any(span.start < c.stop and span_range.stop > c.start for c in claimed):
            continue
        kept.append(span)
        claimed.append(span_range)
    return sorted(kept, key=lambda s: s.start)


# --- Regex tier -------------------------------------------------------

_GUID = r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
# Handles "::" zero-compression (e.g. 2001:db8:85a3::8a2e:370:7334) via a
# permissive hex:colon run rather than the classic (and here, ambiguous
# under Python's first-match alternation) fully-enumerated IPv6 regex.
# Requires >=3 colon-separated groups to avoid matching HH:MM:SS timestamps
# (2 groups) -- a deliberate false-negative on maximally-compressed
# addresses like "::1" in exchange for not flagging every log timestamp.
_IPV6_RE = re.compile(r"\b[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{0,4}){3,7}\b")
_SUBSCRIPTION_CTX_RE = re.compile(r"(?i)subscription[a-z\s]{0,15}?[:=]?\s*(" + _GUID + r")")
_TENANT_CTX_RE = re.compile(r"(?i)tenant[a-z\s]{0,15}?[:=]?\s*(" + _GUID + r")")
_GUID_RE = re.compile(r"\b" + _GUID + r"\b")
_BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9\-_.=]{10,}")
_CONN_STR_RE = re.compile(
    r"(?i)DefaultEndpointsProtocol=[^;\s]+;AccountName=[^;\s]+;AccountKey=[A-Za-z0-9+/=]{20,};?[^\s\"']*"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    re.DOTALL,
)
_SAS_TOKEN_RE = re.compile(r"(?i)\bsv=\d{4}-\d{2}-\d{2}[^\s\"']*?sig=[A-Za-z0-9%]{10,}")
# Windows-style machine names (DESKTOP-AB12CD3, WIN-XYZ123) and FQDNs whose
# first label looks machine-generated (mixes letters and digits). This is a
# best-effort heuristic, not a real hostname grammar -- see
# docs/THREAT_MODEL.md for the false-negative/positive tradeoff.
_WIN_HOSTNAME_RE = re.compile(r"(?i)\b(?:DESKTOP|LAPTOP|WIN|WORKSTATION|PC|SRV|VM)-[A-Z0-9]{6,}\b")
_FQDN_RE = re.compile(
    r"\b(?=[a-z0-9-]{1,63}\.)(?=[a-z0-9-]*\d)[a-z0-9]([a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9-]{2,63}){1,}\b",
    re.IGNORECASE,
)


def _regex_scan(text: str, upn_domains: Optional[List[str]] = None) -> List[Span]:
    """Priority-ordered so merge_spans keeps the more specific match when
    patterns overlap (e.g. a GUID inside a "subscription: <guid>" phrase)."""
    upn_domains = upn_domains or []
    spans: List[Span] = []

    for m in _PRIVATE_KEY_RE.finditer(text):
        spans.append(Span(m.start(), m.end(), "PRIVATE_KEY", 0.99, "regex"))
    for m in _CONN_STR_RE.finditer(text):
        spans.append(Span(m.start(), m.end(), "CONNECTION_STRING", 0.95, "regex"))
    for m in _SAS_TOKEN_RE.finditer(text):
        spans.append(Span(m.start(), m.end(), "SAS_TOKEN", 0.9, "regex"))
    for m in _BEARER_RE.finditer(text):
        spans.append(Span(m.start(), m.end(), "BEARER_TOKEN", 0.9, "regex"))
    for m in _SUBSCRIPTION_CTX_RE.finditer(text):
        spans.append(Span(m.start(1), m.end(1), "AZURE_SUBSCRIPTION_ID", 0.85, "regex"))
    for m in _TENANT_CTX_RE.finditer(text):
        spans.append(Span(m.start(1), m.end(1), "AZURE_TENANT_ID", 0.85, "regex"))
    for m in _GUID_RE.finditer(text):
        spans.append(Span(m.start(), m.end(), "GUID", 0.6, "regex"))
    for m in _IPV6_RE.finditer(text):
        spans.append(Span(m.start(), m.end(), "IPV6", 0.7, "regex"))
    for m in _IPV4_RE.finditer(text):
        spans.append(Span(m.start(), m.end(), "IPV4", 0.85, "regex"))
    for m in _WIN_HOSTNAME_RE.finditer(text):
        spans.append(Span(m.start(), m.end(), "HOSTNAME", 0.75, "regex"))
    for m in _FQDN_RE.finditer(text):
        spans.append(Span(m.start(), m.end(), "HOSTNAME", 0.5, "regex"))
    for m in _EMAIL_RE.finditer(text):
        domain = m.group(0).rsplit("@", 1)[-1].lower()
        entity_type = "UPN" if domain in upn_domains else "EMAIL"
        spans.append(Span(m.start(), m.end(), entity_type, 0.9, "regex"))

    return merge_spans(spans)


class RegexDetector:
    name = "regex"

    def __init__(self, upn_domains: Optional[List[str]] = None):
        self._upn_domains = upn_domains or []

    def detect(self, text: str) -> List[Span]:
        if not text:
            return []
        return _regex_scan(text, self._upn_domains)


# --- Presidio tier (NER; opt-in via requirements-redaction.txt) -------

# NOTE: Presidio's default spaCy-backed recognizer does not emit ORG --
# organization names aren't treated as PII by its out-of-the-box config.
# ORG is kept in the requested list so a custom recognizer can be dropped
# in later without touching call sites, but until then it always scores
# 0% recall in benchmarks/run_redaction_benchmark.py. Documented, not a bug.
_PRESIDIO_ENTITIES = ["PERSON", "LOCATION", "ORG", "PHONE_NUMBER", "CREDIT_CARD"]


_PRESIDIO_CACHE_MAX_ENTRIES = 512


class PresidioDetector:
    """Presidio's NER pass is ~5-13ms/call (see benchmarks/RESULTS.md) --
    real money when a caller fires repeated sequential LLM calls over
    overlapping text, e.g. article_searcher.py's per-candidate scoring
    loop. Cache by content hash, bounded and evicted oldest-first, so an
    exact repeat of the same text is free on every call after the first."""

    name = "presidio"

    def __init__(self):
        from presidio_analyzer import AnalyzerEngine  # deferred: heavy, optional dep

        self._engine = AnalyzerEngine()
        self._cache: "OrderedDict[str, List[Span]]" = OrderedDict()

    def detect(self, text: str) -> List[Span]:
        if not text:
            return []

        cache_key = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

        results = self._engine.analyze(text=text, entities=_PRESIDIO_ENTITIES, language="en")
        spans = [Span(r.start, r.end, r.entity_type, r.score, "presidio") for r in results]

        self._cache[cache_key] = spans
        self._cache.move_to_end(cache_key)
        if len(self._cache) > _PRESIDIO_CACHE_MAX_ENTRIES:
            self._cache.popitem(last=False)
        return spans


def create_presidio_detector() -> Optional["PresidioDetector"]:
    """Returns None (rather than raising) when presidio_analyzer isn't
    installed, so guard/__init__.py can fall back to the regex tier alone."""
    try:
        return PresidioDetector()
    except Exception:
        logger.warning(
            "Presidio detector unavailable (install requirements-redaction.txt); "
            "falling back to the regex tier only.",
            exc_info=True,
        )
        return None


# --- LLM tier (contextual leaks regex/NER miss; opt-in, off by default) --

_LLM_DETECTOR_SYSTEM_PROMPT = """You are a PII/secret detector. Given a block of text, \
find sensitive entities that a regex or NER pass would plausibly miss: things like a \
customer's business context implying identity, an internal project codename tied to a \
named person, a secret embedded in prose rather than a token-shaped string, or a \
description that uniquely identifies a person or machine without a formal token.

Reply with ONLY a JSON array, no prose, no markdown fences. Each element:
{"text": "<exact substring from the input>", "entity_type": "<UPPER_SNAKE_CASE label>"}
If nothing qualifies, reply with []."""


class LLMDetector:
    """Contextual tier. Uses a caller-supplied *raw* (unwrapped) client so
    this detector's own classification call never recurses back through the
    gateway it is feeding."""

    name = "llm"

    def __init__(self, raw_client, deployment_name: str):
        self._client = raw_client
        self._deployment_name = deployment_name

    def detect(self, text: str) -> List[Span]:
        if not text or not text.strip():
            return []
        try:
            response = self._client.chat.completions.create(
                model=self._deployment_name,
                messages=[
                    {"role": "system", "content": _LLM_DETECTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": text[:8000]},
                ],
                temperature=0,
                max_tokens=1000,
            )
            raw = response.choices[0].message.content or "[]"
        except Exception:
            logger.exception("LLMDetector call failed; contributing no additional spans")
            return []
        return _parse_llm_spans(raw, text)


def _parse_llm_spans(raw: str, text: str) -> List[Span]:
    import json

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("LLMDetector returned non-JSON output; ignoring")
        return []

    spans: List[Span] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        surface = item.get("text")
        entity_type = item.get("entity_type")
        if not surface or not entity_type:
            continue
        idx = text.find(surface)
        if idx == -1:
            continue
        spans.append(Span(idx, idx + len(surface), f"LLM_{entity_type}", 0.5, "llm"))
    return merge_spans(spans)
