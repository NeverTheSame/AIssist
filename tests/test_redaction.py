"""Tests for guard/: detectors, vault round-trip, the gateway proxy, and
log redaction. No network access and no real Azure/OpenAI credentials
required -- everything here runs against fakes."""

import logging

import pytest

from guard.detectors import RegexDetector, Span, merge_spans
from guard.gateway import wrap_chat_client
from guard.logfilter import RedactionLogFilter
from guard.settings import GuardSettings
from guard.vault import PseudonymVault

SAMPLE_TEXT = (
    "Contact john.doe@contoso.com from host DESKTOP-AB12CD3, IP 10.0.0.5, "
    "subscription: 11111111-2222-3333-4444-555555555555, "
    "tenant: 66666666-7777-8888-9999-aaaaaaaaaaaa"
)


def make_settings(**overrides) -> GuardSettings:
    defaults = dict(enabled=True, fail_open=False, presidio_enabled=False,
                     llm_detector_enabled=False, injection_defense_enabled=False,
                     upn_domains=[], audit_dir="/tmp/guard_audit_test")
    defaults.update(overrides)
    return GuardSettings(**defaults)


class TestRegexDetector:
    def test_detects_email(self):
        spans = RegexDetector().detect(SAMPLE_TEXT)
        types = {s.entity_type for s in spans}
        assert "EMAIL" in types

    def test_detects_hostname(self):
        spans = RegexDetector().detect(SAMPLE_TEXT)
        matched = [SAMPLE_TEXT[s.start:s.end] for s in spans if s.entity_type == "HOSTNAME"]
        assert "DESKTOP-AB12CD3" in matched

    def test_detects_ipv4(self):
        spans = RegexDetector().detect(SAMPLE_TEXT)
        assert any(s.entity_type == "IPV4" for s in spans)

    def test_subscription_and_tenant_guid_disambiguated_by_context(self):
        spans = RegexDetector().detect(SAMPLE_TEXT)
        types = {s.entity_type for s in spans}
        assert "AZURE_SUBSCRIPTION_ID" in types
        assert "AZURE_TENANT_ID" in types
        # the contextual match should win over the generic GUID pattern
        assert "GUID" not in types

    def test_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"
        spans = RegexDetector().detect(text)
        assert any(s.entity_type == "BEARER_TOKEN" for s in spans)

    def test_private_key_block(self):
        text = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIBOgIBAAJBAK...\n"
            "-----END RSA PRIVATE KEY-----"
        )
        spans = RegexDetector().detect(text)
        assert len(spans) == 1
        assert spans[0].entity_type == "PRIVATE_KEY"

    def test_connection_string(self):
        text = "DefaultEndpointsProtocol=https;AccountName=myacct;AccountKey=abcd1234ABCD5678efgh9012IJKL==;EndpointSuffix=core.windows.net"
        spans = RegexDetector().detect(text)
        assert any(s.entity_type == "CONNECTION_STRING" for s in spans)

    def test_upn_domain_override(self):
        detector = RegexDetector(upn_domains=["contoso.com"])
        spans = detector.detect("Reach john.doe@contoso.com please")
        assert any(s.entity_type == "UPN" for s in spans)
        assert not any(s.entity_type == "EMAIL" for s in spans)

    def test_no_false_positive_on_plain_prose(self):
        text = "The incident was resolved after restarting the service twice."
        spans = RegexDetector().detect(text)
        assert spans == []

    def test_empty_string(self):
        assert RegexDetector().detect("") == []

    def test_non_string_content_ignored_gracefully(self):
        # Gateway never calls detect() on non-str content, but detect()
        # itself should not blow up on an empty/None-like input.
        assert RegexDetector().detect(None) == []


class TestMergeSpans:
    def test_overlapping_spans_prefer_longer_match(self):
        spans = [
            Span(0, 5, "SHORT", 0.5, "regex"),
            Span(0, 10, "LONG", 0.9, "regex"),
        ]
        merged = merge_spans(spans)
        assert len(merged) == 1
        assert merged[0].entity_type == "LONG"

    def test_non_overlapping_spans_both_kept(self):
        spans = [Span(0, 5, "A", 0.5, "regex"), Span(10, 15, "B", 0.5, "regex")]
        merged = merge_spans(spans)
        assert len(merged) == 2

    def test_result_sorted_by_start(self):
        spans = [Span(10, 15, "B", 0.5, "regex"), Span(0, 5, "A", 0.5, "regex")]
        merged = merge_spans(spans)
        assert [s.start for s in merged] == [0, 10]


class TestPresidioCache:
    def test_repeated_text_hits_cache_and_returns_same_spans(self):
        pytest.importorskip("presidio_analyzer")
        from guard.detectors import PresidioDetector

        detector = PresidioDetector()
        text = "My name is John Smith and I live in Seattle."

        first = detector.detect(text)
        assert len(detector._cache) == 1

        second = detector.detect(text)
        assert second == first
        assert len(detector._cache) == 1  # still one entry, not a duplicate

    def test_cache_is_bounded(self):
        pytest.importorskip("presidio_analyzer")
        from guard.detectors import PresidioDetector, _PRESIDIO_CACHE_MAX_ENTRIES

        detector = PresidioDetector()
        for i in range(_PRESIDIO_CACHE_MAX_ENTRIES + 10):
            detector.detect(f"Person number {i} is Jane Doe {i}.")
        assert len(detector._cache) == _PRESIDIO_CACHE_MAX_ENTRIES


class TestPseudonymVault:
    def test_round_trip(self):
        vault = PseudonymVault()
        spans = RegexDetector().detect(SAMPLE_TEXT)
        redacted = vault.redact(SAMPLE_TEXT, spans)
        assert "john.doe@contoso.com" not in redacted
        assert "DESKTOP-AB12CD3" not in redacted
        rehydrated, unmatched = vault.rehydrate(redacted)
        assert rehydrated == SAMPLE_TEXT
        assert unmatched == []

    def test_same_surface_form_gets_same_placeholder(self):
        vault = PseudonymVault()
        text = "john@contoso.com emailed john@contoso.com again"
        spans = RegexDetector().detect(text)
        redacted = vault.redact(text, spans)
        assert redacted.count("<EMAIL_1>") == 2

    def test_case_insensitive_surface_form_reuses_placeholder(self):
        vault = PseudonymVault()
        p1 = vault.placeholder_for("John@Contoso.com", "EMAIL")
        p2 = vault.placeholder_for("john@contoso.com", "EMAIL")
        assert p1 == p2

    def test_unmatched_placeholder_left_verbatim_and_flagged(self):
        vault = PseudonymVault()
        rehydrated, unmatched = vault.rehydrate("The model invented <PERSON_9> out of nowhere")
        assert "<PERSON_9>" in rehydrated
        assert unmatched == ["<PERSON_9>"]

    def test_empty_spans_returns_text_unchanged(self):
        vault = PseudonymVault()
        assert vault.redact("no entities here", []) == "no entities here"


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _RecordingCompletions:
    """Fake chat.completions that records the request it received and lets
    the test script a canned response."""

    def __init__(self, respond_with):
        self.last_kwargs = None
        self._respond_with = respond_with

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse(self._respond_with(kwargs))


def _echo_placeholder_response(kwargs):
    user_content = kwargs["messages"][-1]["content"]
    return f"Summary: {user_content}"


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeAzureOpenAIClient:
    def __init__(self, completions):
        self.chat = _FakeChat(completions)


class TestGateway:
    def test_redacts_outbound_and_rehydrates_inbound_via_wrap_chat_client(self):
        raw_completions = _RecordingCompletions(_echo_placeholder_response)
        raw_client = _FakeAzureOpenAIClient(raw_completions)
        guarded_client = wrap_chat_client(raw_client, [RegexDetector()], PseudonymVault(), make_settings())

        response = guarded_client.chat.completions.create(
            model="gpt-test",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Email me at jane@example.com"},
            ],
        )
        sent_user_content = raw_completions.last_kwargs["messages"][-1]["content"]
        assert "jane@example.com" not in sent_user_content
        assert "<EMAIL_1>" in sent_user_content
        assert "jane@example.com" in response.choices[0].message.content

    def test_guarded_completions_directly(self):
        raw_completions = _RecordingCompletions(_echo_placeholder_response)
        detectors = [RegexDetector()]
        vault = PseudonymVault()
        settings = make_settings()
        from guard.gateway import _GuardedCompletions

        guarded_completions = _GuardedCompletions(raw_completions, detectors, vault, settings)
        response = guarded_completions.create(
            model="gpt-test",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Email me at jane@example.com"},
            ],
        )
        # the raw call must never see the real email
        sent_user_content = raw_completions.last_kwargs["messages"][-1]["content"]
        assert "jane@example.com" not in sent_user_content
        assert "<EMAIL_1>" in sent_user_content
        # the response the caller sees must have the real value back
        assert "jane@example.com" in response.choices[0].message.content

    def test_image_parts_pass_through_unredacted_and_counted(self):
        raw_completions = _RecordingCompletions(lambda kwargs: "ok")
        from guard.gateway import _GuardedCompletions

        guarded_completions = _GuardedCompletions(raw_completions, [RegexDetector()], PseudonymVault(), make_settings())
        guarded_completions.create(
            model="gpt-test",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "contact jane@example.com"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                }
            ],
        )
        sent_content = raw_completions.last_kwargs["messages"][-1]["content"]
        text_part = next(p for p in sent_content if p["type"] == "text")
        image_part = next(p for p in sent_content if p["type"] == "image_url")
        assert "jane@example.com" not in text_part["text"]
        assert image_part["image_url"]["url"] == "data:image/png;base64,AAAA"

    def test_unknown_kwargs_pass_through(self):
        raw_completions = _RecordingCompletions(lambda kwargs: "ok")
        from guard.gateway import _GuardedCompletions

        guarded_completions = _GuardedCompletions(raw_completions, [RegexDetector()], PseudonymVault(), make_settings())
        guarded_completions.create(
            model="gpt-test",
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=500,
            some_future_kwarg="value",
        )
        assert raw_completions.last_kwargs["max_completion_tokens"] == 500
        assert raw_completions.last_kwargs["some_future_kwarg"] == "value"

    def test_fail_closed_on_detector_error(self):
        class BrokenDetector:
            name = "broken"

            def detect(self, text):
                raise RuntimeError("boom")

        from guard.gateway import _GuardedCompletions, GuardBlockedError

        raw_completions = _RecordingCompletions(lambda kwargs: "ok")
        guarded_completions = _GuardedCompletions(
            raw_completions, [BrokenDetector()], PseudonymVault(), make_settings(fail_open=False)
        )
        with pytest.raises(GuardBlockedError):
            guarded_completions.create(model="gpt-test", messages=[{"role": "user", "content": "hi"}])
        assert raw_completions.last_kwargs is None

    def test_fail_open_on_detector_error(self):
        class BrokenDetector:
            name = "broken"

            def detect(self, text):
                raise RuntimeError("boom")

        from guard.gateway import _GuardedCompletions

        raw_completions = _RecordingCompletions(lambda kwargs: "ok")
        guarded_completions = _GuardedCompletions(
            raw_completions, [BrokenDetector()], PseudonymVault(), make_settings(fail_open=True)
        )
        response = guarded_completions.create(model="gpt-test", messages=[{"role": "user", "content": "hi"}])
        assert response.choices[0].message.content == "ok"
        assert raw_completions.last_kwargs is not None


class TestWrapClientDisabled:
    def test_wrap_client_is_noop_when_disabled(self):
        import guard

        class FakeClient:
            pass

        raw = FakeClient()
        wrapped = guard.wrap_client(raw, settings=make_settings(enabled=False))
        assert wrapped is raw


class TestLogRedaction:
    def test_filter_redacts_pii_in_log_record(self):
        records = []

        class ListHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        logger = logging.getLogger("guard_test_logger")
        logger.setLevel(logging.INFO)
        handler = ListHandler()
        handler.addFilter(RedactionLogFilter([RegexDetector()], PseudonymVault()))
        logger.addHandler(handler)

        logger.info("User email: %s", "leak@example.com")

        assert len(records) == 1
        assert records[0].getMessage() == "User email: <EMAIL_1>"
        logger.removeHandler(handler)

    def test_filter_leaves_benign_messages_untouched(self):
        records = []

        class ListHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        logger = logging.getLogger("guard_test_logger_benign")
        logger.setLevel(logging.INFO)
        handler = ListHandler()
        handler.addFilter(RedactionLogFilter([RegexDetector()], PseudonymVault()))
        logger.addHandler(handler)

        logger.info("Processing completed successfully")

        assert records[0].getMessage() == "Processing completed successfully"
        logger.removeHandler(handler)
