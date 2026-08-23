# AIssist security layer: threat model

## What this protects against

AIssist ships customer support incident text -- author aliases/UPNs, emails,
IPs, hostnames, device IDs, machine GUIDs, tenant/subscription IDs, and log
excerpts -- to Azure OpenAI for summarization, and optionally writes the
result into Azure DevOps. Before this work, nothing between Kusto and the
model provider actually de-identified that text; the three functions that
looked like sanitization (`remove_img_data_tags`, `clean_html_content`,
`clean_azure_support_info`) strip formatting and boilerplate, not PII.

`guard/` adds:

1. **A redaction gateway** at the one chokepoint every `AzureOpenAI` client
   in the app is built from (`azure_auth.get_openai_client_with_auth`), so
   all 8 `chat.completions.create` call sites -- including the gitignored
   `team_knowledge/` modules not present in this clone -- are covered by
   editing one file.
2. **Log redaction** attached to every `logging.FileHandler`/`StreamHandler`
   the app creates, so the raw-prompt debug block and summary dump in
   `processor.py`, and the fetcher subprocess dump in `main.py`, are
   scrubbed without touching individual log statements.
3. **Prompt-injection defense**: spotlighting the untrusted incident text
   with explicit delimiters, a heuristic detector for instruction-shaped
   content, and output validation before the one Azure DevOps write path
   that accepts LLM-generated content.
4. **An audit trail**: one JSONL record per guarded call, schema-aligned to
   a future Log Analytics custom table.

Everything is gated by `GUARD_ENABLED` (default `false`); the layer changes
nothing about existing behaviour until a deployment opts in.

## Threat actors and channels

| Actor | Channel | Mitigation |
|---|---|---|
| Passive observer of provider logs/telemetry | `chat.completions.create` payload | Gateway redacts before send, rehydrates response |
| Anyone with read access to `logs/*.log` or `error.log` | Log files | Log-handler redaction filter; `error.log` now gitignored |
| Customer/partner writing into an incident's Teams/email thread | Prompt-injection via incident text concatenated at `processor.py:412` | Spotlighting + heuristic detector (`guard/injection.py`) |
| A future automation that removes the human `input()` gate in `main.py:1102,:1109` | Autonomous Azure DevOps write | Output validation (`guard/validation.py`) pre-positioned now, before that gate is removed |
| Incident A's data leaking into incident B's prompt via mem0 | `processor.py:503-505`, `:1262` | Covered by the same gateway (mem0-enhanced prompts still flow through `chat.completions.create`); a dedicated poisoning test in `tests/test_injection.py` asserts this |

## Detector tiers and why each exists

| Tier | Catches | Cost | Default |
|---|---|---|---|
| `RegexDetector` | Structured tokens: emails/UPNs, IPv4/IPv6, GUIDs, subscription/tenant IDs, bearer tokens, connection strings, private key blocks, SAS tokens, hostnames | ~0.03ms/text, zero deps | Always on |
| `PresidioDetector` | Unstructured NER: PERSON, LOCATION, PHONE_NUMBER, CREDIT_CARD | ~5-13ms/text, +400MB spaCy model | On when `presidio_analyzer` is installed |
| `LLMDetector` | Contextual leaks regex/NER both miss (identity implied by business context, secrets embedded in prose) | One extra model call per text | Opt-in (`GUARD_LLM_DETECTOR_ENABLED`), off by default |

See `benchmarks/RESULTS.md` for measured precision/recall/F1 per entity
type per tier, generated from `benchmarks/run_redaction_benchmark.py`
against the synthetic corpus in `benchmarks/corpus/`.

## Honest limits

- **Images bypass redaction entirely.** `summary_images` (base64
  screenshots) are sent to the model as-is; the gateway passes
  `image_url` parts through untouched and only counts them in the audit
  record. No OCR-based redaction exists. This is a deliberate,
  documented gap, not an oversight -- and it's a real one, since
  `transformer.py:529-547` reconstitutes images from `manual.docx`
  precisely when the upstream text was `** REDACTED **`.
- **Presidio's default recognizer does not detect organization names.**
  `ORG` is requested in `guard/detectors.py`'s entity list for
  forward-compatibility, but scores 0% recall until a custom recognizer
  is registered -- see `benchmarks/RESULTS.md`.
- **Hostname/FQDN detection is a heuristic, not a grammar.** It catches
  Windows-style machine names (`DESKTOP-XXXXXX`) and FQDNs whose first
  label mixes letters and digits; it will miss a hostname that looks like
  an ordinary word and can false-positive on machine-generated-looking
  domains in legitimate URLs.
- **IPv6 detection requires >=3 colon-separated groups**, a deliberate
  tradeoff to avoid matching `HH:MM:SS` timestamps as addresses. This
  misses maximally-compressed forms like `::1`.
- **Rehydration widens the trust boundary.** Placeholders in the model's
  response are substituted back to real values, so a model that emits
  `<EMAIL_1>` somewhere unexpected reintroduces the real value into
  whatever consumes that response (e.g. an Azure DevOps description).
  Output validation (`guard/validation.py`) must run, and does run,
  *after* rehydration, not before.
- **The audit trail is not auditor-facing evidence.** It's an audit trail
  with a schema designed for evidence collection -- field names line up
  with a Log Analytics custom table -- not SOC 2 evidence on its own.
- **Fail-closed by default.** A detector exception blocks the call
  (`GuardBlockedError`) rather than silently letting unredacted text
  through. `GUARD_FAIL_OPEN=true` overrides this for daily local use,
  trading availability for the (small) chance a broken detector lets
  something through undetected.
- **The tool boundary this defends is not live yet.**
  `create_preventative_action_work_item` (`azure_devops_client.py:591`)
  has exactly one caller (`main.py:1115`), and its `title` and
  `repair_item_type` come from human `input()` -- only `description` is
  LLM-generated today. `guard/validation.py` is pre-positioned for the
  automation in `pa_triage_runner.py`/`fetch_new_pa.py`, which is what
  makes this control worth having built before that gate is removed,
  not because it closes a live autonomous-write hole right now.
- **Out of scope:** `processor.save_results` is called at `main.py:790`
  but is not defined anywhere in this clone -- a pre-existing
  `AttributeError` unrelated to this work, noted and not fixed here.
