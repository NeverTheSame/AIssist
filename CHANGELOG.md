# Changelog

## 2026-08-28 — pgvector retrieval backend for article search

Added `pgvector_store.py`, a Postgres+pgvector alternative to the Qdrant
path in `article_searcher.py` for KB/article grounding. Same embedding
space (all-MiniLM-L6-v2, 384 dimensions) and same candidate shape out of
`_semantic_search()`, so callers don't branch on which backend answered.

- `pgvector_store.py` — `connect()`, `ensure_schema()` (table + HNSW cosine
  index), `upsert_articles()`, `search()` (cosine nearest-neighbor via
  `<=>`); identifiers are validated before being formatted into SQL
- `article_searcher.py` — `_load_articles()` opens a pgvector connection
  when `PGVECTOR_DSN` is set (or `VECTOR_DB_PATH` is a `postgres(ql)://`
  URL), tried ahead of Qdrant in `_semantic_search()`; fixed an
  initialization-order bug where `self.qdrant_client`/`self.pgvector_conn`
  were reset to `None` in `__init__` *after* `_load_articles()` had
  already opened them, silently discarding the connection
- `config.py` — `PGVECTOR_DSN`, `PGVECTOR_TABLE`; fixed `_load_env()` to use
  `os.environ.setdefault()` instead of unconditional assignment, so a real
  value already exported by the shell is no longer clobbered by a blank
  `.env.example` placeholder when `.env` is absent
- `tests/test_pgvector_store.py` — 6 tests against a real
  `pgvector/pgvector:pg16` container, skipped unless `PGVECTOR_TEST_DSN` is
  set (not run without live infrastructure means not claimed, matching the
  rest of this project's honesty pattern)
- `requirements.txt`, `.env.example`, `README.md` updated

## 2026-08-23 — LLM security layer: redaction gateway, injection defense, audit trail (`2c8a010`)

Added `guard/`, a package that wraps every `AzureOpenAI` client at its one
factory (`azure_auth.get_openai_client_with_auth`) to redact PII before it
reaches the model provider and rehydrate it in responses, scrubs the same
PII from log output and debug dumps, spotlights attacker-influenceable
incident text against prompt injection, and validates LLM-generated
content before the one Azure DevOps write path that accepts it. Gated
behind `GUARD_ENABLED` (default `false`), so behavior is unchanged until a
deployment opts in.

- `guard/` — settings, PII detectors (regex always-on, Presidio NER
  opt-in with a content-hash cache, LLM tier opt-in), a reversible
  pseudonymization vault, the client-wrapping gateway, log redaction,
  prompt-injection spotlighting/detection, tool-boundary output
  validation, and a JSONL audit trail (`.guard_audit/`, gitignored)
- `benchmarks/` — synthetic labeled corpus (`benchmarks/corpus/`) and
  `run_redaction_benchmark.py`, reporting precision/recall/F1 and latency
  per detector tier (`benchmarks/RESULTS.md`), with committed regression
  thresholds (`benchmarks/thresholds.json`, `check_thresholds.py`)
- `redteam/injections.yaml` + `tests/` — red-team corpus and test suite
  for the injection detector and redaction gateway, including a
  cross-incident mem0-poisoning check
- `cmd/promptguard/` — a standalone Go static checker (own `go.mod`) that
  flags `AzureOpenAI` clients built outside the sanctioned factory,
  `print()` calls that leak sensitive variables past the log filter, and
  file writes to paths `.gitignore` doesn't cover; wired into
  `.github/workflows/security.yml` and `.pre-commit-config.yaml`
- `docs/THREAT_MODEL.md` — architecture, detector-tier tradeoffs, and
  honestly-scoped limitations (image redaction is out of scope, Presidio
  doesn't detect ORG out of the box, hostname/IPv6 detection is
  heuristic, rehydration widens the trust boundary)
- Environment fixes: `config.py` now falls back to `.env.example`
  (with a warning) instead of raising when `.env` is missing, so the repo
  is importable and testable with zero secrets; `.gitignore` no longer
  silently drops `docs/`, `benchmarks/*.md`, `redteam/*.md`, or nested
  `__pycache__` dirs
