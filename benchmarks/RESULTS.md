# Redaction benchmark results

Corpus: `benchmarks/corpus/*.json` (5 synthetic incidents)

## Tier: `regex`

| Entity type | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| AZURE_SUBSCRIPTION_ID | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| AZURE_TENANT_ID | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| BEARER_TOKEN | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| CONNECTION_STRING | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| CREDIT_CARD | 0.00 | 0.00 | 0.00 | 0 | 0 | 1 |
| EMAIL | 1.00 | 1.00 | 1.00 | 3 | 0 | 0 |
| HOSTNAME | 1.00 | 1.00 | 1.00 | 2 | 0 | 0 |
| IPV4 | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| IPV6 | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| LOCATION | 0.00 | 0.00 | 0.00 | 0 | 0 | 1 |
| ORG | 0.00 | 0.00 | 0.00 | 0 | 0 | 1 |
| PERSON | 0.00 | 0.00 | 0.00 | 0 | 0 | 3 |
| PHONE_NUMBER | 0.00 | 0.00 | 0.00 | 0 | 0 | 1 |
| PRIVATE_KEY | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| SAS_TOKEN | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| **overall** | **1.00** | **0.65** | **0.79** | 13 | 0 | 7 |

- p50 latency: 0.02 ms/text, p95 latency: 0.05 ms/text (11 texts)
- estimated cost: $0.0000 per 1,000 incidents (assumes $0.10/compute-hour; detector CPU time only, not LLM token cost)

## Tier: `regex+presidio`

| Entity type | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| AZURE_SUBSCRIPTION_ID | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| AZURE_TENANT_ID | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| BEARER_TOKEN | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| CONNECTION_STRING | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| CREDIT_CARD | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| EMAIL | 1.00 | 1.00 | 1.00 | 3 | 0 | 0 |
| HOSTNAME | 1.00 | 1.00 | 1.00 | 2 | 0 | 0 |
| IPV4 | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| IPV6 | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| LOCATION | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| ORG | 0.00 | 0.00 | 0.00 | 0 | 0 | 1 |
| PERSON | 0.75 | 1.00 | 0.86 | 3 | 1 | 0 |
| PHONE_NUMBER | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| PRIVATE_KEY | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| SAS_TOKEN | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| **overall** | **0.95** | **0.95** | **0.95** | 19 | 1 | 1 |

- p50 latency: 4.52 ms/text, p95 latency: 10.77 ms/text (11 texts)
- estimated cost: $0.0001 per 1,000 incidents (assumes $0.10/compute-hour; detector CPU time only, not LLM token cost)
