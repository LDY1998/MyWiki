---
title: Claims
type: claim-index
status: active
updated: 2026-04-27
---

# Claims

Use this folder for atomic, source-grounded claims and lint reports.

- `claims.yaml` stores structured claims extracted from durable wiki pages.
- `contradiction_reports/` stores lint reports for stale, weak, duplicate, or contradictory claims.

Do not invent claims without source references. If a claim cannot be verified, mark it as unverified when adding it to `claims.yaml`.

## Workflow

Use `knowledge-ingest` to add useful source-grounded claims during ingestion. Use `wiki-lint` to audit claims and write reports under `claims/contradiction_reports/`.

## Minimal Claim Shape

```yaml
- claim_id: claim_YYYY_MM_DD_001
  text: "Atomic claim."
  source_pages:
    - wiki/concepts/Example.md
  source_raw:
    - raw/infra/Example Source.md
  confidence: high
  status: active
  last_verified: YYYY-MM-DD
```

Allowed `confidence`: `low`, `medium`, `high`.

Allowed `status`: `active`, `unverified`, `contradicted`, `stale`, `deprecated`.
