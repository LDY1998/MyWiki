---
name: wiki-lint
description: Use when auditing this MyWiki repo for structural wiki quality, stale or unsupported claims, broken links, duplicate concepts, or missing source coverage.
---

# Wiki Lint

Audit first, fix later. The default output is a report, not a rewrite.

## Inputs

- `wiki/index.md`
- `wiki/Wiki Maintenance Schema.md`
- `wiki/**/*.md`
- `claims/claims.yaml`
- `raw/`
- `tools/scripts/validate_wiki.py`

## Workflow

1. Read the index and maintenance schema.
2. Run `python3 tools/scripts/validate_wiki.py`.
3. Inspect high-signal issues: missing frontmatter, broken wikilinks, orphan pages, raw sources without synthesis, stale or weakly sourced claims, duplicate concepts, and contradictions.
4. Write a dated report to `claims/contradiction_reports/lint-YYYY-MM-DD.md`.
5. Include severity groups: high-priority issues, medium-priority cleanup, low-priority housekeeping.
6. Propose fixes with affected files.
7. Append an entry to `wiki/Infra Wiki Log.md`.

## Rules

- Do not apply major fixes unless the user explicitly asks.
- Do not treat example links inside code blocks or inline code as real wiki links.
- Prefer concrete file paths and wikilinks over vague descriptions.
- If a claim cannot be verified, recommend marking it `unverified`.

## Output

Report the lint file created, validation summary, highest-priority findings, and whether any edits beyond the report were made.
