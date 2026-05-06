---
name: knowledge-ingest
description: Use when ingesting a raw source file into this MyWiki repo, especially files under raw/_inbox, raw/ai, or raw/infra.
---

# Knowledge Ingest

Transform one source file into grounded wiki synthesis. Prefer small, auditable edits over broad rewrites.

## Inputs

- Exactly one raw source file unless the user explicitly asks for a batch.
- `wiki/index.md`
- `wiki/Wiki Maintenance Schema.md`
- Relevant existing wiki pages.

## Workflow

1. Read `wiki/index.md` and `wiki/Wiki Maintenance Schema.md`.
2. Read the source file and identify durable concepts, claims, and source type.
3. Read relevant existing wiki pages before creating new ones.
4. Create or update a source summary in `wiki/sources/` or a paper page in `wiki/papers/`.
5. Update durable synthesis pages in `wiki/concepts/`, `wiki/systems/`, `wiki/comparisons/`, `wiki/maps/`, or `wiki/questions/` when the source changes understanding.
6. Add source links using Obsidian wikilinks for internal pages and Markdown links for external URLs.
7. Add important atomic claims to `claims/claims.yaml` only when they are useful and source-grounded.
8. Update `wiki/index.md`.
9. Append a concise entry to `wiki/Infra Wiki Log.md`.
10. Run `python3 tools/scripts/validate_wiki.py` and report remaining issues.

## Rules

- Do not rewrite files under `raw/`.
- Do not invent citations or unsupported facts.
- Mark uncertain claims as unverified instead of overstating them.
- Preserve user-authored planning notes unless explicitly asked to revise them.
- If ingesting from `raw/_inbox/`, move the raw file to `raw/ai/` or `raw/infra/` after durable synthesis is complete.

## Output

Report the source ingested, wiki pages changed, claims added, index/log updates, raw-file movement, and validation result.
