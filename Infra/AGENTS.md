# Infra Wiki Agent Guide

This repository is an LLM-maintained knowledge base for AI infrastructure, with emphasis on large-scale RL, robotics, world models, and distributed training.

Before making substantial wiki changes, read [wiki/Wiki Maintenance Schema.md](wiki/Wiki%20Maintenance%20Schema.md). That file is the canonical structure and workflow guide for this vault.

## Operating Rules

- Treat `raw/` as source material. Read it, cite it, and move newly ingested files out of `raw/_inbox/`, but do not rewrite source content during wiki maintenance.
- Treat `wiki/` as the synthesized knowledge layer. Agents may create and update pages there.
- Keep [wiki/index.md](wiki/index.md) current whenever pages are added, moved, renamed, or materially changed.
- Append to [wiki/Infra Wiki Log.md](wiki/Infra%20Wiki%20Log.md) after every ingest, structural reorg, lint pass, or major synthesis.
- Prefer Obsidian wikilinks for internal pages, for example `[[FSDP]]` or `[[Parallelism Strategies]]`.
- Use Markdown links only for external URLs.
- Preserve user-authored planning notes unless the user explicitly asks to revise their content.

## Default Workflow

1. Read [wiki/index.md](wiki/index.md) to orient.
2. Read [wiki/Wiki Maintenance Schema.md](wiki/Wiki%20Maintenance%20Schema.md) for structure and conventions.
3. Read relevant existing wiki pages before creating a new one.
4. For a new source, ingest it into the durable synthesis rather than only summarizing it.
5. Update cross-links, index entries, and the log in the same change.
