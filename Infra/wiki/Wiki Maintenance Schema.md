---
title: Wiki Maintenance Schema
type: schema
status: active
tags:
  - knowledge-base
  - maintenance
  - schema
updated: 2026-04-17
---

# Wiki Maintenance Schema

This file defines the structure and operating conventions for the Infra knowledge base. It exists so future LLM agents can maintain the repo consistently instead of treating it as a pile of Markdown files.

The model is from [[llm-wiki]]: raw sources are immutable, the wiki is the durable synthesized layer, and this schema tells agents how to keep the system coherent.

## Repository Layout

```text
Infra/
  AGENTS.md
  llm-wiki.md
  README.md

  raw/
    _inbox/
    sources/
    assets/
    graphify-out/

  wiki/
    index.md
    Infra Wiki Log.md
    Wiki Maintenance Schema.md
    glossary.md

    maps/
    concepts/
    systems/
    papers/
    sources/
    questions/
    comparisons/
    projects/

  plans/

  tools/
    scripts/
    reports/

  archive/
```

## Directory Roles

`raw/` contains source material. Agents may read files here, but source notes should remain stable once ingested.

`raw/_inbox/` is for newly captured articles, papers, transcripts, screenshots, and notes waiting to be processed.

`raw/sources/` holds ingested source documents. Use this for clipped articles, paper notes, docs, book notes, and tutorials.

`raw/assets/` holds local images, PDFs, screenshots, and other attachments.

`wiki/` contains LLM-maintained synthesis. This is the main knowledge layer.

`wiki/index.md` is the content-oriented catalog. It should list important pages by category with one-line summaries.

`wiki/Infra Wiki Log.md` is append-only. It records ingests, questions, lint passes, structural changes, and major synthesis updates.

`wiki/maps/` contains map-of-content pages: broad navigational overviews that connect many lower-level pages.

`wiki/concepts/` contains durable technical concepts such as [[FSDP]], [[ZeRO]], and [[NCCL Collectives]].

`wiki/systems/` contains concrete frameworks, libraries, platforms, and products such as Ray, TorchTitan, verl, Isaac Lab, and vLLM.

`wiki/papers/` contains paper-specific synthesis pages.

`wiki/sources/` contains non-paper source summaries such as blog posts, docs, books, tutorials, and talks.

`wiki/questions/` contains durable answers to useful questions asked during exploration.

`wiki/comparisons/` contains structured comparisons between techniques, systems, or design choices.

`wiki/projects/` contains project specs, portfolio artifacts, and implementation writeups.

`plans/` contains human-facing learning plans, roadmaps, and phase plans.

`tools/` contains helper scripts, generated reports, and local automation.

`archive/` contains inactive or superseded material that should remain available but no longer belongs in the active wiki.

## Page Types

Use the `type` frontmatter field consistently:

- `schema` for maintenance rules.
- `map` for broad navigation pages.
- `concept` for reusable technical concepts.
- `system` for concrete tools, frameworks, and platforms.
- `paper` for paper-specific synthesis.
- `source` for non-paper source summaries.
- `question` for preserved answers to useful questions.
- `comparison` for side-by-side analysis.
- `project` for project specs and implementation artifacts.
- `plan` for learning plans and roadmaps.
- `glossary` for term lists.

## Page Template

Use this shape for new wiki pages unless a page type clearly needs something different:

```markdown
---
title: Page Title
type: concept
status: active
tags:
  - distributed-training
aliases: []
updated: 2026-04-17
sources:
  - "[[Source Page]]"
---

# Page Title

One-paragraph definition or summary.

## Core Idea

## Mechanism

## When It Matters

## Tradeoffs

## Related

- [[Related Page]]

## Sources

- [[Source Page]]
```

## Ingestion Workflow

When processing a new source:

1. Put the raw file in `raw/_inbox/`.
2. Read the source and identify the durable knowledge it contributes.
3. Create or update a source summary in `wiki/sources/` or `wiki/papers/`.
4. Update relevant durable pages in `wiki/concepts/`, `wiki/systems/`, `wiki/comparisons/`, `wiki/maps/`, or `wiki/questions/`.
5. Update [[index|Infra Wiki - Index]].
6. Append an entry to [[Infra Wiki Log]].
7. Move the raw source from `raw/_inbox/` to `raw/sources/`.

A source should rarely produce only one page. If it changes understanding of a durable concept, update that concept page too.

## Query Workflow

When answering a question against the wiki:

1. Read [[index|Infra Wiki - Index]].
2. Search the wiki for relevant pages.
3. Read the most relevant pages and their sources if needed.
4. Answer with internal wikilinks for supporting pages.
5. If the answer is durable, file it under `wiki/questions/`, `wiki/comparisons/`, or another appropriate directory.
6. Update the index and log when filing the answer.

## Lint Workflow

Periodically check for:

- Orphan pages with no inbound links.
- Important terms mentioned repeatedly but missing their own page.
- Contradictions between concept pages.
- Stale claims superseded by newer sources.
- Missing source links.
- Index entries that point to missing or renamed pages.
- Raw sources that have not been ingested.

## Link Conventions

Use Obsidian wikilinks for internal notes:

```markdown
[[FSDP]]
[[Parallelism Strategies]]
[[phase-0-foundations|Phase 0 Foundations]]
```

Use Markdown links for external URLs:

```markdown
[PyTorch FSDP docs](https://pytorch.org/docs/stable/distributed.fsdp.html)
```

Prefer stable page titles over path-heavy links. Obsidian resolves notes by basename, so moving a note between folders should not require changing normal wikilinks.

## Logging Convention

Use this format in [[Infra Wiki Log]]:

```markdown
## [2026-04-17] reorg | Establish wiki schema

Updated:
- [[Wiki Maintenance Schema]]
- [[index|Infra Wiki - Index]]

Notes:
- Created durable directory structure for raw sources, wiki synthesis, plans, tools, and archive.
```
