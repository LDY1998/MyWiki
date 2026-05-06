---
title: Wiki Maintenance Schema
type: schema
status: active
tags:
  - knowledge-base
  - maintenance
  - schema
updated: 2026-04-27
---

# Wiki Maintenance Schema

This file defines the structure and operating conventions for the unified MyWiki Knowledge OS. It exists so future LLM agents can maintain the repo consistently instead of treating it as a pile of Markdown files.

The model is from [[llm-wiki]]: raw sources are immutable, the wiki is the durable synthesized layer, and repo-local skills turn sources, goals, tasks, claims, and learning artifacts into an auditable workflow.

## Runtime Model

The repo is the first runtime:

```text
raw sources
  -> wiki synthesis
  -> claims and lint reports
  -> goals and evidence requirements
  -> daily task notes
  -> flashcards, quizzes, and review schedule
```

Use repo-local skills for judgment-heavy workflows and deterministic scripts for repeatable checks. Do not build a separate app or dashboard until the file workflow is reliable.

## Repository Layout

```text
MyWiki/
  .agents/
    skills/

  AGENTS.md
  README.md

  raw/
    _inbox/
    ai/
    infra/
    assets/

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

  tasks/

  docs/

  claims/
    claims.yaml
    contradiction_reports/

  goals/
    active/
    archived/

  learning/
    flashcards/
    quizzes/
    review_schedule.yaml

  tools/
    scripts/
    reports/

  archive/
```

## Directory Roles

`raw/` contains source material. Agents may read files here, but source notes should remain stable once ingested.

`raw/_inbox/` is for newly captured articles, papers, transcripts, screenshots, and notes waiting to be processed.

`raw/ai/` holds AI and LLM-RL source documents.

`raw/infra/` holds infrastructure, systems, and distributed-training source documents.

`raw/assets/` holds local images, PDFs, screenshots, and other attachments.

`wiki/` contains LLM-maintained synthesis. This is the main knowledge layer.

`wiki/index.md` is the content-oriented catalog. It should list important pages by category with one-line summaries.

`wiki/Infra Wiki Log.md` is append-only. It records ingests, questions, lint passes, structural changes, and major synthesis updates.

`wiki/maps/` contains map-of-content pages: broad navigational overviews that connect many lower-level pages.

`wiki/concepts/` contains durable technical concepts.

`wiki/systems/` contains concrete frameworks, libraries, platforms, and products such as Ray, TorchTitan, verl, Isaac Lab, and vLLM.

`wiki/papers/` contains paper-specific synthesis pages.

`wiki/sources/` contains non-paper source summaries such as blog posts, docs, books, tutorials, and talks.

`wiki/questions/` contains durable answers to useful questions asked during exploration.

`wiki/comparisons/` contains structured comparisons between techniques, systems, or design choices.

`wiki/projects/` contains project specs, portfolio artifacts, and implementation writeups.

`plans/` contains human-facing learning plans, roadmaps, and phase plans.

`tasks/` contains daily operational task notes. Use these notes to record what was read, ingested, synthesized, reorganized, or otherwise changed during a work session. These notes preserve process history; durable technical synthesis still belongs in `wiki/`.

`docs/` contains development specs, implementation plans, and process artifacts that support repo tooling or agent workflows but are not durable knowledge synthesis.

`claims/` contains atomic source-grounded claims and claim lint reports.

`goals/` contains first-principles goals, capability-based subgoals, and evidence requirements.

`learning/` contains flashcards, quizzes, and spaced-review scheduling files generated from grounded wiki pages and recent reading.

`tools/` contains helper scripts, generated reports, and local automation.

`archive/` contains inactive or superseded material that should remain available but no longer belongs in the active wiki.

`.agents/skills/` contains repo-local Codex skills for invokable Knowledge OS workflows.

## Workflow Skills

Use narrow skills instead of one broad Knowledge OS mega-skill.

| Skill | Use When | Primary Outputs |
|---|---|---|
| `knowledge-ingest` | Ingesting one raw source into durable synthesis | source/paper page, concept updates, optional claims, index/log updates |
| `wiki-lint` | Auditing structural quality, missing sources, stale claims, duplicate concepts, or broken links | dated report under `claims/contradiction_reports/` |
| `goal-decompose` | Turning a user goal into first-principles goals, subgoals, and evidence requirements | goal file under `goals/active/` |
| `daily-task-note-generator` | Creating or refreshing `tasks/YYYY-MM-DD.md` | daily note and recursive task archives |
| `learning-review` | Creating flashcards, quizzes, review schedules, or retrieval-practice tasks | artifacts under `learning/` |

Standard skill pattern:

1. Read [[index|Wiki Index]] and this schema.
2. Read only the relevant repo files.
3. Make small, auditable edits.
4. Update cross-links and index entries when pages are added, moved, renamed, or materially changed.
5. Append to [[Infra Wiki Log]] for ingests, reorgs, lint passes, workflow scaffolding, and major synthesis.
6. Run `python3 tools/scripts/validate_wiki.py` and report remaining issues.

## Deterministic Scripts

Scripts enforce invariants that should not depend on LLM judgment.

| Script | Responsibility |
|---|---|
| `tools/scripts/validate_wiki.py` | frontmatter checks, wikilink resolution, index coverage, raw source coverage, loose root-file warnings |

Future deterministic scripts may include `update_index.py`, `task_score.py`, and `review_scheduler.py` when the corresponding workflows become repetitive enough to automate.

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
- `writing` for durable essays or writeups.
- `goal` for goal pages when stored in `wiki/`.
- `task` for durable task pages when stored in `wiki/`.
- `decision` for decision records.
- `review` for durable review summaries.

Non-wiki README files may use operational types such as `task-index`, `claim-index`, `goal-index`, and `learning-index`.

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

Use the `knowledge-ingest` skill when processing a new source:

1. Put the raw file in `raw/_inbox/`.
2. Read the source and identify the durable knowledge it contributes.
3. Create or update a source summary in `wiki/sources/` or `wiki/papers/`.
4. Update relevant durable pages in `wiki/concepts/`, `wiki/systems/`, `wiki/comparisons/`, `wiki/maps/`, or `wiki/questions/`.
5. Update [[index|Wiki Index]].
6. Append an entry to [[Infra Wiki Log]].
7. Move the raw source from `raw/_inbox/` into `raw/ai/` or `raw/infra/`.

A source should rarely produce only one page. If it changes understanding of a durable concept, update that concept page too.

## Query Workflow

When answering a question against the wiki:

1. Read [[index|Wiki Index]].
2. Search the wiki for relevant pages.
3. Read the most relevant pages and their sources if needed.
4. Answer with internal wikilinks for supporting pages.
5. If the answer is durable, file it under `wiki/questions/`, `wiki/comparisons/`, or another appropriate directory.
6. Update the index and log when filing the answer.

## Lint Workflow

Use the `wiki-lint` skill to periodically check for:

- Orphan pages with no inbound links.
- Important terms mentioned repeatedly but missing their own page.
- Contradictions between concept pages.
- Stale claims superseded by newer sources.
- Missing source links.
- Index entries that point to missing or renamed pages.
- Raw sources that have not been ingested.

Write reports under `claims/contradiction_reports/lint-YYYY-MM-DD.md` before applying major fixes.

## Goal Workflow

Use the `goal-decompose` skill when turning a user goal into durable structure.

1. Identify desired outcome, assumptions, truths, constraints, bottlenecks, and leverage points.
2. Create capability-based subgoals, not topic-only subgoals.
3. Define evidence required for each subgoal.
4. Save active goals under `goals/active/`.
5. Link generated tasks back to the relevant goal or subgoal.

## Learning Workflow

Use the `learning-review` skill when turning recent reading into review artifacts.

1. Read recent daily notes and touched wiki pages.
2. Create flashcards under `learning/flashcards/`.
3. Create quizzes under `learning/quizzes/` when misconception checks are useful.
4. Update `learning/review_schedule.yaml`.
5. Every review artifact must cite a wiki page or raw source.

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
- [[index|Wiki Index]]

Notes:
- Created durable directory structure for raw sources, wiki synthesis, plans, tools, and archive.
```

## Daily Task Notes

Use `tasks/YYYY-MM-DD.md` for daily work notes. Use `tasks/templates/Daily Task Note.md` as the insertable Obsidian template when creating a new daily note. Each note should record:

- `Todos` - a short checkbox list seeded when the note is created.
- `Read` - exact source files, wiki pages, plans, or raw materials reviewed.
- `Done` - concrete wiki maintenance, ingest, synthesis, reorg, or planning work completed.
- `Follow-ups` - next actions discovered during the session.

Task notes are intentionally lightweight and may link to durable wiki pages with Obsidian wikilinks. Leave `Read` and `Done` empty on creation so they can be filled manually during the day. Continue to update [[Infra Wiki Log]] for ingests, structural changes, lint passes, and major synthesis.

When reading is tied to a goal or test, trace it explicitly:

```markdown
- `raw/infra/NCCL User Guide.md`
  - Goal: [[Distributed Training Foundations]]
  - Evidence: update [[NCCL Collectives]]
  - Test: answer a retrieval question or create a quiz under `learning/quizzes/`
```
