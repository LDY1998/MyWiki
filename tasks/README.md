---
title: Daily Task Notes
type: task-index
status: active
tags:
  - tasks
  - maintenance
updated: 2026-04-27
---

# Daily Task Notes

Use this folder for short daily notes that record what was read, ingested, synthesized, reorganized, or otherwise changed in the unified wiki.

Daily notes are operational logs, not durable synthesis pages. Put durable technical knowledge in `wiki/`, and use daily notes to preserve the work trail from reading to goals, evidence, and tests.

## Naming

Create one note per active day:

```text
tasks/YYYY-MM-DD.md
```

## Template

Use [Daily Task Note](templates/Daily%20Task%20Note.md) as the insertable Obsidian template for new daily notes.

When a note is first created, the template adds a short `Todos` checklist. Leave `Read` and `Done` empty until you fill them manually during the day.

## Daily Routine

Create or refresh the current daily note at the start of an active work session:

1. Target `tasks/YYYY-MM-DD.md` using the local date.
2. Read up to the five most recent prior daily notes.
3. Carry forward incomplete todos from the most recent prior note.
4. If fewer than two incomplete todos carry forward, add new todos from recent `Follow-ups`, `Done`, `Read`, and relevant `plans/` files until the list has five items.
5. Keep `Read`, `Done`, and `Follow-ups` untouched when refreshing an existing note.
6. Archive older daily notes in batches of five under `tasks/archive/`.

The repo-local `daily-task-note-generator` skill encodes this routine for Codex agents. Use it when asking an agent to create or refresh a daily task note.

## Notes

- Use Obsidian wikilinks for internal wiki pages, such as [[Wiki Maintenance Schema]].
- Keep task notes concise. Link to durable pages, source summaries, plans, or log entries instead of duplicating them.
- In `Read`, record exact file paths when possible, then attach `Goal`, `Evidence`, and `Test` sub-lines when the reading supports a goal.
- Continue updating [[Infra Wiki Log]] for ingests, structural changes, lint passes, and major synthesis.

Example trace:

```markdown
- `raw/infra/NCCL User Guide.md`
  - Goal: [[Distributed Training Foundations]]
  - Evidence: update [[NCCL Collectives]]
  - Test: create `learning/quizzes/YYYY-MM-DD-nccl.md`
```
