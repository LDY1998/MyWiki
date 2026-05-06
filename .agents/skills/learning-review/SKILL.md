---
name: learning-review
description: Use when creating flashcards, quizzes, review schedules, or retrieval-practice tasks from recent reading and wiki pages in this MyWiki repo.
---

# Learning Review

Convert recent grounded reading into retrieval practice. Favor understanding checks over rote memorization.

## Inputs

- Recent `tasks/YYYY-MM-DD.md` notes.
- Relevant wiki pages touched recently.
- `learning/review_schedule.yaml`
- Existing flashcards and quizzes.
- Active goals when review should support a goal.

## Workflow

1. Read recent daily notes and identify exact files/pages read.
2. Select concepts worth reviewing based on recency, importance, and weakness signals.
3. Generate flashcards in `learning/flashcards/` with source wiki pages or raw source references.
4. Generate quizzes in `learning/quizzes/` when misconception checks are useful.
5. Update `learning/review_schedule.yaml` using simple review intervals unless a deterministic scheduler exists.
6. Add optional review tasks to today's task note only when the user asks.
7. Append an entry to `wiki/Infra Wiki Log.md`.

## Rules

- Every card or quiz question must link to a source wiki page or raw source.
- Prefer questions that require explanation, comparison, or application.
- Keep answers concise but grounded.
- Do not create review artifacts from unsupported claims.

## Output

Report flashcards, quizzes, schedule updates, source pages used, and any task/log updates.
