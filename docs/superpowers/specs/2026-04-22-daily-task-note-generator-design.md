---
title: Daily Task Note Generator Skill Design
status: draft
updated: 2026-04-22
---

# Daily Task Note Generator Skill Design

## Goal

Create a repo-local Codex skill that generates or refreshes `tasks/YYYY-MM-DD.md` for the current day.

The skill should:

- Obtain today's date from system time.
- Read up to the 5 most recent prior daily task notes in `tasks/`.
- Use `tasks/templates/Daily Task Note.md` as the base structure.
- Write only the `## Todos` section.
- Preserve all non-`Todos` sections when today's file already exists.

This skill is intended to become the reusable core for a future scheduled daily job, but scheduling is explicitly out of scope for this design.

## Scope

In scope:

- Repo-local skill layout and instructions.
- Deterministic helper script to parse recent notes and write today's note.
- Carry-forward and new-task generation rules.
- Safe update behavior when today's note already exists.

Out of scope:

- macOS `launchd`, cron, or any other recurring scheduler.
- Changes to `wiki/`, `wiki/index.md`, or `wiki/Infra Wiki Log.md`.
- Editing `Read`, `Done`, or `Follow-ups`.
- Rewriting older task notes.

## Inputs

Primary inputs:

- Current local date in `YYYY-MM-DD`.
- `tasks/templates/Daily Task Note.md`.
- Up to 5 most recent prior notes matching `tasks/YYYY-MM-DD.md`.

Secondary inputs used only when new tasks need to be generated:

- `Follow-ups` sections from the recent task notes.
- Relevant files under `plans/` when they materially clarify next steps already implied by recent notes.

## Decision Rules

The skill should inspect the most recent prior daily note before today.

1. Parse the `## Todos` section.
2. Count incomplete checkbox items (`- [ ] ...`).
3. If the number of incomplete items is fewer than 2:
   - Carry those incomplete items forward.
   - Generate enough new todo items to make exactly 5 total todos.
4. If the number of incomplete items is 2 or more:
   - Carry forward all incomplete items.
   - Do not generate new todo items.

The decision to generate new tasks is based only on the count of incomplete items in the most recent prior note.

## New Task Generation Rules

When generation is allowed, the skill should synthesize new todos from the recent work trajectory rather than inventing unrelated work.

Priority order:

1. Unfinished momentum visible across recent `Read`, `Done`, and `Follow-ups` sections.
2. Explicit next steps suggested by the most recent notes.
3. Relevant active plans under `plans/` that clearly align with the current learning path or implementation path.

Constraints:

- Total todo count must be exactly 5 when generation is allowed.
- Avoid duplicating carried-forward items.
- Avoid repeating semantically identical tasks with different wording.
- Prefer concise, operational tasks that fit a single day.
- Preserve Obsidian wikilinks for internal references where natural.
- Do not create broad speculative projects unsupported by recent notes or plans.

## File Update Rules

### If today's note does not exist

Create `tasks/YYYY-MM-DD.md` from the template and populate only `## Todos`.

Expected shape:

- Frontmatter copied from the template with today's date/title.
- Heading copied from the template with today's date/title.
- `## Todos` populated by the skill.
- `## Read`, `## Done`, and `## Follow-ups` left empty.

### If today's note already exists

Update only the `## Todos` section.

Preserve without modification:

- Frontmatter
- Title
- `## Read`
- `## Done`
- `## Follow-ups`
- Any other user-written content outside the `## Todos` section

## Idempotency

Running the skill multiple times on the same day should not duplicate tasks.

Expected behavior:

- Re-carried items should appear once.
- Newly generated items should be re-derived from the same recent context and replace the prior generated todo block, not append another copy.
- If the file already contains the same normalized todo list, the script may leave the file unchanged.

## Parsing Expectations

The helper script should handle the current note format used in this repo:

- Daily notes are named `tasks/YYYY-MM-DD.md`.
- Todo items use Markdown checkboxes.
- Section headers are Markdown `##` headings.

The parser should be conservative:

- Read only files matching the daily-note filename pattern.
- Ignore `tasks/README.md` and template files.
- Treat missing `## Todos` as zero incomplete items.
- Treat malformed checkbox lines as plain text and exclude them from completion counts unless they clearly match `- [ ]` or `- [x]`.

## Recommended Implementation Shape

Repo-local files:

- `.agents/skills/daily-task-note-generator/SKILL.md`
- `.agents/skills/daily-task-note-generator/agents/openai.yaml`
- `.agents/skills/daily-task-note-generator/scripts/generate_daily_task_note.py`

Responsibilities:

- `SKILL.md` explains when to use the skill and the exact workflow.
- `generate_daily_task_note.py` performs file discovery, parsing, carry-forward logic, optional plan reading, todo synthesis, and file writing.
- `agents/openai.yaml` provides UI metadata for the skill.

## Example Outcomes

### Case A: 0 incomplete items yesterday

- Carry forward nothing.
- Generate 5 new todo items from recent notes and plans.

### Case B: 1 incomplete item yesterday

- Carry forward that 1 item.
- Generate 4 new todo items.

### Case C: 2 incomplete items yesterday

- Carry forward both items.
- Generate 0 new items.

### Case D: 5 incomplete items yesterday

- Carry forward all 5 items.
- Generate 0 new items.

## Risks And Guardrails

Risk: the model generates generic todos with weak grounding.

Guardrail:

- Restrict generated tasks to themes supported by recent notes and active plans.

Risk: overwriting user-written content in today's note.

Guardrail:

- Replace only the `## Todos` section.

Risk: unstable behavior from free-form markdown parsing.

Guardrail:

- Keep parsing deterministic and limited to the current note conventions.

## Verification Plan

Implementation should be verified with fixtures or real repo files covering:

- no existing note for today
- existing note for today with handwritten non-`Todos` content
- 0 incomplete todos yesterday
- 1 incomplete todo yesterday
- 2 or more incomplete todos yesterday
- fewer than 5 recent prior notes available
- no prior daily notes available

## Open Questions Resolved

- Skill location: repo-local.
- Scheduler: deferred until the skill is reviewed and accepted.
- Generation threshold: fewer than 2 incomplete items in the most recent prior note.
- Daily target count: 5 total todos only when generation is allowed.
- Existing file behavior: update only `## Todos`.
