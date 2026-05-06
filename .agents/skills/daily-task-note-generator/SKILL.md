---
name: daily-task-note-generator
description: Use when creating or refreshing `tasks/YYYY-MM-DD.md` in this repo and maintaining recursive 5-file task archives for older daily notes.
---

# Daily Task Note Generator

Use this skill when the user wants to create or refresh `tasks/YYYY-MM-DD.md` in this repo.

This is a pure Codex skill. Apply the note update and archive cleanup directly unless the user explicitly asks for a helper script later.

## Inputs

- Today's date from system time unless the user specifies a date.
- `tasks/templates/Daily Task Note.md`
- Up to the 5 most recent prior daily notes in `tasks/`
- Relevant `plans/` files only when they materially clarify the next steps
- Existing archive notes under `tasks/archive/`

## Workflow

1. Determine the target date and target file path `tasks/YYYY-MM-DD.md`.
2. Read up to the 5 most recent prior daily notes before the target date.
3. Inspect only the most recent prior note's `## Todos` section to decide carry-forward behavior.
4. If that note has fewer than 2 incomplete items:
   - carry those items forward
   - generate enough new todo items to make exactly 5 total
5. If that note has 2 or more incomplete items:
   - carry forward all incomplete items
   - do not generate new todo items
6. Use `tasks/templates/Daily Task Note.md` when creating a new file.
7. If the target file already exists, update only `## Todos` and leave everything else untouched.
8. After the target note is written or refreshed, run archive cleanup for older daily notes.

## Generation Rules

When generation is allowed, synthesize new tasks from the recent work trajectory instead of inventing unrelated work.

Priority order:

1. recent `Follow-ups`
2. explicit next steps implied by recent `Read` and `Done` sections
3. relevant active items in `plans/`

Constraints:

- read no more than 5 recent daily notes
- total todos must be exactly 5 only when generation is allowed
- keep todos concise and operational
- avoid duplicates and near-duplicates
- preserve Obsidian wikilinks for internal references where natural
- do not rewrite `Read`, `Done`, or `Follow-ups`
- do not rewrite older daily notes

## Archive Cleanup Rules

Daily notes:

- Use the target date as the anchor for backfilled runs.
- Ignore `README.md`, `templates/`, and anything under `tasks/archive/`.
- Look at all dated daily notes strictly earlier than the target date.
- If at least 5 older daily notes exist, summarize the oldest 5 into one archive note and then remove those 5 source files.
- Continue in batches of 5 until fewer than 5 older daily notes remain.

Archive levels:

- Store archive notes under `tasks/archive/`.
- Treat archived daily-note batches as level `L1`.
- When 5 files exist at level `L1`, summarize the oldest 5 into one `L2` archive note and remove those 5 `L1` files.
- Apply the same rule recursively for higher levels.
- Never mix levels in one archive.
- Never summarize a partial batch of fewer than 5 files.

Archive file format:

- Name files as `L<level>-YYYY-MM-DD-to-YYYY-MM-DD.md`.
- Include frontmatter with `title`, `type: task-archive`, `level`, `range_start`, `range_end`, `created`, and `source_files`.
- Include a title like `Task Archive L1: 2026-04-19 to 2026-04-23`.
- Summarize recurring themes, completed work, readings, and still-relevant follow-ups.
- Preserve useful Obsidian wikilinks where natural.
- Compress repeated operational detail instead of copying full notes.

Safety rules:

- Never delete source files before the new archive note has been written successfully.
- If the computed archive filename already exists, stop and fail loudly instead of overwriting it.
- If a malformed source file is missing sections, summarize the content that is present.
- If an archive file at some level is malformed, stop recursive rollup at that level and leave those files untouched.

## Editing Rules

For a new daily note:

- use the template structure
- fill only `## Todos`
- leave other sections empty

For an existing daily note:

- replace only the contents of `## Todos`
- preserve frontmatter, title, and all non-`Todos` content exactly

## Output Expectations

After applying the change, summarize:

- which date was targeted
- whether todos were carry-forward only or carry-forward plus generation
- how many recent notes were read
- whether archive cleanup ran
- which archive files were created, if any
- which source files were removed, if any

## Example Prompts

- `Generate today's daily task note`
- `Refresh today's task note todos based on the last few days`
- `Create the next daily task note from recent progress`
