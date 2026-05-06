# Infra Wiki Log

Append-only record of wiki maintenance, ingests, queries, lint passes, and structural changes.

## [2026-04-27] docs | Refresh Knowledge OS schema

Updated:
- [[Wiki Maintenance Schema]]
- [[index|Wiki Index]]
- [Agent Guide](../AGENTS.md)
- [README](../README.md)
- [Daily Task Notes](../tasks/README.md)
- [Claims](../claims/README.md)
- [Goals](../goals/README.md)
- [Learning](../learning/README.md)

Notes:
- Updated the maintenance schema to describe the current Knowledge OS runtime model, repo-local workflow skills, deterministic validation, and read-to-goal/test tracing.
- Aligned the root guide and index docs with `claims/`, `goals/`, `learning/`, and `.agents/skills/`.
- Added concise file-shape guidance for claims, goals, flashcards, and quizzes.

## [2026-04-27] skills | Scaffold Knowledge OS workflow skills

Updated:
- [[Knowledge OS Agent Runtime Spec]]
- [knowledge-ingest](../.agents/skills/knowledge-ingest/SKILL.md)
- [wiki-lint](../.agents/skills/wiki-lint/SKILL.md)
- [goal-decompose](../.agents/skills/goal-decompose/SKILL.md)
- [learning-review](../.agents/skills/learning-review/SKILL.md)

Notes:
- Added narrow repo-local skills for source ingestion, wiki linting, goal decomposition, and learning review.
- Kept the existing daily task note generator as the daily planning workflow.
- Added a workflow packaging section to the Knowledge OS runtime spec.

## [2026-04-27] reorg | Conservative cleanup and Knowledge OS scaffold

Updated:
- [[index|Wiki Index]]
- [[Wiki Maintenance Schema]]
- [[Knowledge OS Agent Runtime Spec]]
- [README](../README.md)
- [Root Index](../index.md)

Notes:
- Moved the Knowledge OS runtime spec into `wiki/projects/`.
- Moved the pasted image into `raw/assets/`.
- Archived the empty Obsidian dashboard canvas under `archive/`.
- Removed the empty undocumented `wiki/logs/` directory.
- Added minimal `claims/`, `goals/`, and `learning/` scaffolds for the Knowledge OS expansion layer.

## [2026-04-27] tooling | Add wiki validation routine

Updated:
- [Validator](../tools/scripts/validate_wiki.py)
- [README](../README.md)
- [Daily Task Notes](../tasks/README.md)

Notes:
- Added a read-only wiki validator for frontmatter, wikilink resolution, index coverage, raw source coverage, and loose root-file warnings.
- Documented the validation command in the repo README.
- Documented the daily task-note routine and linked it to the repo-local generator skill.
- Initial validation reports existing frontmatter and link issues; no wiki content was auto-rewritten.

## [2026-04-24] reorg | Unify AI and Infra into repo-root wiki

Updated:
- [[index|Wiki Index]]
- [[Wiki Maintenance Schema]]
- [Daily Task Notes](../tasks/README.md)

Notes:
- Moved AI and Infra durable pages into a shared repo-root `wiki/` tree.
- Moved source material into `raw/ai/` and `raw/infra/`.
- Moved plans, tasks, tools, docs, tests, and the repo-local daily-task skill to repo-root locations.
- Archived `tasks/2026-04-19.md` through `tasks/2026-04-23.md` into a single level-1 task archive.

## [2026-04-20] reorg | Add daily task template

Updated:
- [[Wiki Maintenance Schema]]
- [Daily Task Notes](../tasks/README.md)
- [Daily Task Note Template](../tasks/templates/Daily%20Task%20Note.md)

Notes:
- Moved the daily note template out of `tasks/README.md` into an Obsidian-insertable template file.
- Added a seeded `Todos` checkbox list for newly created daily task notes.
- Left `Read` and `Done` empty by default for manual daily fill-in.
- Added `tasks/2026-04-20.md` using the new task-note shape.

## [2026-04-19] reorg | Add daily task notes

Updated:
- [[Wiki Maintenance Schema]]
- [[index|Wiki Index]]
- [Daily Task Notes](../tasks/README.md)

Notes:
- Created repo-root `tasks/` for daily operational notes about what was read or done.
- Added `tasks/2026-04-19.md` as the first daily note.
- Documented naming, template, and usage conventions for task notes.

## [2026-04-17] reorg | Establish wiki schema

Updated:
- [[Wiki Maintenance Schema]]
- [[index|Wiki Index]]

Notes:
- Added the repo maintenance schema and agent guide.
- Created standard directories for raw inbox/assets/sources, wiki synthesis categories, plans, tools, and archive.
- Moved top-level learning plans into `plans/`.
- Moved ingested raw Markdown sources into the repo-root raw source area.
