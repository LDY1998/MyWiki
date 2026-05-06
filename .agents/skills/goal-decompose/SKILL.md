---
name: goal-decompose
description: Use when converting a user's goal into a first-principles goal file, capability-based subgoals, evidence requirements, or starter tasks in this MyWiki repo.
---

# Goal Decompose

Turn vague ambition into an auditable goal tree with evidence.

## Inputs

- User's goal statement.
- Existing `goals/active/` files.
- Relevant `plans/`, `tasks/`, and wiki pages when they clarify context.

## Workflow

1. Clarify the desired outcome only if the goal is ambiguous enough that assumptions would be risky.
2. Identify assumptions, fundamental truths, constraints, bottlenecks, and leverage points.
3. Create capability-based subgoals, not topic-only subgoals.
4. Define evidence required for each subgoal.
5. Generate a realistic 7-day starter plan when requested or clearly useful.
6. Save the goal under `goals/active/<slug>.md`.
7. Add starter tasks to `tasks/` only if the user asks for execution planning; otherwise list them in the goal file.
8. Link supporting wiki pages and plans with Obsidian wikilinks.
9. Append an entry to `wiki/Infra Wiki Log.md`.

## Rules

- Prefer measurable artifacts: notes, implementations, quizzes, claim updates, diagrams, or decision records.
- Avoid copying course structures unless they follow from first principles.
- Every task should name a goal, subgoal, and evidence artifact.
- Keep goals readable Markdown; use YAML frontmatter for metadata.

## Output

Report the goal file created or updated, subgoals, evidence requirements, starter tasks, and log update.
