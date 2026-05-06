# MyWiki — Index

Unified personal knowledge base for LLM RL, post-training, and AI infrastructure.

## Start Here

- [[index|Wiki Index]] — main catalog for concepts, papers, plans, maps, and operational material.
- [[Wiki Maintenance Schema]] — repository structure and maintenance rules.
- [[Infra Wiki Log]] — append-only record of ingests, reorgs, lint passes, and major synthesis work.

## Repository Areas

- `wiki/` — durable synthesized knowledge.
- `raw/ai/` — AI and LLM-RL source material.
- `raw/infra/` — infrastructure and systems source material.
- `raw/_inbox/` — newly captured material waiting for ingestion.
- `tasks/` — daily operational notes and archives.
- `plans/` — learning plans and roadmaps.
- `docs/` — development specs, implementation plans, and process artifacts.
- `claims/` — atomic claims and lint reports.
- `goals/` — active and archived goals.
- `learning/` — flashcards, quizzes, and review scheduling.
- `tools/` — helper scripts and generated reports.
- `.agents/skills/` — repo-local Codex skills, including the daily task note generator.

## Pattern

See `raw/ai/LLM Wiki.md` for the original system description. In short:
- `raw/` stores source material and captured references.
- `wiki/` stores the durable, interlinked synthesis layer.
- `tasks/` records operational work without duplicating durable knowledge.
- `claims/`, `goals/`, and `learning/` add the lightweight Knowledge OS expansion layer.
- `.agents/skills/` contains invokable workflow skills for ingestion, linting, goals, daily notes, and learning review.
