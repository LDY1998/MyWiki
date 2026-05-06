---
title: Wiki Index
type: map
status: active
tags:
  - index
  - knowledge-base
updated: 2026-04-27
---

# Wiki Index

Unified Knowledge OS for LLM RL, post-training, and AI infrastructure.

## Maintenance

- [[Wiki Maintenance Schema]] - repo structure, page types, ingestion workflow, and link conventions.
- [[Infra Wiki Log]] - append-only record of ingests, reorgs, lint passes, and major synthesis work.
- [[llm-wiki]] - source idea for the persistent LLM-maintained wiki pattern.
- [Daily Task Notes](../tasks/README.md) - operational notes for what was read or done each day.
- [Claims](../claims/README.md) - atomic claim registry and lint-report area.
- [Goals](../goals/README.md) - first-principles goals and evidence requirements.
- [Learning](../learning/README.md) - flashcards, quizzes, and review schedule.

## Workflow Skills

- `knowledge-ingest` - ingest one raw source into durable synthesis.
- `wiki-lint` - audit wiki quality and write lint reports.
- `goal-decompose` - create goals, subgoals, and evidence requirements.
- `daily-task-note-generator` - create or refresh daily task notes.
- `learning-review` - generate review artifacts from recent reading.

Run `python3 tools/scripts/validate_wiki.py` after structural maintenance.

## Maps

- [[Physical AI Infra]] - learning map for distributed systems, training, and world-model infrastructure.

## Projects

- [[Knowledge OS Agent Runtime Spec]] - repo-native design spec for claims, goals, tasks, and learning review.

## Plans

- [[Physical-ai-infra-learning-plan|Physical AI Infra Learning Plan]] - 8-phase roadmap weighted toward inference and policy serving.

## Source Areas

- `raw/ai/` - AI and LLM-RL source notes, chats, and reference material.
- `raw/infra/` - distributed training, systems, and infrastructure source notes.
- `raw/_inbox/` - new material waiting for ingestion.
- `raw/assets/` - attachments and local assets.

## Repository Areas

- `wiki/` - durable knowledge layer.
- `plans/` - learning plans and roadmaps.
- `tasks/` - daily notes and archived task batches.
- `docs/` - development specs, implementation plans, and process artifacts.
- `claims/` - atomic claim registry and contradiction/lint reports.
- `goals/` - active and archived first-principles goals.
- `learning/` - flashcards, quizzes, and review schedule.
- `tools/` - helper scripts and generated reports.
- `archive/` - inactive or superseded material.
