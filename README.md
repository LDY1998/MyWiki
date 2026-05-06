# MyWiki

Repo-native Knowledge OS for LLM RL, post-training, and AI infrastructure.

Start with [wiki/index.md](wiki/index.md). Maintenance rules live in [wiki/Wiki Maintenance Schema.md](wiki/Wiki%20Maintenance%20Schema.md), with agent-facing instructions in [AGENTS.md](AGENTS.md).

Core areas:

- `raw/` stores source material.
- `wiki/` stores durable synthesis.
- `tasks/` stores daily operational notes.
- `docs/` stores development specs and implementation plans.
- `claims/` stores atomic claim tracking.
- `goals/` stores first-principles goals.
- `learning/` stores review artifacts.

Workflow skills live under `.agents/skills/`:

- `knowledge-ingest`
- `wiki-lint`
- `goal-decompose`
- `daily-task-note-generator`
- `learning-review`

## Validation

Run the read-only wiki validator before and after structural maintenance:

```bash
python3 tools/scripts/validate_wiki.py
```

Use `--errors-only` when you want to suppress advisory warnings:

```bash
python3 tools/scripts/validate_wiki.py --errors-only
```
