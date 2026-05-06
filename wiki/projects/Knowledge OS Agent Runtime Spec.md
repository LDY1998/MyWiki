---
title: Knowledge OS Agent Runtime Spec
type: project
status: active
tags:
  - knowledge-os
  - agent-runtime
  - repo-workflow
updated: 2026-04-27
---

# Knowledge OS Agent Runtime Spec

This document describes a repo-based prototype for an agentic personal knowledge system. The goal is to let a coding agent such as Claude Code, Codex, OpenCode, or Cursor build and maintain a knowledge base that can ingest sources, compile a wiki, lint claims, create goals, generate tasks, and produce flashcards or quizzes.

The core idea: **do not start by building a full app**. Start with a disciplined repo, Markdown/YAML files, simple scripts, and agent instructions. The repo itself becomes the first runtime.

---

## 1. Product Vision

Build a personal **Knowledge OS** that turns raw reading and notes into:

1. Grounded wiki pages.
2. Cross-referenced concepts.
3. Atomic claims with source traces.
4. Lint reports for duplicate, stale, weak, or contradictory claims.
5. Goal trees based on first-principles decomposition.
6. Subgoals and daily tasks with difficulty and motivation scores.
7. Reading logs, flashcards, quizzes, and spaced review.
8. Progress tracking based on evidence artifacts.

The desired loop:

```text
raw data → compiled knowledge → audited claims → goals/subgoals → daily tasks → review/quiz → progress evidence
```

The system should behave less like a passive note app and more like a small mission-control room for learning, building, and thinking.

---

## 2. Why Start With a Repo Instead of Full Software?

There are three possible levels:

```text
Level 1: Agent-managed folder
Markdown + Git + AGENTS.md / CLAUDE.md + coding agent

Level 2: Agent runtime
Same folder, but with scripts, schemas, validators, task generator, quiz generator

Level 3: Full software
Web UI, database, auth, dashboards, graph visualization, mobile app, sync
```

Start with **Level 1.5 or Level 2**.

A full app is useful later, but premature UI can become a shiny shell over an unproven workflow. First prove the knowledge workflow in files. The repo is the dragon egg.

---

## 3. Recommended Initial Stack

```text
Editor / Agent:
  Claude Code, Codex, Cursor, OpenCode, or similar coding agent

Storage:
  Markdown + YAML frontmatter + Git

Automation:
  Python scripts

Search:
  ripgrep first
  SQLite FTS later if needed
  vector search later if needed

Database:
  Start without DB
  Add SQLite when Markdown/YAML becomes painful

Viewer:
  VS Code / Obsidian first
  Custom web UI later
```

---

## 4. Repo Structure

Start small:

```text
knowledge-os/
  AGENTS.md
  README.md

  raw/
    papers/
    blogs/
    books/
    chats/

  wiki/
    index.md
    concepts/
    projects/
    literature/

  goals/
    active/
    archived/

  tasks/
    today.md
    backlog.md
    timeline.md
    completed.md

  claims/
    claims.yaml
    contradiction_reports/

  learning/
    flashcards/
    quizzes/
    review_schedule.yaml

  logs/
    activity_log.md
    reading_log.yaml

  prompts/
    agents/
      ingest_agent.md
      lint_agent.md
      goal_agent.md
      task_agent.md
      learning_agent.md
    workflows/
      ingest.md
      goal_decompose.md
      daily_plan.md
      quiz_generate.md

  scripts/
    validate.py
    update_index.py
    link_checker.py
    task_score.py
    review_scheduler.py
```

Do not add every directory immediately if you do not need it. Start minimal, then let the structure grow.

Minimal version:

```text
knowledge-os/
  AGENTS.md
  raw/
  wiki/
    index.md
  goals/
    active/
  tasks/
    today.md
    backlog.md
    timeline.md
  learning/
    flashcards/
    quizzes/
  claims/
    claims.yaml
  logs/
    activity_log.md
    reading_log.yaml
  scripts/
    validate.py
    update_index.py
```

---

## 5. Core Agent Roles

These can start as prompt files, not deployed services.

### 5.1 Ingest Agent

Purpose:

```text
raw source → wiki pages → index/log updates → extracted claims
```

Responsibilities:

- Read files under `raw/`.
- Summarize and extract core concepts.
- Create or update wiki pages.
- Add backlinks and related links.
- Extract important claims into `claims/claims.yaml`.
- Update `wiki/index.md`.
- Append to `logs/activity_log.md`.

---

### 5.2 Wiki Lint Agent

Purpose:

```text
wiki pages + claims → lint report → proposed fixes
```

Checks:

- Duplicate concepts.
- Duplicate claims.
- Contradictory claims.
- Unverified claims.
- Weak citations.
- Missing backlinks.
- Orphan pages.
- Stale pages.
- Pages with no source.
- Terms mentioned often but missing a dedicated page.

Output:

```text
claims/contradiction_reports/lint-YYYY-MM-DD.md
```

---

### 5.3 Goal Agent

Purpose:

```text
user goal → first-principles canvas → goal tree → subgoals → evidence → starter plan
```

Responsibilities:

- Convert vague ambitions into structured goals.
- Use first-principles decomposition.
- Identify assumptions, truths, constraints, bottlenecks, and leverage points.
- Create capability-based subgoals.
- Define evidence required for each subgoal.
- Generate a 7-day starter plan.
- Save goal files under `goals/active/`.

---

### 5.4 Task Agent

Purpose:

```text
active goals + recent activity + unfinished work → daily plan
```

Responsibilities:

- Read active goals.
- Read recent reading logs.
- Read unfinished tasks.
- Generate `tasks/today.md`.
- Score tasks by difficulty, leverage, motivation, and confidence.
- Update timeline and backlog.

A good daily mix:

```text
1 high-leverage hard task
1 medium learning task
1 small cleanup task
1 review/quiz task
```

---

### 5.5 Learning Agent

Purpose:

```text
recent reading + weak concepts → flashcards + quizzes + review schedule
```

Responsibilities:

- Track recently read concepts.
- Generate flashcards from grounded wiki pages.
- Generate quizzes from recent reading.
- Schedule reviews.
- Log mistakes and weak areas.
- Recommend review tasks.

---

## 5.6 Workflow Packaging

Package workflows as narrow repo-local skills, not one broad Knowledge OS mega-skill.

| Skill | Invoke When | Inputs | Outputs | Deterministic Scripts |
|---|---|---|---|---|
| `knowledge-ingest` | Ingesting a raw source | one `raw/...` file, index, schema, related pages | source/paper summary, concept updates, optional claims, index/log updates | `tools/scripts/validate_wiki.py` |
| `wiki-lint` | Auditing wiki quality | `wiki/`, `claims/`, `raw/` | dated lint report under `claims/contradiction_reports/` | `tools/scripts/validate_wiki.py` |
| `goal-decompose` | Creating a goal plan | user goal, active goals, relevant plans/wiki pages | goal file, subgoals, evidence requirements, starter tasks | optional task scoring later |
| `daily-task-note-generator` | Creating or refreshing daily notes | recent task notes, template, plans | `tasks/YYYY-MM-DD.md`, archive rollups | archive cleanup routine |
| `learning-review` | Creating review artifacts | recent reading, touched wiki pages, review schedule | flashcards, quizzes, schedule updates | review scheduler later |

The standard execution pattern:

```text
user invokes skill
  → skill reads current repo context
  → skill performs semantic work
  → deterministic script validates invariants
  → skill updates index/log
```

Skills own judgment-heavy work. Scripts own invariants such as validation, link checking, sorting, and schedule calculations.

---

## 6. First-Principles Goal System

The goal system should not merely copy existing course structures or productivity apps. It should reason from fundamentals.

The first-principles process:

```text
1. Desired outcome
   What capability or result does the user want?

2. Current assumptions
   What is the user assuming must be true?

3. Fundamental truths
   What is definitely true about this domain?

4. Constraints
   Time, skill, money, tools, energy, environment.

5. Bottlenecks
   What currently prevents progress?

6. Leverage points
   What small actions unlock many future actions?

7. Subgoals
   What capabilities must be built?

8. Daily actions
   What can be done today that creates evidence of progress?
```

Example:

```text
Goal:
Build an agentic knowledge base that helps me learn, plan, and execute.

Assumptions:
- I need a beautiful UI first.
- I need a full graph database.
- I need multi-agent orchestration immediately.

First-principles truths:
- Knowledge needs source grounding.
- Goals need measurable progress.
- Motivation improves when progress is visible.
- Agents need bounded permissions.
- Tasks should be generated from actual knowledge gaps.

Bottleneck:
Unclear system architecture and MVP scope.

Leverage point:
Start with Markdown + YAML + Git + one daily planning workflow.

Subgoals:
1. Build source ingestion.
2. Build wiki generation.
3. Build claim linting.
4. Build goal graph.
5. Build task generator.
6. Build learning review system.
```

---

## 7. Capability-Based Subgoals

Subgoals should be capability-based, not merely topic-based.

Bad:

```text
Read about reinforcement learning.
```

Better:

```text
Be able to explain how rollout workers, replay buffers, learners, evaluators, and checkpointing interact in a large-scale RL system.
```

Even better:

```text
Implement a toy actor-learner loop with 4 parallel rollout workers and write a design note explaining bottlenecks.
```

Subgoal types:

| Type | Meaning | Example |
|---|---|---|
| Concept | Understand an idea | Explain reduce-scatter |
| Synthesis | Connect ideas | Compare ZeRO-3 and tensor parallelism |
| Implementation | Build something | Implement toy distributed trainer |
| Writing | Produce artifact | Write architecture note |
| Review | Retain knowledge | Flashcards on NCCL |
| Decision | Choose direction | Pick MVP stack |
| Project | Deliver milestone | Build goal dashboard |

---

## 8. Scoring System

Tasks should be scored across multiple dimensions.

```yaml
task_id: task_2026_04_27_001
title: "Write a first-principles decomposition of the AI knowledge-base runtime"
goal_id: goal_agentic_kb_001
subgoal_id: subgoal_goal_system
type: writing
difficulty: 3
leverage: 5
motivation: 4
confidence: 4
estimated_minutes: 45
why_now: "This defines the architecture before UI work begins."
evidence_required:
  - "A Markdown design note exists."
  - "At least 5 subgoals are created."
status: todo
```

Score meanings:

| Score | Meaning |
|---|---|
| Difficulty | How hard is this task? |
| Leverage | How much future progress does it unlock? |
| Motivation | How emotionally rewarding is it? |
| Confidence | How likely is the user to finish it today? |

Difficulty guide:

| Score | Meaning | Example |
|---:|---|---|
| 1 | Quick cleanup | Add missing backlinks |
| 2 | Small learning task | Summarize one section |
| 3 | Medium synthesis | Compare ZeRO-2 and ZeRO-3 |
| 4 | Hard technical write-up | Explain tensor parallelism with diagrams |
| 5 | Research-level task | Survey RL/world-model infra papers and extract architecture patterns |

---

## 9. Evidence-Based Progress Tracking

Progress should be tracked as evidence, not vibes.

Every task should produce or point to an artifact:

- Markdown note.
- Wiki page.
- Code file.
- Diagram.
- Flashcard set.
- Quiz result.
- Summary.
- Decision record.
- Experiment result.

Example progress ledger:

```markdown
# Progress: Agentic Knowledge Base

## Evidence Collected

- 2026-04-27: Created first-principles goal decomposition.
- 2026-04-28: Built Markdown ingest prototype.
- 2026-04-29: Generated first lint report.
- 2026-04-30: Created 20 flashcards from reading log.

## Current Bottleneck

The system lacks a reliable claim-extraction format.

## Next Leverage Task

Design claim schema and contradiction detection pipeline.
```

---

## 10. File Schemas

### 10.1 Wiki Page Frontmatter

```yaml
title: "Attention Mechanism"
type: concept
status: draft
created: 2026-04-27
updated: 2026-04-27
sources:
  - raw/blogs/illustrated-transformer.md
related:
  - "QKV Attention"
  - "Multi-Head Attention"
  - "Transformer"
```

Allowed `type` values:

```text
concept | project | literature | note | goal | task | decision | review
```

Allowed `status` values:

```text
draft | active | stable | archived | stale
```

---

### 10.2 Claim Schema

```yaml
- claim_id: claim_2026_04_27_001
  text: "Reduce-scatter followed by all-gather can implement all-reduce."
  source_pages:
    - wiki/concepts/allreduce.md
  source_raw:
    - raw/blogs/distributed-finetuning.md
  confidence: high
  status: active
  last_verified: 2026-04-27
  related_claims:
    - claim_2026_04_27_002
```

Allowed `confidence` values:

```text
low | medium | high
```

Allowed `status` values:

```text
active | unverified | contradicted | stale | deprecated
```

---

### 10.3 Goal Schema

```yaml
goal_id: goal_agentic_kb_001
title: "Build an Agentic Knowledge Base Runtime"
created_at: 2026-04-27
status: active
horizon: "3 months"
motivation:
  why: "I want a system that turns reading into goals, tasks, and retained knowledge."
  identity: "Systems engineer building AI-native learning infrastructure."
first_principles:
  - "Knowledge must be grounded in source material."
  - "Memory requires repeated retrieval and use."
  - "Motivation requires visible progress."
  - "Agents need bounded roles and auditable outputs."
success_metrics:
  - "Can ingest one source into raw/ and compile useful wiki pages."
  - "Can generate a daily plan from active goals and recent reading."
  - "Can generate flashcards from recently read content."
subgoals:
  - subgoal_ingestion
  - subgoal_linting
  - subgoal_goal_system
  - subgoal_task_generation
  - subgoal_learning_review
```

---

### 10.4 Subgoal Schema

```yaml
subgoal_id: subgoal_linting
goal_id: goal_agentic_kb_001
title: "Build claim linting and contradiction detection"
type: implementation
status: active
evidence_required:
  - "claims/claims.yaml exists and validates."
  - "At least one lint report is generated."
  - "The lint report identifies missing sources or duplicate content."
```

---

### 10.5 Task Schema

```yaml
task_id: task_2026_04_27_001
title: "Create validate.py for wiki and task schemas"
goal_id: goal_agentic_kb_001
subgoal_id: subgoal_linting
type: coding
difficulty: 3
leverage: 5
motivation: 3
confidence: 4
estimated_minutes: 60
status: todo
due_date: 2026-04-27
evidence_required:
  - "scripts/validate.py exists."
  - "The script checks wiki frontmatter."
  - "The script checks task required fields."
source_pages:
  - wiki/projects/knowledge-os.md
```

Allowed `type` values:

```text
reading | writing | coding | review | synthesis | cleanup | decision | planning
```

Allowed `status` values:

```text
todo | doing | done | blocked | skipped
```

---

### 10.6 Reading Log Schema

```yaml
- event_id: read_2026_04_27_001
  type: read
  source: raw/blogs/karpathy-llm-wiki.md
  wiki_pages_touched:
    - wiki/projects/knowledge-os.md
    - wiki/concepts/llm-wiki.md
  duration_minutes: 35
  depth: skimmed
  concepts:
    - LLM Wiki
    - knowledge compiler
    - linting
    - index.md
    - log.md
  timestamp: 2026-04-27T15:30:00
```

Allowed `depth` values:

```text
skimmed | read | deeply_read | reviewed | practiced
```

---

### 10.7 Flashcard Schema

Markdown format:

```markdown
# Flashcards: Transformer Attention

## Card 1
Q: In attention, if QK gives relevance scores, what is V used for?
A: V carries the content that gets mixed according to the attention weights.

Source: [[Attention Mechanism]]
Difficulty: 2
Review due: 2026-04-29
```

YAML option:

```yaml
- card_id: card_2026_04_27_001
  question: "In attention, if QK gives relevance scores, what is V used for?"
  answer: "V carries the content that gets mixed according to the attention weights."
  source_pages:
    - wiki/concepts/attention.md
  difficulty: 2
  review_due: 2026-04-29
  status: active
```

---

### 10.8 Quiz Schema

```markdown
# Quiz: NCCL AllReduce

## Q1
Why can all-reduce be implemented as reduce-scatter followed by all-gather?

A. Because reduce-scatter first computes the full result on rank 0  
B. Because each rank reduces only one shard, then all-gather distributes all shards  
C. Because all-gather performs the reduction  
D. Because reduce-scatter is equivalent to broadcast  

Correct: B
Explanation: Reduce-scatter partitions the final reduced tensor across ranks. All-gather then shares those reduced shards so every rank has the full result.
Source: [[AllReduce via ReduceScatter]]
```

---

## 11. AGENTS.md Template

Paste this into `AGENTS.md`.

````markdown
# Knowledge OS Agent

You are maintaining a personal knowledge base and goal-oriented learning system.

## Prime directive

Transform raw information into grounded knowledge, then into goals, tasks, reviews, and learning artifacts.

## Repository rules

- Never modify files under `raw/` unless explicitly asked.
- All wiki pages must include YAML frontmatter.
- Every wiki page must be linked from `wiki/index.md`.
- Every generated task must link to a goal or subgoal.
- Every flashcard must link to a source wiki page or raw source.
- Every major change must append an entry to `logs/activity_log.md`.
- Prefer small edits over large rewrites.
- Do not invent citations or sources.
- If a claim cannot be verified, mark it as `unverified`.
- When making factual corrections, create a lint report before editing.
- Keep generated content auditable.

## Wiki page frontmatter

```yaml
title:
type: concept | project | literature | note | goal | task | decision | review
status: draft | active | stable | archived | stale
created:
updated:
sources:
related:
```

## Task format

```yaml
task_id:
title:
goal_id:
subgoal_id:
type: reading | writing | coding | review | synthesis | cleanup | decision | planning
difficulty: 1-5
leverage: 1-5
motivation: 1-5
confidence: 1-5
estimated_minutes:
status: todo | doing | done | blocked | skipped
evidence_required:
```

## Workflows

### Ingest workflow

Input: one raw file.

Steps:
1. Read the source file.
2. Extract key claims and concepts.
3. Create or update wiki pages.
4. Add links to related pages.
5. Extract important atomic claims into `claims/claims.yaml` when appropriate.
6. Update `wiki/index.md`.
7. Append to `logs/activity_log.md`.
8. Add a reading event to `logs/reading_log.yaml` if the source was read or processed.

### Goal workflow

Input: a user goal.

Steps:
1. Convert the user goal into a first-principles canvas.
2. Identify assumptions, fundamental truths, constraints, bottlenecks, and leverage points.
3. Create capability-based subgoals.
4. Define evidence for each subgoal.
5. Generate starter tasks.
6. Save under `goals/active/`.
7. Append to `logs/activity_log.md`.

### Daily planning workflow

Input:
- active goals
- recent reading log
- unfinished tasks
- review schedule

Steps:
1. Read active goals.
2. Read recent reading log.
3. Read unfinished tasks.
4. Select a balanced daily plan:
   - one high-leverage task
   - one learning task
   - one small cleanup task
   - one review task
5. Score all tasks by difficulty, leverage, motivation, and confidence.
6. Update `tasks/today.md`.
7. Update `tasks/timeline.md`.
8. Append to `logs/activity_log.md`.

### Lint workflow

Check:
- duplicate content
- missing backlinks
- weak or unverified claims
- contradictions
- stale pages
- orphan pages
- broken internal links
- missing frontmatter

Output:
- write reports under `claims/contradiction_reports/`
- propose fixes before applying risky edits

### Learning workflow

Input:
- recent reading log
- wiki pages touched recently
- weak concepts
- active goals

Steps:
1. Select concepts worth reviewing.
2. Generate flashcards with source references.
3. Generate quizzes when useful.
4. Update `learning/review_schedule.yaml`.
5. Add review tasks to backlog or today.
6. Append to `logs/activity_log.md`.
````

---

## 12. Prompt Templates

### 12.1 `prompts/agents/goal_agent.md`

```markdown
# Goal Agent Prompt

You are a goal-planning agent inside a personal knowledge base.

Given a user's goal, decompose it using first-principles reasoning.

Do not reason mainly by analogy to existing apps or courses.
Instead:
1. Identify the desired capability or outcome.
2. List assumptions the user may be making.
3. Break the goal into fundamental truths.
4. Identify constraints and bottlenecks.
5. Generate capability-based subgoals.
6. Generate measurable evidence for each subgoal.
7. Generate a 7-day starter plan.
8. Assign difficulty, leverage, confidence, and motivation scores.

Every task must connect to:
- a goal_id
- a subgoal_id
- an evidence artifact
- a source page, if applicable

Output:
- YAML goal file
- readable Markdown summary
- starter task list
```

---

### 12.2 `prompts/agents/ingest_agent.md`

```markdown
# Ingest Agent Prompt

You ingest raw sources into the Knowledge OS.

Rules:
- Never alter raw files.
- Create grounded wiki pages.
- Do not invent facts not supported by the source.
- Use citations or source references to raw files.
- Extract atomic claims only when useful.
- Update `wiki/index.md`.
- Update `logs/activity_log.md`.
- Update `logs/reading_log.yaml`.

For each source:
1. Identify the source type.
2. Extract key concepts.
3. Extract key claims.
4. Create or update relevant wiki pages.
5. Add related links.
6. Mark uncertain or unsupported claims clearly.
```

---

### 12.3 `prompts/agents/lint_agent.md`

```markdown
# Lint Agent Prompt

You audit the Knowledge OS for quality.

Check:
- duplicate claims
- duplicate pages
- weak citations
- unsupported claims
- contradictions
- stale pages
- missing backlinks
- orphan pages
- broken links
- terms that deserve their own page

Output a report with:
1. Summary.
2. High-priority issues.
3. Medium-priority issues.
4. Low-priority cleanup.
5. Suggested fixes.
6. Files affected.

Do not directly rewrite large sections unless explicitly asked.
```

---

### 12.4 `prompts/agents/task_agent.md`

```markdown
# Task Agent Prompt

You generate daily tasks from active goals, recent reading, unfinished work, and review needs.

Rules:
- Every task must link to a goal or subgoal.
- Every task must define evidence required.
- Score each task by difficulty, leverage, motivation, and confidence.
- Prefer a balanced plan:
  - one high-leverage task
  - one learning task
  - one small cleanup task
  - one review task
- Do not overload the day.

Output:
- `tasks/today.md`
- timeline updates if needed
- backlog updates if needed
```

---

### 12.5 `prompts/agents/learning_agent.md`

```markdown
# Learning Agent Prompt

You create learning artifacts from recent reading and weak concepts.

Input:
- reading log
- recently edited wiki pages
- active goals
- previous quiz mistakes

Output:
- flashcards
- quizzes
- review schedule updates
- optional review tasks

Rules:
- Every flashcard must include a source page or raw source.
- Prefer cards that test understanding, not memorization only.
- Prefer quizzes that expose misconceptions.
- Keep explanations concise but grounded.
```

---

## 13. Script Responsibilities

Use scripts for deterministic checks and calculations. Use LLM agents for fuzzy semantic work.

### 13.1 `scripts/validate.py`

Should check:

- Every wiki page has YAML frontmatter.
- Every wiki page has required fields.
- Every task has required fields.
- Every flashcard has a source.
- Every goal has at least one subgoal.
- Every task links to a valid goal/subgoal when possible.
- `wiki/index.md` exists.
- Required folders exist.

Example command:

```bash
python scripts/validate.py
```

---

### 13.2 `scripts/update_index.py`

Should:

- Scan `wiki/`.
- List all pages.
- Group pages by type.
- Update `wiki/index.md`.
- Optionally include backlinks or related links.

---

### 13.3 `scripts/link_checker.py`

Should:

- Scan wiki links such as `[[Page Name]]`.
- Check broken internal links.
- Report orphan pages.
- Report pages not linked from index.

---

### 13.4 `scripts/task_score.py`

Should:

- Validate task scores are in range 1 to 5.
- Sort tasks by priority.
- Optional priority formula:

```text
priority = leverage * confidence + motivation - difficulty
```

This formula is intentionally simple. Tune later.

---

### 13.5 `scripts/review_scheduler.py`

Should:

- Read flashcards.
- Read review schedule.
- Calculate next review date.
- Generate review tasks.

Simple intervals:

```text
first review: 1 day
second review: 3 days
third review: 7 days
fourth review: 14 days
later review: 30 days
```

---

## 14. Example Commands for Coding Agent

### 14.1 Initialize repo

```text
Create the Knowledge OS repo structure described in AGENTS.md.
Add placeholder files for wiki/index.md, logs/activity_log.md, logs/reading_log.yaml, tasks/today.md, tasks/backlog.md, tasks/timeline.md, and claims/claims.yaml.
```

---

### 14.2 Ingest a source

```text
Ingest raw/blogs/karpathy-llm-wiki.md into the wiki following AGENTS.md.
Create useful wiki pages, update index.md, extract important claims, and append activity_log.md.
```

---

### 14.3 Create a goal

```text
Use prompts/agents/goal_agent.md to create a first-principles goal plan for:
"Build an agentic knowledge base that supports goals, tasks, linting, and learning review."

Save it under goals/active/.
Generate a 7-day starter plan.
Update tasks/backlog.md and logs/activity_log.md.
```

---

### 14.4 Generate today’s plan

```text
Using active goals, reading log, backlog, and review schedule, generate tasks/today.md.
Score each task by difficulty, leverage, motivation, and confidence.
Keep the plan balanced and realistic.
```

---

### 14.5 Lint the wiki

```text
Use prompts/agents/lint_agent.md to lint the wiki.
Check for duplicate concepts, weak claims, missing backlinks, stale claims, and contradictions.
Write the report to claims/contradiction_reports/lint-YYYY-MM-DD.md.
Do not apply major fixes yet.
```

---

### 14.6 Generate flashcards

```text
Use prompts/agents/learning_agent.md to generate flashcards from the last 7 days of reading_log.yaml.
Each flashcard must reference a source wiki page or raw source.
Update learning/review_schedule.yaml.
```

---

### 14.7 Add validation scripts

```text
Create scripts/validate.py that checks:
1. every wiki page has frontmatter
2. every task has required fields
3. every flashcard has a source
4. wiki/index.md links to all wiki pages
5. no broken internal wiki links

Then run the script and fix any issues.
```

---

## 15. MVP Build Sequence

### Phase 1: Repo Skeleton

Create:

```text
AGENTS.md
raw/
wiki/index.md
goals/active/
tasks/today.md
tasks/backlog.md
tasks/timeline.md
logs/activity_log.md
logs/reading_log.yaml
claims/claims.yaml
learning/flashcards/
learning/quizzes/
scripts/validate.py
```

Goal:

```text
The coding agent can navigate and follow conventions.
```

---

### Phase 2: Ingest Workflow

Implement:

- Ingest prompt.
- Wiki page template.
- Index update script.
- Activity log update convention.

Goal:

```text
One raw source becomes useful wiki pages.
```

---

### Phase 3: Goal Workflow

Implement:

- Goal agent prompt.
- Goal schema.
- Subgoal schema.
- 7-day starter plan generation.

Goal:

```text
A vague goal becomes a structured goal tree with evidence-based subgoals.
```

---

### Phase 4: Task Workflow

Implement:

- Task schema.
- Daily planning prompt.
- Task scoring.
- Timeline/backlog updates.

Goal:

```text
The system generates realistic daily tasks from active goals and recent activity.
```

---

### Phase 5: Lint Workflow

Implement:

- Claim schema.
- Lint prompt.
- Link checker.
- Lint report format.

Goal:

```text
The system can detect low-quality knowledge structure.
```

---

### Phase 6: Learning Workflow

Implement:

- Reading log schema.
- Flashcard format.
- Quiz format.
- Review schedule.

Goal:

```text
Recent reading becomes flashcards, quizzes, and review tasks.
```

---

### Phase 7: Optional UI

Only after the repo workflow works.

Possible UI panels:

```text
Goal Tree
  Goal → Subgoal → Task → Evidence

First Principles Canvas
  Assumptions → Truths → Constraints → Bottlenecks → Leverage

Daily Mission
  Today's tasks with difficulty and motivation score

Progress Ledger
  Evidence of learning and building

Knowledge Graph
  Pages, claims, sources, and links

Learning Review
  Flashcards, quizzes, weak concepts, review schedule
```

---

## 16. What Belongs in Skills vs Software?

### Put in agent instructions / skills

Use skills for procedures:

```text
how to ingest a source
how to decompose goals
how to score tasks
how to create flashcards
how to write lint reports
how to update logs
```

### Put in scripts/software

Use code for invariants:

```text
file structure validation
YAML schema validation
link checking
duplicate file detection
task sorting
review date calculation
Git commit summaries
statistics and frequency counts
```

### Put in UI later

Use UI for interaction:

```text
goal tree visualization
timeline
daily plan
flashcard review
knowledge graph
progress dashboard
```

---

## 17. Key Design Principles

1. **Ground everything.**  
   Every important generated claim, task, quiz, and summary should point back to source pages or raw files.

2. **Files are memory.**  
   The agent should not rely on hidden session memory. Store state in logs, goals, tasks, claims, and review files.

3. **Tasks need evidence.**  
   Every task should define what artifact proves completion.

4. **Subgoals should build capability.**  
   Avoid vague topic goals like “read RL.” Prefer “implement and explain a toy actor-learner loop.”

5. **Use scripts for deterministic checks.**  
   Do not ask the LLM to count, sort, and validate things that Python can do reliably.

6. **Prefer small edits.**  
   Large agent rewrites are harder to audit.

7. **Lint before fixing.**  
   For contradictions or large changes, generate a report first.

8. **Start terminal-first.**  
   Build the workflow before building the dashboard.

---

## 18. Future Software Version

Once the repo runtime is useful, build a real app around it.

Possible architecture:

```text
Frontend:
  React / Next.js / Tauri / Electron

Backend:
  FastAPI or Node.js

Storage:
  SQLite + Markdown files

Search:
  SQLite FTS + vector search

Agent Runtime:
  Python or TypeScript orchestrator

Views:
  knowledge graph
  goal tree
  daily tasks
  claim audit
  reading dashboard
  flashcard review
```

But the first version should be repo-native.

---

## 19. Final Summary

The first implementation should not be a full software product. It should be:

```text
Claude Code / Codex
+ Markdown repo
+ AGENTS.md
+ prompt-based subagents
+ Python validation scripts
+ Git history
```

The full loop:

```text
raw sources
  ↓
ingest agent
  ↓
wiki pages + index
  ↓
claim extraction + linting
  ↓
first-principles goals
  ↓
subgoals + evidence
  ↓
daily tasks
  ↓
reading logs + flashcards + quizzes
  ↓
progress ledger
```

Build the little knowledge creature in the terminal first. Once it walks, give it a dashboard, graph view, and maybe a velvet cape.
