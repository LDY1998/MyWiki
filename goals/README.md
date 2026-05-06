---
title: Goals
type: goal-index
status: active
updated: 2026-04-27
---

# Goals

Use this folder for first-principles goals, capability-based subgoals, and evidence requirements.

- `active/` contains current goals.
- `archived/` contains completed, abandoned, or superseded goals.

Every generated task should link back to a goal or subgoal once a goal file exists.

## Workflow

Use `goal-decompose` when creating or materially revising a goal. Goals should connect ambition to evidence, not just list topics.

## Goal File Shape

```markdown
---
title: Goal Title
type: goal
status: active
updated: YYYY-MM-DD
---

# Goal Title

## Desired Outcome

## Assumptions

## Fundamental Truths

## Constraints

## Bottlenecks

## Leverage Points

## Subgoals

## Evidence Required

## Starter Tasks
```

Subgoals should be capability-based, for example “implement and explain a toy actor-learner loop,” not only “read about RL.”
