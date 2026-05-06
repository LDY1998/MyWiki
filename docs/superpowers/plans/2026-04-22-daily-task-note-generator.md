# Daily Task Note Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repo-local skill and deterministic Python helper that creates or refreshes `tasks/YYYY-MM-DD.md` by carrying forward incomplete todos and optionally generating new ones from recent task history.

**Architecture:** The implementation uses a repo-local skill folder for discovery and instructions, plus a single Python script that handles date selection, markdown parsing, todo carry-forward rules, recent-note synthesis, and safe file updates. Tests use `unittest` with temporary directories so the behavior is verified without mutating the real `tasks/` folder.

**Tech Stack:** Markdown files, Python 3 standard library (`pathlib`, `re`, `datetime`, `argparse`, `tempfile`, `unittest`), repo-local Codex skills metadata.

---

## File Structure

- Create: `.agents/skills/daily-task-note-generator/SKILL.md`
- Create: `.agents/skills/daily-task-note-generator/agents/openai.yaml`
- Create: `.agents/skills/daily-task-note-generator/scripts/generate_daily_task_note.py`
- Create: `tests/daily_task_note_generator/test_generate_daily_task_note.py`
- Modify: `docs/superpowers/specs/2026-04-22-daily-task-note-generator-design.md`

File responsibilities:

- `.agents/skills/daily-task-note-generator/SKILL.md`: Trigger guidance, workflow, and script invocation instructions.
- `.agents/skills/daily-task-note-generator/agents/openai.yaml`: Human-facing skill metadata.
- `.agents/skills/daily-task-note-generator/scripts/generate_daily_task_note.py`: All parsing, carry-forward logic, synthesis, and write/update behavior.
- `tests/daily_task_note_generator/test_generate_daily_task_note.py`: End-to-end behavior tests using temporary repo fixtures.
- `docs/superpowers/specs/2026-04-22-daily-task-note-generator-design.md`: Spec correction already identified during plan writing.

### Task 1: Lock The Spec And Test Surface

**Files:**
- Modify: `docs/superpowers/specs/2026-04-22-daily-task-note-generator-design.md`
- Create: `tests/daily_task_note_generator/test_generate_daily_task_note.py`

- [ ] **Step 1: Confirm the spec example matches the approved rule**

Ensure Case C in the spec says that 2 incomplete items carry forward and generate 0 new items.

Expected snippet:

```markdown
### Case C: 2 incomplete items yesterday

- Carry forward both items.
- Generate 0 new items.
```

- [ ] **Step 2: Write the failing tests for carry-forward and fill-to-5 behavior**

Create `tests/daily_task_note_generator/test_generate_daily_task_note.py` with this initial scaffold:

```python
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


SCRIPT = Path(".agents/skills/daily-task-note-generator/scripts/generate_daily_task_note.py")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


class DailyTaskNoteGeneratorTests(unittest.TestCase):
    def make_repo(self) -> Path:
        repo = Path(tempfile.mkdtemp(prefix="daily-task-note-"))
        write(
            repo / "tasks/templates/Daily Task Note.md",
            """
            ---
            title: "{{title}} Tasks"
            type: task-note
            status: active
            tags:
              - tasks
            updated: "{{date:YYYY-MM-DD}}"
            ---

            # {{title}} Tasks

            ## Todos

            ## Read

            ## Done

            ## Follow-ups
            """,
        )
        return repo

    def run_script(self, repo: Path, date: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(Path.cwd() / SCRIPT), "--repo-root", str(repo), "--date", date],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_carries_two_or_more_incomplete_items_without_generation(self) -> None:
        repo = self.make_repo()
        write(
            repo / "tasks/2026-04-21.md",
            """
            # 2026-04-21 Tasks

            ## Todos

            - [ ] Keep task A
            - [ ] Keep task B
            - [x] Finished task

            ## Read
            """,
        )

        result = self.run_script(repo, "2026-04-22")

        self.assertNotEqual(result.returncode, 0)

    def test_carries_one_item_and_fills_to_five(self) -> None:
        repo = self.make_repo()
        write(
            repo / "tasks/2026-04-21.md",
            """
            # 2026-04-21 Tasks

            ## Todos

            - [ ] Keep task A
            - [x] Finished task

            ## Follow-ups

            - Continue [[Transformer Memory Math]]
            """,
        )

        result = self.run_script(repo, "2026-04-22")

        self.assertNotEqual(result.returncode, 0)
```

- [ ] **Step 3: Add a failing test for updating only the `Todos` section of an existing file**

Extend the same test file with:

```python
    def test_updates_only_todos_in_existing_today_file(self) -> None:
        repo = self.make_repo()
        write(
            repo / "tasks/2026-04-21.md",
            """
            # 2026-04-21 Tasks

            ## Todos

            - [ ] Carry me

            ## Follow-ups

            - Follow up on [[FSDP]]
            """,
        )
        write(
            repo / "tasks/2026-04-22.md",
            """
            ---
            title: 2026-04-22 Tasks
            ---

            # 2026-04-22 Tasks

            ## Todos

            - [ ] Old generated line

            ## Read

            - leave this alone

            ## Done

            - leave this alone too
            """,
        )

        result = self.run_script(repo, "2026-04-22")

        self.assertNotEqual(result.returncode, 0)
```

- [ ] **Step 4: Run the tests to verify they fail before implementation**

Run:

```bash
python3 -m unittest tests.daily_task_note_generator.test_generate_daily_task_note -v
```

Expected: `FAIL` or `ERROR` because the generator script does not exist yet.

- [ ] **Step 5: Commit the red test scaffold and spec fix**

Run:

```bash
git add docs/superpowers/specs/2026-04-22-daily-task-note-generator-design.md tests/daily_task_note_generator/test_generate_daily_task_note.py
git commit -m "test: add daily task note generator coverage"
```

### Task 2: Implement Parsing And Carry-Forward Logic

**Files:**
- Create: `.agents/skills/daily-task-note-generator/scripts/generate_daily_task_note.py`
- Test: `tests/daily_task_note_generator/test_generate_daily_task_note.py`

- [ ] **Step 1: Write the script skeleton with note discovery and section parsing helpers**

Create `.agents/skills/daily-task-note-generator/scripts/generate_daily_task_note.py` starting with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


DATE_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")
CHECKBOX_RE = re.compile(r"^-\s+\[(?P<state>[ xX])\]\s+(?P<body>.+)$")
SECTION_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$")


@dataclass(frozen=True)
class TodoItem:
    text: str
    checked: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--date", dest="today")
    return parser.parse_args()


def parse_date(raw: str | None) -> date:
    if raw:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    return date.today()


def list_prior_daily_notes(tasks_dir: Path, today_str: str) -> list[Path]:
    notes = [
        path
        for path in tasks_dir.iterdir()
        if path.is_file() and DATE_NAME_RE.match(path.name) and path.stem < today_str
    ]
    return sorted(notes, key=lambda path: path.stem, reverse=True)[:5]
```

- [ ] **Step 2: Implement markdown section parsing and incomplete todo extraction**

Add these functions:

```python
def split_sections(text: str) -> tuple[list[str], dict[str, list[str]]]:
    frontmatter_and_preamble: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = SECTION_RE.match(line)
        if match:
            current = match.group("title")
            sections.setdefault(current, [])
            continue
        if current is None:
            frontmatter_and_preamble.append(line)
        else:
            sections[current].append(line)
    return frontmatter_and_preamble, sections


def parse_todo_items(section_lines: list[str]) -> list[TodoItem]:
    items: list[TodoItem] = []
    for line in section_lines:
        match = CHECKBOX_RE.match(line.strip())
        if not match:
            continue
        items.append(TodoItem(text=match.group("body").strip(), checked=match.group("state").lower() == "x"))
    return items


def incomplete_items(note_path: Path) -> list[str]:
    _, sections = split_sections(note_path.read_text(encoding="utf-8"))
    return [item.text for item in parse_todo_items(sections.get("Todos", [])) if not item.checked]
```

- [ ] **Step 3: Implement carry-forward selection and normalization**

Continue with:

```python
def normalize_task(text: str) -> str:
    return " ".join(text.split())


def carry_forward_items(previous_note: Path | None) -> list[str]:
    if previous_note is None:
        return []
    items = [normalize_task(text) for text in incomplete_items(previous_note)]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped
```

- [ ] **Step 4: Run tests to verify parsing exists and synthesis cases still fail**

Run:

```bash
python3 -m unittest tests.daily_task_note_generator.test_generate_daily_task_note -v
```

Expected: tests still fail because file writing and todo generation are not implemented yet, but import/parsing errors are resolved.

- [ ] **Step 5: Commit the parser and carry-forward layer**

Run:

```bash
git add .agents/skills/daily-task-note-generator/scripts/generate_daily_task_note.py
git commit -m "feat: add daily task note parsing logic"
```

### Task 3: Implement Todo Synthesis And Safe File Updates

**Files:**
- Modify: `.agents/skills/daily-task-note-generator/scripts/generate_daily_task_note.py`
- Test: `tests/daily_task_note_generator/test_generate_daily_task_note.py`

- [ ] **Step 1: Add context collection for generated tasks**

Extend the script with:

```python
def collect_candidate_lines(note_paths: list[Path], repo_root: Path) -> list[str]:
    candidates: list[str] = []
    for note_path in note_paths:
        _, sections = split_sections(note_path.read_text(encoding="utf-8"))
        for section_name in ("Follow-ups", "Done", "Read"):
            for line in sections.get(section_name, []):
                stripped = line.strip()
                if stripped.startswith("- "):
                    candidates.append(stripped[2:].strip())
    plans_dir = repo_root / "plans"
    if plans_dir.exists():
        for plan_path in sorted(plans_dir.glob("*.md")):
            for line in plan_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("- ") and len(stripped) > 4:
                    candidates.append(stripped[2:].strip())
    return candidates


def generate_new_items(candidates: list[str], carried: list[str], limit: int) -> list[str]:
    results: list[str] = []
    seen = {item.casefold() for item in carried}
    for candidate in candidates:
        normalized = normalize_task(candidate)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        results.append(normalized)
        if len(results) == limit:
            break
    return results
```

- [ ] **Step 2: Add template rendering and `Todos`-only replacement**

Add:

```python
def render_todos(items: list[str]) -> list[str]:
    if not items:
        return [""]
    return [""] + [f"- [ ] {item}" for item in items] + [""]


def replace_todos_section(text: str, todo_lines: list[str]) -> str:
    lines = text.splitlines()
    result: list[str] = []
    i = 0
    while i < len(lines):
        result.append(lines[i])
        if lines[i].strip() == "## Todos":
            result.extend(todo_lines)
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
            continue
        i += 1
    return "\n".join(result) + "\n"


def render_new_note(template_text: str, today_str: str, todo_lines: list[str]) -> str:
    rendered = template_text.replace("{{title}}", today_str).replace("{{date:YYYY-MM-DD}}", today_str)
    return replace_todos_section(rendered, todo_lines)
```

- [ ] **Step 3: Wire the decision rule into `main()`**

Finish the script with:

```python
def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    today = parse_date(args.today)
    today_str = today.isoformat()
    tasks_dir = repo_root / "tasks"
    template_path = tasks_dir / "templates" / "Daily Task Note.md"
    prior_notes = list_prior_daily_notes(tasks_dir, today_str)
    previous_note = prior_notes[0] if prior_notes else None
    carried = carry_forward_items(previous_note)
    if len(carried) < 2:
        needed = 5 - len(carried)
        generated = generate_new_items(collect_candidate_lines(prior_notes, repo_root), carried, needed)
        final_items = carried + generated
    else:
        final_items = carried
    todo_lines = render_todos(final_items)
    today_path = tasks_dir / f"{today_str}.md"
    if today_path.exists():
        original = today_path.read_text(encoding="utf-8")
        updated = replace_todos_section(original, todo_lines)
    else:
        updated = render_new_note(template_path.read_text(encoding="utf-8"), today_str, todo_lines)
    today_path.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Tighten the tests to assert concrete output**

Replace the `assertNotEqual(result.returncode, 0)` placeholders in the test file with assertions like:

```python
        self.assertEqual(result.returncode, 0, result.stderr)
        today = (repo / "tasks/2026-04-22.md").read_text(encoding="utf-8")
        self.assertIn("- [ ] Keep task A", today)
        self.assertIn("- [ ] Keep task B", today)
        self.assertNotIn("Old generated line", today)
```

For the fill-to-5 case, assert:

```python
        todo_lines = [line for line in today.splitlines() if line.startswith("- [ ] ")]
        self.assertEqual(len(todo_lines), 5)
        self.assertIn("- [ ] Keep task A", todo_lines)
```

For the existing-file case, assert:

```python
        self.assertIn("- leave this alone", today)
        self.assertIn("- leave this alone too", today)
```

- [ ] **Step 5: Run the tests and verify green**

Run:

```bash
python3 -m unittest tests.daily_task_note_generator.test_generate_daily_task_note -v
```

Expected: all tests `OK`.

- [ ] **Step 6: Commit synthesis and file-update behavior**

Run:

```bash
git add .agents/skills/daily-task-note-generator/scripts/generate_daily_task_note.py tests/daily_task_note_generator/test_generate_daily_task_note.py
git commit -m "feat: generate daily task notes from recent progress"
```

### Task 4: Add Skill Metadata And Usage Instructions

**Files:**
- Create: `.agents/skills/daily-task-note-generator/SKILL.md`
- Create: `.agents/skills/daily-task-note-generator/agents/openai.yaml`

- [ ] **Step 1: Write the skill instructions**

Create `.agents/skills/daily-task-note-generator/SKILL.md`:

```markdown
---
name: daily-task-note-generator
description: Generate or refresh today's task note in this repo by carrying forward incomplete todos and optionally filling to five tasks from recent progress.
---

# Daily Task Note Generator

Use this skill when the user wants to create, refresh, or prepare `tasks/YYYY-MM-DD.md` for the current day in this repo.

## Workflow

1. Run `.agents/skills/daily-task-note-generator/scripts/generate_daily_task_note.py` from the repo root.
2. Let the script determine today's date unless the user explicitly asks for another date.
3. Read the script's result and summarize what changed.

## Rules

- Read at most the 5 most recent prior daily notes.
- If the most recent prior note has fewer than 2 incomplete todo items, carry them forward and generate enough new tasks to reach 5 total.
- If the most recent prior note has 2 or more incomplete todo items, carry them forward and generate nothing new.
- Update only `## Todos` when today's note already exists.
- Leave `Read`, `Done`, and `Follow-ups` untouched.
```

- [ ] **Step 2: Write the skill metadata file**

Create `.agents/skills/daily-task-note-generator/agents/openai.yaml`:

```yaml
display_name: Daily Task Note Generator
short_description: Create or refresh today's daily task note from recent task history.
default_prompt: Generate or refresh today's task note in this repo.
```

- [ ] **Step 3: Run a manual smoke test against the real repo using an explicit date**

Run:

```bash
python3 .agents/skills/daily-task-note-generator/scripts/generate_daily_task_note.py --repo-root . --date 2026-04-22
```

Expected:

- exit code `0`
- `tasks/2026-04-22.md` created or updated
- only the `## Todos` section changes

- [ ] **Step 4: Review the generated note for grounding and formatting**

Check:

- carried-forward items follow the approved `< 2 incomplete` rule
- generated items reflect recent notes and plans
- checkbox formatting is consistent
- non-`Todos` sections remain untouched if the file already existed

- [ ] **Step 5: Commit the skill scaffolding**

Run:

```bash
git add .agents/skills/daily-task-note-generator/SKILL.md .agents/skills/daily-task-note-generator/agents/openai.yaml tasks/2026-04-22.md
git commit -m "feat: add repo-local daily task note generator skill"
```

### Task 5: Final Verification And Review

**Files:**
- Modify: any files touched above if verification reveals issues

- [ ] **Step 1: Run the full automated verification**

Run:

```bash
python3 -m unittest tests.daily_task_note_generator.test_generate_daily_task_note -v
python3 .agents/skills/daily-task-note-generator/scripts/generate_daily_task_note.py --repo-root . --date 2026-04-22
git diff -- tasks/2026-04-22.md
```

Expected:

- tests pass
- the script exits `0`
- the diff for `tasks/2026-04-22.md` shows only `## Todos` updates

- [ ] **Step 2: Re-run the generator to verify idempotency**

Run:

```bash
python3 .agents/skills/daily-task-note-generator/scripts/generate_daily_task_note.py --repo-root . --date 2026-04-22
git diff -- tasks/2026-04-22.md
```

Expected: no new duplicate todo lines appear after the second run.

- [ ] **Step 3: Review the skill trigger text and script path**

Check that:

- the `description` in `SKILL.md` clearly signals the trigger
- the `default_prompt` in `agents/openai.yaml` matches the actual workflow
- the script path referenced in the skill exists exactly

- [ ] **Step 4: Commit any verification fixes**

Run:

```bash
git add .agents/skills/daily-task-note-generator tests/daily_task_note_generator tasks/2026-04-22.md
git commit -m "test: verify daily task note generator workflow"
```

## Self-Review

Spec coverage check:

- repo-local skill layout: covered in Task 4
- deterministic helper script: covered in Tasks 2 and 3
- `< 2 incomplete` rule: covered in Tasks 1, 2, and 3
- max 5 recent prior notes: covered in Task 2
- `Todos`-only update behavior: covered in Tasks 1 and 3
- idempotency: covered in Task 5
- scheduler deferred: no scheduler work included

Placeholder scan:

- No `TODO`, `TBD`, or deferred implementation placeholders remain.

Type consistency:

- Script path, function names, and test commands are consistent across tasks.
