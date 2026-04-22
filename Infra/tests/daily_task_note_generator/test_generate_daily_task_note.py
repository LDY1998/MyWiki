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


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def todos_in(content: str) -> list[str]:
    lines = content.splitlines()
    try:
        start = lines.index("## Todos") + 1
    except ValueError:
        return []

    todos: list[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        if line.strip():
            todos.append(line)
    return todos


def split_around_todos(content: str) -> tuple[str, str, str]:
    lines = content.splitlines()
    try:
        start = lines.index("## Todos")
    except ValueError:
        return content, "", ""

    end = len(lines)
    for index, line in enumerate(lines[start + 1 :], start + 1):
        if line.startswith("## "):
            end = index
            break

    prefix = "\n".join(lines[:start])
    todos = "\n".join(lines[start:end])
    suffix = "\n".join(lines[end:])
    return prefix, todos, suffix


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
        today = repo / "tasks/2026-04-22.md"

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(today.exists())
        self.assertEqual(
            todos_in(read(today)),
            ["- [ ] Keep task A", "- [ ] Keep task B"],
        )

    def test_carries_one_item_and_fills_to_five(self) -> None:
        repo = self.make_repo()
        write(
            repo / "tasks/2026-04-21.md",
            """
            # 2026-04-21 Tasks

            ## Todos

            - [ ] Keep task A
            - [x] Finished task

            ## Read

            - Revisit deterministic markdown parsing

            ## Done

            - Sketched carry-forward rules

            ## Follow-ups

            - Continue [[Transformer Memory Math]]
            - Validate generated todo block idempotency
            """,
        )

        result = self.run_script(repo, "2026-04-22")
        today = repo / "tasks/2026-04-22.md"

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(today.exists())
        todos = todos_in(read(today))
        carried = "- [ ] Keep task A"
        generated = [line for line in todos if line != carried]

        self.assertEqual(len(todos), 5)
        self.assertEqual(len(generated), 4)
        self.assertEqual(len(set(todos)), 5)
        self.assertIn(carried, todos)
        self.assertTrue(any("Transformer Memory Math" in line for line in generated))
        self.assertTrue(any("idempotency" in line.lower() for line in generated))
        self.assertTrue(all(line.startswith("- [ ]") for line in todos))
        self.assertNotIn("- [ ] Finished task", todos)

    def test_running_twice_is_idempotent_for_same_day(self) -> None:
        repo = self.make_repo()
        write(
            repo / "tasks/2026-04-21.md",
            """
            # 2026-04-21 Tasks

            ## Todos

            - [ ] Carry me

            ## Read

            - Keep the block stable on repeat runs
            """,
        )

        first = self.run_script(repo, "2026-04-22")

        self.assertEqual(first.returncode, 0, first.stderr)
        today = repo / "tasks/2026-04-22.md"
        first_output = read(today)

        second = self.run_script(repo, "2026-04-22")
        second_output = read(today)

        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(second_output, first_output)
        self.assertEqual(len(todos_in(second_output)), 5)

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
            type: task-note
            status: active
            tags:
              - tasks
            updated: 2026-04-22
            ---

            # 2026-04-22 Tasks

            Intro paragraph that must remain untouched.

            ## Context

            - existing non-todo content

            ## Todos

            - [ ] Old generated line

            ## Read

            - leave this alone

            ## Done

            - leave this alone too

            ## Follow-ups

            - leave this alone also
            """,
        )
        original_today = read(repo / "tasks/2026-04-22.md")
        original_prefix, original_todos, original_suffix = split_around_todos(original_today)

        result = self.run_script(repo, "2026-04-22")
        updated_today = read(repo / "tasks/2026-04-22.md")
        updated_prefix, updated_todos, updated_suffix = split_around_todos(updated_today)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Intro paragraph that must remain untouched.", updated_prefix)
        self.assertIn("## Context", updated_prefix)
        self.assertIn("- leave this alone also", updated_suffix)
        self.assertEqual(updated_prefix, original_prefix)
        self.assertEqual(updated_suffix, original_suffix)
        self.assertNotEqual(updated_todos, original_todos)
        self.assertEqual(todos_in(updated_today), ["- [ ] Carry me"])

    def test_uses_only_most_recent_prior_note_for_carry_forward_decision(self) -> None:
        repo = self.make_repo()
        write(
            repo / "tasks/2026-04-20.md",
            """
            # 2026-04-20 Tasks

            ## Todos

            - [ ] Older task X
            - [ ] Older task Y

            ## Read
            - older context that should be ignored
            """,
        )
        write(
            repo / "tasks/2026-04-21.md",
            """
            # 2026-04-21 Tasks

            ## Todos

            - [ ] Recent task A
            - [x] Finished task

            ## Follow-ups
            - Recent context that should drive generation
            """,
        )

        result = self.run_script(repo, "2026-04-22")
        today = repo / "tasks/2026-04-22.md"

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(today.exists())
        todos = todos_in(read(today))
        self.assertEqual(len(todos), 5)
        self.assertIn("- [ ] Recent task A", todos)
        self.assertNotIn("- [ ] Older task X", todos)
        self.assertNotIn("- [ ] Older task Y", todos)
