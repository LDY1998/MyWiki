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

    headings = {"## Todos", "## Read", "## Done", "## Follow-ups"}
    todos: list[str] = []
    for line in lines[start:]:
        if line in headings:
            break
        if line.strip():
            todos.append(line)
    return todos


def section_text(content: str, heading: str) -> str:
    lines = content.splitlines()
    try:
        start = lines.index(heading) + 1
    except ValueError:
        return ""

    headings = {"## Todos", "## Read", "## Done", "## Follow-ups"}
    collected: list[str] = []
    for line in lines[start:]:
        if line in headings:
            break
        collected.append(line)
    return "\n".join(collected).strip()


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

            ## Follow-ups

            - Continue [[Transformer Memory Math]]
            """,
        )

        result = self.run_script(repo, "2026-04-22")
        today = repo / "tasks/2026-04-22.md"

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(today.exists())
        self.assertEqual(len(todos_in(read(today))), 5)
        self.assertIn("- [ ] Keep task A", todos_in(read(today)))

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

            ## Follow-ups

            - leave this alone also
            """,
        )
        original_today = read(repo / "tasks/2026-04-22.md")

        result = self.run_script(repo, "2026-04-22")
        updated_today = read(repo / "tasks/2026-04-22.md")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(section_text(updated_today, "## Read"), section_text(original_today, "## Read"))
        self.assertEqual(section_text(updated_today, "## Done"), section_text(original_today, "## Done"))
        self.assertEqual(
            section_text(updated_today, "## Follow-ups"),
            section_text(original_today, "## Follow-ups"),
        )
        self.assertNotEqual(todos_in(updated_today), todos_in(original_today))
        self.assertEqual(todos_in(updated_today), ["- [ ] Carry me"])
