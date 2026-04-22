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
