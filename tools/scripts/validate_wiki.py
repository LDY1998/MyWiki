#!/usr/bin/env python3
"""Validate MyWiki structure without modifying files."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DIR = REPO_ROOT / "wiki"
INDEX_FILE = WIKI_DIR / "index.md"
RAW_DIRS = [REPO_ROOT / "raw" / "ai", REPO_ROOT / "raw" / "infra"]

REQUIRED_FRONTMATTER = {"title", "type", "status", "updated"}
ALLOWED_WIKI_TYPES = {
    "schema",
    "map",
    "concept",
    "system",
    "paper",
    "source",
    "question",
    "comparison",
    "project",
    "plan",
    "glossary",
    "writing",
}
KNOWN_ROOT_FILES = {
    "AGENTS.md",
    "README.md",
    "index.md",
    "knowledge_os_agent_runtime_spec.md",
}
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


@dataclass(frozen=True)
class Issue:
    severity: str
    path: Path
    message: str

    def render(self) -> str:
        rel = self.path.relative_to(REPO_ROOT)
        return f"[{self.severity}] {rel}: {self.message}"


def markdown_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.md") if path.is_file())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_frontmatter(text: str) -> tuple[dict[str, str], bool]:
    if not text.startswith("---\n"):
        return {}, False

    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, False

    fields: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if not line.strip() or line.startswith((" ", "-")):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip().strip('"')
    return fields, True


def strip_fenced_blocks(text: str) -> str:
    lines: list[str] = []
    fence: str | None = None

    for line in text.splitlines():
        stripped = line.lstrip()
        match = re.match(r"(`{3,}|~{3,})", stripped)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker.startswith(fence[0]) and len(marker) >= len(fence):
                fence = None
            continue
        if fence is None:
            lines.append(line)

    return "\n".join(lines)


def validate_frontmatter() -> list[Issue]:
    issues: list[Issue] = []
    for path in markdown_files(WIKI_DIR):
        text = read_text(path)
        fields, has_frontmatter = parse_frontmatter(text)
        if not has_frontmatter:
            issues.append(Issue("error", path, "missing YAML frontmatter"))
            continue

        missing = sorted(REQUIRED_FRONTMATTER - fields.keys())
        if missing:
            issues.append(
                Issue("error", path, f"missing required frontmatter fields: {', '.join(missing)}")
            )

        page_type = fields.get("type")
        if page_type and page_type not in ALLOWED_WIKI_TYPES:
            allowed = ", ".join(sorted(ALLOWED_WIKI_TYPES))
            issues.append(Issue("error", path, f"unknown type '{page_type}' (allowed: {allowed})"))

    return issues


def all_markdown_titles() -> dict[str, list[Path]]:
    titles: dict[str, list[Path]] = {}
    for path in markdown_files(REPO_ROOT):
        if ".git" in path.parts:
            continue
        titles.setdefault(path.stem, []).append(path)
    return titles


def validate_wikilinks() -> list[Issue]:
    issues: list[Issue] = []
    title_index = all_markdown_titles()
    for path in markdown_files(WIKI_DIR):
        text = INLINE_CODE_RE.sub("", strip_fenced_blocks(read_text(path)))
        for target in sorted(set(WIKILINK_RE.findall(text))):
            if target not in title_index:
                issues.append(Issue("error", path, f"unresolved wikilink [[{target}]]"))
    return issues


def validate_index_coverage() -> list[Issue]:
    issues: list[Issue] = []
    if not INDEX_FILE.exists():
        return [Issue("error", INDEX_FILE, "wiki index is missing")]

    index_text = read_text(INDEX_FILE)
    exempt = {"index.md", "Wiki Maintenance Schema.md", "Infra Wiki Log.md"}
    for path in markdown_files(WIKI_DIR):
        if path.name in exempt:
            continue
        if path.stem not in index_text and str(path.relative_to(WIKI_DIR)) not in index_text:
            issues.append(Issue("warning", path, "not referenced from wiki/index.md"))
    return issues


def validate_raw_source_coverage() -> list[Issue]:
    issues: list[Issue] = []
    wiki_text = "\n".join(read_text(path) for path in markdown_files(WIKI_DIR))

    for raw_dir in RAW_DIRS:
        for path in markdown_files(raw_dir):
            if path.stem not in wiki_text and str(path.relative_to(REPO_ROOT)) not in wiki_text:
                issues.append(
                    Issue(
                        "warning",
                        path,
                        "raw source is not referenced by any wiki page; consider a source summary",
                    )
                )
    return issues


def validate_root_files() -> list[Issue]:
    issues: list[Issue] = []
    for path in sorted(REPO_ROOT.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name in KNOWN_ROOT_FILES:
            continue
        issues.append(Issue("warning", path, "loose root file should be classified or documented"))
    return issues


def run_checks(include_warnings: bool) -> list[Issue]:
    checks = [
        validate_frontmatter,
        validate_wikilinks,
        validate_index_coverage,
        validate_raw_source_coverage,
        validate_root_files,
    ]
    issues = [issue for check in checks for issue in check()]
    if not include_warnings:
        issues = [issue for issue in issues if issue.severity == "error"]
    return sorted(issues, key=lambda issue: (issue.severity, str(issue.path), issue.message))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the MyWiki repository structure.")
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="suppress warnings and only report errors",
    )
    args = parser.parse_args()

    issues = run_checks(include_warnings=not args.errors_only)
    if not issues:
        print("Wiki validation passed.")
        return 0

    print(f"Wiki validation found {len(issues)} issue(s):")
    for issue in issues:
        print(issue.render())

    return 1 if any(issue.severity == "error" for issue in issues) else 0


if __name__ == "__main__":
    sys.exit(main())
