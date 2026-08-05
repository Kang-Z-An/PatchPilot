from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .types import ToolResult


_IGNORED_DIRS = {".git", ".patchpilot", ".venv", "__pycache__", "node_modules"}
_MAX_OUTPUT = 20_000
_DIFF_PATHS = [".", ":(exclude).patchpilot", ":(exclude).patchpilot/**"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    arguments: str


class RepositoryTools:
    """Repository-scoped tools exposed to the model."""

    SPECS = (
        ToolSpec("list_files", "List repository files.", '{"path":"optional/subdir"}'),
        ToolSpec("search_code", "Search text in repository files.", '{"query":"text","path":"optional/subdir","max_results":50}'),
        ToolSpec("read_file", "Read a line range from a UTF-8 text file.", '{"path":"file.py","start_line":1,"end_line":200}'),
        ToolSpec(
            "replace_text",
            "Replace one exact, unique text block in a file. Prefer this for small local edits.",
            '{"path":"file.py","old_text":"exact text","new_text":"replacement"}',
        ),
        ToolSpec("apply_patch", "Apply one unified diff after validation.", '{"patch":"*** unified diff text ***"}'),
        ToolSpec("run_tests", "Run the human-configured test command.", "{}"),
        ToolSpec("git_diff", "Show the current working-tree diff.", "{}"),
    )

    def __init__(
        self,
        root: Path,
        test_command: str,
        timeout_seconds: int = 120,
        test_runner: Callable[[], ToolResult] | None = None,
    ) -> None:
        self.root = root.resolve()
        if not self.root.is_dir():
            raise ValueError(f"Repository does not exist: {self.root}")
        self.test_command = shlex.split(test_command)
        if not self.test_command:
            raise ValueError("Test command cannot be empty")
        self.timeout_seconds = timeout_seconds
        self._test_runner = test_runner
        self._handlers: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
            "list_files": self.list_files,
            "search_code": self.search_code,
            "read_file": self.read_file,
            "replace_text": self.replace_text,
            "apply_patch": self.apply_patch,
            "run_tests": self.run_tests,
            "git_diff": self.git_diff,
        }

    @classmethod
    def prompt_description(cls) -> str:
        return "\n".join(
            f'- {spec.name}: {spec.description} Arguments: {spec.arguments}'
            for spec in cls.SPECS
        )

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(False, f"Unknown tool: {name}")
        try:
            return handler(arguments)
        except (OSError, ValueError, TypeError) as exc:
            return ToolResult(False, f"{type(exc).__name__}: {exc}")

    def _resolve(self, raw_path: str, *, require_file: bool = False) -> Path:
        candidate = (self.root / raw_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Path escapes repository root") from exc
        if require_file and not candidate.is_file():
            raise ValueError(f"Not a file: {raw_path}")
        return candidate

    def _iter_files(self, base: Path):
        for directory, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in _IGNORED_DIRS)
            for filename in sorted(filenames):
                path = Path(directory) / filename
                if path.is_symlink():
                    continue
                yield path

    def list_files(self, arguments: dict[str, Any]) -> ToolResult:
        base = self._resolve(str(arguments.get("path", ".")))
        if not base.is_dir():
            return ToolResult(False, "Requested path is not a directory")
        files = [str(path.relative_to(self.root)) for path in self._iter_files(base)]
        return ToolResult(True, self._limit("\n".join(files) or "(no files)"))

    def search_code(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query:
            return ToolResult(False, "query must be a non-empty string")
        base = self._resolve(str(arguments.get("path", ".")))
        maximum = max(1, min(int(arguments.get("max_results", 50)), 200))
        matches: list[str] = []
        files = [base] if base.is_file() else self._iter_files(base)
        for path in files:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if query in line:
                            relative = path.relative_to(self.root)
                            matches.append(f"{relative}:{line_number}:{line.rstrip()}")
                            if len(matches) >= maximum:
                                return ToolResult(True, self._limit("\n".join(matches)))
            except (UnicodeDecodeError, OSError):
                continue
        return ToolResult(True, self._limit("\n".join(matches) or "No matches"))

    def read_file(self, arguments: dict[str, Any]) -> ToolResult:
        raw_path = arguments.get("path")
        if not isinstance(raw_path, str):
            return ToolResult(False, "path must be a string")
        path = self._resolve(raw_path, require_file=True)
        start = max(1, int(arguments.get("start_line", 1)))
        end = min(start + 499, int(arguments.get("end_line", start + 199)))
        if end < start:
            return ToolResult(False, "end_line must be greater than or equal to start_line")
        lines = path.read_text(encoding="utf-8").splitlines()
        selected = (f"{number:4d} | {lines[number - 1]}" for number in range(start, min(end, len(lines)) + 1))
        return ToolResult(True, self._limit("\n".join(selected) or "(empty range)"))

    def apply_patch(self, arguments: dict[str, Any]) -> ToolResult:
        patch = arguments.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            return ToolResult(False, "patch must be a non-empty string")
        patch = self._normalize_patch(patch)
        self._validate_patch_paths(patch)
        check = self._run(["git", "apply", "--check", "--unidiff-zero", "-"], stdin=patch)
        if check.returncode != 0:
            return ToolResult(False, self._format_process(check))
        applied = self._run(["git", "apply", "--unidiff-zero", "-"], stdin=patch)
        return ToolResult(applied.returncode == 0, self._format_process(applied) or "Patch applied")

    def replace_text(self, arguments: dict[str, Any]) -> ToolResult:
        raw_path = arguments.get("path")
        old_text = arguments.get("old_text")
        new_text = arguments.get("new_text")
        if not isinstance(raw_path, str):
            return ToolResult(False, "path must be a string")
        if not isinstance(old_text, str) or not old_text:
            return ToolResult(False, "old_text must be a non-empty string")
        if not isinstance(new_text, str):
            return ToolResult(False, "new_text must be a string")
        if old_text == new_text:
            return ToolResult(False, "old_text and new_text must be different")
        path = self._resolve(raw_path, require_file=True)
        content = path.read_text(encoding="utf-8")
        occurrences = content.count(old_text)
        if occurrences != 1:
            return ToolResult(
                False,
                f"old_text must occur exactly once in {raw_path}; found {occurrences}",
            )
        path.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return ToolResult(True, f"Replaced one occurrence in {raw_path}")

    def run_tests(self, arguments: dict[str, Any]) -> ToolResult:
        if arguments:
            return ToolResult(False, "run_tests does not accept arguments")
        if self._test_runner is not None:
            return self._test_runner()
        result = self._run(self.test_command)
        return ToolResult(result.returncode == 0, self._format_process(result))

    def git_diff(self, arguments: dict[str, Any]) -> ToolResult:
        if arguments:
            return ToolResult(False, "git_diff does not accept arguments")
        intent = self._run(
            ["git", "add", "--intent-to-add", "--all", "--", *_DIFF_PATHS]
        )
        if intent.returncode != 0:
            return ToolResult(False, self._format_process(intent))
        result = self._run(["git", "diff", "--no-ext-diff"])
        return ToolResult(result.returncode == 0, self._format_process(result) or "(no diff)")

    def _run(self, command: list[str], stdin: str | None = None) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command,
                cwd=self.root,
                input=stdin,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            return subprocess.CompletedProcess(command, 124, output, f"Timed out after {self.timeout_seconds}s")

    @staticmethod
    def _normalize_patch(patch: str) -> str:
        """Normalize safe formatting variations without guessing hunk contents."""
        lines = patch.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        if lines and lines[0].strip() == "*** Begin Patch":
            lines.pop(0)
        if lines and lines[-1].strip() == "*** End Patch":
            lines.pop()
        if any(line.startswith(("*** Update File:", "*** Add File:", "*** Delete File:")) for line in lines):
            raise ValueError(
                "Codex-style patch directives are not unified diff headers; "
                "use replace_text or provide --- and +++ headers"
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _validate_patch_paths(patch: str) -> None:
        headers = re.findall(r"^(?:---|\+\+\+)\s+([^\t\n]+)", patch, re.MULTILINE)
        if len(headers) < 2:
            raise ValueError("Patch must contain --- and +++ file headers")
        for header in headers:
            if header == "/dev/null":
                continue
            path = header[2:] if header.startswith(("a/", "b/")) else header
            candidate = Path(path)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"Unsafe patch path: {header}")

    @staticmethod
    def _format_process(result: subprocess.CompletedProcess[str]) -> str:
        combined = result.stdout
        if result.stderr:
            combined += ("\n" if combined else "") + result.stderr
        return RepositoryTools._limit(combined.strip())

    @staticmethod
    def _limit(text: str) -> str:
        if len(text) <= _MAX_OUTPUT:
            return text
        return text[:_MAX_OUTPUT] + "\n...[output truncated]"
