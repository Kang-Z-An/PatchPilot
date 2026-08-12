from __future__ import annotations

import ast
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .types import ToolResult


_IGNORED_DIRS = {".git", ".patchpilot", ".venv", "__pycache__", "node_modules"}
_MAX_OUTPUT = 20_000
_DIFF_PATHS = [".", ":(exclude).patchpilot", ":(exclude).patchpilot/**"]
_TEST_DIRECTORY_NAMES = {"test", "tests", "testing"}
_TEST_FILE_NAMES = {"conftest.py"}


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
            "inspect_python_class",
            "Statically find a Python class and trace its base classes, definition locations, and __slots__ declarations.",
            '{"class_name":"Symbol","path":"optional/subdir","max_depth":4}',
        ),
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
        protect_test_files: bool = False,
    ) -> None:
        self.root = root.resolve()
        if not self.root.is_dir():
            raise ValueError(f"Repository does not exist: {self.root}")
        self.test_command = shlex.split(test_command)
        if not self.test_command:
            raise ValueError("Test command cannot be empty")
        self.timeout_seconds = timeout_seconds
        self._test_runner = test_runner
        if not isinstance(protect_test_files, bool):
            raise ValueError("protect_test_files must be a boolean")
        self.protect_test_files = protect_test_files
        self._ignored_diff_paths: set[str] = set()
        self._handlers: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
            "list_files": self.list_files,
            "search_code": self.search_code,
            "read_file": self.read_file,
            "inspect_python_class": self.inspect_python_class,
            "replace_text": self.replace_text,
            "apply_patch": self.apply_patch,
            "run_tests": self.run_tests,
            "git_diff": self.git_diff,
        }

    def prompt_description(self) -> str:
        description = "\n".join(
            f'- {spec.name}: {spec.description} Arguments: {spec.arguments}'
            for spec in self.SPECS
        )
        if self.protect_test_files:
            description += (
                "\nEditing test files is disabled. You may inspect tests, but "
                "replace_text and apply_patch will reject changes under test/, "
                "tests/, or testing/, and files named test_*.py, *_test.py, or "
                "conftest.py. Fix production code only."
            )
        return description

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(False, f"Unknown tool: {name}")
        try:
            return handler(arguments)
        except (OSError, ValueError, TypeError) as exc:
            return ToolResult(False, f"{type(exc).__name__}: {exc}")

    def ignore_diff_path(self, path: Path) -> None:
        """Exclude an Agent-owned path inside the repository from repair diffs."""
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self.root)
        except ValueError:
            return
        if relative == Path("."):
            raise ValueError("Cannot ignore the entire repository diff")
        self._ignored_diff_paths.add(relative.as_posix())

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

    def inspect_python_class(self, arguments: dict[str, Any]) -> ToolResult:
        raw_name = arguments.get("class_name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            return ToolResult(False, "class_name must be a non-empty string")
        class_name = raw_name.strip().rsplit(".", 1)[-1]
        if not class_name.isidentifier():
            return ToolResult(False, "class_name must end with a valid Python identifier")
        base = self._resolve(str(arguments.get("path", ".")))
        if not base.exists():
            return ToolResult(False, "Requested path does not exist")
        max_depth = int(arguments.get("max_depth", 4))
        if not 1 <= max_depth <= 8:
            return ToolResult(False, "max_depth must be between 1 and 8")

        definitions: dict[str, list[dict[str, Any]]] = {}
        parse_errors = 0
        files = [base] if base.is_file() else self._iter_files(base)
        for path in files:
            if path.suffix != ".py":
                continue
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except (OSError, UnicodeDecodeError, SyntaxError):
                parse_errors += 1
                continue
            relative = str(path.relative_to(self.root))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                slots_line = self._class_slots_line(node)
                record = {
                    "name": node.name,
                    "path": relative,
                    "line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                    "declares_slots": slots_line is not None,
                    "slots_line": slots_line,
                    "base_expressions": [ast.unparse(item) for item in node.bases],
                }
                definitions.setdefault(node.name, []).append(record)

        def expand(
            record: dict[str, Any],
            depth: int,
            lineage: frozenset[tuple[str, int]],
        ) -> dict[str, Any]:
            key = (str(record["path"]), int(record["line"]))
            result = {
                field: value
                for field, value in record.items()
                if field != "base_expressions"
            }
            bases = []
            for expression in record["base_expressions"]:
                simple_name = expression.rsplit(".", 1)[-1]
                candidates = definitions.get(simple_name, [])
                base_result: dict[str, Any] = {
                    "expression": expression,
                    "definitions": [],
                }
                if depth < max_depth:
                    base_result["definitions"] = [
                        expand(candidate, depth + 1, lineage | {key})
                        for candidate in candidates[:5]
                        if (candidate["path"], candidate["line"]) not in lineage | {key}
                    ]
                elif candidates:
                    base_result["definitions"] = [
                        {
                            field: value
                            for field, value in candidate.items()
                            if field != "base_expressions"
                        }
                        for candidate in candidates[:5]
                    ]
                    base_result["depth_limited"] = True
                bases.append(base_result)
            result["bases"] = bases
            return result

        matches = definitions.get(class_name, [])
        output = {
            "class_name": class_name,
            "max_depth": max_depth,
            "definitions_found": len(matches),
            "parse_errors": parse_errors,
            "definitions": [expand(record, 0, frozenset()) for record in matches[:10]],
        }
        return ToolResult(True, self._limit(json.dumps(output, ensure_ascii=False, indent=2)))

    @staticmethod
    def _class_slots_line(node: ast.ClassDef) -> int | None:
        for item in node.body:
            targets: list[ast.expr] = []
            if isinstance(item, ast.Assign):
                targets = item.targets
            elif isinstance(item, ast.AnnAssign):
                targets = [item.target]
            if any(isinstance(target, ast.Name) and target.id == "__slots__" for target in targets):
                return item.lineno
        return None

    def apply_patch(self, arguments: dict[str, Any]) -> ToolResult:
        patch = arguments.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            return ToolResult(False, "patch must be a non-empty string")
        patch = self._normalize_patch(patch)
        self._validate_patch_paths(patch)
        protected = self._protected_patch_paths(patch)
        if protected:
            return ToolResult(
                False,
                "Protected test files are read-only: " + ", ".join(protected),
                {
                    "reason": "protected_test_file",
                    "protected_paths": protected,
                },
            )
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
        if self.protect_test_files and is_test_file_path(raw_path):
            return ToolResult(
                False,
                f"Protected test file is read-only: {raw_path}",
                {
                    "reason": "protected_test_file",
                    "protected_paths": [raw_path],
                },
            )
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
        pathspecs = self._diff_pathspecs()
        intent = self._run(
            ["git", "add", "--intent-to-add", "--all", "--", *pathspecs]
        )
        if intent.returncode != 0:
            return ToolResult(False, self._format_process(intent))
        result = self._run(["git", "diff", "--no-ext-diff", "--", *pathspecs])
        if result.returncode != 0:
            return ToolResult(False, self._format_process(result))
        names = self._run(
            ["git", "diff", "--name-only", "--no-ext-diff", "-z", "--", *pathspecs]
        )
        if names.returncode != 0:
            return ToolResult(False, self._format_process(names))
        changed_files = sorted(path for path in names.stdout.split("\0") if path)
        test_files = [path for path in changed_files if is_test_file_path(path)]
        data = {
            "has_diff": bool(result.stdout.strip()),
            "changed_files": changed_files,
            "test_files_modified": test_files,
        }
        if not data["has_diff"]:
            return ToolResult(
                False,
                "No working-tree diff. A repair must leave a non-empty change.",
                {**data, "reason": "empty_diff"},
            )
        if self.protect_test_files and test_files:
            return ToolResult(
                False,
                "Final diff contains protected test files: " + ", ".join(test_files),
                {**data, "reason": "protected_final_diff"},
            )
        return ToolResult(True, self._format_process(result), data)

    def _diff_pathspecs(self) -> list[str]:
        pathspecs = list(_DIFF_PATHS)
        for path in sorted(self._ignored_diff_paths):
            pathspecs.extend((f":(exclude){path}", f":(exclude){path}/**"))
        return pathspecs

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

    def _protected_patch_paths(self, patch: str) -> list[str]:
        if not self.protect_test_files:
            return []
        paths = {
            normalized
            for header in re.findall(
                r"^(?:---|\+\+\+)\s+([^\t\n]+)", patch, re.MULTILINE
            )
            if (normalized := _normalize_diff_path(header)) is not None
            and is_test_file_path(normalized)
        }
        return sorted(paths)

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


def is_test_file_path(raw_path: str) -> bool:
    """Return whether a repository-relative path is part of the test suite."""
    path = PurePosixPath(raw_path.replace("\\", "/"))
    parts = tuple(part.lower() for part in path.parts if part not in {"", "."})
    if not parts:
        return False
    filename = parts[-1]
    return (
        any(part in _TEST_DIRECTORY_NAMES for part in parts[:-1])
        or filename in _TEST_FILE_NAMES
        or (filename.startswith("test_") and filename.endswith(".py"))
        or filename.endswith("_test.py")
    )


def _normalize_diff_path(header: str) -> str | None:
    if header == "/dev/null":
        return None
    return header[2:] if header.startswith(("a/", "b/")) else header
