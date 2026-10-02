"""A real repair loop with scripted decisions: no API, Docker, or GPU needed."""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from patchpilot.agent import Agent
from patchpilot.tools import RepositoryTools


class ScriptedDemoModel:
    """Fixed actions demonstrate tool execution, not LLM capability."""

    def __init__(self) -> None:
        self.actions = iter([
            {"tool": "read_file", "arguments": {"path": "calculator.py"}},
            {"tool": "replace_text", "arguments": {
                "path": "calculator.py",
                "old_text": "return a - b", "new_text": "return a + b",
            }},
            {"tool": "run_tests", "arguments": {}},
            {"tool": "git_diff", "arguments": {}},
            {"final": "Changed subtraction to addition; tests pass and diff reviewed."},
        ])

    def complete(self, messages: list[dict[str, str]]) -> str:
        return json.dumps(next(self.actions))


def main() -> int:
    trace_dir = Path(".patchpilot/demo") / uuid4().hex[:12]
    command = f"{shlex.quote(sys.executable)} -B -m unittest discover -s . -q"
    with tempfile.TemporaryDirectory(prefix="patchpilot-demo-") as directory:
        repo = Path(directory)
        (repo / "calculator.py").write_text(
            "def add(a, b):\n    return a - b\n", encoding="utf-8",
        )
        (repo / "test_calculator.py").write_text(
            "import unittest\nfrom calculator import add\n\n"
            "class AdditionTest(unittest.TestCase):\n"
            "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run([
            "git", "-C", str(repo), "-c", "user.name=PatchPilot Demo",
            "-c", "user.email=demo@example.invalid", "-c", "commit.gpgsign=false",
            "commit", "-qm", "Buggy demo baseline",
        ], check=True)
        tools = RepositoryTools(repo, command, protect_test_files=True)
        baseline = tools.execute("run_tests", {})
        if baseline.ok:
            raise RuntimeError("Demo baseline unexpectedly passed")
        print("Scripted model; no API calls. Baseline test: FAILED (expected).")
        result = Agent(
            ScriptedDemoModel(), tools, trace_dir.resolve(), max_steps=8,
            trajectory_metadata={"source": "offline_demo", "model": "scripted"},
        ).run("add(2, 3) returns -1; it should return 5.")
        print(tools.execute("git_diff", {}).output)
        print(json.dumps({
            "agent_verified": result.success, "steps": result.steps,
            "trajectory": str(result.trajectory_path),
            "model": "scripted (not a model-quality result)",
        }, indent=2))
        return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
