from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from patchpilot.agent import Agent
from patchpilot.tools import RepositoryTools


class ScriptedModel:
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self.outputs = iter(outputs)

    def complete(self, messages: list[dict[str, str]]) -> str:
        return json.dumps(next(self.outputs))


class AgentEndToEndTest(unittest.TestCase):
    def test_first_edit_deadline_blocks_continued_inspection(self) -> None:
        model = ScriptedModel([
            *[
                {"tool": "read_file", "arguments": {"path": "calculator.py"}}
                for _ in range(6)
            ],
            {
                "tool": "replace_text",
                "arguments": {
                    "path": "calculator.py",
                    "old_text": "return a - b",
                    "new_text": "return a + b",
                },
            },
            {"tool": "run_tests", "arguments": {}},
            {"final": "Fixed and verified."},
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "calculator.py").write_text(
                "def add(a, b):\n    return a - b\n",
                encoding="utf-8",
            )
            (root / "test_calculator.py").write_text(
                "import unittest\nfrom calculator import add\n\n"
                "class CalculatorTest(unittest.TestCase):\n"
                "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n",
                encoding="utf-8",
            )
            result = Agent(
                model,
                RepositoryTools(root, "python -m unittest -q"),
                root / "traces",
                max_steps=10,
            ).run("add subtracts")

        self.assertTrue(result.success)
        self.assertEqual(result.metrics.budget_blocked_actions, 1)
        self.assertEqual(result.metrics.first_edit_step, 7)

    def test_budget_blocks_late_broad_inspection_but_leaves_repair_turns(self) -> None:
        model = ScriptedModel([
            {"tool": "list_files", "arguments": {}},
            {
                "tool": "replace_text",
                "arguments": {
                    "path": "calculator.py",
                    "old_text": "return a - b",
                    "new_text": "return a + b",
                },
            },
            {"tool": "run_tests", "arguments": {}},
            {"tool": "git_diff", "arguments": {}},
            {"final": "Fixed and verified."},
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "calculator.py").write_text(
                "def add(a, b):\n    return a - b\n",
                encoding="utf-8",
            )
            (root / "test_calculator.py").write_text(
                "import unittest\nfrom calculator import add\n\n"
                "class CalculatorTest(unittest.TestCase):\n"
                "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n",
                encoding="utf-8",
            )
            result = Agent(
                model,
                RepositoryTools(root, "python -m unittest -q"),
                root / ".patchpilot" / "trajectory",
                max_steps=5,
            ).run("add subtracts")

        self.assertTrue(result.success)
        self.assertEqual(result.metrics.budget_blocked_actions, 1)
        self.assertEqual(result.metrics.first_edit_step, 2)

    def test_agent_repairs_bug_and_records_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "maths.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            (root / "test_maths.py").write_text(
                "import unittest\nfrom maths import add\n\n"
                "class MathsTest(unittest.TestCase):\n"
                "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n",
                encoding="utf-8",
            )
            patch = """--- a/maths.py
+++ b/maths.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""
            model = ScriptedModel([
                {"tool": "search_code", "arguments": {"query": "def add"}},
                {"tool": "read_file", "arguments": {"path": "maths.py"}},
                {"tool": "apply_patch", "arguments": {"patch": patch}},
                {"tool": "run_tests", "arguments": {}},
                {"tool": "git_diff", "arguments": {}},
                {"final": "Fixed add and verified all tests pass."},
            ])
            result = Agent(
                model,
                RepositoryTools(root, "python -m unittest -q"),
                root / "traces",
                max_steps=10,
            ).run("add subtracts instead of adding")

            self.assertTrue(result.success)
            self.assertEqual(result.steps, 6)
            self.assertIn("return a + b", (root / "maths.py").read_text(encoding="utf-8"))
            events = [json.loads(line) for line in result.trajectory_path.read_text().splitlines()]
            self.assertEqual(events[0]["event"], "run_started")
            self.assertEqual(events[0]["trajectory_schema_version"], 2)
            self.assertEqual(events[0]["source"], "ad_hoc")
            self.assertEqual(events[0]["model"], "unknown")
            self.assertEqual(events[0]["initial_messages"][0]["role"], "system")
            self.assertEqual(events[-1]["event"], "run_finished")
            self.assertEqual(sum(event["event"] == "tool_result" for event in events), 5)
            tool_events = [event for event in events if event["event"] == "tool_result"]
            self.assertTrue(all(isinstance(event["observation"], str) for event in tool_events))
            self.assertEqual(result.metrics.patch_attempts, 1)
            self.assertEqual(result.metrics.patch_successes, 1)
            self.assertEqual(result.metrics.tool_failures, 0)

    def test_agent_cannot_self_report_unverified_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            model = ScriptedModel([{"final": "Everything is fixed."}])
            result = Agent(
                model,
                RepositoryTools(root, "python -m unittest -q"),
                root / "traces",
            ).run("fix it")

            self.assertFalse(result.success)

    def test_passing_tests_without_a_patch_is_not_a_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            model = ScriptedModel([
                {"tool": "run_tests", "arguments": {}},
                {"final": "Tests already pass."},
            ])
            result = Agent(
                model,
                RepositoryTools(root, "python -m unittest -q"),
                root / "traces",
            ).run("fix an unspecified bug")

            self.assertFalse(result.success)

    def test_final_automatically_verifies_diff_after_patch_and_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "maths.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            (root / "test_maths.py").write_text(
                "import unittest\nfrom maths import add\n\n"
                "class MathsTest(unittest.TestCase):\n"
                "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n",
                encoding="utf-8",
            )
            model = ScriptedModel([
                {
                    "tool": "replace_text",
                    "arguments": {
                        "path": "maths.py",
                        "old_text": "return a - b",
                        "new_text": "return a + b",
                    },
                },
                {"tool": "run_tests", "arguments": {}},
                {"final": "Fixed and tested."},
            ])
            result = Agent(
                model,
                RepositoryTools(root, "python -m unittest -q"),
                root / "traces",
            ).run("fix add")

            self.assertTrue(result.success)
            self.assertEqual(result.metrics.replace_text_attempts, 1)
            self.assertEqual(result.metrics.replace_text_successes, 1)
            self.assertEqual(result.metrics.automatic_git_diff_calls, 1)
            self.assertTrue(result.metrics.tests_run_after_latest_edit)
            self.assertEqual(result.metrics.empty_diff_attempts, 0)

    def test_protected_edit_is_counted_and_source_fix_can_recover(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "calculator.py").write_text(
                "def add(a, b):\n    return a - b\n", encoding="utf-8"
            )
            (root / "test_calculator.py").write_text(
                "import unittest\nfrom calculator import add\n\n"
                "class CalculatorTest(unittest.TestCase):\n"
                "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n",
                encoding="utf-8",
            )
            self._commit_baseline(root)
            model = ScriptedModel([
                {
                    "tool": "replace_text",
                    "arguments": {
                        "path": "test_calculator.py",
                        "old_text": "self.assertEqual(add(2, 3), 5)",
                        "new_text": "self.assertEqual(add(2, 3), -1)",
                    },
                },
                {
                    "tool": "replace_text",
                    "arguments": {
                        "path": "calculator.py",
                        "old_text": "return a - b",
                        "new_text": "return a + b",
                    },
                },
                {"tool": "run_tests", "arguments": {}},
                {"final": "Fixed production code and verified it."},
            ])

            result = Agent(
                model,
                RepositoryTools(
                    root,
                    "python -m unittest -q",
                    protect_test_files=True,
                ),
                root / "traces",
                max_steps=8,
            ).run("fix add")

        self.assertTrue(result.success)
        self.assertEqual(result.metrics.protected_edit_attempts, 1)
        self.assertEqual(result.metrics.test_files_modified, 0)
        self.assertTrue(result.metrics.tests_run_after_latest_edit)

    def test_restoring_baseline_cannot_pass_with_empty_diff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "config.txt").write_text("mode=old\n", encoding="utf-8")
            self._commit_baseline(root)
            model = ScriptedModel([
                {
                    "tool": "replace_text",
                    "arguments": {
                        "path": "config.txt",
                        "old_text": "mode=old",
                        "new_text": "mode=new",
                    },
                },
                {
                    "tool": "replace_text",
                    "arguments": {
                        "path": "config.txt",
                        "old_text": "mode=new",
                        "new_text": "mode=old",
                    },
                },
                {"tool": "run_tests", "arguments": {}},
                {"final": "Done."},
            ])

            result = Agent(
                model,
                RepositoryTools(root, "python -c 'pass'"),
                root / "traces",
                max_steps=6,
            ).run("change the configuration")

        self.assertFalse(result.success)
        self.assertEqual(result.metrics.empty_diff_attempts, 1)
        self.assertTrue(result.metrics.tests_run_after_latest_edit)

    def test_edit_after_passing_test_requires_a_fresh_test_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "calculator.py").write_text(
                "def add(a, b):\n    return a - b\n", encoding="utf-8"
            )
            (root / "test_calculator.py").write_text(
                "import unittest\nfrom calculator import add\n\n"
                "class CalculatorTest(unittest.TestCase):\n"
                "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n",
                encoding="utf-8",
            )
            self._commit_baseline(root)
            model = ScriptedModel([
                {
                    "tool": "replace_text",
                    "arguments": {
                        "path": "calculator.py",
                        "old_text": "return a - b",
                        "new_text": "return a + b",
                    },
                },
                {"tool": "run_tests", "arguments": {}},
                {
                    "tool": "replace_text",
                    "arguments": {
                        "path": "calculator.py",
                        "old_text": "def add(a, b):",
                        "new_text": "def add(a, b):  # fixed",
                    },
                },
                {"final": "Fixed."},
            ])

            result = Agent(
                model,
                RepositoryTools(root, "python -m unittest -q"),
                root / "traces",
                max_steps=6,
            ).run("fix add")

        self.assertFalse(result.success)
        self.assertFalse(result.metrics.tests_run_after_latest_edit)

    def test_unprotected_test_edit_is_visible_in_quality_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "calculator.py").write_text(
                "def add(a, b):\n    return a - b\n", encoding="utf-8"
            )
            (root / "test_calculator.py").write_text(
                "import unittest\nfrom calculator import add\n\n"
                "class CalculatorTest(unittest.TestCase):\n"
                "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n",
                encoding="utf-8",
            )
            self._commit_baseline(root)
            model = ScriptedModel([
                {
                    "tool": "replace_text",
                    "arguments": {
                        "path": "test_calculator.py",
                        "old_text": "self.assertEqual(add(2, 3), 5)",
                        "new_text": "self.assertEqual(add(2, 3), -1)",
                    },
                },
                {"tool": "run_tests", "arguments": {}},
                {"tool": "git_diff", "arguments": {}},
                {"final": "Updated the expected behavior."},
            ])

            result = Agent(
                model,
                RepositoryTools(root, "python -m unittest -q"),
                root / "traces",
                max_steps=6,
            ).run("change the test expectation")

        self.assertTrue(result.success)
        self.assertEqual(result.metrics.test_files_modified, 1)

    @staticmethod
    def _commit_baseline(root: Path) -> None:
        subprocess.run(["git", "add", "--all"], cwd=root, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=PatchPilot Test",
                "-c",
                "user.email=test@patchpilot.local",
                "commit",
                "-qm",
                "baseline",
            ],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
