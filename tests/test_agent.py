from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from patchpilot.agent import Agent
from patchpilot.tools import RepositoryTools
from patchpilot.types import ToolResult


class ScriptedModel:
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self.outputs = iter(outputs)

    def complete(self, messages: list[dict[str, str]]) -> str:
        return json.dumps(next(self.outputs))


class AgentEndToEndTest(unittest.TestCase):
    def test_dual_probe_allows_edit_first_but_requires_fresh_pair_and_tests(self) -> None:
        edit = {"tool": "replace_text", "arguments": {
            "path": "maths.py", "old_text": "return a - b", "new_text": "return a + b",
        }}
        pair = {"tool": "run_probe_pair", "arguments": {"code": "assert add(2, 3) == 5"}}
        tests = {"tool": "run_tests", "arguments": {}}
        final = {"final": "Fixed."}
        for actions, expected in (
            ([edit, pair, tests, final], True),
            ([edit, pair, final], False),
            ([edit, tests, final], False),
            ([edit, pair, {"tool": "replace_text", "arguments": {
                "path": "maths.py", "old_text": "return a + b", "new_text": "return b + a",
            }}, tests, final], False),
        ):
            with self.subTest(actions=actions), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                subprocess.run(["git", "init", "-q"], cwd=root, check=True)
                (root / "maths.py").write_text("def add(a, b):\n    return a - b\n")
                result = Agent(
                    ScriptedModel(actions), RepositoryTools(
                        root, "python -c 'pass'", paired_probe_runner=lambda code: ToolResult(True, "red/green", {
                            "red_green": True, "baseline_unchanged": True,
                            "baseline_outcome": "assertion_failed", "candidate_outcome": "passed",
                            "code_sha256": "hash", "base_commit": "base",
                        }),
                    ), root / "traces", max_steps=8, prompt_variant="dual-checkout-probe-v1",
                ).run("add should add")
                self.assertEqual(result.success, expected)
                events = [json.loads(line) for line in result.trajectory_path.read_text().splitlines()]
                self.assertIn("run_probe_pair", events[0]["initial_messages"][0]["content"])
                if pair in actions:
                    self.assertEqual(result.metrics.paired_probe_successes, 1)

    def test_red_green_requires_same_assertion_before_and_after_edit(self) -> None:
        code = "from maths import add\nassert add(2, 3) == 5"
        model = ScriptedModel([
            {"tool": "run_probe", "arguments": {"code": code}},
            {"tool": "replace_text", "arguments": {
                "path": "maths.py", "old_text": "return a - b", "new_text": "return a + b",
            }},
            {"tool": "run_probe", "arguments": {"code": code}},
            {"tool": "run_tests", "arguments": {}},
            {"tool": "git_diff", "arguments": {}},
            {"final": "Fixed and verified."},
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            source = root / "maths.py"
            source.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

            def probe(_code: str) -> ToolResult:
                if "return a - b" in source.read_text(encoding="utf-8"):
                    return ToolResult(
                        False, "AssertionError", {"outcome": "assertion_failed"}
                    )
                return ToolResult(True, "Probe assertions passed", {"outcome": "passed"})

            result = Agent(
                model,
                RepositoryTools(root, "python -c 'pass'", probe_runner=probe),
                root / "traces", max_steps=6, prompt_variant="red-green-probe-v1",
            ).run("add should add")
            events = [json.loads(line) for line in result.trajectory_path.read_text().splitlines()]

        self.assertTrue(result.success)
        self.assertTrue(result.metrics.red_probe_recorded)
        self.assertTrue(result.metrics.green_probe_passed)
        self.assertEqual(result.metrics.tool_failures, 0)
        hashes = [
            event["code_sha256"] for event in events
            if event["event"] in {"probe_red_recorded", "probe_green_passed"}
        ]
        self.assertEqual(len(hashes), 2)
        self.assertEqual(hashes[0], hashes[1])

    def test_red_green_rejects_changed_post_edit_assertion(self) -> None:
        red_code = "assert False"
        calls: list[str] = []
        model = ScriptedModel([
            {"tool": "run_probe", "arguments": {"code": red_code}},
            {"tool": "replace_text", "arguments": {
                "path": "maths.py", "old_text": "return a - b", "new_text": "return a + b",
            }},
            {"tool": "run_probe", "arguments": {"code": "assert True"}},
            {"tool": "run_tests", "arguments": {}},
            {"tool": "git_diff", "arguments": {}},
            {"final": "Fixed."},
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "maths.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

            def probe(code: str) -> ToolResult:
                calls.append(code)
                return ToolResult(False, "AssertionError", {"outcome": "assertion_failed"})

            result = Agent(
                model,
                RepositoryTools(root, "python -c 'pass'", probe_runner=probe),
                root / "traces", max_steps=6, prompt_variant="red-green-probe-v1",
            ).run("add should add")

        self.assertFalse(result.success)
        self.assertEqual(calls, [red_code])
        self.assertEqual(result.metrics.probe_mismatch_attempts, 1)
        self.assertFalse(result.metrics.green_probe_passed)

    def test_failed_post_edit_probe_prevents_agent_verified_status(self) -> None:
        model = ScriptedModel([
            {"tool": "replace_text", "arguments": {
                "path": "maths.py", "old_text": "return a - b", "new_text": "return a + b",
            }},
            {"tool": "run_tests", "arguments": {}},
            {"tool": "run_probe", "arguments": {"code": "assert False"}},
            {"tool": "git_diff", "arguments": {}},
            {"final": "Fixed."},
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "maths.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            result = Agent(
                model,
                RepositoryTools(
                    root, "python -c 'pass'",
                    probe_runner=lambda code: ToolResult(False, "AssertionError"),
                ),
                root / "traces", max_steps=5, prompt_variant="behavior-probe-v1",
            ).run("add should add")

        self.assertFalse(result.success)
        self.assertTrue(result.metrics.tests_run_after_latest_edit)
        self.assertTrue(result.metrics.probe_attempted_after_latest_edit)
        self.assertFalse(result.metrics.probe_passed_after_latest_edit)

    def test_probe_is_traced_but_cannot_replace_fixed_tests(self) -> None:
        model = ScriptedModel([
            {"tool": "run_probe", "arguments": {"code": "assert 1 == 1"}},
            {"tool": "replace_text", "arguments": {
                "path": "maths.py", "old_text": "return a - b", "new_text": "return a + b",
            }},
            {"tool": "run_probe", "arguments": {"code": "assert 1 == 1"}},
            {"tool": "git_diff", "arguments": {}},
            {"final": "Probe passed."},
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "maths.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            tools = RepositoryTools(
                root, "python -m unittest -q",
                probe_runner=lambda code: ToolResult(True, "Probe assertions passed"),
            )
            result = Agent(
                model, tools, root / "traces", max_steps=5,
                prompt_variant="behavior-probe-v1",
            ).run("add should add")
            events = [json.loads(line) for line in result.trajectory_path.read_text().splitlines()]

        self.assertFalse(result.success)
        self.assertEqual(result.metrics.probe_attempts, 2)
        self.assertEqual(result.metrics.probe_successes, 2)
        self.assertTrue(result.metrics.probe_passed_after_latest_edit)
        self.assertFalse(result.metrics.tests_run_after_latest_edit)
        self.assertIn("run_probe", events[0]["initial_messages"][0]["content"])

    def test_behavior_contract_prompt_is_opt_in_and_recorded(self) -> None:
        class CaptureModel:
            def __init__(self) -> None:
                self.messages: list[dict[str, str]] | None = None

            def complete(self, messages: list[dict[str, str]]) -> str:
                self.messages = messages
                return json.dumps({"final": "No edit attempted."})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            model = CaptureModel()
            result = Agent(
                model,
                RepositoryTools(root, "python -m unittest -q"),
                root / "traces",
                max_steps=1,
                prompt_variant="behavior-contract-v1",
            ).run("Example issue")
            started = json.loads(result.trajectory_path.read_text().splitlines()[0])
            default_model = CaptureModel()
            default_result = Agent(
                default_model,
                RepositoryTools(root, "python -m unittest -q"),
                root / "default-traces",
                max_steps=1,
            ).run("Example issue")
            default_started = json.loads(
                default_result.trajectory_path.read_text().splitlines()[0]
            )

        assert model.messages is not None
        assert default_model.messages is not None
        self.assertIn("short behavior contract", model.messages[0]["content"])
        self.assertEqual(started["prompt_variant"], "behavior-contract-v1")
        self.assertNotIn("short behavior contract", default_model.messages[0]["content"])
        self.assertEqual(default_started["prompt_variant"], "default")

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
