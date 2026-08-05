from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from patchpilot.benchmark import BenchmarkRunner, discover_cases
from patchpilot.tools import RepositoryTools


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks" / "cases"


class ScriptedModel:
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self.outputs = iter(outputs)

    def complete(self, messages: list[dict[str, str]]) -> str:
        return json.dumps(next(self.outputs))


class BenchmarkTest(unittest.TestCase):
    def test_bundled_cases_are_valid_and_reproduce_failures(self) -> None:
        cases = discover_cases(BENCHMARK_DIR)

        self.assertEqual(len(cases), 3)
        self.assertEqual(len({case.id for case in cases}), 3)
        for case in cases:
            with self.subTest(case=case.id):
                result = RepositoryTools(case.repo_dir, case.test_command).run_tests({})
                self.assertFalse(result.ok, f"{case.id} should fail before repair")

    def test_runner_repairs_one_case_and_persists_artifacts(self) -> None:
        patch = """--- a/calculator.py
+++ b/calculator.py
@@ -3 +3 @@
-    return a - b
+    return a + b
"""
        model = ScriptedModel([
            {"tool": "search_code", "arguments": {"query": "def add"}},
            {"tool": "read_file", "arguments": {"path": "calculator.py"}},
            {"tool": "apply_patch", "arguments": {"patch": patch}},
            {"tool": "run_tests", "arguments": {}},
            {"tool": "git_diff", "arguments": {}},
            {"final": "Fixed add and verified the tests."},
        ])
        with tempfile.TemporaryDirectory() as directory:
            runner = BenchmarkRunner(
                model=model,
                model_name="scripted-test-model",
                benchmark_dir=BENCHMARK_DIR,
                results_dir=Path(directory),
                max_steps=10,
            )
            summary, summary_path = runner.run({"arithmetic_operator_001"})

            self.assertEqual(summary["total"], 1)
            self.assertEqual(summary["resolved"], 1, summary)
            self.assertEqual(summary["resolve_rate"], 1.0)
            self.assertEqual(summary["metrics"]["patch_attempts"], 1)
            self.assertEqual(summary["metrics"]["patch_success_rate"], 1.0)
            self.assertEqual(summary["metrics"]["tool_failures"], 0)
            self.assertTrue(summary_path.is_file())
            case_result = summary["cases"][0]
            self.assertTrue(Path(case_result["trajectory_path"]).is_file())
            patch_path = Path(case_result["patch_path"])
            self.assertTrue(patch_path.is_file())
            self.assertIn("return a + b", patch_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
