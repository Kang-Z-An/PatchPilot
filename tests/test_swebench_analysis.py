from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from patchpilot.swebench_analysis import analyze_swebench_run, write_analysis


class SWEBenchAnalysisTest(unittest.TestCase):
    def test_audit_preserves_official_score_and_confirms_gold_failure_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._make_run(Path(directory), include_gold=True)

            analysis = analyze_swebench_run(run_dir)

            self.assertEqual(analysis["official"]["resolved"], 1)
            self.assertEqual(analysis["official"]["resolve_rate"], 0.5)
            self.assertNotIn("adjusted_resolve_rate", analysis["official"])
            audit = analysis["audit"]
            self.assertEqual(audit["instances_with_all_target_tests_passed"], 2)
            self.assertEqual(audit["total_regression_test_failures"], 2)
            self.assertEqual(audit["instances_with_test_file_changes"], 1)
            self.assertEqual(
                audit["environment_suspect_instances"], ["example__repo-2"]
            )
            suspect = analysis["cases"][1]
            self.assertEqual(
                suspect["gold_control"]["status"], "same_regression_failures"
            )
            self.assertTrue(suspect["environment_suspect"])
            self.assertEqual(suspect["test_files_modified"], ["tests/test_bug.py"])

            json_path, markdown_path = write_analysis(analysis, run_dir / "analysis")

            self.assertTrue(json_path.is_file())
            report = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Official result: 1/2 resolved (0.5)", report)
            self.assertIn("`example__repo-2`", report)

    def test_target_pass_with_regression_failure_is_unconfirmed_without_gold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._make_run(Path(directory), include_gold=False)

            analysis = analyze_swebench_run(run_dir)

        self.assertEqual(analysis["audit"]["environment_suspect_instances"], [])
        self.assertEqual(
            analysis["audit"]["unconfirmed_environment_candidates"],
            ["example__repo-2"],
        )
        candidate = analysis["cases"][1]
        self.assertTrue(candidate["environment_suspect_candidate"])
        self.assertFalse(candidate["environment_suspect"])
        self.assertEqual(candidate["gold_control"]["status"], "not_available")

    def test_malformed_detailed_report_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._make_run(Path(directory), include_gold=False)
            path = (
                run_dir
                / "official"
                / "logs"
                / "run_evaluation"
                / "official-test"
                / "test-model"
                / "example__repo-1"
                / "report.json"
            )
            path.write_text(
                json.dumps({"example__repo-1": {"tests_status": {}}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "FAIL_TO_PASS"):
                analyze_swebench_run(run_dir)

    @classmethod
    def _make_run(cls, root: Path, *, include_gold: bool) -> Path:
        run_dir = root / "run"
        (run_dir / "cases").mkdir(parents=True)
        primary_root = (
            run_dir
            / "official"
            / "logs"
            / "run_evaluation"
            / "official-test"
            / "test-model"
        )
        summary = {
            "schema_version": 1,
            "run_id": "analysis-test",
            "model": "test-model",
            "total": 2,
            "official": {
                "status": "completed",
                "resolved": 1,
                "unresolved": 1,
                "errors": 0,
                "resolve_rate": 0.5,
            },
            "cases": [
                {
                    "instance_id": "example__repo-1",
                    "official_status": "resolved",
                },
                {
                    "instance_id": "example__repo-2",
                    "official_status": "unresolved",
                },
            ],
        }
        cls._write_json(run_dir / "summary.json", summary)
        cls._write_json(
            run_dir / "official" / "result.json",
            {"status": "completed", "run_id": "official-test"},
        )
        cls._write_instance_report(
            primary_root / "example__repo-1" / "report.json",
            "example__repo-1",
            resolved=True,
            regression_failures=[],
        )
        failures = ["tests/test_timeout.py::test_a", "tests/test_timeout.py::test_b"]
        cls._write_instance_report(
            primary_root / "example__repo-2" / "report.json",
            "example__repo-2",
            resolved=False,
            regression_failures=failures,
        )
        cls._write_patch(
            run_dir,
            "example__repo-1",
            ["src/feature.py"],
        )
        cls._write_patch(
            run_dir,
            "example__repo-2",
            ["src/bug.py", "tests/test_bug.py"],
        )
        if include_gold:
            gold_path = (
                run_dir
                / "official"
                / "logs"
                / "run_evaluation"
                / "gold-control"
                / "gold"
                / "example__repo-2"
                / "report.json"
            )
            cls._write_instance_report(
                gold_path,
                "example__repo-2",
                resolved=False,
                regression_failures=failures,
            )
        return run_dir

    @classmethod
    def _write_patch(cls, run_dir: Path, instance_id: str, paths: list[str]) -> None:
        artifact = run_dir / "artifacts" / instance_id / "attempt" / "model.patch"
        artifact.parent.mkdir(parents=True)
        sections = []
        for path in paths:
            sections.append(
                f"diff --git a/{path} b/{path}\n"
                f"--- a/{path}\n"
                f"+++ b/{path}\n"
                "@@ -1 +1 @@\n-old\n+new\n"
            )
        artifact.write_text("".join(sections), encoding="utf-8")
        cls._write_json(
            run_dir / "cases" / f"{instance_id}.json",
            {"instance_id": instance_id, "patch_path": str(artifact)},
        )

    @staticmethod
    def _write_instance_report(
        path: Path,
        instance_id: str,
        *,
        resolved: bool,
        regression_failures: list[str],
    ) -> None:
        report = {
            instance_id: {
                "patch_successfully_applied": True,
                "resolved": resolved,
                "tests_status": {
                    "FAIL_TO_PASS": {"success": ["tests/test_bug.py::test_fix"], "failure": []},
                    "PASS_TO_PASS": {
                        "success": ["tests/test_existing.py::test_ok"],
                        "failure": regression_failures,
                    },
                },
            }
        }
        SWEBenchAnalysisTest._write_json(path, report)

    @staticmethod
    def _write_json(path: Path, value: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
