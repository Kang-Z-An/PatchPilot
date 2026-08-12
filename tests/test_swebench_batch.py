from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from patchpilot.swebench_batch import (
    SWEBenchBatchRunner,
    build_batch_summary,
    load_batch_manifest,
    run_official_evaluation,
    update_summary_with_official,
)


class ScriptedModel:
    def __init__(self) -> None:
        self.outputs = iter([
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

    def complete(self, messages: list[dict[str, str]]) -> str:
        return json.dumps(next(self.outputs))


class FailsOnceModel(ScriptedModel):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self.failed:
            self.failed = True
            raise RuntimeError("Could not reach model API: temporary failure")
        return super().complete(messages)


class NeverCalledModel:
    def complete(self, messages: list[dict[str, str]]) -> str:
        raise AssertionError("resume should not call the model for completed cases")


class SWEBenchBatchTest(unittest.TestCase):
    def test_manifest_rejects_duplicate_ids_and_skips_disabled_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, commit = self._make_repository(root / "source")
            instance = self._write_instance(root, "example__repo-1", commit)
            record = self._record(instance, source, "example__repo-1")
            disabled_instance = self._write_instance(root, "example__repo-2", commit)
            disabled = {
                **record,
                "instance_id": "example__repo-2",
                "instance_file": disabled_instance.name,
                "enabled": False,
            }
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(record) + "\n" + json.dumps(disabled) + "\n",
                encoding="utf-8",
            )

            cases = load_batch_manifest(manifest)

            self.assertEqual(len(cases), 2)
            self.assertEqual(sum(case.enabled for case in cases), 1)
            manifest.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate instance ids"):
                load_batch_manifest(manifest)

    def test_generate_merges_predictions_and_resume_skips_completed_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = self._make_manifest(root)
            run_dir = root / "results" / "batch-test"
            runner = SWEBenchBatchRunner(
                model=ScriptedModel(),
                model_name="scripted-model",
                cases=cases,
                results_dir=root / "results",
                run_id="batch-test",
            )

            summary, summary_path = runner.generate()

            self.assertEqual(summary["total"], 1)
            self.assertEqual(summary["generation_completed"], 1)
            self.assertEqual(summary["patch_generated"], 1)
            self.assertEqual(summary["agent_verified"], 1)
            self.assertEqual(summary["official"]["status"], "pending")
            self.assertEqual(summary["metrics"]["runs_tested_after_latest_edit"], 1)
            self.assertEqual(summary["metrics"]["test_freshness_rate"], 1.0)
            self.assertTrue(summary["cases"][0]["tests_run_after_latest_edit"])
            self.assertEqual(summary["cases"][0]["test_files_modified"], 0)
            self.assertTrue(summary_path.is_file())
            predictions = (run_dir / "predictions.jsonl").read_text(encoding="utf-8")
            self.assertIn("return a + b", predictions)

            resumed = SWEBenchBatchRunner(
                model=NeverCalledModel(),
                model_name="scripted-model",
                cases=cases,
                results_dir=root / "results",
                run_dir=run_dir,
            )
            resumed_summary, _ = resumed.generate(resume=True)

            self.assertEqual(resumed_summary["generation_completed"], 1)
            case_result = json.loads(
                (run_dir / "cases" / "example__repo-1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(case_result["attempt"], 1)

            wrong_model = SWEBenchBatchRunner(
                model=NeverCalledModel(),
                model_name="different-model",
                cases=cases,
                results_dir=root / "results",
                run_dir=run_dir,
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                wrong_model.generate(resume=True)

            Path(case_result["predictions_path"]).unlink()
            recovered = SWEBenchBatchRunner(
                model=ScriptedModel(),
                model_name="scripted-model",
                cases=cases,
                results_dir=root / "results",
                run_dir=run_dir,
            )
            recovered.generate(resume=True)
            recovered_result = json.loads(
                (run_dir / "cases" / "example__repo-1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(recovered_result["attempt"], 2)

    def test_one_generation_error_does_not_abort_later_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = self._make_manifest(root, count=2)
            runner = SWEBenchBatchRunner(
                model=FailsOnceModel(),
                model_name="scripted-model",
                cases=cases,
                results_dir=root / "results",
                run_id="failure-isolation",
            )

            summary, _ = runner.generate()

            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["generation_errors"], 1)
            self.assertEqual(summary["generation_completed"], 1)
            self.assertEqual(summary["patch_generated"], 1)
            first = json.loads(
                (runner.run_dir / "cases" / "example__repo-1.json").read_text()
            )
            self.assertEqual(first["error_type"], "model_api_error")

    def test_official_report_updates_unified_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = self._make_manifest(root)
            runner = SWEBenchBatchRunner(
                model=ScriptedModel(),
                model_name="scripted-model",
                cases=cases,
                results_dir=root / "results",
                run_id="official-test",
            )
            runner.generate()

            def fake_run(command, **kwargs):
                cwd = Path(kwargs["cwd"])
                run_id = command[command.index("--run_id") + 1]
                report = {
                    "resolved_ids": ["example__repo-1"],
                    "unresolved_ids": [],
                    "error_ids": [],
                    "empty_patch_ids": [],
                    "incomplete_ids": [],
                }
                (cwd / f"scripted-model.{run_id}.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
                return subprocess.CompletedProcess(command, 0, "resolved", "")

            with patch("patchpilot.swebench_batch.subprocess.run", side_effect=fake_run):
                official = run_official_evaluation(runner.run_dir, cases)
            summary, summary_path = update_summary_with_official(
                runner.run_dir,
                "scripted-model",
                cases,
                official,
            )

            self.assertEqual(official.status, "completed")
            self.assertEqual(summary["official"]["resolved"], 1)
            self.assertEqual(summary["official"]["unresolved"], 0)
            self.assertEqual(summary["official"]["resolve_rate"], 1.0)
            self.assertEqual(summary["cases"][0]["official_status"], "resolved")
            self.assertTrue(summary_path.is_file())

    def test_empty_patch_counts_as_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = self._make_manifest(root)
            summary = build_batch_summary(
                "empty-patch-test",
                "scripted-model",
                cases,
                [],
                root / "predictions.jsonl",
                official_report={
                    "resolved_ids": [],
                    "unresolved_ids": [],
                    "error_ids": [],
                    "empty_patch_ids": ["example__repo-1"],
                    "incomplete_ids": [],
                },
            )

            self.assertEqual(summary["official"]["resolved"], 0)
            self.assertEqual(summary["official"]["unresolved"], 1)
            self.assertEqual(summary["official"]["empty_patches"], 1)
            self.assertEqual(summary["official"]["resolve_rate"], 0.0)
            self.assertEqual(summary["cases"][0]["official_status"], "empty_patch")

    def test_manifest_protects_test_files_by_default_and_allows_opt_out(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, commit = self._make_repository(root / "source")
            first = self._write_instance(root, "example__repo-1", commit)
            second = self._write_instance(root, "example__repo-2", commit)
            records = [
                self._record(first, source, "example__repo-1"),
                {
                    **self._record(second, source, "example__repo-2"),
                    "protect_test_files": False,
                },
            ]
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            cases = load_batch_manifest(manifest)

        self.assertTrue(cases[0].protect_test_files)
        self.assertFalse(cases[1].protect_test_files)
        self.assertTrue(cases[0].snapshot()["protect_test_files"])

    def _make_manifest(self, root: Path, count: int = 1):
        source, commit = self._make_repository(root / "source")
        records = []
        for number in range(1, count + 1):
            instance_id = f"example__repo-{number}"
            instance = self._write_instance(root, instance_id, commit)
            records.append(self._record(instance, source, instance_id))
        manifest = root / "manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return load_batch_manifest(manifest)

    @staticmethod
    def _record(instance: Path, source: Path, instance_id: str):
        return {
            "schema_version": 1,
            "instance_id": instance_id,
            "instance_file": instance.name,
            "source_repo": source.name,
            "test_command": "python -m unittest -q",
            "max_steps": 8,
            "timeout": 30,
            "enabled": True,
        }

    @staticmethod
    def _write_instance(root: Path, instance_id: str, commit: str) -> Path:
        path = root / f"{instance_id}.json"
        path.write_text(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "repo": "example/repo",
                    "base_commit": commit,
                    "problem_statement": "add subtracts its second argument",
                }
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def _make_repository(cls, root: Path) -> tuple[Path, str]:
        root.mkdir()
        (root / "calculator.py").write_text(
            "def add(a, b):\n    return a - b\n", encoding="utf-8"
        )
        (root / "test_calculator.py").write_text(
            "import unittest\n"
            "from calculator import add\n\n"
            "class CalculatorTest(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(2, 3), 5)\n",
            encoding="utf-8",
        )
        cls._git(root, "init", "-q")
        cls._git(root, "add", "--all")
        cls._git(
            root,
            "-c",
            "user.name=PatchPilot Test",
            "-c",
            "user.email=test@patchpilot.local",
            "commit",
            "-qm",
            "buggy baseline",
        )
        commit = cls._git(root, "rev-parse", "HEAD").stdout.strip()
        return root, commit

    @staticmethod
    def _git(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(result.stderr or result.stdout)
        return result


if __name__ == "__main__":
    unittest.main()
