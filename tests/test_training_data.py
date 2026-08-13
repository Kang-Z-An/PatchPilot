from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from patchpilot.training_data import ExportConfig, export_sft_dataset, split_for_task


class TrainingDataExportTest(unittest.TestCase):
    def test_exports_exact_verified_conversations_and_groups_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_exact(root / "one.jsonl", "task-a", "first")
            self._write_exact(root / "two.jsonl", "task-a", "second")
            self._write_exact(root / "three.jsonl", "task-b", "third")

            report = export_sft_dataset([root], root / "dataset")

            records = self._records(root / "dataset")
            self.assertEqual(report["accepted_samples"], 3)
            self.assertEqual(report["unique_tasks"], 2)
            self.assertEqual({record["metadata"]["fidelity"] for record in records}, {"exact"})
            task_a_splits = {
                record["split"] for record in records if record["task_id"] == "task-a"
            }
            self.assertEqual(len(task_a_splits), 1)
            self.assertNotIn(str(root), json.dumps(records))
            self.assertEqual(
                split_for_task("task-a", ExportConfig()),
                split_for_task("task-a", ExportConfig()),
            )

    def test_strict_export_rejects_legacy_but_opt_in_labels_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "artifacts" / "example__repo-1" / "attempt"
            trace = run / "trajectory" / "legacy.jsonl"
            self._write_legacy(trace)
            (run / "model.patch").write_text(
                "--- a/source.py\n+++ b/source.py\n@@ -1 +1 @@\n-old\n+new\n",
                encoding="utf-8",
            )

            strict = export_sft_dataset([root], root / "strict")
            permissive = export_sft_dataset(
                [root],
                root / "legacy",
                ExportConfig(allow_legacy=True),
            )

            self.assertEqual(strict["accepted_samples"], 0)
            self.assertEqual(strict["rejection_reasons"]["legacy_not_allowed"], 1)
            self.assertEqual(permissive["accepted_samples"], 1)
            record = self._records(root / "legacy")[0]
            self.assertEqual(record["metadata"]["fidelity"], "legacy_approximate")
            self.assertEqual(record["task_id"], "example__repo-1")

    def test_rejects_official_failure_test_edits_and_possible_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unresolved = root / "unresolved"
            self._write_exact(
                unresolved / "trajectory" / "trace.jsonl",
                "task-unresolved",
                "unresolved",
            )
            self._write_json(
                unresolved / "summary.json",
                {
                    "cases": [
                        {
                            "instance_id": "task-unresolved",
                            "official_status": "unresolved",
                        }
                    ]
                },
            )
            self._write_exact(
                root / "test-edit.jsonl",
                "task-test-edit",
                "edit",
                test_files_modified=1,
            )
            self._write_exact(
                root / "secret.jsonl",
                "task-secret",
                "PATCHPILOT_API_KEY=sk-this-is-a-secret-value",
            )

            report = export_sft_dataset([root], root / "dataset")

            self.assertEqual(report["accepted_samples"], 0)
            self.assertEqual(report["rejection_reasons"]["official_unresolved"], 1)
            self.assertEqual(report["rejection_reasons"]["test_files_modified"], 1)
            self.assertEqual(report["rejection_reasons"]["possible_secret"], 1)

    @staticmethod
    def _write_exact(
        path: Path,
        task_id: str,
        marker: str,
        *,
        test_files_modified: int = 0,
    ) -> None:
        initial = [
            {"role": "system", "content": "Repair code with tools."},
            {"role": "user", "content": f"Issue: {marker}"},
        ]
        observation = json.dumps(
            {"tool": "run_tests", "ok": True, "output": f"passed {marker}"}
        )
        events = [
            {
                "event": "run_started",
                "trajectory_schema_version": 2,
                "task_id": task_id,
                "source": "unit_test",
                "model": "scripted",
                "repository": str(path.parent),
                "issue": marker,
                "initial_messages": initial,
            },
            {
                "event": "model_output",
                "step": 1,
                "raw": '{"tool":"run_tests","arguments":{}}',
            },
            {
                "event": "tool_result",
                "step": 1,
                "tool": "run_tests",
                "ok": True,
                "output": f"passed {marker}",
                "observation": observation,
            },
            {
                "event": "model_output",
                "step": 2,
                "raw": json.dumps({"final": f"fixed {marker}"}),
            },
            {
                "event": "run_finished",
                "step": 2,
                "verified": True,
                "verification": {
                    "edit_applied": True,
                    "tests_passed": True,
                    "tests_run_after_latest_edit": True,
                    "diff_inspected": True,
                    "non_empty_diff": True,
                },
                "metrics": {
                    "tool_calls": 1,
                    "tool_failures": 0,
                    "tool_failure_rate": 0.0,
                    "invalid_actions": 0,
                    "test_files_modified": test_files_modified,
                },
            },
        ]
        TrainingDataExportTest._write_events(path, events)

    @staticmethod
    def _write_legacy(path: Path) -> None:
        events = [
            {"event": "run_started", "issue": "fix source", "repository": "/tmp/repo"},
            {
                "event": "model_output",
                "step": 1,
                "raw": '{"tool":"replace_text","arguments":{}}',
            },
            {
                "event": "tool_result",
                "step": 1,
                "tool": "replace_text",
                "ok": True,
                "output": "updated",
            },
            {"event": "model_output", "step": 2, "raw": '{"final":"fixed"}'},
            {
                "event": "run_finished",
                "step": 2,
                "verified": True,
                "metrics": {
                    "tool_calls": 1,
                    "tool_failures": 0,
                    "invalid_actions": 0,
                },
            },
        ]
        TrainingDataExportTest._write_events(path, events)

    @staticmethod
    def _write_events(path: Path, events: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )

    @staticmethod
    def _write_json(path: Path, value: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def _records(output_dir: Path) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for split in ("train", "dev", "test"):
            text = (output_dir / f"{split}.jsonl").read_text(encoding="utf-8")
            records.extend(json.loads(line) for line in text.splitlines() if line)
        return records


if __name__ == "__main__":
    unittest.main()
