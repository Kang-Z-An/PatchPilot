from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from patchpilot.preference_data import export_preference_dataset


class PreferenceDataExportTest(unittest.TestCase):
    def test_exports_one_exact_official_train_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rejected = root / "rejected"
            chosen = root / "chosen"
            self._write_candidate(rejected, "task-a", "unresolved", "original")
            self._write_candidate(chosen, "task-a", "resolved", "corrected")

            report = export_preference_dataset(
                [rejected],
                [chosen],
                root / "dataset",
            )

            records = self._records(root / "dataset" / "train.jsonl")
            self.assertEqual(report["preference_pairs"], 1)
            self.assertEqual(report["paired_task_ids"], ["task-a"])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["task_id"], "task-a")
            self.assertEqual(records[0]["split"], "train")
            self.assertEqual(
                records[0]["metadata"]["chosen"]["official_status"],
                "resolved",
            )
            self.assertEqual(
                records[0]["metadata"]["rejected"]["official_status"],
                "unresolved",
            )
            self.assertEqual(records[0]["prompt"][0]["role"], "system")
            self.assertEqual(records[0]["chosen"][-1]["role"], "assistant")
            self.assertNotEqual(records[0]["chosen"], records[0]["rejected"])
            self.assertNotIn(str(root), json.dumps(records))

    def test_requires_explicit_official_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rejected = root / "rejected"
            chosen = root / "chosen"
            self._write_candidate(rejected, "task-a", None, "original")
            self._write_candidate(chosen, "task-a", "resolved", "corrected")

            report = export_preference_dataset(
                [rejected],
                [chosen],
                root / "dataset",
            )

            self.assertEqual(report["preference_pairs"], 0)
            self.assertEqual(
                report["candidate_rejection_reasons"]["rejected"],
                {"official_status_missing": 1},
            )

    def test_rejects_non_train_and_hash_fallback_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rejected_dev = root / "rejected-dev"
            rejected_hash = root / "rejected-hash"
            chosen = root / "chosen"
            self._write_candidate(
                rejected_dev,
                "task-dev",
                "unresolved",
                "dev",
                pool_split="dev",
            )
            self._write_candidate(
                rejected_hash,
                "task-hash",
                "unresolved",
                "hash",
                pool_split=None,
            )
            self._write_candidate(chosen, "task-dev", "resolved", "chosen dev")

            report = export_preference_dataset(
                [rejected_dev, rejected_hash],
                [chosen],
                root / "dataset",
            )

            self.assertEqual(report["preference_pairs"], 0)
            self.assertEqual(
                report["candidate_rejection_reasons"]["rejected"],
                {"non_train_split_dev": 1, "preassigned_split_missing": 1},
            )

    def test_rejects_prompt_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rejected = root / "rejected"
            chosen = root / "chosen"
            self._write_candidate(
                rejected,
                "task-a",
                "unresolved",
                "original",
                issue="Fix behavior A",
            )
            self._write_candidate(
                chosen,
                "task-a",
                "resolved",
                "corrected",
                issue="Fix behavior B",
            )

            report = export_preference_dataset(
                [rejected],
                [chosen],
                root / "dataset",
            )

            self.assertEqual(report["preference_pairs"], 0)
            self.assertEqual(report["pair_rejection_reasons"], {"prompt_mismatch": 1})

    def test_refuses_ambiguous_multiple_chosen_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rejected = root / "rejected"
            chosen_one = root / "chosen-one"
            chosen_two = root / "chosen-two"
            self._write_candidate(rejected, "task-a", "unresolved", "original")
            self._write_candidate(chosen_one, "task-a", "resolved", "corrected one")
            self._write_candidate(chosen_two, "task-a", "resolved", "corrected two")

            report = export_preference_dataset(
                [rejected],
                [chosen_one, chosen_two],
                root / "dataset",
            )

            self.assertEqual(report["preference_pairs"], 0)
            self.assertEqual(
                report["pair_rejection_reasons"],
                {"ambiguous_chosen_candidates": 1},
            )

    @staticmethod
    def _write_candidate(
        root: Path,
        task_id: str,
        official_status: str | None,
        outcome: str,
        *,
        pool_split: str | None = "train",
        issue: str = "Fix the shared issue",
    ) -> None:
        trace = root / "artifacts" / task_id / "attempt" / "trajectory" / "trace.jsonl"
        initial_messages = [
            {"role": "system", "content": "Repair code with tools."},
            {"role": "user", "content": f"Issue:\n{issue}"},
        ]
        run_started: dict[str, object] = {
            "event": "run_started",
            "trajectory_schema_version": 2,
            "task_id": task_id,
            "source": "swebench",
            "model": "scripted",
            "repository": str(root / "repo"),
            "issue": issue,
            "initial_messages": initial_messages,
        }
        if pool_split is not None:
            run_started["pool_split"] = pool_split
        observation = json.dumps(
            {
                "tool": "run_tests",
                "ok": True,
                "output": f"tests passed in {root / 'repo'}",
            }
        )
        events = [
            run_started,
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
                "output": "tests passed",
                "observation": observation,
            },
            {
                "event": "model_output",
                "step": 2,
                "raw": json.dumps({"final": outcome}),
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
                    "test_files_modified": 0,
                },
            },
        ]
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )
        cases: list[dict[str, str]] = []
        if official_status is not None:
            cases.append(
                {
                    "instance_id": task_id,
                    "official_status": official_status,
                }
            )
        (root / "summary.json").write_text(
            json.dumps({"cases": cases}),
            encoding="utf-8",
        )

    @staticmethod
    def _records(path: Path) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]


if __name__ == "__main__":
    unittest.main()
