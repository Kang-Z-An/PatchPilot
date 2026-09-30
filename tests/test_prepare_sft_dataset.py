from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from patchpilot.prepare_sft_dataset import IGNORE_INDEX, build_dataset, tokenize_record
from tests.test_audit_sft_tokens import FakeTokenizer


def valid_record() -> dict:
    return {
        "schema_version": 1,
        "sample_id": "sample-1",
        "task_id": "task-1",
        "split": "train",
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "bug"},
            {"role": "assistant", "content": '{"tool":"read"}'},
            {"role": "user", "content": '{"output":"private observation"}'},
            {"role": "assistant", "content": '{"final":"fixed"}'},
        ],
        "metadata": {
            "fidelity": "exact",
            "verified": True,
            "official_status": "resolved",
            "truncated_observations": 0,
            "split_assignment": "preassigned",
        },
    }


class PrepareSFTDatasetTest(unittest.TestCase):
    def test_assistant_only_labels_and_full_attention(self) -> None:
        example = tokenize_record(valid_record(), FakeTokenizer(), max_seq_length=200)
        ids = example["input_ids"]
        labels = example["labels"]
        supervised = bytes(label for label in labels if label != IGNORE_INDEX).decode()

        self.assertEqual(len(ids), len(labels))
        self.assertEqual(example["attention_mask"], [1] * len(ids))
        self.assertEqual(
            supervised,
            '{"tool":"read"}</end>{"final":"fixed"}</end>',
        )
        self.assertNotIn("private observation", supervised)
        self.assertTrue(all(label in {token, IGNORE_INDEX} for token, label in zip(ids, labels)))

    def test_rejects_nontrain_unverified_unresolved_or_truncated(self) -> None:
        for field, value, nested in (
            ("split", "dev", False),
            ("verified", False, True),
            ("official_status", "unresolved", True),
            ("truncated_observations", 1, True),
            ("split_assignment", "hashed", True),
        ):
            with self.subTest(field=field):
                record = valid_record()
                target = record["metadata"] if nested else record
                target[field] = value
                with self.assertRaisesRegex(ValueError, "preassigned train task"):
                    tokenize_record(record, FakeTokenizer(), max_seq_length=200)

    def test_overlength_is_rejected_without_writing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "train.jsonl"
            output_path = Path(directory) / "train.tokens.jsonl"
            input_path.write_text(json.dumps(valid_record()) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no truncation was applied"):
                self._build(input_path, output_path, max_seq_length=40)
            self.assertFalse(output_path.exists())

    def test_build_report_and_written_labels_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "train.jsonl"
            output_path = Path(directory) / "train.tokens.jsonl"
            input_path.write_text(json.dumps(valid_record()) + "\n", encoding="utf-8")
            report = self._build(input_path, output_path)
            example = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(report["samples"], 1)
        self.assertEqual(report["total_tokens"], len(example["input_ids"]))
        self.assertEqual(
            report["assistant_label_tokens"],
            sum(label != IGNORE_INDEX for label in example["labels"]),
        )
        self.assertEqual(report["overlength_policy"], "reject_without_truncation")
        self.assertTrue(report["no_empty_label_samples"])

    def test_duplicate_tasks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "train.jsonl"
            output_path = Path(directory) / "train.tokens.jsonl"
            first = valid_record()
            second = valid_record()
            second["sample_id"] = "sample-2"
            input_path.write_text(
                json.dumps(first) + "\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate task_id"):
                self._build(input_path, output_path)
            self.assertFalse(output_path.exists())

    def test_refuses_to_overwrite_source_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "train.jsonl"
            original = json.dumps(valid_record()) + "\n"
            input_path.write_text(original, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different files"):
                self._build(input_path, input_path)
            self.assertEqual(input_path.read_text(encoding="utf-8"), original)

    @staticmethod
    def _build(input_path: Path, output_path: Path, max_seq_length: int = 200) -> dict:
        return build_dataset(
            input_path,
            output_path,
            FakeTokenizer(),
            max_seq_length=max_seq_length,
            tokenizer_name="fake",
            tokenizer_revision="test",
            transformers_version="test",
        )


if __name__ == "__main__":
    unittest.main()
