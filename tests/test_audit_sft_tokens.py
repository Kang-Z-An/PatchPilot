from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from patchpilot.audit_sft_tokens import assistant_content_mask, audit_dataset


class FakeTokenizer:
    chat_template = "simple-prefix-stable-template"
    model_max_length = 200

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt=False):
        rendered = "".join(
            f"<{message['role']}>{message['content']}</end>" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return list(rendered.encode("utf-8"))


class UnstableTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt=False):
        ids = super().apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )
        return [len(messages), *ids]


class SFTTokenAuditTest(unittest.TestCase):
    def test_masks_system_user_observation_and_assistant_header(self) -> None:
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "bug"},
            {"role": "assistant", "content": '{"tool":"read"}'},
            {"role": "user", "content": '{"tool":"read","output":"secret"}'},
            {"role": "assistant", "content": '{"final":"done"}'},
        ]
        ids, mask, contexts = assistant_content_mask(FakeTokenizer(), messages)
        supervised = bytes(value for value, included in zip(ids, mask) if included).decode()

        self.assertEqual(supervised, '{"tool":"read"}</end>{"final":"done"}</end>')
        self.assertNotIn("secret", supervised)
        self.assertNotIn("<assistant>", supervised)
        self.assertEqual(len(contexts), 2)

    def test_rejects_non_prefix_stable_template(self) -> None:
        messages = [
            {"role": "user", "content": "issue"},
            {"role": "assistant", "content": "answer"},
        ]
        with self.assertRaisesRegex(ValueError, "prefix"):
            assistant_content_mask(UnstableTokenizer(), messages)

    def test_report_has_counts_without_recording_chat_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "task_id": "example-1",
                        "sample_id": "sample-1",
                        "messages": [
                            {"role": "user", "content": "private observation"},
                            {"role": "assistant", "content": "fixed"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = audit_dataset(
                path, FakeTokenizer(), model_name="fake", revision="test", version="test"
            )

        self.assertEqual(report["sample_count"], 1)
        self.assertTrue(report["all_fit_tokenizer_context"])
        self.assertNotIn("private observation", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
