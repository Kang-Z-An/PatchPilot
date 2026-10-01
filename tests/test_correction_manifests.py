from __future__ import annotations

import json
import unittest
from pathlib import Path


class CorrectionManifestTest(unittest.TestCase):
    def test_smoke_and_remaining_cases_partition_frozen_corrections(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "training_pool_v1"

        def records(name: str) -> list[dict]:
            return [
                json.loads(line)
                for line in (root / name).read_text(encoding="utf-8").splitlines()
            ]

        original = records("manifest_train_corrections.jsonl")
        smoke = records("manifest_train_correction_smoke.jsonl")
        remaining = records("manifest_train_corrections_remaining8.jsonl")

        self.assertEqual(len(original), 9)
        self.assertEqual(len(smoke), 1)
        self.assertEqual(len(remaining), 8)
        self.assertEqual(
            sorted(original, key=lambda row: row["instance_id"]),
            sorted(smoke + remaining, key=lambda row: row["instance_id"]),
        )
        self.assertEqual(
            len({row["instance_id"] for row in smoke + remaining}), 9
        )


if __name__ == "__main__":
    unittest.main()
