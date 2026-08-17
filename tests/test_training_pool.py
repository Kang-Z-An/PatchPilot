from __future__ import annotations

import json
import shlex
import tempfile
import unittest
from pathlib import Path

from patchpilot.training_pool import (
    QUALIFIED_IMAGE_TAGS,
    TrainingPoolConfig,
    build_agent_test_command,
    discover_reserved_instance_ids,
    select_training_pool,
    write_training_pool,
)


class TrainingPoolTest(unittest.TestCase):
    def test_public_test_command_uses_the_existing_test_file(self) -> None:
        command, policy = build_agent_test_command(
            instance_id="example__repo-1",
            fail_to_pass=["tests/test_bug.py::test_value[new]"],
            pass_to_pass=[
                'tests/test_bug.py::test_value["quoted]',
                "tests/test_bug.py::test_value[safe]",
            ],
        )

        self.assertNotIn("quoted", command)
        self.assertNotIn("safe", command)
        self.assertIn("tests/test_bug.py", shlex.split(command))
        self.assertEqual(policy, "public_regression_pass_to_pass_file")

    def test_balanced_deterministic_selection_and_split_isolation(self) -> None:
        records = [
            self._record(repo, index)
            for repo in ("example/alpha", "example/beta")
            for index in range(8)
        ]
        config = TrainingPoolConfig()

        first, report = select_training_pool(records, set(), config)
        second, _ = select_training_pool(list(reversed(records)), set(), config)

        self.assertEqual(first, second)
        self.assertEqual(report["selected_records"], 12)
        self.assertEqual(report["split_counts"], {"dev": 2, "test": 2, "train": 8})
        for repo in ("example/alpha", "example/beta"):
            counts = report["repository_split_counts"][repo]
            self.assertEqual(counts, {"dev": 1, "test": 1, "train": 4})
        self.assertEqual(len({item["instance_id"] for item in first}), len(first))

    def test_filters_reserved_test_modifying_and_oversized_candidates(self) -> None:
        records = [self._record("example/alpha", index) for index in range(8)]
        records[0]["patch"] = self._patch("tests/test_bug.py")
        records[1]["FAIL_TO_PASS"] = "[]"
        records[2]["patch"] = self._patch("README.md")

        selected, report = select_training_pool(
            records,
            {records[3]["instance_id"]},
            TrainingPoolConfig(examples_per_repo=4, train_per_repo=2),
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(report["rejection_reasons"]["gold_patch_modifies_tests"], 1)
        self.assertEqual(report["rejection_reasons"]["missing_target_tests"], 1)
        self.assertEqual(report["rejection_reasons"]["non_python_patch"], 1)
        self.assertEqual(report["rejection_reasons"]["reserved_instance"], 1)

    def test_excludes_environment_invalid_repository(self) -> None:
        records = [
            *[self._record("example/valid", index) for index in range(6)],
            *[self._record("example/invalid", index) for index in range(6)],
        ]

        selected, report = select_training_pool(
            records,
            set(),
            TrainingPoolConfig(excluded_repositories=("example/invalid",)),
        )

        self.assertEqual({item["repo"] for item in selected}, {"example/valid"})
        self.assertEqual(report["rejection_reasons"]["excluded_repository"], 6)

    def test_writes_sanitized_instances_manifest_and_selection(self) -> None:
        records = [self._record("example/alpha", index) for index in range(6)]
        config = TrainingPoolConfig()
        selected, report = select_training_pool(records, set(), config)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pool"
            (output / "instances").mkdir(parents=True)
            (output / "instances" / "stale.json").write_text("{}", encoding="utf-8")

            paths = write_training_pool(
                selected,
                report,
                output,
                dataset_name="example/dataset",
                dataset_revision="abc123",
                source_split="dev",
                config=config,
                exclusion_metadata={"verified_dataset": {"instance_ids": 500}},
            )

            instance_id = selected[0]["instance_id"]
            instance = json.loads(
                (output / "instances" / f"{instance_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn("patch", instance)
            self.assertNotIn("test_patch", instance)
            self.assertFalse((output / "instances" / "stale.json").exists())
            self.assertEqual(instance["source"]["revision"], "abc123")
            manifests = {
                split: [
                    json.loads(line)
                    for line in paths[f"manifest_{split}"]
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                for split in ("train", "dev", "test")
            }
            self.assertEqual(
                {split: len(values) for split, values in manifests.items()},
                {"train": 4, "dev": 1, "test": 1},
            )
            manifest = [item for values in manifests.values() for item in values]
            self.assertTrue(all(item["protect_test_files"] for item in manifest))
            smoke = [
                json.loads(line)
                for line in paths["manifest_smoke"]
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(smoke), 1)
            self.assertEqual(smoke[0]["pool_split"], "train")
            self.assertTrue(
                all(item["test_command"].startswith("python -m pytest -q ") for item in manifest)
            )
            for item in manifest:
                written_instance = json.loads(
                    (output / item["instance_file"]).read_text(encoding="utf-8")
                )
                command = shlex.split(item["test_command"])
                self.assertNotIn(written_instance["FAIL_TO_PASS"][0], command)
                self.assertIn("tests/test_feature.py", command)
                self.assertEqual(
                    item["test_policy"], "public_regression_pass_to_pass_file"
                )
            selection_text = paths["selection"].read_text(encoding="utf-8")
            self.assertNotIn("gold answer", selection_text)
            self.assertIn('"instance_ids": 500', selection_text)

    def test_uses_explicit_public_file_fallback_without_pass_to_pass(self) -> None:
        records = [self._record("pydicom/pydicom", index) for index in range(6)]
        records[0]["instance_id"] = "pydicom__pydicom-901"
        records[0]["PASS_TO_PASS"] = "[]"
        config = TrainingPoolConfig()
        selected, report = select_training_pool(records, set(), config)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_training_pool(
                selected,
                report,
                Path(directory) / "pool",
                dataset_name="example/dataset",
                dataset_revision="abc123",
                source_split="dev",
                config=config,
            )
            manifests = [
                json.loads(line)
                for split in ("train", "dev", "test")
                for line in paths[f"manifest_{split}"].read_text(encoding="utf-8").splitlines()
            ]

        record = next(
            item for item in manifests if item["instance_id"] == "pydicom__pydicom-901"
        )
        self.assertEqual(
            record["test_command"],
            "python -m pytest -q pydicom/tests/test_uid.py",
        )
        self.assertEqual(
            record["test_policy"], "public_regression_explicit_fallback"
        )

    def test_writes_qualified_image_tag_for_adapted_instance(self) -> None:
        instance_id = "pylint-dev__astroid-1959"
        records = [self._record("pylint-dev/astroid", index) for index in range(6)]
        records[0]["instance_id"] = instance_id
        config = TrainingPoolConfig()
        selected, report = select_training_pool(records, set(), config)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_training_pool(
                selected,
                report,
                Path(directory) / "pool",
                dataset_name="example/dataset",
                dataset_revision="abc123",
                source_split="dev",
                config=config,
            )
            manifests = [
                json.loads(line)
                for split in ("train", "dev", "test")
                for line in paths[f"manifest_{split}"]
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        record = next(item for item in manifests if item["instance_id"] == instance_id)
        self.assertEqual(
            record["docker_image"],
            f"sweb.eval.x86_64.{instance_id}:{QUALIFIED_IMAGE_TAGS[instance_id]}",
        )

    def test_discovers_nested_reserved_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.json").write_text(
                json.dumps(
                    {
                        "instance_id": "task-one",
                        "cases": [{"instance_id": "task-two"}],
                    }
                ),
                encoding="utf-8",
            )

            reserved = discover_reserved_instance_ids([root])

        self.assertEqual(reserved, {"task-one", "task-two"})

    @staticmethod
    def _record(repo: str, index: int) -> dict[str, str]:
        slug = repo.replace("/", "__")
        return {
            "instance_id": f"{slug}-{index}",
            "repo": repo,
            "base_commit": f"{index:040d}",
            "problem_statement": "A production function returns an incorrect value. " * 4,
            "patch": TrainingPoolTest._patch("src/feature.py"),
            "test_patch": "gold answer must never be exported",
            "FAIL_TO_PASS": json.dumps(["tests/test_feature.py::test_bug"]),
            "PASS_TO_PASS": json.dumps(["tests/test_feature.py::test_existing"]),
            "version": "1.0",
            "environment_setup_commit": "",
        }

    @staticmethod
    def _patch(path: str) -> str:
        return (
            f"diff --git a/{path} b/{path}\n"
            f"--- a/{path}\n"
            f"+++ b/{path}\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        )


if __name__ == "__main__":
    unittest.main()
