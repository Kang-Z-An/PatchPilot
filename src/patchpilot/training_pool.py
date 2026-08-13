from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .tools import is_test_file_path


QUALIFIED_IMAGE_TAGS = {
    "marshmallow-code__marshmallow-1810": "compat",
    "pydicom__pydicom-1194": "precache",
    "pylint-dev__astroid-1959": "compat",
    "pylint-dev__astroid-1978": "compat",
    "sqlfluff__sqlfluff-4041": "localcache",
}


@dataclass(frozen=True)
class TrainingPoolConfig:
    seed: str = "patchpilot-training-pool-v1"
    examples_per_repo: int = 6
    train_per_repo: int = 4
    dev_per_repo: int = 1
    test_per_repo: int = 1
    max_patch_files: int = 3
    max_changed_lines: int = 80
    max_target_tests: int = 5
    min_problem_chars: int = 100
    max_problem_chars: int = 6_000
    excluded_repositories: tuple[str, ...] = ()

    def validate(self) -> None:
        counts = (self.train_per_repo, self.dev_per_repo, self.test_per_repo)
        if any(count < 0 for count in counts):
            raise ValueError("Per-repository split counts cannot be negative")
        if sum(counts) != self.examples_per_repo:
            raise ValueError("Per-repository split counts must equal examples_per_repo")
        if self.examples_per_repo < 1:
            raise ValueError("examples_per_repo must be positive")
        if self.max_patch_files < 1 or self.max_changed_lines < 1:
            raise ValueError("Patch-size limits must be positive")
        if self.max_target_tests < 1:
            raise ValueError("max_target_tests must be positive")
        if not 0 <= self.min_problem_chars <= self.max_problem_chars:
            raise ValueError("Invalid problem length limits")


def select_training_pool(
    records: Iterable[dict[str, Any]],
    reserved_ids: set[str],
    config: TrainingPoolConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = config or TrainingPoolConfig()
    config.validate()
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejection_reasons: Counter[str] = Counter()
    seen_ids: set[str] = set()

    for record in records:
        reason, normalized = _normalize_candidate(record, reserved_ids, config)
        if reason is not None:
            rejection_reasons[reason] += 1
            continue
        assert normalized is not None
        instance_id = normalized["instance_id"]
        if instance_id in seen_ids:
            raise ValueError(f"Duplicate instance_id in source records: {instance_id}")
        seen_ids.add(instance_id)
        eligible[normalized["repo"]].append(normalized)

    insufficient = {
        repo: len(candidates)
        for repo, candidates in eligible.items()
        if len(candidates) < config.examples_per_repo
    }
    if insufficient:
        detail = ", ".join(f"{repo}={count}" for repo, count in sorted(insufficient.items()))
        raise ValueError(f"Repositories have too few eligible examples: {detail}")
    if not eligible:
        raise ValueError("No eligible repositories found")

    selected: list[dict[str, Any]] = []
    repo_counts: dict[str, dict[str, int]] = {}
    for repo in sorted(eligible):
        ranked = sorted(
            eligible[repo],
            key=lambda item: (
                _stable_rank(config.seed, item["instance_id"]),
                item["instance_id"],
            ),
        )[: config.examples_per_repo]
        split_sequence = (
            ["train"] * config.train_per_repo
            + ["dev"] * config.dev_per_repo
            + ["test"] * config.test_per_repo
        )
        counts: Counter[str] = Counter()
        for item, split in zip(ranked, split_sequence, strict=True):
            selected.append({**item, "pool_split": split})
            counts[split] += 1
        repo_counts[repo] = dict(sorted(counts.items()))

    selected.sort(key=lambda item: (item["pool_split"], item["repo"], item["instance_id"]))
    report = {
        "source_records": sum(rejection_reasons.values()) + sum(len(v) for v in eligible.values()),
        "reserved_ids": len(reserved_ids),
        "eligible_records": sum(len(candidates) for candidates in eligible.values()),
        "eligible_repositories": len(eligible),
        "selected_records": len(selected),
        "selected_repositories": len({item["repo"] for item in selected}),
        "split_counts": dict(sorted(Counter(item["pool_split"] for item in selected).items())),
        "repository_split_counts": repo_counts,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }
    return selected, report


def write_training_pool(
    selected: list[dict[str, Any]],
    report: dict[str, Any],
    output_dir: Path,
    *,
    dataset_name: str,
    dataset_revision: str,
    source_split: str,
    config: TrainingPoolConfig,
    exclusion_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    instances_dir = output_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)
    expected_instance_files = {
        f"{item['instance_id']}.json" for item in selected
    }
    for stale_path in instances_dir.glob("*.json"):
        if stale_path.name not in expected_instance_files:
            stale_path.unlink()
    manifest_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    for item in selected:
        instance_id = item["instance_id"]
        instance_path = instances_dir / f"{instance_id}.json"
        instance = {
            "instance_id": instance_id,
            "repo": item["repo"],
            "base_commit": item["base_commit"],
            "problem_statement": item["problem_statement"],
            "version": item["version"],
            "environment_setup_commit": item["environment_setup_commit"],
            "FAIL_TO_PASS": item["fail_to_pass"],
            "PASS_TO_PASS": item["pass_to_pass"],
            "pool_split": item["pool_split"],
            "source": {
                "dataset": dataset_name,
                "revision": dataset_revision,
                "split": source_split,
            },
        }
        _write_json(instance_path, instance)
        test_command = "python -m pytest -q " + " ".join(
            shlex.quote(test_id) for test_id in item["fail_to_pass"]
        )
        manifest_records.append(
            {
                "schema_version": 1,
                "instance_id": instance_id,
                "instance_file": f"instances/{instance_id}.json",
                "docker_image": (
                    f"sweb.eval.x86_64.{instance_id}:"
                    f"{QUALIFIED_IMAGE_TAGS.get(instance_id, 'latest')}"
                ),
                "test_command": test_command,
                "max_steps": 30,
                "timeout": 300,
                "enabled": True,
                "protect_test_files": True,
                "pool_split": item["pool_split"],
            }
        )
        selection_records.append(
            {
                "instance_id": instance_id,
                "repo": item["repo"],
                "pool_split": item["pool_split"],
                "gold_patch_files": item["patch_files"],
                "gold_changed_lines": item["changed_lines"],
                "target_test_count": len(item["fail_to_pass"]),
                "ranking_hash": _stable_rank(config.seed, instance_id),
            }
        )

    selection_path = output_dir / "selection.json"
    report_path = output_dir / "report.json"
    manifest_paths = {
        split: output_dir / f"manifest_{split}.jsonl" for split in ("train", "dev", "test")
    }
    for split, path in manifest_paths.items():
        _write_jsonl(
            path,
            [record for record in manifest_records if record["pool_split"] == split],
        )
    smoke_records: list[dict[str, Any]] = []
    for repo in sorted({item["repo"] for item in selected}):
        repo_train = [
            item
            for item in selected
            if item["repo"] == repo and item["pool_split"] == "train"
        ]
        chosen = min(
            repo_train,
            key=lambda item: (_stable_rank(config.seed, item["instance_id"]), item["instance_id"]),
        )
        smoke_records.append(
            next(
                record
                for record in manifest_records
                if record["instance_id"] == chosen["instance_id"]
            )
        )
    smoke_path = output_dir / "manifest_smoke.jsonl"
    _write_jsonl(smoke_path, smoke_records)
    _write_json(
        selection_path,
        {
            "schema_version": 1,
            "dataset": dataset_name,
            "dataset_revision": dataset_revision,
            "source_split": source_split,
            "selection_seed": config.seed,
            "exclusions": exclusion_metadata or {},
            "policy": {
                "examples_per_repo": config.examples_per_repo,
                "train_per_repo": config.train_per_repo,
                "dev_per_repo": config.dev_per_repo,
                "test_per_repo": config.test_per_repo,
                "max_patch_files": config.max_patch_files,
                "max_changed_lines": config.max_changed_lines,
                "max_target_tests": config.max_target_tests,
                "problem_chars": [config.min_problem_chars, config.max_problem_chars],
                "python_source_only": True,
                "gold_patches_excluded_from_instance_files": True,
                "excluded_repositories": list(config.excluded_repositories),
            },
            "instances": selection_records,
        },
    )
    _write_json(
        report_path,
        {
            "schema_version": 1,
            **report,
            "exclusions": exclusion_metadata or {},
        },
    )
    return {
        "manifest_train": manifest_paths["train"],
        "manifest_dev": manifest_paths["dev"],
        "manifest_test": manifest_paths["test"],
        "manifest_smoke": smoke_path,
        "selection": selection_path,
        "report": report_path,
        "instances": instances_dir,
    }


def discover_reserved_instance_ids(paths: Iterable[Path]) -> set[str]:
    reserved: set[str] = set()
    for raw in paths:
        path = raw.resolve()
        candidates = [path] if path.is_file() else path.rglob("*.json")
        for candidate in candidates:
            try:
                value = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            _collect_instance_ids(value, reserved)
    return reserved


def _normalize_candidate(
    record: dict[str, Any],
    reserved_ids: set[str],
    config: TrainingPoolConfig,
) -> tuple[str | None, dict[str, Any] | None]:
    required = ("instance_id", "repo", "base_commit", "problem_statement", "patch")
    if any(not isinstance(record.get(key), str) or not record[key].strip() for key in required):
        return "missing_required_field", None
    instance_id = record["instance_id"].strip()
    if instance_id in reserved_ids:
        return "reserved_instance", None
    repo = record["repo"].strip()
    if repo in config.excluded_repositories:
        return "excluded_repository", None
    problem = record["problem_statement"].strip()
    if not config.min_problem_chars <= len(problem) <= config.max_problem_chars:
        return "problem_length", None
    fail_to_pass = _parse_string_list(record.get("FAIL_TO_PASS"))
    pass_to_pass = _parse_string_list(record.get("PASS_TO_PASS"))
    if not fail_to_pass:
        return "missing_target_tests", None
    if len(fail_to_pass) > config.max_target_tests:
        return "too_many_target_tests", None
    test_patch = record.get("test_patch")
    if not isinstance(test_patch, str) or not test_patch.strip():
        return "missing_test_patch", None
    patch = record["patch"]
    patch_files = _patch_files(patch)
    if not patch_files:
        return "missing_patch_files", None
    if len(patch_files) > config.max_patch_files:
        return "too_many_patch_files", None
    if any(PurePosixPath(path).suffix != ".py" for path in patch_files):
        return "non_python_patch", None
    if any(is_test_file_path(path) for path in patch_files):
        return "gold_patch_modifies_tests", None
    changed_lines = sum(
        1
        for line in patch.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )
    if changed_lines > config.max_changed_lines:
        return "patch_too_large", None
    return None, {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": record["base_commit"].strip(),
        "problem_statement": problem,
        "version": record.get("version", ""),
        "environment_setup_commit": record.get("environment_setup_commit", ""),
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "patch_files": patch_files,
        "changed_lines": changed_lines,
    }


def _parse_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return []
    return value


def _patch_files(patch: str) -> list[str]:
    return sorted(set(re.findall(r"^diff --git a/(.+?) b/[^\n]+$", patch, re.MULTILINE)))


def _stable_rank(seed: str, instance_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{instance_id}".encode("utf-8")).hexdigest()


def _collect_instance_ids(value: Any, output: set[str]) -> None:
    if isinstance(value, dict):
        instance_id = value.get("instance_id")
        if isinstance(instance_id, str) and instance_id.strip():
            output.add(instance_id.strip())
        for child in value.values():
            _collect_instance_ids(child, output)
    elif isinstance(value, list):
        for child in value:
            _collect_instance_ids(child, output)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )
