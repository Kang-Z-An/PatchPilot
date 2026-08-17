from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .training_data import (
    ExportConfig,
    ParsedTrajectory,
    _contains_possible_secret,
    _modified_test_files,
    _quality_rejection,
    _redact_repository,
    _split_for_trajectory,
    discover_trajectories,
    parse_trajectory,
)


@dataclass(frozen=True)
class PreferenceExportConfig:
    """Strict gates for train-only chosen/rejected trajectory pairs."""

    max_invalid_actions: int = 0
    max_tool_failure_rate: float = 0.25
    max_observation_chars: int = 12_000

    def validate(self) -> None:
        ExportConfig(
            max_invalid_actions=self.max_invalid_actions,
            max_tool_failure_rate=self.max_tool_failure_rate,
            max_observation_chars=self.max_observation_chars,
        ).validate()

    def trajectory_config(self) -> ExportConfig:
        return ExportConfig(
            allow_legacy=False,
            max_invalid_actions=self.max_invalid_actions,
            max_tool_failure_rate=self.max_tool_failure_rate,
            max_observation_chars=self.max_observation_chars,
        )


@dataclass(frozen=True)
class _Candidate:
    parsed: ParsedTrajectory
    prompt: tuple[dict[str, str], ...]
    continuation: tuple[dict[str, str], ...]
    conversation_hash: str
    split_provenance: str


def export_preference_dataset(
    rejected_inputs: list[Path],
    chosen_inputs: list[Path],
    output_dir: Path,
    config: PreferenceExportConfig | None = None,
) -> dict[str, Any]:
    """Export officially graded, train-only trajectory preference pairs.

    Rejected inputs must contain official unresolved trajectories. Chosen inputs
    must contain later official resolved corrections for the same task and exact
    initial prompt. The gold patch itself is never converted into a trajectory.
    """

    config = config or PreferenceExportConfig()
    config.validate()
    quality_config = config.trajectory_config()

    rejected_paths = discover_trajectories(rejected_inputs)
    chosen_paths = discover_trajectories(chosen_inputs)
    rejected_reasons: Counter[str] = Counter()
    chosen_reasons: Counter[str] = Counter()
    rejected = _collect_candidates(
        rejected_paths,
        expected_status="unresolved",
        quality_config=quality_config,
        rejection_counts=rejected_reasons,
    )
    chosen = _collect_candidates(
        chosen_paths,
        expected_status="resolved",
        quality_config=quality_config,
        rejection_counts=chosen_reasons,
    )

    pair_reasons: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    all_task_ids = sorted(set(rejected) | set(chosen))
    for task_id in all_task_ids:
        rejected_candidates = rejected.get(task_id, [])
        chosen_candidates = chosen.get(task_id, [])
        if not rejected_candidates or not chosen_candidates:
            continue
        if len(rejected_candidates) != 1:
            pair_reasons["ambiguous_rejected_candidates"] += 1
            continue
        if len(chosen_candidates) != 1:
            pair_reasons["ambiguous_chosen_candidates"] += 1
            continue

        rejected_candidate = rejected_candidates[0]
        chosen_candidate = chosen_candidates[0]
        if rejected_candidate.prompt != chosen_candidate.prompt:
            pair_reasons["prompt_mismatch"] += 1
            continue
        if rejected_candidate.parsed.source != chosen_candidate.parsed.source:
            pair_reasons["source_mismatch"] += 1
            continue
        if (
            rejected_candidate.conversation_hash
            == chosen_candidate.conversation_hash
        ):
            pair_reasons["identical_trajectories"] += 1
            continue

        pair_material = {
            "task_id": task_id,
            "prompt": rejected_candidate.prompt,
            "chosen": chosen_candidate.continuation,
            "rejected": rejected_candidate.continuation,
        }
        pair_hash = _content_hash(pair_material)
        prompt_hash = _content_hash(list(rejected_candidate.prompt))
        records.append(
            {
                "schema_version": 1,
                "pair_id": pair_hash[:20],
                "task_id": task_id,
                "split": "train",
                "prompt": list(rejected_candidate.prompt),
                "chosen": list(chosen_candidate.continuation),
                "rejected": list(rejected_candidate.continuation),
                "metadata": {
                    "source": chosen_candidate.parsed.source,
                    "prompt_hash": prompt_hash,
                    "chosen": _candidate_metadata(
                        chosen_candidate,
                        official_status="resolved",
                    ),
                    "rejected": _candidate_metadata(
                        rejected_candidate,
                        official_status="unresolved",
                    ),
                },
            }
        )

    records.sort(key=lambda record: (record["task_id"], record["pair_id"]))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "train.jsonl", records)

    rejected_task_ids = set(rejected)
    chosen_task_ids = set(chosen)
    paired_task_ids = {record["task_id"] for record in records}
    report = {
        "schema_version": 1,
        "rejected_inputs": len(rejected_inputs),
        "chosen_inputs": len(chosen_inputs),
        "rejected_trajectories_found": len(rejected_paths),
        "chosen_trajectories_found": len(chosen_paths),
        "eligible_rejected_trajectories": sum(map(len, rejected.values())),
        "eligible_chosen_trajectories": sum(map(len, chosen.values())),
        "preference_pairs": len(records),
        "paired_task_ids": sorted(paired_task_ids),
        "unpaired_rejected_task_ids": sorted(rejected_task_ids - paired_task_ids),
        "unpaired_chosen_task_ids": sorted(chosen_task_ids - paired_task_ids),
        "candidate_rejection_reasons": {
            "rejected": dict(sorted(rejected_reasons.items())),
            "chosen": dict(sorted(chosen_reasons.items())),
        },
        "pair_rejection_reasons": dict(sorted(pair_reasons.items())),
        "policy": {
            "split": "train_only",
            "preassigned_split_required": True,
            "exact_schema_v2_required": True,
            "exact_initial_prompt_match_required": True,
            "rejected_official_status": "unresolved",
            "chosen_official_status": "resolved",
            "agent_verification_required_both_sides": True,
            "test_file_changes_rejected": True,
            "possible_secrets_rejected": True,
            "gold_patch_as_trajectory_forbidden": True,
            "max_invalid_actions": config.max_invalid_actions,
            "max_tool_failure_rate": config.max_tool_failure_rate,
            "max_observation_chars": config.max_observation_chars,
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _collect_candidates(
    paths: list[Path],
    *,
    expected_status: str,
    quality_config: ExportConfig,
    rejection_counts: Counter[str],
) -> dict[str, list[_Candidate]]:
    candidates: dict[str, list[_Candidate]] = defaultdict(list)
    for path in paths:
        try:
            parsed = parse_trajectory(path, quality_config.max_observation_chars)
        except ValueError:
            rejection_counts["malformed_trajectory"] += 1
            continue
        reason = _quality_rejection(parsed, quality_config)
        if reason is not None:
            rejection_counts[reason] += 1
            continue
        if parsed.fidelity != "exact":
            rejection_counts["legacy_not_allowed"] += 1
            continue
        if parsed.source != "swebench":
            rejection_counts["not_swebench"] += 1
            continue

        official_status = _strict_official_status(parsed.path, parsed.task_id)
        if official_status is None:
            rejection_counts["official_status_missing"] += 1
            continue
        if official_status != expected_status:
            rejection_counts[f"official_status_{official_status}"] += 1
            continue

        test_files = _modified_test_files(parsed)
        if test_files is None:
            rejection_counts["test_file_status_unknown"] += 1
            continue
        if test_files:
            rejection_counts["test_files_modified"] += 1
            continue
        if _contains_possible_secret(parsed.messages):
            rejection_counts["possible_secret"] += 1
            continue

        try:
            split, split_provenance = _split_for_trajectory(parsed, quality_config)
        except ValueError:
            rejection_counts["invalid_split_assignment"] += 1
            continue
        if split != "train":
            rejection_counts[f"non_train_split_{split}"] += 1
            continue
        if split_provenance == "task_id_hash":
            rejection_counts["preassigned_split_missing"] += 1
            continue

        messages = _redact_repository(
            parsed.messages,
            parsed.start.get("repository"),
        )
        initial_messages = parsed.start.get("initial_messages")
        assert isinstance(initial_messages, list)
        prompt_length = len(initial_messages)
        if prompt_length < 1 or prompt_length >= len(messages):
            rejection_counts["empty_prompt_or_continuation"] += 1
            continue
        prompt = messages[:prompt_length]
        continuation = messages[prompt_length:]
        conversation_hash = _content_hash(list(messages))
        candidates[parsed.task_id].append(
            _Candidate(
                parsed=parsed,
                prompt=prompt,
                continuation=continuation,
                conversation_hash=conversation_hash,
                split_provenance=split_provenance,
            )
        )
    return dict(candidates)


def _strict_official_status(path: Path, task_id: str) -> str | None:
    """Return only an explicit official status, never an Agent-local status."""

    for parent in path.parents:
        summary_path = parent / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cases = summary.get("cases") if isinstance(summary, dict) else None
        if not isinstance(cases, list):
            continue
        for case in cases:
            if not isinstance(case, dict):
                continue
            candidate_id = case.get("instance_id", case.get("id"))
            if candidate_id != task_id:
                continue
            status = case.get("official_status")
            if isinstance(status, str) and status not in {"pending", "unknown"}:
                return status
    return None


def _candidate_metadata(
    candidate: _Candidate,
    *,
    official_status: str,
) -> dict[str, Any]:
    metrics = candidate.parsed.terminal.get("metrics")
    return {
        "model": candidate.parsed.model,
        "fidelity": candidate.parsed.fidelity,
        "verified": True,
        "official_status": official_status,
        "steps": candidate.parsed.terminal.get("step"),
        "split_provenance": candidate.split_provenance,
        "conversation_hash": candidate.conversation_hash,
        "truncated_observations": candidate.parsed.truncated_observations,
        "metrics": metrics if isinstance(metrics, dict) else {},
    }


def _content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
