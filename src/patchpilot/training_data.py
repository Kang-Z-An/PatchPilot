from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .tools import is_test_file_path


_SPLITS = ("train", "dev", "test")
_SECRET_PATTERNS = (
    re.compile(r"\bas_sk_[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"PATCHPILOT_API_KEY\s*=\s*\S+", re.IGNORECASE),
    re.compile(
        r"(?:api[_-]?key|authorization)\s*[:=]\s*['\"]?[A-Za-z0-9_-]{16,}",
        re.IGNORECASE,
    ),
)
_LEGACY_SYSTEM_PROMPT = (
    "You are PatchPilot, a code-repair agent. Fix the reported issue by emitting "
    "one JSON tool action per turn, then finish with a JSON final answer. This is "
    "a legacy trajectory whose exact original system prompt was not recorded."
)


@dataclass(frozen=True)
class ExportConfig:
    allow_legacy: bool = False
    max_invalid_actions: int = 0
    max_tool_failure_rate: float = 0.25
    max_observation_chars: int = 12_000
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: str = "patchpilot-v1"

    def validate(self) -> None:
        if self.max_invalid_actions < 0:
            raise ValueError("max_invalid_actions cannot be negative")
        if not 0 <= self.max_tool_failure_rate <= 1:
            raise ValueError("max_tool_failure_rate must be between 0 and 1")
        if self.max_observation_chars < 1:
            raise ValueError("max_observation_chars must be positive")
        ratios = (self.train_ratio, self.dev_ratio, self.test_ratio)
        if any(ratio < 0 for ratio in ratios):
            raise ValueError("split ratios cannot be negative")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("train/dev/test ratios must sum to 1")


@dataclass(frozen=True)
class ParsedTrajectory:
    path: Path
    start: dict[str, Any]
    terminal: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    fidelity: str
    task_id: str
    source: str
    model: str
    messages: tuple[dict[str, str], ...]
    truncated_observations: int


def export_sft_dataset(
    inputs: list[Path],
    output_dir: Path,
    config: ExportConfig | None = None,
) -> dict[str, Any]:
    config = config or ExportConfig()
    config.validate()
    trajectories = discover_trajectories(inputs)
    rejection_counts: Counter[str] = Counter()
    fidelity_seen: Counter[str] = Counter()
    fidelity_accepted: Counter[str] = Counter()
    split_assignment_counts: Counter[str] = Counter()
    split_provenance_counts: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    task_splits: dict[str, str] = {}

    for path in trajectories:
        try:
            parsed = parse_trajectory(path, config.max_observation_chars)
        except ValueError:
            rejection_counts["malformed_trajectory"] += 1
            continue
        fidelity_seen[parsed.fidelity] += 1
        reason = _quality_rejection(parsed, config)
        if reason is not None:
            rejection_counts[reason] += 1
            continue
        official_status = _external_status(parsed.path, parsed.task_id)
        if official_status in {"unresolved", "error"}:
            rejection_counts[f"official_{official_status}"] += 1
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

        messages = _redact_repository(parsed.messages, parsed.start.get("repository"))
        content_hash = hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if content_hash in seen_samples:
            rejection_counts["duplicate_conversation"] += 1
            continue
        seen_samples.add(content_hash)
        try:
            split, split_provenance = _split_for_trajectory(parsed, config)
        except ValueError:
            rejection_counts["invalid_split_assignment"] += 1
            continue
        previous_split = task_splits.setdefault(parsed.task_id, split)
        if previous_split != split:
            raise ValueError(
                "Conflicting split assignments for task "
                f"{parsed.task_id!r}: {previous_split!r} and {split!r}"
            )
        split_assignment = (
            "hashed" if split_provenance == "task_id_hash" else "preassigned"
        )
        metrics = parsed.terminal.get("metrics")
        accepted.append(
            {
                "schema_version": 1,
                "sample_id": content_hash[:20],
                "task_id": parsed.task_id,
                "split": split,
                "messages": list(messages),
                "metadata": {
                    "source": parsed.source,
                    "model": parsed.model,
                    "fidelity": parsed.fidelity,
                    "verified": True,
                    "official_status": official_status or "not_available",
                    "steps": parsed.terminal.get("step"),
                    "truncated_observations": parsed.truncated_observations,
                    "split_assignment": split_assignment,
                    "split_provenance": split_provenance,
                    "metrics": metrics if isinstance(metrics, dict) else {},
                },
            }
        )
        fidelity_accepted[parsed.fidelity] += 1
        split_assignment_counts[split_assignment] += 1
        split_provenance_counts[split_provenance] += 1

    accepted.sort(key=lambda item: (item["split"], item["task_id"], item["sample_id"]))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {}
    split_task_ids: dict[str, list[str]] = {}
    for split in _SPLITS:
        records = [record for record in accepted if record["split"] == split]
        _write_jsonl(output_dir / f"{split}.jsonl", records)
        split_counts[split] = len(records)
        split_task_ids[split] = sorted({record["task_id"] for record in records})

    report = {
        "schema_version": 1,
        "inputs": len(inputs),
        "trajectories_found": len(trajectories),
        "accepted_samples": len(accepted),
        "rejected_samples": len(trajectories) - len(accepted),
        "unique_tasks": len({record["task_id"] for record in accepted}),
        "fidelity_seen": dict(sorted(fidelity_seen.items())),
        "fidelity_accepted": dict(sorted(fidelity_accepted.items())),
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "split_counts": split_counts,
        "split_task_ids": split_task_ids,
        "split_assignment_counts": dict(sorted(split_assignment_counts.items())),
        "split_provenance_counts": dict(sorted(split_provenance_counts.items())),
        "split_policy": {
            "unit": "task_id",
            "mode": "prefer_preassigned",
            "preassigned_sources": [
                "trajectory",
                "manifest_snapshot",
                "instance_file",
            ],
            "fallback": "deterministic_task_id_hash",
            "seed": config.split_seed,
            "ratios": {
                "train": config.train_ratio,
                "dev": config.dev_ratio,
                "test": config.test_ratio,
            },
        },
        "quality_policy": {
            "verified_only": True,
            "official_unresolved_rejected_when_available": True,
            "test_file_changes_rejected": True,
            "possible_secrets_rejected": True,
            "allow_legacy": config.allow_legacy,
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


def discover_trajectories(inputs: Iterable[Path]) -> list[Path]:
    candidates: set[Path] = set()
    for raw in inputs:
        path = raw.resolve()
        if path.is_file():
            paths = [path]
        elif path.is_dir():
            paths = path.rglob("*.jsonl")
        else:
            raise ValueError(f"Training-data input does not exist: {path}")
        for candidate in paths:
            try:
                with candidate.open(encoding="utf-8") as handle:
                    first = handle.readline()
                value = json.loads(first)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and value.get("event") == "run_started":
                candidates.add(candidate.resolve())
    return sorted(candidates)


def parse_trajectory(path: Path, max_observation_chars: int = 12_000) -> ParsedTrajectory:
    events = _load_events(path)
    start = events[0]
    terminal = events[-1]
    if start.get("event") != "run_started":
        raise ValueError(f"Trajectory does not begin with run_started: {path}")
    if terminal.get("event") not in {"run_finished", "run_stopped"}:
        raise ValueError(f"Trajectory has no terminal event: {path}")
    exact = (
        start.get("trajectory_schema_version") == 2
        and _valid_messages(start.get("initial_messages"))
    )
    messages, truncated = (
        _exact_messages(events, max_observation_chars)
        if exact
        else _legacy_messages(events, max_observation_chars)
    )
    task_id = start.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        task_id = _infer_task_id(path, str(start.get("issue", "")))
    source = start.get("source")
    if not isinstance(source, str) or not source.strip():
        source = "swebench" if "__" in task_id else "legacy"
    model = start.get("model")
    if not isinstance(model, str) or not model.strip():
        model = _infer_model(path) or "unknown"
    return ParsedTrajectory(
        path=path.resolve(),
        start=start,
        terminal=terminal,
        events=tuple(events),
        fidelity="exact" if exact else "legacy_approximate",
        task_id=task_id.strip(),
        source=source.strip(),
        model=model.strip(),
        messages=tuple(messages),
        truncated_observations=truncated,
    )


def split_for_task(task_id: str, config: ExportConfig) -> str:
    digest = hashlib.sha256(
        f"{config.split_seed}\0{task_id}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < config.train_ratio:
        return "train"
    if value < config.train_ratio + config.dev_ratio:
        return "dev"
    return "test"


def _split_for_trajectory(
    parsed: ParsedTrajectory,
    config: ExportConfig,
) -> tuple[str, str]:
    trajectory_split = parsed.start.get("pool_split")
    if trajectory_split is not None:
        return _validate_pool_split(trajectory_split, "trajectory"), "trajectory"

    snapshot_path = _nearest_file(parsed.path, "manifest.snapshot.jsonl")
    if snapshot_path is not None:
        record = _snapshot_record(snapshot_path, parsed.task_id)
        if record is not None:
            snapshot_split = record.get("pool_split")
            if snapshot_split is not None:
                return (
                    _validate_pool_split(snapshot_split, str(snapshot_path)),
                    "manifest_snapshot",
                )
            instance_split = _split_from_instance_file(record, snapshot_path)
            if instance_split is not None:
                return instance_split, "instance_file"

    return split_for_task(parsed.task_id, config), "task_id_hash"


def _snapshot_record(path: Path, task_id: str) -> dict[str, Any] | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid manifest snapshot JSON at {path}:{line_number}: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"Manifest snapshot record must be an object at {path}:{line_number}"
            )
        if value.get("instance_id") == task_id:
            return value
    return None


def _split_from_instance_file(
    snapshot_record: dict[str, Any],
    snapshot_path: Path,
) -> str | None:
    raw_path = snapshot_record.get("instance_file")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    instance_path = Path(raw_path).expanduser()
    if not instance_path.is_absolute():
        instance_path = snapshot_path.parent / instance_path
    try:
        value = json.loads(instance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    records = value if isinstance(value, list) else [value]
    task_id = snapshot_record.get("instance_id")
    for record in records:
        if not isinstance(record, dict) or record.get("instance_id") != task_id:
            continue
        split = record.get("pool_split")
        if split is None:
            return None
        return _validate_pool_split(split, str(instance_path))
    return None


def _validate_pool_split(value: Any, source: str) -> str:
    if not isinstance(value, str) or value not in _SPLITS:
        raise ValueError(
            f"pool_split must be one of {list(_SPLITS)} in {source}, got {value!r}"
        )
    return value


def _quality_rejection(
    parsed: ParsedTrajectory,
    config: ExportConfig,
) -> str | None:
    terminal = parsed.terminal
    if terminal.get("event") != "run_finished" or terminal.get("verified") is not True:
        return "not_verified"
    if parsed.fidelity != "exact" and not config.allow_legacy:
        return "legacy_not_allowed"
    metrics = terminal.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    invalid_actions = metrics.get("invalid_actions", 0)
    if not isinstance(invalid_actions, (int, float)):
        return "invalid_metrics"
    if invalid_actions > config.max_invalid_actions:
        return "too_many_invalid_actions"
    rate = metrics.get("tool_failure_rate")
    if not isinstance(rate, (int, float)):
        calls = metrics.get("tool_calls", 0)
        failures = metrics.get("tool_failures", 0)
        if not isinstance(calls, (int, float)) or not isinstance(
            failures, (int, float)
        ):
            return "invalid_metrics"
        rate = failures / calls if calls else 0.0
    if rate > config.max_tool_failure_rate:
        return "tool_failure_rate_too_high"
    if parsed.fidelity == "exact":
        verification = terminal.get("verification")
        if not isinstance(verification, dict) or not all(
            verification.get(key) is True
            for key in (
                "edit_applied",
                "tests_passed",
                "tests_run_after_latest_edit",
                "diff_inspected",
                "non_empty_diff",
            )
        ):
            return "incomplete_verification"
    return None


def _exact_messages(
    events: list[dict[str, Any]], max_chars: int
) -> tuple[list[dict[str, str]], int]:
    messages = [dict(message) for message in events[0]["initial_messages"]]
    truncated = 0
    for event in events[1:]:
        kind = event.get("event")
        if kind == "model_output":
            raw = event.get("raw")
            if not isinstance(raw, str):
                raise ValueError("model_output.raw must be a string")
            messages.append({"role": "assistant", "content": raw})
        elif kind in {"tool_result", "invalid_action"}:
            observation = event.get("observation")
            if observation is None and event.get("automatic") is True:
                continue
            if not isinstance(observation, str):
                raise ValueError(f"{kind}.observation must be a string")
            observation, was_truncated = _truncate(observation, max_chars)
            truncated += was_truncated
            messages.append({"role": "user", "content": observation})
    return messages, truncated


def _legacy_messages(
    events: list[dict[str, Any]], max_chars: int
) -> tuple[list[dict[str, str]], int]:
    issue = events[0].get("issue")
    if not isinstance(issue, str) or not issue.strip():
        raise ValueError("Legacy trajectory has no issue")
    messages = [
        {"role": "system", "content": _LEGACY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Issue:\n{issue}"},
    ]
    truncated = 0
    for event in events[1:]:
        kind = event.get("event")
        if kind == "model_output":
            raw = event.get("raw")
            if not isinstance(raw, str):
                raise ValueError("model_output.raw must be a string")
            messages.append({"role": "assistant", "content": raw})
        elif kind == "tool_result" and event.get("automatic") is not True:
            observation = json.dumps(
                {
                    "tool": event.get("tool"),
                    "ok": event.get("ok"),
                    "output": event.get("output"),
                },
                ensure_ascii=False,
            )
            observation, was_truncated = _truncate(observation, max_chars)
            truncated += was_truncated
            messages.append({"role": "user", "content": observation})
        elif kind == "invalid_action":
            observation = f"Invalid action: {event.get('error')}"
            messages.append({"role": "user", "content": observation})
    return messages, truncated


def _modified_test_files(parsed: ParsedTrajectory) -> list[str] | None:
    metrics = parsed.terminal.get("metrics")
    if isinstance(metrics, dict):
        count = metrics.get("test_files_modified")
        if isinstance(count, int):
            return ["<recorded-test-file>"] if count > 0 else []
    patch_path = _nearest_file(parsed.path, "model.patch")
    if patch_path is None:
        return None
    try:
        patch = patch_path.read_text(encoding="utf-8")
    except OSError:
        return None
    paths: set[str] = set()
    for raw in re.findall(r"^(?:---|\+\+\+)\s+([^\t\n]+)", patch, re.MULTILINE):
        if raw == "/dev/null":
            continue
        normalized = raw[2:] if raw.startswith(("a/", "b/")) else raw
        if is_test_file_path(normalized):
            paths.add(normalized)
    return sorted(paths)


def _external_status(path: Path, task_id: str) -> str | None:
    for parent in path.parents:
        summary_path = parent / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(summary, dict) or not isinstance(summary.get("cases"), list):
            continue
        for case in summary["cases"]:
            if not isinstance(case, dict):
                continue
            candidate_id = case.get("instance_id", case.get("id"))
            if candidate_id != task_id:
                continue
            official = case.get("official_status")
            if isinstance(official, str) and official not in {"pending", "unknown"}:
                return official
            status = case.get("status")
            if isinstance(status, str):
                return status
    return None


def _infer_task_id(path: Path, issue: str) -> str:
    parts = path.parts
    for marker in ("artifacts", "trajectories"):
        if marker in parts:
            index = len(parts) - 1 - parts[::-1].index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    if "results" in parts:
        index = len(parts) - 1 - parts[::-1].index("results")
        if index + 1 < len(parts) and "__" in parts[index + 1]:
            return parts[index + 1]
    return "legacy-" + hashlib.sha256(issue.encode("utf-8")).hexdigest()[:16]


def _infer_model(path: Path) -> str | None:
    for parent in path.parents:
        for filename in ("run.json", "summary.json"):
            candidate = parent / filename
            if not candidate.is_file():
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            model = data.get("model") if isinstance(data, dict) else None
            if isinstance(model, str) and model.strip():
                return model.strip()
    return None


def _nearest_file(path: Path, filename: str) -> Path | None:
    for parent in path.parents:
        candidate = parent / filename
        if candidate.is_file():
            return candidate
        if (parent / "summary.json").is_file():
            break
    return None


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read trajectory {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Trajectory event must be an object at {path}:{line_number}")
        events.append(value)
    if not events:
        raise ValueError(f"Empty trajectory: {path}")
    return events


def _valid_messages(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(message, dict)
        and message.get("role") in {"system", "user", "assistant"}
        and isinstance(message.get("content"), str)
        for message in value
    )


def _truncate(value: str, max_chars: int) -> tuple[str, int]:
    if len(value) <= max_chars:
        return value, 0
    return value[:max_chars] + "\n...[observation truncated during export]", 1


def _contains_possible_secret(messages: tuple[dict[str, str], ...]) -> bool:
    return any(
        pattern.search(message["content"])
        for message in messages
        for pattern in _SECRET_PATTERNS
    )


def _redact_repository(
    messages: tuple[dict[str, str], ...], repository: Any
) -> tuple[dict[str, str], ...]:
    if not isinstance(repository, str) or not repository:
        return messages
    return tuple(
        {
            "role": message["role"],
            "content": message["content"].replace(repository, "<REPOSITORY>"),
        }
        for message in messages
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
