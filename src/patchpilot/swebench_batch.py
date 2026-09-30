from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .agent import AgentMetrics
from .swebench import SWEBenchRunner, load_instance
from .types import Model


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_POOL_SPLITS = {"train", "dev", "test"}
_COUNT_METRICS = (
    "tool_calls",
    "tool_failures",
    "invalid_actions",
    "edit_attempts",
    "edit_successes",
    "patch_attempts",
    "patch_successes",
    "replace_text_attempts",
    "replace_text_successes",
    "inspection_calls",
    "budget_blocked_actions",
    "automatic_git_diff_calls",
    "protected_edit_attempts",
    "empty_diff_attempts",
    "test_files_modified",
)


@dataclass(frozen=True)
class SWEBenchBatchCase:
    instance_id: str
    instance_file: Path
    test_command: str
    test_policy: str | None = None
    pool_split: str | None = None
    docker_image: str | None = None
    source_repo: Path | None = None
    max_steps: int = 25
    timeout: int = 180
    enabled: bool = True
    protect_test_files: bool = True

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        manifest_path: Path,
        record_number: int,
    ) -> "SWEBenchBatchCase":
        location = f"{manifest_path}:{record_number}"
        if data.get("schema_version") != 1:
            raise ValueError(f"Unsupported or missing schema_version in {location}")
        required = ("instance_id", "instance_file", "test_command")
        missing = [
            field
            for field in required
            if not isinstance(data.get(field), str) or not data[field].strip()
        ]
        if missing:
            raise ValueError(f"Missing non-empty string fields {missing} in {location}")
        enabled = data.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"enabled must be a boolean in {location}")
        protect_test_files = data.get("protect_test_files", True)
        if not isinstance(protect_test_files, bool):
            raise ValueError(f"protect_test_files must be a boolean in {location}")
        max_steps = data.get("max_steps", 25)
        timeout = data.get("timeout", 180)
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError(f"max_steps must be a positive integer in {location}")
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
            raise ValueError(f"timeout must be a positive integer in {location}")

        docker_image = data.get("docker_image")
        if docker_image is not None and (
            not isinstance(docker_image, str) or not docker_image.strip()
        ):
            raise ValueError(f"docker_image must be a non-empty string in {location}")
        source_repo_value = data.get("source_repo")
        if source_repo_value is not None and (
            not isinstance(source_repo_value, str) or not source_repo_value.strip()
        ):
            raise ValueError(f"source_repo must be a non-empty string in {location}")
        test_policy = data.get("test_policy")
        if test_policy is not None and (
            not isinstance(test_policy, str) or not test_policy.strip()
        ):
            raise ValueError(f"test_policy must be a non-empty string in {location}")
        pool_split = data.get("pool_split")
        if pool_split is not None and (
            not isinstance(pool_split, str) or pool_split not in _POOL_SPLITS
        ):
            raise ValueError(
                f"pool_split must be one of {sorted(_POOL_SPLITS)} in {location}"
            )

        base = manifest_path.parent
        instance_file = _resolve_manifest_path(base, data["instance_file"])
        source_repo = (
            _resolve_manifest_path(base, source_repo_value)
            if source_repo_value is not None
            else None
        )
        instance_id = data["instance_id"].strip()
        instance = load_instance(instance_file, instance_id)
        if instance.instance_id != instance_id:
            raise ValueError(f"Instance id mismatch in {location}")
        if source_repo is not None and not source_repo.is_dir():
            raise ValueError(f"source_repo does not exist in {location}: {source_repo}")
        return cls(
            instance_id=instance_id,
            instance_file=instance_file,
            test_command=data["test_command"].strip(),
            test_policy=test_policy.strip() if test_policy is not None else None,
            pool_split=pool_split,
            docker_image=docker_image.strip() if docker_image is not None else None,
            source_repo=source_repo,
            max_steps=max_steps,
            timeout=timeout,
            enabled=enabled,
            protect_test_files=protect_test_files,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "instance_id": self.instance_id,
            "instance_file": str(self.instance_file),
            "test_command": self.test_command,
            "test_policy": self.test_policy,
            "pool_split": self.pool_split,
            "docker_image": self.docker_image,
            "source_repo": str(self.source_repo) if self.source_repo is not None else None,
            "max_steps": self.max_steps,
            "timeout": self.timeout,
            "enabled": self.enabled,
            "protect_test_files": self.protect_test_files,
        }


@dataclass(frozen=True)
class SWEBenchBatchCaseResult:
    instance_id: str
    status: str
    attempt: int
    patch_generated: bool
    agent_verified: bool
    steps: int
    duration_seconds: float
    metadata_path: str | None
    predictions_path: str | None
    patch_path: str | None
    trajectory_path: str | None
    metrics: dict[str, int | float | bool]
    error_type: str | None = None
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SWEBenchBatchCaseResult":
        return cls(**data)


@dataclass(frozen=True)
class OfficialEvaluationResult:
    status: str
    run_id: str
    report_path: str | None
    log_path: str
    returncode: int | None
    error: str | None = None
    image_aliases: tuple[dict[str, str], ...] = ()
    cleanup_errors: tuple[str, ...] = ()


@dataclass
class _DockerImageAlias:
    source_image: str
    target_image: str
    source_id: str
    previous_target_id: str | None
    backup_image: str | None
    restored: bool = False


class _OfficialImageAliasSession:
    def __init__(self, cases: list[SWEBenchBatchCase], namespace: str) -> None:
        self.cases = [case for case in cases if case.enabled]
        self.namespace = namespace.strip().rstrip("/")
        self.aliases: list[_DockerImageAlias] = []
        self.audit_records: list[dict[str, str]] = []
        self.messages: list[str] = []

    def prepare(self) -> None:
        for case in self.cases:
            source = case.docker_image
            if source is None:
                continue
            target = _official_image_target(source, self.namespace)
            if source == target:
                continue
            source_id = _inspect_docker_image(source)
            assert source_id is not None
            previous_id = _inspect_docker_image(target, allow_missing=True)
            if previous_id == source_id:
                self.audit_records.append(
                    {"source": source, "target": target, "action": "reused"}
                )
                self.messages.append(
                    f"Reused existing official image alias {target} for {source}."
                )
                continue

            backup = None
            if previous_id is not None:
                backup = _backup_image_name(target)
                _docker_tag(target, backup)
            try:
                _docker_tag(source, target)
            except RuntimeError as exc:
                if backup is not None:
                    try:
                        _remove_docker_tag(backup)
                    except RuntimeError as cleanup_exc:
                        raise RuntimeError(
                            f"{exc}; backup-tag cleanup also failed: {cleanup_exc}"
                        ) from exc
                raise
            alias = _DockerImageAlias(
                source_image=source,
                target_image=target,
                source_id=source_id,
                previous_target_id=previous_id,
                backup_image=backup,
            )
            self.aliases.append(alias)
            action = "replaced" if previous_id is not None else "created"
            self.audit_records.append(
                {"source": source, "target": target, "action": action}
            )
            self.messages.append(
                f"Temporarily {action} official image alias {target} from {source}."
            )

    def restore(self) -> list[str]:
        errors: list[str] = []
        for alias in reversed(self.aliases):
            if alias.restored:
                continue
            try:
                current_id = _inspect_docker_image(
                    alias.target_image,
                    allow_missing=True,
                )
                if current_id not in {None, alias.source_id}:
                    raise RuntimeError(
                        "Refusing to restore Docker image alias because its target "
                        f"changed during evaluation: {alias.target_image}"
                    )
                if alias.previous_target_id is None:
                    if current_id is not None:
                        _remove_docker_tag(alias.target_image)
                else:
                    assert alias.backup_image is not None
                    backup_id = _inspect_docker_image(alias.backup_image)
                    if backup_id != alias.previous_target_id:
                        raise RuntimeError(
                            "Docker image alias backup changed during evaluation: "
                            f"{alias.backup_image}"
                        )
                    _docker_tag(alias.backup_image, alias.target_image)
                    _remove_docker_tag(alias.backup_image)
                alias.restored = True
                self.messages.append(
                    f"Restored official image alias {alias.target_image}."
                )
            except RuntimeError as exc:
                errors.append(str(exc))
        return errors


def load_batch_manifest(path: Path) -> list[SWEBenchBatchCase]:
    path = path.resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read SWE-bench batch manifest {path}: {exc}") from exc
    records: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL in {path} at line {line_number}: {exc.msg}"
            ) from exc
    if not records:
        raise ValueError(f"No cases found in SWE-bench batch manifest {path}")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Every manifest record must be a JSON object: {path}")
    cases = [
        SWEBenchBatchCase.from_dict(record, path, index)
        for index, record in enumerate(records, 1)
    ]
    ids = [case.instance_id for case in cases]
    if len(ids) != len(set(ids)):
        duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        raise ValueError(f"Duplicate instance ids in {path}: {duplicates}")
    if not any(case.enabled for case in cases):
        raise ValueError(f"No enabled cases found in {path}")
    return cases


def preflight_cases(
    cases: list[SWEBenchBatchCase],
    docker_platform: str = "linux/amd64",
    check_docker: bool = True,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        status = "disabled" if not case.enabled else "ready"
        error = None
        if case.enabled and check_docker and case.docker_image is not None:
            try:
                result = subprocess.run(
                    [
                        "docker",
                        "image",
                        "inspect",
                        "--platform",
                        docker_platform,
                        case.docker_image,
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
            except OSError as exc:
                status = "error"
                error = f"Cannot run Docker: {exc}"
            else:
                if result.returncode != 0:
                    status = "error"
                    error = result.stderr.strip() or result.stdout.strip()
        results.append(
            {"instance_id": case.instance_id, "status": status, "error": error}
        )
    enabled = [result for result in results if result["status"] != "disabled"]
    return {
        "total": len(cases),
        "enabled": len(enabled),
        "ready": sum(result["status"] == "ready" for result in results),
        "errors": sum(result["status"] == "error" for result in results),
        "cases": results,
    }


class SWEBenchBatchRunner:
    def __init__(
        self,
        model: Model,
        model_name: str,
        cases: list[SWEBenchBatchCase],
        results_dir: Path,
        run_dir: Path | None = None,
        run_id: str | None = None,
        docker_platform: str = "linux/amd64",
        docker_cpus: float = 2.0,
        docker_memory: str = "4g",
        docker_pids_limit: int = 512,
        temperature: float = 0.0,
        thinking_mode: str | None = None,
    ) -> None:
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if thinking_mode not in {None, "enabled", "disabled"}:
            raise ValueError("thinking_mode must be enabled, disabled, or unset")
        self.model = model
        self.model_name = model_name
        self.temperature = temperature
        self.thinking_mode = thinking_mode
        self.cases = [case for case in cases if case.enabled]
        self.results_dir = results_dir.resolve()
        self.docker_platform = docker_platform
        self.docker_cpus = docker_cpus
        self.docker_memory = docker_memory
        self.docker_pids_limit = docker_pids_limit
        if run_dir is not None and run_id is not None:
            raise ValueError("Pass either run_dir or run_id, not both")
        if run_dir is not None:
            self.run_dir = run_dir.resolve()
            self.run_id = self.run_dir.name
        else:
            self.run_id = run_id or _new_run_id()
            if not _RUN_ID.fullmatch(self.run_id):
                raise ValueError(f"Unsafe batch run id: {self.run_id!r}")
            self.run_dir = self.results_dir / self.run_id

    def generate(self, resume: bool = False) -> tuple[dict[str, Any], Path]:
        if resume:
            if not self.run_dir.is_dir():
                raise ValueError(f"Cannot resume missing batch run: {self.run_dir}")
        elif self.run_dir.exists():
            raise ValueError(f"Batch run directory already exists: {self.run_dir}")
        self._prepare_run_directory(resume)
        case_results: list[SWEBenchBatchCaseResult] = []
        for case in self.cases:
            record_path = self.run_dir / "cases" / f"{case.instance_id}.json"
            existing = _load_case_result(record_path)
            if resume and _completed_artifacts_available(existing):
                assert existing is not None
                case_results.append(existing)
                continue
            attempt = (existing.attempt + 1) if existing is not None else 1
            result = self._generate_case(case, attempt)
            _write_json(record_path, asdict(result))
            case_results.append(result)
        predictions_path = self._merge_predictions(case_results)
        summary = build_batch_summary(
            self.run_id,
            self.model_name,
            self.cases,
            case_results,
            predictions_path,
            official_report=None,
        )
        summary_path = self.run_dir / "summary.json"
        _write_json(summary_path, summary)
        return summary, summary_path

    def _prepare_run_directory(self, resume: bool) -> None:
        snapshot_path = self.run_dir / "manifest.snapshot.jsonl"
        expected = "".join(
            json.dumps(case.snapshot(), ensure_ascii=False) + "\n" for case in self.cases
        )
        if resume:
            try:
                existing = snapshot_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"Cannot read run manifest snapshot: {exc}") from exc
            if existing != expected:
                raise ValueError("Current manifest does not match the resumed run snapshot")
            metadata_path = self.run_dir / "run.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot read resumed run metadata: {exc}") from exc
            if metadata.get("model") != self.model_name:
                raise ValueError(
                    "Current model does not match the model recorded by the resumed run"
                )
            if metadata.get("temperature", 0.0) != self.temperature:
                raise ValueError(
                    "Current temperature does not match the resumed run"
                )
            if metadata.get("thinking_mode") != self.thinking_mode:
                raise ValueError(
                    "Current thinking mode does not match the resumed run"
                )
            return
        (self.run_dir / "cases").mkdir(parents=True, exist_ok=False)
        (self.run_dir / "artifacts").mkdir()
        snapshot_path.write_text(expected, encoding="utf-8")
        _write_json(
            self.run_dir / "run.json",
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "model": self.model_name,
                "temperature": self.temperature,
                "thinking_mode": self.thinking_mode,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _generate_case(
        self,
        case: SWEBenchBatchCase,
        attempt: int,
    ) -> SWEBenchBatchCaseResult:
        started = time.perf_counter()
        try:
            if case.docker_image is not None:
                _require_docker_image(case.docker_image, self.docker_platform)
            instance = load_instance(case.instance_file, case.instance_id)
            runner = SWEBenchRunner(
                model=self.model,
                model_name=self.model_name,
                results_dir=self.run_dir / "artifacts",
                test_command=case.test_command,
                max_steps=case.max_steps,
                timeout_seconds=case.timeout,
                docker_image=case.docker_image,
                docker_platform=self.docker_platform,
                docker_cpus=self.docker_cpus,
                docker_memory=self.docker_memory,
                docker_pids_limit=self.docker_pids_limit,
                protect_test_files=case.protect_test_files,
            )
            result = runner.run(instance, case.source_repo)
            return SWEBenchBatchCaseResult(
                instance_id=case.instance_id,
                status="completed",
                attempt=attempt,
                patch_generated=result.patch_generated,
                agent_verified=result.agent_verified,
                steps=result.steps,
                duration_seconds=round(time.perf_counter() - started, 3),
                metadata_path=result.metadata_path,
                predictions_path=result.predictions_path,
                patch_path=result.patch_path,
                trajectory_path=result.trajectory_path,
                metrics=result.metrics,
            )
        except Exception as exc:  # One broken instance must not abort the batch.
            return SWEBenchBatchCaseResult(
                instance_id=case.instance_id,
                status="error",
                attempt=attempt,
                patch_generated=False,
                agent_verified=False,
                steps=0,
                duration_seconds=round(time.perf_counter() - started, 3),
                metadata_path=None,
                predictions_path=None,
                patch_path=None,
                trajectory_path=None,
                metrics=AgentMetrics().as_dict(),
                error_type=_classify_error(exc),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _merge_predictions(
        self,
        case_results: list[SWEBenchBatchCaseResult],
    ) -> Path:
        predictions: list[dict[str, Any]] = []
        for result in case_results:
            if result.status != "completed" or result.predictions_path is None:
                continue
            path = Path(result.predictions_path)
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Cannot merge prediction {path}: {exc}") from exc
            if record.get("instance_id") != result.instance_id:
                raise RuntimeError(f"Prediction instance mismatch in {path}")
            predictions.append(record)
        path = self.run_dir / "predictions.jsonl"
        text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in predictions)
        _write_text_atomic(path, text)
        return path


def run_official_evaluation(
    run_dir: Path,
    cases: list[SWEBenchBatchCase],
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    namespace: str = "",
    max_workers: int = 1,
    timeout_seconds: int = 3600,
    official_run_id: str | None = None,
) -> OfficialEvaluationResult:
    run_dir = run_dir.resolve()
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.is_file() or not predictions_path.read_text(encoding="utf-8").strip():
        raise ValueError(f"No merged predictions found in {predictions_path}")
    run_id = official_run_id or f"patchpilot-{run_dir.name}"
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"Unsafe official run id: {run_id!r}")
    instance_ids = [case.instance_id for case in cases if case.enabled]
    official_dir = run_dir / "official"
    official_dir.mkdir(exist_ok=True)
    log_path = official_dir / "harness.log"
    if not official_harness_available():
        error = (
            "Official SWE-bench harness is unavailable. Re-run this stage with "
            "`uv run --with swebench==4.1.0 python -m "
            "patchpilot.evaluate_swebench_batch ... --stage official`."
        )
        log_path.write_text(error + "\n", encoding="utf-8")
        result = OfficialEvaluationResult(
            status="error",
            run_id=run_id,
            report_path=None,
            log_path=str(log_path),
            returncode=None,
            error=error,
        )
        _write_json(official_dir / "result.json", asdict(result))
        return result
    image_aliases = _OfficialImageAliasSession(cases, namespace)
    try:
        image_aliases.prepare()
    except RuntimeError as exc:
        cleanup_errors = image_aliases.restore()
        error = f"Cannot prepare official Docker image aliases: {exc}"
        if cleanup_errors:
            error += "; cleanup also failed: " + "; ".join(cleanup_errors)
        log_path.write_text(
            _official_log(image_aliases.messages, error, cleanup_errors),
            encoding="utf-8",
        )
        result = OfficialEvaluationResult(
            status="error",
            run_id=run_id,
            report_path=None,
            log_path=str(log_path),
            returncode=None,
            error=error,
            image_aliases=tuple(image_aliases.audit_records),
            cleanup_errors=tuple(cleanup_errors),
        )
        _write_json(official_dir / "result.json", asdict(result))
        return result
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max_workers),
        "--instance_ids",
        *instance_ids,
        "--run_id",
        run_id,
        "--namespace",
        namespace,
    ]
    completed: subprocess.CompletedProcess[str] | None = None
    harness_output = ""
    harness_error: str | None = None
    try:
        try:
            completed = subprocess.run(
                command,
                cwd=official_dir,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            harness_output = _combined_output(exc.stdout or "", exc.stderr or "")
            harness_error = f"Official evaluation timed out after {timeout_seconds}s"
        except OSError as exc:
            harness_error = f"Cannot start official evaluator: {exc}"
        else:
            harness_output = _combined_output(completed.stdout, completed.stderr)
    finally:
        cleanup_errors = image_aliases.restore()
    log_path.write_text(
        _official_log(image_aliases.messages, harness_output, cleanup_errors),
        encoding="utf-8",
    )
    reports = sorted(official_dir.glob(f"*.{run_id}.json"))
    report_path = reports[-1] if reports else None
    error = harness_error
    status = "completed"
    if harness_error is not None:
        status = "error"
    elif completed is None:
        status = "error"
        error = "Official evaluator did not return a result"
    elif completed.returncode != 0:
        status = "error"
        error = f"Official evaluator exited with code {completed.returncode}"
    elif report_path is None:
        status = "error"
        error = "Official evaluator did not create a report"
    if cleanup_errors:
        status = "error"
        cleanup_error = "Cannot restore official Docker image aliases: " + "; ".join(
            cleanup_errors
        )
        error = f"{error}; {cleanup_error}" if error else cleanup_error
    result = OfficialEvaluationResult(
        status=status,
        run_id=run_id,
        report_path=str(report_path) if report_path is not None else None,
        log_path=str(log_path),
        returncode=completed.returncode if completed is not None else None,
        error=error,
        image_aliases=tuple(image_aliases.audit_records),
        cleanup_errors=tuple(cleanup_errors),
    )
    _write_json(official_dir / "result.json", asdict(result))
    return result


def official_harness_available() -> bool:
    try:
        return importlib.util.find_spec("swebench.harness.run_evaluation") is not None
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        return False


def _official_image_target(source_image: str, namespace: str) -> str:
    repository = _image_repository(source_image)
    if namespace:
        repository = f"{namespace}/{repository.rsplit('/', 1)[-1]}"
    return f"{repository}:latest"


def _image_repository(image: str) -> str:
    if "@" in image:
        raise RuntimeError(
            f"Digest-pinned Docker images cannot be aliased for official evaluation: {image}"
        )
    last_slash = image.rfind("/")
    last_colon = image.rfind(":")
    return image[:last_colon] if last_colon > last_slash else image


def _backup_image_name(target: str) -> str:
    repository = _image_repository(target)
    return f"{repository}:patchpilot-backup-{uuid4().hex}"


def _inspect_docker_image(image: str, allow_missing: bool = False) -> str | None:
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Cannot run Docker while inspecting {image}: {exc}") from exc
    if completed.returncode == 0:
        image_id = completed.stdout.strip()
        if image_id:
            return image_id
        raise RuntimeError(f"Docker returned no image id for {image}")
    detail = _combined_output(completed.stdout, completed.stderr).strip()
    missing = "no such image" in detail.lower() or "no such object" in detail.lower()
    if allow_missing and missing:
        return None
    raise RuntimeError(f"Cannot inspect Docker image {image}: {detail or 'unknown error'}")


def _docker_tag(source: str, target: str) -> None:
    _run_docker_mutation(["docker", "tag", source, target], f"tag {source} as {target}")


def _remove_docker_tag(image: str) -> None:
    _run_docker_mutation(["docker", "image", "rm", image], f"remove tag {image}")


def _run_docker_mutation(command: list[str], action: str) -> None:
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Cannot run Docker to {action}: {exc}") from exc
    if completed.returncode != 0:
        detail = _combined_output(completed.stdout, completed.stderr).strip()
        raise RuntimeError(f"Cannot {action}: {detail or 'unknown Docker error'}")


def _official_log(
    alias_messages: list[str],
    harness_output: str,
    cleanup_errors: list[str],
) -> str:
    sections: list[str] = []
    if alias_messages:
        sections.append("Docker image aliases:\n" + "\n".join(alias_messages))
    if harness_output.strip():
        sections.append("Official harness:\n" + harness_output.strip())
    if cleanup_errors:
        sections.append("Docker alias cleanup errors:\n" + "\n".join(cleanup_errors))
    return "\n\n".join(sections).rstrip() + "\n"


def update_summary_with_official(
    run_dir: Path,
    model_name: str,
    cases: list[SWEBenchBatchCase],
    official_result: OfficialEvaluationResult,
) -> tuple[dict[str, Any], Path]:
    run_dir = run_dir.resolve()
    case_results = [
        result
        for case in cases
        if case.enabled
        if (result := _load_case_result(run_dir / "cases" / f"{case.instance_id}.json"))
        is not None
    ]
    report = None
    if official_result.report_path is not None:
        try:
            report = json.loads(Path(official_result.report_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read official report: {exc}") from exc
    summary = build_batch_summary(
        run_dir.name,
        model_name,
        [case for case in cases if case.enabled],
        case_results,
        run_dir / "predictions.jsonl",
        official_report=report,
        official_result=official_result,
    )
    path = run_dir / "summary.json"
    _write_json(path, summary)
    return summary, path


def build_batch_summary(
    run_id: str,
    model_name: str,
    cases: list[SWEBenchBatchCase],
    case_results: list[SWEBenchBatchCaseResult],
    predictions_path: Path,
    official_report: dict[str, Any] | None,
    official_result: OfficialEvaluationResult | None = None,
) -> dict[str, Any]:
    completed = [result for result in case_results if result.status == "completed"]
    result_by_id = {result.instance_id: result for result in case_results}
    metric_totals = {
        key: sum(int(result.metrics.get(key, 0)) for result in completed)
        for key in _COUNT_METRICS
    }
    first_edit_steps = [
        int(result.metrics.get("first_edit_step", 0))
        for result in completed
        if int(result.metrics.get("first_edit_step", 0)) > 0
    ]
    fresh_test_runs = sum(
        bool(result.metrics.get("tests_run_after_latest_edit", False))
        for result in completed
    )
    resolved_ids = set(official_report.get("resolved_ids", [])) if official_report else set()
    unresolved_ids = set(official_report.get("unresolved_ids", [])) if official_report else set()
    error_ids = set(official_report.get("error_ids", [])) if official_report else set()
    empty_ids = set(official_report.get("empty_patch_ids", [])) if official_report else set()
    incomplete_ids = set(official_report.get("incomplete_ids", [])) if official_report else set()
    case_summaries: list[dict[str, Any]] = []
    for case in cases:
        result = result_by_id.get(case.instance_id)
        if official_report is None:
            official_status = "pending"
        elif case.instance_id in resolved_ids:
            official_status = "resolved"
        elif case.instance_id in unresolved_ids:
            official_status = "unresolved"
        elif case.instance_id in error_ids:
            official_status = "error"
        elif case.instance_id in empty_ids:
            official_status = "empty_patch"
        elif case.instance_id in incomplete_ids:
            official_status = "incomplete"
        else:
            official_status = "not_submitted"
        case_summaries.append(
            {
                "instance_id": case.instance_id,
                "generation_status": result.status if result is not None else "missing",
                "patch_generated": result.patch_generated if result is not None else False,
                "agent_verified": result.agent_verified if result is not None else False,
                "official_status": official_status,
                "error_type": result.error_type if result is not None else None,
                "protected_edit_attempts": (
                    int(result.metrics.get("protected_edit_attempts", 0))
                    if result is not None
                    else 0
                ),
                "test_files_modified": (
                    int(result.metrics.get("test_files_modified", 0))
                    if result is not None
                    else 0
                ),
                "tests_run_after_latest_edit": (
                    bool(result.metrics.get("tests_run_after_latest_edit", False))
                    if result is not None
                    else False
                ),
            }
        )
    total = len(cases)
    case_ids = {case.instance_id for case in cases}
    official_resolved = len(resolved_ids & case_ids)
    official_errors = len(error_ids & case_ids)
    # SWE-bench reports empty and incomplete patches separately from
    # ``unresolved_ids``.  At the batch level they are still attempted cases
    # that did not resolve the issue, so keep the headline counts exhaustive:
    # total == resolved + unresolved + errors.
    official_unresolved = max(0, total - official_resolved - official_errors)
    official_empty = len(empty_ids & case_ids)
    official_incomplete = len(incomplete_ids & case_ids)
    official_not_submitted = sum(
        case["official_status"] == "not_submitted" for case in case_summaries
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "model": model_name,
        "total": total,
        "generation_completed": len(completed),
        "generation_errors": sum(result.status == "error" for result in case_results),
        "patch_generated": sum(result.patch_generated for result in completed),
        "agent_verified": sum(result.agent_verified for result in completed),
        "average_steps": _average([result.steps for result in completed]),
        "average_duration_seconds": _average(
            [result.duration_seconds for result in completed], digits=3
        ),
        "average_first_edit_step": _average(first_edit_steps),
        "predictions_path": str(predictions_path),
        "metrics": {
            **metric_totals,
            "tool_failure_rate": _rate(
                metric_totals["tool_failures"], metric_totals["tool_calls"]
            ),
            "edit_success_rate": _rate(
                metric_totals["edit_successes"], metric_totals["edit_attempts"]
            ),
            "patch_success_rate": _rate(
                metric_totals["patch_successes"], metric_totals["patch_attempts"]
            ),
            "replace_text_success_rate": _rate(
                metric_totals["replace_text_successes"],
                metric_totals["replace_text_attempts"],
            ),
            "runs_tested_after_latest_edit": fresh_test_runs,
            "test_freshness_rate": _rate(fresh_test_runs, len(completed)),
        },
        "official": {
            "status": official_result.status if official_result is not None else "pending",
            "resolved": official_resolved if official_report is not None else None,
            "unresolved": official_unresolved if official_report is not None else None,
            "errors": official_errors if official_report is not None else None,
            "empty_patches": official_empty if official_report is not None else None,
            "incomplete": official_incomplete if official_report is not None else None,
            "not_submitted": (
                official_not_submitted if official_report is not None else None
            ),
            "resolve_rate": (
                round(official_resolved / total, 4)
                if official_report is not None and total
                else None
            ),
            "report_path": (
                official_result.report_path if official_result is not None else None
            ),
            "error": official_result.error if official_result is not None else None,
        },
        "cases": case_summaries,
    }
    return summary


def _resolve_manifest_path(base: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid4().hex[:8]}"


def _require_docker_image(image: str, platform: str) -> None:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--platform", platform, image],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Cannot run Docker: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Docker image is unavailable: {image}: {detail}")


def _load_case_result(path: Path) -> SWEBenchBatchCaseResult | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read batch case result {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Batch case result must be a JSON object: {path}")
    return SWEBenchBatchCaseResult.from_dict(data)


def _completed_artifacts_available(
    result: SWEBenchBatchCaseResult | None,
) -> bool:
    if result is None or result.status != "completed":
        return False
    if result.predictions_path is None:
        return False
    return Path(result.predictions_path).is_file()


def _classify_error(exc: Exception) -> str:
    message = str(exc)
    if any(
        marker in message
        for marker in ("Model API", "Could not reach model API", "Unexpected model response")
    ):
        return "model_api_error"
    if any(marker in message.lower() for marker in ("docker", "repository", "git ")):
        return "infrastructure_error"
    return "generation_error"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _combined_output(stdout: str | bytes, stderr: str | bytes) -> str:
    values = [
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        for value in (stdout, stderr)
    ]
    parts = [part.strip() for part in values if part.strip()]
    return "\n".join(parts) + ("\n" if parts else "")


def _average(values: list[int | float], digits: int = 2) -> float:
    return round(sum(values) / len(values), digits) if values else 0.0


def _rate(successes: int, attempts: int) -> float:
    return round(successes / attempts, 4) if attempts else 0.0
