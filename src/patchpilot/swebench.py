from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .agent import Agent
from .docker import DockerTestRunner
from .tools import RepositoryTools
from .types import Model


_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_COMMIT = re.compile(r"^[0-9a-fA-F]{7,40}$")


@dataclass(frozen=True)
class SWEBenchInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SWEBenchInstance":
        required = ("instance_id", "repo", "base_commit", "problem_statement")
        missing = [
            field
            for field in required
            if not isinstance(data.get(field), str) or not data[field].strip()
        ]
        if missing:
            raise ValueError(f"Missing non-empty SWE-bench fields: {missing}")
        instance_id = data["instance_id"].strip()
        repo = data["repo"].strip()
        base_commit = data["base_commit"].strip()
        if not _INSTANCE_ID.fullmatch(instance_id):
            raise ValueError(f"Unsafe SWE-bench instance id: {instance_id!r}")
        if not _REPOSITORY.fullmatch(repo):
            raise ValueError(f"Invalid GitHub repository name: {repo!r}")
        if not _COMMIT.fullmatch(base_commit):
            raise ValueError(f"Invalid base commit: {base_commit!r}")
        return cls(instance_id, repo, base_commit, data["problem_statement"].strip())


@dataclass(frozen=True)
class SWEBenchRunResult:
    instance_id: str
    model: str
    docker_image: str | None
    agent_verified: bool
    patch_generated: bool
    steps: int
    answer: str
    repository_path: str
    trajectory_path: str
    patch_path: str
    predictions_path: str
    metadata_path: str
    metrics: dict[str, int | float | bool]


def load_instance(path: Path, instance_id: str | None = None) -> SWEBenchInstance:
    """Load one official-style instance from a JSON object, array, or JSONL file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read SWE-bench instance file {path}: {exc}") from exc

    records: list[Any]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}: {exc.msg}"
                ) from exc
    else:
        records = payload if isinstance(payload, list) else [payload]

    if not records:
        raise ValueError(f"No SWE-bench instances found in {path}")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Every SWE-bench record must be a JSON object: {path}")

    if instance_id is None:
        if len(records) != 1:
            raise ValueError(
                f"{path} contains {len(records)} instances; pass --instance-id to select one"
            )
        selected = records[0]
    else:
        matches = [record for record in records if record.get("instance_id") == instance_id]
        if not matches:
            raise ValueError(f"Instance {instance_id!r} was not found in {path}")
        if len(matches) > 1:
            raise ValueError(f"Instance {instance_id!r} occurs more than once in {path}")
        selected = matches[0]
    return SWEBenchInstance.from_dict(selected)


def prepare_repository(
    instance: SWEBenchInstance,
    target: Path,
    source_repo: Path | None = None,
) -> Path:
    """Clone a disposable repository and check out the instance's exact base commit."""
    if target.exists():
        raise ValueError(f"Repository target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source = (
        str(source_repo.resolve())
        if source_repo is not None
        else f"https://github.com/{instance.repo}.git"
    )
    if source_repo is not None and not source_repo.resolve().is_dir():
        raise ValueError(f"Local source repository does not exist: {source_repo}")
    _run_git(
        ["git", "clone", "--quiet", "--no-hardlinks", source, str(target)],
        cwd=target.parent,
        description="clone repository",
    )
    _run_git(
        ["git", "checkout", "--quiet", "--detach", instance.base_commit],
        cwd=target,
        description="check out base commit",
    )
    return target.resolve()


def prepare_repository_from_image(
    instance: SWEBenchInstance,
    target: Path,
    image: str,
    platform: str = "linux/amd64",
) -> Path:
    """Copy the pinned /testbed checkout from an existing SWE-bench image."""
    if target.exists():
        raise ValueError(f"Repository target already exists: {target}")
    if not image.strip():
        raise ValueError("Docker image cannot be empty")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    container_name = f"patchpilot-export-{uuid4().hex[:12]}"
    created = False
    try:
        _run_command(
            [
                "docker",
                "create",
                "--platform",
                platform,
                "--name",
                container_name,
                image,
                "/bin/true",
            ],
            cwd=target.parent,
            description="create repository export container",
        )
        created = True
        _run_command(
            ["docker", "cp", f"{container_name}:/testbed/.", str(target)],
            cwd=target.parent,
            description="copy repository from SWE-bench image",
        )
    except Exception:
        if target.exists():
            _remove_partial_tree(target)
        raise
    finally:
        if created:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                text=True,
                capture_output=True,
                check=False,
            )

    if not (target / ".git").is_dir():
        _remove_partial_tree(target)
        raise RuntimeError(f"Docker image does not contain a Git checkout at /testbed: {image}")
    _run_git(
        ["git", "reset", "--hard", instance.base_commit],
        cwd=target,
        description="reset exported repository to base commit",
    )
    return target.resolve()


class SWEBenchRunner:
    """Generate one SWE-bench prediction; the official harness performs final grading."""

    def __init__(
        self,
        model: Model,
        model_name: str,
        results_dir: Path,
        test_command: str,
        max_steps: int = 20,
        timeout_seconds: int = 120,
        docker_image: str | None = None,
        docker_platform: str = "linux/amd64",
        docker_cpus: float = 2.0,
        docker_memory: str = "4g",
        docker_pids_limit: int = 512,
        protect_test_files: bool = True,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.results_dir = results_dir.resolve()
        self.test_command = test_command
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.docker_image = docker_image
        self.docker_platform = docker_platform
        self.docker_cpus = docker_cpus
        self.docker_memory = docker_memory
        self.docker_pids_limit = docker_pids_limit
        self.protect_test_files = protect_test_files

    def run(
        self,
        instance: SWEBenchInstance,
        source_repo: Path | None = None,
    ) -> SWEBenchRunResult:
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + f"-{uuid4().hex[:8]}"
        )
        run_dir = self.results_dir / instance.instance_id / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        if self.docker_image is not None and source_repo is None:
            repo = prepare_repository_from_image(
                instance,
                run_dir / "repo",
                self.docker_image,
                self.docker_platform,
            )
        else:
            repo = prepare_repository(instance, run_dir / "repo", source_repo)
        docker_runner = None
        if self.docker_image is not None:
            docker_runner = DockerTestRunner(
                repository=repo,
                image=self.docker_image,
                test_command=self.test_command,
                timeout_seconds=self.timeout_seconds,
                platform=self.docker_platform,
                cpus=self.docker_cpus,
                memory=self.docker_memory,
                pids_limit=self.docker_pids_limit,
            )
        tools = RepositoryTools(
            repo,
            self.test_command,
            self.timeout_seconds,
            test_runner=docker_runner.run if docker_runner is not None else None,
            protect_test_files=self.protect_test_files,
        )
        agent_result = Agent(
            self.model,
            tools,
            run_dir / "trajectory",
            self.max_steps,
        ).run(instance.problem_statement)

        patch = collect_git_diff(repo)
        patch_path = run_dir / "model.patch"
        patch_path.write_text(patch, encoding="utf-8")
        prediction = {
            "instance_id": instance.instance_id,
            "model_name_or_path": self.model_name,
            "model_patch": patch,
        }
        predictions_path = run_dir / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(prediction, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        metadata_path = run_dir / "run.json"
        result = SWEBenchRunResult(
            instance_id=instance.instance_id,
            model=self.model_name,
            docker_image=self.docker_image,
            agent_verified=agent_result.success,
            patch_generated=bool(patch.strip()),
            steps=agent_result.steps,
            answer=agent_result.answer,
            repository_path=str(repo),
            trajectory_path=str(agent_result.trajectory_path),
            patch_path=str(patch_path),
            predictions_path=str(predictions_path),
            metadata_path=str(metadata_path),
            metrics=agent_result.metrics.as_dict(),
        )
        metadata_path.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result


def collect_git_diff(repo: Path) -> str:
    """Collect a complete binary-safe diff, including newly-created files."""
    _run_git(
        [
            "git",
            "add",
            "--intent-to-add",
            "--all",
            "--",
            ".",
            ":(exclude).patchpilot",
            ":(exclude).patchpilot/**",
        ],
        cwd=repo,
        description="mark untracked files for diff",
    )
    result = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Failed to collect model patch: {detail}")
    if result.stdout and not result.stdout.endswith("\n"):
        return result.stdout + "\n"
    return result.stdout


def _run_git(command: list[str], cwd: Path, description: str) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Failed to {description}: {detail}")


def _run_command(command: list[str], cwd: Path, description: str) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Failed to {description}: {detail}")


def _remove_partial_tree(path: Path) -> None:
    """Remove only a newly-created run directory after a failed Docker export."""
    import shutil

    shutil.rmtree(path)
