from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .agent import Agent, AgentMetrics
from .tools import RepositoryTools
from .types import Model


_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    issue: str
    test_command: str
    language: str
    category: str
    repo_dir: Path

    @classmethod
    def load(cls, manifest_path: Path) -> "BenchmarkCase":
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read benchmark manifest {manifest_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Manifest must contain a JSON object: {manifest_path}")
        if data.get("schema_version") != 1:
            raise ValueError(f"Unsupported or missing schema_version in {manifest_path}")
        required = ("id", "issue", "test_command", "language", "category")
        missing = [field for field in required if not isinstance(data.get(field), str) or not data[field].strip()]
        if missing:
            raise ValueError(f"Missing non-empty string fields {missing} in {manifest_path}")
        case_id = data["id"]
        if not _CASE_ID.fullmatch(case_id):
            raise ValueError(f"Unsafe case id {case_id!r} in {manifest_path}")
        repo_dir = manifest_path.parent / "repo"
        if not repo_dir.is_dir() or not any(repo_dir.iterdir()):
            raise ValueError(f"Case repository is missing or empty: {repo_dir}")
        return cls(
            id=case_id,
            issue=data["issue"],
            test_command=data["test_command"],
            language=data["language"],
            category=data["category"],
            repo_dir=repo_dir,
        )


@dataclass(frozen=True)
class CaseResult:
    id: str
    category: str
    status: str
    resolved: bool
    steps: int
    duration_seconds: float
    initial_test_output: str
    answer: str
    trajectory_path: str | None
    patch_path: str | None
    metrics: dict[str, int | float]
    error: str | None = None


def discover_cases(benchmark_dir: Path, selected_ids: set[str] | None = None) -> list[BenchmarkCase]:
    manifests = sorted(benchmark_dir.glob("*/case.json"))
    cases = [BenchmarkCase.load(path) for path in manifests]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark case ids must be unique")
    if selected_ids:
        unknown = selected_ids - set(ids)
        if unknown:
            raise ValueError(f"Unknown benchmark case ids: {sorted(unknown)}")
        cases = [case for case in cases if case.id in selected_ids]
    if not cases:
        raise ValueError(f"No benchmark cases found in {benchmark_dir}")
    return cases


class BenchmarkRunner:
    def __init__(
        self,
        model: Model,
        model_name: str,
        benchmark_dir: Path,
        results_dir: Path,
        max_steps: int = 20,
        timeout_seconds: int = 120,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.benchmark_dir = benchmark_dir.resolve()
        self.results_dir = results_dir.resolve()
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds

    def run(self, selected_ids: set[str] | None = None) -> tuple[dict[str, Any], Path]:
        cases = discover_cases(self.benchmark_dir, selected_ids)
        started_at = datetime.now(timezone.utc)
        run_id = started_at.strftime("%Y%m%dT%H%M%SZ") + f"-{uuid4().hex[:8]}"
        run_dir = self.results_dir / run_id
        case_results_dir = run_dir / "cases"
        trajectories_dir = run_dir / "trajectories"
        patches_dir = run_dir / "patches"
        case_results_dir.mkdir(parents=True, exist_ok=False)
        trajectories_dir.mkdir()
        patches_dir.mkdir()

        results: list[CaseResult] = []
        for case in cases:
            result = self._run_case(case, trajectories_dir, patches_dir)
            results.append(result)
            self._write_json(case_results_dir / f"{case.id}.json", asdict(result))

        attempted = [result for result in results if result.status != "invalid_baseline"]
        resolved = sum(result.resolved for result in results)
        metric_counts = (
            "tool_calls",
            "tool_failures",
            "invalid_actions",
            "edit_attempts",
            "edit_successes",
            "patch_attempts",
            "patch_successes",
            "replace_text_attempts",
            "replace_text_successes",
        )
        metric_totals = {
            key: sum(int(result.metrics[key]) for result in results)
            for key in metric_counts
        }
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "model": self.model_name,
            "benchmark": str(self.benchmark_dir),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "total": len(results),
            "attempted": len(attempted),
            "resolved": resolved,
            "unresolved": sum(result.status == "unresolved" for result in results),
            "errors": sum(result.status == "error" for result in results),
            "invalid_baselines": sum(result.status == "invalid_baseline" for result in results),
            "resolve_rate": round(resolved / len(attempted), 4) if attempted else 0.0,
            "average_steps": round(sum(result.steps for result in attempted) / len(attempted), 2) if attempted else 0.0,
            "average_duration_seconds": round(
                sum(result.duration_seconds for result in attempted) / len(attempted), 3
            ) if attempted else 0.0,
            "metrics": {
                **metric_totals,
                "tool_failure_rate": self._rate(
                    metric_totals["tool_failures"],
                    metric_totals["tool_calls"],
                ),
                "edit_success_rate": self._rate(
                    metric_totals["edit_successes"],
                    metric_totals["edit_attempts"],
                ),
                "patch_success_rate": self._rate(
                    metric_totals["patch_successes"],
                    metric_totals["patch_attempts"],
                ),
                "replace_text_success_rate": self._rate(
                    metric_totals["replace_text_successes"],
                    metric_totals["replace_text_attempts"],
                ),
            },
            "cases": [asdict(result) for result in results],
        }
        summary_path = run_dir / "summary.json"
        self._write_json(summary_path, summary)
        return summary, summary_path

    def _run_case(
        self,
        case: BenchmarkCase,
        trajectories_dir: Path,
        patches_dir: Path,
    ) -> CaseResult:
        started = time.perf_counter()
        trajectory_path: str | None = None
        patch_path: str | None = None
        initial_output = ""
        try:
            with tempfile.TemporaryDirectory(prefix=f"patchpilot-{case.id}-") as directory:
                repo = Path(directory) / "repo"
                shutil.copytree(
                    case.repo_dir,
                    repo,
                    ignore=shutil.ignore_patterns(".git", ".patchpilot", "__pycache__", "*.pyc"),
                )
                self._initialize_git_repository(repo)
                tools = RepositoryTools(repo, case.test_command, self.timeout_seconds)
                initial_test = tools.run_tests({})
                initial_output = initial_test.output
                if initial_test.ok:
                    return CaseResult(
                        id=case.id,
                        category=case.category,
                        status="invalid_baseline",
                        resolved=False,
                        steps=0,
                        duration_seconds=round(time.perf_counter() - started, 3),
                        initial_test_output=initial_output,
                        answer="Initial tests passed; the case does not reproduce its bug.",
                        trajectory_path=None,
                        patch_path=None,
                        metrics=AgentMetrics().as_dict(),
                    )

                agent_result = Agent(
                    self.model,
                    tools,
                    trajectories_dir / case.id,
                    self.max_steps,
                ).run(case.issue)
                trajectory_path = str(agent_result.trajectory_path)
                diff = tools.git_diff({})
                patch_file = patches_dir / f"{case.id}.patch"
                patch_file.write_text(diff.output if diff.ok else "", encoding="utf-8")
                patch_path = str(patch_file)
                return CaseResult(
                    id=case.id,
                    category=case.category,
                    status="resolved" if agent_result.success else "unresolved",
                    resolved=agent_result.success,
                    steps=agent_result.steps,
                    duration_seconds=round(time.perf_counter() - started, 3),
                    initial_test_output=initial_output,
                    answer=agent_result.answer,
                    trajectory_path=trajectory_path,
                    patch_path=patch_path,
                    metrics=agent_result.metrics.as_dict(),
                )
        except Exception as exc:  # Keep one failed case from aborting the benchmark.
            return CaseResult(
                id=case.id,
                category=case.category,
                status="error",
                resolved=False,
                steps=0,
                duration_seconds=round(time.perf_counter() - started, 3),
                initial_test_output=initial_output,
                answer="Benchmark case failed before completion.",
                trajectory_path=trajectory_path,
                patch_path=patch_path,
                metrics=AgentMetrics().as_dict(),
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _initialize_git_repository(repo: Path) -> None:
        commands = (
            ["git", "init", "-q"],
            ["git", "add", "--all"],
            [
                "git",
                "-c",
                "user.name=PatchPilot Benchmark",
                "-c",
                "user.email=benchmark@patchpilot.local",
                "commit",
                "-qm",
                "benchmark baseline",
            ],
        )
        for command in commands:
            result = subprocess.run(command, cwd=repo, text=True, capture_output=True, check=False)
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"Failed to initialize case repository: {detail}")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _rate(successes: int, attempts: int) -> float:
        return round(successes / attempts, 4) if attempts else 0.0
