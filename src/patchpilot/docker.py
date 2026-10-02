from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .types import ToolResult


_MAX_OUTPUT = 20_000
_MAX_PROBE_CHARS = 4_000
_ASSERTION_FAILED_MARKER = "PATCHPILOT_PROBE_ASSERTION_FAILED"


@dataclass(frozen=True)
class DockerTestRunner:
    """Run a fixed test command in a disposable, resource-limited container."""

    repository: Path
    image: str
    test_command: str
    timeout_seconds: int = 120
    platform: str = "linux/amd64"
    cpus: float = 2.0
    memory: str = "4g"
    pids_limit: int = 512
    network: str = "none"

    def __post_init__(self) -> None:
        repository = self.repository.resolve()
        if not repository.is_dir():
            raise ValueError(f"Repository does not exist: {repository}")
        if not self.image.strip():
            raise ValueError("Docker image cannot be empty")
        if not self.test_command.strip():
            raise ValueError("Test command cannot be empty")
        if self.timeout_seconds < 1:
            raise ValueError("Docker test timeout must be positive")
        if self.cpus <= 0:
            raise ValueError("Docker CPU limit must be positive")
        if self.pids_limit < 1:
            raise ValueError("Docker PID limit must be positive")
        object.__setattr__(self, "repository", repository)

    def run(self) -> ToolResult:
        command = [
            "docker",
            "run",
            "--rm",
            "--platform",
            self.platform,
            "--network",
            self.network,
            "--cpus",
            str(self.cpus),
            "--memory",
            self.memory,
            "--pids-limit",
            str(self.pids_limit),
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--mount",
            f"type=bind,source={self.repository},target=/testbed",
            "--workdir",
            "/testbed",
            self.image,
            "/opt/miniconda3/bin/conda",
            "run",
            "--no-capture-output",
            "-n",
            "testbed",
            "/bin/bash",
            "-lc",
            self.test_command,
        ]
        try:
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = _combined_output(exc.stdout or "", exc.stderr or "")
            detail = output or "(no output)"
            return ToolResult(
                False,
                _limit(f"{detail}\nTimed out after {self.timeout_seconds}s"),
            )
        except OSError as exc:
            return ToolResult(False, f"Cannot run Docker: {exc}")

        output = _combined_output(result.stdout, result.stderr) or "(no output)"
        return ToolResult(result.returncode == 0, _limit(output))

    def run_probe(self, code: str, *, classify_assertions: bool = False) -> ToolResult:
        """Execute a small assertion probe without write access to the checkout."""
        if not isinstance(code, str) or not code.strip():
            return ToolResult(False, "Probe code must be a non-empty string")
        if len(code) > _MAX_PROBE_CHARS or "\0" in code:
            return ToolResult(False, "Probe code is too large or contains a NUL byte")
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return ToolResult(False, f"Invalid probe Python: {exc.msg}")
        if not any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
            return ToolResult(False, "Probe must contain at least one Python assert")

        executable_code = code
        if classify_assertions:
            executable_code = (
                "import traceback\n"
                "try:\n"
                f"    exec(compile({code!r}, '<issue-probe>', 'exec'), "
                "{'__name__': '__main__'})\n"
                "except AssertionError as exc:\n"
                "    traceback.print_exc()\n"
                "    frames = traceback.extract_tb(exc.__traceback__)\n"
                "    if not frames or frames[-1].filename != '<issue-probe>':\n"
                "        raise SystemExit(24)\n"
                f"    print({_ASSERTION_FAILED_MARKER!r})\n"
                "    raise SystemExit(23)\n"
            )

        container_name = f"patchpilot-probe-{uuid4().hex}"
        command = [
            "docker", "run", "--rm", "--name", container_name,
            "--platform", self.platform,
            "--network", "none", "--cpus", str(self.cpus),
            "--memory", self.memory, "--pids-limit", str(self.pids_limit),
            "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
            "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=64m",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "PYTHONPATH=/testbed",
            "--mount", f"type=bind,source={self.repository},target=/testbed,readonly",
            "--workdir", "/testbed", self.image,
            "/opt/miniconda3/envs/testbed/bin/python", "-c", executable_code,
        ]
        timeout = min(self.timeout_seconds, 60)
        try:
            result = subprocess.run(
                command, text=True, capture_output=True, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            # Killing only the Docker CLI may leave the container running.
            try:
                subprocess.run(
                    ["docker", "rm", "--force", container_name],
                    text=True, capture_output=True, timeout=10, check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
            output = _combined_output(exc.stdout or "", exc.stderr or "")
            return ToolResult(False, _limit(f"{output}\nTimed out after {timeout}s"))
        except OSError as exc:
            return ToolResult(False, f"Cannot run Docker probe: {exc}")
        output = _combined_output(result.stdout, result.stderr) or (
            "Probe assertions passed" if result.returncode == 0 else "Probe failed without output"
        )
        return ToolResult(
            result.returncode == 0,
            _limit(output),
            {
                "kind": "issue_probe",
                "read_only_repository": True,
                "outcome": (
                    "assertion_failed"
                    if classify_assertions
                    and result.returncode == 23
                    and _ASSERTION_FAILED_MARKER in result.stdout
                    else "passed" if result.returncode == 0 else "error"
                ),
            },
        )


@dataclass(frozen=True)
class DualCheckoutProbeRunner:
    """Compare one assertion on a clean base checkout and the current repair."""

    baseline: DockerTestRunner
    candidate: DockerTestRunner
    base_commit: str

    def __post_init__(self) -> None:
        base_path = self.baseline.repository.resolve()
        candidate_path = self.candidate.repository.resolve()
        if (
            base_path == candidate_path
            or base_path.is_relative_to(candidate_path)
            or candidate_path.is_relative_to(base_path)
        ):
            raise ValueError("Probe baseline and candidate must be separate checkouts")
        if (
            self.baseline.image != self.candidate.image
            or self.baseline.platform != self.candidate.platform
        ):
            raise ValueError("Probe environments must use the same image and platform")

    def _baseline_is_clean(self) -> bool:
        for args, expected in (
            (["rev-parse", "HEAD"], self.base_commit),
            (["status", "--porcelain=v1", "--untracked-files=all", "--ignored"], ""),
        ):
            try:
                result = subprocess.run(
                    ["git", *args], cwd=self.baseline.repository,
                    text=True, capture_output=True, timeout=10, check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                return False
            if result.returncode != 0 or result.stdout.strip() != expected:
                return False
        return True

    def run(self, code: str) -> ToolResult:
        if not self._baseline_is_clean():
            return ToolResult(
                False, "Original checkout no longer matches the clean base commit.",
                {"reason": "baseline_changed", "baseline_unchanged": False},
            )
        baseline = self.baseline.run_probe(code, classify_assertions=True)
        candidate = self.candidate.run_probe(code, classify_assertions=True)
        unchanged = self._baseline_is_clean()
        red_green = (
            unchanged
            and baseline.data.get("outcome") == "assertion_failed"
            and candidate.data.get("outcome") == "passed"
            and candidate.ok
        )
        data = {
            "kind": "dual_checkout_probe",
            "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "base_commit": self.base_commit,
            "baseline_outcome": baseline.data.get("outcome", "error"),
            "candidate_outcome": candidate.data.get("outcome", "error"),
            "baseline_unchanged": unchanged,
            "red_green": red_green,
        }
        output = json.dumps(
            {
                **data,
                "baseline_output": baseline.output,
                "candidate_output": candidate.output,
            },
            ensure_ascii=False,
        )
        return ToolResult(red_green, _limit(output), data)


def _combined_output(stdout: str, stderr: str) -> str:
    parts = [part.strip() for part in (stdout, stderr) if part.strip()]
    return "\n".join(parts)


def _limit(text: str) -> str:
    if len(text) <= _MAX_OUTPUT:
        return text
    return text[:_MAX_OUTPUT] + "\n...[output truncated]"
