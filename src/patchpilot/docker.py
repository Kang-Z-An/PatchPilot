from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .types import ToolResult


_MAX_OUTPUT = 20_000


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


def _combined_output(stdout: str, stderr: str) -> str:
    parts = [part.strip() for part in (stdout, stderr) if part.strip()]
    return "\n".join(parts)


def _limit(text: str) -> str:
    if len(text) <= _MAX_OUTPUT:
        return text
    return text[:_MAX_OUTPUT] + "\n...[output truncated]"
