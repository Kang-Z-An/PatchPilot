from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from patchpilot.docker import DockerTestRunner


class DockerTestRunnerTest(unittest.TestCase):
    def test_runs_fixed_command_in_limited_offline_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            completed = subprocess.CompletedProcess(
                ["docker", "run"],
                0,
                "2 passed\n",
                "",
            )
            with patch("patchpilot.docker.subprocess.run", return_value=completed) as run:
                result = DockerTestRunner(
                    repository=repository,
                    image="sweb.eval.example:latest",
                    test_command="python -m pytest -q",
                    timeout_seconds=90,
                    cpus=1.5,
                    memory="3g",
                    pids_limit=256,
                ).run()

        self.assertTrue(result.ok)
        self.assertIn("2 passed", result.output)
        command = run.call_args.args[0]
        self.assertEqual(command[:3], ["docker", "run", "--rm"])
        self.assertIn("none", command)
        self.assertIn("no-new-privileges", command)
        self.assertIn("ALL", command)
        self.assertIn("1.5", command)
        self.assertIn("3g", command)
        self.assertEqual(command[-1], "python -m pytest -q")
        self.assertEqual(run.call_args.kwargs["timeout"], 90)

    def test_reports_container_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            timeout = subprocess.TimeoutExpired(
                ["docker", "run"],
                5,
                output="partial output",
            )
            with patch("patchpilot.docker.subprocess.run", side_effect=timeout):
                result = DockerTestRunner(
                    repository=Path(directory),
                    image="sweb.eval.example:latest",
                    test_command="python -m pytest -q",
                    timeout_seconds=5,
                ).run()

        self.assertFalse(result.ok)
        self.assertIn("partial output", result.output)
        self.assertIn("Timed out after 5s", result.output)


if __name__ == "__main__":
    unittest.main()
