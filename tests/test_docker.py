from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from patchpilot.docker import DockerTestRunner, DualCheckoutProbeRunner
from patchpilot.types import ToolResult


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

    def test_probe_runs_assertions_offline_with_read_only_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            completed = subprocess.CompletedProcess(["docker", "run"], 0, "ok\n", "")
            with patch("patchpilot.docker.subprocess.run", return_value=completed) as run:
                runner = DockerTestRunner(
                    repository=repository,
                    image="sweb.eval.example:latest",
                    test_command="python -m pytest -q",
                )
                result = runner.run_probe("assert 2 + 2 == 4")

        self.assertTrue(result.ok)
        self.assertTrue(result.data["read_only_repository"])
        command = run.call_args.args[0]
        self.assertIn("--read-only", command)
        self.assertIn("none", command)
        self.assertIn(f"type=bind,source={repository.resolve()},target=/testbed,readonly", command)
        self.assertEqual(command[-3:], ["/opt/miniconda3/envs/testbed/bin/python", "-c", "assert 2 + 2 == 4"])
        self.assertLessEqual(run.call_args.kwargs["timeout"], 60)

    def test_probe_rejects_non_assertion_and_invalid_python_before_docker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = DockerTestRunner(
                repository=Path(directory),
                image="sweb.eval.example:latest",
                test_command="python -m pytest -q",
            )
            with patch("patchpilot.docker.subprocess.run") as run:
                self.assertFalse(runner.run_probe("print(1)").ok)
                self.assertFalse(runner.run_probe("assert (").ok)
                self.assertFalse(runner.run_probe("assert True\n" * 1000).ok)
                run.assert_not_called()

    def test_probe_timeout_forces_its_named_container_to_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = DockerTestRunner(
                repository=Path(directory),
                image="sweb.eval.example:latest",
                test_command="python -m pytest -q",
                timeout_seconds=5,
            )
            timeout = subprocess.TimeoutExpired(["docker", "run"], 5)
            cleanup = subprocess.CompletedProcess(["docker", "rm"], 0, "", "")
            with patch(
                "patchpilot.docker.subprocess.run", side_effect=[timeout, cleanup]
            ) as run:
                result = runner.run_probe("assert True")

        self.assertFalse(result.ok)
        self.assertIn("Timed out after 5s", result.output)
        name = run.call_args_list[0].args[0][4]
        self.assertEqual(
            run.call_args_list[1].args[0], ["docker", "rm", "--force", name]
        )

    def test_classified_probe_distinguishes_assertion_failure_from_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = DockerTestRunner(
                repository=Path(directory),
                image="sweb.eval.example:latest",
                test_command="python -m pytest -q",
            )
            assertion = subprocess.CompletedProcess(
                ["docker", "run"], 23,
                "PATCHPILOT_PROBE_ASSERTION_FAILED\n", "AssertionError\n",
            )
            import_error = subprocess.CompletedProcess(
                ["docker", "run"], 1, "", "ModuleNotFoundError\n",
            )
            with patch(
                "patchpilot.docker.subprocess.run",
                side_effect=[assertion, import_error],
            ) as run:
                red = runner.run_probe("assert False", classify_assertions=True)
                error = runner.run_probe("import missing\nassert True", classify_assertions=True)

        self.assertEqual(red.data["outcome"], "assertion_failed")
        self.assertEqual(error.data["outcome"], "error")
        self.assertIn("exec(compile(", run.call_args_list[0].args[0][-1])


class DualCheckoutProbeTest(unittest.TestCase):
    def test_same_code_requires_red_green_and_pristine_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            baseline.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=baseline, check=True)
            (baseline / "source.py").write_text("x = 1\n")
            subprocess.run(["git", "add", "."], cwd=baseline, check=True)
            subprocess.run([
                "git", "-c", "user.name=Test", "-c", "user.email=test@example.org",
                "commit", "-qm", "baseline",
            ], cwd=baseline, check=True)
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=baseline, text=True).strip()
            candidate = root / "candidate"
            candidate.mkdir()
            runner = DualCheckoutProbeRunner(
                DockerTestRunner(baseline, "same-image", "unused"),
                DockerTestRunner(candidate, "same-image", "unused"), commit,
            )
            code = "assert x == 2"
            red = ToolResult(False, "AssertionError", {"outcome": "assertion_failed"})
            green = ToolResult(True, "ok", {"outcome": "passed"})
            error = ToolResult(False, "ImportError", {"outcome": "error"})
            for outputs, expected in (([red, green], True), ([green, green], False), ([error, green], False), ([red, red], False)):
                with patch.object(DockerTestRunner, "run_probe", side_effect=outputs) as probe:
                    result = runner.run(code)
                self.assertEqual(result.ok, expected)
                self.assertTrue(result.data["baseline_unchanged"])
                self.assertEqual([call.args[0] for call in probe.call_args_list], [code, code])
            (baseline / "source.py").write_text("x = 9\n")
            with patch.object(DockerTestRunner, "run_probe") as probe:
                result = runner.run(code)
                probe.assert_not_called()
            self.assertFalse(result.ok)
            self.assertEqual(result.data["reason"], "baseline_changed")

    def test_baseline_mutation_during_probe_fails_closed(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        base = Path(directory.name) / "base"
        candidate = Path(directory.name) / "candidate"
        base.mkdir()
        candidate.mkdir()
        runner = DualCheckoutProbeRunner(
            DockerTestRunner(base, "image", "unused"),
            DockerTestRunner(candidate, "image", "unused"), "a" * 40,
        )
        with patch.object(DualCheckoutProbeRunner, "_baseline_is_clean", side_effect=[True, False]), patch.object(
            DockerTestRunner, "run_probe", side_effect=[
                ToolResult(False, "red", {"outcome": "assertion_failed"}),
                ToolResult(True, "green", {"outcome": "passed"}),
            ],
        ):
            result = runner.run("assert False")
        self.assertFalse(result.ok)
        self.assertFalse(result.data["baseline_unchanged"])

    def test_nested_checkouts_rejected_and_git_errors_fail_closed(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        base = Path(directory.name) / "base"
        candidate = Path(directory.name) / "candidate"
        base.mkdir()
        candidate.mkdir()
        nested = candidate / "base"
        nested.mkdir()
        with self.assertRaises(ValueError):
            DualCheckoutProbeRunner(
                DockerTestRunner(nested, "image", "unused"),
                DockerTestRunner(candidate, "image", "unused"), "a" * 40,
            )
        runner = DualCheckoutProbeRunner(
            DockerTestRunner(base, "image", "unused"),
            DockerTestRunner(candidate, "image", "unused"), "a" * 40,
        )
        with patch("patchpilot.docker.subprocess.run", side_effect=OSError("missing")):
            self.assertFalse(runner.run("assert False").ok)


if __name__ == "__main__":
    unittest.main()
