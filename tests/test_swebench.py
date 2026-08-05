from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from patchpilot.swebench import SWEBenchInstance, SWEBenchRunner, load_instance


class ScriptedModel:
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self.outputs = iter(outputs)

    def complete(self, messages: list[dict[str, str]]) -> str:
        return json.dumps(next(self.outputs))


class SWEBenchTest(unittest.TestCase):
    def test_loads_one_instance_from_jsonl(self) -> None:
        record = {
            "instance_id": "example__repo-1",
            "repo": "example/repo",
            "base_commit": "a" * 40,
            "problem_statement": "Fix the bug.",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "instances.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            instance = load_instance(path)

        self.assertEqual(instance.instance_id, "example__repo-1")
        self.assertEqual(instance.problem_statement, "Fix the bug.")

    def test_runner_clones_base_commit_and_writes_official_prediction(self) -> None:
        model = ScriptedModel([
            {
                "tool": "replace_text",
                "arguments": {
                    "path": "calculator.py",
                    "old_text": "return a - b",
                    "new_text": "return a + b",
                },
            },
            {"tool": "run_tests", "arguments": {}},
            {"tool": "git_diff", "arguments": {}},
            {"final": "Fixed the arithmetic bug and verified the tests."},
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "calculator.py").write_text(
                "def add(a, b):\n    return a - b\n",
                encoding="utf-8",
            )
            (source / "test_calculator.py").write_text(
                "import unittest\n"
                "from calculator import add\n\n"
                "class CalculatorTest(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        self.assertEqual(add(2, 3), 5)\n",
                encoding="utf-8",
            )
            self._git(source, "init", "-q")
            self._git(source, "add", "--all")
            self._git(
                source,
                "-c",
                "user.name=PatchPilot Test",
                "-c",
                "user.email=test@patchpilot.local",
                "commit",
                "-qm",
                "buggy baseline",
            )
            commit = self._git(source, "rev-parse", "HEAD").stdout.strip()
            instance = SWEBenchInstance(
                instance_id="example__repo-1",
                repo="example/repo",
                base_commit=commit,
                problem_statement="add subtracts its second argument",
            )
            runner = SWEBenchRunner(
                model=model,
                model_name="scripted-model",
                results_dir=root / "results",
                test_command="python -m unittest -q",
                max_steps=8,
            )

            result = runner.run(instance, source)

            self.assertTrue(result.agent_verified)
            self.assertTrue(result.patch_generated)
            self.assertIsNone(result.docker_image)
            prediction = json.loads(
                Path(result.predictions_path).read_text(encoding="utf-8")
            )
            self.assertEqual(prediction["instance_id"], instance.instance_id)
            self.assertEqual(prediction["model_name_or_path"], "scripted-model")
            self.assertIn("return a + b", prediction["model_patch"])
            self.assertTrue(Path(result.trajectory_path).is_file())

    @staticmethod
    def _git(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(result.stderr or result.stdout)
        return result


if __name__ == "__main__":
    unittest.main()
