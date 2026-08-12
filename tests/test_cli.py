from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from patchpilot.cli import build_parser, load_model_config
from patchpilot.evaluate import build_parser as build_evaluate_parser
from patchpilot.evaluate_swebench import build_parser as build_swebench_parser


class CliTest(unittest.TestCase):
    def test_default_env_file(self) -> None:
        args = build_parser().parse_args([
            "--repo",
            "/tmp/repo",
            "--issue",
            "fix it",
            "--model",
            "test-model",
        ])

        self.assertEqual(args.env_file, Path(".env"))

    def test_model_is_optional_in_both_commands(self) -> None:
        agent_args = build_parser().parse_args([
            "--repo",
            "/tmp/repo",
            "--issue",
            "fix it",
        ])
        evaluate_args = build_evaluate_parser().parse_args([
            "--benchmark",
            "benchmarks/cases",
        ])

        self.assertIsNone(agent_args.model)
        self.assertIsNone(evaluate_args.model)

    def test_custom_env_file(self) -> None:
        args = build_parser().parse_args([
            "--repo",
            "/tmp/repo",
            "--issue",
            "fix it",
            "--model",
            "test-model",
            "--env-file",
            "private/provider.env",
        ])

        self.assertEqual(args.env_file, Path("private/provider.env"))

    def test_test_file_edit_policy_cli_defaults(self) -> None:
        agent_args = build_parser().parse_args([
            "--repo",
            "/tmp/repo",
            "--issue",
            "fix it",
        ])
        protected_agent_args = build_parser().parse_args([
            "--repo",
            "/tmp/repo",
            "--issue",
            "fix it",
            "--protect-test-files",
        ])
        swebench_args = build_swebench_parser().parse_args([
            "--instance-file",
            "instance.json",
            "--test-command",
            "python -m pytest -q",
        ])
        unprotected_swebench_args = build_swebench_parser().parse_args([
            "--instance-file",
            "instance.json",
            "--test-command",
            "python -m pytest -q",
            "--allow-test-file-edits",
        ])

        self.assertFalse(agent_args.protect_test_files)
        self.assertTrue(protected_agent_args.protect_test_files)
        self.assertFalse(swebench_args.allow_test_file_edits)
        self.assertTrue(unprotected_swebench_args.allow_test_file_edits)

    def test_loads_model_config_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "PATCHPILOT_API_KEY=from-file\n"
                "PATCHPILOT_BASE_URL=https://example.test/v1\n"
                "PATCHPILOT_MODEL=from-file-model\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                api_key, base_url, model_name = load_model_config(env_file)

        self.assertEqual(api_key, "from-file")
        self.assertEqual(base_url, "https://example.test/v1")
        self.assertEqual(model_name, "from-file-model")

    def test_environment_takes_precedence_over_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("PATCHPILOT_API_KEY=from-file\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "PATCHPILOT_API_KEY": "from-environment",
                    "PATCHPILOT_MODEL": "from-environment-model",
                },
                clear=True,
            ):
                api_key, _, model_name = load_model_config(env_file, "from-cli-model")

        self.assertEqual(api_key, "from-environment")
        self.assertEqual(model_name, "from-cli-model")


if __name__ == "__main__":
    unittest.main()
