from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .agent import Agent
from .model import OpenAICompatibleModel
from .tools import RepositoryTools


def load_model_config(env_file: Path, model_override: str | None = None) -> tuple[str, str, str]:
    """Load model settings, with CLI and process environment values taking precedence."""
    load_dotenv(dotenv_path=env_file, override=False)
    api_key = os.environ.get("PATCHPILOT_API_KEY")
    if not api_key:
        raise SystemExit(
            f"PATCHPILOT_API_KEY is required; set it in the environment or {env_file}"
        )
    base_url = os.environ.get("PATCHPILOT_BASE_URL", "https://api.openai.com/v1")
    model_name = model_override or os.environ.get("PATCHPILOT_MODEL")
    if not model_name or not model_name.strip():
        raise SystemExit(
            f"A model is required; pass --model or set PATCHPILOT_MODEL in {env_file}"
        )
    return api_key, base_url, model_name.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a traceable code-repair agent")
    parser.add_argument("--repo", type=Path, required=True, help="Repository to repair")
    parser.add_argument("--issue", required=True, help="Bug report or task description")
    parser.add_argument("--model", help="Model name; overrides PATCHPILOT_MODEL")
    parser.add_argument("--test-command", default="python -m pytest -q")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=120, help="Tool timeout in seconds")
    parser.add_argument(
        "--protect-test-files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Make conventional test files read-only to the agent",
    )
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="dotenv file to load (default: .env in the current directory)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key, base_url, model_name = load_model_config(args.env_file, args.model)
    repo = args.repo.resolve()
    trajectory_dir = args.trajectory_dir or repo / ".patchpilot" / "trajectories"
    tools = RepositoryTools(
        repo,
        args.test_command,
        args.timeout,
        protect_test_files=args.protect_test_files,
    )
    model = OpenAICompatibleModel(model_name, api_key, base_url, args.timeout)
    result = Agent(model, tools, trajectory_dir, args.max_steps).run(args.issue)
    print(result.answer)
    print(f"trajectory: {result.trajectory_path}")
    return 0 if result.success else 1
