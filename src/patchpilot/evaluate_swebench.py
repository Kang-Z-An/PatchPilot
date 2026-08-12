from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .cli import load_model_config
from .model import OpenAICompatibleModel
from .swebench import SWEBenchRunner, load_instance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one PatchPilot prediction for the SWE-bench harness"
    )
    parser.add_argument(
        "--instance-file",
        type=Path,
        required=True,
        help="JSON or JSONL file containing official SWE-bench instance fields",
    )
    parser.add_argument(
        "--instance-id",
        help="Instance to select when --instance-file contains multiple records",
    )
    parser.add_argument(
        "--source-repo",
        type=Path,
        help="Optional local Git repository to clone instead of downloading from GitHub",
    )
    parser.add_argument(
        "--test-command",
        required=True,
        help="Test command available in the prepared repository environment",
    )
    parser.add_argument("--model", help="Model name; overrides PATCHPILOT_MODEL")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(".patchpilot/swebench/results"),
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--docker-image",
        help=(
            "Run tests in this existing SWE-bench instance image and copy its "
            "pinned /testbed checkout when --source-repo is omitted"
        ),
    )
    parser.add_argument("--docker-platform", default="linux/amd64")
    parser.add_argument("--docker-cpus", type=float, default=2.0)
    parser.add_argument("--docker-memory", default="4g")
    parser.add_argument("--docker-pids-limit", type=int, default=512)
    parser.add_argument(
        "--allow-test-file-edits",
        action="store_true",
        help="Disable the default SWE-bench protection against editing test files",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    instance = load_instance(args.instance_file, args.instance_id)
    api_key, base_url, model_name = load_model_config(args.env_file, args.model)
    model = OpenAICompatibleModel(model_name, api_key, base_url, args.timeout)
    runner = SWEBenchRunner(
        model=model,
        model_name=model_name,
        results_dir=args.results_dir,
        test_command=args.test_command,
        max_steps=args.max_steps,
        timeout_seconds=args.timeout,
        docker_image=args.docker_image,
        docker_platform=args.docker_platform,
        docker_cpus=args.docker_cpus,
        docker_memory=args.docker_memory,
        docker_pids_limit=args.docker_pids_limit,
        protect_test_files=not args.allow_test_file_edits,
    )
    result = runner.run(instance, args.source_repo)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    print("official_evaluation: pending")
    return 0 if result.patch_generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
