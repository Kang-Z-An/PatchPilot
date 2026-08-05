from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cli import load_model_config
from .model import OpenAICompatibleModel
from .swebench_batch import (
    SWEBenchBatchRunner,
    load_batch_manifest,
    preflight_cases,
    run_official_evaluation,
    update_summary_with_official,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and officially evaluate a resumable SWE-bench batch"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("generate", "official", "all"),
        default="generate",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate manifest files and Docker images without calling the model",
    )
    parser.add_argument(
        "--skip-docker-check",
        action="store_true",
        help="Skip Docker image inspection during --validate-only",
    )
    parser.add_argument("--model", help="Model name; overrides PATCHPILOT_MODEL")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(".patchpilot/swebench/batch"),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--docker-platform", default="linux/amd64")
    parser.add_argument("--docker-cpus", type=float, default=2.0)
    parser.add_argument("--docker-memory", default="4g")
    parser.add_argument("--docker-pids-limit", type=int, default=512)
    parser.add_argument(
        "--dataset-name",
        default="princeton-nlp/SWE-bench_Verified",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--official-namespace", default="")
    parser.add_argument("--official-run-id")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--official-timeout", type=int, default=3600)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cases = load_batch_manifest(args.manifest)
    if args.validate_only:
        result = preflight_cases(
            cases,
            docker_platform=args.docker_platform,
            check_docker=not args.skip_docker_check,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["errors"] == 0 else 1

    if args.resume and args.run_dir is None:
        parser.error("--resume requires --run-dir")
    if args.run_dir is not None and args.run_id is not None:
        parser.error("--run-dir and --run-id cannot be used together")
    if args.stage == "official" and args.run_dir is None:
        parser.error("--stage official requires --run-dir")
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    if args.official_timeout < 1:
        parser.error("--official-timeout must be positive")

    run_dir = args.run_dir.resolve() if args.run_dir is not None else None
    model_name: str
    generation_errors = 0
    if args.stage in {"generate", "all"}:
        api_key, base_url, model_name = load_model_config(args.env_file, args.model)
        model = OpenAICompatibleModel(
            model_name,
            api_key,
            base_url,
            max(case.timeout for case in cases if case.enabled),
        )
        runner = SWEBenchBatchRunner(
            model=model,
            model_name=model_name,
            cases=cases,
            results_dir=args.results_dir,
            run_dir=run_dir,
            run_id=args.run_id,
            docker_platform=args.docker_platform,
            docker_cpus=args.docker_cpus,
            docker_memory=args.docker_memory,
            docker_pids_limit=args.docker_pids_limit,
        )
        summary, summary_path = runner.generate(resume=args.resume)
        run_dir = summary_path.parent
        generation_errors = int(summary["generation_errors"])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"summary: {summary_path}")
    else:
        assert run_dir is not None
        model_name = _load_run_model(run_dir)

    if args.stage in {"official", "all"}:
        assert run_dir is not None
        official_result = run_official_evaluation(
            run_dir=run_dir,
            cases=cases,
            dataset_name=args.dataset_name,
            split=args.split,
            namespace=args.official_namespace,
            max_workers=args.max_workers,
            timeout_seconds=args.official_timeout,
            official_run_id=args.official_run_id,
        )
        summary, summary_path = update_summary_with_official(
            run_dir,
            model_name,
            cases,
            official_result,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"summary: {summary_path}")
        return 0 if official_result.status == "completed" else 2
    return 0 if generation_errors == 0 else 1


def _load_run_model(run_dir: Path) -> str:
    path = run_dir / "run.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read batch run metadata {path}: {exc}") from exc
    model = data.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"Batch run metadata has no model name: {path}")
    return model


if __name__ == "__main__":
    raise SystemExit(main())
