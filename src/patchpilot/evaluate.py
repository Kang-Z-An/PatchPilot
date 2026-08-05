from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import BenchmarkRunner
from .cli import load_model_config
from .model import OpenAICompatibleModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PatchPilot on a local benchmark")
    parser.add_argument("--benchmark", type=Path, required=True, help="Directory containing case folders")
    parser.add_argument("--model", help="Model name; overrides PATCHPILOT_MODEL")
    parser.add_argument("--results-dir", type=Path, help="Output directory; defaults to a sibling results directory")
    parser.add_argument("--case", action="append", dest="case_ids", help="Run one case id; may be repeated")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key, base_url, model_name = load_model_config(args.env_file, args.model)
    results_dir = args.results_dir or args.benchmark.parent / "results"
    model = OpenAICompatibleModel(model_name, api_key, base_url, args.timeout)
    runner = BenchmarkRunner(
        model=model,
        model_name=model_name,
        benchmark_dir=args.benchmark,
        results_dir=results_dir,
        max_steps=args.max_steps,
        timeout_seconds=args.timeout,
    )
    summary, summary_path = runner.run(set(args.case_ids) if args.case_ids else None)
    concise = {
        key: summary[key]
        for key in (
            "run_id",
            "model",
            "total",
            "attempted",
            "resolved",
            "unresolved",
            "errors",
            "invalid_baselines",
            "resolve_rate",
            "average_steps",
            "average_duration_seconds",
        )
    }
    concise["metrics"] = summary["metrics"]
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
