from __future__ import annotations

import argparse
import json
from pathlib import Path

from .training_data import ExportConfig, export_sft_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export verified PatchPilot trajectories as SFT chat data"
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Trajectory JSONL file or directory to scan; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="Include approximate pre-schema-v2 trajectories with an explicit fidelity label",
    )
    parser.add_argument("--max-invalid-actions", type=int, default=0)
    parser.add_argument("--max-tool-failure-rate", type=float, default=0.25)
    parser.add_argument("--max-observation-chars", type=int, default=12_000)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Hash-fallback train ratio when no pool_split is available",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.1,
        help="Hash-fallback dev ratio when no pool_split is available",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Hash-fallback test ratio when no pool_split is available",
    )
    parser.add_argument(
        "--split-seed",
        default="patchpilot-v1",
        help="Seed for the deterministic fallback when no pool_split is available",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExportConfig(
        allow_legacy=args.allow_legacy,
        max_invalid_actions=args.max_invalid_actions,
        max_tool_failure_rate=args.max_tool_failure_rate,
        max_observation_chars=args.max_observation_chars,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )
    try:
        report = export_sft_dataset(args.input, args.output_dir, config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"dataset: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
