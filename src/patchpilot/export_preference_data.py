from __future__ import annotations

import argparse
import json
from pathlib import Path

from .preference_data import PreferenceExportConfig, export_preference_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export train-only preference pairs from official unresolved and "
            "official resolved PatchPilot trajectories"
        )
    )
    parser.add_argument(
        "--rejected-input",
        type=Path,
        action="append",
        required=True,
        help="Original unresolved trajectory file or directory; may be repeated",
    )
    parser.add_argument(
        "--chosen-input",
        type=Path,
        action="append",
        required=True,
        help="Corrected resolved trajectory file or directory; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-invalid-actions", type=int, default=0)
    parser.add_argument("--max-tool-failure-rate", type=float, default=0.25)
    parser.add_argument("--max-observation-chars", type=int, default=12_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PreferenceExportConfig(
        max_invalid_actions=args.max_invalid_actions,
        max_tool_failure_rate=args.max_tool_failure_rate,
        max_observation_chars=args.max_observation_chars,
    )
    try:
        report = export_preference_dataset(
            args.rejected_input,
            args.chosen_input,
            args.output_dir,
            config,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"dataset: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
