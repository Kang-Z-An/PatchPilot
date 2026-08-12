from __future__ import annotations

import argparse
import json
from pathlib import Path

from .swebench_analysis import analyze_swebench_run, write_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit one PatchPilot SWE-bench batch without changing its official score"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--gold-report-root",
        type=Path,
        action="append",
        default=[],
        help="Optional additional root containing **/gold/<instance>/report.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <run-dir>/analysis)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    analysis = analyze_swebench_run(run_dir, args.gold_report_root)
    output_dir = args.output_dir or run_dir / "analysis"
    json_path, markdown_path = write_analysis(analysis, output_dir)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    print(f"summary: {json_path}")
    print(f"report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
