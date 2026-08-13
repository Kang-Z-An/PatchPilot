from __future__ import annotations

import argparse
import json
from pathlib import Path

from .training_pool import (
    TrainingPoolConfig,
    discover_reserved_instance_ids,
    select_training_pool,
    write_training_pool,
)


DATASET_NAME = "princeton-nlp/SWE-bench"
DATASET_REVISION = "e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6"
SOURCE_SPLIT = "dev"
VERIFIED_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
VERIFIED_DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
VERIFIED_SPLIT = "test"
ENVIRONMENT_INVALID_REPOSITORIES = (
    "pvlib/pvlib-python",
    "pyvista/pyvista",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze a balanced source-only SWE-bench post-training pool"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/training_pool_v1"),
    )
    parser.add_argument(
        "--reserved",
        type=Path,
        action="append",
        default=[Path("benchmarks/swebench")],
        help="JSON file/directory whose instance ids must be excluded; may be repeated",
    )
    parser.add_argument("--seed", default="patchpilot-training-pool-v1")
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument(
        "--verified-dataset-revision",
        default=VERIFIED_DATASET_REVISION,
        help="Pinned Verified revision whose instance ids are always excluded",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The optional datasets package is required; run with `uv run --with datasets`."
        ) from exc
    config = TrainingPoolConfig(
        seed=args.seed,
        excluded_repositories=ENVIRONMENT_INVALID_REPOSITORIES,
    )
    dataset = load_dataset(
        DATASET_NAME,
        revision=args.dataset_revision,
        split=SOURCE_SPLIT,
    )
    verified_dataset = load_dataset(
        VERIFIED_DATASET_NAME,
        revision=args.verified_dataset_revision,
        split=VERIFIED_SPLIT,
    )
    records = [dict(record) for record in dataset]
    verified_ids = {
        record["instance_id"]
        for record in verified_dataset
        if isinstance(record.get("instance_id"), str)
    }
    local_reserved_ids = discover_reserved_instance_ids(args.reserved)
    reserved_ids = verified_ids | local_reserved_ids
    selected, report = select_training_pool(records, reserved_ids, config)
    paths = write_training_pool(
        selected,
        report,
        args.output_dir,
        dataset_name=DATASET_NAME,
        dataset_revision=args.dataset_revision,
        source_split=SOURCE_SPLIT,
        config=config,
        exclusion_metadata={
            "verified_dataset": {
                "dataset": VERIFIED_DATASET_NAME,
                "revision": args.verified_dataset_revision,
                "split": VERIFIED_SPLIT,
                "instance_ids": len(verified_ids),
            },
            "local_reserved_ids": len(local_reserved_ids),
        },
    )
    output = {
        **report,
        "dataset": DATASET_NAME,
        "dataset_revision": args.dataset_revision,
        "source_split": SOURCE_SPLIT,
        "verified_exclusion": {
            "dataset": VERIFIED_DATASET_NAME,
            "revision": args.verified_dataset_revision,
            "split": VERIFIED_SPLIT,
            "instance_ids": len(verified_ids),
        },
        "local_reserved_ids": len(local_reserved_ids),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
