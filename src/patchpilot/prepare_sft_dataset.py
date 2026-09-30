"""Build an assistant-only causal-LM dataset from verified train chats.

This is CPU-only data preparation. It never loads model weights or trains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .audit_sft_tokens import (
    DEFAULT_REVISION,
    DEFAULT_TOKENIZER,
    assistant_content_mask,
)


IGNORE_INDEX = -100


def _validate_record(record: dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if (
        record.get("schema_version") != 1
        or not isinstance(record.get("sample_id"), str)
        or not isinstance(record.get("task_id"), str)
        or record.get("split") != "train"
        or not isinstance(metadata, dict)
        or metadata.get("fidelity") != "exact"
        or metadata.get("verified") is not True
        or metadata.get("official_status") != "resolved"
        or metadata.get("truncated_observations") != 0
        or metadata.get("split_assignment") != "preassigned"
    ):
        raise ValueError(
            "SFT input must be an exact, untruncated, officially resolved "
            "sample from a preassigned train task"
        )


def tokenize_record(
    record: dict[str, Any], tokenizer: Any, *, max_seq_length: int
) -> dict[str, Any]:
    _validate_record(record)
    ids, assistant_mask, _ = assistant_content_mask(tokenizer, record["messages"])
    if len(ids) > max_seq_length:
        raise ValueError(
            f"Sample {record['task_id']} has {len(ids)} tokens, exceeding "
            f"max_seq_length={max_seq_length}; no truncation was applied"
        )
    labels = [
        token if supervised else IGNORE_INDEX
        for token, supervised in zip(ids, assistant_mask)
    ]
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError(f"Sample {record['task_id']} has no assistant labels")
    return {
        "sample_id": record["sample_id"],
        "task_id": record["task_id"],
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
    }


def build_dataset(
    input_path: Path,
    output_path: Path,
    tokenizer: Any,
    *,
    max_seq_length: int,
    tokenizer_name: str,
    tokenizer_revision: str,
    transformers_version: str,
) -> dict[str, Any]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and tokenized output must be different files")
    if not 0 < max_seq_length <= tokenizer.model_max_length:
        raise ValueError(
            f"max_seq_length must be between 1 and tokenizer model_max_length "
            f"({tokenizer.model_max_length})"
        )
    if not isinstance(tokenizer.chat_template, str):
        raise ValueError("Tokenizer must have a single text chat template")

    raw_input = input_path.read_bytes()
    records = [json.loads(line) for line in raw_input.splitlines() if line.strip()]
    if not records:
        raise ValueError("SFT input is empty")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("SFT input rows must be JSON objects")
    for record in records:
        _validate_record(record)
    task_ids = [record.get("task_id") for record in records]
    sample_ids = [record.get("sample_id") for record in records]
    if len(set(task_ids)) != len(task_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate task_id or sample_id in SFT input")

    examples = [
        tokenize_record(record, tokenizer, max_seq_length=max_seq_length)
        for record in records
    ]
    output_bytes = "".join(
        json.dumps(example, separators=(",", ":")) + "\n" for example in examples
    ).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output_bytes)

    lengths = [len(example["input_ids"]) for example in examples]
    label_counts = [
        sum(label != IGNORE_INDEX for label in example["labels"])
        for example in examples
    ]
    return {
        "schema_version": 1,
        "input_sha256": hashlib.sha256(raw_input).hexdigest(),
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "tokenizer": {
            "model": tokenizer_name,
            "revision": tokenizer_revision,
            "transformers_version": transformers_version,
            "chat_template_sha256": hashlib.sha256(
                tokenizer.chat_template.encode()
            ).hexdigest(),
        },
        "max_seq_length": max_seq_length,
        "overlength_policy": "reject_without_truncation",
        "mask_policy": "assistant content and end-of-turn only; all other tokens use -100",
        "split": "train",
        "samples": len(examples),
        "total_tokens": sum(lengths),
        "assistant_label_tokens": sum(label_counts),
        "masked_context_tokens": sum(lengths) - sum(label_counts),
        "min_max_tokens": [min(lengths), max(lengths)],
        "all_attention_masks_are_one": True,
        "no_empty_label_samples": all(label_counts),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Strict train JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Tokenized JSONL")
    parser.add_argument("--report", type=Path, help="Optional aggregate JSON report")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-seq-length", type=int, default=32768)
    args = parser.parse_args(argv)
    if args.report and args.report.resolve() in {
        args.input.resolve(),
        args.output.resolve(),
    }:
        parser.error("Report path must differ from input and tokenized output")

    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        parser.error(f"Data builder requires transformers and jinja2: {exc}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    try:
        report = build_dataset(
            args.input,
            args.output,
            tokenizer,
            max_seq_length=args.max_seq_length,
            tokenizer_name=args.tokenizer,
            tokenizer_revision=args.revision,
            transformers_version=transformers.__version__,
        )
    except ValueError as exc:
        parser.error(str(exc))
    serialized_report = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized_report, encoding="utf-8")
    print(serialized_report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
