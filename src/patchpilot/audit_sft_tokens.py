"""Audit token budgets and assistant-only labels for exported SFT chats.

This module loads only a tokenizer. It does not load model weights or train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_TOKENIZER = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"


def assistant_content_mask(
    tokenizer: Any, messages: list[dict[str, str]]
) -> tuple[list[int], list[bool], list[int]]:
    """Return token IDs, content-only assistant mask, and assistant context lengths.

    Re-render each prefix to fail closed if the template is not prefix-stable.
    Assistant role headers are context; content and the end-of-turn marker are
    supervised. This avoids relying on tokenizer-specific generation tags.
    """
    if not messages or any(
        not isinstance(message, dict)
        or message.get("role") not in {"system", "user", "assistant"}
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise ValueError("Expected a nonempty list of system/user/assistant text messages")

    full_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    mask = [False] * len(full_ids)
    previous_ids: list[int] = []
    context_lengths: list[int] = []
    for index, message in enumerate(messages):
        current_ids = tokenizer.apply_chat_template(messages[: index + 1], tokenize=True)
        if current_ids[: len(previous_ids)] != previous_ids:
            raise ValueError(f"Chat template is not prefix-stable at message {index}")
        if full_ids[: len(current_ids)] != current_ids:
            raise ValueError(f"Full chat differs from prefix at message {index}")
        if message["role"] == "assistant":
            header_ids = tokenizer.apply_chat_template(
                messages[:index], tokenize=True, add_generation_prompt=True
            )
            if (
                header_ids[: len(previous_ids)] != previous_ids
                or current_ids[: len(header_ids)] != header_ids
                or len(header_ids) >= len(current_ids)
            ):
                raise ValueError(f"Cannot isolate assistant content at message {index}")
            mask[len(header_ids) : len(current_ids)] = [True] * (
                len(current_ids) - len(header_ids)
            )
            context_lengths.append(len(header_ids))
        previous_ids = current_ids
    if previous_ids != full_ids or not any(mask):
        raise ValueError("Chat rendering is inconsistent or has no assistant labels")
    return full_ids, mask, context_lengths


def audit_dataset(
    path: Path,
    tokenizer: Any,
    *,
    model_name: str,
    revision: str,
    version: str,
) -> dict[str, Any]:
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not records:
        raise ValueError("SFT dataset is empty")
    if not isinstance(tokenizer.chat_template, str):
        raise ValueError("Tokenizer has no single chat template")
    samples = []
    for record in records:
        messages = record["messages"]
        ids, mask, contexts = assistant_content_mask(tokenizer, messages)
        samples.append(
            {
                "task_id": record["task_id"],
                "sample_id": record["sample_id"],
                "total_tokens": len(ids),
                "assistant_label_tokens": sum(mask),
                "context_only_tokens": len(ids) - sum(mask),
                "assistant_turns": len(contexts),
                "longest_assistant_prefix_tokens": max(contexts),
                "fits_tokenizer_context": len(ids) <= tokenizer.model_max_length,
            }
        )
    lengths = [sample["total_tokens"] for sample in samples]
    labels = [sample["assistant_label_tokens"] for sample in samples]
    return {
        "schema_version": 1,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "tokenizer": {
            "model": model_name,
            "revision": revision,
            "transformers_version": version,
            "chat_template_sha256": hashlib.sha256(
                tokenizer.chat_template.encode()
            ).hexdigest(),
            "model_max_length": tokenizer.model_max_length,
            "automatic_assistant_mask_supported": "{% generation"
            in tokenizer.chat_template,
        },
        "mask_policy": (
            "assistant content plus end-of-turn marker; role headers and all "
            "system/user text masked"
        ),
        "sample_count": len(samples),
        "total_tokens": sum(lengths),
        "assistant_label_tokens": sum(labels),
        "assistant_label_fraction": round(sum(labels) / sum(lengths), 6),
        "min_median_max_total_tokens": [
            min(lengths),
            statistics.median(lengths),
            max(lengths),
        ],
        "max_assistant_prefix_tokens": max(
            sample["longest_assistant_prefix_tokens"] for sample in samples
        ),
        "all_fit_tokenizer_context": all(
            sample["fits_tokenizer_context"] for sample in samples
        ),
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Exported SFT JSONL")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args(argv)

    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        parser.error(f"Tokenizer audit requires transformers and jinja2: {exc}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    report = audit_dataset(
        args.input,
        tokenizer,
        model_name=args.tokenizer,
        revision=args.revision,
        version=transformers.__version__,
    )
    result = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    print(result, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
