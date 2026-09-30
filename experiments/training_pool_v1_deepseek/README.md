# PatchPilot training-pool v1 baseline — DeepSeek

This directory freezes the compact evidence for PatchPilot's first complete
rollout over all 16 tasks in the `train` partition of training pool v1. Large
repository checkouts, trajectories, Docker logs, model responses, and exported
chat data remain under Git-ignored `.patchpilot/` directories.

## Result

- Model: `deepseek-v4-flash`
- Dataset: `princeton-nlp/SWE-bench` `dev` at revision
  `e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6`
- Frozen pool partition: 16 train tasks, four tasks from each of four repositories
- Generation: 16/16 patches generated, zero generation errors
- Agent-side verification: 16/16
- Official SWE-bench result: **7/16 resolved (43.75%)**
- Official environment errors: 0
- Test-file modifications: 0
- Strict SFT export: 7 accepted exact trajectories, 9 rejected unresolved trajectories

All nine unresolved predictions applied cleanly and preserved their official
regression tests. They failed issue-specific target tests, so none is labeled an
environment failure. The 16/16 Agent-side result is therefore a process metric,
not a task-resolution result.

Machine-readable aggregate metrics are in [`summary.json`](summary.json),
per-instance metrics and labels are in [`cases.json`](cases.json), and the
gold-patch-informed error analysis is in
[`failure_analysis.md`](failure_analysis.md).
The pre-correction preference-export dry run is recorded in
[`preference_pipeline_audit.json`](preference_pipeline_audit.json).
The full-observation SFT export and its pre-training data audit are recorded in
[`sft_data_audit.json`](sft_data_audit.json).
The pinned Qwen2.5-Coder-7B-Instruct tokenizer and assistant-label audit is in
[`tokenizer_audit_qwen25coder7b.json`](tokenizer_audit_qwen25coder7b.json).
The CPU-only, training-ready JSONL build is summarized in
[`sft_tokenized_dataset_audit.json`](sft_tokenized_dataset_audit.json); the
tokenized rows themselves remain ignored under `.patchpilot/`.

## SFT token audit

The seven accepted train conversations contain 92,297 tokens in total under
Qwen2.5-Coder-7B-Instruct's pinned chat template. Only 4,685 tokens (5.08%)
belong to assistant content/end-of-turn labels; system instructions, issues,
tool observations, and assistant role headers are context-only. Individual
conversations range from 7,016 to 18,427 tokens (median 11,772), all below
the tokenizer's 32,768-token configured limit. The longest prefix before an
assistant answer is 18,356 tokens. This is a context-length check, **not** a
GPU memory or training-throughput validation. The tokenizer's chat template
does not provide a generation-tag mask, so the audit verifies stable rendered
message prefixes and constructs labels explicitly; it fails if boundaries
cannot be isolated.

Reproduce without downloading model weights or using a GPU:

```bash
uv run --with 'transformers==4.57.6' --with jinja2 \
  python -m patchpilot.audit_sft_tokens \
  --input .patchpilot/training/v051-train-full16-untruncated-v1/train.jsonl \
  --output experiments/training_pool_v1_deepseek/tokenizer_audit_qwen25coder7b.json
```

The report pins the tokenizer commit, chat-template hash, Transformers version,
and input SHA-256. It records counts but no raw messages. It does not establish
that seven examples are sufficient for a useful fine-tune.

The subsequent data builder produced seven `input_ids` / `attention_mask` /
`labels` rows with 92,297 input tokens, of which 4,685 are assistant labels and
87,612 are masked context. It rejects overlength rows without truncation and
requires preassigned, officially resolved train examples. This output is ready
for a training-loader smoke test, but no SFT/LoRA job has run.

## Interpretation

The primary gap is semantic verification. Public regression files showed that
the patches were syntactically valid and did not break known behavior, but they
did not prove that the new issue behavior was implemented. Four failed patches
changed the wrong abstraction layer, four covered only part of the required
behavior, and one implemented the wrong standard-prescribed algorithm.

Resolved runs were shorter on this small sample: 14.714 mean steps and 100.301
seconds, versus 20 steps and 214.892 seconds for unresolved runs. This is an
observed association, not evidence that a shorter budget would improve
correctness.

Repository-level results were 4/4 for Marshmallow and 1/4 each for pydicom,
Astroid, and SQLFluff. The task count is too small and the selected tasks differ
in difficulty, so these numbers must not be presented as repository-level model
benchmarks.

## Frozen inputs and isolation

- Selection: `benchmarks/training_pool_v1/selection.json`
- Train manifest: `benchmarks/training_pool_v1/manifest_train.jsonl`
- Sanitized instances: `benchmarks/training_pool_v1/instances/`
- Environment adaptations: `benchmarks/training_pool_v1/environment_adaptations/`

The source records come from SWE-bench `dev`, but this project assigned them to
its own frozen train/dev/test partitions before model rollout. The 16 tasks in
this report belong only to the local train partition. The four local dev and
four local test tasks remain outside this training export.

Agent-side commands use public tests present at `base_commit`; they never name
the withheld `FAIL_TO_PASS` tests introduced by `test_patch`. Official grading
uses the separate SWE-bench harness with `--split dev`.

## Retained local run IDs

- `v050-train-smoke-public-tests-r1` — 4 tasks
- `v051-train-half1-public-tests-r1` — 8 tasks
- `v051-train-remaining4-public-tests-r1` — 4 tasks

The historical `run.json` schema records timestamps and model names but not Git
commit hashes. The `v050` and `v051` code labels in `summary.json` are therefore
derived from the explicit run IDs. This report was frozen with PatchPilot
`v0.5.2` (`9d4bc46`), but it does not claim that every prediction was generated
by that exact commit. The external model name also does not pin a provider-side
model revision.

## Reproduction commands

Validate the complete train manifest without calling the model:

```bash
uv run python -m patchpilot.evaluate_swebench_batch \
  --manifest benchmarks/training_pool_v1/manifest_train.jsonl \
  --validate-only
```

For a new full baseline, generate predictions under a new run ID, then grade
the same run with the official `dev` split:

```bash
uv run python -m patchpilot.evaluate_swebench_batch \
  --manifest benchmarks/training_pool_v1/manifest_train.jsonl \
  --stage generate \
  --run-id <new-run-id>

uv run --with swebench python -m patchpilot.evaluate_swebench_batch \
  --manifest benchmarks/training_pool_v1/manifest_train.jsonl \
  --stage official \
  --run-dir .patchpilot/swebench/batch/<new-run-id> \
  --dataset-name princeton-nlp/SWE-bench \
  --split dev \
  --max-workers 1
```

PatchPilot v0.5.2 automatically exposes qualified non-`latest` images to the
official harness using temporary, audited Docker aliases and restores prior
tags afterward.

## Reporting boundary

This is a training-pool baseline, not a SWE-bench leaderboard submission and
not a held-out post-training result. No API key, absolute local path, raw model
response, full trajectory, gold patch, or third-party checkout is committed in
this experiment record.

The nine unresolved train tasks are listed in
`benchmarks/training_pool_v1/manifest_train_corrections.jsonl`. A future
correction run must be officially graded before
`patchpilot.export_preference_data` can pair its resolved trajectories with the
original unresolved ones.
