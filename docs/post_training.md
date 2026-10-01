# PatchPilot post-training pipeline

PatchPilot's post-training phase starts with data quality, not with launching a
GPU job. A trained model is only comparable with the v0.5.0 Agent baseline when
the examples are reconstructable, tasks do not leak between splits, and the
same frozen evaluator is used before and after training.

## Data flow

```text
Agent run
  -> schema-v2 trajectory
  -> verification and safety filters
  -> frozen pool split (or deterministic task hash fallback)
  -> train/dev/test chat JSONL
  -> SFT/LoRA
  -> frozen local + SWE-bench evaluation

Official unresolved train rollout + officially resolved corrected rollout
  -> exact prompt and task match
  -> chosen/rejected conversation pair
  -> preference optimization (only after the SFT baseline)
```

## Trajectory contract

Schema-v2 `run_started` events include:

- `initial_messages`: the exact system and user messages sent to the model;
- `task_id`: stable benchmark identity used as the split unit;
- `source`: for example `local_benchmark`, `swebench`, or `ad_hoc`;
- `model`: the model that generated the trajectory;
- the issue, step budget, and inspection/edit limits.

Every non-automatic `tool_result` and `invalid_action` stores the exact
`observation` returned to the model. Interleaving these events with
`model_output` events reproduces the supervised chat sequence. Automatic final
diff verification is retained as audit evidence but is not inserted after the
model's final answer.

Pre-v2 trajectories do not contain this information. The exporter can build an
approximate conversation from their event logs only when `--allow-legacy` is
explicitly supplied. Such records carry `fidelity=legacy_approximate`.

## Default quality gates

A strict SFT sample must satisfy all available evidence:

1. The terminal event is `run_finished` with `verified=true`.
2. A successful edit, fresh passing tests, and non-empty diff inspection are
   present in schema-v2 verification metadata.
3. If an official benchmark status exists, it is not unresolved or error.
4. The final patch does not modify a conventional test file.
5. Invalid actions and tool-failure rate stay within configured limits.
6. No recognizable API key or authorization secret appears in the conversation.
7. An identical conversation has not already been exported.

Strict export now keeps each original tool observation in full. An optional
`--max-observation-chars` limit rejects an over-limit trajectory instead of
silently truncating context and labeling it exact. Earlier 12,000-character
exports shortened three observations across two of the seven accepted train
samples; the corrected export retains all seven without truncation. See
`experiments/training_pool_v1_deepseek/sft_data_audit.json`.

For SFT, supervise assistant messages containing the JSON tool actions and
final answer. Keep system/user messages and tool observations as context only.
The seven exported conversations are 24,607–61,229 characters each. The pinned
Qwen2.5-Coder-7B-Instruct audit measures 7,016–18,427 tokens per conversation;
all seven fit its tokenizer's 32,768-token limit. This is not a GPU-memory
measurement or evidence of model improvement.

Convert the strict, officially resolved train partition into causal-LM rows:

```bash
uv run --with 'transformers==4.57.6' --with jinja2 \
  python -m patchpilot.prepare_sft_dataset \
  --input .patchpilot/training/v051-train-full16-untruncated-v1/train.jsonl \
  --output .patchpilot/training/v051-train-full16-untruncated-v1/train.qwen25coder7b.tokens.jsonl \
  --report experiments/training_pool_v1_deepseek/sft_tokenized_dataset_audit.json
```

Each row contains `input_ids`, an all-ones `attention_mask`, and aligned
`labels`: assistant content and its end-of-turn suffix retain their token IDs;
all other positions are `-100` and do not contribute to the causal-LM loss.
The builder requires exact, untruncated, verified, officially resolved samples
from preassigned train tasks. It rejects duplicates and sequences above
`--max-seq-length` (default 32,768) without silently clipping any observation.
The tokenized JSONL is kept under ignored `.patchpilot/`; only hashes and
aggregate counts are committed. No model weights or GPU are used in this step.

For frozen benchmark pools, samples preserve the task's preassigned
`pool_split` from the batch snapshot. Older snapshots are recovered through
their recorded instance file. Ad-hoc runs without either source use a seeded
hash of `task_id` as a deterministic fallback. The exporter records this
provenance in each sample and report, and aborts if the same task is assigned to
different splits.

## Corpus audits

The first audit scanned 24 historical pre-schema-v2 trajectories:

- strict exact mode: **0 accepted** because all completed historical runs
  predate schema v2;
- explicit legacy audit: **5 approximate candidates** across five tasks;
- 19 were rejected due to incomplete/malformed runs, unknown test-edit status,
  actual test-file edits, or an invalid-action quality threshold.

The five legacy candidates are evidence that the converter works, not a useful
training corpus. Four landed in train and one in test under the current stable
hash; dev is empty because the corpus is tiny. No SFT result should be claimed
from this historical dataset.

The later training-pool v1 rollout produced 16 exact train trajectories. The
official harness resolved seven and left nine unresolved, so strict SFT export
accepts seven and rejects nine. The baseline and failure analysis are frozen in
`experiments/training_pool_v1_deepseek/`. The nine unresolved task IDs form
`manifest_train_corrections.jsonl`; they remain rejected examples until a
same-prompt corrected rollout is officially resolved.

The first correction smoke test has now resolved `pydicom__pydicom-800` under
the pinned official harness, with zero invalid actions. Its original unresolved
and new resolved trajectories pass the exact-prompt preference pairing gate.
Combining the accepted correction with the unchanged original baseline gives
eight strict train SFT samples. A first correction attempt on the other eight
tasks generated seven patches, but official grading resolved 0/8 with zero
environment errors; strict SFT export accepted none. Those tasks remain open,
and the eight-sample corpus is unchanged. See the correction smoke and
remaining-eight audits in `experiments/training_pool_v1_deepseek/`.

Reproduce the two audits:

```bash
uv run python -m patchpilot.export_training_data \
  --input .patchpilot/swebench \
  --input benchmarks/results \
  --output-dir .patchpilot/training/strict-v1

uv run python -m patchpilot.export_training_data \
  --input .patchpilot/swebench \
  --input benchmarks/results \
  --output-dir .patchpilot/training/legacy-audit-v1 \
  --allow-legacy
```

Generated datasets remain under Git-ignored `.patchpilot/`; only aggregate,
non-sensitive experiment reports should be versioned.

## Preference-pair contract

Preference export is limited to tasks explicitly assigned to the frozen train
partition. Hash-assigned ad-hoc tasks, dev tasks, and test tasks are rejected.
For each task, the exporter requires exactly one original trajectory with
`official_status=unresolved` and one corrected trajectory with
`official_status=resolved`. Both sides must pass the same schema-v2 quality,
verification, source-only, and secret checks used for strict SFT data.

The initial system and user messages must match exactly after repository-path
redaction. The output keeps this shared `prompt` once, followed by the complete
chosen and rejected continuations, including tool calls and tool observations.
This prevents a changed prompt from being misrepresented as a preference over
two answers to the same input.

Gold patches may guide analysis on the train split, but a gold diff is not a
chosen trajectory. A chosen example must be a coherent Agent rollout and must
pass the separate official harness. Generate corrections only from
`manifest_train_corrections.jsonl`, grade that run, and then export pairs:

```bash
uv run python -m patchpilot.export_preference_data \
  --rejected-input .patchpilot/swebench/batch/<baseline-run> \
  --chosen-input .patchpilot/swebench/batch/<officially-graded-correction-run> \
  --output-dir .patchpilot/training/preferences-v1
```

## Next training sequence

The first task pool is now frozen in
`benchmarks/training_pool_v1/`: 24 source-only Python tasks from the official
SWE-bench `dev` split, balanced across four repositories as 16 train, 4 dev, and
4 test tasks. It has no instance overlap with `benchmarks/swebench/`. Separate
manifests make accidental rollout collection on a held-out task less likely.
An official Docker smoke test retained repositories only when a non-empty
marker baseline was unresolved and the dataset gold patch was resolved. The
four retained repositories passed 0/4 baseline and 4/4 gold controls with zero
infrastructure errors. Two initially considered repositories were excluded
after two gold controls per repository showed systematic environment drift;
the evidence is frozen in `benchmarks/training_pool_v1/smoke_report.json`.

1. Prepare Docker images and validate gold-patch execution for all 24 tasks
   before paying for model rollouts. Remove or replace environment-invalid cases
   through a versioned selection update, never silently.
2. Collect schema-v2 trajectories from `manifest_train.jsonl`, retaining failed
   attempts separately for later preference construction. Do not export dev or
   test trajectories into SFT data.
3. Freeze the surviving held-out task set before training. Never select examples
   after looking at trained-model answers.
4. Train a small SFT/LoRA pilot and save the base-model revision, tokenizer,
   prompt template, adapter configuration, seed, and dataset report.
5. Compare base and trained models using the same Agent code, tool policy, step
   budget, local benchmark, and frozen SWE-bench subset.
6. Report official resolution as the primary outcome, with steps, duration,
   invalid actions, tool failures, edit success, and test freshness as process
   metrics.
7. Only after the SFT baseline is stable, construct chosen/rejected pairs for
   preference optimization or reward-guided methods. Do not manufacture GRPO
   rewards before the evaluator can reliably distinguish correct patches from
   environment failures.

For SWE-bench rollouts, Agent-side tests are deliberately different from the
official target tests. The Agent runs relevant public regression files already
present at `base_commit`; it never receives `test_patch` or names tests
introduced by that patch. A strict training sample therefore needs both fresh
public regression success inside the trajectory and `resolved=true` from the
separate official evaluator.
