# PatchPilot

PatchPilot is a small, inspectable code-repair agent. Given a bug report and a
local Python repository, it lets an LLM inspect files, search code, apply a
unified diff, run a fixed test command, and return a final answer. Every model
decision and tool result is written to a JSONL trajectory for later evaluation
or post-training.

The first milestone intentionally avoids an agent framework. The loop, tool
boundaries, prompts, and traces are all visible in this repository.

## Current MVP

- Repository-scoped `list_files`, `search_code`, and `read_file` tools
- Static `inspect_python_class` inheritance tracing with definition locations
  and `__slots__` detection; repository code is parsed but never imported
- Exact, single-occurrence `replace_text` for reliable small edits
- Unified-diff `apply_patch` tool with path validation and `git apply --check`
  (including minimal zero-context diffs)
- Fixed `run_tests` tool; the model cannot choose an arbitrary command
- `git_diff` inspection
- OpenAI-compatible chat-completions client
- Deterministic fake model for offline end-to-end tests
- Schema-v2 JSONL trajectories that preserve the exact model conversation
- Audited SFT chat-data export with quality filters and task-level data splits
- Train-only preference export pairing official unresolved and resolved trajectories

## Quick start

Python 3.11 or newer is supported.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m patchpilot --help
```

Run against a local repository with an OpenAI-compatible endpoint:

```bash
cp .env.example .env
# Edit .env and replace PATCHPILOT_API_KEY with your real key.

PYTHONPATH=src python -m patchpilot \
  --repo /path/to/buggy/repo \
  --issue "The add function subtracts its second argument" \
  --test-command "python -m pytest -q"
```

Optional environment variables:

- `PATCHPILOT_BASE_URL` (default: `https://api.openai.com/v1`)
- `PATCHPILOT_API_KEY`
- `PATCHPILOT_MODEL`

PatchPilot loads `.env` from the current directory. Existing process environment
variables take precedence over values in the file. To use another file, pass
`--env-file /path/to/config.env`. The real `.env` file is ignored by Git; only
the placeholder `.env.example` should be committed. `--model` remains available
as a one-run override for `PATCHPILOT_MODEL`.

Traces are saved under `.patchpilot/trajectories/` in the target repository by
default. Use `--trajectory-dir` to keep them elsewhere.

## Action protocol

The model emits exactly one JSON object per turn:

```json
{"tool":"search_code","arguments":{"query":"def add"}}
```

Available tools are `list_files`, `search_code`, `read_file`,
`inspect_python_class`, `replace_text`, `apply_patch`, `run_tests`, and
`git_diff`. Use `inspect_python_class` for Python inheritance, MRO, or
`__slots__` issues. Prefer `replace_text` for one exact local replacement and
reserve `apply_patch` for complex or multi-file edits.
A run ends with:

```json
{"final":"Fixed the arithmetic bug and verified the test suite."}
```

A run is only marked successful after a real non-empty edit, a passing test run
after the latest successful edit, and a successful `git_diff` inspection of
that final state. Agent-owned trajectory paths are excluded from the diff, so
artifacts cannot masquerade as a repair. When test protection is enabled, a
final diff containing a protected test file is rejected even if that change
was created outside the editing tools.

Trajectories and benchmark summaries record tool failures; edit, patch, and
replacement attempt/success rates; blocked test-file edits; empty-diff
attempts; test files present in the final patch; and whether tests were rerun
after the latest edit.

## Safety boundary

File tools reject absolute paths and traversal outside the selected repository.
The test command is supplied by the human at process startup and runs with a
timeout. It is still local code execution: only run PatchPilot on repositories
you trust, ideally inside a disposable container. Stronger OS-level sandboxing
is planned, not claimed by this MVP.

## Roadmap

1. Establish baselines on a small curated Python bug set.
2. Add hybrid retrieval (lexical first, embeddings as an experiment).
3. Expand from one validated SWE-bench Verified instance to a pinned subset.
4. Add automated official-image preparation and batch evaluation.
5. Build an audited, leakage-resistant SFT trajectory exporter. (implemented)
6. Collect a larger exact-trajectory corpus, run SFT/LoRA, and compare it with
   the frozen pre-training baseline.

## Local benchmark

The repository includes three intentionally buggy Python cases. The evaluator
copies each fixture into an isolated temporary Git repository, confirms its
tests fail, runs the agent, and stores a summary, patch, and trajectory:

```bash
uv run python -m patchpilot.evaluate \
  --benchmark benchmarks/cases
```

See `benchmarks/README.md` for case selection and output details.

## Post-training data

New runs write trajectory schema v2. In addition to model outputs and tool
results, it records the exact initial messages and each observation sent back to
the model, plus stable `task_id`, source, and model metadata. This makes the chat
conversation reconstructable without guessing which prompt version was used.

Export verified trajectories into OpenAI-style chat JSONL files:

```bash
uv run python -m patchpilot.export_training_data \
  --input .patchpilot/swebench \
  --input benchmarks/results \
  --output-dir .patchpilot/training/strict-v1
```

The output directory contains `train.jsonl`, `dev.jsonl`, `test.jsonl`, and
`report.json`. The strict default rejects legacy trajectories, unsuccessful or
incompletely verified runs, known official failures, test-modifying patches,
high tool-failure/invalid-action runs, and conversations containing recognizable
secret patterns. Absolute repository paths are replaced with `<REPOSITORY>`.

Frozen benchmark tasks retain their preassigned `pool_split`; older batch runs
recover it from the instance recorded in their manifest snapshot. Ad-hoc runs
fall back to a deterministic `task_id` hash. Conflicting assignments for the
same task stop the export, preventing repeated runs of one issue from leaking
across train/dev/test. Old schema-v1 traces can be audited with
`--allow-legacy`, but they are labeled `legacy_approximate` and should not be
mixed silently with exact training examples.

After a failed train trajectory has been corrected and officially resolved,
export a conversation-level preference pair:

```bash
uv run python -m patchpilot.export_preference_data \
  --rejected-input .patchpilot/swebench/batch/<baseline-run> \
  --chosen-input .patchpilot/swebench/batch/<correction-run> \
  --output-dir .patchpilot/training/preferences-v1
```

This exporter is deliberately stricter than the SFT exporter. Both trajectories
must be exact schema-v2 SWE-bench runs from an explicitly preassigned train
task, pass Agent-side verification, leave tests untouched, and have identical
initial prompts. The rejected side must be officially unresolved and the chosen
side officially resolved. A gold diff alone is never accepted as a chosen
trajectory.

See [`docs/post_training.md`](docs/post_training.md) for the data contract,
current corpus audit, frozen 24-task training pool, and the SFT/LoRA evaluation
sequence.

## SWE-bench adapter

PatchPilot can generate one prediction in the JSONL format expected by the
official SWE-bench evaluator. The adapter reads an official-style instance,
clones its repository at `base_commit`, runs PatchPilot in the disposable
clone, and saves both `model.patch` and `predictions.jsonl`:

```bash
uv run python -m patchpilot.evaluate_swebench \
  --instance-file /path/to/swebench-dev.jsonl \
  --instance-id sympy__sympy-20590 \
  --test-command "python -m pytest -q"
```

The instance file may be one JSON object, a JSON array, or JSONL. Each selected
record must contain `instance_id`, `repo`, `base_commit`, and
`problem_statement`. For offline development, `--source-repo` can point to an
existing local Git clone; PatchPilot still creates and edits a separate clone.

For a realistic run, pass an existing official SWE-bench instance image. The
adapter copies the image's pinned `/testbed` checkout into a disposable host
working tree, lets the agent edit that copy, and runs the fixed test command in
a fresh container after every `run_tests` action:

```bash
uv run python -m patchpilot.evaluate_swebench \
  --instance-file benchmarks/swebench/sympy__sympy-20590.json \
  --docker-image sweb.eval.x86_64.sympy__sympy-20590:latest \
  --test-command "python -c 'from sympy import Symbol; assert not hasattr(Symbol(\"s\"), \"__dict__\")'" \
  --max-steps 25 \
  --timeout 180
```

Container tests default to `linux/amd64`, no network, 2 CPUs, 4 GiB memory,
and 512 processes. These limits can be changed with `--docker-platform`,
`--docker-cpus`, `--docker-memory`, and `--docker-pids-limit`. The model cannot
change the image, test command, or limits during a run.

SWE-bench runs also make conventional test files read-only to the Agent by
default. Tests remain available to search and read, but edits under `test/`,
`tests/`, or `testing/`, and edits to `test_*.py`, `*_test.py`, or `conftest.py`
are rejected by both editing tools. Use `--allow-test-file-edits` only for a
task that explicitly requires changing tests. The general `patchpilot` command
keeps its previous permissive default and offers `--protect-test-files` when
the same source-only policy is wanted.

`agent_verified` means that PatchPilot left a non-empty diff, the configured
container test passed after the latest edit, and the final diff passed its
inspection and test-file policy. It is a process signal, not the official
benchmark result. The generated `predictions.jsonl` must still pass the
SWE-bench harness.

## Batch SWE-bench evaluation

Batch runs are driven by a versioned JSONL manifest. Each record pins the
instance file, Docker image, test command, step budget, and timeout:

```json
{"schema_version":1,"instance_id":"sympy__sympy-20590","instance_file":"sympy__sympy-20590.json","docker_image":"sweb.eval.x86_64.sympy__sympy-20590:latest","test_command":"python -c 'from sympy import Symbol; assert not hasattr(Symbol(\"s\"), \"__dict__\")'","max_steps":25,"timeout":180,"enabled":true,"protect_test_files":true}
```

Relative paths are resolved from the manifest's directory. Validate all files
and local Docker images without loading an API key or calling the model:

```bash
uv run python -m patchpilot.evaluate_swebench_batch \
  --manifest benchmarks/swebench/dev_manifest.jsonl \
  --validate-only
```

`protect_test_files` defaults to `true` when omitted and is stored in the run's
manifest snapshot. Set it to `false` only for a deliberately test-editing case.

Generate one prediction per enabled case. A failure is written to that case's
result file and does not abort later cases:

```bash
uv run python -m patchpilot.evaluate_swebench_batch \
  --manifest benchmarks/swebench/dev_manifest.jsonl \
  --stage generate
```

Every run writes `.patchpilot/swebench/batch/<run-id>/` containing a manifest
snapshot, per-case state, individual trajectories and patches, one merged
`predictions.jsonl`, and `summary.json`. Resume an interrupted run by passing
its exact directory:

```bash
uv run python -m patchpilot.evaluate_swebench_batch \
  --manifest benchmarks/swebench/dev_manifest.jsonl \
  --stage generate \
  --run-dir .patchpilot/swebench/batch/<run-id> \
  --resume
```

Completed cases with intact prediction artifacts are skipped. Error cases and
completed records whose prediction artifact was lost are attempted again. The
manifest snapshot and model name must match the original run.

Run official grading separately so an evaluator or Docker failure never causes
another paid model rollout:

Pin `swebench==4.1.0` for this evaluator. The newer 5.0.2 harness changed its
CLI and instance schema; an unpinned install is not an equivalent evaluation.

```bash
uv run --with 'swebench==4.1.0' python -m patchpilot.evaluate_swebench_batch \
  --manifest benchmarks/swebench/dev_manifest.jsonl \
  --stage official \
  --run-dir .patchpilot/swebench/batch/<run-id>
```

Use `--stage all` to generate and grade in one process when the optional
`swebench` package is available:

```bash
uv run --with 'swebench==4.1.0' python -m patchpilot.evaluate_swebench_batch \
  --manifest benchmarks/swebench/dev_manifest.jsonl \
  --stage all
```

Both `official` and `all` now fail before evaluation (and before paid generation
for `all`) with an actionable command when the optional harness is missing. The
final summary separates generation errors, patch generation, `agent_verified`,
official resolved/unresolved/error states, tool and patch-quality metrics, test
freshness, first-edit step, duration, and official resolve rate.

When an enabled manifest case names a qualified Docker image with a non-`latest`
tag, the official stage temporarily exposes that exact image under the `latest`
name expected by the SWE-bench harness. A pre-existing, different `latest` image
is first protected by a unique backup tag and restored after the harness exits,
including timeout and launch-error paths. Cleanup verifies image IDs and refuses
to overwrite a tag changed concurrently by another process. Alias actions and
cleanup errors are recorded in `official/result.json` and `official/harness.log`.

Audit a completed official run without calling the model again:

```bash
uv run python -m patchpilot.analyze_swebench_run \
  --run-dir .patchpilot/swebench/batch/<run-id>
```

The analyzer reads the official per-instance `FAIL_TO_PASS` and `PASS_TO_PASS`
results, reports modified test files, and writes `analysis/summary.json` plus
`analysis/report.md`. It only labels an unresolved case as environment-suspect
when its target tests passed and an available gold-patch control reproduced the
identical non-empty regression-failure set. Without that gold control, the case
remains an unconfirmed candidate. These annotations never replace or adjust the
official resolved count or resolve rate.

Run the official Docker evaluator on the generated prediction separately:

```bash
uv run --with 'swebench==4.1.0' python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --split test \
  --predictions_path /path/to/predictions.jsonl \
  --max_workers 1 \
  --instance_ids sympy__sympy-20590 \
  --run_id patchpilot-single \
  --namespace ''
```

On Apple Silicon, `--namespace ''` builds or reuses local amd64 images. The
first build is slow because it installs the shared Ubuntu/Miniconda environment;
later instances can reuse those layers. Official evaluation executes third-party
repository code and can consume substantial disk space, so start with one
verified instance and keep Docker resource limits enabled.

The `sympy__sympy-20590` development run was validated end to end: the gold
patch resolved 1/1 instance, and PatchPilot with `deepseek-v4-flash` generated a
functionally equivalent patch that also resolved 1/1 under the official harness.

## Recorded experiments

Versioned experiment reports keep compact, reviewable evidence in Git without
committing disposable repositories or full Docker logs. The first frozen
multi-repository run is documented in
[`experiments/pilot_v1_deepseek/`](experiments/pilot_v1_deepseek/README.md):
PatchPilot generated and locally verified 8/8 patches, with an official
SWE-bench result of 7/8 resolved (87.5%). Its one unresolved instance is kept in
the official count and accompanied by a separate gold-patch environment audit.
The same evidence can now be regenerated from the ignored run artifacts with
`patchpilot.analyze_swebench_run`.

The complete 16-task local training partition is frozen separately in
[`experiments/training_pool_v1_deepseek/`](experiments/training_pool_v1_deepseek/README.md).
It records an official result of 7/16 resolved (43.75%), the seven strict SFT
trajectory candidates, and a gold-informed diagnosis of all nine unresolved
predictions. This training-pool result is not a held-out benchmark score.
