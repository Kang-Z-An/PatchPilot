# Local benchmark

Each folder under `cases/` contains a `case.json` manifest and an intentionally
buggy `repo/` fixture. Fixtures are immutable inputs: the evaluator copies each
one to a temporary directory, initializes a fresh Git repository, verifies that
the initial tests fail, and only then starts the agent.

Run all cases:

```bash
uv run python -m patchpilot.evaluate \
  --benchmark benchmarks/cases
```

Run one case while debugging:

```bash
uv run python -m patchpilot.evaluate \
  --benchmark benchmarks/cases \
  --case arithmetic_operator_001
```

Every run creates `results/<run-id>/` with `summary.json`, per-case JSON files,
model/tool trajectories, and the final patches. The `results/` directory is
ignored by Git. Summary metrics include tool failures, invalid actions, total
edit attempts, unified-diff patch success rate, and exact-replacement success
rate. They also report protected edit attempts, empty final diffs, modified
test files, and the share of runs tested after their latest edit.

## SWE-bench development manifest

`swebench/dev_manifest.jsonl` is the pinned development batch. Validate it
without calling the model:

```bash
uv run python -m patchpilot.evaluate_swebench_batch \
  --manifest benchmarks/swebench/dev_manifest.jsonl \
  --validate-only
```

The current SymPy case is an integration/development instance because it was
used to tune the agent loop. It must not be reported as an untouched test-set
result. Add future development and held-out cases as separate, versioned
manifests rather than silently changing this file between experiments.

## SWE-bench pilot v1

`swebench/pilot_v1_manifest.jsonl` pins eight SWE-bench Verified instances from
eight different repositories. The selection was recorded before running the
agent and excludes the SymPy development instance. Selection metadata and broad
bug categories are in `swebench/pilot_v1_selection.json`.

Each `test_command` is a small public reproduction derived from the issue's
observable behavior. It does not depend on SWE-bench's hidden `test_patch`.
Before freezing the manifest, every reproduction was checked to fail on the
pinned base image for the expected reason and to pass after applying the
official gold patch.

Batch SWE-bench cases protect conventional test files from Agent edits by
default, even when `protect_test_files` is omitted from an older manifest.
Protection is part of the generated run snapshot. A future case that genuinely
requires test changes must opt out explicitly with `"protect_test_files": false`.

Validate the instance files without requiring Docker images:

```bash
uv run python -m patchpilot.evaluate_swebench_batch \
  --manifest benchmarks/swebench/pilot_v1_manifest.jsonl \
  --validate-only \
  --skip-docker-check
```

After preparing all pinned Docker images, remove `--skip-docker-check` for the
full preflight. Use a new run ID for each experiment and never edit the manifest
between generation and official grading.

Prepare the pilot's local amd64 images with the official SWE-bench harness:

```bash
uv run --with swebench python -m swebench.harness.prepare_images \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --split test \
  --instance_ids \
    sympy__sympy-16886 \
    sphinx-doc__sphinx-9698 \
    matplotlib__matplotlib-13989 \
    scikit-learn__scikit-learn-14141 \
    pydata__xarray-4629 \
    pytest-dev__pytest-5262 \
    pylint-dev__pylint-4970 \
    psf__requests-5414 \
  --max_workers 1 \
  --force_rebuild false \
  --namespace '' \
  --tag latest \
  --env_image_tag latest
```

The first build downloads full repository histories and repository-specific
dependencies. Build logs are written under the ignored `logs/build_images/`
directory. Later runs reuse the resulting environment and instance layers.

Generate predictions with the configured OpenAI-compatible model:

```bash
uv run python -m patchpilot.evaluate_swebench_batch \
  --manifest benchmarks/swebench/pilot_v1_manifest.jsonl \
  --stage generate \
  --run-id pilot-v1
```

After generation, use the printed run directory to run the official SWE-bench
grader. Keeping generation and grading separate makes interrupted runs
resumable and avoids accidentally regenerating predictions:

```bash
uv run --with swebench python -m patchpilot.evaluate_swebench_batch \
  --manifest benchmarks/swebench/pilot_v1_manifest.jsonl \
  --stage official \
  --run-dir /absolute/path/to/the/generated/run
```

Audit the official per-test results and any gold-patch controls already stored
under that run directory:

```bash
uv run python -m patchpilot.analyze_swebench_run \
  --run-dir /absolute/path/to/the/generated/run
```

The audit is written to `<run-dir>/analysis/`. It preserves the official score;
environment-suspect and test-file-change fields are diagnostic annotations.
