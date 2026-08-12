# PatchPilot pilot v1 — DeepSeek

This directory is the versioned, compact record of PatchPilot's first frozen
multi-repository SWE-bench Verified experiment. Large disposable repositories,
Docker logs, trajectories, and model API data remain under the Git-ignored
`.patchpilot/` directory and are intentionally not committed.

## Result

- Model: `deepseek-v4-flash`
- PatchPilot code: commit `9b7d126`
- Agent baseline: `v0.3.0` (`01ae04f`)
- Benchmark: 8 frozen SWE-bench Verified instances from 8 repositories
- Generation: 8/8 patches generated, 0 generation errors
- Local public reproductions: 8/8 passed after the Agent edit
- Official SWE-bench result: **7/8 resolved (87.5%)**
- Mean steps: 13.25
- Mean duration: 81.758 seconds
- Tool failure rate: 6.38%
- Instances whose generated patch changed a test file: 3

The unresolved Requests instance passed its issue-specific target test. An
Agent-patch retry and a gold-patch control both failed the same four unrelated
timeout regression tests. See
[`requests_environment_analysis.md`](requests_environment_analysis.md). The
official 7/8 result is preserved rather than rewritten.

Machine-readable results are in [`summary.json`](summary.json), with per-case
steps, timing, modified files, and status in [`cases.json`](cases.json).

## Frozen inputs

The selection policy and complete official instance records are versioned in:

- `benchmarks/swebench/pilot_v1_selection.json`
- `benchmarks/swebench/pilot_v1_manifest.jsonl`
- `benchmarks/swebench/<instance_id>.json`

The pilot excludes `sympy__sympy-20590`, which had already been used while
developing the Agent loop. This pilot is now a development benchmark: future
Agent changes may be compared against it, but it should not be described as a
fresh held-out evaluation again.

## Commands used

Preflight the frozen manifest and its local images:

```bash
uv run python -m patchpilot.evaluate_swebench_batch \
  --manifest benchmarks/swebench/pilot_v1_manifest.jsonl \
  --validate-only
```

Generate predictions with the model configuration from `.env`:

```bash
uv run python -m patchpilot.evaluate_swebench_batch \
  --manifest benchmarks/swebench/pilot_v1_manifest.jsonl \
  --stage generate \
  --run-id pilot-v1-deepseek
```

Grade the generated predictions with the official harness:

```bash
uv run --with swebench python -m patchpilot.evaluate_swebench_batch \
  --manifest benchmarks/swebench/pilot_v1_manifest.jsonl \
  --stage official \
  --run-dir .patchpilot/swebench/batch/pilot-v1-deepseek \
  --max-workers 1
```

Regenerate the compact per-test audit from the retained official reports and
gold-patch control:

```bash
uv run python -m patchpilot.analyze_swebench_run \
  --run-dir .patchpilot/swebench/batch/pilot-v1-deepseek
```

This writes ignored machine-readable and Markdown audit files below the run's
`analysis/` directory. It keeps the official 7/8 result unchanged and marks the
Requests case as environment-suspect only because the gold control reproduces
the identical regression failures.

## Interpretation and limitations

Eight deliberately small instances are enough for an integration and
capability pilot, not for a statistically strong leaderboard claim. The model,
prompt, tool policy, local reproduction commands, instance selection, and
Docker environment all affect the result. Three successful predictions also
modified repository test files; this observation motivated the source-only
test-file protection implemented in PatchPilot v0.4.0. The historical patches
remain unchanged so the recorded pilot evidence stays reproducible.

No API key, model response content, private environment value, absolute local
path, or full third-party checkout is included in this record.
