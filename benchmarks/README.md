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
rate.

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
