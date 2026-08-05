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
