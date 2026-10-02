# Contributing to PatchPilot

Small, reproducible changes are welcome. Start with the Agent loop, tool
contracts, evaluator, or training-data checks described in [README.md](README.md).

## Development

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python examples/offline_demo.py
uv run python scripts/check_release.py
```

The tests use scripted models and mocked Docker calls. They do not require
model credentials, model weights, a GPU, or a Docker daemon. CI runs the test
suite on Python 3.11 and 3.13 and checks that the package builds and installs.
Actual SWE-bench execution is a separate, opt-in integration test.

## Changes and evidence

Describe the bug or behavior change, its reproduction, and the verification
performed. Add focused regression coverage when changing tool boundaries,
verification state, resume behavior, or data filtering. Include a minimal
sanitized example rather than a full local trajectory.

Preserve historical reports and frozen manifests. New experiments should use
new run IDs and record model settings, prompt variant, dataset revision, and
environment adaptations. Keep official resolution, Agent-side verification,
and training-data acceptance separate. Do not tune on held-out tasks or send
gold/test patches to the Agent.

New prompt variants should be opt-in. Include negative results when reporting
an experiment; a passing toy test or a tiny smoke batch is not a benchmark-wide
performance claim. Post-training gains require an actual frozen pre/post run.

Keep credentials, raw conversations, model weights, third-party checkouts, and
Docker logs out of pull requests. Do not modify the real `.env` during tests.
See [SECURITY.md](SECURITY.md) for sensitive reports and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for data provenance.
