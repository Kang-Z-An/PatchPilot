# PatchPilot

[中文介绍](README.zh-CN.md) · [Usage & evaluation](docs/usage.md) · [Post-training data](docs/post_training.md)

**An inspectable code-repair agent with reproducible SWE-bench evaluation and audited training-data exports.**

PatchPilot takes an issue and a Python repository through an explicit
inspect → edit → test → review loop. It records model decisions and tool
observations so you can study successful repairs and failures. The core uses
Python's standard library plus `python-dotenv`; no agent framework is required.

This is an early research project. The Agent, evaluator, and data pipeline are
implemented. RAG retrieval and SFT/LoRA/RL training remain future work.

## Try it in two minutes

Requirements: Python 3.11+, Git, and [uv](https://docs.astral.sh/uv/).
Clone the repository, then run the offline demo:

```bash
git clone https://github.com/Kang-Z-An/PatchPilot.git
cd PatchPilot
uv sync --locked
uv run python examples/offline_demo.py
```

The demo repairs an arithmetic bug with a **scripted model**, runs real
tests, prints the diff, and saves a trace under `.patchpilot/demo/`. It needs
no API key, Docker, or GPU. It demonstrates the execution loop, not model
capability. Its disposable repository is removed afterward.

Run the offline test suite:

```bash
uv run python -m unittest discover -s tests -v
```

Without uv, install from this checkout using `python -m pip install -e .`,
then run the same Python commands without the `uv run` prefix. This setup
does not install a package by name from PyPI.

## Repair a repository with an LLM

Copy `.env.example` to `.env` and set the values for your provider:

```dotenv
PATCHPILOT_API_KEY=replace-with-your-api-key
PATCHPILOT_BASE_URL=https://your-provider.example/v1
PATCHPILOT_MODEL=your-model-name
```

Then point PatchPilot at a disposable local Git checkout:

```bash
uv run patchpilot \
  --repo /absolute/path/to/disposable-checkout \
  --issue "Describe the failing behavior and expected result" \
  --test-command "python -m unittest discover -s tests -q" \
  --protect-test-files \
  --max-steps 20
```

The command edits that checkout **in place** and makes paid API requests.
Repository snippets and test output are sent to your configured provider.
Install the target project's test dependencies in the test environment first.
Use `--test-command "python -m pytest -q"` for a pytest project once its
dependencies are available.

Existing environment variables override `.env`; `--model` overrides
`PATCHPILOT_MODEL`. Traces may contain source code and issue details, so
review them before sharing. See [the usage guide](docs/usage.md) for Docker
workflows and `--env-file` / `--trajectory-dir` options.

## What is implemented

- Repository-scoped search, reading, and static Python inheritance inspection.
- Exact `replace_text`, validated unified diffs, and protected test files.
- Fixed regression commands, fresh-test checks, and final diff inspection.
- Optional Docker assertion probes, including the same code on original and edited checkouts.
- Resumable SWE-bench batches with separate generation and official grading.
- Exact schema-v2 trajectories, task-disjoint SFT export, same-task/same-prompt
  preference pairs, and assistant-only token-label auditing.

`agent_verified` records completion of configured checks. Official
SWE-bench `resolved` is a separate outcome. An assertion can pass while the
complete issue remains unresolved.

## Recorded evidence

These are small development experiments with requested model name
`deepseek-v4-flash`, not full SWE-bench leaderboard results or measured
post-training gains. Provider model revisions were not pinned.

| Experiment | Official result | Interpretation |
| --- | --- | --- |
| [Historical Verified pilot](experiments/pilot_v1_deepseek/README.md) | 7/8 resolved | 8 selected tasks; 3 patches edited tests under the historical policy; now a development set |
| [Frozen training partition](experiments/training_pool_v1_deepseek/README.md) | 7/16 resolved | Training-data collection, not a held-out score; no test edits |
| [Dual-checkout probe smoke](experiments/training_pool_v1_deepseek/v056_dual_checkout_probe_smoke_audit.json) | 1/2 resolved | 2 selected train failures; 0 newly accepted strict SFT trajectories |

The audited corpus has **8 accepted train trajectories and 1 valid preference
pair**, following a separate correction run. Compact reports are included;
raw conversations, model weights, and Docker images are not. No SFT/LoRA/RL
job or pre/post-training improvement has been validated.

## Evaluation

Try the three local toy cases with your configured model:

```bash
uv run python -m patchpilot.evaluate --benchmark benchmarks/cases
```

Inspect a SWE-bench manifest without Docker or model calls:

```bash
uv run python -m patchpilot.evaluate_swebench_batch \
  --manifest benchmarks/swebench/pilot_v1_manifest.jsonl \
  --validate-only --skip-docker-check
```

Real grading requires Docker and prepared instance images. This adapter uses
`swebench==4.1.0`; follow [the evaluation guide](docs/usage.md) and
[environment recipes](benchmarks/training_pool_v1/environment_adaptations/README.md).
The training pool originates from SWE-bench `dev`, so its official grading
uses `--dataset-name princeton-nlp/SWE-bench --split dev`.

## Find your way around

| Path | Purpose |
| --- | --- |
| [src/patchpilot/agent.py](src/patchpilot/agent.py) | Model/action loop, prompt variants, verification and metrics |
| [src/patchpilot/tools.py](src/patchpilot/tools.py) | Repository tools and editing boundaries |
| [src/patchpilot/docker.py](src/patchpilot/docker.py) | Container tests and issue probes |
| [src/patchpilot/swebench_batch.py](src/patchpilot/swebench_batch.py) | Resumable generation and official evaluation |
| [benchmarks/](benchmarks/README.md) | Toy fixtures, frozen manifests and provenance |
| [experiments/](experiments/training_pool_v1_deepseek/README.md) | Compact positive and negative experiment records |
| [docs/post_training.md](docs/post_training.md) | Data contracts, export gates and training plan |
| [tests/](tests/) | Offline unit and integration tests |

## Boundaries and next work

Local test commands execute repository code with your permissions. Docker
tests use resource limits and offline networking, but PatchPilot is not a
hardened sandbox for arbitrary hostile repositories. Use disposable, trusted
checkouts; see [SECURITY.md](SECURITY.md).

Next priorities are resolving differences between public and official test
expectations, improving issue coverage, and collecting higher-quality
trajectories. Retrieval and a frozen base-versus-LoRA comparison follow those
evaluation and data checks.

Contributions are welcome through [CONTRIBUTING.md](CONTRIBUTING.md).
PatchPilot's original code is [MIT licensed](LICENSE); benchmark material
retains upstream terms and attribution in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
