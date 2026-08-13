# PatchPilot training pool v1

This directory freezes the first post-training task pool. It is selected from
the official `princeton-nlp/SWE-bench` **dev** split at revision
`e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6`. The 19,008-record train split was
not used because it does not provide `FAIL_TO_PASS` test identifiers required by
PatchPilot's verified rollout pipeline.

## Composition

- 24 tasks from 4 Python repositories;
- 16 training-rollout tasks;
- 4 development-evaluation tasks;
- 4 final internal-test tasks;
- exactly 4 train, 1 dev, and 1 test task per repository.

Candidates require a non-empty target-test set and test patch, a Python-only
gold patch touching production files only, at most three changed files, at most
80 changed lines, at most five target tests, and a bounded issue length. The
selection is ranked deterministically using the seed in `selection.json`.

The instance files deliberately omit `patch` and `test_patch`. Gold patch file
names and aggregate size are retained only in `selection.json` for dataset
auditing; the Agent receives only the issue and repository checkout.

## Isolation rules

Use the manifests for their named purpose:

- `manifest_train.jsonl`: collect schema-v2 trajectories for training-data
  filtering;
- `manifest_dev.jsonl`: choose checkpoints and training settings, never feed
  its model outputs into training;
- `manifest_test.jsonl`: run only for frozen pre/post-training comparison.
- `manifest_smoke.jsonl`: one fixed train task per repository for validating
  Docker and gold-patch execution before expanding to all 24 tasks.

All three sets are task-disjoint. They are also disjoint from every instance in
`benchmarks/swebench/`, including the 8-task Verified pilot and the earlier
SymPy development instance. The generator additionally loads all 500 IDs from
SWE-bench Verified revision
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a` and excludes them programmatically.

The initial smoke test considered six repositories. `pvlib/pvlib-python` and
`pyvista/pyvista` were excluded after two deterministically ranked instances
from each repository produced unresolved official gold controls with large
regression-failure sets. The four retained repositories passed the acceptance
rule: marker baseline 0/4 resolved, gold control 4/4 resolved, and zero
infrastructure errors. See `smoke_report.json` for the versioned evidence.

The full-pool gold control now resolves all 24 selected instances with zero
harness errors. Nineteen use the official generated `:latest` instance image.
Five use explicitly tagged environment-only adaptations because the original
build or test environment drifted after the historical task was created:

- `marshmallow-code__marshmallow-1810:compat` uses a commit-pinned shallow
  history and an already qualified same-repository test dependency layer;
- `pydicom__pydicom-1194:precache` pins pytest before removal of the legacy
  setup/teardown hooks and caches two public data files with verified hashes;
- `pylint-dev__astroid-1959:compat` and `pylint-dev__astroid-1978:compat`
  preserve editable installs while bypassing an obsolete isolated build
  backend;
- `sqlfluff__sqlfluff-4041:localcache` uses a commit-pinned shallow repository
  and an already qualified same-repository dependency layer.

Each adapted image was checked to preserve the exact base Git tree and to
contain no gold patch. The frozen manifests reference these explicit tags.
See `full_gold_report.json` for the aggregate acceptance result and boundaries.

## Reproduction and validation

Regenerate the exact selection without calling a model:

```bash
uv run --with datasets python -m patchpilot.prepare_training_pool \
  --output-dir benchmarks/training_pool_v1
```

Validate one split before preparing Docker images:

```bash
uv run python -m patchpilot.evaluate_swebench_batch \
  --manifest benchmarks/training_pool_v1/manifest_train.jsonl \
  --validate-only \
  --skip-docker-check
```

The source dataset is SWE-bench `dev`, so official grading must explicitly use:

```bash
--dataset-name princeton-nlp/SWE-bench --split dev
```

Do not use the evaluator defaults, which target SWE-bench Verified `test`.
