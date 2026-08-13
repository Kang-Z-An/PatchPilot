# Environment adaptation recipes

These recipes rebuild the five non-`:latest` images referenced by the frozen
training-pool manifests. They change only source acquisition, package tooling,
or public test-data availability. They must never copy a prediction or gold
patch into an image.

For the four commit-pinned recipes, create the named build-context directory
with a one-commit shallow fetch of the exact base commit before running
`docker build`. For example:

```bash
mkdir -p .patchpilot/training_pool_v1/source_cache/astroid-1959
git init .patchpilot/training_pool_v1/source_cache/astroid-1959
git -C .patchpilot/training_pool_v1/source_cache/astroid-1959 \
  remote add origin https://github.com/pylint-dev/astroid.git
git -C .patchpilot/training_pool_v1/source_cache/astroid-1959 \
  fetch --depth=1 origin 49691cc04f2d38b174787280f7ed38f818c828bd
git -C .patchpilot/training_pool_v1/source_cache/astroid-1959 \
  checkout --detach FETCH_HEAD
```

Use the corresponding commit and repository for the other directories:

| Context directory | Repository | Base commit |
| --- | --- | --- |
| `astroid-1959` | `pylint-dev/astroid` | `49691cc04f2d38b174787280f7ed38f818c828bd` |
| `astroid-1978` | `pylint-dev/astroid` | `0c9ab0fe56703fa83c73e514a1020d398d23fa7f` |
| `marshmallow-1810` | `marshmallow-code/marshmallow` | `23d0551569d748460c504af85996451edd685371` |
| `sqlfluff-4041` | `sqlfluff/sqlfluff` | `66fd602d0824138f212082c8fdf381266a9edad3` |

Run each `docker build` from the directory containing those context folders,
using the matching Dockerfile and the exact tag written in the manifests. The
base images named by the Dockerfiles must already be present; they are either
official SWE-bench environment images or gold-qualified same-repository
instance images.

For `pydicom-1194`, download the two files from commit
`39a2eb31815eec435dc26c322c27aec5cfcbddb6` of `pydicom/pydicom-data` into the
build context. The Dockerfile verifies both SHA-256 values before the image is
created.

After a build, verify that the synthetic `SWE-bench` commit has the requested
base commit as its parent, that `HEAD^{tree}` equals `<base>^{tree}`, and that
`git status --short` is empty. `full_gold_report.json` records the gold-control
acceptance result for these exact tags.
